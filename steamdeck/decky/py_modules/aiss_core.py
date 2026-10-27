"""Core logic of the AISS DLSS->NSS Decky plugin.

Nothing here imports `decky`, so the module can be exercised standalone (see tests/test_backend.py). main.py wraps an
`AissCore` in the Decky `Plugin` class.

Per enabled game the plugin owns:
  * pfx/drive_c/windows/system32/_nvngx.dll        the NSS NGX core (a foreign file there is backed up and restored)
  * <exe dir>/dbghelp.dll or <exe dir>/dxgi.dll   the loader shim (never overwrites a file that is not ours)
  * HKLM\\Software\\NVIDIA Corporation\\Global {41FCC608-...}=hex:01 in pfx/system.reg (only while the game is not
    running, with a one-time backup of system.reg)
  * its share of the Steam launch options (computed here, applied by the frontend through SteamClient)
and once, globally, the Vulkan layer in ~/.local/share/vulkan/implicit_layer.d (inert unless AISS_LAYER=1).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------------------------

BIN_FILES = {
    "core": "nvngx.dll",
    "dbghelp": "dbghelp.dll",
    "dxgi": "dxgi.dll",
    "layer_so": "libVkLayer_aiss_ngx.so",
    "layer_json": "VkLayer_aiss_ngx.json",
}
CORE_DEST_NAME = "_nvngx.dll"
CORE_BACKUP_SUFFIX = ".aiss-orig"
REG_BACKUP_SUFFIX = ".aiss-bak"

# Strings every build of our binaries contains; used to recognise an older/newer build of ours that is not in the
# manifest (e.g. one copied by hand during development) so it can be replaced instead of being reported as a conflict.
SHIM_MARKER = "AISS_SIGNED_DLL".encode("utf-16-le")
CORE_MARKER = b"AISS_RATIO"

SIG_OVERRIDE_VALUE = "{41FCC608-8496-4DEF-B43E-7D9BD675A6FF}"
REG_KEY_PATH = "Software\\NVIDIA Corporation\\Global"  # relative to HKLM, as Wine stores it in system.reg

DLSS_DLL = "nvngx_dlss.dll"
SL_INTERPOSER = "sl.interposer.dll"
SCAN_MAX_DEPTH = 10          # Unreal keeps nvngx_dlss.dll 8 levels deep (Engine/Plugins/.../Binaries/ThirdParty/Win64)
SCAN_MAX_ENTRIES = 250_000   # per game, guards against pathological trees

SKIP_NAME_SUBSTRINGS = ("proton", "steam linux runtime", "steamworks common redistributables")

BAD_EXE_SUBSTRINGS = (
    "crashreport", "crashpad", "crashhandler", "bugsplat", "bssndrpt", "unitycrashhandler", "eac", "easyanticheat",
    "beclient", "battleye", "eosbootstrap", "benchmark", "uninstall", "unins0", "setup", "installer", "launcher",
    "updater", "bootstrap", "_redist", "redist", "prereq", "vcredist", "directx", "dotnet", "ue4prereq", "cefprocess",
    "protocolselector", "helper", "reporter", "touchup", "quicksfv",
)

# Launch options
MANAGED_ENV = ("AISS_LAYER", "AISS_RATIO", "AISS_NSS_QUALITY", "AISS_LOG", "AISS_CONTROL")  # always owned by the plugin
# AISS_RATIO and AISS_NSS_QUALITY are no longer written (the model is chosen live through AISS_CONTROL); they stay
# managed so older launch options lose them on the next sync.
NVAPI_ENV = "PROTON_FORCE_NVAPI"
WDO_ENV = "WINEDLLOVERRIDES"
COMMAND = "%command%"

DEFAULT_SETTINGS = {
    "quality": 0,        # NSS model: 0 = full network (default), 2 = fast small network (more ghosting); applied live
    "log": 1,            # AISS_LOG level; 1 is the core's default and is not written out
    "fast_math": 0,      # 1 = fp32 requantisation in the ML graph (~2% faster, not bit exact); applied live
}

STATUS_INSTALLED = "installed"
STATUS_NOT_INSTALLED = "not_installed"
STATUS_CONFLICT = "conflict"
STATUS_PREFIX_MISSING = "prefix_missing"
STATUS_PARTIAL = "needs_repair"


class Logger:
    def info(self, msg):  # pragma: no cover - replaced by decky.logger
        print("[aiss] " + msg)

    def warning(self, msg):  # pragma: no cover
        print("[aiss] WARN " + msg)

    def error(self, msg):  # pragma: no cover
        print("[aiss] ERROR " + msg)


# --------------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_contains(path: Path, marker: bytes, limit: int = 64 << 20) -> bool:
    try:
        if path.stat().st_size > limit:
            return False
        with open(path, "rb") as f:
            return marker in f.read()
    except OSError:
        return False


def atomic_copy(src: Path, dst: Path) -> None:
    """Copy via a temp file in the destination folder + rename, so a running game never sees a half-written DLL."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".aiss-", dir=str(dst.parent))
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.chmod(tmp, 0o755)
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".aiss-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
            f.write(text)
        try:
            os.chmod(tmp, path.stat().st_mode & 0o7777)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------------------------------------------------
# VDF (Valve KeyValues text) parser, enough for libraryfolders.vdf and appmanifest_*.acf
# --------------------------------------------------------------------------------------------------------------------

def parse_vdf(text: str) -> dict:
    tokens = re.findall(r'"((?:[^"\\]|\\.)*)"|([{}])', text)
    root: dict = {}
    stack = [root]
    key = None
    for quoted, brace in tokens:
        if brace == "{":
            child: dict = {}
            if key is not None:
                stack[-1][key] = child
            stack.append(child)
            key = None
        elif brace == "}":
            if len(stack) > 1:
                stack.pop()
            key = None
        else:
            value = quoted.replace("\\\\", "\\").replace('\\"', '"')
            if key is None:
                key = value
            else:
                stack[-1][key] = value
                key = None
    return root


def _ci_get(d: dict, name: str, default=None):
    for k, v in d.items():
        if k.lower() == name.lower():
            return v
    return default


# --------------------------------------------------------------------------------------------------------------------
# Launch options
# --------------------------------------------------------------------------------------------------------------------

_ENV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.S)


def tokenize(s: str) -> list[str]:
    """Split on unquoted whitespace, keeping each token's original text (quotes included)."""
    tokens: list[str] = []
    cur = ""
    quote = None
    i = 0
    in_token = False
    while i < len(s):
        c = s[i]
        if quote:
            cur += c
            if c == "\\" and quote == '"' and i + 1 < len(s):
                cur += s[i + 1]
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
            cur += c
            in_token = True
        elif c == "\\" and i + 1 < len(s):
            cur += c + s[i + 1]
            i += 1
            in_token = True
        elif c.isspace():
            if in_token:
                tokens.append(cur)
                cur = ""
                in_token = False
        else:
            cur += c
            in_token = True
        i += 1
    if in_token:
        tokens.append(cur)
    return tokens


def _unquote(v: str) -> str:
    out = ""
    quote = None
    i = 0
    while i < len(v):
        c = v[i]
        if quote:
            if c == quote:
                quote = None
            elif c == "\\" and quote == '"' and i + 1 < len(v) and v[i + 1] in '"\\$`':
                out += v[i + 1]
                i += 1
            else:
                out += c
        elif c in "\"'":
            quote = c
        elif c == "\\" and i + 1 < len(v):
            out += v[i + 1]
            i += 1
        else:
            out += c
        i += 1
    return out


def _env_name(tok: str):
    m = _ENV_RE.match(tok)
    return m.group(1) if m else None


def _wdo_entries(value: str) -> list[tuple[list[str], str | None]]:
    entries = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            names, mode = part.split("=", 1)
        else:
            names, mode = part, None
        entries.append(([n.strip() for n in names.split(",") if n.strip()], mode))
    return entries


def _wdo_join(entries) -> str:
    parts = []
    for names, mode in entries:
        if not names:
            continue
        parts.append(",".join(names) + ("" if mode is None else "=" + mode))
    return ";".join(parts)


def _same_dll(a: str, b: str) -> bool:
    a = a.lower()
    b = b.lower()
    if a.endswith(".dll"):
        a = a[:-4]
    if b.endswith(".dll"):
        b = b[:-4]
    return a == b


def _wdo_mode(entries, shim: str):
    for names, mode in entries:
        if any(_same_dll(n, shim) for n in names):
            return True, mode
    return False, None


def _wdo_remove(entries, shim: str):
    out = []
    for names, mode in entries:
        names = [n for n in names if not _same_dll(n, shim)]
        if names:
            out.append((names, mode))
    return out



def pe_imports(path: str) -> set[str]:
    """Lower-case names of the DLLs a PE file imports statically (empty on any parse problem)."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
            if head[:2] != b"MZ":
                return set()
            pe = struct.unpack_from("<I", head, 0x3C)[0]
            if head[pe:pe + 4] != b"PE\0\0":
                return set()
            nsec, optsz = struct.unpack_from("<H", head, pe + 6)[0], struct.unpack_from("<H", head, pe + 20)[0]
            opt = pe + 24
            magic = struct.unpack_from("<H", head, opt)[0]
            dirs = opt + (112 if magic == 0x20B else 96)
            secs = [struct.unpack_from("<8sIIII", head, opt + optsz + i * 40) for i in range(nsec)]

            def off(rva):
                for _, vsize, va, rsize, raw in secs:
                    if va <= rva < va + max(vsize, rsize):
                        return rva - va + raw
                return None

            rva = struct.unpack_from("<I", head, dirs + 8)[0]  # data directory 1: imports
            names = set()
            while rva and len(names) < 512:
                o = off(rva)
                if o is None:
                    break
                f.seek(o)
                desc = f.read(20)
                if len(desc) < 20:
                    break
                name_rva = struct.unpack_from("<I", desc, 12)[0]
                if not name_rva:
                    break
                no = off(name_rva)
                if no is None:
                    break
                f.seek(no)
                names.add(f.read(256).split(b"\0")[0].decode("ascii", "replace").lower())
                rva += 20
            return names
    except (OSError, struct.error):
        return set()


PROXY_LOG = "aiss_nvngx.log"  # written by the proxy next to the running exe, rewritten on every launch


def last_run(exe_dir: str | None) -> dict | None:
    """Summary of the proxy's log from the game's last launch, or None if it never loaded there."""
    if not exe_dir:
        return None
    path = os.path.join(exe_dir, PROXY_LOG)
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
        when = int(os.path.getmtime(path))
    except OSError:
        return None
    api, size, error, upscaling = None, None, None, False
    for line in lines:
        m = re.search(r"\b(D3D12|D3D11|Vulkan) init\b", line)
        if m:
            api = m.group(1)
        m = re.search(r"DLSS create: (\d+x\d+) -> (\d+x\d+)", line)
        if m:
            size = f"{m.group(1)} -> {m.group(2)}"
        if "NSS context " in line:
            upscaling = True
        for bad in ("only runs under Proton", "failed", "missing or foreign", "not found"):
            if bad in line:
                error = line.replace("[aiss] ", "").strip()
    if error:
        state, text = "error", error
    elif upscaling:
        state, text = "active", f"DLSS on NSS ({api or '?'}, {size})"
    elif api:
        state, text = "loaded", f"proxy loaded ({api}); DLSS not switched on in the game"
    else:
        state, text = "loaded", "proxy loaded; the game did not initialise DLSS"
    return {"state": state, "text": text, "when": when}

def _split_options(current: str):
    toks = tokenize(current.strip())
    if COMMAND in toks:
        i = toks.index(COMMAND)
        pre, post, has_cmd = toks[:i], toks[i + 1:], True
    else:
        pre, post, has_cmd = [], toks, False
    n = 0
    while n < len(pre) and _env_name(pre[n]):
        n += 1
    return pre[:n], pre[n:], post, has_cmd


def _join_options(lead, wrappers, post, with_cmd: bool) -> str:
    if not with_cmd:
        return " ".join(lead + wrappers + post).strip()
    return " ".join(lead + wrappers + [COMMAND] + post).strip()


def settings_env(settings: dict) -> list[str]:
    env = ["AISS_LAYER=1"]
    control = str(settings.get("control") or "")
    if control:
        env.append(f'AISS_CONTROL="{control}"' if " " in control else f"AISS_CONTROL={control}")
    log = int(settings.get("log", 1))
    if log != 1:
        env.append(f"AISS_LOG={log}")
    return env


def strip_launch_options(current: str, rec: dict) -> str:
    """Remove everything the plugin added (per `rec`) from `current`; leaves the user's own options alone."""
    if not rec:
        return current
    lead, wrappers, post, has_cmd = _split_options(current)
    shim = rec.get("shim")
    new_lead = []
    for tok in lead:
        name = _env_name(tok)
        if name in MANAGED_ENV:
            continue
        if name == NVAPI_ENV and rec.get("nvapi_added"):
            continue
        if name == WDO_ENV and shim and rec.get("wdo_changed"):
            value = _unquote(tok.split("=", 1)[1])
            if value == rec.get("wdo_new_value"):
                orig = rec.get("wdo_orig_token")
                if orig:
                    new_lead.append(orig)
                continue
            entries = _wdo_remove(_wdo_entries(value), shim)
            prev_mode = rec.get("wdo_prev_mode")
            if prev_mode is not None or rec.get("wdo_prev_present"):
                entries.append(([shim], prev_mode))
            joined = _wdo_join(entries)
            if joined:
                new_lead.append(f'{WDO_ENV}="{joined}"')
            continue
        new_lead.append(tok)
    keep_cmd = has_cmd and not (rec.get("command_added") and not new_lead and not wrappers)
    if has_cmd and not keep_cmd:
        return _join_options([], [], post, False)
    return _join_options(new_lead, wrappers, post, has_cmd)


def build_launch_options(current: str, shim: str, settings: dict, rec: dict | None) -> tuple[str, dict]:
    """Merge the plugin's options into `current`. Idempotent; returns (new options, new ownership record)."""
    base = strip_launch_options(current, rec or {})
    lead, wrappers, post, has_cmd = _split_options(base)
    new_rec: dict = {"shim": shim, "command_added": not has_cmd}

    ours = []
    if any(_env_name(t) == NVAPI_ENV for t in lead):
        new_rec["nvapi_added"] = False
        # make sure it says 1
        lead = [f"{NVAPI_ENV}=1" if _env_name(t) == NVAPI_ENV else t for t in lead]
    else:
        new_rec["nvapi_added"] = True
        ours.append(f"{NVAPI_ENV}=1")
    ours += settings_env(settings)

    wdo_idx = next((i for i, t in enumerate(lead) if _env_name(t) == WDO_ENV), None)
    if wdo_idx is None:
        value = f"{shim}=n,b"
        ours.append(f'{WDO_ENV}="{value}"')
        new_rec.update(wdo_changed=True, wdo_new_value=value, wdo_orig_token=None, wdo_prev_present=False,
                       wdo_prev_mode=None)
    else:
        orig_tok = lead[wdo_idx]
        entries = _wdo_entries(_unquote(orig_tok.split("=", 1)[1]))
        present, mode = _wdo_mode(entries, shim)
        if present and mode is not None and mode.replace(" ", "") == "n,b":
            new_rec.update(wdo_changed=False)
        else:
            entries = _wdo_remove(entries, shim) + [([shim], "n,b")]
            value = _wdo_join(entries)
            lead[wdo_idx] = f'{WDO_ENV}="{value}"'
            new_rec.update(wdo_changed=True, wdo_new_value=value, wdo_orig_token=orig_tok,
                           wdo_prev_present=present, wdo_prev_mode=mode)

    new_lead = ours + [t for t in lead if _env_name(t) not in MANAGED_ENV]
    return _join_options(new_lead, wrappers, post, True), new_rec


# --------------------------------------------------------------------------------------------------------------------
# Wine registry (system.reg) editing
# --------------------------------------------------------------------------------------------------------------------

def _reg_header_re(key: str):
    escaped = key.replace("\\", "\\\\")
    return re.compile(r"^\[" + re.escape(escaped) + r"\]", re.I)


def _reg_value_re(name: str):
    return re.compile(r'^"' + re.escape(name) + r'"=', re.I)


def reg_has_value(text: str, key: str, name: str, data: str) -> bool:
    lines = text.split("\n")
    hdr = _reg_header_re(key)
    val = _reg_value_re(name)
    in_key = False
    for line in lines:
        if line.startswith("["):
            in_key = bool(hdr.match(line))
            continue
        if in_key and val.match(line):
            return line.split("=", 1)[1].strip().lower() == data.lower()
    return False


def reg_set_value(text: str, key: str, name: str, data: str, now: float | None = None) -> tuple[str, bool]:
    """Set `"name"=data` under [key]. Returns (new text, created_key)."""
    now = time.time() if now is None else now
    lines = text.split("\n")
    hdr = _reg_header_re(key)
    val = _reg_value_re(name)
    new_line = f'"{name}"={data}'
    start = next((i for i, l in enumerate(lines) if hdr.match(l)), None)
    if start is None:
        filetime = int((now + 11644473600) * 10_000_000)
        block = [f"[{key.replace(chr(92), chr(92) * 2)}] {int(now)}", f"#time={filetime:x}", new_line, ""]
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines + [""] + block), True
    end = start + 1
    while end < len(lines) and not lines[end].startswith("["):
        end += 1
    for i in range(start + 1, end):
        if val.match(lines[i]):
            lines[i] = new_line
            return "\n".join(lines), False
    insert = start + 1
    while insert < end and lines[insert].startswith("#"):
        insert += 1
    lines.insert(insert, new_line)
    return "\n".join(lines), False


def reg_delete_value(text: str, key: str, name: str, drop_empty_key: bool) -> str:
    lines = text.split("\n")
    hdr = _reg_header_re(key)
    val = _reg_value_re(name)
    start = next((i for i, l in enumerate(lines) if hdr.match(l)), None)
    if start is None:
        return text
    end = start + 1
    while end < len(lines) and not lines[end].startswith("["):
        end += 1
    body = [l for l in lines[start + 1:end] if not val.match(l)]
    if drop_empty_key and all(l == "" or l.startswith("#") for l in body):
        # drop the whole section including the blank line that separates it from the next one
        del lines[start:end]
        if start > 0 and start < len(lines) and lines[start - 1] == "" and lines[start] == "":
            del lines[start]
        while len(lines) > 1 and lines[-1] == "" and lines[-2] == "":
            lines.pop()
        return "\n".join(lines)
    lines[start + 1:end] = body
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------------------------------------------------

class AissCore:
    def __init__(self, *, home: Path, bin_dir: Path, state_file: Path, steam_roots: list[Path] | None = None,
                 layer_dir: Path | None = None, logger=None, proc_dir: Path = Path("/proc"),
                 running_check: Callable[[str], bool] | None = None):
        self.home = Path(home)
        self.bin_dir = Path(bin_dir)
        self.state_file = Path(state_file)
        self.steam_roots = steam_roots or [
            self.home / ".local/share/Steam",
            self.home / ".steam/steam",
            self.home / ".var/app/com.valvesoftware.Steam/.local/share/Steam",
        ]
        self.layer_dir = layer_dir or (self.home / ".local/share/vulkan/implicit_layer.d")
        self.log = logger or Logger()
        self.proc_dir = proc_dir
        self._running_check = running_check
        self._bundle_hash: dict[str, str] = {}
        self._scan_cache: dict[str, tuple[str, dict]] = {}
        self.state = self._load_state()

    # ---- state ----------------------------------------------------------------------------------------------------

    def _load_state(self) -> dict:
        state = {"version": 1, "settings": dict(DEFAULT_SETTINGS), "games": {}, "layer": {}}
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                state.update({k: v for k, v in loaded.items() if k in state})
                state["settings"] = {**DEFAULT_SETTINGS, **(loaded.get("settings") or {})}
        except FileNotFoundError:
            pass
        except Exception as e:
            self.log.error(f"state file unreadable ({e}); starting fresh")
        return state

    def save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.state_file, json.dumps(self.state, indent=2, sort_keys=True))

    def _game_state(self, appid: str) -> dict:
        return self.state["games"].setdefault(str(appid), {"enabled": False})

    # ---- bundle ---------------------------------------------------------------------------------------------------

    def bundle_path(self, key: str) -> Path:
        return self.bin_dir / BIN_FILES[key]

    def bundle_hash(self, key: str) -> str:
        if key not in self._bundle_hash:
            self._bundle_hash[key] = sha256_file(self.bundle_path(key))
        return self._bundle_hash[key]

    def bundle_missing(self) -> list[str]:
        return [name for key, name in BIN_FILES.items() if not self.bundle_path(key).is_file()]

    def bundle_info(self) -> dict:
        info = {"missing": self.bundle_missing(), "version": ""}
        try:
            info["version"] = (self.bin_dir / "VERSION").read_text(encoding="utf-8").strip()
        except OSError:
            pass
        return info

    # ---- Steam library --------------------------------------------------------------------------------------------

    def library_paths(self) -> list[Path]:
        libs: list[Path] = []
        seen: set[str] = set()

        def add(p: Path):
            try:
                key = str(p.resolve())
            except OSError:
                key = str(p)
            if key not in seen:
                seen.add(key)
                libs.append(p)

        for root in self.steam_roots:
            vdf = root / "steamapps" / "libraryfolders.vdf"
            if (root / "steamapps").is_dir():
                add(root)
            if not vdf.is_file():
                continue
            try:
                data = parse_vdf(vdf.read_text(encoding="utf-8", errors="replace"))
            except OSError as e:
                self.log.error(f"cannot read {vdf}: {e}")
                continue
            folders = _ci_get(data, "libraryfolders") or {}
            for entry in folders.values():
                if isinstance(entry, dict) and _ci_get(entry, "path"):
                    add(Path(_ci_get(entry, "path")))
        return [p for p in libs if (p / "steamapps").is_dir()]

    def installed_apps(self) -> list[dict]:
        apps: dict[str, dict] = {}
        for lib in self.library_paths():
            steamapps = lib / "steamapps"
            for acf in sorted(steamapps.glob("appmanifest_*.acf")):
                try:
                    st = _ci_get(parse_vdf(acf.read_text(encoding="utf-8", errors="replace")), "AppState") or {}
                except OSError:
                    continue
                appid = str(_ci_get(st, "appid", "") or "")
                name = _ci_get(st, "name", "") or ""
                installdir = _ci_get(st, "installdir", "") or ""
                if not appid or not installdir:
                    continue
                if any(s in name.lower() for s in SKIP_NAME_SUBSTRINGS):
                    continue
                apps[appid] = {
                    "appid": appid,
                    "name": name or installdir,
                    "library": str(lib),
                    "install_path": str(steamapps / "common" / installdir),
                    "updated": str(_ci_get(st, "LastUpdated", "") or ""),
                    "buildid": str(_ci_get(st, "buildid", "") or ""),
                }
        return sorted(apps.values(), key=lambda a: a["name"].lower())

    @staticmethod
    def walk_game(root: Path) -> dict:
        """Depth-limited walk collecting DLSS / Streamline DLLs and executables (paths relative to root)."""
        found = {"dlss": [], "sl": [], "exes": []}
        entries = 0
        stack = [(str(root), 0)]
        while stack:
            d, depth = stack.pop()
            try:
                it = os.scandir(d)
            except OSError:
                continue
            with it:
                for e in it:
                    entries += 1
                    if entries > SCAN_MAX_ENTRIES:
                        return found
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if depth < SCAN_MAX_DEPTH:
                                stack.append((e.path, depth + 1))
                            continue
                        if not e.is_file():
                            continue
                    except OSError:
                        continue
                    low = e.name.lower()
                    if low == DLSS_DLL:
                        found["dlss"].append(e.path)
                    elif low == SL_INTERPOSER:
                        found["sl"].append(e.path)
                    elif low.endswith(".exe"):
                        found["exes"].append(e.path)
        return found

    @staticmethod
    def exe_score(exe: str, root: str, game_name: str) -> int:
        rel = os.path.relpath(exe, root).replace("\\", "/").lower()
        name = os.path.basename(rel)
        stem = name[:-4]
        score = 0
        if stem.endswith("-win64-shipping") or stem.endswith("-wingdk-shipping"):
            score += 400
        elif "shipping" in stem:
            score += 200
        if "/binaries/win64/" in "/" + rel:
            score += 150
        if rel.startswith("engine/"):
            score -= 400
        if "/" not in rel:
            score += 40
        clean = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        g, r, s = clean(game_name), clean(os.path.basename(root)), clean(stem)
        if s and g and (s in g or g in s):
            score += 120
        if s and r and (s in r or r in s):
            score += 100
        for bad in BAD_EXE_SUBSTRINGS:
            if bad in name:
                score -= 300
        score -= 5 * rel.count("/")
        return score

    def analyse_game(self, app: dict, force: bool = False) -> dict:
        """Cached per (install path, LastUpdated, buildid)."""
        key = f"{app['install_path']}|{app.get('updated')}|{app.get('buildid')}"
        cached = self._scan_cache.get(app["appid"])
        if cached and cached[0] == key and not force:
            return cached[1]
        root = app["install_path"]
        info = {"has_dlss": False, "exe": None, "exe_dir": None, "sl_dirs": [], "shim": None, "targets": []}
        if os.path.isdir(root):
            found = self.walk_game(Path(root))
            info["has_dlss"] = bool(found["dlss"])
            if info["has_dlss"]:
                exes = found["exes"]
                if exes:
                    best = max(exes, key=lambda e: (self.exe_score(e, root, app["name"]), -len(e)))
                    info["exe"] = best
                    info["exe_dir"] = os.path.dirname(best)
                else:
                    info["exe_dir"] = root
                sl_dirs = sorted({os.path.dirname(p) for p in found["sl"]})
                info["sl_dirs"] = sl_dirs
                # The shim must be loaded before NGX initialises, so it takes the name of a DLL the exe imports at
                # start-up: dxgi.dll (most D3D12/D3D11 games, Unreal included) or else dbghelp.dll (e.g. Enshrouded,
                # a Vulkan game). Only when the exe imports neither does the Streamline layout decide: Streamline's
                # sl.interposer.dll imports dbghelp. Unreal ships and preloads its own dbghelp.dll, which is why
                # dxgi wins when both are imported.
                imports = pe_imports(info["exe"]) if info["exe"] else set()
                if "dxgi.dll" in imports or "dbghelp.dll" in imports:
                    info["shim"] = "dxgi" if "dxgi.dll" in imports else "dbghelp"
                    info["targets"] = [info["exe_dir"]]
                elif sl_dirs:
                    info["shim"] = "dbghelp"
                    targets = [info["exe_dir"]]
                    # also next to sl.interposer.dll when it lives elsewhere (e.g. DOOM: streamline/production)
                    if info["exe_dir"] not in sl_dirs:
                        pick = sorted(sl_dirs, key=lambda d: ("debug" in d.lower() or "develop" in d.lower(),
                                                              "production" not in d.lower(), len(d)))[0]
                        targets.append(pick)
                    info["targets"] = targets
                else:
                    info["shim"] = "dxgi"
                    info["targets"] = [info["exe_dir"]]
        self._scan_cache[app["appid"]] = (key, info)
        return info

    def find_prefix(self, appid: str, library: str | None = None) -> Path | None:
        libs = self.library_paths()
        if library:
            libs.sort(key=lambda p: str(p) != library)
        for lib in libs:
            pfx = lib / "steamapps" / "compatdata" / str(appid) / "pfx"
            if (pfx / "drive_c" / "windows" / "system32").is_dir() and (pfx / "system.reg").is_file():
                return pfx
        return None

    def is_running(self, appid: str) -> bool:
        if self._running_check:
            return self._running_check(str(appid))
        needles = (f"SteamAppId={appid}\0".encode(), f"STEAM_COMPAT_APP_ID={appid}\0".encode(),
                   f"SteamGameId={appid}\0".encode())
        try:
            pids = [p for p in os.listdir(self.proc_dir) if p.isdigit()]
        except OSError:
            return False
        for pid in pids:
            try:
                with open(self.proc_dir / pid / "environ", "rb") as f:
                    env = b"\0" + f.read()
            except OSError:
                continue
            if any(b"\0" + n in env for n in needles):
                return True
        return False

    # ---- file ownership -------------------------------------------------------------------------------------------

    def _classify(self, path: Path, bundle_key: str, recorded: str | None, marker: bytes) -> str:
        """'missing' | 'current' (identical to bundle) | 'ours' (other build of ours) | 'foreign'."""
        if not path.exists():
            return "missing"
        try:
            if path.stat().st_size == self.bundle_path(bundle_key).stat().st_size and \
                    sha256_file(path) == self.bundle_hash(bundle_key):
                return "current"
            if recorded and sha256_file(path) == recorded:
                return "ours"
        except OSError:
            return "foreign"
        return "ours" if file_contains(path, marker) else "foreign"

    # ---- per-game status ------------------------------------------------------------------------------------------

    def game_view(self, app: dict, info: dict) -> dict:
        appid = app["appid"]
        gs = self.state["games"].get(appid, {})
        pfx = self.find_prefix(appid, app["library"])
        view = {
            "appid": appid, "name": app["name"], "install_path": app["install_path"],
            "shim": info["shim"], "exe": info["exe"], "targets": info["targets"],
            "streamline": bool(info["sl_dirs"]), "prefix": str(pfx) if pfx else None,
            "enabled": bool(gs.get("enabled")), "message": gs.get("message", ""),
            "launch_applied": bool(gs.get("launch")), "reg_pending": bool(gs.get("reg_pending")),
            "last_run": last_run(info["exe_dir"]),
        }
        conflicts = []
        shim_ok = True
        for t in info["targets"]:
            c = self._classify(Path(t) / f"{info['shim']}.dll", info["shim"],
                               (gs.get("files") or {}).get(str(Path(t) / f"{info['shim']}.dll")), SHIM_MARKER)
            if c == "foreign":
                conflicts.append(str(Path(t) / f"{info['shim']}.dll"))
            if c != "current":
                shim_ok = False
        core_ok = False
        if pfx:
            core_ok = self._classify(pfx / "drive_c/windows/system32" / CORE_DEST_NAME, "core", None,
                                     CORE_MARKER) == "current"
        if conflicts:
            status = STATUS_CONFLICT
            view["message"] = "Existing file is not ours: " + ", ".join(conflicts)
        elif not pfx:
            status = STATUS_PREFIX_MISSING
        elif shim_ok and core_ok:
            status = STATUS_INSTALLED if view["enabled"] else STATUS_NOT_INSTALLED
            if not view["enabled"]:
                view["message"] = "files present but not enabled"
        elif view["enabled"]:
            status = STATUS_PARTIAL
        else:
            status = STATUS_NOT_INSTALLED
        view["status"] = status
        view["conflicts"] = conflicts
        return view

    def scan(self, force: bool = False) -> list[dict]:
        out = []
        for app in self.installed_apps():
            info = self.analyse_game(app, force)
            gs = self.state["games"].get(app["appid"])
            if info["has_dlss"] or (gs and (gs.get("enabled") or gs.get("launch"))):
                if not info["has_dlss"]:
                    info = {**info, "targets": gs.get("targets", []), "shim": gs.get("shim")}
                out.append(self.game_view(app, info))
        return out

    def _app(self, appid: str) -> dict | None:
        return next((a for a in self.installed_apps() if a["appid"] == str(appid)), None)

    # ---- install / uninstall --------------------------------------------------------------------------------------

    def ensure_layer(self) -> dict:
        rec = self.state.setdefault("layer", {})
        files = rec.setdefault("files", {})
        result = {"ok": True, "message": ""}
        for key in ("layer_so", "layer_json"):
            dst = self.layer_dir / BIN_FILES[key]
            src = self.bundle_path(key)
            if not src.is_file():
                return {"ok": False, "message": f"bundle missing {src.name}"}
            if dst.exists():
                h = sha256_file(dst)
                if h == self.bundle_hash(key):
                    files[str(dst)] = h
                    continue
                if files.get(str(dst)) != h:
                    result = {"ok": False, "message": f"{dst} differs from the bundled build and was not installed "
                                                      f"by the plugin; left untouched"}
                    continue
            atomic_copy(src, dst)
            if key == "layer_so":
                os.chmod(dst, 0o755)
            else:
                os.chmod(dst, 0o644)
            files[str(dst)] = self.bundle_hash(key)
        return result

    def remove_layer(self) -> None:
        files = self.state.get("layer", {}).get("files", {})
        for path, h in list(files.items()):
            p = Path(path)
            try:
                if p.exists() and sha256_file(p) == h:
                    p.unlink()
            except OSError as e:
                self.log.error(f"cannot remove {p}: {e}")
            files.pop(path, None)

    def _install_core(self, gs: dict, pfx: Path) -> None:
        dst = pfx / "drive_c/windows/system32" / CORE_DEST_NAME
        c = self._classify(dst, "core", gs.get("core_hash"), CORE_MARKER)
        if c != "current":
            backup = dst.with_name(dst.name + CORE_BACKUP_SUFFIX)
            if c != "missing" and not gs.get("core_hash") and not backup.exists():
                # first time we take over this prefix: keep whatever was there, restored on uninstall
                shutil.copy2(dst, backup)
                gs["core_backup"] = str(backup)
            atomic_copy(self.bundle_path("core"), dst)
        gs["core_hash"] = self.bundle_hash("core")
        gs["core_path"] = str(dst)

    def _remove_core(self, gs: dict) -> list[str]:
        warnings = []
        path = gs.get("core_path")
        if not path:
            return warnings
        dst = Path(path)
        c = self._classify(dst, "core", gs.get("core_hash"), CORE_MARKER)
        if c in ("current", "ours"):
            dst.unlink()
        elif c == "foreign":
            warnings.append(f"{dst} was replaced by something else; left in place")
        backup = gs.get("core_backup")
        if backup and Path(backup).exists() and not dst.exists():
            os.replace(backup, dst)
        gs.pop("core_path", None)
        gs.pop("core_hash", None)
        gs.pop("core_backup", None)
        return warnings

    def _apply_registry(self, appid: str, gs: dict, pfx: Path) -> None:
        reg = pfx / "system.reg"
        text = reg.read_text(encoding="utf-8", errors="surrogateescape")
        if reg_has_value(text, REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01"):
            gs["reg_pending"] = False
            gs.setdefault("reg_added", False)
            return
        if self.is_running(appid):
            gs["reg_pending"] = True
            return
        backup = reg.with_name(reg.name + REG_BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(reg, backup)
        new_text, created = reg_set_value(text, REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01")
        atomic_write_text(reg, new_text)
        gs["reg_added"] = True
        gs["reg_created_key"] = created
        gs["reg_pending"] = False
        gs["reg_path"] = str(reg)

    def _remove_registry(self, appid: str, gs: dict) -> bool:
        """Returns False if it had to be deferred (game running)."""
        if not gs.get("reg_added") or not gs.get("reg_path"):
            gs.pop("reg_added", None)
            return True
        reg = Path(gs["reg_path"])
        if not reg.exists():
            gs.pop("reg_added", None)
            return True
        if self.is_running(appid):
            return False
        text = reg.read_text(encoding="utf-8", errors="surrogateescape")
        new_text = reg_delete_value(text, REG_KEY_PATH, SIG_OVERRIDE_VALUE, bool(gs.get("reg_created_key")))
        if new_text != text:
            atomic_write_text(reg, new_text)
        for k in ("reg_added", "reg_created_key", "reg_path", "reg_pending"):
            gs.pop(k, None)
        return True

    def install_game(self, appid: str, force_rescan: bool = True) -> dict:
        appid = str(appid)
        app = self._app(appid)
        if not app:
            return {"ok": False, "status": "error", "message": "game not installed"}
        missing = self.bundle_missing()
        if missing:
            return {"ok": False, "status": "error", "message": "plugin bundle incomplete: " + ", ".join(missing)}
        info = self.analyse_game(app, force=force_rescan)
        if not info["has_dlss"]:
            return {"ok": False, "status": "error", "message": f"{DLSS_DLL} not found in the game folder"}
        gs = self._game_state(appid)
        files = gs.setdefault("files", {})

        # A game update may have switched the shim (Streamline added/removed): drop the old one first.
        if gs.get("shim") and gs["shim"] != info["shim"]:
            self._remove_shims(gs)
            files = gs.setdefault("files", {})

        shim_src = self.bundle_path(info["shim"])
        dsts = [Path(t) / f"{info['shim']}.dll" for t in info["targets"]]
        conflicts = [str(d) for d in dsts
                     if self._classify(d, info["shim"], files.get(str(d)), SHIM_MARKER) == "foreign"]
        if conflicts:
            gs["message"] = "Existing file is not ours: " + ", ".join(conflicts)
            self.save_state()
            return {"ok": False, "status": STATUS_CONFLICT, "message": gs["message"], "conflicts": conflicts}

        for d in dsts:
            if self._classify(d, info["shim"], files.get(str(d)), SHIM_MARKER) != "current":
                atomic_copy(shim_src, d)
            files[str(d)] = self.bundle_hash(info["shim"])
        gs.update(enabled=True, name=app["name"], shim=info["shim"], targets=info["targets"],
                  install_path=app["install_path"])

        layer = self.ensure_layer()
        messages = [] if layer["ok"] else [layer["message"]]

        pfx = self.find_prefix(appid, app["library"])
        status = STATUS_INSTALLED
        if pfx is None:
            status = STATUS_PREFIX_MISSING
            messages.append("Proton prefix not created yet: launch the game once; setup finishes automatically")
        else:
            self._install_core(gs, pfx)
            try:
                self._apply_registry(appid, gs, pfx)
                if gs.get("reg_pending"):
                    messages.append("registry update deferred until the game exits")
            except OSError as e:
                messages.append(f"registry not updated: {e}")
        gs["message"] = "; ".join(messages)
        self.save_state()
        return {"ok": True, "status": status, "message": gs["message"], "shim": info["shim"]}

    def _remove_shims(self, gs: dict) -> list[str]:
        warnings = []
        for path, h in list((gs.get("files") or {}).items()):
            p = Path(path)
            shim = p.stem.lower() if p.stem.lower() in ("dbghelp", "dxgi") else None
            if shim and p.exists():
                c = self._classify(p, shim, h, SHIM_MARKER)
                if c in ("current", "ours"):
                    p.unlink()
                else:
                    warnings.append(f"{p} is no longer ours; left in place")
            gs["files"].pop(path, None)
        return warnings

    def uninstall_game(self, appid: str) -> dict:
        appid = str(appid)
        gs = self.state["games"].get(appid)
        if not gs:
            return {"ok": True, "status": STATUS_NOT_INSTALLED, "message": ""}
        warnings = self._remove_shims(gs)
        warnings += self._remove_core(gs)
        if not self._remove_registry(appid, gs):
            gs["reg_remove_pending"] = True
            warnings.append("registry cleanup deferred until the game exits")
        else:
            gs.pop("reg_remove_pending", None)
        gs["enabled"] = False
        gs["message"] = "; ".join(warnings)
        self.save_state()
        return {"ok": True, "status": STATUS_NOT_INSTALLED, "message": gs["message"]}

    def _needs_repair(self, app: dict, info: dict, gs: dict) -> bool:
        if gs.get("shim") != info["shim"] or sorted(gs.get("targets") or []) != sorted(info["targets"]):
            return True
        for t in info["targets"]:
            d = Path(t) / f"{info['shim']}.dll"
            if self._classify(d, info["shim"], (gs.get("files") or {}).get(str(d)), SHIM_MARKER) != "current":
                return True  # missing (game update), older build of ours, or a conflict to report
        pfx = self.find_prefix(app["appid"], app["library"])
        if pfx is None:
            return False  # nothing more to do until the game has been launched once
        if self._classify(pfx / "drive_c/windows/system32" / CORE_DEST_NAME, "core", gs.get("core_hash"),
                          CORE_MARKER) != "current":
            return True
        return bool(gs.get("reg_pending")) or not gs.get("core_path")

    def verify_all(self) -> list[dict]:
        """Repair enabled games (a game update can delete the shim, the prefix may exist by now, a newer plugin
        bundles newer binaries) and finish deferred registry work."""
        try:
            self.write_control()
        except OSError as e:
            self.log.warning(f"cannot write {self.control_path}: {e}")
        results = []
        apps = {a["appid"]: a for a in self.installed_apps()}
        for appid, gs in list(self.state["games"].items()):
            try:
                if gs.get("enabled"):
                    app = apps.get(appid)
                    if not app:
                        continue
                    # cached per LastUpdated/buildid, so a game update triggers a fresh walk
                    info = self.analyse_game(app)
                    if not info["has_dlss"] or not self._needs_repair(app, info, gs):
                        continue
                    r = self.install_game(appid, force_rescan=False)
                    r["appid"] = appid
                    results.append(r)
                    self.log.info(f"repair {appid}: {r['status']} {r['message']}")
                elif gs.get("reg_remove_pending"):
                    if self._remove_registry(appid, gs):
                        gs.pop("reg_remove_pending", None)
                        self.save_state()
            except Exception as e:  # keep going for the other games
                self.log.error(f"verify {appid} failed: {e}")
                results.append({"appid": appid, "ok": False, "status": "error", "message": str(e)})
        if any(gs.get("enabled") for gs in self.state["games"].values()):
            self.ensure_layer()
            self.save_state()
        return results

    # ---- launch options -------------------------------------------------------------------------------------------

    def launch_options(self, appid: str, current: str, enable: bool) -> dict:
        """Compute the new launch options; the frontend applies them and then calls confirm_launch_options."""
        gs = self._game_state(str(appid))
        rec = gs.get("launch") or {}
        if enable:
            shim = gs.get("shim")
            if not shim:
                return {"ok": False, "options": current, "message": "game not set up"}
            new, new_rec = build_launch_options(current or "", shim, self.env_settings(), rec)
            gs["launch"] = new_rec
        else:
            new = strip_launch_options(current or "", rec)
            gs.pop("launch", None)
        self.save_state()
        return {"ok": True, "options": new, "changed": new != (current or "")}

    # The proxy re-reads this file while the game runs, so a model change applies without restarting the game.
    @property
    def control_path(self) -> Path:
        return self.home / ".local/share/aiss-dlss-nss/control"

    def env_settings(self) -> dict:
        return {**self.state["settings"], "control": str(self.control_path)}

    def write_control(self) -> None:
        q = int(self.state["settings"].get("quality", 0))
        if q not in (0, 1, 2):
            q = 0
        fast = 1 if self.state["settings"].get("fast_math") else 0
        capture = int(self.state.get("capture_seq", 0))
        self.control_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.control_path.with_suffix(".tmp")
        tmp.write_text(f"{q} {fast} {capture}\n")
        os.replace(tmp, self.control_path)

    @property
    def captures_dir(self) -> Path:
        return self.control_path.parent / "captures"

    def request_capture(self) -> dict:
        """Ask every running game with the proxy to save its next frames' inputs (the "Capture frames" button)."""
        self.state["capture_seq"] = int(self.state.get("capture_seq", 0)) + 1
        self.save_state()
        self.write_control()
        return {"seq": self.state["capture_seq"], "dir": str(self.captures_dir)}

    def latest_capture(self, since: float = 0.0) -> dict | None:
        """Newest capture folder modified after `since` (epoch seconds), with how many frames it holds so far."""
        if not self.captures_dir.is_dir():
            return None
        dirs = [d for d in self.captures_dir.iterdir() if d.is_dir() and d.stat().st_mtime >= since]
        if not dirs:
            return None
        d = max(dirs, key=lambda p: p.stat().st_mtime)
        frames = len(list(d.glob("*_output.raw")))
        return {"name": d.name, "path": str(d), "frames": frames}

    def set_settings(self, settings: dict) -> dict:
        s = dict(self.state["settings"])
        s.pop("ratio", None)  # option removed; drop it from older state files
        if "quality" in settings:
            q = int(settings["quality"])
            if q not in (-1, 0, 1, 2):
                raise ValueError("quality must be 0, 1 or 2")
            s["quality"] = 0 if q == -1 else q
        if "log" in settings:
            lv = int(settings["log"])
            if not 0 <= lv <= 4:
                raise ValueError("log must be 0..4")
            s["log"] = lv
        if "fast_math" in settings:
            fm = int(settings["fast_math"])
            if fm not in (0, 1):
                raise ValueError("fast_math must be 0 or 1")
            s["fast_math"] = fm
        self.state["settings"] = s
        self.save_state()
        self.write_control()
        return s
