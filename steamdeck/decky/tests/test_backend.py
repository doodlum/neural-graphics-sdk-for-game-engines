"""Standalone tests for the plugin backend (no Decky needed).

    python3 steamdeck/decky/tests/test_backend.py [-v]

Everything that writes happens in a temporary fake Steam tree. The real Steam library is only *scanned*
(test_real_library_scan_readonly), with state and layer paths redirected to the temp dir.
Set AISS_TEST_BIN to use another bin/ dir (default: ../bin, falling back to the repo build dirs).
"""

import hashlib
import os
import shutil
import sys
import struct
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN = HERE.parent
REPO = PLUGIN.parent.parent
sys.path.insert(0, str(PLUGIN / "py_modules"))

import aiss_core as core  # noqa: E402
from aiss_core import (AissCore, build_launch_options, strip_launch_options, reg_set_value,  # noqa: E402
                       reg_delete_value, reg_has_value, REG_KEY_PATH, SIG_OVERRIDE_VALUE)

S0 = {"upscaler": 0, "log": 1, "fast_math": 0}


def bin_dir(tmp: Path) -> Path:
    env = os.environ.get("AISS_TEST_BIN")
    if env:
        return Path(env)
    if all((PLUGIN / "bin" / n).is_file() for k, n in core.BIN_FILES.items() if k not in core.OPTIONAL_BIN):
        return PLUGIN / "bin"
    d = tmp / "bin"
    d.mkdir(exist_ok=True)
    for src in (REPO / "build_win_nvngx/nvngx.dll", REPO / "build_win_loader/dbghelp/dbghelp.dll",
                REPO / "build_win_loader/dxgi/dxgi.dll", REPO / "build_layer/libVkLayer_aiss_ngx.so",
                REPO / "build_layer/VkLayer_aiss_ngx.json"):
        shutil.copy(src, d / src.name)
    xess = REPO.parent / "third_party/xess/sdk"
    if (xess / "bin/libxess.dll").is_file():
        os.symlink(xess / "bin/libxess.dll", d / "libxess.dll")  # 78 MB: linked, not copied
        shutil.copy(xess / "LICENSE.txt", d / "libxess-LICENSE.txt")
    else:  # stand-ins: the tests only move the files around
        (d / "libxess.dll").write_bytes(b"MZ stand-in for Intel libxess.dll")
        (d / "libxess-LICENSE.txt").write_text("stand-in licence\n")
    for amd in ("amdvkffx64.dll", "amdxcffx64.dll"):
        src = REPO.parent / "third_party/amd" / amd
        if src.is_file():
            os.symlink(src, d / amd)  # tens of MB: linked, not copied
        else:
            (d / amd).write_bytes(b"MZ stand-in for AMD " + amd.encode())
    return d



def sha256_of(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def minimal_pe(imports: list[str]) -> bytes:
    """A PE32+ image whose import table names `imports` (enough for pe_imports)."""
    img = bytearray(0x400)
    img[0:2] = b"MZ"
    struct.pack_into("<I", img, 0x3C, 0x40)
    img[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", img, 0x44, 0x8664, 1, 0, 0, 0, 240, 0x22)
    opt = 0x58
    struct.pack_into("<H", img, opt, 0x20B)
    struct.pack_into("<II", img, opt + 112 + 8, 0x1000, 20 * (len(imports) + 1))  # import directory
    struct.pack_into("<8sIIIII", img, opt + 240, b".idata", 0x200, 0x1000, 0x200, 0x200, 0)
    name_rva = 0x1100
    for i, name in enumerate(imports):
        struct.pack_into("<IIIII", img, 0x200 + 20 * i, 0, 0, 0, name_rva, 0)
        raw = name.encode() + b"\0"
        img[0x200 + (name_rva - 0x1000):0x200 + (name_rva - 0x1000) + len(raw)] = raw
        name_rva += 0x20
    return bytes(img)

class Quiet:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(m)

    warning = error = info


# ------------------------------------------------------------------------------------------------------------------
class LaunchOptionTests(unittest.TestCase):
    def roundtrip(self, current, shim="dxgi", settings=S0, expect=None):
        new, rec = build_launch_options(current, shim, settings, None)
        if expect is not None:
            self.assertEqual(new, expect)
        again, rec2 = build_launch_options(new, shim, settings, rec)
        self.assertEqual(again, new, "not idempotent")
        self.assertEqual(strip_launch_options(again, rec2), current.strip(), "strip did not restore")
        return new, rec2

    def test_empty(self):
        self.roundtrip("", expect='PROTON_FORCE_NVAPI=1 AISS_LAYER=1 WINEDLLOVERRIDES="dxgi=n,b" %command%')

    def test_args_without_command(self):
        self.roundtrip("-dx12 -skipintro",
                       expect='PROTON_FORCE_NVAPI=1 AISS_LAYER=1 WINEDLLOVERRIDES="dxgi=n,b" %command% -dx12 -skipintro')

    def test_wrapper(self):
        self.roundtrip("gamemoderun %command% -foo",
                       expect='PROTON_FORCE_NVAPI=1 AISS_LAYER=1 WINEDLLOVERRIDES="dxgi=n,b" gamemoderun %command% -foo')

    def test_existing_wdo_framegen_style(self):
        new, _ = self.roundtrip("WINEDLLOVERRIDES=dxgi=n,b SteamDeck=0 %command%", shim="dbghelp")
        self.assertIn('WINEDLLOVERRIDES="dxgi=n,b;dbghelp=n,b"', new)

    def test_existing_wdo_already_ours(self):
        cur = 'WINEDLLOVERRIDES="dxgi=n,b" %command%'
        new, rec = self.roundtrip(cur)
        self.assertFalse(rec["wdo_changed"])
        self.assertEqual(new.count("dxgi=n,b"), 1)

    def test_existing_wdo_other_mode_restored(self):
        new, _ = self.roundtrip('WINEDLLOVERRIDES="dxgi=b;winmm=n,b" %command%')
        self.assertIn('WINEDLLOVERRIDES="winmm=n,b;dxgi=n,b"', new)

    def test_preexisting_nvapi_kept(self):
        new, rec = self.roundtrip("PROTON_FORCE_NVAPI=1 DXVK_HUD=fps %command%")
        self.assertFalse(rec["nvapi_added"])
        self.assertEqual(new.count("PROTON_FORCE_NVAPI"), 1)

    def test_quoted_user_values(self):
        self.roundtrip('MANGOHUD_CONFIG="fps_limit=30, no_display" mangohud %command% --name "My Save"')

    def test_settings_change(self):
        new, rec = build_launch_options("", "dxgi", S0, None)
        s2 = {"log": 2}
        new2, rec2 = build_launch_options(new, "dxgi", s2, rec)
        self.assertEqual(new2, 'PROTON_FORCE_NVAPI=1 AISS_LAYER=1 AISS_LOG=2 '
                               'WINEDLLOVERRIDES="dxgi=n,b" %command%')
        new3, rec3 = build_launch_options(new2, "dxgi", S0, rec2)
        self.assertEqual(new3, new)
        self.assertEqual(strip_launch_options(new3, rec3), "")

    def test_user_edit_after_enable(self):
        new, rec = build_launch_options("%command% -foo", "dxgi", S0, None)
        edited = new.replace('"dxgi=n,b"', '"dxgi=n,b;foo=d"').replace("%command% -foo", "%command% -foo -bar")
        self.assertEqual(strip_launch_options(edited, rec), 'WINEDLLOVERRIDES="foo=d" %command% -foo -bar')

    def test_shim_switch(self):
        new, rec = build_launch_options("-x", "dxgi", S0, None)
        new2, rec2 = build_launch_options(new, "dbghelp", S0, rec)
        self.assertIn('"dbghelp=n,b"', new2)
        self.assertNotIn("dxgi", new2)
        self.assertEqual(strip_launch_options(new2, rec2), "-x")


# ------------------------------------------------------------------------------------------------------------------
REG_SAMPLE = """WINE REGISTRY Version 2
;; All keys relative to \\\\Machine

#arch=win64

[Software\\\\Microsoft\\\\Foo] 1700000000
#time=1d9
"A"="b"

[Software\\\\NVIDIA Corporation\\\\Global\\\\NGXCore] 1791989093
#time=1dd5bea929bf48a
"FullPath"="C:\\\\Windows\\\\System32"

[Software\\\\Zed] 1700000000
"Z"=dword:00000001
"""


class RegistryTests(unittest.TestCase):
    def test_add_remove_new_key(self):
        new, created = reg_set_value(REG_SAMPLE, REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01", now=1792970778)
        self.assertTrue(created)
        self.assertIn('[Software\\\\NVIDIA Corporation\\\\Global] 1792970778', new)
        self.assertTrue(reg_has_value(new, REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01"))
        self.assertFalse(reg_has_value(REG_SAMPLE, REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01"))
        self.assertEqual(reg_delete_value(new, REG_KEY_PATH, SIG_OVERRIDE_VALUE, True), REG_SAMPLE)

    def test_existing_key_without_value(self):
        base = REG_SAMPLE.replace("[Software\\\\Zed]", "[Software\\\\NVIDIA Corporation\\\\Global] 1\n#time=5\n"
                                  "\"Other\"=dword:1\n\n[Software\\\\Zed]")
        new, created = reg_set_value(base, REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01")
        self.assertFalse(created)
        self.assertIn('#time=5\n"{41FCC608-8496-4DEF-B43E-7D9BD675A6FF}"=hex:01\n"Other"', new)
        self.assertEqual(reg_delete_value(new, REG_KEY_PATH, SIG_OVERRIDE_VALUE, False), base)

    def test_real_testprefix_registry_readonly(self):
        reg = Path("/home/deck/aiss/testprefix/pfx/system.reg")
        if not reg.exists():
            self.skipTest("no test prefix")
        text = reg.read_text(encoding="utf-8", errors="surrogateescape")
        self.assertTrue(reg_has_value(text, REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01"))
        # removing and re-adding inside the existing key round-trips the text
        stripped = reg_delete_value(text, REG_KEY_PATH, SIG_OVERRIDE_VALUE, False)
        self.assertFalse(reg_has_value(stripped, REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01"))
        self.assertEqual(reg_set_value(stripped, REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01")[0], text)


# ------------------------------------------------------------------------------------------------------------------
def tree_digest(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class FakeLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aiss-test-"))
        self.bin = bin_dir(self.tmp)
        self.home = self.tmp / "home"
        self.steam = self.home / ".local/share/Steam"
        self.lib2 = self.tmp / "sdcard"
        self.running: set[str] = set()
        (self.steam / "steamapps/common").mkdir(parents=True)
        (self.lib2 / "steamapps/common").mkdir(parents=True)
        (self.steam / "steamapps/libraryfolders.vdf").write_text(f'''"libraryfolders"
{{
\t"0"\n\t{{\n\t\t"path"\t\t"{self.steam}"\n\t\t"apps"\n\t\t{{\n\t\t}}\n\t}}
\t"1"\n\t{{\n\t\t"path"\t\t"{self.lib2}"\n\t}}
\t"2"\n\t{{\n\t\t"path"\t\t"{self.tmp}/unmounted"\n\t}}
}}
''')
        dlss = b"MZ fake nvidia dlss"
        # 1001: Unreal layout, plain DLSS -> dxgi in <Game>/Binaries/Win64
        g = self.game(self.steam, "1001", "UE Game", "UEGame", prefix=True)
        self.put(g / "UEGame.exe")
        self.put(g / "UEGame/Binaries/Win64/UEGame-Win64-Shipping.exe")
        self.put(g / "Engine/Binaries/Win64/CrashReportClient.exe")
        self.put(g / "Engine/Plugins/Runtime/Nvidia/DLSS/Binaries/ThirdParty/Win64/nvngx_dlss.dll", dlss)
        # 1002: Streamline next to exe -> dbghelp; foreign _nvngx.dll already in the prefix
        g = self.game(self.steam, "1002", "SL Game", "SLGame", prefix=True)
        self.put(g / "SLGame.exe")
        self.put(g / "sl.interposer.dll")
        self.put(g / "nvngx_dlss.dll", dlss)
        self.put(self.pfx(self.steam, "1002") / "drive_c/windows/system32/_nvngx.dll", b"MZ someone else's core")
        # 1003: foreign dxgi.dll (e.g. OptiScaler) -> conflict
        g = self.game(self.steam, "1003", "Conflict Game", "ConflictGame", prefix=True)
        self.put(g / "Conflict.exe")
        self.put(g / "nvngx_dlss.dll", dlss)
        self.put(g / "dxgi.dll", b"MZ OptiScaler")
        # 1004: on the second library, never launched (no prefix)
        g = self.game(self.lib2, "1004", "Fresh Game", "FreshGame", prefix=False)
        self.put(g / "bin/x64/Fresh.exe")
        self.put(g / "bin/x64/nvngx_dlss.dll", dlss)
        self.put(g / "bin/x64/FreshLauncher.exe")
        # 1005: no DLSS
        g = self.game(self.steam, "1005", "No DLSS", "NoDLSS", prefix=True)
        self.put(g / "NoDLSS.exe")
        # 1006: DOOM-like, Streamline in a subfolder
        g = self.game(self.steam, "1006", "DOOM Like", "DoomLike", prefix=True)
        self.put(g / "DoomLike.exe")
        self.put(g / "idTechLauncher.exe")
        self.put(g / "streamline/production/sl.interposer.dll")
        self.put(g / "streamline/production/nvngx_dlss.dll", dlss)
        # 228980: Steamworks redistributables -> ignored by name
        self.game(self.steam, "228980", "Steamworks Common Redistributables", "Steamworks Shared", prefix=False)

        self.layer = self.home / ".local/share/vulkan/implicit_layer.d"
        self.before_games = {k: tree_digest(self.steam / "steamapps/common" / k)
                             for k in ("UEGame", "SLGame", "ConflictGame", "DoomLike")}
        self.before_pfx = tree_digest(self.steam / "steamapps/compatdata")
        self.core = self.make_core()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_core(self, bin_dir=None):
        return AissCore(home=self.home, bin_dir=bin_dir or self.bin, state_file=self.tmp / "settings/state.json",
                        steam_roots=[self.steam], logger=Quiet(), running_check=lambda a: a in self.running)

    # helpers
    def put(self, path: Path, data: bytes = b"MZ"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def pfx(self, lib: Path, appid: str) -> Path:
        return lib / "steamapps/compatdata" / appid / "pfx"

    def make_prefix(self, lib: Path, appid: str):
        p = self.pfx(lib, appid)
        (p / "drive_c/windows/system32").mkdir(parents=True, exist_ok=True)
        (p / "system.reg").write_text(REG_SAMPLE)

    def game(self, lib: Path, appid: str, name: str, installdir: str, prefix: bool) -> Path:
        (lib / f"steamapps/appmanifest_{appid}.acf").write_text(
            f'"AppState"\n{{\n\t"appid"\t\t"{appid}"\n\t"name"\t\t"{name}"\n\t"installdir"\t\t"{installdir}"\n'
            f'\t"LastUpdated"\t\t"1"\n\t"buildid"\t\t"1"\n}}\n')
        if prefix:
            self.make_prefix(lib, appid)
        d = lib / "steamapps/common" / installdir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def view(self, appid):
        return next(g for g in self.core.scan(force=True) if g["appid"] == appid)

    # tests
    def test_detection(self):
        games = {g["appid"]: g for g in self.core.scan(force=True)}
        self.assertEqual(sorted(games), ["1001", "1002", "1003", "1004", "1006"])
        c = self.steam / "steamapps/common"
        self.assertEqual(games["1001"]["shim"], "dxgi")
        self.assertEqual(games["1001"]["targets"], [str(c / "UEGame/UEGame/Binaries/Win64")])
        self.assertEqual(games["1002"]["shim"], "dbghelp")
        self.assertEqual(games["1002"]["targets"], [str(c / "SLGame")])
        self.assertEqual(games["1003"]["status"], "conflict")
        self.assertEqual(games["1004"]["status"], "prefix_missing")
        self.assertEqual(games["1004"]["targets"], [str(self.lib2 / "steamapps/common/FreshGame/bin/x64")])
        self.assertEqual(games["1006"]["targets"], [str(c / "DoomLike"), str(c / "DoomLike/streamline/production")])
        for a in ("1001", "1002", "1006"):
            self.assertEqual(games[a]["status"], "not_installed")

    def test_install_uninstall_roundtrip(self):
        c = self.steam / "steamapps/common"
        for a in ("1001", "1002", "1006"):
            r = self.core.install_game(a)
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["status"], "installed", r)
            self.assertEqual(self.view(a)["status"], "installed")
        self.assertEqual((c / "UEGame/UEGame/Binaries/Win64/dxgi.dll").read_bytes(), (self.bin / "dxgi.dll").read_bytes())
        self.assertEqual((c / "SLGame/dbghelp.dll").read_bytes(), (self.bin / "dbghelp.dll").read_bytes())
        self.assertTrue((c / "DoomLike/dbghelp.dll").exists())
        self.assertTrue((c / "DoomLike/streamline/production/dbghelp.dll").exists())
        for a in ("1001", "1002", "1006"):
            p = self.pfx(self.steam, a)
            self.assertEqual((p / "drive_c/windows/system32/_nvngx.dll").read_bytes(),
                             (self.bin / "nvngx.dll").read_bytes())
            # XeSS (Intel's binary) and FSR 4 (AMD's) go next to the core, unmodified
            self.assertEqual((p / "drive_c/windows/system32/libxess.dll").read_bytes(),
                             (self.bin / "libxess.dll").read_bytes())
            # AMD's FSR 4 DLLs go in a private system32/aiss subfolder (Proton's own amdxcffx64 in system32 can't shadow them)
            for amd in ("amdvkffx64.dll", "amdxcffx64.dll"):
                dst = p / "drive_c/windows/system32/aiss" / amd
                if (self.bin / amd).is_file():
                    self.assertEqual(sha256_of(dst), sha256_of(self.bin / amd))
                else:
                    self.assertFalse(dst.exists())
            self.assertTrue(reg_has_value((p / "system.reg").read_text(), REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01"))
            self.assertEqual((p / "system.reg.aiss-bak").read_text(), REG_SAMPLE)
        self.assertTrue((self.pfx(self.steam, "1002") / "drive_c/windows/system32/_nvngx.dll.aiss-orig").exists())
        self.assertTrue((self.layer / "libVkLayer_aiss_ngx.so").exists())
        self.assertTrue((self.layer / "VkLayer_aiss_ngx.json").exists())
        # install again: idempotent
        self.assertEqual(self.core.install_game("1001")["status"], "installed")
        # state survives a restart
        self.core = self.make_core()
        self.assertTrue(self.core.state["games"]["1002"]["enabled"])
        for a in ("1001", "1002", "1006"):
            r = self.core.uninstall_game(a)
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["message"], "")
        for k, digest in self.before_games.items():
            self.assertEqual(tree_digest(c / k), digest, f"{k} not restored")
        after = tree_digest(self.steam / "steamapps/compatdata")
        after = {k: v for k, v in after.items() if not k.endswith(".aiss-bak")}
        self.assertEqual(after, self.before_pfx, "prefixes not restored")
        self.core.remove_layer()
        self.assertFalse((self.layer / "libVkLayer_aiss_ngx.so").exists())

    def test_xess_dll_foreign_kept_and_old_install_repaired(self):
        sys32 = self.pfx(self.steam, "1001") / "drive_c/windows/system32"
        self.put(sys32 / "libxess.dll", b"MZ a game's own libxess")
        self.assertEqual(self.core.install_game("1001")["status"], "installed")
        self.assertEqual((sys32 / "libxess.dll").read_bytes(), (self.bin / "libxess.dll").read_bytes())
        self.assertEqual((sys32 / "libxess.dll.aiss-orig").read_bytes(), b"MZ a game's own libxess")
        # an install from before XeSS was bundled (no libxess.dll recorded) is repaired by verify_all
        gs = self.core.state["games"]["1001"]
        (sys32 / "libxess.dll").unlink()
        gs.pop("xess_path")
        gs.pop("xess_hash")
        self.assertEqual(self.core.verify_all()[0]["status"], "installed")
        self.assertTrue((sys32 / "libxess.dll").exists())
        self.assertEqual(self.view("1001")["status"], "installed")
        self.core.uninstall_game("1001")
        self.assertEqual((sys32 / "libxess.dll").read_bytes(), b"MZ a game's own libxess")
        self.assertFalse((sys32 / "_nvngx.dll").exists())

    def test_fsr4_dlls_private_subfolder_and_optional(self):
        sys32 = self.pfx(self.steam, "1001") / "drive_c/windows/system32"
        aiss  = sys32 / "aiss"
        # a bundle with both AMD DLLs (the build skips whichever it cannot find locally)
        withfsr = self.tmp / "bin_fsr4"
        withfsr.mkdir()
        for f in self.bin.iterdir():
            if f.name not in ("amdvkffx64.dll", "amdxcffx64.dll"):
                os.symlink(f.resolve(), withfsr / f.name)
        for amd in ("amdvkffx64.dll", "amdxcffx64.dll"):
            if (self.bin / amd).is_file():
                os.symlink((self.bin / amd).resolve(), withfsr / amd)
            else:
                (withfsr / amd).write_bytes(b"MZ stand-in for AMD " + amd.encode())
        # Proton's own amdxcffx64 sits in system32; ours must land in the private aiss subfolder, not shadow it
        self.put(sys32 / "amdxcffx64.dll", b"MZ proton builtin amdxcffx64 4.1.1")
        self.core = self.make_core(bin_dir=withfsr)
        self.assertEqual(self.core.install_game("1001")["status"], "installed")
        self.assertEqual(sha256_of(aiss / "amdvkffx64.dll"), sha256_of(withfsr / "amdvkffx64.dll"))
        self.assertEqual(sha256_of(aiss / "amdxcffx64.dll"), sha256_of(withfsr / "amdxcffx64.dll"))
        self.assertEqual((sys32 / "amdxcffx64.dll").read_bytes(), b"MZ proton builtin amdxcffx64 4.1.1")  # untouched
        self.assertTrue(self.core.bundle_info()["fsr4"])
        self.core.uninstall_game("1001")
        self.assertFalse(aiss.exists())  # the private subfolder is removed on uninstall
        self.assertEqual((sys32 / "amdxcffx64.dll").read_bytes(), b"MZ proton builtin amdxcffx64 4.1.1")
        (sys32 / "amdxcffx64.dll").unlink()
        # a bundle built without the AMD DLLs: not reported missing, nothing installed for them, games still "installed"
        nofsr = self.tmp / "bin_nofsr4"
        nofsr.mkdir()
        for f in self.bin.iterdir():
            if f.name not in ("amdvkffx64.dll", "amdxcffx64.dll"):
                os.symlink(f.resolve(), nofsr / f.name)
        self.core = self.make_core(bin_dir=nofsr)
        self.assertEqual(self.core.bundle_missing(), [])
        self.assertFalse(self.core.bundle_info()["fsr4"])
        self.assertEqual(self.core.install_game("1001")["status"], "installed")
        self.assertFalse((sys32 / "aiss").exists())
        self.assertEqual(self.view("1001")["status"], "installed")
        self.core.uninstall_game("1001")
        self.assertFalse((sys32 / "_nvngx.dll").exists())

    def test_conflict_untouched(self):
        r = self.core.install_game("1003")
        self.assertFalse(r["ok"])
        self.assertEqual(r["status"], "conflict")
        self.assertEqual(tree_digest(self.steam / "steamapps/common/ConflictGame"), self.before_games["ConflictGame"])
        self.assertFalse((self.pfx(self.steam, "1003") / "drive_c/windows/system32/_nvngx.dll").exists())
        self.assertFalse(self.core.state["games"]["1003"]["enabled"])
        lo = self.core.launch_options("1003", "-foo", True)
        self.assertFalse(lo["ok"])

    def test_prefix_created_later(self):
        r = self.core.install_game("1004")
        self.assertTrue(r["ok"])
        self.assertEqual(r["status"], "prefix_missing")
        self.assertTrue((self.lib2 / "steamapps/common/FreshGame/bin/x64/dxgi.dll").exists())
        self.assertEqual(self.core.verify_all(), [])  # nothing to do yet
        # first launch creates the prefix (Steam puts it in the game's library)
        self.make_prefix(self.lib2, "1004")
        res = self.core.verify_all()
        self.assertEqual([r["status"] for r in res], ["installed"])
        self.assertTrue((self.pfx(self.lib2, "1004") / "drive_c/windows/system32/_nvngx.dll").exists())
        self.assertEqual(self.view("1004")["status"], "installed")

    def test_repair_after_game_update(self):
        self.core.install_game("1001")
        shim = self.steam / "steamapps/common/UEGame/UEGame/Binaries/Win64/dxgi.dll"
        shim.unlink()  # Steam update/verify deleted it
        self.assertEqual(self.view("1001")["status"], "needs_repair")
        res = self.core.verify_all()
        self.assertEqual(res[0]["status"], "installed")
        self.assertEqual(shim.read_bytes(), (self.bin / "dxgi.dll").read_bytes())
        # an older build of ours (has the marker, unknown hash) is upgraded, not reported as a conflict
        shim.write_bytes(b"MZ old build " + core.SHIM_MARKER)
        self.assertEqual(self.view("1001")["status"], "needs_repair")
        self.assertEqual(self.core.verify_all()[0]["status"], "installed")
        self.assertEqual(shim.read_bytes(), (self.bin / "dxgi.dll").read_bytes())
        # someone else's dxgi.dll replaced ours -> conflict, left alone
        shim.write_bytes(b"MZ ReShade")
        self.assertEqual(self.view("1001")["status"], "conflict")
        self.assertEqual(self.core.verify_all()[0]["status"], "conflict")
        self.assertEqual(shim.read_bytes(), b"MZ ReShade")
        r = self.core.uninstall_game("1001")
        self.assertIn("no longer ours", r["message"])
        self.assertEqual(shim.read_bytes(), b"MZ ReShade")
        self.assertEqual(self.core.verify_all(), [])  # nothing done for disabled game

    def test_update_adds_streamline(self):
        self.core.install_game("1001")
        g = self.steam / "steamapps/common/UEGame"
        self.put(g / "UEGame/Binaries/Win64/sl.interposer.dll")
        (g.parent.parent / "appmanifest_1001.acf").write_text(
            (g.parent.parent / "appmanifest_1001.acf").read_text().replace('"LastUpdated"\t\t"1"', '"LastUpdated"\t\t"2"'))
        res = self.core.verify_all()
        self.assertEqual(res[0]["shim"], "dbghelp")
        self.assertFalse((g / "UEGame/Binaries/Win64/dxgi.dll").exists())
        self.assertTrue((g / "UEGame/Binaries/Win64/dbghelp.dll").exists())
        lo = self.core.launch_options("1001", "", True)
        self.assertIn('"dbghelp=n,b"', lo["options"])

    def test_running_game_defers_registry(self):
        self.running.add("1002")
        r = self.core.install_game("1002")
        self.assertTrue(r["ok"])
        self.assertIn("deferred", r["message"])
        reg = self.pfx(self.steam, "1002") / "system.reg"
        self.assertEqual(reg.read_text(), REG_SAMPLE)
        self.assertEqual(self.core.verify_all()[0]["message"], "registry update deferred until the game exits")
        self.running.clear()
        self.core.verify_all()
        self.assertTrue(reg_has_value(reg.read_text(), REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01"))
        self.assertFalse(self.core.state["games"]["1002"]["reg_pending"])
        # disable while running: files go, registry waits
        self.running.add("1002")
        r = self.core.uninstall_game("1002")
        self.assertIn("deferred", r["message"])
        self.assertTrue(reg_has_value(reg.read_text(), REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01"))
        self.running.clear()
        self.core.verify_all()
        self.assertEqual(reg.read_text(), REG_SAMPLE)

    def test_layer_not_ours_untouched(self):
        self.layer.mkdir(parents=True)
        (self.layer / "libVkLayer_aiss_ngx.so").write_bytes(b"dev build")
        r = self.core.install_game("1002")
        self.assertIn("left untouched", r["message"])
        self.assertEqual((self.layer / "libVkLayer_aiss_ngx.so").read_bytes(), b"dev build")
        self.assertTrue((self.layer / "VkLayer_aiss_ngx.json").exists())

    def test_streamline_game_importing_dxgi_gets_dxgi(self):
        # Unreal + Streamline (e.g. STAR WARS Galactic Racer): Unreal preloads its own dbghelp.dll, so the shim must
        # be the dxgi.dll the exe imports at start-up.
        g = self.game(self.steam, "1007", "UE SL Game", "UESL", prefix=True)
        self.put(g / "UESL/Binaries/Win64/UESL-Win64-Shipping.exe", minimal_pe(["kernel32.dll", "dxgi.dll"]))
        self.put(g / "Engine/Binaries/ThirdParty/DbgHelp/dbghelp.dll", b"MZ microsoft dbghelp")
        self.put(g / "Engine/Plugins/Marketplace/DLSS/Binaries/ThirdParty/Win64/nvngx_dlss.dll", b"MZ dlss")
        self.put(g / "Engine/Plugins/Marketplace/StreamlineCore/Binaries/ThirdParty/Win64/sl.interposer.dll")
        games = {x["appid"]: x for x in self.core.scan(force=True)}
        self.assertEqual(games["1007"]["shim"], "dxgi")
        self.assertEqual(games["1007"]["targets"], [str(g / "UESL/Binaries/Win64")])
        self.core.install_game("1007")
        self.assertTrue((g / "UESL/Binaries/Win64/dxgi.dll").exists())
        self.assertFalse((g / "UESL/Binaries/Win64/dbghelp.dll").exists())
        self.assertIn('"dxgi=n,b"', self.core.launch_options("1007", "", True)["options"])

    def test_vulkan_game_importing_dbghelp_gets_dbghelp(self):
        # Enshrouded: Vulkan, no Streamline, exe imports dbghelp but not dxgi.
        g = self.game(self.steam, "1008", "VK Game", "VKGame", prefix=True)
        self.put(g / "vkgame.exe", minimal_pe(["kernel32.dll", "dbghelp.dll", "steam_api64.dll"]))
        self.put(g / "nvngx_dlss.dll", b"MZ dlss")
        games = {x["appid"]: x for x in self.core.scan(force=True)}
        self.assertEqual(games["1008"]["shim"], "dbghelp")
        self.assertEqual(games["1008"]["targets"], [str(g)])

    def test_launch_options_flow_and_settings(self):
        self.core.install_game("1002")
        control = self.home / ".local/share/aiss-dlss-nss/control"
        lo = self.core.launch_options("1002", "-novid", True)
        self.assertEqual(lo["options"], f'PROTON_FORCE_NVAPI=1 AISS_LAYER=1 AISS_CONTROL={control} '
                                        'WINEDLLOVERRIDES="dbghelp=n,b" %command% -novid')
        # The control file: a legacy first line "<upscaler> <fast_math> <capture>", then key=value lines. Old proxies
        # read the first line; new ones also read the keys. Everything switches live, not through the launch options.
        def ctl():
            text = control.read_text()
            d = {"_first": text.splitlines()[0]}
            for ln in text.splitlines():
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    d[k] = v
            return d
        self.core.set_settings({"upscaler": 2, "log": 0})
        self.assertEqual(ctl()["_first"], "2 0 0")
        self.assertEqual(ctl()["upscaler"], "2")
        self.assertEqual(ctl()["log"], "0")
        self.core.set_settings({"upscaler": 0})
        self.assertEqual(ctl()["_first"], "0 0 0")
        # Fast maths also goes through the control file and keeps the launch options unchanged.
        self.core.set_settings({"fast_math": 1})
        self.assertEqual(ctl()["_first"], "0 1 0")
        self.assertEqual(ctl()["fast_math"], "1")
        # Each capture request bumps the third value; settings changes keep it.
        r = self.core.request_capture()
        self.assertEqual(r["seq"], 1)
        self.assertEqual(ctl()["_first"], "0 1 1")
        self.core.request_capture()
        self.core.set_settings({"upscaler": 2})
        self.assertEqual(ctl()["_first"], "2 1 2")
        # XeSS (3), and FSR 4 (upscaler 4) with a model: Auto=8, Lite=4, INT8=5, INT8 RDNA 2=6.
        self.core.set_settings({"upscaler": 3})
        self.assertEqual(ctl()["_first"], "3 1 2")
        for model, code in ((0, 8), (1, 4), (2, 5), (3, 6)):
            self.core.set_settings({"upscaler": 4, "fsr4_model": model})
            self.assertEqual(ctl()["_first"], f"{code} 1 2")
            self.assertEqual(ctl()["upscaler"], str(code))
        # FSR 4 path, watermark, gpu timing and xess preset all go through the control file.
        self.core.set_settings({"fsr4_path": 2, "watermark": 1, "gpu_timing": 1, "xess_quality": 101})
        self.assertEqual((ctl()["fsr4_path"], ctl()["watermark"], ctl()["gpu_timing"], ctl()["xess_quality"]),
                         ("2", "1", "1", "101"))
        for bad in ({"upscaler": 5}, {"fsr4_model": 4}, {"fsr4_path": 3}, {"xess_quality": 99}, {"fast_math": 2},
                    {"quality": 7}):
            with self.assertRaises(ValueError):
                self.core.set_settings(bad)
        self.core.set_settings({"upscaler": 2})
        self.assertIsNone(self.core.latest_capture())
        cap = self.core.captures_dir / "Game_20261027_120000"
        cap.mkdir(parents=True)
        (cap / "000_output.raw").write_bytes(b"x")
        self.assertEqual(self.core.latest_capture()["frames"], 1)
        self.core.set_settings({"upscaler": 0, "watermark": 0, "gpu_timing": 0, "xess_quality": 0})
        # AISS_RATIO / AISS_NSS_QUALITY left over from older plugin versions are removed on the next sync.
        old = lo["options"].replace("AISS_LAYER=1", "AISS_LAYER=1 AISS_RATIO=2.0 AISS_NSS_QUALITY=0")
        lo2 = self.core.launch_options("1002", old, True)
        self.assertEqual(lo2["options"], f'PROTON_FORCE_NVAPI=1 AISS_LAYER=1 AISS_CONTROL={control} '
                                         'AISS_LOG=0 WINEDLLOVERRIDES="dbghelp=n,b" %command% -novid')
        self.core = self.make_core()  # ownership record persisted
        off = self.core.launch_options("1002", lo2["options"], False)
        self.assertEqual(off["options"], "-novid")

    def test_main_py_loads_with_stub_decky(self):
        stub = types.ModuleType("decky")
        stub.DECKY_PLUGIN_DIR = str(PLUGIN)
        stub.DECKY_USER_HOME = str(self.home)
        stub.DECKY_PLUGIN_SETTINGS_DIR = str(self.tmp / "settings")
        stub.logger = Quiet()
        sys.modules["decky"] = stub
        import importlib.util
        spec = importlib.util.spec_from_file_location("aiss_main", PLUGIN / "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import asyncio
        p = mod.Plugin()
        mod.Plugin.core = None
        st = asyncio.run(p.get_state(True))
        self.assertIn("games", st)
        self.assertIn("settings", st)
        r = asyncio.run(p.set_settings({"upscaler": 9}))
        self.assertFalse(r["ok"])
        mod.Plugin.core = None


# ------------------------------------------------------------------------------------------------------------------
class RealLibraryReadOnly(unittest.TestCase):
    def test_real_library_scan_readonly(self):
        home = Path("/home/deck")
        if not (home / ".local/share/Steam/steamapps/libraryfolders.vdf").exists():
            self.skipTest("no Steam library")
        tmp = Path(tempfile.mkdtemp(prefix="aiss-real-"))
        try:
            c = AissCore(home=home, bin_dir=bin_dir(tmp), state_file=tmp / "state.json",
                         layer_dir=tmp / "layer", logger=Quiet())
            games = c.scan(force=True)
            self.assertFalse((tmp / "state.json").exists(), "scan must not write")
            print("\n  real library: %d DLSS game(s)" % len(games))
            for g in games:
                print(f"   {g['appid']:>8} {g['name'][:28]:28} shim={g['shim']:7} status={g['status']:14} "
                      f"running={c.is_running(g['appid'])}")
                for t in g["targets"]:
                    print(f"            -> {t}/{g['shim']}.dll")
                if g["conflicts"]:
                    print(f"            conflicts: {g['conflicts']}")
            self.assertTrue(all(g["shim"] in ("dbghelp", "dxgi") for g in games))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1, argv=[a for a in sys.argv if a != "-v"])
