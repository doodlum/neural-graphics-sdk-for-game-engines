"""Standalone tests for the plugin backend (no Decky needed).

    python3 steamdeck/decky/tests/test_backend.py [-v]

Everything that writes happens in a temporary fake Steam tree. The real Steam library is only *scanned*
(test_real_library_scan_readonly), with state and layer paths redirected to the temp dir.
Set DU_TEST_BIN to use another bin/ dir (default: ../bin, falling back to the repo build dirs).
"""

import hashlib
import json
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

import deckyupscaler_core as core  # noqa: E402
from deckyupscaler_core import (UpscalerCore, build_launch_options, strip_launch_options, reg_set_value,  # noqa: E402
                       reg_delete_value, reg_has_value, REG_KEY_PATH, SIG_OVERRIDE_VALUE)

S0 = {"upscaler": 0, "nss_quality": 0, "log": 1}


def bin_dir(tmp: Path) -> Path:
    env = os.environ.get("DU_TEST_BIN")
    if env:
        return Path(env)
    if all((PLUGIN / "bin" / n).is_file() for k, n in core.BIN_FILES.items() if k not in core.OPTIONAL_BIN):
        return PLUGIN / "bin"
    d = tmp / "bin"
    d.mkdir(exist_ok=True)
    for src in (REPO / "build_win_nvngx/nvngx.dll", REPO / "build_win_loader/dbghelp/dbghelp.dll",
                REPO / "build_win_loader/dxgi/dxgi.dll", REPO / "build_layer/libVkLayer_deckyupscaler.so",
                REPO / "build_layer/VkLayer_deckyupscaler.json"):
        shutil.copy(src, d / src.name)
    xess = REPO.parent / "third_party/xess/sdk"
    if (xess / "bin/libxess.dll").is_file():
        os.symlink(xess / "bin/libxess.dll", d / "libxess.dll")  # 78 MB: linked, not copied
        shutil.copy(xess / "LICENSE.txt", d / "libxess-LICENSE.txt")
    else:  # stand-ins: the tests only move the files around
        (d / "libxess.dll").write_bytes(b"MZ stand-in for Intel libxess.dll")
        (d / "libxess-LICENSE.txt").write_text("stand-in licence\n")
    for amd in ("amdxcffx64.dll",):
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

# AMD FSR 4 bundle files (optional, never in the repo).
AMD_FSR4_FILES = ("amdvkffx64.dll", "amdxcffx64.dll", "amd_fidelityfx_upscaler_vk.dll", "amd_fidelityfx-LICENSE.md", "fsr4_shaders.bin",
                  "fsr4_shaders_dot2.bin")

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
        self.roundtrip("", expect='PROTON_FORCE_NVAPI=1 DU_LAYER=1 WINEDLLOVERRIDES="dxgi=n,b" %command%')

    def test_args_without_command(self):
        self.roundtrip("-dx12 -skipintro",
                       expect='PROTON_FORCE_NVAPI=1 DU_LAYER=1 WINEDLLOVERRIDES="dxgi=n,b" %command% -dx12 -skipintro')

    def test_wrapper(self):
        self.roundtrip("gamemoderun %command% -foo",
                       expect='PROTON_FORCE_NVAPI=1 DU_LAYER=1 WINEDLLOVERRIDES="dxgi=n,b" gamemoderun %command% -foo')

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
        self.assertEqual(new2, 'PROTON_FORCE_NVAPI=1 DU_LAYER=1 DU_LOG=2 '
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
        self.tmp = Path(tempfile.mkdtemp(prefix="deckyupscaler-test-"))
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
        self.put(g / "dxgi.dll", b"MZ ReShade")  # another mod that takes dxgi.dll (OptiScaler is removed instead)
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
        return UpscalerCore(home=self.home, bin_dir=bin_dir or self.bin, state_file=self.tmp / "settings/state.json",
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
            # AMD's FSR 4 DLL goes in a private system32/deckyupscaler subfolder (Proton's own amdxcffx64 in system32
            # can't shadow it); the Vulkan DLL is no longer installed
            dst = p / "drive_c/windows/system32/deckyupscaler/amdxcffx64.dll"
            if (self.bin / "amdxcffx64.dll").is_file():
                self.assertEqual(sha256_of(dst), sha256_of(self.bin / "amdxcffx64.dll"))
            else:
                self.assertFalse(dst.exists())
            self.assertFalse((p / "drive_c/windows/system32/deckyupscaler/amdvkffx64.dll").exists())
            self.assertTrue(reg_has_value((p / "system.reg").read_text(), REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01"))
            self.assertEqual((p / "system.reg.deckyupscaler-bak").read_text(), REG_SAMPLE)
        self.assertTrue((self.pfx(self.steam, "1002") / "drive_c/windows/system32/_nvngx.dll.deckyupscaler-orig")
                        .exists())
        self.assertTrue((self.layer / "libVkLayer_deckyupscaler.so").exists())
        self.assertTrue((self.layer / "VkLayer_deckyupscaler.json").exists())
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
        after = {k: v for k, v in after.items() if not k.endswith(".deckyupscaler-bak")}
        self.assertEqual(after, self.before_pfx, "prefixes not restored")
        self.core.remove_layer()
        self.assertFalse((self.layer / "libVkLayer_deckyupscaler.so").exists())

    def test_xess_dll_foreign_kept_and_old_install_repaired(self):
        sys32 = self.pfx(self.steam, "1001") / "drive_c/windows/system32"
        self.put(sys32 / "libxess.dll", b"MZ a game's own libxess")
        self.assertEqual(self.core.install_game("1001")["status"], "installed")
        self.assertEqual((sys32 / "libxess.dll").read_bytes(), (self.bin / "libxess.dll").read_bytes())
        self.assertEqual((sys32 / "libxess.dll.deckyupscaler-orig").read_bytes(), b"MZ a game's own libxess")
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

    def test_fsr4_dll_private_subfolder_optional_and_legacy_removed(self):
        sys32 = self.pfx(self.steam, "1001") / "drive_c/windows/system32"
        sub   = sys32 / "deckyupscaler"
        # a bundle with the AMD DLL (the build skips it when it cannot find it locally)
        withfsr = self.tmp / "bin_fsr4"
        withfsr.mkdir()
        for f in self.bin.iterdir():
            if f.name not in AMD_FSR4_FILES:
                os.symlink(f.resolve(), withfsr / f.name)
        if (self.bin / "amdxcffx64.dll").is_file():
            os.symlink((self.bin / "amdxcffx64.dll").resolve(), withfsr / "amdxcffx64.dll")
        else:
            (withfsr / "amdxcffx64.dll").write_bytes(b"MZ stand-in for AMD amdxcffx64.dll")
        (withfsr / "amd_fidelityfx_upscaler_vk.dll").write_bytes(b"MZ stand-in for AMD's FSR SDK Vulkan upscaler")
        (withfsr / "amd_fidelityfx-LICENSE.md").write_text("AMD FSR SDK licence stand-in")
        (withfsr / "fsr4_shaders.bin").write_bytes(b"DUSPACK1\0\0\0\0")
        (withfsr / "fsr4_shaders_dot2.bin").write_bytes(b"DUSPACK1\0\0\0\0")
        # Proton's own amdxcffx64 sits in system32; ours must land in the private deckyupscaler subfolder, not shadow it
        self.put(sys32 / "amdxcffx64.dll", b"MZ proton builtin amdxcffx64 4.1.1")
        self.core = self.make_core(bin_dir=withfsr)
        self.assertEqual(self.core.install_game("1001")["status"], "installed")
        self.assertEqual(sha256_of(sub / "amdxcffx64.dll"), sha256_of(withfsr / "amdxcffx64.dll"))
        for name in ("amd_fidelityfx_upscaler_vk.dll", "fsr4_shaders.bin", "fsr4_shaders_dot2.bin"):  # the Vulkan FSR 4 path's files
            self.assertEqual(sha256_of(sub / name), sha256_of(withfsr / name))
        self.assertFalse((sub / "amd_fidelityfx-LICENSE.md").exists())  # the licence stays with the plugin
        self.assertEqual((sys32 / "amdxcffx64.dll").read_bytes(), b"MZ proton builtin amdxcffx64 4.1.1")  # untouched
        self.assertTrue(self.core.bundle_info()["fsr4"])
        # an earlier build (still named aiss-dlss-nss) also installed aiss/amdvkffx64.dll: removed on the next install
        # (upgrade) with its folder ...
        old = sys32 / "aiss/amdvkffx64.dll"
        self.put(old, b"MZ old amdvkffx64")
        gs = self.core.state["games"]["1001"]
        gs["fsr4_path"], gs["fsr4_hash"] = str(old), sha256_of(old)
        self.assertEqual(self.core.install_game("1001")["status"], "installed")
        self.assertFalse(old.exists())
        self.assertFalse(old.parent.exists())
        self.assertNotIn("fsr4_path", self.core.state["games"]["1001"])
        # ... and on uninstall
        self.put(old, b"MZ old amdvkffx64")
        gs = self.core.state["games"]["1001"]
        gs["fsr4_path"], gs["fsr4_hash"] = str(old), sha256_of(old)
        self.core.uninstall_game("1001")
        self.assertFalse(old.exists())
        self.assertFalse(old.parent.exists())
        self.assertFalse(sub.exists())  # the private subfolder is removed on uninstall
        self.assertEqual((sys32 / "amdxcffx64.dll").read_bytes(), b"MZ proton builtin amdxcffx64 4.1.1")
        (sys32 / "amdxcffx64.dll").unlink()
        # a later bundle without the shader packs (FSR 4 on AMD's own shaders): reinstalling removes the packs it had put
        # in the prefix, so the proxy stops loading them, and keeps the rest
        self.assertEqual(self.core.install_game("1001")["status"], "installed")
        nopacks = self.tmp / "bin_nopacks"
        nopacks.mkdir()
        for f in withfsr.iterdir():
            if f.name not in ("fsr4_shaders.bin", "fsr4_shaders_dot2.bin"):
                os.symlink(f.resolve(), nopacks / f.name)
        self.core = self.make_core(bin_dir=nopacks)
        self.assertEqual(self.core.install_game("1001")["status"], "installed")
        self.assertFalse((sub / "fsr4_shaders.bin").exists())
        self.assertFalse((sub / "fsr4_shaders_dot2.bin").exists())
        self.assertTrue((sub / "amd_fidelityfx_upscaler_vk.dll").exists())
        self.assertNotIn("fsr4_shaders_path", self.core.state["games"]["1001"])
        self.core.uninstall_game("1001")
        self.assertFalse(sub.exists())
        # a bundle built without the AMD DLL: not reported missing, nothing installed for it, games still "installed"
        nofsr = self.tmp / "bin_nofsr4"
        nofsr.mkdir()
        for f in self.bin.iterdir():
            if f.name not in AMD_FSR4_FILES:
                os.symlink(f.resolve(), nofsr / f.name)
        self.core = self.make_core(bin_dir=nofsr)
        self.assertEqual(self.core.bundle_missing(), [])
        self.assertFalse(self.core.bundle_info()["fsr4"])
        self.assertEqual(self.core.install_game("1001")["status"], "installed")
        self.assertFalse(sub.exists())
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

    def test_optiscaler_removed_and_originals_restored(self):
        # DOOM: The Dark Ages as Decky Framegen and HelixSR left it: OptiScaler as winmm.dll, the game's dbghelp.dll
        # backed up as .b (our shim takes that name), Framegen's support DLLs (one with a backed-up original), and
        # HelixSR's FSR drop-in with the DLL it replaced kept as .original.dll.
        g = self.steam / "steamapps/common/DoomLike"
        (g / "winmm.dll").write_bytes(b"MZ OptiScaler injector")
        (g / "OptiScaler.ini").write_text("[Spoofing]\n")
        (g / "FRAMEGEN_PATCH").write_text('{"dll_name": "winmm.dll"}')
        (g / "D3D12_Optiscaler").mkdir()
        (g / "D3D12_Optiscaler/D3D12Core.dll").write_bytes(b"MZ")
        (g / "plugins").mkdir()
        (g / "plugins/OptiPatcher.asi").write_bytes(b"MZ")
        (g / "plugins/game_plugin.dll").write_bytes(b"MZ game")
        (g / "fakenvapi.dll").write_bytes(b"MZ")
        (g / "libxess.dll").write_bytes(b"MZ framegen copy")
        (g / "amd_fidelityfx_vk.dll").write_bytes(b"MZ framegen copy")
        (g / "amd_fidelityfx_vk.dll.b").write_bytes(b"MZ game original")
        (g / "dbghelp.dll.b").write_bytes(b"MZ game dbghelp")
        (g / "amd_fidelityfx_upscaler_dx12.dll").write_bytes(b"MZ HLX FSR 4")
        (g / "amd_fidelityfx_upscaler_dx12.original.dll").write_bytes(b"MZ framegen copy")
        (g / "helixsr-install.json").write_text('{"helixsr": "2.0.0", "files": ["amd_fidelityfx_upscaler_dx12.dll"]}')
        other = self.steam / "steamapps/common/SLGame/libxess.dll"  # a game's own XeSS, no OptiScaler there
        other.write_bytes(b"MZ game xess")
        self.assertEqual(self.view("1006")["status"], "not_installed")
        r = self.core.install_game("1006")
        self.assertTrue(r["ok"], r)
        for name in ("winmm.dll", "OptiScaler.ini", "FRAMEGEN_PATCH", "D3D12_Optiscaler", "plugins/OptiPatcher.asi",
                     "fakenvapi.dll", "libxess.dll", "amd_fidelityfx_upscaler_dx12.dll",
                     "amd_fidelityfx_upscaler_dx12.original.dll", "helixsr-install.json", "amd_fidelityfx_vk.dll.b"):
            self.assertFalse((g / name).exists(), name)
        self.assertEqual((g / "amd_fidelityfx_vk.dll").read_bytes(), b"MZ game original")
        archived = list((g / ".deckyupscaler-optiscaler-removed").iterdir())  # moved aside, not deleted
        self.assertEqual(len(archived), 1)
        self.assertEqual((archived[0] / "winmm.dll").read_bytes(), b"MZ OptiScaler injector")
        self.assertEqual((archived[0] / "libxess.dll").read_bytes(), b"MZ framegen copy")
        self.assertTrue((archived[0] / "D3D12_Optiscaler/D3D12Core.dll").is_file())
        self.assertEqual((archived[0] / "amd_fidelityfx_vk.dll").read_bytes(), b"MZ framegen copy")
        self.assertEqual((archived[0] / "amd_fidelityfx_upscaler_dx12.dll").read_bytes(), b"MZ HLX FSR 4")
        self.assertEqual((archived[0] / "amd_fidelityfx_upscaler_dx12.dll.1").read_bytes(), b"MZ framegen copy")
        self.assertEqual((g / "plugins/game_plugin.dll").read_bytes(), b"MZ game")
        self.assertEqual((g / "dbghelp.dll").read_bytes(), (self.bin / "dbghelp.dll").read_bytes())  # our shim
        self.assertEqual((g / "dbghelp.dll.b").read_bytes(), b"MZ game dbghelp")
        self.assertEqual(other.read_bytes(), b"MZ game xess")
        self.assertEqual(self.view("1006")["status"], "installed")
        # a later OptiScaler install is removed again by the periodic repair, but not while the game runs
        (g / "dxgi.dll").write_bytes(b"MZ OptiScaler again")
        self.core.is_running = lambda appid: True
        self.core.verify_all()
        self.assertTrue((g / "dxgi.dll").exists())
        self.core.is_running = lambda appid: False
        self.core.verify_all()
        self.assertFalse((g / "dxgi.dll").exists())

    def test_ffx_forwarder_install_update_uninstall(self):
        fwd = REPO / "build_win_loader/ffx/amd_fidelityfx_vk.dll"
        if not fwd.is_file():
            self.skipTest("forwarder not built")
        b = self.tmp / "bin_ffx"
        b.mkdir()
        for f in self.bin.iterdir():
            os.symlink(f.resolve(), b / f.name)
        shutil.copy(fwd, b / "deckyupscaler_ffx_vk.dll")
        self.core = self.make_core(bin_dir=b)
        ffx = self.steam / "steamapps/common/DoomLike/amd-ffx"
        self.put(ffx / "amd_fidelityfx_vk.dll", b"MZ AMD ffx")
        r = self.core.install_game("1006")
        self.assertTrue(r["ok"], r)
        self.assertEqual((ffx / "amd_fidelityfx_vk.dll").read_bytes(), fwd.read_bytes())
        self.assertEqual((ffx / "amd_fidelityfx_vk.deckyupscaler-orig.dll").read_bytes(), b"MZ AMD ffx")
        self.assertEqual(self.view("1006")["status"], "installed")
        # a game update brings a new AMD DLL under the forwarder's name: it becomes the kept original
        (ffx / "amd_fidelityfx_vk.dll").write_bytes(b"MZ AMD ffx v2")
        self.core.verify_all()
        self.assertEqual((ffx / "amd_fidelityfx_vk.dll").read_bytes(), fwd.read_bytes())
        self.assertEqual((ffx / "amd_fidelityfx_vk.deckyupscaler-orig.dll").read_bytes(), b"MZ AMD ffx v2")
        self.core.uninstall_game("1006")
        self.assertEqual((ffx / "amd_fidelityfx_vk.dll").read_bytes(), b"MZ AMD ffx v2")
        self.assertFalse((ffx / "amd_fidelityfx_vk.deckyupscaler-orig.dll").exists())

    def test_id_tech_launcher_bypassed(self):
        g = self.steam / "steamapps/common/DoomLike"
        self.put(g / "idTechLauncher.exe")
        self.assertTrue(self.core.install_game("1006")["ok"])
        lo = self.core.launch_options("1006", "-skipintro", True)
        self.assertTrue(lo["ok"])
        self.assertIn("""bash -c 'exec "${@/idTechLauncher.exe/DoomLike.exe}"' -- %command% -skipintro""", lo["options"])
        again = self.core.launch_options("1006", lo["options"], True)  # idempotent
        self.assertEqual(again["options"], lo["options"])
        off = self.core.launch_options("1006", lo["options"], False)
        self.assertEqual(off["options"], "-skipintro")
        # games without a launcher get no wrapper
        self.assertTrue(self.core.install_game("1002")["ok"])
        self.assertNotIn("bash -c", self.core.launch_options("1002", "", True)["options"])

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
        (self.layer / "libVkLayer_deckyupscaler.so").write_bytes(b"dev build")
        r = self.core.install_game("1002")
        self.assertIn("left untouched", r["message"])
        self.assertEqual((self.layer / "libVkLayer_deckyupscaler.so").read_bytes(), b"dev build")
        self.assertTrue((self.layer / "VkLayer_deckyupscaler.json").exists())

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
        control = self.home / ".local/share/deckyupscaler/control"
        lo = self.core.launch_options("1002", "-novid", True)
        self.assertEqual(lo["options"], f'PROTON_FORCE_NVAPI=1 DU_LAYER=1 DU_CONTROL={control} '
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
        # NSS Quality is the default; the NSS quality is its own setting (Arm's 0 Quality, 1 Balanced, 2 Performance).
        self.core.set_settings({"log": 0})
        self.assertEqual(ctl()["_first"], "0 0 0")
        self.assertEqual(ctl()["log"], "0")
        self.core.set_settings({"nss_quality": 2})
        self.assertEqual(ctl()["_first"], "2 0 0")
        self.assertEqual(ctl()["upscaler"], "2")
        # Fast maths is gone: the old key is ignored, never written, and the legacy first line keeps it off.
        self.core.set_settings({"fast_math": 1})
        self.assertEqual(ctl()["_first"], "2 0 0")
        self.assertNotIn("fast_math", ctl())
        self.assertNotIn("fast_math", self.core.state["settings"])
        # Each capture request bumps the third value; settings changes keep it.
        r = self.core.request_capture()
        self.assertEqual(r["seq"], 1)
        self.assertEqual(ctl()["_first"], "2 0 1")
        self.core.request_capture()
        self.core.set_settings({"nss_quality": 1})
        self.assertEqual(ctl()["_first"], "1 0 2")
        # XeSS (3), and FSR 4 (upscaler 4) with a model: Auto=8, Lite=4, INT8=5, INT8 RDNA 2=6. The NSS quality is
        # remembered for when NSS is picked again.
        self.core.set_settings({"upscaler": 3})
        self.assertEqual(ctl()["_first"], "3 0 2")
        for model, code in ((0, 8), (1, 4), (2, 5), (3, 6)):
            self.core.set_settings({"upscaler": 4, "fsr4_model": model})
            self.assertEqual(ctl()["_first"], f"{code} 0 2")
            self.assertEqual(ctl()["upscaler"], str(code))
        self.core.set_settings({"upscaler": 0})
        self.assertEqual(ctl()["_first"], "1 0 2")
        # Older frontends sent NSS Balanced / Performance as upscaler 1 / 2.
        self.core.set_settings({"upscaler": 2})
        self.assertEqual((self.core.state["settings"]["upscaler"], self.core.state["settings"]["nss_quality"]), (0, 2))
        # watermark, gpu timing and xess preset all go through the control file; the removed fsr4_path is ignored
        self.core.set_settings({"fsr4_path": 2, "watermark": 1, "gpu_timing": 1, "xess_quality": 101})
        self.assertEqual((ctl()["watermark"], ctl()["gpu_timing"], ctl()["xess_quality"]), ("1", "1", "101"))
        # Motion vector dilation for FSR: off by default, switchable through the control file. Equal-cost render
        # scale is gone: no longer written, and an old saved setting is dropped.
        self.assertEqual(ctl()["dilate_mv"], "0")
        self.core.set_settings({"dilate_mv": 1})
        self.assertEqual(ctl()["dilate_mv"], "1")
        self.core.set_settings({"dilate_mv": 0})
        self.assertNotIn("equal_cost", ctl())
        self.assertNotIn("equal_cost", self.core.state["settings"])
        # FSR 3.1 (from AMD's SDK Vulkan DLL) is upscaler 9, written as such.
        self.core.set_settings({"upscaler": 9})
        self.assertEqual(ctl()["upscaler"], "9")
        self.core.set_settings({"upscaler": 0})
        self.assertNotIn("fsr4_path", ctl())
        self.assertNotIn("fsr4_path", self.core.state["settings"])
        for bad in ({"upscaler": 5}, {"fsr4_model": 4}, {"xess_quality": 99}, {"nss_quality": 3}, {"dilate_mv": 2}, {"upscaler": 7},
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
        # What the plugin wrote under its former name (AISS_RATIO / AISS_NSS_QUALITY in early versions) is removed on
        # the next sync; the user's own DU_ variables stay.
        old = lo["options"].replace("DU_LAYER=1", "AISS_LAYER=1 AISS_RATIO=2.0 AISS_NSS_QUALITY=0 DU_RATIO=1.5")
        lo2 = self.core.launch_options("1002", old, True)
        self.assertEqual(lo2["options"], f'PROTON_FORCE_NVAPI=1 DU_LAYER=1 DU_CONTROL={control} '
                                         'DU_LOG=0 WINEDLLOVERRIDES="dbghelp=n,b" DU_RATIO=1.5 %command% -novid')
        self.core = self.make_core()  # ownership record persisted
        off = self.core.launch_options("1002", lo2["options"], False)
        self.assertEqual(off["options"], "DU_RATIO=1.5 %command% -novid")

    # ---- the plugin's former name (aiss-dlss-nss) -----------------------------------------------------------------
    def old_install(self) -> dict:
        """Games 1002 and 1006 as the plugin left them under its former name: its builds under the old names and
        markers, its backups with the old suffixes, its FSR 4 files in system32/aiss, its layer, and its state file in
        <settings>/aiss-dlss-nss/. Returns the old state."""
        c = self.steam / "steamapps/common"
        games = {g["appid"]: g for g in self.core.scan(force=True)}
        old_shim = b"MZ old shim " + "AISS_SIGNED_DLL".encode("utf-16-le")
        state = {"version": 1, "games": {}, "layer": {"files": {}},
                 "settings": {"upscaler": 3, "nss_quality": 2, "fsr4_model": 1, "log": 2}}
        for appid in ("1002", "1006"):
            sys32 = self.pfx(self.steam, appid) / "drive_c/windows/system32"
            gs = {"enabled": True, "name": games[appid]["name"], "shim": "dbghelp", "targets": games[appid]["targets"],
                  "install_path": games[appid]["install_path"], "files": {}, "message": ""}
            for t in games[appid]["targets"]:
                self.put(Path(t) / "dbghelp.dll", old_shim)
                gs["files"][str(Path(t) / "dbghelp.dll")] = sha256_of(Path(t) / "dbghelp.dll")
            for sk, rel, data in (("core", "_nvngx.dll", b"MZ old core AISS_RATIO"),
                                  ("xess", "libxess.dll", b"MZ old libxess"),
                                  ("fsr4_dx12", "aiss/amdxcffx64.dll", b"MZ old amdxcffx64"),
                                  ("fsr4_shaders", "aiss/fsr4_shaders.bin", b"AISSSPV1 old pack")):
                if sk == "core" and (sys32 / rel).exists():  # 1002's own core, kept by the old plugin
                    os.replace(sys32 / rel, sys32 / (rel + ".aiss-orig"))
                    gs["core_backup"] = str(sys32 / (rel + ".aiss-orig"))
                self.put(sys32 / rel, data)
                gs[f"{sk}_path"], gs[f"{sk}_hash"] = str(sys32 / rel), sha256_of(sys32 / rel)
            reg = self.pfx(self.steam, appid) / "system.reg"
            (reg.parent / "system.reg.aiss-bak").write_text(REG_SAMPLE)
            reg.write_text(reg_set_value(REG_SAMPLE, REG_KEY_PATH, SIG_OVERRIDE_VALUE, "hex:01")[0])
            gs.update(reg_added=True, reg_created_key=True, reg_pending=False, reg_path=str(reg))
            state["games"][appid] = gs
        # 1006 also has a game FSR (amd_fidelityfx_vk.dll): the old forwarder, AMD's DLL kept under the old name
        ffx = c / "DoomLike/amd-ffx"
        self.put(ffx / "amd_fidelityfx_vk.dll", b"MZ old forwarder AISS_FfxForwarderVersion")
        self.put(ffx / "amd_fidelityfx_vk.aiss-orig.dll", b"MZ AMD ffx")
        state["games"]["1006"]["ffx_files"] = {str(ffx / "amd_fidelityfx_vk.dll"): "0" * 64}
        state["games"]["1002"]["launch"] = {
            "shim": "dbghelp", "command_added": True, "nvapi_added": True, "wdo_changed": True,
            "wdo_new_value": "dbghelp=n,b", "wdo_orig_token": None, "wdo_prev_present": False, "wdo_prev_mode": None}
        # the old layer: the .so as recorded, the manifest recognised by its layer name only
        self.put(self.layer / "libVkLayer_aiss_ngx.so", b"\x7fELF old layer VK_LAYER_AISS_ngx")
        self.put(self.layer / "VkLayer_aiss_ngx.json", b'{"layer": {"name": "VK_LAYER_AISS_ngx"}}')
        state["layer"]["files"][str(self.layer / "libVkLayer_aiss_ngx.so")] = \
            sha256_of(self.layer / "libVkLayer_aiss_ngx.so")
        old_state = self.tmp / "aiss-dlss-nss/state.json"
        old_state.parent.mkdir(parents=True)
        old_state.write_text(json.dumps(state))
        return state

    def migration_bundle(self) -> Path:
        """This bundle plus a stand-in FFX forwarder (with the current marker)."""
        b = self.tmp / "bin_migration"
        b.mkdir()
        for f in self.bin.iterdir():
            os.symlink(f.resolve(), b / f.name)
        (b / "deckyupscaler_ffx_vk.dll").write_bytes(b"MZ new forwarder " + core.FFX_MARKER)
        return b

    def test_migration_from_old_plugin(self):
        old_state = self.old_install()
        old_text = (self.tmp / "aiss-dlss-nss/state.json").read_bytes()
        bundle = self.migration_bundle()
        c = self.steam / "steamapps/common"
        self.core = self.make_core(bin_dir=bundle)
        # 1. the old state is imported on the first start (and saved as ours), the old file left as it was
        self.assertTrue((self.tmp / "settings/state.json").is_file())
        self.assertEqual((self.tmp / "aiss-dlss-nss/state.json").read_bytes(), old_text)
        st = self.core.state
        self.assertEqual(sorted(a for a, g in st["games"].items() if g["enabled"]), ["1002", "1006"])
        self.assertEqual((st["settings"]["upscaler"], st["settings"]["nss_quality"], st["settings"]["log"]), (3, 2, 2))
        self.assertEqual(st["games"]["1002"]["launch"], old_state["games"]["1002"]["launch"])
        self.assertEqual(st["games"]["1002"]["core_backup"], old_state["games"]["1002"]["core_backup"])
        gs = st["games"]["1006"]
        self.assertNotIn("fsr4_dx12_path", gs)  # system32/aiss paths became legacy records
        self.assertEqual(sorted(Path(p).name for p in gs["legacy_files"]), ["amdxcffx64.dll", "fsr4_shaders.bin"])
        self.assertEqual(st["layer"]["files"], {})
        self.assertIn(str(self.layer / "libVkLayer_aiss_ngx.so"), st["layer"]["legacy_files"])
        # 2. the old builds count as ours: an upgrade to repair, not a conflict
        for a in ("1002", "1006"):
            self.assertEqual(self.view(a)["status"], "needs_repair")
        res = self.core.verify_all()
        self.assertEqual(sorted((r["appid"], r["status"]) for r in res), [("1002", "installed"), ("1006", "installed")])
        for a in ("1002", "1006"):
            self.assertEqual(self.view(a)["status"], "installed")
            sys32 = self.pfx(self.steam, a) / "drive_c/windows/system32"
            self.assertEqual(sha256_of(sys32 / "_nvngx.dll"), sha256_of(bundle / "nvngx.dll"))
            self.assertEqual(sha256_of(sys32 / "libxess.dll"), sha256_of(bundle / "libxess.dll"))
            # 3. the old FSR 4 folder is gone, the files are in the new one
            self.assertFalse((sys32 / "aiss").exists())
            if (bundle / "amdxcffx64.dll").is_file():
                self.assertEqual(sha256_of(sys32 / "deckyupscaler/amdxcffx64.dll"), sha256_of(bundle / "amdxcffx64.dll"))
            # the old backups are reused, not doubled
            self.assertFalse((sys32 / "_nvngx.dll.deckyupscaler-orig").exists())
            self.assertFalse((sys32.parent.parent.parent / "system.reg.deckyupscaler-bak").exists())
        self.assertEqual((self.pfx(self.steam, "1002") / "drive_c/windows/system32/_nvngx.dll.aiss-orig").read_bytes(),
                         b"MZ someone else's core")
        for t in self.core.state["games"]["1006"]["targets"]:
            self.assertEqual(sha256_of(Path(t) / "dbghelp.dll"), sha256_of(bundle / "dbghelp.dll"))
        ffx = c / "DoomLike/amd-ffx"
        self.assertEqual((ffx / "amd_fidelityfx_vk.dll").read_bytes(), (bundle / "deckyupscaler_ffx_vk.dll").read_bytes())
        self.assertEqual((ffx / "amd_fidelityfx_vk.deckyupscaler-orig.dll").read_bytes(), b"MZ AMD ffx")
        self.assertFalse((ffx / "amd_fidelityfx_vk.aiss-orig.dll").exists())
        self.assertFalse((self.layer / "libVkLayer_aiss_ngx.so").exists())
        self.assertFalse((self.layer / "VkLayer_aiss_ngx.json").exists())
        self.assertTrue((self.layer / "libVkLayer_deckyupscaler.so").exists())
        self.assertNotIn("legacy_files", self.core.state["layer"])
        self.assertNotIn("legacy_files", self.core.state["games"]["1006"])
        self.assertEqual(self.core.verify_all(), [])  # done: nothing more to repair
        # 4. launch options: the old managed variables give way to the new ones
        old_lo = ('PROTON_FORCE_NVAPI=1 AISS_LAYER=1 AISS_CONTROL=/home/deck/.local/share/aiss-dlss-nss/control '
                  'AISS_LOG=2 WINEDLLOVERRIDES="dbghelp=n,b" %command% -novid')
        lo = self.core.launch_options("1002", old_lo, True)["options"]
        control = self.home / ".local/share/deckyupscaler/control"
        self.assertEqual(lo, f'PROTON_FORCE_NVAPI=1 DU_LAYER=1 DU_CONTROL={control} DU_LOG=2 '
                             'WINEDLLOVERRIDES="dbghelp=n,b" %command% -novid')
        self.assertEqual(self.core.launch_options("1002", lo, False)["options"], "-novid")
        # uninstall puts back what the old plugin backed up, under its old names
        self.core.uninstall_game("1002")
        sys32 = self.pfx(self.steam, "1002") / "drive_c/windows/system32"
        self.assertEqual((sys32 / "_nvngx.dll").read_bytes(), b"MZ someone else's core")
        self.assertFalse((sys32 / "_nvngx.dll.aiss-orig").exists())
        self.assertEqual((self.pfx(self.steam, "1002") / "system.reg").read_text(), REG_SAMPLE)
        self.core.uninstall_game("1006")
        self.assertEqual((ffx / "amd_fidelityfx_vk.dll").read_bytes(), b"MZ AMD ffx")
        # the import happens once: the next start reads its own state
        self.core = self.make_core(bin_dir=bundle)
        self.assertFalse(self.core.state["games"]["1002"]["enabled"])
        self.assertEqual((self.tmp / "aiss-dlss-nss/state.json").read_bytes(), old_text)

    def test_migration_leaves_what_is_not_the_old_plugins(self):
        self.old_install()
        bundle = self.migration_bundle()
        sys32 = self.pfx(self.steam, "1002") / "drive_c/windows/system32"
        # changed since the old plugin installed them: someone else's now
        (sys32 / "aiss/amdxcffx64.dll").write_bytes(b"MZ hand-copied amdxcffx64")
        self.put(sys32 / "aiss/readme.txt", b"user notes")
        (self.layer / "VkLayer_aiss_ngx.json").write_text('{"layer": {"name": "VK_LAYER_other"}}')
        self.core = self.make_core(bin_dir=bundle)
        self.assertEqual(self.core.install_game("1002")["status"], "installed")
        self.assertEqual((sys32 / "aiss/amdxcffx64.dll").read_bytes(), b"MZ hand-copied amdxcffx64")
        self.assertEqual((sys32 / "aiss/readme.txt").read_bytes(), b"user notes")
        self.assertFalse((sys32 / "aiss/fsr4_shaders.bin").exists())  # still as recorded: removed
        self.core.verify_all()
        self.assertTrue((self.layer / "VkLayer_aiss_ngx.json").exists())
        self.assertFalse((self.layer / "libVkLayer_aiss_ngx.so").exists())
        # an old-suffix backup the state does not record is restored on uninstall too
        self.core.state["games"]["1002"].pop("core_backup")
        self.core.uninstall_game("1002")
        self.assertEqual((sys32 / "_nvngx.dll").read_bytes(), b"MZ someone else's core")
        # without the old plugin's state nothing is imported
        shutil.rmtree(self.tmp / "settings")
        shutil.rmtree(self.tmp / "aiss-dlss-nss")
        self.core = self.make_core(bin_dir=bundle)
        self.assertEqual(self.core.state["games"], {})
        self.assertFalse((self.tmp / "settings/state.json").exists())

    def test_settings_migration(self):
        # Saved settings from older plugins: NSS Balanced / Performance were upscaler 1 / 2, fast maths was a setting,
        # and before that a single 'quality' key held everything.
        m = core.migrate_settings({"upscaler": 2, "fast_math": 1})
        self.assertEqual((m["upscaler"], m["nss_quality"]), (0, 2))
        self.assertNotIn("fast_math", m)
        m = core.migrate_settings({"quality": 1})
        self.assertEqual((m["upscaler"], m["nss_quality"]), (0, 1))
        m = core.migrate_settings({"quality": 5})
        self.assertEqual((m["upscaler"], m["fsr4_model"]), (4, 2))
        m = core.migrate_settings({})
        self.assertEqual((m["upscaler"], m["nss_quality"]), (0, 0))  # NSS Quality by default

    def test_main_py_loads_with_stub_decky(self):
        stub = types.ModuleType("decky")
        stub.DECKY_PLUGIN_DIR = str(PLUGIN)
        stub.DECKY_USER_HOME = str(self.home)
        stub.DECKY_PLUGIN_SETTINGS_DIR = str(self.tmp / "settings")
        stub.logger = Quiet()
        sys.modules["decky"] = stub
        import importlib.util
        spec = importlib.util.spec_from_file_location("deckyupscaler_main", PLUGIN / "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import asyncio
        p = mod.Plugin()
        mod.Plugin.core = None
        st = asyncio.run(p.get_state(True))
        self.assertIn("games", st)
        self.assertIn("settings", st)
        r = asyncio.run(p.set_settings({"upscaler": 7}))  # FP8: not offered
        self.assertFalse(r["ok"])
        mod.Plugin.core = None


# ------------------------------------------------------------------------------------------------------------------
class RealLibraryReadOnly(unittest.TestCase):
    def test_real_library_scan_readonly(self):
        home = Path("/home/deck")
        if not (home / ".local/share/Steam/steamapps/libraryfolders.vdf").exists():
            self.skipTest("no Steam library")
        tmp = Path(tempfile.mkdtemp(prefix="deckyupscaler-real-"))
        try:
            c = UpscalerCore(home=home, bin_dir=bin_dir(tmp), state_file=tmp / "state.json",
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
