#!/usr/bin/env python3
"""Packs replacement FSR 4 shader modules into the file the proxy loads from beside AMD's SDK Vulkan DLL
(system32\\aiss\\fsr4_shaders.bin, see Shaders() in nvngx/fsr4_vk.cpp).

    pack_fsr4_shaders.py DIR OUT    DIR holds <hash>.spv files: <hash> is the 16-hex-digit FNV-1a hash of the original
                                    module the file replaces (as AISS_FSR4_VK_DUMP names them)

Format: b"AISSSPV1", u32 count, then per module u64 hash, u32 word count and the SPIR-V words (little endian).
"""
import struct
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    entries = []
    for f in sorted(Path(sys.argv[1]).glob("*.spv")):
        code = f.read_bytes()
        if len(f.stem) != 16 or len(code) % 4 or code[:4] != b"\x03\x02\x23\x07":
            print(f"skipping {f.name}: not <16-hex-hash>.spv SPIR-V")
            continue
        entries.append((int(f.stem, 16), code))
    out = bytearray(b"AISSSPV1" + struct.pack("<I", len(entries)))
    for h, code in entries:
        out += struct.pack("<QI", h, len(code) // 4) + code
    Path(sys.argv[2]).write_bytes(out)
    print(f"{sys.argv[2]}: {len(entries)} modules, {len(out)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
