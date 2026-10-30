#!/usr/bin/env python3
"""Packs replacement FSR 4 shader modules into the file the proxy loads from beside AMD's SDK Vulkan DLL
(system32\\deckyupscaler\\fsr4_shaders.bin, see Shaders() in nvngx/fsr4_vk.cpp).

    pack_fsr4_shaders.py DIR OUT [--cleans-colour] [--skip-dispatch HASH]...

    DIR holds <hash>.spv files: <hash> is the 16-hex-digit FNV-1a hash of the original module the file replaces (as
    DU_FSR4_VK_DUMP names them). The options add the set's options entry (reserved hash 0; or DIR holds
    0000000000000000.spv): --cleans-colour when the set's colour readers clean Inf / NaN themselves (the proxy then skips
    its clean-up pass), --skip-dispatch for each original module whose dispatches the set makes unnecessary.

Format: b"DUSPACK1", u32 count, then per module u64 hash, u32 word count and the SPIR-V words (little endian). The
options entry is a SPIR-V header (5 words) followed by u32 flags, u32 count and count u64 hashes. The proxy also reads
packs written by older builds of this tool (magic b"AISSSPV1").
"""
import struct
import sys
from pathlib import Path


def options_entry(cleans_colour: bool, skip: list) -> bytes:
    words = struct.pack("<5I", 0x07230203, 0x00010000, 0, 0, 0)
    words += struct.pack("<II", 1 if cleans_colour else 0, len(skip))
    return words + b"".join(struct.pack("<Q", h) for h in skip)


def main() -> int:
    args = sys.argv[1:]
    cleans_colour = "--cleans-colour" in args
    args = [a for a in args if a != "--cleans-colour"]
    skip = []
    while "--skip-dispatch" in args:
        i = args.index("--skip-dispatch")
        if i + 1 >= len(args):
            print(__doc__)
            return 2
        skip.append(int(args[i + 1], 16))
        del args[i : i + 2]
    if len(args) != 2:
        print(__doc__)
        return 2
    entries = []
    for f in sorted(Path(args[0]).glob("*.spv")):
        code = f.read_bytes()
        if len(f.stem) != 16 or len(code) % 4 or code[:4] != b"\x03\x02\x23\x07":
            print(f"skipping {f.name}: not <16-hex-hash>.spv SPIR-V")
            continue
        entries.append((int(f.stem, 16), code))
    if cleans_colour or skip:
        entries = [e for e in entries if e[0] != 0] + [(0, options_entry(cleans_colour, skip))]
    out = bytearray(b"DUSPACK1" + struct.pack("<I", len(entries)))
    for h, code in entries:
        out += struct.pack("<QI", h, len(code) // 4) + code
    Path(args[1]).write_bytes(out)
    print(f"{args[1]}: {len(entries)} entries, {len(out)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
