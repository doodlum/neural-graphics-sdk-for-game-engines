#!/usr/bin/env bash
# Builds the "AISS DLSS-NSS" Decky plugin.
#   1. copies the built binaries into bin/ (from the repo build dirs, override with env vars below)
#   2. builds the frontend (dist/index.js) with pnpm, inside the `aissbuild` distrobox when node is not on the host
#   3. assembles out/aiss-dlss-nss/ (the folder that goes into ~/homebrew/plugins/) and out/aiss-dlss-nss.zip
#
#   steamdeck/decky/build.sh            full build
#   SKIP_FRONTEND=1 steamdeck/decky/build.sh   refresh binaries + package only (reuses existing dist/)
#   PNPM="node /path/pnpm/bin/pnpm.cjs" steamdeck/decky/build.sh   use a pnpm that is not on PATH
#
# Install (Decky plugin dirs are root-owned):
#   sudo rm -rf ~/homebrew/plugins/aiss-dlss-nss
#   sudo cp -r steamdeck/decky/out/aiss-dlss-nss ~/homebrew/plugins/
#   sudo systemctl restart plugin_loader
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NVNGX_DLL="${NVNGX_DLL:-$REPO/build_win_nvngx/nvngx.dll}"
LOADER_DIR="${LOADER_DIR:-$REPO/build_win_loader}"
LAYER_DIR="${LAYER_DIR:-$REPO/build_layer}"
XESS_SDK="${XESS_SDK:-$REPO/../third_party/xess/sdk}"  # Intel XeSS SDK: libxess.dll + LICENSE.txt, shipped unmodified
DISTROBOX="${DISTROBOX:-aissbuild}"
NAME="aiss-dlss-nss"

echo "== binaries -> bin/"
mkdir -p "$HERE/bin"
for f in "$NVNGX_DLL" "$LOADER_DIR/dbghelp/dbghelp.dll" "$LOADER_DIR/dxgi/dxgi.dll" \
         "$LAYER_DIR/libVkLayer_aiss_ngx.so" "$LAYER_DIR/VkLayer_aiss_ngx.json"; do
    [ -f "$f" ] || { echo "missing $f - build it first" >&2; exit 1; }
    install -m 0755 "$f" "$HERE/bin/"
done
chmod 0644 "$HERE/bin/VkLayer_aiss_ngx.json"
[ -f "$XESS_SDK/bin/libxess.dll" ] && [ -f "$XESS_SDK/LICENSE.txt" ] || { echo "missing XeSS SDK at $XESS_SDK" >&2; exit 1; }
install -m 0644 "$XESS_SDK/bin/libxess.dll" "$HERE/bin/libxess.dll"
install -m 0644 "$XESS_SDK/LICENSE.txt" "$HERE/bin/libxess-LICENSE.txt"
echo "$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$REPO" diff --quiet 2>/dev/null || echo -dirty) built $(date -u +%Y-%m-%dT%H:%MZ)" > "$HERE/bin/VERSION"
(cd "$HERE/bin" && sha256sum nvngx.dll dbghelp.dll dxgi.dll libVkLayer_aiss_ngx.so VkLayer_aiss_ngx.json libxess.dll \
    libxess-LICENSE.txt > SHA256SUMS)

if [ "${SKIP_FRONTEND:-0}" != 1 ]; then
    echo "== frontend"
    # PNPM may name a pnpm command, e.g. PNPM="node /path/to/pnpm/bin/pnpm.cjs"
    PNPM="${PNPM:-pnpm}"
    FRONTEND_CMD="cd '$HERE' && $PNPM install --frozen-lockfile=false && $PNPM run build"
    if command -v node >/dev/null 2>&1 && { [ "$PNPM" != pnpm ] || command -v pnpm >/dev/null 2>&1; }; then
        bash -c "$FRONTEND_CMD"
    else
        distrobox enter "$DISTROBOX" -- bash -c "command -v node >/dev/null && { [ '$PNPM' != pnpm ] || command -v pnpm >/dev/null; } || sudo pacman -S --noconfirm --needed nodejs pnpm; $FRONTEND_CMD"
    fi
fi
[ -f "$HERE/dist/index.js" ] || { echo "dist/index.js missing" >&2; exit 1; }

echo "== package -> out/$NAME"
OUT="$HERE/out/$NAME"
rm -rf "$OUT" "$HERE/out/$NAME.zip"
mkdir -p "$OUT/py_modules" "$OUT/bin" "$OUT/dist"
cp "$HERE/plugin.json" "$HERE/package.json" "$HERE/main.py" "$OUT/"
cp "$HERE/py_modules/aiss_core.py" "$OUT/py_modules/"
cp "$HERE/dist/index.js" "$OUT/dist/"
[ -f "$HERE/dist/index.js.map" ] && cp "$HERE/dist/index.js.map" "$OUT/dist/"
cp -p "$HERE/bin/"* "$OUT/bin/"
if command -v zip >/dev/null 2>&1; then
    (cd "$HERE/out" && zip -qr "$NAME.zip" "$NAME")
else
    python3 -c "import shutil,sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2], sys.argv[3])" \
        "$HERE/out/$NAME" "$HERE/out" "$NAME"
fi
echo "done: $OUT  ($HERE/out/$NAME.zip)"
echo "install: sudo rm -rf ~/homebrew/plugins/$NAME && sudo cp -r '$OUT' ~/homebrew/plugins/ && sudo systemctl restart plugin_loader"
