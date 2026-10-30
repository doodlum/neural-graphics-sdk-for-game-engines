#!/usr/bin/env bash
# Builds the DeckyUpscaler Decky plugin.
#   1. copies the built binaries into bin/ (from the repo build dirs, override with env vars below)
#   2. builds the frontend (dist/index.js) with pnpm, inside the `aissbuild` distrobox when node is not on the host
#   3. assembles out/deckyupscaler/ (the folder that goes into ~/homebrew/plugins/) and out/deckyupscaler.zip
#
#   steamdeck/decky/build.sh            full build
#   SKIP_FRONTEND=1 steamdeck/decky/build.sh   refresh binaries + package only (reuses existing dist/)
#   PNPM="node /path/pnpm/bin/pnpm.cjs" steamdeck/decky/build.sh   use a pnpm that is not on PATH
#
# Install (Decky plugin dirs are root-owned). The plugin's former folder, aiss-dlss-nss, must go too: on its first
# start the new plugin takes over the old one's games, files and settings.
#   sudo rm -rf ~/homebrew/plugins/deckyupscaler ~/homebrew/plugins/aiss-dlss-nss
#   sudo cp -r steamdeck/decky/out/deckyupscaler ~/homebrew/plugins/
#   sudo systemctl restart plugin_loader
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NVNGX_DLL="${NVNGX_DLL:-$REPO/build_win_nvngx/nvngx.dll}"
LOADER_DIR="${LOADER_DIR:-$REPO/build_win_loader}"
LAYER_DIR="${LAYER_DIR:-$REPO/build_layer}"
XESS_SDK="${XESS_SDK:-$REPO/../third_party/xess/sdk}"  # Intel XeSS SDK: libxess.dll + LICENSE.txt, shipped unmodified
# AMD FSR 4: amdxcffx64.dll from AMD's D3D12 driver (Adrenalin), shipped unmodified; optional, kept outside the repo
FSR4_DX12_DLL="${FSR4_DX12_DLL:-$REPO/../third_party/amd/amdxcffx64.dll}"
# AMD FSR SDK 2.4 Vulkan upscaler (amd_fidelityfx_upscaler_vk.dll, FSR 4.1.2) and its licence, shipped unmodified; and
# replacement FSR 4 Lite shader modules (fsr4_shaders.bin). All optional, kept outside the repo.
FSR4_VK_DIR="${FSR4_VK_DIR:-$REPO/../third_party/amd/sdk240}"
FSR4_SHADERS="${FSR4_SHADERS:-$REPO/../third_party/amd/fsr4_shaders.bin}"
FSR4_SHADERS_DOT2="${FSR4_SHADERS_DOT2:-$REPO/../third_party/amd/fsr4_shaders_dot2.bin}"
DISTROBOX="${DISTROBOX:-aissbuild}"
NAME="deckyupscaler"

echo "== binaries -> bin/"
mkdir -p "$HERE/bin"
rm -f "$HERE/bin/libVkLayer_aiss_ngx.so" "$HERE/bin/VkLayer_aiss_ngx.json" "$HERE/bin/aiss_ffx_vk.dll"  # former names
for f in "$NVNGX_DLL" "$LOADER_DIR/dbghelp/dbghelp.dll" "$LOADER_DIR/dxgi/dxgi.dll" \
         "$LAYER_DIR/libVkLayer_deckyupscaler.so" "$LAYER_DIR/VkLayer_deckyupscaler.json"; do
    [ -f "$f" ] || { echo "missing $f - build it first" >&2; exit 1; }
    install -m 0755 "$f" "$HERE/bin/"
done
chmod 0644 "$HERE/bin/VkLayer_deckyupscaler.json"
[ -f "$XESS_SDK/bin/libxess.dll" ] && [ -f "$XESS_SDK/LICENSE.txt" ] || { echo "missing XeSS SDK at $XESS_SDK" >&2; exit 1; }
install -m 0644 "$XESS_SDK/bin/libxess.dll" "$HERE/bin/libxess.dll"
install -m 0644 "$XESS_SDK/LICENSE.txt" "$HERE/bin/libxess-LICENSE.txt"
FSR4_SUM=""
rm -f "$HERE/bin/amdvkffx64.dll"  # no longer used (FSR 4 runs only through amdxcffx64)
if [ -f "$FSR4_DX12_DLL" ]; then
    install -m 0644 "$FSR4_DX12_DLL" "$HERE/bin/amdxcffx64.dll"
    FSR4_SUM="amdxcffx64.dll"
else
    rm -f "$HERE/bin/amdxcffx64.dll"
    echo "warning: $FSR4_DX12_DLL not found - building without FSR 4 (the plugin hides its upscaler option)" >&2
fi
for pair in "$FSR4_VK_DIR/amd_fidelityfx_upscaler_vk.dll:amd_fidelityfx_upscaler_vk.dll" \
            "$FSR4_VK_DIR/license.md:amd_fidelityfx-LICENSE.md" "$FSR4_SHADERS:fsr4_shaders.bin" \
            "$FSR4_SHADERS_DOT2:fsr4_shaders_dot2.bin"; do
    src="${pair%%:*}"; dst="${pair##*:}"
    if [ -f "$src" ]; then
        install -m 0644 "$src" "$HERE/bin/$dst"
        FSR4_SUM="$FSR4_SUM $dst"
    else
        rm -f "$HERE/bin/$dst"
        echo "note: $src not found - building without $dst" >&2
    fi
done
echo "$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$REPO" diff --quiet 2>/dev/null || echo -dirty) built $(date -u +%Y-%m-%dT%H:%MZ)" > "$HERE/bin/VERSION"
# The FFX forwarder (stands in for a game's amd_fidelityfx_vk.dll), bundled under its own name.
install -m 0755 "$LOADER_DIR/ffx/amd_fidelityfx_vk.dll" "$HERE/bin/deckyupscaler_ffx_vk.dll"
(cd "$HERE/bin" && sha256sum nvngx.dll dbghelp.dll dxgi.dll libVkLayer_deckyupscaler.so VkLayer_deckyupscaler.json libxess.dll deckyupscaler_ffx_vk.dll \
    libxess-LICENSE.txt $FSR4_SUM > SHA256SUMS)

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
cp "$HERE/py_modules/deckyupscaler_core.py" "$OUT/py_modules/"
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
echo "install: sudo rm -rf ~/homebrew/plugins/$NAME ~/homebrew/plugins/aiss-dlss-nss && sudo cp -r '$OUT' ~/homebrew/plugins/ && sudo systemctl restart plugin_loader"
