#!/bin/bash
# Builds everything for the Steam Deck: the SDK as Windows static libraries, nvngx.dll (the NGX proxy), the loader
# shims, the native Vulkan layer and the test harnesses. Run inside a Linux environment with mingw-w64, cmake, ninja,
# make, glslc and python (e.g. the Arch distrobox described in steamdeck/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
TOOLCHAIN=$ROOT/steamdeck/cmake/mingw-w64-x86_64.cmake

# SDK (NSS only) as static Windows libraries in bin/. Unix Makefiles: the shader compiler step relies on make's
# serial invocation of the permutation tool.
cmake -S . -B build_win_mk -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DFFX_API_BACKEND=vk_windows_x64 \
    -DFFX_FI=OFF -DFFX_OF=OFF -DFFX_BUILD_AS_DLL=OFF -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DVulkan_LIBRARY="$ROOT/steamdeck/cmake/vulkan/libvulkan-1.dll.a"
make -C build_win_mk -j"$(nproc)"

for part in nvngx loader tests; do
    cmake -S steamdeck/$part -B build_win_$part -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN"
    ninja -C build_win_$part
done

cmake -S steamdeck/layer -B build_layer -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build_layer

echo
echo "nvngx.dll:      build_win_nvngx/nvngx.dll"
echo "loader shims:   build_win_loader/{dbghelp,dxgi}/*.dll"
echo "Vulkan layer:   build_layer/{libVkLayer_aiss_ngx.so,VkLayer_aiss_ngx.json}"
echo "D3D11 harness:  build_win_tests/ngx_d3d11_test.exe"
