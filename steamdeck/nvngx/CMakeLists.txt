# nvngx.dll: NGX core replacement that runs Arm NSS for DLSS Super Resolution under Proton.
# Links the SDK's static Windows libraries (build them first, see steamdeck/build.sh).
cmake_minimum_required(VERSION 3.20)
project(aiss_nvngx CXX)

set(NGSDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "Neural graphics SDK checkout")
set(NGSDK_LIBS "${NGSDK_ROOT}/bin" CACHE PATH "Directory holding libngsdk_windows_x64.a")
find_program(GLSLC glslc REQUIRED)

set(SPV_INC "${CMAKE_CURRENT_BINARY_DIR}/depth_copy.spv.inc")
add_custom_command(
    OUTPUT "${SPV_INC}"
    COMMAND "${GLSLC}" -O --target-env=vulkan1.1 -mfmt=num -o "${SPV_INC}" "${CMAKE_CURRENT_SOURCE_DIR}/depth_copy.comp"
    DEPENDS depth_copy.comp
    COMMENT "Compiling depth_copy.comp")

add_library(nvngx SHARED
    common.cpp
    params.cpp
    nss_upscaler.cpp
    ngx_common.cpp
    ngx_d3d12.cpp
    ngx_vulkan.cpp
    ngx_misc.cpp
    "${SPV_INC}")

set_target_properties(nvngx PROPERTIES PREFIX "" OUTPUT_NAME nvngx)
target_compile_features(nvngx PRIVATE cxx_std_17)
target_compile_options(nvngx PRIVATE -O2 -Wall -Wno-unused-parameter)
target_compile_definitions(nvngx PRIVATE FFX_BACKEND_VK VK_USE_PLATFORM_WIN32_KHR WIN32_LEAN_AND_MEAN NOMINMAX)
target_include_directories(nvngx PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${NGSDK_ROOT}/ffx-api/include"
    "${NGSDK_ROOT}/sdk/include"
    "${NGSDK_ROOT}/sdk/include/vulkan-headers")
target_link_libraries(nvngx PRIVATE
    "${NGSDK_LIBS}/libngsdk_windows_x64.a"
    "${NGSDK_LIBS}/ffx_sdk/windows_x64/libffx_nss_windows_x64.a"
    "${NGSDK_LIBS}/ffx_sdk/windows_x64/libffx_backend_vk_windows_x64.a"
    "${NGSDK_LIBS}/ffx_sdk/windows_x64/libSPIRV-Tools-opt.a"
    "${NGSDK_LIBS}/ffx_sdk/windows_x64/libSPIRV-Tools.a"
    dxguid)
# Only the NGX entry points are exported (they carry __declspec(dllexport)).
target_link_options(nvngx PRIVATE -Wl,--exclude-all-symbols -s)
