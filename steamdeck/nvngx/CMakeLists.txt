# nvngx.dll: NGX core replacement that runs Arm NSS for DLSS Super Resolution under Proton.
# Links the SDK's static Windows libraries (build them first, see steamdeck/build.sh).
cmake_minimum_required(VERSION 3.20)
project(aiss_nvngx CXX)

set(NGSDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "Neural graphics SDK checkout")
set(NGSDK_LIBS "${NGSDK_ROOT}/bin" CACHE PATH "Directory holding libngsdk_windows_x64.a")
find_program(GLSLC glslc REQUIRED)
# Intel XeSS SDK: only its headers are used at build time; libxess.dll is loaded at runtime.
set(XESS_SDK "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/xess/sdk" CACHE PATH "Intel XeSS SDK (headers)")

set(SPV_INC "${CMAKE_CURRENT_BINARY_DIR}/depth_copy.spv.inc")
add_custom_command(
    OUTPUT "${SPV_INC}"
    COMMAND "${GLSLC}" -O --target-env=vulkan1.1 -mfmt=num -o "${SPV_INC}" "${CMAKE_CURRENT_SOURCE_DIR}/depth_copy.comp"
    DEPENDS depth_copy.comp
    COMMENT "Compiling depth_copy.comp")
set(EXPOSURE_SPV_INC "${CMAKE_CURRENT_BINARY_DIR}/exposure.spv.inc")
add_custom_command(
    OUTPUT "${EXPOSURE_SPV_INC}"
    COMMAND "${GLSLC}" -O --target-env=vulkan1.1 -mfmt=num -o "${EXPOSURE_SPV_INC}" "${CMAKE_CURRENT_SOURCE_DIR}/exposure.comp"
    DEPENDS exposure.comp
    COMMENT "Compiling exposure.comp")

# Region copy for colour / motion vectors whose texture is larger than the render size, one variant per output format.
set(REGION_COPY_INCS "")
foreach(fmt rgba16f rgba32f r11f_g11f_b10f rgba8 rgb10_a2 rg16f rg32f)
    set(inc "${CMAKE_CURRENT_BINARY_DIR}/region_copy_${fmt}.spv.inc")
    add_custom_command(
        OUTPUT "${inc}"
        COMMAND "${GLSLC}" -O --target-env=vulkan1.1 -mfmt=num -DOUT_FORMAT=${fmt} -o "${inc}" "${CMAKE_CURRENT_SOURCE_DIR}/region_copy.comp"
        DEPENDS region_copy.comp
        COMMENT "Compiling region_copy.comp (${fmt})")
    list(APPEND REGION_COPY_INCS "${inc}")
endforeach()

add_library(nvngx SHARED
    common.cpp
    params.cpp
    nss_upscaler.cpp
    ngx_common.cpp
    ngx_d3d11.cpp
    ngx_d3d12.cpp
    ngx_vulkan.cpp
    ngx_misc.cpp
    xess_upscaler.cpp
    fsr4_common.cpp
    fsr4_dx12.cpp
    fsr4_upscaler.cpp
    spirv_half_images.cpp
    frame_dump.cpp
    gpu_timing.cpp
    "${SPV_INC}"
    "${EXPOSURE_SPV_INC}"
    ${REGION_COPY_INCS})

set_target_properties(nvngx PROPERTIES PREFIX "" OUTPUT_NAME nvngx)
target_compile_features(nvngx PRIVATE cxx_std_17)
target_compile_options(nvngx PRIVATE -O2 -Wall -Wno-unused-parameter)
target_compile_definitions(nvngx PRIVATE FFX_BACKEND_VK VK_USE_PLATFORM_WIN32_KHR WIN32_LEAN_AND_MEAN NOMINMAX)
target_include_directories(nvngx PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${NGSDK_ROOT}/ffx-api/include"
    "${NGSDK_ROOT}/sdk/include"
    "${NGSDK_ROOT}/sdk/include/vulkan-headers"
    "${XESS_SDK}/inc")
target_link_libraries(nvngx PRIVATE
    "${NGSDK_LIBS}/libngsdk_windows_x64.a"
    "${NGSDK_LIBS}/ffx_sdk/windows_x64/libffx_nss_windows_x64.a"
    "${NGSDK_LIBS}/ffx_sdk/windows_x64/libffx_backend_vk_windows_x64.a"
    "${NGSDK_LIBS}/ffx_sdk/windows_x64/libSPIRV-Tools-opt.a"
    "${NGSDK_LIBS}/ffx_sdk/windows_x64/libSPIRV-Tools.a"
    dxguid
    version)
# Only the NGX entry points are exported (they carry __declspec(dllexport)).
target_link_options(nvngx PRIVATE -Wl,--exclude-all-symbols -s)
