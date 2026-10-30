# VK_LAYER_DECKYUPSCALER: native Linux implicit Vulkan layer (see deckyupscaler_layer.c).
cmake_minimum_required(VERSION 3.20)
project(deckyupscaler_layer C)

set(NGSDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "Neural graphics SDK checkout")
# cmd_trampolines.inc is generated and committed: python3 gen_cmd_trampolines.py <vulkan_core.h> cmd_trampolines.inc
add_library(VkLayer_deckyupscaler SHARED deckyupscaler_layer.c)
target_include_directories(VkLayer_deckyupscaler PRIVATE "${NGSDK_ROOT}/sdk/include/vulkan-headers")
target_compile_options(VkLayer_deckyupscaler PRIVATE -O2 -Wall -Wno-unused-parameter -fvisibility=hidden)
target_link_options(VkLayer_deckyupscaler PRIVATE -Wl,--no-undefined -s)
target_link_libraries(VkLayer_deckyupscaler PRIVATE pthread)
configure_file(VkLayer_deckyupscaler.json "${CMAKE_CURRENT_BINARY_DIR}/VkLayer_deckyupscaler.json" COPYONLY)
