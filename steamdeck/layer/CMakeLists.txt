# VK_LAYER_AISS_ngx: native Linux implicit Vulkan layer (see aiss_layer.c).
cmake_minimum_required(VERSION 3.20)
project(aiss_layer C)

set(NGSDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "Neural graphics SDK checkout")
add_library(VkLayer_aiss_ngx SHARED aiss_layer.c)
target_include_directories(VkLayer_aiss_ngx PRIVATE "${NGSDK_ROOT}/sdk/include/vulkan-headers")
target_compile_options(VkLayer_aiss_ngx PRIVATE -O2 -Wall -Wno-unused-parameter -fvisibility=hidden)
target_link_options(VkLayer_aiss_ngx PRIVATE -Wl,--no-undefined -s)
target_link_libraries(VkLayer_aiss_ngx PRIVATE pthread)
configure_file(VkLayer_aiss_ngx.json "${CMAKE_CURRENT_BINARY_DIR}/VkLayer_aiss_ngx.json" COPYONLY)
