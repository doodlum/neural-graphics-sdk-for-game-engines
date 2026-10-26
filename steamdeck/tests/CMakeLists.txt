# Harnesses that drive nvngx.dll like a game's NGX code does (run under Proton/Wine).
cmake_minimum_required(VERSION 3.20)
project(aiss_tests CXX)
add_executable(ngx_d3d11_test ngx_d3d11_test.cpp)
target_compile_features(ngx_d3d11_test PRIVATE cxx_std_17)
target_link_libraries(ngx_d3d11_test PRIVATE d3d11)
