# Cross-compile the SDK and the NGX proxy as Windows x64 PE binaries for Proton.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_RC_COMPILER x86_64-w64-mingw32-windres)
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# No libstdc++/libgcc DLLs next to the game.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -static-libgcc -static-libstdc++")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-static -static-libgcc -static-libstdc++")
# Case shims for MSVC-style includes such as <Windows.h>.
set(CMAKE_C_FLAGS_INIT "-I${CMAKE_CURRENT_LIST_DIR}/mingw-shims")
set(CMAKE_CXX_FLAGS_INIT "-I${CMAKE_CURRENT_LIST_DIR}/mingw-shims")
