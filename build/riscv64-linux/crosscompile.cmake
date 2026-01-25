# CMake toolchain file for cross compiling x265 for riscv64
# This feature is only supported as experimental. Use with caution.
# Please report bugs on bitbucket
# Run cmake with: cmake -DCMAKE_TOOLCHAIN_FILE=crosscompile.cmake -G "Unix Makefiles" ../../source && ccmake ../../source

set(CROSS_COMPILE_RISCV64 1)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# specify the cross compiler (giving precedence to user-supplied CC/CXX)
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
endif()

# specify the target environment
SET(CMAKE_FIND_ROOT_PATH  /usr/riscv64-unknown-linux-gnu)
