#!/bin/sh

# This will generate a cross-compile environment, compiling an aarch64
# Win64 target from a 32bit MinGW32 host environment.  If your MinGW
# install is 64bit, you can use the native compiler batch file:
# make-Makefiles.sh

cmake -G "MSYS Makefiles" -DCMAKE_TOOLCHAIN_FILE=toolchain-aarch64-w64-mingw32.cmake ../../source && cmake-gui ../../source
