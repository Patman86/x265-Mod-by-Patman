@echo off
setlocal enabledelayedexpansion

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if "%VS18_PATH%" == "" (
    for /f "usebackq tokens=1* delims=: " %%i in (`"!VSWHERE!" -latest -version [18.0^,19.0^)`) do (
        if /i "%%i"=="productPath" (
            set VS18_PATH=%%j
        )
    )
)
setx VS18_PATH "!VS18_PATH!"
call "%VS18_PATH%\..\..\tools\VsDevCmd.bat"

@mkdir 12bit
@mkdir 10bit
@mkdir 8bit

@cd 12bit
cmake -G "Visual Studio 18 2026" ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN12=ON
if exist x265.slnx (
    MSBuild /p:Configuration="Release" x265.slnx
    copy /y Release\x265-static.lib ..\8bit\x265-static-main12.lib
)

@cd ..\10bit
cmake -G "Visual Studio 18 2026" ../../../source -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF
if exist x265.slnx (
    MSBuild /p:Configuration="Release" x265.slnx
    copy /y Release\x265-static.lib ..\8bit\x265-static-main10.lib
)

@cd ..\8bit
if not exist x265-static-main10.lib (
    msg "%username%" "10bit build failed"
    exit 1
)
if not exist x265-static-main12.lib (
    msg "%username%" "12bit build failed"
    exit 1
)
cmake -G "Visual Studio 18 2026" ../../../source -DEXTRA_LIB="x265-static-main10.lib;x265-static-main12.lib" -DLINKED_10BIT=ON -DLINKED_12BIT=ON
if exist x265.slnx (
    MSBuild /p:Configuration="Release" x265.slnx
    :: combine static libraries (ignore warnings caused by winxp.cpp hacks)
    move Release\x265-static.lib x265-static-main.lib
    LIB.EXE /ignore:4006 /ignore:4221 /OUT:Release\x265-static.lib x265-static-main.lib x265-static-main10.lib x265-static-main12.lib
)

pause
