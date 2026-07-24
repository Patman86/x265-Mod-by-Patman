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

if "%VS18_PATH%" == "" (
    msg "%username%" "Visual Studio 18 not detected"
    exit 1
)
echo %VS18_PATH%
setx VS18_PATH "!VS18_PATH!"
if not exist x265.slnx (
    call make-solutions.bat
)
if exist x265.slnx (
    call "%VS18_PATH%\..\..\tools\VsDevCmd.bat"
    MSBuild /p:Configuration="Release" x265.slnx
    MSBuild /p:Configuration="Debug" x265.slnx
    MSBuild /p:Configuration="RelWithDebInfo" x265.slnx
)
