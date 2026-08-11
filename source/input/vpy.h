/*****************************************************************************
 * Copyright (C) 2013-2020 MulticoreWare, Inc
 *
 * Authors: Vladimir Kontserenko <djatom@beatrice-raws.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at license @ x265.com.
 *****************************************************************************/

#ifndef X265_VPY_H
#define X265_VPY_H

#include <atomic>
#include <map>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>

#include <vapoursynth/VapourSynth4.h>

#include "input.h"

#if _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
using lib_path_t = std::wstring;
using lib_t = HMODULE;
using func_t = FARPROC;
#else
#include <unistd.h>
#define Sleep(x) usleep((x) * 1000)
#include <dlfcn.h>
#define __stdcall
using lib_path_t = std::string;
using lib_t = void*;
using func_t = void*;
#endif

#if defined(_WIN32) || defined(_WIN64)
#define CloseEvent CloseHandle
#else
#include "event.h"
#endif

namespace X265_NS {

using vss_api = const VSSCRIPTAPI* (VS_CC*)(int version);
using vss_last_error_func = const char* (VS_CC*)();

class VPYInput : public InputFile
{
protected:
    struct FrameSlot
    {
        int frameNumber = -1;
        bool ready = false;
        const VSFrame* frame = nullptr;
        HANDLE event = nullptr;
    };

    std::vector<FrameSlot> frameSlots;
    std::atomic<int> requestedFrames{ 0 };
    std::atomic<int> completedFrames{ 0 };
    std::atomic<int> pendingFrames{ 0 };
    std::atomic<bool> isRunning{ false };
    int framesToRequest{ 0 };
    int nextFrame{ 0 };
    int parallelRequests{ -1 };
    int nodeIndex{ 0 };
    int asyncFailedFrame{ -1 };
    bool abortAsync{ false };
    bool useScriptSar{ false };
    bool vpyFailed{ false };
    char frameError[512]{};
    size_t frameSize{ 0 };
    uint8_t* frameBuffer{ nullptr };
    InputFileInfo _info{};
    lib_t vss_library{ nullptr };

#if _WIN32
    lib_path_t vss_library_path{ L"vsscript" };
    void vs_open() { vss_library = LoadLibraryW(vss_library_path.c_str()); }
    void vs_close() { if (vss_library) { FreeLibrary(vss_library); vss_library = nullptr; } }
    func_t vs_address(LPCSTR func) { return GetProcAddress(vss_library, func); }
#else
#ifdef __MACH__
    lib_path_t vss_library_path{ "libvsscript.dylib" };
#else
    lib_path_t vss_library_path{ "libvsscript.so" };
#endif
    void vs_open() { vss_library = dlopen(vss_library_path.c_str(), RTLD_GLOBAL | RTLD_NOW); }
    void vs_close() { if (vss_library) { dlclose(vss_library); vss_library = nullptr; } }
    func_t vs_address(const char* func) { return dlsym(vss_library, func); }
#endif

    lib_path_t convertLibraryPath(const std::string& path);
    void parseVpyOptions(const char* options);
    void load_vs();
    bool tryLoadLibraryPath(const lib_path_t& path);
    void applyEnvironmentLibraryPath();
    int clampParallelRequests(int requested, int numThreads, int totalFrames) const;
    int slotIndex(int frameNumber) const;
    FrameSlot* getSlot(int frameNumber);
    void resetSlot(FrameSlot& slot, bool closeEvent = false);
    bool createSlotEvents();
    void freeSlotFrames();
    void releaseSlots();
    void requestFrame(int n);
    const VSFrame* getAsyncFrame(int n);

    const VSAPI* vsapi = nullptr;
    vss_api getVSScriptAPI = nullptr;
    vss_last_error_func getVSScriptAPILastError = nullptr;
    const VSSCRIPTAPI* vssapi = nullptr;
    VSScript* script = nullptr;
    VSNode* node = nullptr;
    VSCore* core = nullptr;

public:
    VPYInput(InputFileInfo& info);
    ~VPYInput() {}

    void setAsyncFrame(int n, const VSFrame* f, const char* errorMsg);
    void release();
    bool isEof() const { return nextFrame >= _info.frameCount; }
    bool isFail() { return vpyFailed; }
    void startReader();
    void stopReader();
    bool readPicture(x265_picture& pic);

    const char* getName() const { return "vpy"; }
    int getWidth() const { return _info.width; }
    int getHeight() const { return _info.height; }
    int outputFrame() { return nextFrame; }
};

}

#endif // X265_VPY_H
