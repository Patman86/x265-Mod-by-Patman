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

#include "vpy.h"
#include "cli_log.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

using namespace X265_NS;

static void __stdcall frameDoneCallback(void* userData, const VSFrame* f, int n, VSNode*, const char* errorMsg)
{
    reinterpret_cast<VPYInput*>(userData)->setAsyncFrame(n, f, errorMsg);
}

static void VS_CC logMessageHandler(int msgType, const char* msg, void*)
{
    auto vsToX265LogLevel = [msgType]()
    {
        switch (msgType)
        {
        case mtDebug:       return X265_LOG_DEBUG;
        case mtInformation: return X265_LOG_INFO;
        case mtWarning:     return X265_LOG_WARNING;
        case mtCritical:    return X265_LOG_WARNING;
        case mtFatal:       return X265_LOG_ERROR;
        default:            return X265_LOG_FULL;
        }
    };
    vpy_log(vsToX265LogLevel(), "%s\n", msg ? msg : "");
}

lib_path_t VPYInput::convertLibraryPath(const std::string& path)
{
#if defined(_WIN32)
    if (path.empty())
        return std::wstring();

    int size_needed = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (size_needed <= 0)
        return std::wstring();

    std::wstring wide((size_t)size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wide[0], size_needed);
    if (!wide.empty() && wide.back() == L'\0')
        wide.pop_back();
    return wide;
#else
    return path;
#endif
}

void VPYInput::parseVpyOptions(const char* optionsCStr)
{
    if (!optionsCStr || !*optionsCStr)
        return;

    std::string options{ optionsCStr };
    options += ';';

    const std::string optSeparator{ ";" };
    const std::string valSeparator{ "=" };
    const std::map<std::string, int> knownOptions
    {
        { "library", 1 },
        { "output", 2 },
        { "requests", 3 },
        { "use-script-sar", 4 }
    };

    size_t start = 0;
    while (true)
    {
        const size_t end = options.find(optSeparator, start);
        if (end == std::string::npos)
            break;

        const std::string option = options.substr(start, end - start);
        const size_t valuePos = option.find(valSeparator);
        if (valuePos != std::string::npos)
        {
            const std::string key = option.substr(0, valuePos);
            const std::string value = option.substr(valuePos + 1);
            const auto it = knownOptions.find(key);
            if (it == knownOptions.end())
            {
                vpy_log(X265_LOG_WARNING, "unknown vpy option \"%s\" ignored\n", option.c_str());
            }
            else
            {
                switch (it->second)
                {
                case 1:
                    vss_library_path = convertLibraryPath(value);
                    vpy_log(X265_LOG_INFO, "using external VapourSynth library from: \"%s\"\n", value.c_str());
                    break;
                case 2:
                    nodeIndex = std::max(0, std::stoi(value));
                    break;
                case 3:
                    parallelRequests = std::max(1, std::stoi(value));
                    break;
                case 4:
                    useScriptSar = (std::stoi(value) != 0);
                    break;
                }
            }
        }
        else if (!option.empty())
        {
            vpy_log(X265_LOG_ERROR, "invalid option \"%s\" ignored\n", option.c_str());
        }

        start = end + optSeparator.length();
    }
}

bool VPYInput::tryLoadLibraryPath(const lib_path_t& path)
{
    if (path.empty())
        return false;

    vss_library_path = path;
    vs_open();
    return vss_library != nullptr;
}

void VPYInput::applyEnvironmentLibraryPath()
{
#if !defined(_WIN32)
    const char* envPath = std::getenv("VSSCRIPT_PATH");
    if (envPath && *envPath)
    {
        vss_library_path = envPath;
        vpy_log(X265_LOG_INFO, "using VSSCRIPT_PATH: \"%s\"\n", envPath);
    }
#endif
}

void VPYInput::load_vs()
{
#if !defined(_WIN32)
    const bool hasExplicitDefaultName =
#ifdef __MACH__
        (vss_library_path == "libvsscript.dylib" || vss_library_path == "libvapoursynth-script.dylib");
#else
        (vss_library_path == "libvsscript.so" || vss_library_path == "libvapoursynth-script.so");
#endif

    if (hasExplicitDefaultName)
        applyEnvironmentLibraryPath();
#endif

    vs_open();

#if !defined(_WIN32)
    if (!vss_library)
    {
#ifdef __MACH__
        if (vss_library_path == "libvsscript.dylib")
        {
            if (!tryLoadLibraryPath("libvapoursynth-script.dylib"))
                tryLoadLibraryPath("/opt/homebrew/lib/libvsscript.dylib");
            if (!vss_library)
                tryLoadLibraryPath("/usr/local/lib/libvsscript.dylib");
            if (!vss_library)
                tryLoadLibraryPath("/opt/homebrew/lib/libvapoursynth-script.dylib");
            if (!vss_library)
                tryLoadLibraryPath("/usr/local/lib/libvapoursynth-script.dylib");
        }
#else
        if (vss_library_path == "libvsscript.so")
        {
            if (!tryLoadLibraryPath("libvapoursynth-script.so"))
                tryLoadLibraryPath("/usr/lib/libvsscript.so");
            if (!vss_library)
                tryLoadLibraryPath("/usr/local/lib/libvsscript.so");
        }
#endif
    }
#endif

    if (!vss_library)
    {
        vpy_log(X265_LOG_ERROR, "failed to load VSScript library\n");
        vpyFailed = true;
        return;
    }

    getVSScriptAPI = reinterpret_cast<vss_api>(vs_address("getVSScriptAPI"));
    if (!getVSScriptAPI)
    {
        vpy_log(X265_LOG_ERROR, "failed to load getVSScriptAPI function. Upgrade VapourSynth to R55 or newer!\n");
        vpyFailed = true;
        return;
    }

    getVSScriptAPILastError = reinterpret_cast<vss_last_error_func>(vs_address("getVSScriptAPILastError"));

    vssapi = getVSScriptAPI(VSSCRIPT_API_VERSION);
    if (!vssapi)
    {
        const char* detail = getVSScriptAPILastError ? getVSScriptAPILastError() : nullptr;
        vpy_log(X265_LOG_ERROR, "failed to initialize VSScript%s%s\n",
            detail ? ": " : "",
            detail ? detail : "");
        vpyFailed = true;
        return;
    }

    vsapi = vssapi->getVSAPI(VAPOURSYNTH_API_VERSION);
    if (!vsapi)
    {
        vpy_log(X265_LOG_ERROR, "failed to get VapourSynth API pointer\n");
        vpyFailed = true;
        return;
    }

}

int VPYInput::clampParallelRequests(int requested, int numThreads, int totalFrames) const
{
    const int safeFrames = std::max(1, totalFrames);
    const int safeThreads = std::max(1, numThreads);
    const int defaultRequests = std::min(safeFrames, safeThreads);

    if (requested <= 0)
        return defaultRequests;

    return std::max(1, std::min(requested, safeFrames));
}

int VPYInput::slotIndex(int frameNumber) const
{
    return frameSlots.empty() ? 0 : (frameNumber % (int)frameSlots.size());
}

VPYInput::FrameSlot* VPYInput::getSlot(int frameNumber)
{
    if (frameSlots.empty())
        return nullptr;
    return &frameSlots[slotIndex(frameNumber)];
}

void VPYInput::resetSlot(FrameSlot& slot, bool closeEvent)
{
    if (slot.frame)
    {
        vsapi->freeFrame(slot.frame);
        slot.frame = nullptr;
    }

    slot.frameNumber = -1;
    slot.ready = false;

    if (closeEvent && slot.event)
    {
        CloseEvent(slot.event);
        slot.event = nullptr;
    }
}

bool VPYInput::createSlotEvents()
{
    for (size_t i = 0; i < frameSlots.size(); ++i)
    {
        frameSlots[i].event = CreateEvent(nullptr, false, false, nullptr);
        if (!frameSlots[i].event)
        {
            vpy_log(X265_LOG_ERROR, "failed to create async event for slot %d\n", (int)i);
            return false;
        }
    }
    return true;
}

void VPYInput::freeSlotFrames()
{
    for (auto& slot : frameSlots)
    {
        if (slot.frame)
        {
            vsapi->freeFrame(slot.frame);
            slot.frame = nullptr;
        }
        slot.ready = false;
        slot.frameNumber = -1;
    }
}

void VPYInput::releaseSlots()
{
    for (auto& slot : frameSlots)
        resetSlot(slot, true);
    frameSlots.clear();
}

void VPYInput::requestFrame(int n)
{
    FrameSlot* slot = getSlot(n);
    if (!slot)
    {
        snprintf(frameError, sizeof(frameError), "internal error: no slot available for frame %d", n);
        vpyFailed = true;
        asyncFailedFrame = (asyncFailedFrame < 0 || n < asyncFailedFrame) ? n : asyncFailedFrame;
        return;
    }

    if (slot->ready || slot->frame)
    {
        snprintf(frameError, sizeof(frameError), "async slot collision before requesting frame %d", n);
        vpy_log(X265_LOG_ERROR, "%s\n", frameError);
        vpyFailed = true;
        asyncFailedFrame = (asyncFailedFrame < 0 || n < asyncFailedFrame) ? n : asyncFailedFrame;
        return;
    }

    slot->frameNumber = n;
    slot->ready = false;
#ifdef _WIN32
    ResetEvent(slot->event);
#endif
    vsapi->getFrameAsync(n, node, frameDoneCallback, this);
    pendingFrames.fetch_add(1);
}

void VPYInput::setAsyncFrame(int n, const VSFrame* f, const char* errorMsg)
{
    FrameSlot* slot = getSlot(n);
    if (!slot)
    {
        if (f)
            vsapi->freeFrame(f);
        pendingFrames.fetch_sub(1);
        return;
    }

    if (!f)
    {
        if (asyncFailedFrame < 0 || n < asyncFailedFrame)
            asyncFailedFrame = n;
        snprintf(frameError, sizeof(frameError), "%s", errorMsg ? errorMsg : "unknown VapourSynth error");
        vpy_log(X265_LOG_ERROR, "async frame request #%d failed: %s\n", n, frameError);
        vpyFailed = true;
    }
    else if (abortAsync)
    {
        vsapi->freeFrame(f);
    }
    else if (slot->frameNumber != n || slot->ready || slot->frame)
    {
        vsapi->freeFrame(f);
        if (asyncFailedFrame < 0 || n < asyncFailedFrame)
            asyncFailedFrame = n;
        snprintf(frameError, sizeof(frameError), "async slot collision at frame %d", n);
        vpy_log(X265_LOG_ERROR, "%s\n", frameError);
        vpyFailed = true;
    }
    else
    {
        slot->frame = f;
        slot->ready = true;
        completedFrames.fetch_add(1);
    }

    pendingFrames.fetch_sub(1);
    if (slot->event)
        SetEvent(slot->event);
}

const VSFrame* VPYInput::getAsyncFrame(int n)
{
    if (asyncFailedFrame >= 0 && asyncFailedFrame <= n)
        return nullptr;

    FrameSlot* slot = getSlot(n);
    if (!slot)
        return nullptr;

    while (!slot->ready && asyncFailedFrame < 0)
        WaitForSingleObject(slot->event, INFINITE);

    if (asyncFailedFrame >= 0 && asyncFailedFrame <= n)
        return nullptr;

    if (slot->frameNumber != n || !slot->frame)
    {
        snprintf(frameError, sizeof(frameError), "frame %d not ready in expected slot", n);
        vpyFailed = true;
        return nullptr;
    }

    const VSFrame* frame = slot->frame;
    slot->frame = nullptr;
    slot->ready = false;
    slot->frameNumber = -1;

    if (!abortAsync && requestedFrames.load() < framesToRequest && asyncFailedFrame < 0)
    {
        const int inFlight = requestedFrames.load() - nextFrame;
        if (inFlight < parallelRequests)
        {
            const int requestIndex = requestedFrames.load();
            requestFrame(requestIndex);
            if (!vpyFailed)
                requestedFrames.store(requestIndex + 1);
        }
    }

    return frame;
}

VPYInput::VPYInput(InputFileInfo& info)
{
    if (info.readerOpts)
        parseVpyOptions(info.readerOpts);

    load_vs();
    if (vpyFailed)
        return;

    if (info.skipFrames > 0)
        nextFrame = info.skipFrames;

    core = vsapi->createCore(0);
    if (!core)
    {
        vpy_log(X265_LOG_ERROR, "failed to create VapourSynth core\n");
        vpyFailed = true;
        return;
    }

    vsapi->addLogHandler(logMessageHandler, nullptr, nullptr, core);
    script = vssapi->createScript(core);
    if (!script)
    {
        vpy_log(X265_LOG_ERROR, "failed to create VapourSynth script\n");
        vpyFailed = true;
        return;
    }

    vssapi->evalSetWorkingDir(script, 1);
    vssapi->evaluateFile(script, info.filename);
    if (vssapi->getError(script))
    {
        vpy_log(X265_LOG_ERROR, "script evaluation failed: %s\n", vssapi->getError(script));
        vpyFailed = true;
        return;
    }

    if (nodeIndex > 0)
        vpy_log(X265_LOG_INFO, "output node changed to %d\n", nodeIndex);

    node = vssapi->getOutputNode(script, nodeIndex);
    if (!node)
    {
        vpy_log(X265_LOG_ERROR, "`%s` does not provide output node %d\n", info.filename, nodeIndex);
        vpyFailed = true;
        return;
    }

    if (vsapi->getNodeType(node) != mtVideo)
    {
        vpy_log(X265_LOG_ERROR, "`%s` at output node %d has no video data\n", info.filename, nodeIndex);
        vpyFailed = true;
        return;
    }

    VSCoreInfo core_info{};
    vsapi->getCoreInfo(vssapi->getCore(script), &core_info);
    vpy_log(X265_LOG_INFO, "VapourSynth Core R%d\n", core_info.core);

    const VSVideoInfo* vi = vsapi->getVideoInfo(node);
    if (!vi)
    {
        vpy_log(X265_LOG_ERROR, "failed to query video info from output node %d\n", nodeIndex);
        vpyFailed = true;
        return;
    }

    if (!vsh::isConstantVideoFormat(vi))
    {
        vpy_log(X265_LOG_ERROR, "only constant video formats are supported\n");
        vpyFailed = true;
        return;
    }

    info.width = vi->width;
    info.height = vi->height;

    char errbuf[512]{};
    const VSFrame* frame0 = vsapi->getFrame(nextFrame, node, errbuf, sizeof(errbuf));
    if (!frame0)
    {
        vpy_log(X265_LOG_ERROR, "%s occurred while getting frame %d\n", errbuf[0] ? errbuf : "unknown error", nextFrame);
        vpyFailed = true;
        return;
    }

    const VSMap* frameProps0 = vsapi->getFramePropertiesRO(frame0);
    info.sarWidth = (useScriptSar && vsapi->mapNumElements(frameProps0, "_SARNum") > 0)
        ? (int)vsapi->mapGetInt(frameProps0, "_SARNum", 0, nullptr) : 0;
    info.sarHeight = (useScriptSar && vsapi->mapNumElements(frameProps0, "_SARDen") > 0)
        ? (int)vsapi->mapGetInt(frameProps0, "_SARDen", 0, nullptr) : 0;

    if (vi->fpsNum == 0 && vi->fpsDen == 0)
    {
        int errDurNum = 0;
        int errDurDen = 0;
        int64_t rateDen = vsapi->mapGetInt(frameProps0, "_DurationNum", 0, &errDurNum);
        int64_t rateNum = vsapi->mapGetInt(frameProps0, "_DurationDen", 0, &errDurDen);

        if (errDurNum || errDurDen)
        {
            vsapi->freeFrame(frame0);
            vpy_log(X265_LOG_ERROR, "VFR: missing FPS values at frame %d\n", nextFrame);
            vpyFailed = true;
            return;
        }

        if (!rateNum)
        {
            vsapi->freeFrame(frame0);
            vpy_log(X265_LOG_ERROR, "VFR: FPS numerator is zero at frame %d\n", nextFrame);
            vpyFailed = true;
            return;
        }

        info.fpsNum = (uint32_t)rateNum;
        info.fpsDenom = (uint32_t)rateDen;
        vpy_log(X265_LOG_INFO, "VideoNode is VFR, but x265 does not support that at the moment. Forcing CFR\n");
    }
    else
    {
        info.fpsNum = vi->fpsNum;
        info.fpsDenom = vi->fpsDen;
    }

    info.frameCount = vi->numFrames;
    info.depth = vi->format.bitsPerSample;
    framesToRequest = info.frameCount;

    if (info.encodeToFrame > 0)
        framesToRequest = std::min(info.frameCount, info.encodeToFrame + nextFrame);

    parallelRequests = clampParallelRequests(parallelRequests, core_info.numThreads, framesToRequest - nextFrame);

    bool cspSupported = false;
    if (vi->format.bitsPerSample >= 8 && vi->format.bitsPerSample <= 16)
    {
        if (vi->format.colorFamily == cfYUV)
        {
            if (vi->format.subSamplingW == 0 && vi->format.subSamplingH == 0)
            {
                info.csp = X265_CSP_I444;
                cspSupported = true;
            }
            else if (vi->format.subSamplingW == 1 && vi->format.subSamplingH == 0)
            {
                info.csp = X265_CSP_I422;
                cspSupported = true;
            }
            else if (vi->format.subSamplingW == 1 && vi->format.subSamplingH == 1)
            {
                info.csp = X265_CSP_I420;
                cspSupported = true;
            }
        }
        else if (vi->format.colorFamily == cfGray)
        {
            info.csp = X265_CSP_I400;
            cspSupported = true;
        }
    }

    if (!cspSupported)
    {
        char format_name[64]{};
        vsapi->getVideoFormatName(&vi->format, format_name);
        vsapi->freeFrame(frame0);
        vpy_log(X265_LOG_ERROR, "video colorspace %s is not supported\n", format_name[0] ? format_name : "<unknown>");
        vpyFailed = true;
        return;
    }

    vsapi->freeFrame(frame0);

    _info = info;
    requestedFrames.store(nextFrame);
    completedFrames.store(nextFrame);
    pendingFrames.store(0);
    asyncFailedFrame = -1;
    abortAsync = false;

    frameSlots.resize((size_t)parallelRequests + 1);
    if (!createSlotEvents())
    {
        vpyFailed = true;
        return;
    }

    isRunning = true;
}

void VPYInput::startReader()
{
    if (vpyFailed || !isRunning)
        return;

    vpy_log(X265_LOG_INFO, "using %d parallel requests\n", parallelRequests);

    const int initialRequests = std::min(parallelRequests, framesToRequest - nextFrame);
    for (int n = 0; n < initialRequests; ++n)
    {
        const int frameNumber = nextFrame + n;
        requestFrame(frameNumber);
        if (vpyFailed)
        {
            isRunning = false;
            return;
        }
        requestedFrames.store(frameNumber + 1);
    }
}

void VPYInput::stopReader()
{
    isRunning = false;
    abortAsync = true;

    while (pendingFrames.load() > 0)
    {
        vpy_log(X265_LOG_INFO, "waiting completion of %d requested frames...\r", pendingFrames.load());
        Sleep(1);
    }

    freeSlotFrames();
}

void VPYInput::release()
{
    isRunning = false;
    abortAsync = true;

    while (pendingFrames.load() > 0)
        Sleep(1);

    freeSlotFrames();
    releaseSlots();

    if (node)
    {
        vsapi->freeNode(node);
        node = nullptr;
    }

    if (script)
    {
        vssapi->freeScript(script);
        script = nullptr;
    }

    vs_close();

    if (frameBuffer)
    {
        x265_free(frameBuffer);
        frameBuffer = nullptr;
    }

    delete this;
}

bool VPYInput::readPicture(x265_picture& pic)
{
    if (nextFrame >= framesToRequest || !isRunning || abortAsync)
        return false;

    const VSFrame* currentFrame = getAsyncFrame(nextFrame);
    if (!currentFrame)
    {
        fprintf(stderr, "%*s\r", 130, " ");
        vpy_log(X265_LOG_ERROR, "error occurred while reading frame %d: %s\n", nextFrame, frameError[0] ? frameError : "unknown VapourSynth error");
        vpyFailed = true;
        abortAsync = true;
        framesToRequest = nextFrame;
        return false;
    }

    pic.width = _info.width;
    pic.height = _info.height;
    pic.colorSpace = _info.csp;
    pic.bitDepth = _info.depth;

    if (frameSize == 0 || frameBuffer == nullptr)
    {
        for (int i = 0; i < x265_cli_csps[_info.csp].planes; ++i)
            frameSize += (size_t)vsapi->getFrameHeight(currentFrame, i) * (size_t)vsapi->getStride(currentFrame, i);

        frameBuffer = reinterpret_cast<uint8_t*>(x265_malloc(frameSize));
        if (!frameBuffer)
        {
            vsapi->freeFrame(currentFrame);
            vpy_log(X265_LOG_ERROR, "failed to allocate %zu bytes for VapourSynth frame buffer\n", frameSize);
            vpyFailed = true;
            abortAsync = true;
            return false;
        }
    }

    pic.framesize = frameSize;

    uint8_t* ptr = frameBuffer;
    for (int i = 0; i < x265_cli_csps[_info.csp].planes; ++i)
    {
        pic.stride[i] = vsapi->getStride(currentFrame, i);
        pic.planes[i] = ptr;
        const size_t len = (size_t)vsapi->getFrameHeight(currentFrame, i) * (size_t)pic.stride[i];
        memcpy(pic.planes[i], vsapi->getReadPtr(currentFrame, i), len);
        ptr += len;
    }

    vsapi->freeFrame(currentFrame);
    ++nextFrame;
    return true;
}
