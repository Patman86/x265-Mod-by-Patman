/*****************************************************************************
 * MIT License
 *
 * Copyright (c) 2022 Xinyue Lu
 *
 * Authors: Xinyue Lu <i@7086.in>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *****************************************************************************
 * The MIT License applies to this file only.
 *****************************************************************************/

#pragma once

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <utility>

static constexpr int TIME_WAIT = 30;

inline void sleep_seconds(int seconds)
{
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

enum class GOP_ISOM_MATRIX_INDEX {
    UNSPECIFIED = 2
};

class GOPEngine
{
private:
    // Input
    const char*     inputFilename;
    const InputFileInfo inputInfo;

    // Input Args
    std::string     gopFilename;
    std::string     dirPrefix;
    std::string     filenamePrefix;
    int             frameOffset = 0;

    // Internal State
    int             currentFrame = 0;
    FILE*           gopFP = nullptr;
    FILE*           dataFP = nullptr;

public:
    bool fail = false;

    GOPEngine(const char* fname, InputFileInfo& info)
        : inputFilename(fname)
        , inputInfo(info)
    {
        ParseInputFilename();
        ParsePrefix();
        gopFP = OpenFileForWrite(dirPrefix + gopFilename, false);
    }

    void SetParam(x265_param* p_param)
    {
        p_param->bAnnexB        = false;
        p_param->bRepeatHeaders = false;

        std::fprintf(gopFP, "#options %s.options\n", filenamePrefix.c_str());
        ProduceOptFile(p_param);
    }

    int WriteHeaders(const x265_nal* p_nal, uint32_t nalcount)
    {
        currentFrame = 0;
        int totalWrites = 0;

        if (nalcount < 3)
            LogError("Too few headers, expect 3+, actual %d", nalcount);

        FILE* hdr_file = OpenFileForWrite(dirPrefix + filenamePrefix + ".headers", false);
        if (!hdr_file)
            return -1;

        std::fprintf(gopFP, "#headers %s.headers\n", filenamePrefix.c_str());

        for (uint32_t i = 0; i < nalcount; i++) {
            SmartWrite(p_nal[i].payload, p_nal[i].sizeBytes, hdr_file);
            totalWrites += static_cast<int>(p_nal[i].sizeBytes);
        }

        std::fclose(hdr_file);
        return totalWrites;
    }

    int WriteFrame(const x265_nal* p_nalu, uint32_t nalcount, x265_picture& pic)
    {
        const bool is_keyframe = pic.sliceType == X265_TYPE_IDR;
        int totalWrites = 0;

        if (is_keyframe) {
            if (dataFP) {
                std::fclose(dataFP);
                dataFP = nullptr;
            }

            std::stringstream ss;
            ss << filenamePrefix << "-" << std::setfill('0') << std::setw(6)
               << (currentFrame + frameOffset) << ".hevc-gop-data";
            std::string data_filename = ss.str();

            dataFP = OpenFileForWrite(dirPrefix + data_filename, currentFrame > 0);
            if (!dataFP)
                return -1;

            std::fprintf(gopFP, "%s\n", data_filename.c_str());
            std::fflush(gopFP);
        }

        constexpr int8_t ts_len  = 2 * static_cast<int8_t>(sizeof(int64_t));
        constexpr int8_t ts_lenx[4] = { 0, 0, 0, ts_len };

        SmartWrite(&ts_lenx, sizeof(ts_lenx), dataFP);
        SmartWrite(&pic.pts, sizeof(int64_t), dataFP);
        SmartWrite(&pic.dts, sizeof(int64_t), dataFP);

        for (uint8_t i = 0; i < nalcount; i++) {
            SmartWrite(p_nalu[i].payload, p_nalu[i].sizeBytes, dataFP);
            totalWrites += static_cast<int>(p_nalu[i].sizeBytes);
        }

        currentFrame++;
        return totalWrites;
    }

    void Release()
    {
        if (dataFP)
            std::fclose(dataFP);

        if (gopFP) {
            std::fprintf(gopFP, "# %d frames written, last frame %d\n",
                         currentFrame, currentFrame + frameOffset);
            std::fclose(gopFP);
        }
    }

private:
    void ParseInputFilename()
    {
        std::string input(inputFilename);

        // split "?"
        auto sz = input.find_first_of('?');
        if (sz == std::string::npos) {
            gopFilename = input;
        } else {
            gopFilename = input.substr(0, sz);
            std::string args = input.substr(sz + 1);
            ParseInputArgs(args);
        }
    }

    void ParseInputArgs(std::string args)
    {
        std::stringstream ss(args);
        std::string arg;
        std::string key;
        std::string value;

        while (std::getline(ss, arg, '&')) {
            auto sz = arg.find_first_of('=');
            if (sz == std::string::npos) {
                key   = arg;   // ganze Zeichenfolge ist der Key
                value = "1";
            } else {
                key   = arg.substr(0, sz);
                value = arg.substr(sz + 1);
            }

            if (key == "start")
                frameOffset = std::stoi(value);
        }
    }

    void ParsePrefix()
    {
        std::size_t pos;

        if ((pos = gopFilename.find_last_of("/\\")) != std::string::npos) {
            dirPrefix   = gopFilename.substr(0, pos + 1);
            gopFilename = gopFilename.substr(pos + 1);
        }

        if ((pos = gopFilename.find_last_of('.')) != std::string::npos)
            filenamePrefix = gopFilename.substr(0, pos);
        else
            filenamePrefix = gopFilename;
    }

    void ProduceOptFile(x265_param* p_param)
    {
        FILE* optFP = OpenFileForWrite(dirPrefix + filenamePrefix + ".options", false);
        if (!optFP)
            return;

        std::fprintf(optFP, "b-frames %d\n",          p_param->bframes);
        std::fprintf(optFP, "b-pyramid %d\n",         p_param->bBPyramid);
        std::fprintf(optFP, "input-timebase-num %d\n",  inputInfo.timebaseNum);
        std::fprintf(optFP, "input-timebase-den %d\n",  inputInfo.timebaseDenom);
        std::fprintf(optFP, "output-fps-num %u\n",    p_param->fpsNum);
        std::fprintf(optFP, "output-fps-den %u\n",    p_param->fpsDenom);
        std::fprintf(optFP, "source-width %d\n",      p_param->sourceWidth);
        std::fprintf(optFP, "source-height %d\n",     p_param->sourceHeight);
        std::fprintf(optFP, "sar-width %d\n",         p_param->vui.sarWidth);
        std::fprintf(optFP, "sar-height %d\n",        p_param->vui.sarHeight);
        std::fprintf(optFP, "primaries-index %d\n",   p_param->vui.colorPrimaries);
        std::fprintf(optFP, "transfer-index %d\n",    p_param->vui.transferCharacteristics);
        std::fprintf(optFP, "matrix-index %d\n",
                     p_param->vui.matrixCoeffs >= 0
                         ? p_param->vui.matrixCoeffs
                         : static_cast<int>(GOP_ISOM_MATRIX_INDEX::UNSPECIFIED));
        std::fprintf(optFP, "full-range %d\n",
                     p_param->vui.bEnableVideoFullRangeFlag >= 0
                         ? p_param->vui.bEnableVideoFullRangeFlag
                         : 0);

        std::fclose(optFP);
    }

    FILE* OpenFileForWrite(const std::string& fname, bool retry)
    {
        while (true) {
            FILE* fp = x265_fopen(fname.c_str(), "wb");
            if (fp != nullptr)
                return fp;

            if (!retry)
                break;

            // Retrying
            LogWarning("unable to open file %s for writing, error %d %s, retrying in %d seconds.\n",
                       fname.c_str(), errno, std::strerror(errno), TIME_WAIT);
            sleep_seconds(TIME_WAIT);
        }

        // Failed
        fail = true;
        LogError("unable to open file %s for writing, error %d %s.\n",
                 fname.c_str(), errno, std::strerror(errno));
        return nullptr;
    }

    void SmartWrite(const void* data, std::size_t size, FILE* file)
    {
        long data_pos = std::ftell(file);
        if (data_pos < 0)
            data_pos = 0;

        while (true) {
            std::size_t written = std::fwrite(data, 1, size, file);
            if (written == size)
                break;

            // ENOSPC
            LogWarning("unable to write, error %d %s, retrying in %d seconds.\n",
                       errno, std::strerror(errno), TIME_WAIT);
            std::fseek(file, data_pos, SEEK_SET);
            sleep_seconds(TIME_WAIT);
        }
    }

    template <typename... Params>
    void LogWarning(const char* fmt, Params&&... params)
    {
        general_log(nullptr, "gop+", X265_LOG_WARNING, fmt, std::forward<Params>(params)...);
    }

    template <typename... Params>
    void LogError(const char* fmt, Params&&... params)
    {
        general_log(nullptr, "gop+", X265_LOG_ERROR, fmt, std::forward<Params>(params)...);
    }
};