#pragma once

#include "common.h"
#include "x265.h"

namespace X265_CLI
{
    struct CliLogContext
    {
        const x265_param* param;

        CliLogContext() : param(nullptr) {}
        explicit CliLogContext(const x265_param* p) : param(p) {}

        void log(const char* caller, int level, const char* fmt, ...) const
        {
            if (!param)
                return;

            if (param->logLevel == -1)
                return;

            va_list args;
            va_start(args, fmt);
            X265_NS::general_log(param, caller, level, fmt, args);
            va_end(args);
        }
    };

    void setCliLogContext(const CliLogContext& ctx);
    const CliLogContext& getCliLogContext();
}

#define avs_log(level, fmt, ...)                                             \
    do {                                                                     \
        const X265_CLI::CliLogContext& _ctx = X265_CLI::getCliLogContext();  \
        if (_ctx.param && _ctx.param->logLevel == -1) break;                 \
        X265_NS::general_log(_ctx.param, "avs+", level, fmt, ##__VA_ARGS__); \
    } while (0)

#define vpy_log(level, fmt, ...)                                             \
    do {                                                                     \
        const X265_CLI::CliLogContext& _ctx = X265_CLI::getCliLogContext();  \
        if (_ctx.param && _ctx.param->logLevel == -1) break;                 \
        X265_NS::general_log(_ctx.param, "vpy", level, fmt, ##__VA_ARGS__);  \
    } while (0)
