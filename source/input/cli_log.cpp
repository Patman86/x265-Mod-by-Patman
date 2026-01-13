#include "cli_log.h"

namespace X265_CLI
{
    static CliLogContext g_cliLogCtx;

    void setCliLogContext(const CliLogContext& ctx)
    {
        g_cliLogCtx = ctx;
    }

    const CliLogContext& getCliLogContext()
    {
        return g_cliLogCtx;
    }
}
