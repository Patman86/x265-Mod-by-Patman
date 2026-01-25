/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Changsheng Wu <wu.changsheng@sanechips.com.cn>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at license @ x265.com.
 *****************************************************************************/

#ifndef X265_COMMON_RISCV64_CPU_H
#define X265_COMMON_RISCV64_CPU_H

#include "x265.h"

#if RISCV64_RUNTIME_CPU_DETECT

#if defined(__linux__)

#include <sys/auxv.h>

#define HWCAP_RISCV64_RVV     (1 << ('V' - 'A'))

static int parse_proc_cpuinfo(const char *flag) {
    FILE *file = fopen("/proc/cpuinfo", "r");
    if (file == NULL)
        return 0;

    char line[1024];
    int found = 0;

    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, flag) != NULL) {
            found = 1;
            break;
        }
    }

    fclose(file);
    return found;
}

static inline uint32_t riscv64_cpu_detect()
{
    uint32_t flags = 0;

    unsigned long hwcap = getauxval(AT_HWCAP);

    if (hwcap & HWCAP_RISCV64_RVV) {
        flags |= X265_CPU_RVV;

        if (parse_proc_cpuinfo("zbb"))
            flags |= X265_CPU_ZBB;
    }

    return flags;
}

#else // defined(__linux__)
#error                                                                 \
    "Run-time CPU feature detection selected, but no detection method" \
    "available for your platform. Rerun cmake configure with"          \
    "-DRISCV64_RUNTIME_CPU_DETECT=OFF."
#endif // defined(__linux__)

#else // if AARCH64_RUNTIME_CPU_DETECT

static inline uint32_t riscv64_cpu_detect()
{
    uint32_t flags = 0;

#if HAVE_RVV
    flags |= X265_CPU_RVV;
    flags |= X265_CPU_ZBB;
#endif
    return flags;
}

#endif // if RISCV64_RUNTIME_CPU_DETECT

#endif // ifndef X265_COMMON_RISCV64_CPU_H
