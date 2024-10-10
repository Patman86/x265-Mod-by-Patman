/*****************************************************************************
 * Copyright (C) 2024 MulticoreWare, Inc
 *
 * Authors: Hari Limaye <hari.limaye@arm.com>
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

#ifndef X265_COMMON_AARCH64_CPU_H
#define X265_COMMON_AARCH64_CPU_H

#include "x265.h"

#if AARCH64_RUNTIME_CPU_DETECT

#if defined(__linux__)

#include <sys/auxv.h>

#define X265_AARCH64_HWCAP_ASIMDDP (1 << 20)
#define X265_AARCH64_HWCAP_SVE (1 << 22)
#define X265_AARCH64_HWCAP2_SVE2 (1 << 1)
#define X265_AARCH64_HWCAP2_I8MM (1 << 13)

static inline int aarch64_get_cpu_flags()
{
    int flags = 0;

#if HAVE_NEON_DOTPROD || HAVE_SVE
    unsigned long hwcap = getauxval(AT_HWCAP);
#endif
#if HAVE_NEON_I8MM || HAVE_SVE2
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
#endif

#if HAVE_NEON
    flags |= X265_CPU_NEON;
#endif
#if HAVE_NEON_DOTPROD
    if (hwcap & X265_AARCH64_HWCAP_ASIMDDP) flags |= X265_CPU_NEON_DOTPROD;
#endif
#if HAVE_NEON_I8MM
    if (hwcap2 & X265_AARCH64_HWCAP2_I8MM) flags |= X265_CPU_NEON_I8MM;
#endif
#if HAVE_SVE
    if (hwcap & X265_AARCH64_HWCAP_SVE) flags |= X265_CPU_SVE;
#endif
#if HAVE_SVE2
    if (hwcap2 & X265_AARCH64_HWCAP2_SVE2) flags |= X265_CPU_SVE2;
#endif

    return flags;
}

#else // defined(__linux__)
#error                                                                 \
    "Run-time CPU feature detection selected, but no detection method" \
    "available for your platform. Rerun cmake configure with"          \
    "-DAARCH64_RUNTIME_CPU_DETECT=OFF."
#endif // defined(__linux__)

static inline int aarch64_cpu_detect()
{
    int flags = aarch64_get_cpu_flags();

    // Restrict flags: FEAT_I8MM assumes that FEAT_DotProd is available.
    if (!(flags & X265_CPU_NEON_DOTPROD)) flags &= ~X265_CPU_NEON_I8MM;

    // Restrict flags: SVE assumes that FEAT_{DotProd,I8MM} are available.
    if (!(flags & X265_CPU_NEON_DOTPROD)) flags &= ~X265_CPU_SVE;
    if (!(flags & X265_CPU_NEON_I8MM)) flags &= ~X265_CPU_SVE;

    // Restrict flags: SVE2 assumes that FEAT_SVE is available.
    if (!(flags & X265_CPU_SVE)) flags &= ~X265_CPU_SVE2;

    return flags;
}

#else // if AARCH64_RUNTIME_CPU_DETECT

static inline int aarch64_cpu_detect()
{
    int flags = 0;

#if HAVE_NEON
    flags |= X265_CPU_NEON;
#endif
#if HAVE_NEON_DOTPROD
    flags |= X265_CPU_NEON_DOTPROD;
#endif
#if HAVE_NEON_I8MM
    flags |= X265_CPU_NEON_I8MM;
#endif
#if HAVE_SVE
    flags |= X265_CPU_SVE;
#endif
#if HAVE_SVE2
    flags |= X265_CPU_SVE2;
#endif
    return flags;
}

#endif // if AARCH64_RUNTIME_CPU_DETECT

#endif // ifndef X265_COMMON_AARCH64_CPU_H
