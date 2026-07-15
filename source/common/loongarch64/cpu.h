/*****************************************************************************
 * Copyright (C) 2026 MulticoreWare, Inc
 *
 * Authors: PengXu <pengxu@loongson.cn>
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

#ifndef X265_COMMON_LOONGARCH64_CPU_H
#define X265_COMMON_LOONGARCH64_CPU_H

#include "x265.h"

#if LOONGARCH64_RUNTIME_CPU_DETECT

#if HAVE_GETAUXVAL

#include <sys/auxv.h>

#define LA_HWCAP_LSX    ( 1U << 4 )
#define LA_HWCAP_LASX   ( 1U << 5 )

static inline uint32_t loongarch64_get_cpu_flags()
{
    uint32_t flags = 0;
#if HAVE_LSX || HAVE_LSX
    unsigned long hwcap = x265_getauxval(AT_HWCAP);
#endif

#if HAVE_LSX
    if( hwcap & LA_HWCAP_LSX )
        flags |= X265_CPU_LSX;
#endif
#if HAVE_LASX
    if( hwcap & LA_HWCAP_LASX )
        flags |= X265_CPU_LASX;
#endif

    return flags;
}

#else // HAVE_GETAUXVAL
#error                                                                 \
    "Run-time CPU feature detection selected, but no detection method" \
    "available for your platform. Rerun cmake configure with"          \
    "-DLOONGARCH64_RUNTIME_CPU_DETECT=OFF."
#endif // HAVE_GETAUXVAL

static inline uint32_t loongarch64_cpu_detect()
{
    uint32_t flags = loongarch64_get_cpu_flags();

    return flags;
}

#else // if LOONGARCH64_RUNTIME_CPU_DETECT

static inline uint32_t loongarch64_cpu_detect()
{
    uint32_t flags = 0;

#if HAVE_LSX
    flags |= X265_CPU_LSX;
#endif
#if HAVE_LASX
    flags |= X265_CPU_LASX;
#endif
    return flags;
}

#endif // if LOONGARCH64_RUNTIME_CPU_DETECT

#endif // ifndef X265_COMMON_LOONGARCH64_CPU_H
