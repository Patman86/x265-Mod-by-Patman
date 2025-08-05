/*****************************************************************************
 * Copyright (C) 2020 MulticoreWare, Inc
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


#include "common.h"
#include "primitives.h"
#include "x265.h"
#include "cpu.h"

extern "C" {
#include "fun-decls.h"
}

#if defined(__GNUC__)
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif

namespace X265_NS
{
// private x265 namespace


void setupRVVPrimitives(EncoderPrimitives &p)
{
    // copy_count
    p.cu[BLOCK_4x4].copy_cnt     = PFX(copy_cnt_4_v);
    p.cu[BLOCK_8x8].copy_cnt     = PFX(copy_cnt_8_v);
    p.cu[BLOCK_16x16].copy_cnt   = PFX(copy_cnt_16_v);
    p.cu[BLOCK_32x32].copy_cnt   = PFX(copy_cnt_32_v);

    p.cu[BLOCK_4x4].count_nonzero     = PFX(count_nonzero_4_v);
    p.cu[BLOCK_8x8].count_nonzero     = PFX(count_nonzero_8_v);
    p.cu[BLOCK_16x16].count_nonzero   = PFX(count_nonzero_16_v);
    p.cu[BLOCK_32x32].count_nonzero   = PFX(count_nonzero_32_v);
}

void setupAssemblyPrimitives(EncoderPrimitives &p, int cpuMask)
{
    if (cpuMask & X265_CPU_RVV)
    {
        setupRVVPrimitives(p);
    }
}

void setupIntrinsicPrimitives(EncoderPrimitives &p, int cpuMask)
{
    (void)p;
    if (cpuMask & X265_CPU_RVV)
    {
    }
}

} // namespace X265_NS
