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

    p.idst4x4               = PFX(idst4_v);
    p.cu[BLOCK_4x4].idct    = PFX(idct4_v);
    p.cu[BLOCK_8x8].idct    = PFX(idct8_v);
    p.cu[BLOCK_16x16].idct  = PFX(idct16_v);
    p.cu[BLOCK_32x32].idct  = PFX(idct32_v);
    p.denoiseDct            = PFX(denoiseDct_v);

    p.dst4x4                = PFX(dst4_v);
    p.cu[BLOCK_4x4].dct     = PFX(dct4_v);
    p.cu[BLOCK_8x8].dct     = PFX(dct8_v);
    p.cu[BLOCK_16x16].dct   = PFX(dct16_v);
    p.cu[BLOCK_32x32].dct   = PFX(dct32_v);

    p.cu[BLOCK_4x4].nonPsyRdoQuant      = PFX(nonPsyRdoQuant2_v);
    p.cu[BLOCK_8x8].nonPsyRdoQuant      = PFX(nonPsyRdoQuant3_v);
    p.cu[BLOCK_16x16].nonPsyRdoQuant    = PFX(nonPsyRdoQuant4_v);
    p.cu[BLOCK_32x32].nonPsyRdoQuant    = PFX(nonPsyRdoQuant5_v);
    p.cu[BLOCK_4x4].psyRdoQuant         = PFX(PsyRdoQuant2_v);
    p.cu[BLOCK_8x8].psyRdoQuant         = PFX(PsyRdoQuant3_v);
    p.cu[BLOCK_16x16].psyRdoQuant       = PFX(PsyRdoQuant4_v);
    p.cu[BLOCK_32x32].psyRdoQuant       = PFX(PsyRdoQuant5_v);
    p.cu[BLOCK_4x4].psyRdoQuant_1p      = PFX(nonPsyRdoQuant2_v);
    p.cu[BLOCK_8x8].psyRdoQuant_1p      = PFX(nonPsyRdoQuant3_v);
    p.cu[BLOCK_16x16].psyRdoQuant_1p    = PFX(nonPsyRdoQuant4_v);
    p.cu[BLOCK_32x32].psyRdoQuant_1p    = PFX(nonPsyRdoQuant5_v);
    p.cu[BLOCK_4x4].psyRdoQuant_2p      = PFX(PsyRdoQuant2_v);
    p.cu[BLOCK_8x8].psyRdoQuant_2p      = PFX(PsyRdoQuant3_v);
    p.cu[BLOCK_16x16].psyRdoQuant_2p    = PFX(PsyRdoQuant4_v);
    p.cu[BLOCK_32x32].psyRdoQuant_2p    = PFX(PsyRdoQuant5_v);
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
