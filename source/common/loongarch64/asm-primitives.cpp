/*****************************************************************************
 * Copyright (C) 2024 MulticoreWare, Inc
 *
 * Authors: Hao Chen <chenhao@loongson.cn>
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

#include "common.h"
#include "primitives.h"
#include "x265.h"
#include "cpu.h"
#include "loopfilter.h"
#include "ipfilter8.h"
#include "dct.h"
#include "pixel.h"
#include "intrapred.h"
#include "blockcopy8.h"

#define ASSIGN2(func, fname) \
    func[ALIGNED] = PFX(fname); \
    func[NONALIGNED] = PFX(fname)

namespace X265_NS {
// private x265 namespace

template<int size>
int psyCost_pp_lsx(const pixel *source, intptr_t sstride, const pixel *recon, intptr_t rstride)
{
    static pixel zeroBuf[8] /* = { 0 } */;

    if (size)
    {
        int dim = 1 << (size + 2);
        uint32_t totEnergy = 0;
        for (int i = 0; i < dim; i += 8)
        {
            for (int j = 0; j < dim; j += 8)
            {
                /* AC energy, measured by sa8d (AC + DC) minus SAD (DC) */
                int sourceEnergy = PFX(pixel_sa8d_8x8_lsx)(source + i * sstride + j, sstride, zeroBuf, 0) -
                                   (PFX(pixel_sad_8x8_lsx)(source + i * sstride + j, sstride, zeroBuf, 0) >> 2);
                int reconEnergy =  PFX(pixel_sa8d_8x8_lsx)(recon + i * rstride + j, rstride, zeroBuf, 0) -
                                   (PFX(pixel_sad_8x8_lsx)(recon + i * rstride + j, rstride, zeroBuf, 0) >> 2);

                totEnergy += abs(sourceEnergy - reconEnergy);
            }
        }
        return totEnergy;
    }
    else
    {
        int sourceEnergy = PFX(pixel_satd_4x4_lsx)(source, sstride, zeroBuf, 0) - (PFX(pixel_sad_4x4_lsx)(source, sstride, zeroBuf, 0) >> 2);
        int reconEnergy = PFX(pixel_satd_4x4_lsx)(recon, rstride, zeroBuf, 0) - (PFX(pixel_sad_4x4_lsx)(recon, rstride, zeroBuf, 0) >> 2);
        return abs(sourceEnergy - reconEnergy);
    }
}

template<int size>
int psyCost_pp_lasx(const pixel *source, intptr_t sstride, const pixel *recon, intptr_t rstride)
{
    static pixel zeroBuf[8] /* = { 0 } */;

    if (size)
    {
        int dim = 1 << (size + 2);
        uint32_t totEnergy = 0;
        for (int i = 0; i < dim; i += 8)
        {
            for (int j = 0; j < dim; j += 8)
            {
                /* AC energy, measured by sa8d (AC + DC) minus SAD (DC) */
                int sourceEnergy = PFX(pixel_sa8d_8x8_lasx)(source + i * sstride + j, sstride, zeroBuf, 0) -
                                   (PFX(pixel_sad_8x8_lsx)(source + i * sstride + j, sstride, zeroBuf, 0) >> 2);
                int reconEnergy =  PFX(pixel_sa8d_8x8_lasx)(recon + i * rstride + j, rstride, zeroBuf, 0) -
                                   (PFX(pixel_sad_8x8_lsx)(recon + i * rstride + j, rstride, zeroBuf, 0) >> 2);

                totEnergy += abs(sourceEnergy - reconEnergy);
            }
        }
        return totEnergy;
    }
    else
    {
        int sourceEnergy = PFX(pixel_satd_4x4_lsx)(source, sstride, zeroBuf, 0) - (PFX(pixel_sad_4x4_lsx)(source, sstride, zeroBuf, 0) >> 2);
        int reconEnergy = PFX(pixel_satd_4x4_lsx)(recon, rstride, zeroBuf, 0) - (PFX(pixel_sad_4x4_lsx)(recon, rstride, zeroBuf, 0) >> 2);
        return abs(sourceEnergy - reconEnergy);
    }
}

typedef void (*intra_pred_ang_lsx)(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);

intra_pred_ang_lsx PFX(intra_pred_ang8_19_lsx) = PFX(intra_pred_ang8_17_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_20_lsx) = PFX(intra_pred_ang8_16_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_21_lsx) = PFX(intra_pred_ang8_15_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_22_lsx) = PFX(intra_pred_ang8_14_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_23_lsx) = PFX(intra_pred_ang8_13_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_24_lsx) = PFX(intra_pred_ang8_12_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_25_lsx) = PFX(intra_pred_ang8_11_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_27_lsx) = PFX(intra_pred_ang8_9_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_28_lsx) = PFX(intra_pred_ang8_8_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_29_lsx) = PFX(intra_pred_ang8_7_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_30_lsx) = PFX(intra_pred_ang8_6_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_31_lsx) = PFX(intra_pred_ang8_5_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_32_lsx) = PFX(intra_pred_ang8_4_lsx);
intra_pred_ang_lsx PFX(intra_pred_ang8_33_lsx) = PFX(intra_pred_ang8_3_lsx);

intra_pred_ang_lsx PFX(intra_pred_ang16_34_lsx) = PFX(intra_pred_ang16_2_lsx);

#define intra_pred_lsx(number)                                                 \
intra_pred_ang_lsx intra_pred##number##_lsx[35] __attribute__((aligned(8)))= { \
        NULL,                                                                  \
        NULL,                                                                  \
        PFX(intra_pred_ang##number##_2_lsx),                                   \
        PFX(intra_pred_ang##number##_3_lsx),                                   \
        PFX(intra_pred_ang##number##_4_lsx),                                   \
        PFX(intra_pred_ang##number##_5_lsx),                                   \
        PFX(intra_pred_ang##number##_6_lsx),                                   \
        PFX(intra_pred_ang##number##_7_lsx),                                   \
        PFX(intra_pred_ang##number##_8_lsx),                                   \
        PFX(intra_pred_ang##number##_9_lsx),                                   \
        PFX(intra_pred_ang##number##_10_lsx),                                  \
        PFX(intra_pred_ang##number##_11_lsx),                                  \
        PFX(intra_pred_ang##number##_12_lsx),                                  \
        PFX(intra_pred_ang##number##_13_lsx),                                  \
        PFX(intra_pred_ang##number##_14_lsx),                                  \
        PFX(intra_pred_ang##number##_15_lsx),                                  \
        PFX(intra_pred_ang##number##_16_lsx),                                  \
        PFX(intra_pred_ang##number##_17_lsx),                                  \
        PFX(intra_pred_ang##number##_18_lsx),                                  \
        PFX(intra_pred_ang##number##_19_lsx),                                  \
        PFX(intra_pred_ang##number##_20_lsx),                                  \
        PFX(intra_pred_ang##number##_21_lsx),                                  \
        PFX(intra_pred_ang##number##_22_lsx),                                  \
        PFX(intra_pred_ang##number##_23_lsx),                                  \
        PFX(intra_pred_ang##number##_24_lsx),                                  \
        PFX(intra_pred_ang##number##_25_lsx),                                  \
        PFX(intra_pred_ang##number##_26_lsx),                                  \
        PFX(intra_pred_ang##number##_27_lsx),                                  \
        PFX(intra_pred_ang##number##_28_lsx),                                  \
        PFX(intra_pred_ang##number##_29_lsx),                                  \
        PFX(intra_pred_ang##number##_30_lsx),                                  \
        PFX(intra_pred_ang##number##_31_lsx),                                  \
        PFX(intra_pred_ang##number##_32_lsx),                                  \
        PFX(intra_pred_ang##number##_33_lsx),                                  \
        PFX(intra_pred_ang##number##_34_lsx)                                   \
};

intra_pred_lsx(4);
intra_pred_lsx(8);
intra_pred_lsx(16);
intra_pred_lsx(32);

#define all_angs_pred_lsx(number1, number2)                                     \
static void all_angs_pred##number1##_lsx(pixel *dest, pixel *refPix,            \
                                        pixel *filtPix, int bLuma)              \
{                                                                               \
    const int size = number1;                                                   \
    for (int mode = 2; mode <= 34; mode++)                                      \
    {                                                                           \
        pixel *srcPix  = (g_intraFilterFlags[mode] & size ? filtPix  : refPix); \
        pixel *out = dest + ((mode - 2) << (number2));                          \
                                                                                \
        intra_pred##number1##_lsx[mode](out, size, srcPix, mode, bLuma);        \
                                                                                \
        bool modeHor = (mode < 18);                                             \
        if (modeHor)                                                            \
        {                                                                       \
            if (size == 8)                                                      \
            {                                                                   \
                x265_transpose_8x8_lsx(out, out, size);                         \
            }                                                                   \
            else if (size == 16)                                                \
            {                                                                   \
                x265_transpose_16x16_lsx(out, out, size);                       \
            }                                                                   \
            else if (size == 32)                                                \
            {                                                                   \
                x265_transpose_32x32_lsx(out, out, size);                       \
            }                                                                   \
            else                                                                \
            {                                                                   \
                x265_transpose_4x4_lsx(out, out, size);                         \
            }                                                                   \
        }                                                                       \
    }                                                                           \
}

all_angs_pred_lsx(4, 4)
all_angs_pred_lsx(8, 6)
all_angs_pred_lsx(16, 8)
all_angs_pred_lsx(32, 10)

#define LUMA(W, H, CPU) \
    p.pu[LUMA_ ## W ## x ## H].luma_vpp     = PFX(interp_8tap_vert_pp_ ## W ## x ## H ## _ ##CPU); \
    p.pu[LUMA_ ## W ## x ## H].luma_vps     = PFX(interp_8tap_vert_ps_ ## W ## x ## H ## _ ##CPU); \
    p.pu[LUMA_ ## W ## x ## H].luma_vsp     = PFX(interp_8tap_vert_sp_ ## W ## x ## H ## _ ##CPU); \
    p.pu[LUMA_ ## W ## x ## H].luma_vss     = PFX(interp_8tap_vert_ss_ ## W ## x ## H ## _ ##CPU); \
    p.pu[LUMA_ ## W ## x ## H].luma_hpp     = PFX(interp_8tap_horiz_pp_ ## W ## x ## H ## _ ##CPU); \
    p.pu[LUMA_ ## W ## x ## H].luma_hps     = PFX(interp_8tap_horiz_ps_ ## W ## x ## H ## _ ##CPU); \
    p.pu[LUMA_ ## W ## x ## H].luma_hvpp     = PFX(interp_8tap_hv_pp_ ## W ## x ## H ## _ ##CPU); \
    p.pu[LUMA_ ## W ## x ## H].convert_p2s[NONALIGNED] = PFX(filterPixelToShort_ ## W ## x ## H ## _ ##CPU); \
    p.pu[LUMA_ ## W ## x ## H].convert_p2s[ALIGNED] = PFX(filterPixelToShort_ ## W ## x ## H ## _ ##CPU);

#define CHROMA_420(W, H, CPU) \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_hpp = PFX(interp_4tap_horiz_pp_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_hps = PFX(interp_4tap_horiz_ps_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_vpp = PFX(interp_4tap_vert_pp_ ## W ## x ## H ## _ ##CPU);  \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_vps = PFX(interp_4tap_vert_ps_ ## W ## x ## H ## _ ##CPU);  \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_vsp = PFX(interp_4tap_vert_sp_ ## W ## x ## H ## _ ##CPU);  \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_vss = PFX(interp_4tap_vert_ss_ ## W ## x ## H ## _ ##CPU);  \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].p2s[NONALIGNED] = PFX(filterPixelToShort_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].p2s[ALIGNED] = PFX(filterPixelToShort_ ## W ## x ## H ## _ ##CPU);

#define CHROMA_422(W, H, CPU) \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_hpp = PFX(interp_4tap_horiz_pp_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_hps = PFX(interp_4tap_horiz_ps_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_vpp = PFX(interp_4tap_vert_pp_ ## W ## x ## H ## _ ##CPU);  \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_vps = PFX(interp_4tap_vert_ps_ ## W ## x ## H ## _ ##CPU);  \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_vsp = PFX(interp_4tap_vert_sp_ ## W ## x ## H ## _ ##CPU);  \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_vss = PFX(interp_4tap_vert_ss_ ## W ## x ## H ## _ ##CPU);  \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].p2s[NONALIGNED] = PFX(filterPixelToShort_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].p2s[ALIGNED] = PFX(filterPixelToShort_ ## W ## x ## H ## _ ##CPU);

#define CHROMA_444(W, H, CPU) \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_hpp = PFX(interp_4tap_horiz_pp_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_hps = PFX(interp_4tap_horiz_ps_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_vpp = PFX(interp_4tap_vert_pp_ ## W ## x ## H ## _ ##CPU);  \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_vps = PFX(interp_4tap_vert_ps_ ## W ## x ## H ## _ ##CPU);  \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_vsp = PFX(interp_4tap_vert_sp_ ## W ## x ## H ## _ ##CPU);  \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_vss = PFX(interp_4tap_vert_ss_ ## W ## x ## H ## _ ##CPU);  \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].p2s[NONALIGNED] = PFX(filterPixelToShort_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].p2s[ALIGNED] = PFX(filterPixelToShort_ ## W ## x ## H ## _ ##CPU);

#define LUMA_CU(W, H, CPU) \
    p.cu[BLOCK_ ## W ## x ## H].psy_cost_pp = psyCost_pp_ ##CPU<BLOCK_ ## W ## x ## H>; \
    p.cu[BLOCK_ ## W ## x ## H].sse_pp      = PFX(pixel_sse_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].sse_ss      = PFX(pixel_sse_ss_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].var         = PFX(pixel_var_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].ssd_s[NONALIGNED] = PFX(pixel_ssd_s_## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].ssd_s[ALIGNED] = PFX(pixel_ssd_s_## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].copy_ss     = PFX(blockcopy_ss_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].copy_sp     = PFX(blockcopy_sp_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].copy_ps     = PFX(blockcopy_ps_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].cpy2Dto1D_shl = PFX(cpy2Dto1D_shl_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].cpy2Dto1D_shr = PFX(cpy2Dto1D_shr_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].cpy1Dto2D_shl[NONALIGNED] = PFX(cpy1Dto2D_shl_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].cpy1Dto2D_shl[ALIGNED] = PFX(cpy1Dto2D_shl_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].cpy1Dto2D_shr = PFX(cpy1Dto2D_shr_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].transpose     = PFX(transpose_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].calcresidual[NONALIGNED] = PFX(pixel_getResidual_ ## W ## x ## H ##_ ## CPU); \
    p.cu[BLOCK_ ## W ## x ## H].calcresidual[ALIGNED]    = PFX(pixel_getResidual_ ## W ## x ## H ##_ ## CPU);;

#define LUMA_PU(W, H, CPU) \
    p.pu[LUMA_ ## W ## x ## H].copy_pp = PFX(blockcopy_pp_ ## W ## x ## H ## _ ##CPU); \
    p.pu[LUMA_ ## W ## x ## H].addAvg[NONALIGNED] = PFX(addAvg_ ## W ## x ## H ## _ ##CPU); \
    p.pu[LUMA_ ## W ## x ## H].addAvg[ALIGNED] = PFX(addAvg_ ## W ## x ## H ## _ ##CPU);

#define CHROMA_PU_420(W, H, CPU) \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].copy_pp = PFX(blockcopy_pp_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].addAvg[NONALIGNED]  = PFX(addAvg_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].addAvg[ALIGNED]  = PFX(addAvg_ ## W ## x ## H ## _ ##CPU);

#define CHROMA_PU_422(W, H, CPU) \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].copy_pp = PFX(blockcopy_pp_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].addAvg[NONALIGNED]  = PFX(addAvg_ ## W ## x ## H ## _ ##CPU); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].addAvg[ALIGNED]  = PFX(addAvg_ ## W ## x ## H ## _ ##CPU);

#define CHROMA_CU_420(W, H, CPU) \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_ss = PFX(blockcopy_ss_ ## W ## x ## H ##_ ## CPU); \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_sp = PFX(blockcopy_sp_ ## W ## x ## H ##_ ## CPU);

#define CHROMA_CU_422(W, H, CPU) \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_ss = PFX(blockcopy_ss_ ## W ## x ## H ##_ ## CPU); \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_sp = PFX(blockcopy_sp_ ## W ## x ## H ##_ ## CPU); \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_ps = PFX(blockcopy_ps_ ## W ## x ## H ##_ ## CPU);

#define ALL_LUMA_PU_TYPED(prim, fncdef, fname, cpu) \
    p.pu[LUMA_4x4].prim   = fncdef PFX(fname ## _4x4_ ## cpu); \
    p.pu[LUMA_8x8].prim   = fncdef PFX(fname ## _8x8_ ## cpu); \
    p.pu[LUMA_16x16].prim = fncdef PFX(fname ## _16x16_ ## cpu); \
    p.pu[LUMA_32x32].prim = fncdef PFX(fname ## _32x32_ ## cpu); \
    p.pu[LUMA_64x64].prim = fncdef PFX(fname ## _64x64_ ## cpu); \
    p.pu[LUMA_8x4].prim   = fncdef PFX(fname ## _8x4_ ## cpu); \
    p.pu[LUMA_4x8].prim   = fncdef PFX(fname ## _4x8_ ## cpu); \
    p.pu[LUMA_16x8].prim  = fncdef PFX(fname ## _16x8_ ## cpu); \
    p.pu[LUMA_8x16].prim  = fncdef PFX(fname ## _8x16_ ## cpu); \
    p.pu[LUMA_16x32].prim = fncdef PFX(fname ## _16x32_ ## cpu); \
    p.pu[LUMA_32x16].prim = fncdef PFX(fname ## _32x16_ ## cpu); \
    p.pu[LUMA_64x32].prim = fncdef PFX(fname ## _64x32_ ## cpu); \
    p.pu[LUMA_32x64].prim = fncdef PFX(fname ## _32x64_ ## cpu); \
    p.pu[LUMA_16x12].prim = fncdef PFX(fname ## _16x12_ ## cpu); \
    p.pu[LUMA_12x16].prim = fncdef PFX(fname ## _12x16_ ## cpu); \
    p.pu[LUMA_16x4].prim  = fncdef PFX(fname ## _16x4_ ## cpu); \
    p.pu[LUMA_4x16].prim  = fncdef PFX(fname ## _4x16_ ## cpu); \
    p.pu[LUMA_32x24].prim = fncdef PFX(fname ## _32x24_ ## cpu); \
    p.pu[LUMA_24x32].prim = fncdef PFX(fname ## _24x32_ ## cpu); \
    p.pu[LUMA_32x8].prim  = fncdef PFX(fname ## _32x8_ ## cpu); \
    p.pu[LUMA_8x32].prim  = fncdef PFX(fname ## _8x32_ ## cpu); \
    p.pu[LUMA_64x48].prim = fncdef PFX(fname ## _64x48_ ## cpu); \
    p.pu[LUMA_48x64].prim = fncdef PFX(fname ## _48x64_ ## cpu); \
    p.pu[LUMA_64x16].prim = fncdef PFX(fname ## _64x16_ ## cpu); \
    p.pu[LUMA_16x64].prim = fncdef PFX(fname ## _16x64_ ## cpu)

#define ALL_LUMA_TU_TYPED(prim, fncdef, fname, cpu) \
    p.cu[BLOCK_4x4].prim   = fncdef PFX(fname ## _4x4_ ## cpu); \
    p.cu[BLOCK_8x8].prim   = fncdef PFX(fname ## _8x8_ ## cpu); \
    p.cu[BLOCK_16x16].prim = fncdef PFX(fname ## _16x16_ ## cpu); \
    p.cu[BLOCK_32x32].prim = fncdef PFX(fname ## _32x32_ ## cpu); \
    p.cu[BLOCK_64x64].prim = fncdef PFX(fname ## _64x64_ ## cpu)

void setupAssemblyPrimitives(EncoderPrimitives &p, int cpuMask)
{
#if HAVE_LSX
    if (cpuMask & X265_CPU_LSX)
    {
        // loopfilter
        p.saoCuOrgE0 = PFX(saoCuOrgE0_lsx);
        p.saoCuOrgE1 = PFX(saoCuOrgE1_lsx);
        p.saoCuOrgE1_2Rows = PFX(saoCuOrgE1_2Rows_lsx);
        p.saoCuOrgE2[0] = PFX(saoCuOrgE2_lsx);
        p.saoCuOrgE2[1] = PFX(saoCuOrgE2_lsx);
        p.saoCuOrgE3[0] = PFX(saoCuOrgE3_lsx);
        p.saoCuOrgE3[1] = PFX(saoCuOrgE3_lsx);
        p.saoCuOrgB0 = PFX(saoCuOrgB0_lsx);
        p.sign = PFX(calSign_lsx);
        p.pelFilterLumaStrong[0] = PFX(pelFilterLumaStrong_V_lsx);
        p.pelFilterLumaStrong[1] = PFX(pelFilterLumaStrong_H_lsx);
        p.pelFilterChroma[0] = PFX(pelFilterChroma_V_lsx);
        p.pelFilterChroma[1] = PFX(pelFilterChroma_H_lsx);
        p.saoCuStatsE0 = PFX(saoCuStatsE0_lsx);
        p.saoCuStatsE1 = PFX(saoCuStatsE1_lsx);
        p.saoCuStatsE2 = PFX(saoCuStatsE2_lsx);
        p.saoCuStatsE3 = PFX(saoCuStatsE3_lsx);
        p.saoCuStatsBO = PFX(saoCuStatsB0_lsx);

        // ipfilter
        LUMA(4,  4,  lsx)
        LUMA(4,  8,  lsx)
        LUMA(4,  16, lsx)
        LUMA(8,  4,  lsx)
        LUMA(8,  8,  lsx)
        LUMA(8,  16, lsx)
        LUMA(8,  32, lsx)
        LUMA(12, 16, lsx)
        LUMA(16, 4,  lsx)
        LUMA(16, 8,  lsx)
        LUMA(16, 12, lsx)
        LUMA(16, 16, lsx)
        LUMA(16, 32, lsx)
        LUMA(16, 64, lsx)
        LUMA(24, 32, lsx)
        LUMA(32, 8,  lsx)
        LUMA(32, 16, lsx)
        LUMA(32, 24, lsx)
        LUMA(32, 32, lsx)
        LUMA(32, 64, lsx)
        LUMA(48, 64, lsx)
        LUMA(64, 16, lsx)
        LUMA(64, 32, lsx)
        LUMA(64, 48, lsx)
        LUMA(64, 64, lsx)

        CHROMA_420(2,  4,  lsx)
        CHROMA_420(2,  8,  lsx)
        CHROMA_420(4,  2,  lsx)
        CHROMA_420(4,  4,  lsx)
        CHROMA_420(4,  8,  lsx)
        CHROMA_420(4,  16, lsx)
        CHROMA_420(6,  8,  lsx)
        CHROMA_420(8,  2,  lsx)
        CHROMA_420(8,  4,  lsx)
        CHROMA_420(8,  6,  lsx)
        CHROMA_420(8,  8,  lsx)
        CHROMA_420(8,  16, lsx)
        CHROMA_420(8,  32, lsx)
        CHROMA_420(12, 16, lsx)
        CHROMA_420(16, 4,  lsx)
        CHROMA_420(16, 8,  lsx)
        CHROMA_420(16, 12, lsx)
        CHROMA_420(16, 16, lsx)
        CHROMA_420(16, 32, lsx)
        CHROMA_420(24, 32, lsx)
        CHROMA_420(32, 8,  lsx)
        CHROMA_420(32, 16, lsx)
        CHROMA_420(32, 24, lsx)
        CHROMA_420(32, 32, lsx)

        CHROMA_422(2,  4,  lsx)
        CHROMA_422(2,  8,  lsx)
        CHROMA_422(2,  16, lsx)
        CHROMA_422(4,  4,  lsx)
        CHROMA_422(4,  8,  lsx)
        CHROMA_422(4,  16, lsx)
        CHROMA_422(4,  32, lsx)
        CHROMA_422(6,  16, lsx)
        CHROMA_422(8,  4,  lsx)
        CHROMA_422(8,  8,  lsx)
        CHROMA_422(8,  12, lsx)
        CHROMA_422(8,  16, lsx)
        CHROMA_422(8,  32, lsx)
        CHROMA_422(8,  64, lsx)
        CHROMA_422(12, 32, lsx)
        CHROMA_422(16, 8,  lsx)
        CHROMA_422(16, 16, lsx)
        CHROMA_422(16, 24, lsx)
        CHROMA_422(16, 32, lsx)
        CHROMA_422(16, 64, lsx)
        CHROMA_422(24, 64, lsx)
        CHROMA_422(32, 16, lsx)
        CHROMA_422(32, 32, lsx)
        CHROMA_422(32, 48, lsx)
        CHROMA_422(32, 64, lsx)

        CHROMA_444(4,  4,  lsx)
        CHROMA_444(4,  8,  lsx)
        CHROMA_444(4,  16, lsx)
        CHROMA_444(8,  4,  lsx)
        CHROMA_444(8,  8,  lsx)
        CHROMA_444(8,  16, lsx)
        CHROMA_444(8,  32, lsx)
        CHROMA_444(12, 16, lsx)
        CHROMA_444(16, 4,  lsx)
        CHROMA_444(16, 8,  lsx)
        CHROMA_444(16, 12, lsx)
        CHROMA_444(16, 16, lsx)
        CHROMA_444(16, 32, lsx)
        CHROMA_444(16, 64, lsx)
        CHROMA_444(24, 32, lsx)
        CHROMA_444(32, 8,  lsx)
        CHROMA_444(32, 16, lsx)
        CHROMA_444(32, 24, lsx)
        CHROMA_444(32, 32, lsx)
        CHROMA_444(32, 64, lsx)
        CHROMA_444(48, 64, lsx)
        CHROMA_444(64, 16, lsx)
        CHROMA_444(64, 32, lsx)
        CHROMA_444(64, 48, lsx)
        CHROMA_444(64, 64, lsx)

        //dct
        p.cu[BLOCK_32x32].dct  = PFX(dct32_lsx);
        p.cu[BLOCK_16x16].dct  = PFX(dct16_lsx);
        p.cu[BLOCK_8x8].dct    = PFX(dct8_lsx);
        p.cu[BLOCK_4x4].dct    = PFX(dct4_lsx);
        p.cu[BLOCK_4x4].idct   = PFX(idct4_lsx);
        p.cu[BLOCK_8x8].idct   = PFX(idct8_lsx);
        p.cu[BLOCK_16x16].idct = PFX(idct16_lsx);
        p.cu[BLOCK_32x32].idct = PFX(idct32_lsx);
        p.dst4x4               = PFX(dst4_lsx);
        p.idst4x4              = PFX(idst4_lsx);
        p.scanPosLast          = PFX(scanPosLast_lsx);
        p.costCoeffNxN         = PFX(costCoeffNxN_lsx);
        p.quant                = PFX(quant_lsx);
        p.dequant_normal       = PFX(dequant_normal_lsx);
        p.dequant_scaling      = PFX(dequant_scaling_lsx);
        p.nquant               = PFX(nquant_lsx);
        p.cu[BLOCK_32x32].count_nonzero = PFX(count_nonzero_32_lsx);
        p.cu[BLOCK_16x16].count_nonzero = PFX(count_nonzero_16_lsx);
        p.cu[BLOCK_8x8].count_nonzero   = PFX(count_nonzero_8_lsx);
        p.cu[BLOCK_4x4].count_nonzero   = PFX(count_nonzero_4_lsx);
        p.costC1C2Flag         = PFX(costC1C2Flag_lsx);
        p.costCoeffRemain      = PFX(costCoeffRemain_lsx);
        p.findPosFirstLast     = PFX(findPosFirstLast_lsx);
        p.denoiseDct           = PFX(denoiseDct_lsx);

        //pixelAvg_pp
        ASSIGN2(p.pu[LUMA_64x64].pixelavg_pp, pixel_avg_64x64_lsx);
        ASSIGN2(p.pu[LUMA_64x48].pixelavg_pp, pixel_avg_64x48_lsx);
        ASSIGN2(p.pu[LUMA_64x32].pixelavg_pp, pixel_avg_64x32_lsx);
        ASSIGN2(p.pu[LUMA_64x16].pixelavg_pp, pixel_avg_64x16_lsx);
        ASSIGN2(p.pu[LUMA_48x64].pixelavg_pp, pixel_avg_48x64_lsx);
        ASSIGN2(p.pu[LUMA_32x64].pixelavg_pp, pixel_avg_32x64_lsx);
        ASSIGN2(p.pu[LUMA_32x32].pixelavg_pp, pixel_avg_32x32_lsx);
        ASSIGN2(p.pu[LUMA_32x24].pixelavg_pp, pixel_avg_32x24_lsx);
        ASSIGN2(p.pu[LUMA_32x16].pixelavg_pp, pixel_avg_32x16_lsx);
        ASSIGN2(p.pu[LUMA_32x8].pixelavg_pp, pixel_avg_32x8_lsx);
        ASSIGN2(p.pu[LUMA_24x32].pixelavg_pp, pixel_avg_24x32_lsx);
        ASSIGN2(p.pu[LUMA_16x64].pixelavg_pp, pixel_avg_16x64_lsx);
        ASSIGN2(p.pu[LUMA_16x32].pixelavg_pp, pixel_avg_16x32_lsx);
        ASSIGN2(p.pu[LUMA_16x16].pixelavg_pp, pixel_avg_16x16_lsx);
        ASSIGN2(p.pu[LUMA_16x12].pixelavg_pp, pixel_avg_16x12_lsx);
        ASSIGN2(p.pu[LUMA_16x8].pixelavg_pp, pixel_avg_16x8_lsx);
        ASSIGN2(p.pu[LUMA_16x4].pixelavg_pp, pixel_avg_16x4_lsx);
        ASSIGN2(p.pu[LUMA_12x16].pixelavg_pp, pixel_avg_12x16_lsx);
        ASSIGN2(p.pu[LUMA_8x32].pixelavg_pp, pixel_avg_8x32_lsx);
        ASSIGN2(p.pu[LUMA_8x16].pixelavg_pp, pixel_avg_8x16_lsx);
        ASSIGN2(p.pu[LUMA_8x8].pixelavg_pp, pixel_avg_8x8_lsx);
        ASSIGN2(p.pu[LUMA_8x4].pixelavg_pp, pixel_avg_8x4_lsx);
        ASSIGN2(p.pu[LUMA_4x16].pixelavg_pp, pixel_avg_4x16_lsx);
        ASSIGN2(p.pu[LUMA_4x8].pixelavg_pp, pixel_avg_4x8_lsx);
        ASSIGN2(p.pu[LUMA_4x4].pixelavg_pp, pixel_avg_4x4_lsx);

        LUMA_PU(4,   4, lsx);
        LUMA_PU(8,   8, lsx);
        LUMA_PU(16, 16, lsx);
        LUMA_PU(32, 32, lsx);
        LUMA_PU(64, 64, lsx);
        LUMA_PU(4,   8, lsx);
        LUMA_PU(8,   4, lsx);
        LUMA_PU(16,  8, lsx);
        LUMA_PU(8,  16, lsx);
        LUMA_PU(16, 12, lsx);
        LUMA_PU(12, 16, lsx);
        LUMA_PU(16,  4, lsx);
        LUMA_PU(4,  16, lsx);
        LUMA_PU(32, 16, lsx);
        LUMA_PU(16, 32, lsx);
        LUMA_PU(32, 24, lsx);
        LUMA_PU(24, 32, lsx);
        LUMA_PU(32,  8, lsx);
        LUMA_PU(8,  32, lsx);
        LUMA_PU(64, 32, lsx);
        LUMA_PU(32, 64, lsx);
        LUMA_PU(64, 48, lsx);
        LUMA_PU(48, 64, lsx);
        LUMA_PU(64, 16, lsx);
        LUMA_PU(16, 64, lsx);

        CHROMA_PU_420(2,   2, lsx);
        CHROMA_PU_420(2,   4, lsx);
        CHROMA_PU_420(4,   4, lsx);
        CHROMA_PU_420(8,   8, lsx);
        CHROMA_PU_420(16, 16, lsx);
        CHROMA_PU_420(32, 32, lsx);
        CHROMA_PU_420(4,   2, lsx);
        CHROMA_PU_420(8,   4, lsx);
        CHROMA_PU_420(4,   8, lsx);
        CHROMA_PU_420(8,   6, lsx);
        CHROMA_PU_420(6,   8, lsx);
        CHROMA_PU_420(8,   2, lsx);
        CHROMA_PU_420(2,   8, lsx);
        CHROMA_PU_420(16,  8, lsx);
        CHROMA_PU_420(8,  16, lsx);
        CHROMA_PU_420(16, 12, lsx);
        CHROMA_PU_420(12, 16, lsx);
        CHROMA_PU_420(16,  4, lsx);
        CHROMA_PU_420(4,  16, lsx);
        CHROMA_PU_420(32, 16, lsx);
        CHROMA_PU_420(16, 32, lsx);
        CHROMA_PU_420(32, 24, lsx);
        CHROMA_PU_420(24, 32, lsx);
        CHROMA_PU_420(32,  8, lsx);
        CHROMA_PU_420(8,  32, lsx);

        CHROMA_PU_422(2,   4, lsx);
        CHROMA_PU_422(4,   8, lsx);
        CHROMA_PU_422(8,  16, lsx);
        CHROMA_PU_422(16, 32, lsx);
        CHROMA_PU_422(32, 64, lsx);
        CHROMA_PU_422(4,   4, lsx);
        CHROMA_PU_422(2,   8, lsx);
        CHROMA_PU_422(8,   8, lsx);
        CHROMA_PU_422(4,  16, lsx);
        CHROMA_PU_422(8,  12, lsx);
        CHROMA_PU_422(6,  16, lsx);
        CHROMA_PU_422(8,   4, lsx);
        CHROMA_PU_422(2,  16, lsx);
        CHROMA_PU_422(16, 16, lsx);
        CHROMA_PU_422(8,  32, lsx);
        CHROMA_PU_422(16, 24, lsx);
        CHROMA_PU_422(12, 32, lsx);
        CHROMA_PU_422(16,  8, lsx);
        CHROMA_PU_422(4,  32, lsx);
        CHROMA_PU_422(32, 32, lsx);
        CHROMA_PU_422(16, 64, lsx);
        CHROMA_PU_422(32, 48, lsx);
        CHROMA_PU_422(24, 64, lsx);
        CHROMA_PU_422(32, 16, lsx);
        CHROMA_PU_422(8,  64, lsx);

        // intra_pred
        p.cu[BLOCK_4x4].intra_filter = PFX(intra_filter_4x4_lsx);
        p.cu[BLOCK_8x8].intra_filter = PFX(intra_filter_8x8_lsx);
        p.cu[BLOCK_16x16].intra_filter = PFX(intra_filter_16x16_lsx);
        p.cu[BLOCK_32x32].intra_filter = PFX(intra_filter_32x32_lsx);

        p.cu[BLOCK_4x4].intra_pred[PLANAR_IDX]   = PFX(intra_pred_planar_4x4_lsx);
        p.cu[BLOCK_8x8].intra_pred[PLANAR_IDX]   = PFX(intra_pred_planar_8x8_lsx);
        p.cu[BLOCK_16x16].intra_pred[PLANAR_IDX] = PFX(intra_pred_planar_16x16_lsx);
        p.cu[BLOCK_32x32].intra_pred[PLANAR_IDX] = PFX(intra_pred_planar_32x32_lsx);

        p.cu[BLOCK_4x4].intra_pred[2] = PFX(intra_pred_ang4_2_lsx);
        p.cu[BLOCK_4x4].intra_pred[3] = PFX(intra_pred_ang4_3_lsx);
        p.cu[BLOCK_4x4].intra_pred[4] = PFX(intra_pred_ang4_4_lsx);
        p.cu[BLOCK_4x4].intra_pred[5] = PFX(intra_pred_ang4_5_lsx);
        p.cu[BLOCK_4x4].intra_pred[6] = PFX(intra_pred_ang4_6_lsx);
        p.cu[BLOCK_4x4].intra_pred[7] = PFX(intra_pred_ang4_7_lsx);
        p.cu[BLOCK_4x4].intra_pred[8] = PFX(intra_pred_ang4_8_lsx);
        p.cu[BLOCK_4x4].intra_pred[9] = PFX(intra_pred_ang4_9_lsx);
        p.cu[BLOCK_4x4].intra_pred[10] = PFX(intra_pred_ang4_10_lsx);
        p.cu[BLOCK_4x4].intra_pred[11] = PFX(intra_pred_ang4_11_lsx);
        p.cu[BLOCK_4x4].intra_pred[12] = PFX(intra_pred_ang4_12_lsx);
        p.cu[BLOCK_4x4].intra_pred[13] = PFX(intra_pred_ang4_13_lsx);
        p.cu[BLOCK_4x4].intra_pred[14] = PFX(intra_pred_ang4_14_lsx);
        p.cu[BLOCK_4x4].intra_pred[15] = PFX(intra_pred_ang4_15_lsx);
        p.cu[BLOCK_4x4].intra_pred[16] = PFX(intra_pred_ang4_16_lsx);
        p.cu[BLOCK_4x4].intra_pred[17] = PFX(intra_pred_ang4_17_lsx);
        p.cu[BLOCK_4x4].intra_pred[18] = PFX(intra_pred_ang4_18_lsx);
        p.cu[BLOCK_4x4].intra_pred[19] = PFX(intra_pred_ang4_19_lsx);
        p.cu[BLOCK_4x4].intra_pred[20] = PFX(intra_pred_ang4_20_lsx);
        p.cu[BLOCK_4x4].intra_pred[21] = PFX(intra_pred_ang4_21_lsx);
        p.cu[BLOCK_4x4].intra_pred[22] = PFX(intra_pred_ang4_22_lsx);
        p.cu[BLOCK_4x4].intra_pred[23] = PFX(intra_pred_ang4_23_lsx);
        p.cu[BLOCK_4x4].intra_pred[24] = PFX(intra_pred_ang4_24_lsx);
        p.cu[BLOCK_4x4].intra_pred[25] = PFX(intra_pred_ang4_25_lsx);
        p.cu[BLOCK_4x4].intra_pred[26] = PFX(intra_pred_ang4_26_lsx);
        p.cu[BLOCK_4x4].intra_pred[27] = PFX(intra_pred_ang4_27_lsx);
        p.cu[BLOCK_4x4].intra_pred[28] = PFX(intra_pred_ang4_28_lsx);
        p.cu[BLOCK_4x4].intra_pred[29] = PFX(intra_pred_ang4_29_lsx);
        p.cu[BLOCK_4x4].intra_pred[30] = PFX(intra_pred_ang4_30_lsx);
        p.cu[BLOCK_4x4].intra_pred[31] = PFX(intra_pred_ang4_31_lsx);
        p.cu[BLOCK_4x4].intra_pred[32] = PFX(intra_pred_ang4_32_lsx);
        p.cu[BLOCK_4x4].intra_pred[33] = PFX(intra_pred_ang4_33_lsx);
        p.cu[BLOCK_4x4].intra_pred[34] = PFX(intra_pred_ang4_34_lsx);

        p.cu[BLOCK_8x8].intra_pred[2] = PFX(intra_pred_ang8_2_lsx);
        p.cu[BLOCK_8x8].intra_pred[3] = PFX(intra_pred_ang8_3_lsx);
        p.cu[BLOCK_8x8].intra_pred[4] = PFX(intra_pred_ang8_4_lsx);
        p.cu[BLOCK_8x8].intra_pred[5] = PFX(intra_pred_ang8_5_lsx);
        p.cu[BLOCK_8x8].intra_pred[6] = PFX(intra_pred_ang8_6_lsx);
        p.cu[BLOCK_8x8].intra_pred[7] = PFX(intra_pred_ang8_7_lsx);
        p.cu[BLOCK_8x8].intra_pred[8] = PFX(intra_pred_ang8_8_lsx);
        p.cu[BLOCK_8x8].intra_pred[9] = PFX(intra_pred_ang8_9_lsx);
        p.cu[BLOCK_8x8].intra_pred[10] = PFX(intra_pred_ang8_10_lsx);
        p.cu[BLOCK_8x8].intra_pred[11] = PFX(intra_pred_ang8_11_lsx);
        p.cu[BLOCK_8x8].intra_pred[12] = PFX(intra_pred_ang8_12_lsx);
        p.cu[BLOCK_8x8].intra_pred[13] = PFX(intra_pred_ang8_13_lsx);
        p.cu[BLOCK_8x8].intra_pred[14] = PFX(intra_pred_ang8_14_lsx);
        p.cu[BLOCK_8x8].intra_pred[15] = PFX(intra_pred_ang8_15_lsx);
        p.cu[BLOCK_8x8].intra_pred[16] = PFX(intra_pred_ang8_16_lsx);
        p.cu[BLOCK_8x8].intra_pred[17] = PFX(intra_pred_ang8_17_lsx);
        p.cu[BLOCK_8x8].intra_pred[18] = PFX(intra_pred_ang8_18_lsx);
        p.cu[BLOCK_8x8].intra_pred[19] = PFX(intra_pred_ang8_17_lsx);
        p.cu[BLOCK_8x8].intra_pred[20] = PFX(intra_pred_ang8_16_lsx);
        p.cu[BLOCK_8x8].intra_pred[21] = PFX(intra_pred_ang8_15_lsx);
        p.cu[BLOCK_8x8].intra_pred[22] = PFX(intra_pred_ang8_14_lsx);
        p.cu[BLOCK_8x8].intra_pred[23] = PFX(intra_pred_ang8_13_lsx);
        p.cu[BLOCK_8x8].intra_pred[24] = PFX(intra_pred_ang8_12_lsx);
        p.cu[BLOCK_8x8].intra_pred[25] = PFX(intra_pred_ang8_11_lsx);
        p.cu[BLOCK_8x8].intra_pred[26] = PFX(intra_pred_ang8_26_lsx);
        p.cu[BLOCK_8x8].intra_pred[27] = PFX(intra_pred_ang8_9_lsx);
        p.cu[BLOCK_8x8].intra_pred[28] = PFX(intra_pred_ang8_8_lsx);
        p.cu[BLOCK_8x8].intra_pred[29] = PFX(intra_pred_ang8_7_lsx);
        p.cu[BLOCK_8x8].intra_pred[30] = PFX(intra_pred_ang8_6_lsx);
        p.cu[BLOCK_8x8].intra_pred[31] = PFX(intra_pred_ang8_5_lsx);
        p.cu[BLOCK_8x8].intra_pred[32] = PFX(intra_pred_ang8_4_lsx);
        p.cu[BLOCK_8x8].intra_pred[33] = PFX(intra_pred_ang8_3_lsx);
        p.cu[BLOCK_8x8].intra_pred[34] = PFX(intra_pred_ang8_34_lsx);

        p.cu[BLOCK_16x16].intra_pred[2] = PFX(intra_pred_ang16_2_lsx);
        p.cu[BLOCK_16x16].intra_pred[3] = PFX(intra_pred_ang16_3_lsx);
        p.cu[BLOCK_16x16].intra_pred[4] = PFX(intra_pred_ang16_4_lsx);
        p.cu[BLOCK_16x16].intra_pred[5] = PFX(intra_pred_ang16_5_lsx);
        p.cu[BLOCK_16x16].intra_pred[6] = PFX(intra_pred_ang16_6_lsx);
        p.cu[BLOCK_16x16].intra_pred[7] = PFX(intra_pred_ang16_7_lsx);
        p.cu[BLOCK_16x16].intra_pred[8] = PFX(intra_pred_ang16_8_lsx);
        p.cu[BLOCK_16x16].intra_pred[9] = PFX(intra_pred_ang16_9_lsx);
        p.cu[BLOCK_16x16].intra_pred[10] = PFX(intra_pred_ang16_10_lsx);
        p.cu[BLOCK_16x16].intra_pred[11] = PFX(intra_pred_ang16_11_lsx);
        p.cu[BLOCK_16x16].intra_pred[12] = PFX(intra_pred_ang16_12_lsx);
        p.cu[BLOCK_16x16].intra_pred[13] = PFX(intra_pred_ang16_13_lsx);
        p.cu[BLOCK_16x16].intra_pred[14] = PFX(intra_pred_ang16_14_lsx);
        p.cu[BLOCK_16x16].intra_pred[15] = PFX(intra_pred_ang16_15_lsx);
        p.cu[BLOCK_16x16].intra_pred[16] = PFX(intra_pred_ang16_16_lsx);
        p.cu[BLOCK_16x16].intra_pred[17] = PFX(intra_pred_ang16_17_lsx);
        p.cu[BLOCK_16x16].intra_pred[18] = PFX(intra_pred_ang16_18_lsx);
        p.cu[BLOCK_16x16].intra_pred[19] = PFX(intra_pred_ang16_19_lsx);
        p.cu[BLOCK_16x16].intra_pred[20] = PFX(intra_pred_ang16_20_lsx);
        p.cu[BLOCK_16x16].intra_pred[21] = PFX(intra_pred_ang16_21_lsx);
        p.cu[BLOCK_16x16].intra_pred[22] = PFX(intra_pred_ang16_22_lsx);
        p.cu[BLOCK_16x16].intra_pred[23] = PFX(intra_pred_ang16_23_lsx);
        p.cu[BLOCK_16x16].intra_pred[24] = PFX(intra_pred_ang16_24_lsx);
        p.cu[BLOCK_16x16].intra_pred[25] = PFX(intra_pred_ang16_25_lsx);
        p.cu[BLOCK_16x16].intra_pred[26] = PFX(intra_pred_ang16_26_lsx);
        p.cu[BLOCK_16x16].intra_pred[27] = PFX(intra_pred_ang16_27_lsx);
        p.cu[BLOCK_16x16].intra_pred[28] = PFX(intra_pred_ang16_28_lsx);
        p.cu[BLOCK_16x16].intra_pred[29] = PFX(intra_pred_ang16_29_lsx);
        p.cu[BLOCK_16x16].intra_pred[30] = PFX(intra_pred_ang16_30_lsx);
        p.cu[BLOCK_16x16].intra_pred[31] = PFX(intra_pred_ang16_31_lsx);
        p.cu[BLOCK_16x16].intra_pred[32] = PFX(intra_pred_ang16_32_lsx);
        p.cu[BLOCK_16x16].intra_pred[33] = PFX(intra_pred_ang16_33_lsx);
        p.cu[BLOCK_16x16].intra_pred[34] = PFX(intra_pred_ang16_2_lsx);

        p.cu[BLOCK_32x32].intra_pred[2] = PFX(intra_pred_ang32_2_lsx);
        p.cu[BLOCK_32x32].intra_pred[3] = PFX(intra_pred_ang32_3_lsx);
        p.cu[BLOCK_32x32].intra_pred[4] = PFX(intra_pred_ang32_4_lsx);
        p.cu[BLOCK_32x32].intra_pred[5] = PFX(intra_pred_ang32_5_lsx);
        p.cu[BLOCK_32x32].intra_pred[6] = PFX(intra_pred_ang32_6_lsx);
        p.cu[BLOCK_32x32].intra_pred[7] = PFX(intra_pred_ang32_7_lsx);
        p.cu[BLOCK_32x32].intra_pred[8] = PFX(intra_pred_ang32_8_lsx);
        p.cu[BLOCK_32x32].intra_pred[9] = PFX(intra_pred_ang32_9_lsx);
        p.cu[BLOCK_32x32].intra_pred[10] = PFX(intra_pred_ang32_10_lsx);
        p.cu[BLOCK_32x32].intra_pred[11] = PFX(intra_pred_ang32_11_lsx);
        p.cu[BLOCK_32x32].intra_pred[12] = PFX(intra_pred_ang32_12_lsx);
        p.cu[BLOCK_32x32].intra_pred[13] = PFX(intra_pred_ang32_13_lsx);
        p.cu[BLOCK_32x32].intra_pred[14] = PFX(intra_pred_ang32_14_lsx);
        p.cu[BLOCK_32x32].intra_pred[15] = PFX(intra_pred_ang32_15_lsx);
        p.cu[BLOCK_32x32].intra_pred[16] = PFX(intra_pred_ang32_16_lsx);
        p.cu[BLOCK_32x32].intra_pred[17] = PFX(intra_pred_ang32_17_lsx);
        p.cu[BLOCK_32x32].intra_pred[18] = PFX(intra_pred_ang32_18_lsx);
        p.cu[BLOCK_32x32].intra_pred[19] = PFX(intra_pred_ang32_19_lsx);
        p.cu[BLOCK_32x32].intra_pred[20] = PFX(intra_pred_ang32_20_lsx);
        p.cu[BLOCK_32x32].intra_pred[21] = PFX(intra_pred_ang32_21_lsx);
        p.cu[BLOCK_32x32].intra_pred[22] = PFX(intra_pred_ang32_22_lsx);
        p.cu[BLOCK_32x32].intra_pred[23] = PFX(intra_pred_ang32_23_lsx);
        p.cu[BLOCK_32x32].intra_pred[24] = PFX(intra_pred_ang32_24_lsx);
        p.cu[BLOCK_32x32].intra_pred[25] = PFX(intra_pred_ang32_25_lsx);
        p.cu[BLOCK_32x32].intra_pred[26] = PFX(intra_pred_ang32_26_lsx);
        p.cu[BLOCK_32x32].intra_pred[27] = PFX(intra_pred_ang32_27_lsx);
        p.cu[BLOCK_32x32].intra_pred[28] = PFX(intra_pred_ang32_28_lsx);
        p.cu[BLOCK_32x32].intra_pred[29] = PFX(intra_pred_ang32_29_lsx);
        p.cu[BLOCK_32x32].intra_pred[30] = PFX(intra_pred_ang32_30_lsx);
        p.cu[BLOCK_32x32].intra_pred[31] = PFX(intra_pred_ang32_31_lsx);
        p.cu[BLOCK_32x32].intra_pred[32] = PFX(intra_pred_ang32_32_lsx);
        p.cu[BLOCK_32x32].intra_pred[33] = PFX(intra_pred_ang32_33_lsx);
        p.cu[BLOCK_32x32].intra_pred[34] = PFX(intra_pred_ang32_34_lsx);

        p.cu[BLOCK_4x4].intra_pred_allangs   = all_angs_pred4_lsx;
        p.cu[BLOCK_8x8].intra_pred_allangs   = all_angs_pred8_lsx;
        p.cu[BLOCK_16x16].intra_pred_allangs = all_angs_pred16_lsx;
        p.cu[BLOCK_32x32].intra_pred_allangs = all_angs_pred32_lsx;

        //intra_pred_dc
        p.cu[BLOCK_4x4].intra_pred[DC_IDX] = PFX(intra_pred_dc4_lsx);
        p.cu[BLOCK_8x8].intra_pred[DC_IDX] = PFX(intra_pred_dc8_lsx);
        p.cu[BLOCK_16x16].intra_pred[DC_IDX] = PFX(intra_pred_dc16_lsx);
        p.cu[BLOCK_32x32].intra_pred[DC_IDX] = PFX(intra_pred_dc32_lsx);

        // luma satd
        ALL_LUMA_PU_TYPED(satd, , pixel_satd, lsx);

        // chroma satd
        p.chroma[X265_CSP_I420].pu[CHROMA_420_2x2].satd   = NULL;
        p.chroma[X265_CSP_I420].pu[CHROMA_420_4x4].satd   = PFX(pixel_satd_4x4_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_8x8].satd   = PFX(pixel_satd_8x8_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_16x16].satd = PFX(pixel_satd_16x16_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_32x32].satd = PFX(pixel_satd_32x32_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_4x2].satd   = NULL;
        p.chroma[X265_CSP_I420].pu[CHROMA_420_2x4].satd   = NULL;
        p.chroma[X265_CSP_I420].pu[CHROMA_420_8x4].satd   = PFX(pixel_satd_8x4_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_4x8].satd   = PFX(pixel_satd_4x8_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_16x8].satd  = PFX(pixel_satd_16x8_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_8x16].satd  = PFX(pixel_satd_8x16_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_32x16].satd = PFX(pixel_satd_32x16_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_16x32].satd = PFX(pixel_satd_16x32_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_8x6].satd   = NULL;
        p.chroma[X265_CSP_I420].pu[CHROMA_420_6x8].satd   = NULL;
        p.chroma[X265_CSP_I420].pu[CHROMA_420_8x2].satd   = NULL;
        p.chroma[X265_CSP_I420].pu[CHROMA_420_2x8].satd   = NULL;
        p.chroma[X265_CSP_I420].pu[CHROMA_420_16x12].satd = PFX(pixel_satd_16x12_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_12x16].satd = PFX(pixel_satd_12x16_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_16x4].satd  = PFX(pixel_satd_16x4_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_4x16].satd  = PFX(pixel_satd_4x16_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_32x24].satd = PFX(pixel_satd_32x24_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_24x32].satd = PFX(pixel_satd_24x32_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_32x8].satd  = PFX(pixel_satd_32x8_lsx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_8x32].satd  = PFX(pixel_satd_8x32_lsx);

        p.chroma[X265_CSP_I422].pu[CHROMA_422_4x4].satd   = PFX(pixel_satd_4x4_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_4x8].satd   = PFX(pixel_satd_4x8_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_4x16].satd  = PFX(pixel_satd_4x16_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_8x4].satd   = PFX(pixel_satd_8x4_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_8x8].satd   = PFX(pixel_satd_8x8_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_8x16].satd  = PFX(pixel_satd_8x16_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_16x8].satd  = PFX(pixel_satd_16x8_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_16x16].satd = PFX(pixel_satd_16x16_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_2x4].satd   = NULL;
        p.chroma[X265_CSP_I422].pu[CHROMA_422_16x32].satd = PFX(pixel_satd_16x32_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_32x64].satd = PFX(pixel_satd_32x64_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_2x8].satd   = NULL;
        p.chroma[X265_CSP_I422].pu[CHROMA_422_8x32].satd  = PFX(pixel_satd_8x32_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_32x32].satd = PFX(pixel_satd_32x32_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_16x64].satd = PFX(pixel_satd_16x64_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_8x12].satd  = PFX(pixel_satd_8x12_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_6x16].satd  = NULL;
        p.chroma[X265_CSP_I422].pu[CHROMA_422_2x16].satd  = NULL;
        p.chroma[X265_CSP_I422].pu[CHROMA_422_16x24].satd = PFX(pixel_satd_16x24_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_12x32].satd = PFX(pixel_satd_12x32_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_4x32].satd  = PFX(pixel_satd_4x32_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_32x48].satd = PFX(pixel_satd_32x48_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_24x64].satd = PFX(pixel_satd_24x64_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_32x16].satd = PFX(pixel_satd_32x16_lsx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_8x64].satd  = PFX(pixel_satd_8x64_lsx);

        // sa8d
        p.cu[BLOCK_4x4].sa8d   = PFX(pixel_satd_4x4_lsx);
        p.cu[BLOCK_8x8].sa8d   = PFX(pixel_sa8d_8x8_lsx);
        p.cu[BLOCK_16x16].sa8d = PFX(pixel_sa8d_16x16_lsx);
        p.cu[BLOCK_32x32].sa8d = PFX(pixel_sa8d_32x32_lsx);
        p.cu[BLOCK_64x64].sa8d = PFX(pixel_sa8d_64x64_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_8x8].sa8d       = PFX(pixel_satd_4x4_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_16x16].sa8d     = PFX(pixel_sa8d_16x16_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_32x32].sa8d     = PFX(pixel_sa8d_32x32_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_64x64].sa8d     = PFX(pixel_sa8d_64x64_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_8x8].sa8d       = PFX(pixel_satd_4x8_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].sa8d  = PFX(pixel_sa8d_8x16_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].sa8d = PFX(pixel_sa8d_16x32_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].sa8d = PFX(pixel_sa8d_32x64_lsx);

        // sad_x4
        ALL_LUMA_PU_TYPED(sad_x4, , pixel_sad_x4, lsx);

        // sad
        ALL_LUMA_PU_TYPED(sad, , pixel_sad, lsx);

        LUMA_CU(4, 4, lsx);
        LUMA_CU(8, 8, lsx);
        LUMA_CU(16, 16, lsx);
        LUMA_CU(32, 32, lsx);
        LUMA_CU(64, 64, lsx);

        CHROMA_CU_420(2, 2, lsx);
        CHROMA_CU_420(4, 4, lsx);
        CHROMA_CU_420(8, 8, lsx);
        CHROMA_CU_420(16, 16, lsx);
        CHROMA_CU_420(32, 32, lsx);

        CHROMA_CU_422(2, 4, lsx);
        CHROMA_CU_422(4, 8, lsx);
        CHROMA_CU_422(8, 16, lsx);
        CHROMA_CU_422(16, 32, lsx);
        CHROMA_CU_422(32, 64, lsx);

        p.chroma[X265_CSP_I420].cu[BLOCK_420_4x4].copy_ps  = PFX(blockcopy_ps_4x4_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_8x8].copy_ps  = PFX(blockcopy_ps_8x8_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].copy_ps= PFX(blockcopy_ps_16x16_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].copy_ps= PFX(blockcopy_ps_32x32_lsx);

        p.chroma[X265_CSP_I420].cu[BLOCK_420_4x4].sse_pp   = PFX(pixel_sse_4x4_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_8x8].sse_pp   = PFX(pixel_sse_8x8_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].sse_pp = PFX(pixel_sse_16x16_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].sse_pp = PFX(pixel_sse_32x32_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_4x8].sse_pp   = PFX(pixel_sse_4x8_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].sse_pp  = PFX(pixel_sse_8x16_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].sse_pp = PFX(pixel_sse_16x32_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].sse_pp = PFX(pixel_sse_32x64_lsx);

        p.weight_pp = PFX(weight_pp_lsx);
        p.weight_sp = PFX(weight_sp_lsx);

        p.cu[BLOCK_4x4].sub_ps   = PFX(pixel_sub_ps_4x4_lsx);
        p.cu[BLOCK_8x8].sub_ps   = PFX(pixel_sub_ps_8x8_lsx);
        p.cu[BLOCK_16x16].sub_ps = PFX(pixel_sub_ps_16x16_lsx);
        p.cu[BLOCK_32x32].sub_ps = PFX(pixel_sub_ps_32x32_lsx);
        p.cu[BLOCK_64x64].sub_ps = PFX(pixel_sub_ps_64x64_lsx);

        // chroma sub_ps
        p.chroma[X265_CSP_I420].cu[BLOCK_420_4x4].sub_ps   = PFX(pixel_sub_ps_4x4_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_8x8].sub_ps   = PFX(pixel_sub_ps_8x8_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].sub_ps = PFX(pixel_sub_ps_16x16_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].sub_ps = PFX(pixel_sub_ps_32x32_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_4x8].sub_ps   = PFX(pixel_sub_ps_4x8_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].sub_ps  = PFX(pixel_sub_ps_8x16_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].sub_ps = PFX(pixel_sub_ps_16x32_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].sub_ps = PFX(pixel_sub_ps_32x64_lsx);

        p.cu[BLOCK_4x4].add_ps[NONALIGNED]   = PFX(pixel_add_ps_4x4_lsx);
        p.cu[BLOCK_8x8].add_ps[NONALIGNED]   = PFX(pixel_add_ps_8x8_lsx);
        p.cu[BLOCK_16x16].add_ps[NONALIGNED] = PFX(pixel_add_ps_16x16_lsx);
        p.cu[BLOCK_32x32].add_ps[NONALIGNED] = PFX(pixel_add_ps_32x32_lsx);
        p.cu[BLOCK_64x64].add_ps[NONALIGNED] = PFX(pixel_add_ps_64x64_lsx);

        p.cu[BLOCK_4x4].add_ps[ALIGNED]   = PFX(pixel_add_ps_4x4_lsx);
        p.cu[BLOCK_8x8].add_ps[ALIGNED]   = PFX(pixel_add_ps_8x8_lsx);
        p.cu[BLOCK_16x16].add_ps[ALIGNED] = PFX(pixel_add_ps_16x16_lsx);
        p.cu[BLOCK_32x32].add_ps[ALIGNED] = PFX(pixel_add_ps_32x32_lsx);
        p.cu[BLOCK_64x64].add_ps[ALIGNED] = PFX(pixel_add_ps_64x64_lsx);

        // chroma add_ps
        p.chroma[X265_CSP_I420].cu[BLOCK_420_4x4].add_ps[NONALIGNED]   = PFX(pixel_add_ps_4x4_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_8x8].add_ps[NONALIGNED]   = PFX(pixel_add_ps_8x8_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].add_ps[NONALIGNED] = PFX(pixel_add_ps_16x16_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].add_ps[NONALIGNED] = PFX(pixel_add_ps_32x32_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_4x8].add_ps[NONALIGNED]   = PFX(pixel_add_ps_4x8_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].add_ps[NONALIGNED]  = PFX(pixel_add_ps_8x16_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].add_ps[NONALIGNED] = PFX(pixel_add_ps_16x32_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].add_ps[NONALIGNED] = PFX(pixel_add_ps_32x64_lsx);

        p.chroma[X265_CSP_I420].cu[BLOCK_420_4x4].add_ps[ALIGNED]   = PFX(pixel_add_ps_4x4_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_8x8].add_ps[ALIGNED]   = PFX(pixel_add_ps_8x8_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].add_ps[ALIGNED] = PFX(pixel_add_ps_16x16_lsx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].add_ps[ALIGNED] = PFX(pixel_add_ps_32x32_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_4x8].add_ps[ALIGNED]   = PFX(pixel_add_ps_4x8_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].add_ps[ALIGNED]  = PFX(pixel_add_ps_8x16_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].add_ps[ALIGNED] = PFX(pixel_add_ps_16x32_lsx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].add_ps[ALIGNED] = PFX(pixel_add_ps_32x64_lsx);

        // sad_x3
        ALL_LUMA_PU_TYPED(sad_x3, , pixel_sad_x3, lsx);

        // Block_fill
        ALL_LUMA_TU_TYPED(blockfill_s[ALIGNED], , blockfill_s, lsx);
        ALL_LUMA_TU_TYPED(blockfill_s[NONALIGNED], , blockfill_s, lsx);

        // normFact_c
        p.cu[BLOCK_8x8].normFact   = PFX(pixel_normFact_8x8_lsx);
        p.cu[BLOCK_16x16].normFact = PFX(pixel_normFact_16x16_lsx);
        p.cu[BLOCK_32x32].normFact = PFX(pixel_normFact_32x32_lsx);
        p.cu[BLOCK_64x64].normFact = PFX(pixel_normFact_64x64_lsx);

        p.planecopy_cp = PFX(pixel_planecopy_cp_lsx);
        p.planecopy_sp = PFX(pixel_planecopy_sp_lsx);
        p.planecopy_sp_shl = PFX(pixel_planecopy_sp_shl_lsx);
        p.planecopy_pp_shr = PFX(pixel_planecopy_pp_shr_lsx);

        p.scale1D_128to64[NONALIGNED] = PFX(scale1D_128to64_lsx);
        p.scale1D_128to64[ALIGNED] = PFX(scale1D_128to64_lsx);
        p.scale2D_64to32 = PFX(scale2D_64to32_lsx);

        // ssimDist
        p.cu[BLOCK_4x4].ssimDist   = x265_ssimDist4_lsx;
        p.cu[BLOCK_8x8].ssimDist   = x265_ssimDist8_lsx;
        p.cu[BLOCK_16x16].ssimDist = x265_ssimDist16_lsx;
        p.cu[BLOCK_32x32].ssimDist = x265_ssimDist32_lsx;
        p.cu[BLOCK_64x64].ssimDist = x265_ssimDist64_lsx;

        p.ssim_end_4 = PFX(ssim_end4_lsx);
        p.ssim_4x4x2_core = PFX(ssim_4x4x2_lsx);

        // mc
        p.frameInitLowres    = PFX(frame_init_lowres_core_lsx);
        p.frameInitLowerRes  = PFX(frame_init_lowres_core_lsx);
        p.frameSubSampleLuma = PFX(frame_subsample_luma_lsx);

        p.fix8Unpack = PFX(cutree_fix8_unpack_lsx);
        p.fix8Pack   = PFX(cutree_fix8_pack_lsx);

        //copy_cnt
        p.cu[BLOCK_4x4].copy_cnt   = PFX(copy_count_4_lsx);
        p.cu[BLOCK_8x8].copy_cnt   = PFX(copy_count_8_lsx);
        p.cu[BLOCK_16x16].copy_cnt = PFX(copy_count_16_lsx);
        p.cu[BLOCK_32x32].copy_cnt = PFX(copy_count_32_lsx);
    }
#endif /* HAVE_LSX */

#if HAVE_LASX
    if (cpuMask & X265_CPU_LASX)
    {
        // loopfilter
        p.sign = PFX(calSign_lasx);

        // ipfilter
        LUMA(16, 4,  lasx)
        LUMA(16, 8,  lasx)
        LUMA(16, 12, lasx)
        LUMA(16, 16, lasx)
        LUMA(16, 32, lasx)
        LUMA(16, 64, lasx)
        LUMA(24, 32, lasx)
        LUMA(32, 8,  lasx)
        LUMA(32, 16, lasx)
        LUMA(32, 24, lasx)
        LUMA(32, 32, lasx)
        LUMA(32, 64, lasx)
        LUMA(48, 64, lasx)
        LUMA(64, 16, lasx)
        LUMA(64, 32, lasx)
        LUMA(64, 48, lasx)
        LUMA(64, 64, lasx)

        CHROMA_420(16, 4,  lasx)
        CHROMA_420(16, 8,  lasx)
        CHROMA_420(16, 12, lasx)
        CHROMA_420(16, 16, lasx)
        CHROMA_420(16, 32, lasx)
        CHROMA_420(24, 32, lasx)
        CHROMA_420(32, 8,  lasx)
        CHROMA_420(32, 16, lasx)
        CHROMA_420(32, 24, lasx)
        CHROMA_420(32, 32, lasx)

        CHROMA_422(16, 8,  lasx)
        CHROMA_422(16, 16, lasx)
        CHROMA_422(16, 24, lasx)
        CHROMA_422(16, 32, lasx)
        CHROMA_422(16, 64, lasx)
        CHROMA_422(24, 64, lasx)
        CHROMA_422(32, 16, lasx)
        CHROMA_422(32, 32, lasx)
        CHROMA_422(32, 48, lasx)
        CHROMA_422(32, 64, lasx)

        CHROMA_444(16, 4,  lasx)
        CHROMA_444(16, 8,  lasx)
        CHROMA_444(16, 12, lasx)
        CHROMA_444(16, 16, lasx)
        CHROMA_444(16, 32, lasx)
        CHROMA_444(16, 64, lasx)
        CHROMA_444(24, 32, lasx)
        CHROMA_444(32, 8,  lasx)
        CHROMA_444(32, 16, lasx)
        CHROMA_444(32, 24, lasx)
        CHROMA_444(32, 32, lasx)
        CHROMA_444(32, 64, lasx)
        CHROMA_444(48, 64, lasx)
        CHROMA_444(64, 16, lasx)
        CHROMA_444(64, 32, lasx)
        CHROMA_444(64, 48, lasx)
        CHROMA_444(64, 64, lasx)

        //dct
        p.cu[BLOCK_16x16].dct  = PFX(dct16_lasx);
        p.cu[BLOCK_32x32].dct  = PFX(dct32_lasx);
        p.cu[BLOCK_16x16].idct = PFX(idct16_lasx);
        p.cu[BLOCK_32x32].idct = PFX(idct32_lasx);
        p.quant                = PFX(quant_lasx);
        p.dequant_normal       = PFX(dequant_normal_lasx);
        p.dequant_scaling      = PFX(dequant_scaling_lasx);
        p.nquant               = PFX(nquant_lasx);
        p.cu[BLOCK_32x32].count_nonzero = PFX(count_nonzero_32_lasx);
        p.cu[BLOCK_16x16].count_nonzero = PFX(count_nonzero_16_lasx);
        p.cu[BLOCK_8x8].count_nonzero   = PFX(count_nonzero_8_lasx);
        p.cu[BLOCK_4x4].count_nonzero   = PFX(count_nonzero_4_lasx);
        p.findPosFirstLast     = PFX(findPosFirstLast_lasx);
        p.denoiseDct           = PFX(denoiseDct_lasx);

        //pixelAvg_pp
        ASSIGN2(p.pu[LUMA_64x64].pixelavg_pp, pixel_avg_64x64_lasx);
        ASSIGN2(p.pu[LUMA_64x48].pixelavg_pp, pixel_avg_64x48_lasx);
        ASSIGN2(p.pu[LUMA_64x32].pixelavg_pp, pixel_avg_64x32_lasx);
        ASSIGN2(p.pu[LUMA_64x16].pixelavg_pp, pixel_avg_64x16_lasx);
        ASSIGN2(p.pu[LUMA_48x64].pixelavg_pp, pixel_avg_48x64_lasx);
        ASSIGN2(p.pu[LUMA_32x64].pixelavg_pp, pixel_avg_32x64_lasx);
        ASSIGN2(p.pu[LUMA_32x32].pixelavg_pp, pixel_avg_32x32_lasx);
        ASSIGN2(p.pu[LUMA_32x24].pixelavg_pp, pixel_avg_32x24_lasx);
        ASSIGN2(p.pu[LUMA_32x16].pixelavg_pp, pixel_avg_32x16_lasx);
        ASSIGN2(p.pu[LUMA_32x8].pixelavg_pp, pixel_avg_32x8_lasx);

        //addAvg_pp
        ASSIGN2(p.pu[LUMA_24x32].addAvg, addAvg_24x32_lasx);
        ASSIGN2(p.pu[LUMA_32x8].addAvg, addAvg_32x8_lasx);
        ASSIGN2(p.pu[LUMA_32x16].addAvg, addAvg_32x16_lasx);
        ASSIGN2(p.pu[LUMA_32x24].addAvg, addAvg_32x24_lasx);
        ASSIGN2(p.pu[LUMA_32x32].addAvg, addAvg_32x32_lasx);
        ASSIGN2(p.pu[LUMA_32x64].addAvg, addAvg_32x64_lasx);
        ASSIGN2(p.pu[LUMA_48x64].addAvg, addAvg_48x64_lasx);
        ASSIGN2(p.pu[LUMA_64x16].addAvg, addAvg_64x16_lasx);
        ASSIGN2(p.pu[LUMA_64x32].addAvg, addAvg_64x32_lasx);
        ASSIGN2(p.pu[LUMA_64x48].addAvg, addAvg_64x48_lasx);
        ASSIGN2(p.pu[LUMA_64x64].addAvg, addAvg_64x64_lasx);
        ASSIGN2(p.chroma[X265_CSP_I420].pu[CHROMA_420_24x32].addAvg, addAvg_24x32_lasx);
        ASSIGN2(p.chroma[X265_CSP_I420].pu[CHROMA_420_32x8].addAvg, addAvg_32x8_lasx);
        ASSIGN2(p.chroma[X265_CSP_I420].pu[CHROMA_420_32x16].addAvg, addAvg_32x16_lasx);
        ASSIGN2(p.chroma[X265_CSP_I420].pu[CHROMA_420_32x24].addAvg, addAvg_32x24_lasx);
        ASSIGN2(p.chroma[X265_CSP_I420].pu[CHROMA_420_32x32].addAvg, addAvg_32x32_lasx);
        ASSIGN2(p.chroma[X265_CSP_I422].pu[CHROMA_422_24x64].addAvg, addAvg_24x64_lasx);
        ASSIGN2(p.chroma[X265_CSP_I422].pu[CHROMA_422_32x16].addAvg, addAvg_32x16_lasx);
        ASSIGN2(p.chroma[X265_CSP_I422].pu[CHROMA_422_32x32].addAvg, addAvg_32x32_lasx);
        ASSIGN2(p.chroma[X265_CSP_I422].pu[CHROMA_422_32x48].addAvg, addAvg_32x48_lasx);
        ASSIGN2(p.chroma[X265_CSP_I422].pu[CHROMA_422_32x64].addAvg, addAvg_32x64_lasx);

        // intra_filter
        p.cu[BLOCK_4x4].intra_filter = PFX(intra_filter_4x4_lasx);
        p.cu[BLOCK_8x8].intra_filter = PFX(intra_filter_8x8_lasx);
        p.cu[BLOCK_16x16].intra_filter = PFX(intra_filter_16x16_lasx);
        p.cu[BLOCK_32x32].intra_filter = PFX(intra_filter_32x32_lasx);

        //intra_pred
        p.cu[BLOCK_8x8].intra_pred[3] = PFX(intra_pred_ang8_3_lasx);
        p.cu[BLOCK_8x8].intra_pred[4] = PFX(intra_pred_ang8_4_lasx);
        p.cu[BLOCK_8x8].intra_pred[5] = PFX(intra_pred_ang8_5_lasx);
        p.cu[BLOCK_8x8].intra_pred[6] = PFX(intra_pred_ang8_6_lasx);
        p.cu[BLOCK_8x8].intra_pred[7] = PFX(intra_pred_ang8_7_lasx);
        p.cu[BLOCK_8x8].intra_pred[8] = PFX(intra_pred_ang8_8_lasx);
        p.cu[BLOCK_8x8].intra_pred[9] = PFX(intra_pred_ang8_9_lasx);
        p.cu[BLOCK_8x8].intra_pred[11] = PFX(intra_pred_ang8_11_lasx);
        p.cu[BLOCK_8x8].intra_pred[12] = PFX(intra_pred_ang8_12_lasx);
        p.cu[BLOCK_8x8].intra_pred[13] = PFX(intra_pred_ang8_13_lasx);
        p.cu[BLOCK_8x8].intra_pred[14] = PFX(intra_pred_ang8_14_lasx);
        p.cu[BLOCK_8x8].intra_pred[15] = PFX(intra_pred_ang8_15_lasx);
        p.cu[BLOCK_8x8].intra_pred[16] = PFX(intra_pred_ang8_16_lasx);
        p.cu[BLOCK_8x8].intra_pred[17] = PFX(intra_pred_ang8_17_lasx);
        p.cu[BLOCK_8x8].intra_pred[19] = PFX(intra_pred_ang8_19_lasx);
        p.cu[BLOCK_8x8].intra_pred[20] = PFX(intra_pred_ang8_20_lasx);
        p.cu[BLOCK_8x8].intra_pred[21] = PFX(intra_pred_ang8_21_lasx);
        p.cu[BLOCK_8x8].intra_pred[22] = PFX(intra_pred_ang8_22_lasx);
        p.cu[BLOCK_8x8].intra_pred[23] = PFX(intra_pred_ang8_23_lasx);
        p.cu[BLOCK_8x8].intra_pred[24] = PFX(intra_pred_ang8_24_lasx);
        p.cu[BLOCK_8x8].intra_pred[25] = PFX(intra_pred_ang8_25_lasx);
        p.cu[BLOCK_8x8].intra_pred[27] = PFX(intra_pred_ang8_27_lasx);
        p.cu[BLOCK_8x8].intra_pred[28] = PFX(intra_pred_ang8_28_lasx);
        p.cu[BLOCK_8x8].intra_pred[29] = PFX(intra_pred_ang8_29_lasx);
        p.cu[BLOCK_8x8].intra_pred[30] = PFX(intra_pred_ang8_30_lasx);
        p.cu[BLOCK_8x8].intra_pred[31] = PFX(intra_pred_ang8_31_lasx);
        p.cu[BLOCK_8x8].intra_pred[32] = PFX(intra_pred_ang8_32_lasx);
        p.cu[BLOCK_8x8].intra_pred[33] = PFX(intra_pred_ang8_33_lasx);
        p.cu[BLOCK_16x16].intra_pred[3] = PFX(intra_pred_ang16_3_lasx);
        p.cu[BLOCK_16x16].intra_pred[4] = PFX(intra_pred_ang16_4_lasx);
        p.cu[BLOCK_16x16].intra_pred[5] = PFX(intra_pred_ang16_5_lasx);
        p.cu[BLOCK_16x16].intra_pred[6] = PFX(intra_pred_ang16_6_lasx);
        p.cu[BLOCK_16x16].intra_pred[7] = PFX(intra_pred_ang16_7_lasx);
        p.cu[BLOCK_16x16].intra_pred[8] = PFX(intra_pred_ang16_8_lasx);
        p.cu[BLOCK_16x16].intra_pred[9] = PFX(intra_pred_ang16_9_lasx);
        p.cu[BLOCK_16x16].intra_pred[11] = PFX(intra_pred_ang16_11_lasx);
        p.cu[BLOCK_16x16].intra_pred[12] = PFX(intra_pred_ang16_12_lasx);
        p.cu[BLOCK_16x16].intra_pred[13] = PFX(intra_pred_ang16_13_lasx);
        p.cu[BLOCK_16x16].intra_pred[14] = PFX(intra_pred_ang16_14_lasx);
        p.cu[BLOCK_16x16].intra_pred[15] = PFX(intra_pred_ang16_15_lasx);
        p.cu[BLOCK_16x16].intra_pred[16] = PFX(intra_pred_ang16_16_lasx);
        p.cu[BLOCK_16x16].intra_pred[17] = PFX(intra_pred_ang16_17_lasx);

        // luma satd
        p.pu[LUMA_4x8].satd   = PFX(pixel_satd_4x8_lasx);
        p.pu[LUMA_4x16].satd  = PFX(pixel_satd_4x16_lasx);
        p.pu[LUMA_8x8].satd   = PFX(pixel_satd_8x8_lasx);
        p.pu[LUMA_8x16].satd  = PFX(pixel_satd_8x16_lasx);
        p.pu[LUMA_8x32].satd  = PFX(pixel_satd_8x32_lasx);
        p.pu[LUMA_12x16].satd = PFX(pixel_satd_12x16_lasx);
        p.pu[LUMA_16x8].satd  = PFX(pixel_satd_16x8_lasx);
        p.pu[LUMA_16x16].satd = PFX(pixel_satd_16x16_lasx);
        p.pu[LUMA_16x32].satd = PFX(pixel_satd_16x32_lasx);
        p.pu[LUMA_16x64].satd = PFX(pixel_satd_16x64_lasx);
        p.pu[LUMA_24x32].satd = PFX(pixel_satd_24x32_lasx);
        p.pu[LUMA_32x8].satd  = PFX(pixel_satd_32x8_lasx);
        p.pu[LUMA_32x16].satd = PFX(pixel_satd_32x16_lasx);
        p.pu[LUMA_32x24].satd = PFX(pixel_satd_32x24_lasx);
        p.pu[LUMA_32x32].satd = PFX(pixel_satd_32x32_lasx);
        p.pu[LUMA_32x64].satd = PFX(pixel_satd_32x64_lasx);
        p.pu[LUMA_48x64].satd = PFX(pixel_satd_48x64_lasx);
        p.pu[LUMA_64x16].satd = PFX(pixel_satd_64x16_lasx);
        p.pu[LUMA_64x32].satd = PFX(pixel_satd_64x32_lasx);
        p.pu[LUMA_64x48].satd = PFX(pixel_satd_64x48_lasx);
        p.pu[LUMA_64x64].satd = PFX(pixel_satd_64x64_lasx);

        // chroma satd
        p.chroma[X265_CSP_I420].pu[CHROMA_420_4x8].satd   = PFX(pixel_satd_4x8_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_4x16].satd  = PFX(pixel_satd_4x16_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_8x16].satd  = PFX(pixel_satd_8x16_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_8x8].satd   = PFX(pixel_satd_8x8_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_16x8].satd  = PFX(pixel_satd_16x8_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_16x16].satd = PFX(pixel_satd_16x16_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_8x32].satd  = PFX(pixel_satd_8x32_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_16x32].satd = PFX(pixel_satd_16x32_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_16x32].satd = PFX(pixel_satd_16x64_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_32x8].satd  = PFX(pixel_satd_32x8_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_32x16].satd = PFX(pixel_satd_32x16_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_32x24].satd = PFX(pixel_satd_32x24_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_32x32].satd = PFX(pixel_satd_32x32_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_8x32].satd  = PFX(pixel_satd_8x32_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_12x16].satd = PFX(pixel_satd_12x16_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_24x32].satd = PFX(pixel_satd_24x32_lasx);

        p.chroma[X265_CSP_I422].pu[CHROMA_422_4x8].satd   = PFX(pixel_satd_4x8_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_4x16].satd  = PFX(pixel_satd_4x16_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_8x8].satd   = PFX(pixel_satd_8x8_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_8x16].satd  = PFX(pixel_satd_8x16_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_16x8].satd  = PFX(pixel_satd_16x8_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_16x16].satd = PFX(pixel_satd_16x16_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_8x32].satd  = PFX(pixel_satd_8x32_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_8x64].satd  = PFX(pixel_satd_8x64_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_16x24].satd = PFX(pixel_satd_16x24_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_16x32].satd = PFX(pixel_satd_16x32_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_16x64].satd = PFX(pixel_satd_16x64_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_32x16].satd = PFX(pixel_satd_32x16_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_32x32].satd = PFX(pixel_satd_32x32_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_32x64].satd = PFX(pixel_satd_32x64_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_32x48].satd = PFX(pixel_satd_32x48_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_8x32].satd  = PFX(pixel_satd_8x32_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_8x64].satd  = PFX(pixel_satd_8x64_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_4x32].satd  = PFX(pixel_satd_4x32_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_12x32].satd = PFX(pixel_satd_12x32_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_24x64].satd = PFX(pixel_satd_24x64_lasx);

        p.pu[LUMA_24x32].sad = x265_pixel_sad_24x32_lasx;
        p.pu[LUMA_32x8].sad  = x265_pixel_sad_32x8_lasx;
        p.pu[LUMA_32x16].sad = x265_pixel_sad_32x16_lasx;
        p.pu[LUMA_32x24].sad = x265_pixel_sad_32x24_lasx;
        p.pu[LUMA_32x32].sad = x265_pixel_sad_32x32_lasx;
        p.pu[LUMA_32x64].sad = x265_pixel_sad_32x64_lasx;
        p.pu[LUMA_48x64].sad = x265_pixel_sad_48x64_lasx;
        p.pu[LUMA_64x16].sad = x265_pixel_sad_64x16_lasx;
        p.pu[LUMA_64x32].sad = x265_pixel_sad_64x32_lasx;
        p.pu[LUMA_64x48].sad = x265_pixel_sad_64x48_lasx;
        p.pu[LUMA_64x64].sad = x265_pixel_sad_64x64_lasx;

        // sad_x4
        p.pu[LUMA_8x4].sad_x4   = x265_pixel_sad_x4_8x4_lasx;
        p.pu[LUMA_8x8].sad_x4   = x265_pixel_sad_x4_8x8_lasx;
        p.pu[LUMA_8x16].sad_x4  = x265_pixel_sad_x4_8x16_lasx;
        p.pu[LUMA_8x32].sad_x4  = x265_pixel_sad_x4_8x32_lasx;
        p.pu[LUMA_12x16].sad_x4 = x265_pixel_sad_x4_12x16_lasx;
        p.pu[LUMA_16x4].sad_x4  = x265_pixel_sad_x4_16x4_lasx;
        p.pu[LUMA_16x8].sad_x4  = x265_pixel_sad_x4_16x8_lasx;
        p.pu[LUMA_16x12].sad_x4 = x265_pixel_sad_x4_16x12_lasx;
        p.pu[LUMA_16x16].sad_x4 = x265_pixel_sad_x4_16x16_lasx;
        p.pu[LUMA_16x32].sad_x4 = x265_pixel_sad_x4_16x32_lasx;
        p.pu[LUMA_16x64].sad_x4 = x265_pixel_sad_x4_16x64_lasx;
        p.pu[LUMA_24x32].sad_x4 = x265_pixel_sad_x4_24x32_lasx;
        p.pu[LUMA_32x8].sad_x4  = x265_pixel_sad_x4_32x8_lasx;
        p.pu[LUMA_32x16].sad_x4 = x265_pixel_sad_x4_32x16_lasx;
        p.pu[LUMA_32x24].sad_x4 = x265_pixel_sad_x4_32x24_lasx;
        p.pu[LUMA_32x32].sad_x4 = x265_pixel_sad_x4_32x32_lasx;
        p.pu[LUMA_32x64].sad_x4 = x265_pixel_sad_x4_32x64_lasx;
        p.pu[LUMA_64x16].sad_x4 = x265_pixel_sad_x4_64x16_lasx;
        p.pu[LUMA_64x32].sad_x4 = x265_pixel_sad_x4_64x32_lasx;
        p.pu[LUMA_64x48].sad_x4 = x265_pixel_sad_x4_64x48_lasx;
        p.pu[LUMA_64x64].sad_x4 = x265_pixel_sad_x4_64x64_lasx;
        p.pu[LUMA_48x64].sad_x4 = x265_pixel_sad_x4_48x64_lasx;

        // sad_x3
        p.pu[LUMA_12x16].sad_x3 = x265_pixel_sad_x3_12x16_lasx;
        p.pu[LUMA_16x8].sad_x3  = x265_pixel_sad_x3_16x8_lasx;
        p.pu[LUMA_16x12].sad_x3 = x265_pixel_sad_x3_16x12_lasx;
        p.pu[LUMA_16x16].sad_x3 = x265_pixel_sad_x3_16x16_lasx;
        p.pu[LUMA_16x32].sad_x3 = x265_pixel_sad_x3_16x32_lasx;
        p.pu[LUMA_16x64].sad_x3 = x265_pixel_sad_x3_16x64_lasx;
        p.pu[LUMA_24x32].sad_x3 = x265_pixel_sad_x3_24x32_lasx;
        p.pu[LUMA_32x8].sad_x3  = x265_pixel_sad_x3_32x8_lasx;
        p.pu[LUMA_32x24].sad_x3 = x265_pixel_sad_x3_32x24_lasx;
        p.pu[LUMA_32x16].sad_x3 = x265_pixel_sad_x3_32x16_lasx;
        p.pu[LUMA_32x32].sad_x3 = x265_pixel_sad_x3_32x32_lasx;
        p.pu[LUMA_32x64].sad_x3 = x265_pixel_sad_x3_32x64_lasx;
        p.pu[LUMA_64x16].sad_x3 = x265_pixel_sad_x3_64x16_lasx;
        p.pu[LUMA_64x32].sad_x3 = x265_pixel_sad_x3_64x32_lasx;
        p.pu[LUMA_64x48].sad_x3 = x265_pixel_sad_x3_64x48_lasx;
        p.pu[LUMA_64x64].sad_x3 = x265_pixel_sad_x3_64x64_lasx;
        p.pu[LUMA_48x64].sad_x3 = x265_pixel_sad_x3_48x64_lasx;

        // sa8d
        p.cu[BLOCK_8x8].sa8d   = PFX(pixel_sa8d_8x8_lasx);
        p.cu[BLOCK_16x16].sa8d = PFX(pixel_sa8d_16x16_lasx);
        p.cu[BLOCK_32x32].sa8d = PFX(pixel_sa8d_32x32_lasx);
        p.cu[BLOCK_64x64].sa8d = PFX(pixel_sa8d_64x64_lasx);
        p.chroma[X265_CSP_I420].cu[BLOCK_16x16].sa8d     = PFX(pixel_sa8d_16x16_lasx);
        p.chroma[X265_CSP_I420].cu[BLOCK_32x32].sa8d     = PFX(pixel_sa8d_32x32_lasx);
        p.chroma[X265_CSP_I420].cu[BLOCK_64x64].sa8d     = PFX(pixel_sa8d_64x64_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_8x8].sa8d       = PFX(pixel_satd_4x8_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].sa8d  = PFX(pixel_sa8d_8x16_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].sa8d = PFX(pixel_sa8d_16x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].sa8d = PFX(pixel_sa8d_32x64_lasx);

        // psy_cost_pp
        p.cu[BLOCK_4x4].psy_cost_pp   = psyCost_pp_lasx<BLOCK_4x4>;
        p.cu[BLOCK_8x8].psy_cost_pp   = psyCost_pp_lasx<BLOCK_8x8>;
        p.cu[BLOCK_16x16].psy_cost_pp = psyCost_pp_lasx<BLOCK_16x16>;
        p.cu[BLOCK_32x32].psy_cost_pp = psyCost_pp_lasx<BLOCK_32x32>;
        p.cu[BLOCK_64x64].psy_cost_pp = psyCost_pp_lasx<BLOCK_64x64>;

        p.cu[BLOCK_32x32].sub_ps = PFX(pixel_sub_ps_32x32_lasx);
        p.cu[BLOCK_64x64].sub_ps = PFX(pixel_sub_ps_64x64_lasx);

        // chroma sub_ps
        p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].sub_ps = PFX(pixel_sub_ps_32x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].sub_ps = PFX(pixel_sub_ps_32x64_lasx);

        p.cu[BLOCK_16x16].add_ps[NONALIGNED] = PFX(pixel_add_ps_16x16_lasx);
        p.cu[BLOCK_32x32].add_ps[NONALIGNED] = PFX(pixel_add_ps_32x32_lasx);
        p.cu[BLOCK_64x64].add_ps[NONALIGNED] = PFX(pixel_add_ps_64x64_lsx);

        p.cu[BLOCK_16x16].add_ps[ALIGNED] = PFX(pixel_add_ps_16x16_lasx);
        p.cu[BLOCK_32x32].add_ps[ALIGNED] = PFX(pixel_add_ps_32x32_lasx);
        p.cu[BLOCK_64x64].add_ps[ALIGNED] = PFX(pixel_add_ps_64x64_lasx);

        // chroma add_ps
        p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].add_ps[NONALIGNED] = PFX(pixel_add_ps_16x16_lasx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].add_ps[NONALIGNED] = PFX(pixel_add_ps_32x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].add_ps[NONALIGNED] = PFX(pixel_add_ps_16x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].add_ps[NONALIGNED] = PFX(pixel_add_ps_32x64_lasx);

        p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].add_ps[ALIGNED] = PFX(pixel_add_ps_16x16_lasx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].add_ps[ALIGNED] = PFX(pixel_add_ps_32x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].add_ps[ALIGNED] = PFX(pixel_add_ps_16x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].add_ps[ALIGNED] = PFX(pixel_add_ps_32x64_lasx);

        // var
        p.cu[BLOCK_8x8].var   = PFX(pixel_var_8x8_lasx);
        p.cu[BLOCK_16x16].var = PFX(pixel_var_16x16_lasx);
        p.cu[BLOCK_32x32].var = PFX(pixel_var_32x32_lasx);
        p.cu[BLOCK_64x64].var = PFX(pixel_var_64x64_lasx);

        // sse_pp
        p.cu[BLOCK_16x16].sse_pp = PFX(pixel_sse_16x16_lasx);
        p.cu[BLOCK_32x32].sse_pp = PFX(pixel_sse_32x32_lasx);
        p.cu[BLOCK_64x64].sse_pp = PFX(pixel_sse_64x64_lasx);

        p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].sse_pp = PFX(pixel_sse_16x16_lasx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].sse_pp = PFX(pixel_sse_32x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].sse_pp = PFX(pixel_sse_16x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].sse_pp = PFX(pixel_sse_32x64_lasx);

        // sse_ss
        p.cu[BLOCK_8x8].sse_ss = PFX(pixel_sse_ss_8x8_lasx);
        p.cu[BLOCK_16x16].sse_ss = PFX(pixel_sse_ss_16x16_lasx);
        p.cu[BLOCK_32x32].sse_ss = PFX(pixel_sse_ss_32x32_lasx);
        p.cu[BLOCK_64x64].sse_ss = PFX(pixel_sse_ss_64x64_lasx);

        // ssd_s
        p.cu[BLOCK_16x16].ssd_s[NONALIGNED] = PFX(pixel_ssd_s_16x16_lasx);
        p.cu[BLOCK_16x16].ssd_s[ALIGNED]    = PFX(pixel_ssd_s_16x16_lasx);
        p.cu[BLOCK_32x32].ssd_s[NONALIGNED] = PFX(pixel_ssd_s_32x32_lasx);
        p.cu[BLOCK_32x32].ssd_s[ALIGNED]    = PFX(pixel_ssd_s_32x32_lasx);
        p.cu[BLOCK_64x64].ssd_s[NONALIGNED] = PFX(pixel_ssd_s_64x64_lasx);
        p.cu[BLOCK_64x64].ssd_s[ALIGNED]    = PFX(pixel_ssd_s_64x64_lasx);

        // calcresidual
        p.cu[BLOCK_64x64].calcresidual[NONALIGNED] = PFX(pixel_getResidual_64x64_lasx);
        p.cu[BLOCK_32x32].calcresidual[NONALIGNED] = PFX(pixel_getResidual_32x32_lasx);
        p.cu[BLOCK_64x64].calcresidual[ALIGNED] = PFX(pixel_getResidual_64x64_lasx);
        p.cu[BLOCK_32x32].calcresidual[ALIGNED] = PFX(pixel_getResidual_32x32_lasx);

        // normFact_c
        p.cu[BLOCK_8x8].normFact   = PFX(pixel_normFact_8x8_lasx);
        p.cu[BLOCK_16x16].normFact = PFX(pixel_normFact_16x16_lasx);
        p.cu[BLOCK_32x32].normFact = PFX(pixel_normFact_32x32_lasx);
        p.cu[BLOCK_64x64].normFact = PFX(pixel_normFact_64x64_lasx);

        p.scale1D_128to64[NONALIGNED] = PFX(scale1D_128to64_lasx);
        p.scale1D_128to64[ALIGNED] = PFX(scale1D_128to64_lasx);
        p.scale2D_64to32 = PFX(scale2D_64to32_lasx);

        // ssimDist
        p.cu[BLOCK_8x8].ssimDist   = x265_ssimDist8_lasx;
        p.cu[BLOCK_16x16].ssimDist = x265_ssimDist16_lasx;
        p.cu[BLOCK_32x32].ssimDist = x265_ssimDist32_lasx;
        p.cu[BLOCK_64x64].ssimDist = x265_ssimDist64_lasx;

        // mc
        p.frameInitLowres    = PFX(frame_init_lowres_core_lasx);
        p.frameInitLowerRes  = PFX(frame_init_lowres_core_lasx);
        p.frameSubSampleLuma = PFX(frame_subsample_luma_lasx);

        p.fix8Unpack = PFX(cutree_fix8_unpack_lasx);
        p.fix8Pack   = PFX(cutree_fix8_pack_lasx);

        // Block_fill
        p.cu[BLOCK_16x16].blockfill_s[NONALIGNED] = PFX(blockfill_s_16x16_lasx);
        p.cu[BLOCK_32x32].blockfill_s[NONALIGNED] = PFX(blockfill_s_32x32_lasx);
        p.cu[BLOCK_64x64].blockfill_s[NONALIGNED] = PFX(blockfill_s_64x64_lasx);

        p.cu[BLOCK_16x16].blockfill_s[ALIGNED] = PFX(blockfill_s_16x16_lasx);
        p.cu[BLOCK_32x32].blockfill_s[ALIGNED] = PFX(blockfill_s_32x32_lasx);
        p.cu[BLOCK_64x64].blockfill_s[ALIGNED] = PFX(blockfill_s_64x64_lasx);

        //copy_pp
        p.pu[LUMA_24x32].copy_pp = PFX(blockcopy_pp_24x32_lasx);
        p.pu[LUMA_32x8].copy_pp = PFX(blockcopy_pp_32x8_lasx);
        p.pu[LUMA_32x16].copy_pp = PFX(blockcopy_pp_32x16_lasx);
        p.pu[LUMA_32x24].copy_pp = PFX(blockcopy_pp_32x24_lasx);
        p.pu[LUMA_32x32].copy_pp = PFX(blockcopy_pp_32x32_lasx);
        p.pu[LUMA_32x64].copy_pp = PFX(blockcopy_pp_32x64_lasx);
        p.pu[LUMA_48x64].copy_pp = PFX(blockcopy_pp_48x64_lasx);
        p.pu[LUMA_64x16].copy_pp = PFX(blockcopy_pp_64x16_lasx);
        p.pu[LUMA_64x32].copy_pp = PFX(blockcopy_pp_64x32_lasx);
        p.pu[LUMA_64x48].copy_pp = PFX(blockcopy_pp_64x48_lasx);
        p.pu[LUMA_64x64].copy_pp = PFX(blockcopy_pp_64x64_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_24x32].copy_pp = PFX(blockcopy_pp_24x32_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_32x8].copy_pp = PFX(blockcopy_pp_32x8_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_32x16].copy_pp = PFX(blockcopy_pp_32x16_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_32x24].copy_pp = PFX(blockcopy_pp_32x24_lasx);
        p.chroma[X265_CSP_I420].pu[CHROMA_420_32x32].copy_pp = PFX(blockcopy_pp_32x32_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_24x64].copy_pp = PFX(blockcopy_pp_24x64_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_32x16].copy_pp = PFX(blockcopy_pp_32x16_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_32x32].copy_pp = PFX(blockcopy_pp_32x32_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_32x48].copy_pp = PFX(blockcopy_pp_32x48_lasx);
        p.chroma[X265_CSP_I422].pu[CHROMA_422_32x64].copy_pp = PFX(blockcopy_pp_32x64_lasx);

        //copy_ss
        p.cu[BLOCK_16x16].copy_ss = PFX(blockcopy_ss_16x16_lasx);
        p.cu[BLOCK_32x32].copy_ss = PFX(blockcopy_ss_32x32_lasx);
        p.cu[BLOCK_64x64].copy_ss = PFX(blockcopy_ss_64x64_lasx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].copy_ss = PFX(blockcopy_ss_16x16_lasx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].copy_ss = PFX(blockcopy_ss_32x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].copy_ss = PFX(blockcopy_ss_16x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].copy_ss = PFX(blockcopy_ss_32x64_lasx);
        //copy_sp
        p.cu[BLOCK_16x16].copy_sp = PFX(blockcopy_sp_16x16_lasx);
        p.cu[BLOCK_32x32].copy_sp = PFX(blockcopy_sp_32x32_lasx);
        p.cu[BLOCK_64x64].copy_sp = PFX(blockcopy_sp_64x64_lasx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].copy_sp = PFX(blockcopy_sp_16x16_lasx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].copy_sp = PFX(blockcopy_sp_32x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].copy_sp = PFX(blockcopy_sp_16x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].copy_sp = PFX(blockcopy_sp_32x64_lasx);
        //copy_ps
        p.cu[BLOCK_16x16].copy_ps = PFX(blockcopy_ps_16x16_lasx);
        p.cu[BLOCK_32x32].copy_ps = PFX(blockcopy_ps_32x32_lasx);
        p.cu[BLOCK_64x64].copy_ps = PFX(blockcopy_ps_64x64_lasx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].copy_ps = PFX(blockcopy_ps_16x16_lasx);
        p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].copy_ps = PFX(blockcopy_ps_32x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].copy_ps = PFX(blockcopy_ps_16x32_lasx);
        p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].copy_ps = PFX(blockcopy_ps_32x64_lasx);
        //copy_cnt
        p.cu[BLOCK_16x16].copy_cnt = PFX(copy_count_16_lasx);
        p.cu[BLOCK_32x32].copy_cnt = PFX(copy_count_32_lasx);

        // cpy2Dto1D_shl
        p.cu[BLOCK_8x8].cpy2Dto1D_shl   = PFX(cpy2Dto1D_shl_8x8_lasx);
        p.cu[BLOCK_16x16].cpy2Dto1D_shl = PFX(cpy2Dto1D_shl_16x16_lasx);
        p.cu[BLOCK_32x32].cpy2Dto1D_shl = PFX(cpy2Dto1D_shl_32x32_lasx);
        p.cu[BLOCK_64x64].cpy2Dto1D_shl = PFX(cpy2Dto1D_shl_64x64_lasx);

        // cpy2Dto1D_shr
        p.cu[BLOCK_8x8].cpy2Dto1D_shr   = PFX(cpy2Dto1D_shr_8x8_lasx);
        p.cu[BLOCK_16x16].cpy2Dto1D_shr = PFX(cpy2Dto1D_shr_16x16_lasx);
        p.cu[BLOCK_32x32].cpy2Dto1D_shr = PFX(cpy2Dto1D_shr_32x32_lasx);
        p.cu[BLOCK_64x64].cpy2Dto1D_shr = PFX(cpy2Dto1D_shr_64x64_lasx);

        // cpy1Dto2D_shl
        p.cu[BLOCK_8x8].cpy1Dto2D_shl[NONALIGNED]   = PFX(cpy1Dto2D_shl_8x8_lasx);
        p.cu[BLOCK_16x16].cpy1Dto2D_shl[NONALIGNED] = PFX(cpy1Dto2D_shl_16x16_lasx);
        p.cu[BLOCK_32x32].cpy1Dto2D_shl[NONALIGNED] = PFX(cpy1Dto2D_shl_32x32_lasx);
        p.cu[BLOCK_64x64].cpy1Dto2D_shl[NONALIGNED] = PFX(cpy1Dto2D_shl_64x64_lasx);

        p.cu[BLOCK_8x8].cpy1Dto2D_shl[ALIGNED]   = PFX(cpy1Dto2D_shl_8x8_lasx);
        p.cu[BLOCK_16x16].cpy1Dto2D_shl[ALIGNED] = PFX(cpy1Dto2D_shl_16x16_lasx);
        p.cu[BLOCK_32x32].cpy1Dto2D_shl[ALIGNED] = PFX(cpy1Dto2D_shl_32x32_lasx);
        p.cu[BLOCK_64x64].cpy1Dto2D_shl[ALIGNED] = PFX(cpy1Dto2D_shl_64x64_lasx);

        // cpy1Dto2D_shr
        p.cu[BLOCK_8x8].cpy1Dto2D_shr   = PFX(cpy1Dto2D_shr_8x8_lasx);
        p.cu[BLOCK_16x16].cpy1Dto2D_shr = PFX(cpy1Dto2D_shr_16x16_lasx);
        p.cu[BLOCK_32x32].cpy1Dto2D_shr = PFX(cpy1Dto2D_shr_32x32_lasx);
        p.cu[BLOCK_64x64].cpy1Dto2D_shr = PFX(cpy1Dto2D_shr_64x64_lasx);
    }
#endif /* HAVE_LASX */
}

}
