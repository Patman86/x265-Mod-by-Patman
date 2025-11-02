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

#define ALL_LUMA_PU_TYPED(prim, fncdef, fname, cpu)                            \
  p.pu[LUMA_4x4].prim = fncdef PFX(fname##_4x4_##cpu);                         \
  p.pu[LUMA_8x8].prim = fncdef PFX(fname##_8x8_##cpu);                         \
  p.pu[LUMA_16x16].prim = fncdef PFX(fname##_16x16_##cpu);                     \
  p.pu[LUMA_32x32].prim = fncdef PFX(fname##_32x32_##cpu);                     \
  p.pu[LUMA_64x64].prim = fncdef PFX(fname##_64x64_##cpu);                     \
  p.pu[LUMA_8x4].prim = fncdef PFX(fname##_8x4_##cpu);                         \
  p.pu[LUMA_4x8].prim = fncdef PFX(fname##_4x8_##cpu);                         \
  p.pu[LUMA_16x8].prim = fncdef PFX(fname##_16x8_##cpu);                       \
  p.pu[LUMA_8x16].prim = fncdef PFX(fname##_8x16_##cpu);                       \
  p.pu[LUMA_16x32].prim = fncdef PFX(fname##_16x32_##cpu);                     \
  p.pu[LUMA_32x16].prim = fncdef PFX(fname##_32x16_##cpu);                     \
  p.pu[LUMA_64x32].prim = fncdef PFX(fname##_64x32_##cpu);                     \
  p.pu[LUMA_32x64].prim = fncdef PFX(fname##_32x64_##cpu);                     \
  p.pu[LUMA_16x12].prim = fncdef PFX(fname##_16x12_##cpu);                     \
  p.pu[LUMA_12x16].prim = fncdef PFX(fname##_12x16_##cpu);                     \
  p.pu[LUMA_16x4].prim = fncdef PFX(fname##_16x4_##cpu);                       \
  p.pu[LUMA_4x16].prim = fncdef PFX(fname##_4x16_##cpu);                       \
  p.pu[LUMA_32x24].prim = fncdef PFX(fname##_32x24_##cpu);                     \
  p.pu[LUMA_24x32].prim = fncdef PFX(fname##_24x32_##cpu);                     \
  p.pu[LUMA_32x8].prim = fncdef PFX(fname##_32x8_##cpu);                       \
  p.pu[LUMA_8x32].prim = fncdef PFX(fname##_8x32_##cpu);                       \
  p.pu[LUMA_64x48].prim = fncdef PFX(fname##_64x48_##cpu);                     \
  p.pu[LUMA_48x64].prim = fncdef PFX(fname##_48x64_##cpu);                     \
  p.pu[LUMA_64x16].prim = fncdef PFX(fname##_64x16_##cpu);                     \
  p.pu[LUMA_16x64].prim = fncdef PFX(fname##_16x64_##cpu)
#define ALL_LUMA_PU(prim, fname, cpu) ALL_LUMA_PU_TYPED(prim, , fname, cpu)

#define ALL_CHROMA_422_PU_TYPED(prim, fncdef, fname, cpu)               \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x8].prim   = fncdef PFX(fname ## _4x8_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x16].prim  = fncdef PFX(fname ## _8x16_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x32].prim = fncdef PFX(fname ## _16x32_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x64].prim = fncdef PFX(fname ## _32x64_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x4].prim   = fncdef PFX(fname ## _4x4_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_2x8].prim   = fncdef PFX(fname ## _2x8_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x8].prim   = fncdef PFX(fname ## _8x8_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x16].prim  = fncdef PFX(fname ## _4x16_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x16].prim = fncdef PFX(fname ## _16x16_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x32].prim  = fncdef PFX(fname ## _8x32_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x32].prim = fncdef PFX(fname ## _32x32_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x64].prim = fncdef PFX(fname ## _16x64_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x12].prim  = fncdef PFX(fname ## _8x12_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_6x16].prim  = fncdef PFX(fname ## _6x16_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x4].prim   = fncdef PFX(fname ## _8x4_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_2x16].prim  = fncdef PFX(fname ## _2x16_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x24].prim = fncdef PFX(fname ## _16x24_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_12x32].prim = fncdef PFX(fname ## _12x32_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x8].prim  = fncdef PFX(fname ## _16x8_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x32].prim  = fncdef PFX(fname ## _4x32_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x48].prim = fncdef PFX(fname ## _32x48_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_24x64].prim = fncdef PFX(fname ## _24x64_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x16].prim = fncdef PFX(fname ## _32x16_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x64].prim  = fncdef PFX(fname ## _8x64_ ## cpu)
#define ALL_CHROMA_422_PU(prim, fname, cpu) ALL_CHROMA_422_PU_TYPED(prim, , fname, cpu)

#define LUMA_CU(W, H) \
    p.cu[BLOCK_ ## W ## x ## H].sub_ps        = pixel_sub_ps_c<W, H>;

#if defined(__GNUC__)
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif

#include "pixel-prim.h"

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

    // ssd_s
    p.cu[BLOCK_4x4].ssd_s[NONALIGNED]   = PFX(pixel_ssd_s_4x4_rvv);
    p.cu[BLOCK_8x8].ssd_s[NONALIGNED]   = PFX(pixel_ssd_s_8x8_rvv);
    p.cu[BLOCK_16x16].ssd_s[NONALIGNED]   = PFX(pixel_ssd_s_16x16_rvv);
    p.cu[BLOCK_32x32].ssd_s[NONALIGNED]   = PFX(pixel_ssd_s_32x32_rvv);
    p.cu[BLOCK_64x64].ssd_s[NONALIGNED]   = PFX(pixel_ssd_s_64x64_rvv);

    p.cu[BLOCK_4x4].ssd_s[ALIGNED]   = PFX(pixel_ssd_s_4x4_rvv);
    p.cu[BLOCK_8x8].ssd_s[ALIGNED]   = PFX(pixel_ssd_s_8x8_rvv);
    p.cu[BLOCK_16x16].ssd_s[ALIGNED] = PFX(pixel_ssd_s_16x16_rvv);
    p.cu[BLOCK_32x32].ssd_s[ALIGNED] = PFX(pixel_ssd_s_32x32_rvv);
    p.cu[BLOCK_64x64].ssd_s[ALIGNED] = PFX(pixel_ssd_s_64x64_rvv);

    // sse_pp
    p.cu[BLOCK_4x4].sse_pp   = PFX(pixel_sse_pp_4x4_rvv);
    p.cu[BLOCK_8x8].sse_pp   = PFX(pixel_sse_pp_8x8_rvv);
    p.cu[BLOCK_16x16].sse_pp = PFX(pixel_sse_pp_16x16_rvv);
    p.cu[BLOCK_32x32].sse_pp = PFX(pixel_sse_pp_32x32_rvv);
    p.cu[BLOCK_64x64].sse_pp = PFX(pixel_sse_pp_64x64_rvv);

    p.chroma[X265_CSP_I420].cu[BLOCK_420_4x4].sse_pp   = PFX(pixel_sse_pp_4x4_rvv);
    p.chroma[X265_CSP_I420].cu[BLOCK_420_8x8].sse_pp   = PFX(pixel_sse_pp_8x8_rvv);
    p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].sse_pp = PFX(pixel_sse_pp_16x16_rvv);
    p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].sse_pp = PFX(pixel_sse_pp_32x32_rvv);
    p.chroma[X265_CSP_I422].cu[BLOCK_422_4x8].sse_pp   = PFX(pixel_sse_pp_4x8_rvv);
    p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].sse_pp  = PFX(pixel_sse_pp_8x16_rvv);
    p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].sse_pp = PFX(pixel_sse_pp_16x32_rvv);
    p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].sse_pp = PFX(pixel_sse_pp_32x64_rvv);

    // sse_ss
    p.cu[BLOCK_4x4].sse_ss   = PFX(pixel_sse_ss_4x4_rvv);
    p.cu[BLOCK_8x8].sse_ss   = PFX(pixel_sse_ss_8x8_rvv);
    p.cu[BLOCK_16x16].sse_ss = PFX(pixel_sse_ss_16x16_rvv);
    p.cu[BLOCK_32x32].sse_ss = PFX(pixel_sse_ss_32x32_rvv);
    p.cu[BLOCK_64x64].sse_ss = PFX(pixel_sse_ss_64x64_rvv);

#if !HIGH_BIT_DEPTH
    // sad
    ALL_LUMA_PU(sad, pixel_sad, rvv);
    ALL_LUMA_PU(sad_x3, sad_x3, rvv);
    ALL_LUMA_PU(sad_x4, sad_x4, rvv);

    // pixel_avg_pp
    ALL_LUMA_PU(pixelavg_pp[NONALIGNED], pixel_avg_pp, rvv);
    ALL_LUMA_PU(pixelavg_pp[ALIGNED], pixel_avg_pp, rvv);

    // ssimDist
    p.cu[BLOCK_4x4].ssimDist = PFX(ssimDist4_v);
    p.cu[BLOCK_8x8].ssimDist = PFX(ssimDist8_v);
    p.cu[BLOCK_16x16].ssimDist = PFX(ssimDist16_v);
    p.cu[BLOCK_32x32].ssimDist = PFX(ssimDist32_v);
    p.cu[BLOCK_64x64].ssimDist = PFX(ssimDist64_v);

    // normFact
    p.cu[BLOCK_8x8].normFact = PFX(normFact_v);
    p.cu[BLOCK_16x16].normFact = PFX(normFact_v);
    p.cu[BLOCK_32x32].normFact = PFX(normFact_v);
    p.cu[BLOCK_64x64].normFact = PFX(normFact_v);
#endif

    // pixel_var
    p.cu[BLOCK_8x8].var   = PFX(pixel_var_8x8_v);
    p.cu[BLOCK_16x16].var = PFX(pixel_var_16x16_v);
    p.cu[BLOCK_32x32].var = PFX(pixel_var_32x32_v);
    p.cu[BLOCK_64x64].var = PFX(pixel_var_64x64_v);

    // calc_Residual
    p.cu[BLOCK_4x4].calcresidual[NONALIGNED]   = PFX(getResidual4_v);
    p.cu[BLOCK_8x8].calcresidual[NONALIGNED]   = PFX(getResidual8_v);
    p.cu[BLOCK_16x16].calcresidual[NONALIGNED] = PFX(getResidual16_v);
    p.cu[BLOCK_32x32].calcresidual[NONALIGNED] = PFX(getResidual32_v);

    p.cu[BLOCK_4x4].calcresidual[ALIGNED]   = PFX(getResidual4_v);
    p.cu[BLOCK_8x8].calcresidual[ALIGNED]   = PFX(getResidual8_v);
    p.cu[BLOCK_16x16].calcresidual[ALIGNED] = PFX(getResidual16_v);
    p.cu[BLOCK_32x32].calcresidual[ALIGNED] = PFX(getResidual32_v);

    // pixel_sub_ps
    p.cu[BLOCK_4x4].sub_ps   = PFX(pixel_sub_ps_4x4_v);
    p.cu[BLOCK_8x8].sub_ps   = PFX(pixel_sub_ps_8x8_v);
    p.cu[BLOCK_16x16].sub_ps = PFX(pixel_sub_ps_16x16_v);
    p.cu[BLOCK_32x32].sub_ps = PFX(pixel_sub_ps_32x32_v);
    p.cu[BLOCK_64x64].sub_ps = PFX(pixel_sub_ps_64x64_v);

    // chroma sub_ps
    p.chroma[X265_CSP_I420].cu[BLOCK_420_4x4].sub_ps   = PFX(pixel_sub_ps_4x4_v);
    p.chroma[X265_CSP_I420].cu[BLOCK_420_8x8].sub_ps   = PFX(pixel_sub_ps_8x8_v);
    p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].sub_ps = PFX(pixel_sub_ps_16x16_v);
    p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].sub_ps = PFX(pixel_sub_ps_32x32_v);

    p.chroma[X265_CSP_I422].cu[BLOCK_422_4x8].sub_ps   = PFX(pixel_sub_ps_4x8_v);
    p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].sub_ps  = PFX(pixel_sub_ps_8x16_v);
    p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].sub_ps = PFX(pixel_sub_ps_16x32_v);
    p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].sub_ps = PFX(pixel_sub_ps_32x64_v);

    // pixel_add_ps
    p.cu[BLOCK_4x4].add_ps[NONALIGNED]   = PFX(pixel_add_ps_4x4_v);
    p.cu[BLOCK_8x8].add_ps[NONALIGNED]   = PFX(pixel_add_ps_8x8_v);
    p.cu[BLOCK_16x16].add_ps[NONALIGNED] = PFX(pixel_add_ps_16x16_v);
    p.cu[BLOCK_32x32].add_ps[NONALIGNED] = PFX(pixel_add_ps_32x32_v);
    p.cu[BLOCK_64x64].add_ps[NONALIGNED] = PFX(pixel_add_ps_64x64_v);

    // chroma add_ps
    p.chroma[X265_CSP_I420].cu[BLOCK_420_4x4].add_ps[NONALIGNED]   = PFX(pixel_add_ps_4x4_v);
    p.chroma[X265_CSP_I420].cu[BLOCK_420_8x8].add_ps[NONALIGNED]   = PFX(pixel_add_ps_8x8_v);
    p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].add_ps[NONALIGNED] = PFX(pixel_add_ps_16x16_v);
    p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].add_ps[NONALIGNED] = PFX(pixel_add_ps_32x32_v);
    p.chroma[X265_CSP_I422].cu[BLOCK_422_4x8].add_ps[NONALIGNED]   = PFX(pixel_add_ps_4x8_v);
    p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].add_ps[NONALIGNED]  = PFX(pixel_add_ps_8x16_v);
    p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].add_ps[NONALIGNED] = PFX(pixel_add_ps_16x32_v);
    p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].add_ps[NONALIGNED] = PFX(pixel_add_ps_32x64_v);

    //scale2D_64to32
    p.scale2D_64to32  = PFX(scale2D_64to32_v);

    // scale1D_128to64
    p.scale1D_128to64[NONALIGNED] = PFX(scale1D_128to64_v);
    p.scale1D_128to64[ALIGNED] = PFX(scale1D_128to64_v);

    // quant
    p.dequant_scaling = PFX(dequant_scaling_v);
    p.dequant_normal = PFX(dequant_normal_v);
    p.quant = PFX(quant_v);
    p.nquant = PFX(nquant_v);

    // ssim_4x4x2_core
    p.ssim_4x4x2_core = PFX(ssim_4x4x2_core_v);

    p.scanPosLast = PFX(scanPosLast_v);
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
#if HAVE_RVV_INTRINSIC
    if (cpuMask & X265_CPU_RVV)
    {
        setupPixelPrimitives_rvv(p);
    }
#else
    (void)p;
    (void)cpuMask;
#endif
}

} // namespace X265_NS
