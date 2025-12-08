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

#define ALL_LUMA_TU_TYPED_L(prim, fncdef, fname, cpu) \
    p.cu[BLOCK_4x4].prim   = fncdef PFX(fname ## _2_ ## cpu); \
    p.cu[BLOCK_8x8].prim   = fncdef PFX(fname ## _3_ ## cpu); \
    p.cu[BLOCK_16x16].prim = fncdef PFX(fname ## _4_ ## cpu); \
    p.cu[BLOCK_32x32].prim = fncdef PFX(fname ## _5_ ## cpu);
#define ALL_LUMA_TU_L(prim, fname, cpu)  ALL_LUMA_TU_TYPED_L(prim, , fname, cpu)

#define ALL_LUMA_TU_TYPED_S(prim, fncdef, fname, cpu) \
    p.cu[BLOCK_4x4].prim   = fncdef PFX(fname ## _4_ ## cpu); \
    p.cu[BLOCK_8x8].prim   = fncdef PFX(fname ## _8_ ## cpu); \
    p.cu[BLOCK_16x16].prim = fncdef PFX(fname ## _16_ ## cpu); \
    p.cu[BLOCK_32x32].prim = fncdef PFX(fname ## _32_ ## cpu);
#define ALL_LUMA_TU_S(prim, fname, cpu)  ALL_LUMA_TU_TYPED_S(prim, , fname, cpu)

#define ALL_LUMA_BLOCKS_TYPED_S(prim, fncdef, fname, cpu) \
    p.cu[BLOCK_4x4].prim   = fncdef PFX(fname ## _4_ ## cpu); \
    p.cu[BLOCK_8x8].prim   = fncdef PFX(fname ## _8_ ## cpu); \
    p.cu[BLOCK_16x16].prim = fncdef PFX(fname ## _16_ ## cpu); \
    p.cu[BLOCK_32x32].prim = fncdef PFX(fname ## _32_ ## cpu); \
    p.cu[BLOCK_64x64].prim = fncdef PFX(fname ## _64_ ## cpu);
#define ALL_LUMA_BLOCKS_S(prim, fname, cpu)  ALL_LUMA_BLOCKS_TYPED_S(prim, , fname, cpu)

#define ALL_LUMA_BLOCKS_TYPED(prim, fncdef, fname, cpu) \
    p.cu[BLOCK_4x4].prim   = fncdef PFX(fname ## _4x4_ ## cpu); \
    p.cu[BLOCK_8x8].prim   = fncdef PFX(fname ## _8x8_ ## cpu); \
    p.cu[BLOCK_16x16].prim = fncdef PFX(fname ## _16x16_ ## cpu); \
    p.cu[BLOCK_32x32].prim = fncdef PFX(fname ## _32x32_ ## cpu); \
    p.cu[BLOCK_64x64].prim = fncdef PFX(fname ## _64x64_ ## cpu);
#define ALL_LUMA_BLOCKS(prim, fname, cpu)  ALL_LUMA_BLOCKS_TYPED(prim, , fname, cpu)

#define ALL_LUMA_BLOCKS_TYPED_B(prim, fncdef, fname, cpu) \
    p.cu[BLOCK_8x8].prim   = fncdef PFX(fname ## _8x8_ ## cpu); \
    p.cu[BLOCK_16x16].prim = fncdef PFX(fname ## _16x16_ ## cpu); \
    p.cu[BLOCK_32x32].prim = fncdef PFX(fname ## _32x32_ ## cpu); \
    p.cu[BLOCK_64x64].prim = fncdef PFX(fname ## _64x64_ ## cpu);
#define ALL_LUMA_BLOCKS_B(prim, fname, cpu)  ALL_LUMA_BLOCKS_TYPED_B(prim, , fname, cpu)

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

#define ALL_CHROMA_420_CU_TYPED(prim, fncdef, fname, cpu) \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_4x4].prim   = fncdef PFX(fname ## _4x4_ ## cpu); \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_8x8].prim   = fncdef PFX(fname ## _8x8_ ## cpu); \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].prim = fncdef PFX(fname ## _16x16_ ## cpu); \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].prim = fncdef PFX(fname ## _32x32_ ## cpu)
#define ALL_CHROMA_420_CU(prim, fname, cpu) ALL_CHROMA_420_CU_TYPED(prim, , fname, cpu)

#define ALL_CHROMA_420_CU_TYPED_B(prim, fncdef, fname, cpu) \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_8x8].prim   = fncdef PFX(fname ## _8x8_ ## cpu); \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_16x16].prim = fncdef PFX(fname ## _16x16_ ## cpu); \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_32x32].prim = fncdef PFX(fname ## _32x32_ ## cpu)
#define ALL_CHROMA_420_CU_B(prim, fname, cpu) ALL_CHROMA_420_CU_TYPED_B(prim, , fname, cpu)

#define ALL_CHROMA_422_CU_TYPED(prim, fncdef, fname, cpu) \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_4x8].prim   = fncdef PFX(fname ## _4x8_ ## cpu); \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].prim  = fncdef PFX(fname ## _8x16_ ## cpu); \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].prim = fncdef PFX(fname ## _16x32_ ## cpu); \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].prim = fncdef PFX(fname ## _32x64_ ## cpu)
#define ALL_CHROMA_422_CU(prim, fname, cpu) ALL_CHROMA_422_CU_TYPED(prim, , fname, cpu)

#define ALL_CHROMA_422_CU_TYPED_B(prim, fncdef, fname, cpu) \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].prim  = fncdef PFX(fname ## _8x16_ ## cpu); \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].prim = fncdef PFX(fname ## _16x32_ ## cpu); \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].prim = fncdef PFX(fname ## _32x64_ ## cpu)
#define ALL_CHROMA_422_CU_B(prim, fname, cpu) ALL_CHROMA_422_CU_TYPED_B(prim, , fname, cpu)

#define ALL_CHROMA_420_PU_TYPED(prim, fncdef, fname, cpu)                                  \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x4].prim   = fncdef PFX(fname ## _4x4_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x8].prim   = fncdef PFX(fname ## _8x8_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x16].prim = fncdef PFX(fname ## _16x16_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x32].prim = fncdef PFX(fname ## _32x32_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x2].prim   = fncdef PFX(fname ## _4x2_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x4].prim   = fncdef PFX(fname ## _8x4_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x8].prim   = fncdef PFX(fname ## _4x8_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x8].prim  = fncdef PFX(fname ## _16x8_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x16].prim  = fncdef PFX(fname ## _8x16_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x16].prim = fncdef PFX(fname ## _32x16_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x32].prim = fncdef PFX(fname ## _16x32_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x6].prim   = fncdef PFX(fname ## _8x6_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_6x8].prim   = fncdef PFX(fname ## _6x8_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x2].prim   = fncdef PFX(fname ## _8x2_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x12].prim = fncdef PFX(fname ## _16x12_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_12x16].prim = fncdef PFX(fname ## _12x16_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x4].prim  = fncdef PFX(fname ## _16x4_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x16].prim  = fncdef PFX(fname ## _4x16_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x24].prim = fncdef PFX(fname ## _32x24_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_24x32].prim = fncdef PFX(fname ## _24x32_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x8].prim  = fncdef PFX(fname ## _32x8_ ## cpu); \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x32].prim  = fncdef PFX(fname ## _8x32_ ## cpu)
#define ALL_CHROMA_420_PU(prim, fname, cpu) ALL_CHROMA_420_PU_TYPED(prim, , fname, cpu)

#define ALL_CHROMA_422_PU_TYPED(prim, fncdef, fname, cpu)               \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x8].prim   = fncdef PFX(fname ## _4x8_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x16].prim  = fncdef PFX(fname ## _8x16_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x32].prim = fncdef PFX(fname ## _16x32_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x64].prim = fncdef PFX(fname ## _32x64_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x4].prim   = fncdef PFX(fname ## _4x4_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x8].prim   = fncdef PFX(fname ## _8x8_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x16].prim  = fncdef PFX(fname ## _4x16_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x16].prim = fncdef PFX(fname ## _16x16_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x32].prim  = fncdef PFX(fname ## _8x32_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x32].prim = fncdef PFX(fname ## _32x32_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x64].prim = fncdef PFX(fname ## _16x64_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x12].prim  = fncdef PFX(fname ## _8x12_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_6x16].prim  = fncdef PFX(fname ## _6x16_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x4].prim   = fncdef PFX(fname ## _8x4_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x24].prim = fncdef PFX(fname ## _16x24_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_12x32].prim = fncdef PFX(fname ## _12x32_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x8].prim  = fncdef PFX(fname ## _16x8_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x32].prim  = fncdef PFX(fname ## _4x32_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x48].prim = fncdef PFX(fname ## _32x48_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_24x64].prim = fncdef PFX(fname ## _24x64_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x16].prim = fncdef PFX(fname ## _32x16_ ## cpu); \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x64].prim  = fncdef PFX(fname ## _8x64_ ## cpu)
#define ALL_CHROMA_422_PU(prim, fname, cpu) ALL_CHROMA_422_PU_TYPED(prim, , fname, cpu)

#define ALL_CHROMA_444_PU_TYPED(prim, fncdef, fname, cpu) \
    p.chroma[X265_CSP_I444].pu[LUMA_4x4].prim   = fncdef PFX(fname ## _4x4_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_8x8].prim   = fncdef PFX(fname ## _8x8_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_16x16].prim = fncdef PFX(fname ## _16x16_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_32x32].prim = fncdef PFX(fname ## _32x32_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_64x64].prim = fncdef PFX(fname ## _64x64_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_8x4].prim   = fncdef PFX(fname ## _8x4_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_4x8].prim   = fncdef PFX(fname ## _4x8_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_16x8].prim  = fncdef PFX(fname ## _16x8_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_8x16].prim  = fncdef PFX(fname ## _8x16_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_16x32].prim = fncdef PFX(fname ## _16x32_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_32x16].prim = fncdef PFX(fname ## _32x16_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_64x32].prim = fncdef PFX(fname ## _64x32_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_32x64].prim = fncdef PFX(fname ## _32x64_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_16x12].prim = fncdef PFX(fname ## _16x12_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_12x16].prim = fncdef PFX(fname ## _12x16_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_16x4].prim  = fncdef PFX(fname ## _16x4_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_4x16].prim  = fncdef PFX(fname ## _4x16_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_32x24].prim = fncdef PFX(fname ## _32x24_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_24x32].prim = fncdef PFX(fname ## _24x32_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_32x8].prim  = fncdef PFX(fname ## _32x8_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_8x32].prim  = fncdef PFX(fname ## _8x32_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_64x48].prim = fncdef PFX(fname ## _64x48_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_48x64].prim = fncdef PFX(fname ## _48x64_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_64x16].prim = fncdef PFX(fname ## _64x16_ ## cpu); \
    p.chroma[X265_CSP_I444].pu[LUMA_16x64].prim = fncdef PFX(fname ## _16x64_ ## cpu)
#define ALL_CHROMA_444_PU(prim, fname, cpu) ALL_CHROMA_444_PU_TYPED(prim, , fname, cpu)

#define LUMA_CU(W, H) \
    p.cu[BLOCK_ ## W ## x ## H].sub_ps        = pixel_sub_ps_c<W, H>;

#if defined(__GNUC__)
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif

#include "fun-decls-prim.h"

namespace X265_NS
{
// private x265 namespace

void setupRVVPrimitives(EncoderPrimitives &p)
{
    // p2s
    ALL_CHROMA_420_PU(p2s[NONALIGNED], filterPixelToShort, rvv);
    ALL_CHROMA_422_PU(p2s[ALIGNED], filterPixelToShort, rvv);
    ALL_CHROMA_444_PU(p2s[ALIGNED], filterPixelToShort, rvv);
    ALL_LUMA_PU(convert_p2s[ALIGNED], filterPixelToShort, rvv);
    ALL_CHROMA_420_PU(p2s[ALIGNED], filterPixelToShort, rvv);
    ALL_CHROMA_422_PU(p2s[NONALIGNED], filterPixelToShort, rvv);
    ALL_CHROMA_444_PU(p2s[NONALIGNED], filterPixelToShort, rvv);
    ALL_LUMA_PU(convert_p2s[NONALIGNED], filterPixelToShort, rvv);

    // copy_count
    ALL_LUMA_TU_S(copy_cnt, copy_cnt, v);
    ALL_LUMA_TU_S(count_nonzero, count_nonzero, v);

    p.idst4x4               = PFX(idst4_v);
    p.denoiseDct            = PFX(denoiseDct_v);
    p.dst4x4                = PFX(dst4_v);

    ALL_LUMA_TU_S(dct, dct, v);
    ALL_LUMA_TU_S(idct, idct, v);

    ALL_LUMA_TU_L(nonPsyRdoQuant, nonPsyRdoQuant, v);
    ALL_LUMA_TU_L(psyRdoQuant, PsyRdoQuant, v);
    ALL_LUMA_TU_L(psyRdoQuant_1p, nonPsyRdoQuant, v);
    ALL_LUMA_TU_L(psyRdoQuant_2p, PsyRdoQuant, v);

    // ssd_s
    ALL_LUMA_BLOCKS(ssd_s[NONALIGNED], pixel_ssd_s, rvv);
    ALL_LUMA_BLOCKS(ssd_s[ALIGNED], pixel_ssd_s, rvv);

    // sse_pp
    ALL_LUMA_BLOCKS(sse_pp, pixel_sse_pp, rvv);

    ALL_CHROMA_420_CU(sse_pp, pixel_sse_pp, rvv);
    ALL_CHROMA_422_CU(sse_pp, pixel_sse_pp, rvv);

    // sse_ss
    ALL_LUMA_BLOCKS(sse_ss, pixel_sse_ss, rvv);

    // sad
    ALL_LUMA_PU(sad, pixel_sad, rvv);
    ALL_LUMA_PU(sad_x3, sad_x3, rvv);
    ALL_LUMA_PU(sad_x4, sad_x4, rvv);

#if !HIGH_BIT_DEPTH
    // pixel_avg_pp
    ALL_LUMA_PU(pixelavg_pp[NONALIGNED], pixel_avg_pp, rvv);
    ALL_LUMA_PU(pixelavg_pp[ALIGNED], pixel_avg_pp, rvv);

    // intra_pred_planar
    p.cu[BLOCK_4x4].intra_pred[PLANAR_IDX] = PFX(intra_pred_planar4_rvv);
    p.cu[BLOCK_8x8].intra_pred[PLANAR_IDX] = PFX(intra_pred_planar8_rvv);
    p.cu[BLOCK_16x16].intra_pred[PLANAR_IDX] = PFX(intra_pred_planar16_rvv);

    // ssimDist
    ALL_LUMA_BLOCKS_S(ssimDist, ssimDist, v);

    // normFact
    p.cu[BLOCK_8x8].normFact = PFX(normFact_v);
    p.cu[BLOCK_16x16].normFact = PFX(normFact_v);
    p.cu[BLOCK_32x32].normFact = PFX(normFact_v);
    p.cu[BLOCK_64x64].normFact = PFX(normFact_v);

    p.costCoeffNxN = PFX(costCoeffNxN_rvv);

    p.cu[BLOCK_4x4].sa8d   = PFX(satd4_4x4_rvv);
    ALL_LUMA_BLOCKS_B(sa8d, sa8d, rvv);

    p.chroma[X265_CSP_I420].cu[BLOCK_16x16].sa8d = PFX(sa8d_8x8_rvv);
    p.chroma[X265_CSP_I420].cu[BLOCK_32x32].sa8d = PFX(sa8d_16x16_rvv);
    p.chroma[X265_CSP_I420].cu[BLOCK_64x64].sa8d = PFX(sa8d_32x32_rvv);

    p.chroma[X265_CSP_I422].cu[BLOCK_16x16].sa8d = PFX(sa8d_8x16_rvv);
    p.chroma[X265_CSP_I422].cu[BLOCK_32x32].sa8d = PFX(sa8d_16x32_rvv);
    p.chroma[X265_CSP_I422].cu[BLOCK_64x64].sa8d = PFX(sa8d_32x64_rvv);

    ALL_CHROMA_422_CU_B(sa8d, sa8d, rvv);

    p.pu[LUMA_4x4].satd     = PFX(satd4_4x4_rvv);
    p.pu[LUMA_4x8].satd     = PFX(satd4_4x8_rvv);
    p.pu[LUMA_4x16].satd    = PFX(satd4_4x16_rvv);
    p.pu[LUMA_12x16].satd   = PFX(satd4_12x16_rvv);
    p.pu[LUMA_8x4].satd     = PFX(satd8_8x4_rvv);
    p.pu[LUMA_8x8].satd     = PFX(satd8_8x8_rvv);
    p.pu[LUMA_8x16].satd    = PFX(satd8_8x16_rvv);
    p.pu[LUMA_8x32].satd    = PFX(satd8_8x32_rvv);
    p.pu[LUMA_16x4].satd    = PFX(satd8_16x4_rvv);
    p.pu[LUMA_16x8].satd    = PFX(satd8_16x8_rvv);
    p.pu[LUMA_16x12].satd   = PFX(satd8_16x12_rvv);
    p.pu[LUMA_16x16].satd   = PFX(satd8_16x16_rvv);
    p.pu[LUMA_16x32].satd   = PFX(satd8_16x32_rvv);
    p.pu[LUMA_16x64].satd   = PFX(satd8_16x64_rvv);
    p.pu[LUMA_24x32].satd   = PFX(satd8_24x32_rvv);
    p.pu[LUMA_32x8].satd    = PFX(satd8_32x8_rvv);
    p.pu[LUMA_32x16].satd   = PFX(satd8_32x16_rvv);
    p.pu[LUMA_32x24].satd   = PFX(satd8_32x24_rvv);
    p.pu[LUMA_32x32].satd   = PFX(satd8_32x32_rvv);
    p.pu[LUMA_32x64].satd   = PFX(satd8_32x64_rvv);
    p.pu[LUMA_48x64].satd   = PFX(satd8_48x64_rvv);
    p.pu[LUMA_64x16].satd   = PFX(satd8_64x16_rvv);
    p.pu[LUMA_64x32].satd   = PFX(satd8_64x32_rvv);
    p.pu[LUMA_64x48].satd   = PFX(satd8_64x48_rvv);
    p.pu[LUMA_64x64].satd   = PFX(satd8_64x64_rvv);

    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x4].satd    = PFX(satd4_4x4_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x8].satd    = PFX(satd4_4x8_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x16].satd   = PFX(satd4_4x16_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_12x16].satd  = PFX(satd4_12x16_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x4].satd    = PFX(satd8_8x4_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x8].satd    = PFX(satd8_8x8_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x16].satd   = PFX(satd8_8x16_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x32].satd   = PFX(satd8_8x32_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x4].satd   = PFX(satd8_16x4_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x8].satd   = PFX(satd8_16x8_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x12].satd  = PFX(satd8_16x12_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x16].satd  = PFX(satd8_16x16_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x32].satd  = PFX(satd8_16x32_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_24x32].satd  = PFX(satd8_24x32_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x8].satd   = PFX(satd8_32x8_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x16].satd  = PFX(satd8_32x16_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x24].satd  = PFX(satd8_32x24_rvv);
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x32].satd  = PFX(satd8_32x32_rvv);

    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x4].satd   = PFX(satd4_4x4_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x8].satd   = PFX(satd4_4x8_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x16].satd   = PFX(satd4_4x16_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x32].satd   = PFX(satd4_4x32_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_12x32].satd   = PFX(satd4_12x32_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x4].satd  = PFX(satd8_8x4_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x8].satd  = PFX(satd8_8x8_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x12].satd  = PFX(satd8_8x12_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x16].satd  = PFX(satd8_8x16_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x32].satd  = PFX(satd8_8x32_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x64].satd  = PFX(satd8_8x64_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x32].satd  = PFX(satd4_4x32_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_12x32].satd = PFX(satd4_12x32_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x8].satd  = PFX(satd8_16x8_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x16].satd  = PFX(satd8_16x16_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x24].satd  = PFX(satd8_16x24_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x32].satd  = PFX(satd8_16x32_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x64].satd  = PFX(satd8_16x64_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_24x64].satd  = PFX(satd8_24x64_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x16].satd  = PFX(satd8_32x16_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x32].satd  = PFX(satd8_32x32_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x48].satd  = PFX(satd8_32x48_rvv);
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x64].satd  = PFX(satd8_32x64_rvv);
#endif

    // pixel_var
    ALL_LUMA_BLOCKS_B(var, pixel_var, v);

    // calc_Residual
    ALL_LUMA_TU_S(calcresidual[NONALIGNED], getResidual, v);
    ALL_LUMA_TU_S(calcresidual[ALIGNED], getResidual, v);

    // pixel_sub_ps
    ALL_LUMA_BLOCKS(sub_ps, pixel_sub_ps, v);
    ALL_CHROMA_420_CU(sub_ps, pixel_sub_ps, v);
    ALL_CHROMA_422_CU(sub_ps, pixel_sub_ps, v);

    // pixel_add_ps
    ALL_LUMA_BLOCKS(add_ps[NONALIGNED], pixel_add_ps, v);
    ALL_LUMA_BLOCKS(add_ps[ALIGNED], pixel_add_ps, v);
    ALL_CHROMA_420_CU(add_ps[NONALIGNED], pixel_add_ps, v);
    ALL_CHROMA_420_CU(add_ps[ALIGNED], pixel_add_ps, v);
    ALL_CHROMA_422_CU(add_ps[NONALIGNED], pixel_add_ps, v);
    ALL_CHROMA_422_CU(add_ps[ALIGNED], pixel_add_ps, v);

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

    p.saoCuStatsE0 = PFX(saoCuStatsE0_rvv);
    p.saoCuStatsE1 = PFX(saoCuStatsE1_rvv);
    p.saoCuStatsE2 = PFX(saoCuStatsE2_rvv);
    p.saoCuStatsE3 = PFX(saoCuStatsE3_rvv);

    p.saoCuOrgE0 = PFX(processSaoCUE0_rvv);
    p.saoCuOrgE1 = PFX(processSaoCUE1_rvv);
    p.saoCuOrgE1_2Rows = PFX(processSaoCUE1_2Rows_rvv);
    p.saoCuOrgE2[0] = PFX(processSaoCUE2_rvv);
    p.saoCuOrgE2[1] = PFX(processSaoCUE2_rvv);
    p.saoCuOrgE3[0] = PFX(processSaoCUE3_rvv);
    p.saoCuOrgE3[1] = PFX(processSaoCUE3_rvv);
    p.saoCuOrgB0 = PFX(processSaoCUB0_rvv);
    p.sign = PFX(calSign_rvv);
    p.pelFilterLumaStrong[0] = PFX(pelFilterLumaStrong_v_rvv);
    p.pelFilterLumaStrong[1] = PFX(pelFilterLumaStrong_h_rvv);
    //p.pelFilterChroma[0]     = PFX(pelFilterChroma_V_rvv);
    //p.pelFilterChroma[1]     = PFX(pelFilterChroma_H_rvv);

    p.weight_pp = PFX(weight_pp_v);
    p.weight_sp = PFX(weight_sp_v);

    p.planecopy_cp = PFX(planecopy_cp_v);
    p.planecopy_sp = PFX(planecopy_sp_v);
    p.planecopy_sp_shl = PFX(planecopy_sp_shl_v);
    p.planecopy_pp_shr = PFX(planecopy_pp_shr_v);

    ALL_LUMA_PU(addAvg[NONALIGNED], addAvg, v);
    ALL_LUMA_PU(addAvg[ALIGNED], addAvg, v);
    ALL_CHROMA_420_PU(addAvg[NONALIGNED], addAvg, v);
    ALL_CHROMA_420_PU(addAvg[ALIGNED], addAvg, v);
    ALL_CHROMA_422_PU(addAvg[NONALIGNED], addAvg, v);
    ALL_CHROMA_422_PU(addAvg[ALIGNED], addAvg, v);

    ALL_LUMA_PU(copy_pp, blockcopy_pp, v);
    ALL_CHROMA_420_PU(copy_pp, blockcopy_pp, v);
    ALL_CHROMA_422_PU(copy_pp, blockcopy_pp, v);

    ALL_LUMA_BLOCKS(copy_ss, blockcopy_ss, v);
    ALL_CHROMA_420_CU(copy_ss, blockcopy_ss, v);
    ALL_CHROMA_422_CU(copy_ss, blockcopy_ss, v);

    ALL_LUMA_BLOCKS(copy_sp, blockcopy_sp, v);
    ALL_CHROMA_420_CU(copy_sp, blockcopy_sp, v);
    ALL_CHROMA_422_CU(copy_sp, blockcopy_sp, v);

    ALL_LUMA_BLOCKS(copy_ps, blockcopy_ps, v);
    ALL_CHROMA_420_CU(copy_ps, blockcopy_ps, v);
    ALL_CHROMA_422_CU(copy_ps, blockcopy_ps, v);

    ALL_LUMA_BLOCKS_S(blockfill_s[NONALIGNED], blockfill_s, v);
    ALL_LUMA_BLOCKS_S(blockfill_s[ALIGNED], blockfill_s, v);
    ALL_LUMA_BLOCKS_S(cpy2Dto1D_shl, cpy2Dto1D_shl, v);
    ALL_LUMA_BLOCKS_S(cpy2Dto1D_shr, cpy2Dto1D_shr, v);
    ALL_LUMA_BLOCKS_S(cpy1Dto2D_shl[NONALIGNED], cpy1Dto2D_shl, v);
    ALL_LUMA_BLOCKS_S(cpy1Dto2D_shl[ALIGNED], cpy1Dto2D_shl, v);
    ALL_LUMA_BLOCKS_S(cpy1Dto2D_shr, cpy1Dto2D_shr, v);
}

void setupAssemblyPrimitives(EncoderPrimitives &p, int cpuMask)
{
    if (cpuMask & X265_CPU_RVV)
    {
        setupRVVPrimitives(p);

        if (cpuMask & X265_CPU_ZBB) {
#if !HIGH_BIT_DEPTH
            ALL_LUMA_BLOCKS(psy_cost_pp, psyCost_pp, rvv);
#endif
        }
    }
}

void setupIntrinsicPrimitives(EncoderPrimitives &p, int cpuMask)
{
#if HAVE_RVV_INTRINSIC
    if (cpuMask & X265_CPU_RVV)
    {
        setupPixelPrimitives_rvv(p);
        setupSaoPrimitives_rvv(p);
        setupIntraPrimitives_rvv(p);
#if !HIGH_BIT_DEPTH
        setupFilterPrimitives_rvv(p);
#endif
    }
#else
    (void)p;
    (void)cpuMask;
#endif
}

} // namespace X265_NS
