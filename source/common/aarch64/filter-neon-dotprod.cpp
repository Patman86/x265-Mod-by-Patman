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

#include "filter-neon-dotprod.h"

#if !HIGH_BIT_DEPTH
#include "mem-neon.h"
#include <arm_neon.h>

namespace {
static const uint8_t dotprod_permute_tbl[48] = {
    0, 1,  2,  3, 1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5, 6,
    4, 5,  6,  7, 5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10,
    8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14
};

static const uint8_t dot_prod_merge_block_tbl[48] = {
    // Shift left and insert new last column in transposed 4x4 block.
    1, 2, 3, 16, 5, 6, 7, 20, 9, 10, 11, 24, 13, 14, 15, 28,
    // Shift left and insert two new columns in transposed 4x4 block.
    2, 3, 16, 17, 6, 7, 20, 21, 10, 11, 24, 25, 14, 15, 28, 29,
    // Shift left and insert three new columns in transposed 4x4 block.
    3, 16, 17, 18, 7, 20, 21, 22, 11, 24, 25, 26, 15, 28, 29, 30
};

uint8x8_t inline filter8_8_pp(uint8x16_t samples, const int8x8_t filter,
                              const int32x4_t constant, const uint8x16x3_t tbl)
{
    // Transform sample range from uint8_t to int8_t for signed dot product.
    int8x16_t samples_s8 =
        vreinterpretq_s8_u8(vsubq_u8(samples, vdupq_n_u8(128)));

    // Permute input samples for dot product.
    // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
    int8x16_t perm_samples_0 = vqtbl1q_s8(samples_s8, tbl.val[0]);
    // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
    int8x16_t perm_samples_1 = vqtbl1q_s8(samples_s8, tbl.val[1]);
    // { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 }
    int8x16_t perm_samples_2 = vqtbl1q_s8(samples_s8, tbl.val[2]);

    int32x4_t dotprod_lo = vdotq_lane_s32(constant, perm_samples_0, filter, 0);
    int32x4_t dotprod_hi = vdotq_lane_s32(constant, perm_samples_1, filter, 0);
    dotprod_lo = vdotq_lane_s32(dotprod_lo, perm_samples_1, filter, 1);
    dotprod_hi = vdotq_lane_s32(dotprod_hi, perm_samples_2, filter, 1);

    // Narrow and combine.
    int16x8_t dotprod = vcombine_s16(vmovn_s32(dotprod_lo),
                                     vmovn_s32(dotprod_hi));
    return vqrshrun_n_s16(dotprod, IF_FILTER_PREC);
}

void inline init_sample_permute(uint8x8_t *samples, const uint8x16x3_t tbl,
                                int8x16_t *d)
{
    // Transform sample range from uint8_t to int8_t for signed dot product.
    int8x8_t samples_s8[4];
    samples_s8[0] = vreinterpret_s8_u8(vsub_u8(samples[0], vdup_n_u8(128)));
    samples_s8[1] = vreinterpret_s8_u8(vsub_u8(samples[1], vdup_n_u8(128)));
    samples_s8[2] = vreinterpret_s8_u8(vsub_u8(samples[2], vdup_n_u8(128)));
    samples_s8[3] = vreinterpret_s8_u8(vsub_u8(samples[3], vdup_n_u8(128)));

    // Permute input samples for dot product.
    // { 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6 }
    d[0] = vqtbl1q_s8(vcombine_s8(samples_s8[0], vdup_n_s8(0)), tbl.val[0]);
    d[1] = vqtbl1q_s8(vcombine_s8(samples_s8[1], vdup_n_s8(0)), tbl.val[0]);
    d[2] = vqtbl1q_s8(vcombine_s8(samples_s8[2], vdup_n_s8(0)), tbl.val[0]);
    d[3] = vqtbl1q_s8(vcombine_s8(samples_s8[3], vdup_n_s8(0)), tbl.val[0]);
}

uint8x8_t inline filter8_8_pp_reuse(uint8x16_t samples, const int8x8_t filter,
                                    const int32x4_t constant,
                                    const uint8x16x3_t tbl,
                                    int8x16_t &perm_samples_0)
{
    // Transform sample range from uint8_t to int8_t for signed dot product.
    int8x16_t samples_s8 =
        vreinterpretq_s8_u8(vsubq_u8(samples, vdupq_n_u8(128)));

    // Permute input samples for dot product.
    // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
    // Already in perm_samples_0.
    // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
    int8x16_t perm_samples_1 = vqtbl1q_s8(samples_s8, tbl.val[1]);
    // { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 }
    int8x16_t perm_samples_2 = vqtbl1q_s8(samples_s8, tbl.val[2]);

    int32x4_t dotprod_lo = vdotq_lane_s32(constant, perm_samples_0, filter, 0);
    int32x4_t dotprod_hi = vdotq_lane_s32(constant, perm_samples_1, filter, 0);
    dotprod_lo = vdotq_lane_s32(dotprod_lo, perm_samples_1, filter, 1);
    dotprod_hi = vdotq_lane_s32(dotprod_hi, perm_samples_2, filter, 1);

    // Save for re-use in next iteration.
    perm_samples_0 = perm_samples_2;

    // Narrow and combine.
    int16x8_t dotprod = vcombine_s16(vmovn_s32(dotprod_lo),
                                     vmovn_s32(dotprod_hi));
    return vqrshrun_n_s16(dotprod, IF_FILTER_PREC);
}

int16x4_t inline filter8_4_ps(uint8x16_t samples, const int8x8_t filter,
                              const uint8x16x3_t tbl)
{
    // Transform sample range from uint8_t to int8_t for signed dot product.
    int8x16_t samples_s8 =
        vreinterpretq_s8_u8(vsubq_u8(samples, vdupq_n_u8(128)));

    // Permute input samples for dot product.
    // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
    int8x16_t perm_samples_0 = vqtbl1q_s8(samples_s8, tbl.val[0]);
    // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
    int8x16_t perm_samples_1 = vqtbl1q_s8(samples_s8, tbl.val[1]);

    // Correction accounting for sample range transform cancels to 0.
    int32x4_t constant = vdupq_n_s32(0);
    int32x4_t dotprod = vdotq_lane_s32(constant, perm_samples_0, filter, 0);
    dotprod = vdotq_lane_s32(dotprod, perm_samples_1, filter, 1);

    // Narrow.
    return vmovn_s32(dotprod);
}

int16x8_t inline filter8_8_ps(uint8x16_t samples, const int8x8_t filter,
                              const uint8x16x3_t tbl)
{
    // Transform sample range from uint8_t to int8_t for signed dot product.
    int8x16_t samples_s8 =
        vreinterpretq_s8_u8(vsubq_u8(samples, vdupq_n_u8(128)));

    // Permute input samples for dot product.
    // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
    int8x16_t perm_samples_0 = vqtbl1q_s8(samples_s8, tbl.val[0]);
    // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
    int8x16_t perm_samples_1 = vqtbl1q_s8(samples_s8, tbl.val[1]);
    // { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 }
    int8x16_t perm_samples_2 = vqtbl1q_s8(samples_s8, tbl.val[2]);

    // Correction accounting for sample range transform cancels to 0.
    int32x4_t constant = vdupq_n_s32(0);
    int32x4_t dotprod_lo = vdotq_lane_s32(constant, perm_samples_0, filter, 0);
    int32x4_t dotprod_hi = vdotq_lane_s32(constant, perm_samples_1, filter, 0);
    dotprod_lo = vdotq_lane_s32(dotprod_lo, perm_samples_1, filter, 1);
    dotprod_hi = vdotq_lane_s32(dotprod_hi, perm_samples_2, filter, 1);

    // Narrow and combine.
    return vcombine_s16(vmovn_s32(dotprod_lo), vmovn_s32(dotprod_hi));
}

int16x8_t inline filter8_8_ps_reuse(uint8x16_t samples, const int8x8_t filter,
                                    const uint8x16x3_t tbl,
                                    int8x16_t &perm_samples_0)
{
    // Transform sample range from uint8_t to int8_t for signed dot product.
    int8x16_t samples_s8 =
        vreinterpretq_s8_u8(vsubq_u8(samples, vdupq_n_u8(128)));

    // Permute input samples for dot product.
    // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
    // Already in perm_samples_0.
    // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
    int8x16_t perm_samples_1 = vqtbl1q_s8(samples_s8, tbl.val[1]);
    // { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 }
    int8x16_t perm_samples_2 = vqtbl1q_s8(samples_s8, tbl.val[2]);

    // Correction accounting for sample range transform cancels to 0.
    int32x4_t constant = vdupq_n_s32(0);
    int32x4_t dotprod_lo = vdotq_lane_s32(constant, perm_samples_0, filter, 0);
    int32x4_t dotprod_hi = vdotq_lane_s32(constant, perm_samples_1, filter, 0);
    dotprod_lo = vdotq_lane_s32(dotprod_lo, perm_samples_1, filter, 1);
    dotprod_hi = vdotq_lane_s32(dotprod_hi, perm_samples_2, filter, 1);

    // Save for re-use in next iteration.
    perm_samples_0 = perm_samples_2;

    // Narrow and combine.
    return vcombine_s16(vmovn_s32(dotprod_lo), vmovn_s32(dotprod_hi));
}

uint8x8_t inline filter4_8_pp(uint8x16_t samples, const int8x8_t filter,
                              const int32x4_t constant, const uint8x16x2_t tbl)
{
    // Transform sample range from uint8_t to int8_t for signed dot product.
    int8x16_t samples_s8 =
        vreinterpretq_s8_u8(vsubq_u8(samples, vdupq_n_u8(128)));

    // Permute input samples for dot product.
    // {0,  1,  2,  3 ,  1,  2,  3,  4 ,  2,  3,  4,  5 ,  3,  4,  5,  6}
    int8x16_t perm_samples_0 = vqtbl1q_s8(samples_s8, tbl.val[0]);
    // {4,  5,  6,  7 ,  5,  6,  7,  8 ,  6,  7,  8,  9 ,  7,  8,  9, 10}
    int8x16_t perm_samples_1 = vqtbl1q_s8(samples_s8, tbl.val[1]);

    int32x4_t dotprod_lo = vdotq_lane_s32(constant, perm_samples_0, filter, 0);
    int32x4_t dotprod_hi = vdotq_lane_s32(constant, perm_samples_1, filter, 0);

    // Narrow and combine.
    int16x8_t dotprod = vcombine_s16(vmovn_s32(dotprod_lo),
                                     vmovn_s32(dotprod_hi));
    return vqrshrun_n_s16(dotprod, IF_FILTER_PREC);
}

int16x8_t inline filter4_8_ps(uint8x16_t samples, const int8x8_t filter,
                              const uint8x16x2_t tbl)
{
    // Transform sample range from uint8_t to int8_t for signed dot product.
    int8x16_t samples_s8 =
        vreinterpretq_s8_u8(vsubq_u8(samples, vdupq_n_u8(128)));

    // Permute input samples for dot product.
    // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
    int8x16_t perm_samples_0 = vqtbl1q_s8(samples_s8, tbl.val[0]);
    // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
    int8x16_t perm_samples_1 = vqtbl1q_s8(samples_s8, tbl.val[1]);

    // Correction accounting for sample range transform cancels to 0.
    int32x4_t constant = vdupq_n_s32(0);
    int32x4_t dotprod_lo = vdotq_lane_s32(constant, perm_samples_0, filter, 0);
    int32x4_t dotprod_hi = vdotq_lane_s32(constant, perm_samples_1, filter, 0);

    // Narrow and combine.
    return vcombine_s16(vmovn_s32(dotprod_lo), vmovn_s32(dotprod_hi));
}

void inline transpose_concat_4x4(const int8x8_t *s, int8x16_t &d)
{
    // Transpose 8-bit elements and concatenate result rows as follows:
    // s0: 00, 01, 02, 03, XX, XX, XX, XX
    // s1: 10, 11, 12, 13, XX, XX, XX, XX
    // s2: 20, 21, 22, 23, XX, XX, XX, XX
    // s3: 30, 31, 32, 33, XX, XX, XX, XX
    //
    // d: 00, 10, 20, 30, 01, 11, 21, 31, 02, 12, 22, 32, 03, 13, 23, 33
    int8x16_t s0q = vcombine_s8(s[0], vdup_n_s8(0));
    int8x16_t s1q = vcombine_s8(s[1], vdup_n_s8(0));
    int8x16_t s2q = vcombine_s8(s[2], vdup_n_s8(0));
    int8x16_t s3q = vcombine_s8(s[3], vdup_n_s8(0));

    int8x16_t s01 = vzipq_s8(s0q, s1q).val[0];
    int8x16_t s23 = vzipq_s8(s2q, s3q).val[0];

    int16x8_t s0123 =
        vzipq_s16(vreinterpretq_s16_s8(s01), vreinterpretq_s16_s8(s23)).val[0];

    d = vreinterpretq_s8_s16(s0123);
}

void inline transpose_concat_8x4(const int8x8_t *s, int8x16_t &d0,
                                 int8x16_t &d1)
{
    // Transpose 8-bit elements and concatenate result rows as follows:
    // s0: 00, 01, 02, 03, 04, 05, 06, 07
    // s1: 10, 11, 12, 13, 14, 15, 16, 17
    // s2: 20, 21, 22, 23, 24, 25, 26, 27
    // s3: 30, 31, 32, 33, 34, 35, 36, 37
    //
    // d0: 00, 10, 20, 30, 01, 11, 21, 31, 02, 12, 22, 32, 03, 13, 23, 33
    // d1: 04, 14, 24, 34, 05, 15, 25, 35, 06, 16, 26, 36, 07, 17, 27, 37
    int8x16_t s0q = vcombine_s8(s[0], vdup_n_s8(0));
    int8x16_t s1q = vcombine_s8(s[1], vdup_n_s8(0));
    int8x16_t s2q = vcombine_s8(s[2], vdup_n_s8(0));
    int8x16_t s3q = vcombine_s8(s[3], vdup_n_s8(0));

    int8x16_t s01 = vzipq_s8(s0q, s1q).val[0];
    int8x16_t s23 = vzipq_s8(s2q, s3q).val[0];

    int16x8x2_t s0123 =
        vzipq_s16(vreinterpretq_s16_s8(s01), vreinterpretq_s16_s8(s23));

    d0 = vreinterpretq_s8_s16(s0123.val[0]);
    d1 = vreinterpretq_s8_s16(s0123.val[1]);
}

int16x4_t inline filter8_4_ps_partial(const int8x16_t s0, const int8x16_t s1,
                                      const int8x8_t filter)

{
    int32x4_t dotprod = vdotq_lane_s32(vdupq_n_s32(0), s0, filter, 0);
    dotprod = vdotq_lane_s32(dotprod, s1, filter, 1);
    return vmovn_s32(dotprod);
}

int16x8_t inline filter8_8_ps_partial(const int8x16_t s0, const int8x16_t s1,
                                      const int8x16_t s2, const int8x16_t s3,
                                      const int8x8_t filter)
{
    int32x4_t dotprod_lo = vdotq_lane_s32(vdupq_n_s32(0), s0, filter, 0);
    dotprod_lo = vdotq_lane_s32(dotprod_lo, s2, filter, 1);
    int32x4_t dotpro_hi = vdotq_lane_s32(vdupq_n_s32(0), s1, filter, 0);
    dotpro_hi = vdotq_lane_s32(dotpro_hi, s3, filter, 1);

    // Narrow and combine.
    return vcombine_s16(vmovn_s32(dotprod_lo), vmovn_s32(dotpro_hi));
}
} // Unnamed namespace.

namespace X265_NS {
template<int width, int height>
void interp8_horiz_pp_dotprod(const uint8_t *src, intptr_t srcStride,
                              uint8_t *dst, intptr_t dstStride, int coeffIdx)
{
    const int N_TAPS = 8;

    src -= N_TAPS / 2 - 1;

    const uint8x16x3_t tbl = vld1q_u8_x3(dotprod_permute_tbl);
    const int8x8_t filter = vmovn_s16(vld1q_s16(g_lumaFilter[coeffIdx]));
    // Correction accounting for sample range transform from uint8_t to int8_t.
    const int32x4_t c = vdupq_n_s32(64 * 128);

    int row;
    for (row = 0; row < height; row += 4)
    {
        int col = 0;
        if (width >= 32)
        {
            // Peel first sample permute to enable passing between iterations.
            uint8x8_t s0[4];
            load_u8x8xn<4>(src, srcStride, s0);
            int8x16_t ps0[4];
            init_sample_permute(s0, tbl, ps0);

            for (; col + 16 <= width; col += 16)
            {
                uint8x16_t s_lo[4], s_hi[4];
                load_u8x16xn<4>(src + col + 0, srcStride, s_lo);
                load_u8x16xn<4>(src + col + 8, srcStride, s_hi);

                uint8x8_t d_lo[4];
                d_lo[0] = filter8_8_pp_reuse(s_lo[0], filter, c, tbl, ps0[0]);
                d_lo[1] = filter8_8_pp_reuse(s_lo[1], filter, c, tbl, ps0[1]);
                d_lo[2] = filter8_8_pp_reuse(s_lo[2], filter, c, tbl, ps0[2]);
                d_lo[3] = filter8_8_pp_reuse(s_lo[3], filter, c, tbl, ps0[3]);

                uint8x8_t d_hi[4];
                d_hi[0] = filter8_8_pp_reuse(s_hi[0], filter, c, tbl, ps0[0]);
                d_hi[1] = filter8_8_pp_reuse(s_hi[1], filter, c, tbl, ps0[1]);
                d_hi[2] = filter8_8_pp_reuse(s_hi[2], filter, c, tbl, ps0[2]);
                d_hi[3] = filter8_8_pp_reuse(s_hi[3], filter, c, tbl, ps0[3]);

                store_u8x8xn<4>(dst + col + 0, dstStride, d_lo);
                store_u8x8xn<4>(dst + col + 8, dstStride, d_hi);
            }
        }
        else
        {
            for (; col + 8 <= width; col += 8)
            {
                uint8x16_t s[4];
                load_u8x16xn<4>(src + col, srcStride, s);

                uint8x8_t d[4];
                d[0] = filter8_8_pp(s[0], filter, c, tbl);
                d[1] = filter8_8_pp(s[1], filter, c, tbl);
                d[2] = filter8_8_pp(s[2], filter, c, tbl);
                d[3] = filter8_8_pp(s[3], filter, c, tbl);

                store_u8x8xn<4>(dst + col, dstStride, d);
            }
        }
        for (; col < width; col += 4)
        {
            uint8x16_t s[4];
            load_u8x16xn<4>(src + col, srcStride, s);

            uint8x8_t d[4];
            d[0] = filter8_8_pp(s[0], filter, c, tbl);
            d[1] = filter8_8_pp(s[1], filter, c, tbl);
            d[2] = filter8_8_pp(s[2], filter, c, tbl);
            d[3] = filter8_8_pp(s[3], filter, c, tbl);

            store_u8x4xn<4>(dst + col, dstStride, d);
        }

        src += 4 * srcStride;
        dst += 4 * dstStride;
    }
}

template<int width, int height>
void interp8_horiz_ps_dotprod(const uint8_t *src, intptr_t srcStride,
                              int16_t *dst, intptr_t dstStride, int coeffIdx,
                              int isRowExt)
{
    const int N_TAPS = 8;
    int blkheight = height;

    src -= N_TAPS / 2 - 1;
    if (isRowExt)
    {
        src -= (N_TAPS / 2 - 1) * srcStride;
        blkheight += N_TAPS - 1;
    }

    const uint8x16x3_t tbl = vld1q_u8_x3(dotprod_permute_tbl);
    const int8x8_t filter = vmovn_s16(vld1q_s16(g_lumaFilter[coeffIdx]));

    for (int row = 0; row + 4 <= blkheight; row += 4)
    {
        int col = 0;
        if (width >= 32)
        {
            // Peel first sample permute to enable passing between iterations.
            uint8x8_t s0[4];
            load_u8x8xn<4>(src, srcStride, s0);
            int8x16_t ps0[4];
            init_sample_permute(s0, tbl, ps0);

            for (; col + 16 <= width; col += 16)
            {
                uint8x16_t s_lo[4], s_hi[4];
                load_u8x16xn<4>(src + col + 0, srcStride, s_lo);
                load_u8x16xn<4>(src + col + 8, srcStride, s_hi);

                int16x8_t d_lo[4];
                d_lo[0] = filter8_8_ps_reuse(s_lo[0], filter, tbl, ps0[0]);
                d_lo[1] = filter8_8_ps_reuse(s_lo[1], filter, tbl, ps0[1]);
                d_lo[2] = filter8_8_ps_reuse(s_lo[2], filter, tbl, ps0[2]);
                d_lo[3] = filter8_8_ps_reuse(s_lo[3], filter, tbl, ps0[3]);

                int16x8_t d_hi[4];
                d_hi[0] = filter8_8_ps_reuse(s_hi[0], filter, tbl, ps0[0]);
                d_hi[1] = filter8_8_ps_reuse(s_hi[1], filter, tbl, ps0[1]);
                d_hi[2] = filter8_8_ps_reuse(s_hi[2], filter, tbl, ps0[2]);
                d_hi[3] = filter8_8_ps_reuse(s_hi[3], filter, tbl, ps0[3]);

                store_s16x8xn<4>(dst + col + 0, dstStride, d_lo);
                store_s16x8xn<4>(dst + col + 8, dstStride, d_hi);
            }
        }
        else
        {
            for (; col + 8 <= width; col += 8)
            {
                uint8x16_t s[4];
                load_u8x16xn<4>(src + col, srcStride, s);

                int16x8_t d[4];
                d[0] = filter8_8_ps(s[0], filter, tbl);
                d[1] = filter8_8_ps(s[1], filter, tbl);
                d[2] = filter8_8_ps(s[2], filter, tbl);
                d[3] = filter8_8_ps(s[3], filter, tbl);

                store_s16x8xn<4>(dst + col, dstStride, d);
            }
        }
        for (; col < width; col += 4)
        {
            uint8x16_t s[4];
            load_u8x16xn<4>(src + col, srcStride, s);

            int16x4_t d[4];
            d[0] = filter8_4_ps(s[0], filter, tbl);
            d[1] = filter8_4_ps(s[1], filter, tbl);
            d[2] = filter8_4_ps(s[2], filter, tbl);
            d[3] = filter8_4_ps(s[3], filter, tbl);

            store_s16x4xn<4>(dst + col, dstStride, d);
        }

        src += 4 * srcStride;
        dst += 4 * dstStride;
    }

    if (isRowExt)
    {
        // Process final 3 rows.
        int col = 0;
        for (; (col + 8) <= width; col += 8)
        {
            uint8x16_t s[3];
            load_u8x16xn<3>(src + col, srcStride, s);

            int16x8_t d[3];
            d[0] = filter8_8_ps(s[0], filter, tbl);
            d[1] = filter8_8_ps(s[1], filter, tbl);
            d[2] = filter8_8_ps(s[2], filter, tbl);

            store_s16x8xn<3>(dst + col, dstStride, d);
        }

        for (; col < width; col += 4)
        {
            uint8x16_t s[3];
            load_u8x16xn<3>(src + col, srcStride, s);

            int16x4_t d[3];
            d[0] = filter8_4_ps(s[0], filter, tbl);
            d[1] = filter8_4_ps(s[1], filter, tbl);
            d[2] = filter8_4_ps(s[2], filter, tbl);

            store_s16x4xn<3>(dst + col, dstStride, d);
        }
    }
}

template<int N, int width, int height>
void interp_horiz_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst,
                          intptr_t dstStride, int coeffIdx);

template<int width, int height>
void interp4_horiz_pp_dotprod(const uint8_t *src, intptr_t srcStride,
                              uint8_t *dst, intptr_t dstStride, int coeffIdx)
{
    const int N_TAPS = 4;

    if (coeffIdx == 4)
        return interp_horiz_pp_neon<N_TAPS, width, height>(src, srcStride, dst,
                                                           dstStride, coeffIdx);

    src -= N_TAPS / 2 - 1;

    const uint8x16x2_t tbl = vld1q_u8_x2(dotprod_permute_tbl);
    const int16x4_t filter_16 = vld1_s16(g_chromaFilter[coeffIdx]);
    const int8x8_t filter = vmovn_s16(vcombine_s16(filter_16, vdup_n_s16(0)));

    // Correction accounting for sample range transform from uint8_t to int8_t.
    const int32x4_t c = vdupq_n_s32(64 * 128);

    for (int row = 0; row + 4 <= height; row += 4)
    {
        int col = 0;
        for (; col + 16 <= width; col += 16)
        {
            uint8x16_t s0[4], s1[4];
            load_u8x16xn<4>(src + col, srcStride, s0);
            load_u8x16xn<4>(src + col + 8, srcStride, s1);

            uint8x8_t d_lo[4];
            d_lo[0] = filter4_8_pp(s0[0], filter, c, tbl);
            d_lo[1] = filter4_8_pp(s0[1], filter, c, tbl);
            d_lo[2] = filter4_8_pp(s0[2], filter, c, tbl);
            d_lo[3] = filter4_8_pp(s0[3], filter, c, tbl);

            uint8x8_t d_hi[4];
            d_hi[0] = filter4_8_pp(s1[0], filter, c, tbl);
            d_hi[1] = filter4_8_pp(s1[1], filter, c, tbl);
            d_hi[2] = filter4_8_pp(s1[2], filter, c, tbl);
            d_hi[3] = filter4_8_pp(s1[3], filter, c, tbl);

            uint8x16_t d[4];
            d[0] = vcombine_u8(d_lo[0], d_hi[0]);
            d[1] = vcombine_u8(d_lo[1], d_hi[1]);
            d[2] = vcombine_u8(d_lo[2], d_hi[2]);
            d[3] = vcombine_u8(d_lo[3], d_hi[3]);

            store_u8x16xn<4>(dst + col, dstStride, d);
        }

        for (; col + 8 <= width; col += 8)
        {
            uint8x16_t s[4];
            load_u8x16xn<4>(src + col, srcStride, s);

            uint8x8_t d[4];
            d[0] = filter4_8_pp(s[0], filter, c, tbl);
            d[1] = filter4_8_pp(s[1], filter, c, tbl);
            d[2] = filter4_8_pp(s[2], filter, c, tbl);
            d[3] = filter4_8_pp(s[3], filter, c, tbl);

            store_u8x8xn<4>(dst + col, dstStride, d);
        }

        // Block sizes 12xH, 6xH, 4xH, 2xH.
        if (width % 8 != 0)
        {
            uint8x16_t s[4];
            load_u8x16xn<4>(src + col, srcStride, s);

            uint8x8_t d[4];
            d[0] = filter4_8_pp(s[0], filter, c, tbl);
            d[1] = filter4_8_pp(s[1], filter, c, tbl);
            d[2] = filter4_8_pp(s[2], filter, c, tbl);
            d[3] = filter4_8_pp(s[3], filter, c, tbl);

            const int n_store = width < 8 ? width : 4;
            store_u8xnxm<n_store, 4>(dst + col, dstStride, d);
        }

        src += 4 * srcStride;
        dst += 4 * dstStride;
    }

    // Block sizes 8x6, 8x2, 4x2.
    if (height & 2)
    {
        uint8x16_t s[2];
        load_u8x16xn<2>(src, srcStride, s);

        uint8x8_t d[2];
        d[0] = filter4_8_pp(s[0], filter, c, tbl);
        d[1] = filter4_8_pp(s[1], filter, c, tbl);

        const int n_store = width < 8 ? width : 8;
        store_u8xnxm<n_store, 2>(dst, dstStride, d);
    }
}

template<int width, int height>
void interp4_horiz_ps_dotprod(const uint8_t *src, intptr_t srcStride,
                              int16_t *dst, intptr_t dstStride, int coeffIdx,
                              int isRowExt)
{
    const int N_TAPS = 4;
    int blkheight = height;

    src -= N_TAPS / 2 - 1;
    if (isRowExt)
    {
        src -= (N_TAPS / 2 - 1) * srcStride;
        blkheight += N_TAPS - 1;
    }

    const uint8x16x2_t tbl = vld1q_u8_x2(dotprod_permute_tbl);
    const int16x4_t filter_16 = vld1_s16(g_chromaFilter[coeffIdx]);
    const int8x8_t filter = vmovn_s16(vcombine_s16(filter_16, vdup_n_s16(0)));

    int row = 0;
    for (; row + 4 <= blkheight; row += 4)
    {
        int col = 0;
        for (; col + 16 <= width; col += 16)
        {
            uint8x16_t s_lo[4], s_hi[4];
            load_u8x16xn<4>(src + col + 0, srcStride, s_lo);
            load_u8x16xn<4>(src + col + 8, srcStride, s_hi);

            int16x8_t d_lo[4];
            d_lo[0] = filter4_8_ps(s_lo[0], filter, tbl);
            d_lo[1] = filter4_8_ps(s_lo[1], filter, tbl);
            d_lo[2] = filter4_8_ps(s_lo[2], filter, tbl);
            d_lo[3] = filter4_8_ps(s_lo[3], filter, tbl);

            int16x8_t d_hi[4];
            d_hi[0] = filter4_8_ps(s_hi[0], filter, tbl);
            d_hi[1] = filter4_8_ps(s_hi[1], filter, tbl);
            d_hi[2] = filter4_8_ps(s_hi[2], filter, tbl);
            d_hi[3] = filter4_8_ps(s_hi[3], filter, tbl);

            store_s16x8xn<4>(dst + col + 0, dstStride, d_lo);
            store_s16x8xn<4>(dst + col + 8, dstStride, d_hi);
        }

        for (; col + 8 <= width; col += 8)
        {
            uint8x16_t s[4];
            load_u8x16xn<4>(src + col, srcStride, s);

            int16x8_t d[4];
            d[0] = filter4_8_ps(s[0], filter, tbl);
            d[1] = filter4_8_ps(s[1], filter, tbl);
            d[2] = filter4_8_ps(s[2], filter, tbl);
            d[3] = filter4_8_ps(s[3], filter, tbl);

            store_s16x8xn<4>(dst + col, dstStride, d);
        }

        // Block sizes 12xH, 6xH, 4xH, 2xH.
        if (width % 8 != 0)
        {
            uint8x16_t s[4];
            load_u8x16xn<4>(src + col, srcStride, s);

            int16x8_t d[4];
            d[0] = filter4_8_ps(s[0], filter, tbl);
            d[1] = filter4_8_ps(s[1], filter, tbl);
            d[2] = filter4_8_ps(s[2], filter, tbl);
            d[3] = filter4_8_ps(s[3], filter, tbl);

            const int n_store = width < 8 ? width : 4;
            store_s16xnxm<n_store, 4>(d, dst + col, dstStride);
        }

        src += 4 * srcStride;
        dst += 4 * dstStride;
    }

    // Process remaining rows.
    for (; row < blkheight; ++row)
    {
        int col = 0;
        for (; (col + 8) <= width; col += 8)
        {
            uint8x16_t s = vld1q_u8(src + col);

            int16x8_t d = filter4_8_ps(s, filter, tbl);

            vst1q_s16(dst + col, d);
        }

        // Block sizes 12xH, 6xH, 4xH, 2xH.
        if (width % 8 != 0)
        {
            uint8x16_t s = vld1q_u8(src + col);

            int16x8_t d = filter4_8_ps(s, filter, tbl);

            const int n_store = width < 8 ? width : 4;
            store_s16xnxm<n_store, 1>(&d, dst + col, dstStride);
        }

        src += srcStride;
        dst += dstStride;
    }
}

template<int width, int height>
void interp8_vert_ps_dotprod(const uint8_t *src, intptr_t srcStride,
                             int16_t *dst, intptr_t dstStride, int coeffIdx)
{
    const int N_TAPS = 8;

    src -= (N_TAPS / 2 - 1) * srcStride;

    const uint8x16x3_t merge_block_tbl = vld1q_u8_x3(dot_prod_merge_block_tbl);
    const int8x8_t filter = vmovn_s16(vld1q_s16(g_lumaFilter[coeffIdx]));

    if (width % 8 != 0)
    {
        uint8x8_t s_u8[11];
        int8x8_t s_s8[11];
        int8x16x2_t samples_tbl;
        int8x16_t s_lo[8];
        int8x16_t s_hi[8];
        const uint8_t *src_ptr = src;
        int16_t *dst_ptr = dst;

        if (width == 12)
        {
            load_u8x8xn<7>(src_ptr, srcStride, s_u8);

            // Transform sample range from uint8_t to int8_t.
            s_s8[0] = vreinterpret_s8_u8(vsub_u8(s_u8[0], vdup_n_u8(128)));
            s_s8[1] = vreinterpret_s8_u8(vsub_u8(s_u8[1], vdup_n_u8(128)));
            s_s8[2] = vreinterpret_s8_u8(vsub_u8(s_u8[2], vdup_n_u8(128)));
            s_s8[3] = vreinterpret_s8_u8(vsub_u8(s_u8[3], vdup_n_u8(128)));
            s_s8[4] = vreinterpret_s8_u8(vsub_u8(s_u8[4], vdup_n_u8(128)));
            s_s8[5] = vreinterpret_s8_u8(vsub_u8(s_u8[5], vdup_n_u8(128)));
            s_s8[6] = vreinterpret_s8_u8(vsub_u8(s_u8[6], vdup_n_u8(128)));
            s_s8[7] = vdup_n_s8(0);
            s_s8[8] = vdup_n_s8(0);
            s_s8[9] = vdup_n_s8(0);

            transpose_concat_8x4(s_s8 + 0, s_lo[0], s_hi[0]);
            transpose_concat_8x4(s_s8 + 1, s_lo[1], s_hi[1]);
            transpose_concat_8x4(s_s8 + 2, s_lo[2], s_hi[2]);
            transpose_concat_8x4(s_s8 + 3, s_lo[3], s_hi[3]);
            transpose_concat_8x4(s_s8 + 4, s_lo[4], s_hi[4]);
            transpose_concat_8x4(s_s8 + 5, s_lo[5], s_hi[5]);
            transpose_concat_8x4(s_s8 + 6, s_lo[6], s_hi[6]);

            src_ptr += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u8x8xn<4>(src_ptr, srcStride, s_u8 + 7);

                // Transform sample range from uint8_t to int8_t.
                s_s8[7] = vreinterpret_s8_u8(vsub_u8(s_u8[7], vdup_n_u8(128)));
                s_s8[8] = vreinterpret_s8_u8(vsub_u8(s_u8[8], vdup_n_u8(128)));
                s_s8[9] = vreinterpret_s8_u8(vsub_u8(s_u8[9], vdup_n_u8(128)));
                s_s8[10] = vreinterpret_s8_u8(vsub_u8(s_u8[10],
                                                      vdup_n_u8(128)));

                transpose_concat_8x4(s_s8 + 7, s_lo[7], s_hi[7]);

                // Merge new data into block from previous iteration.
                samples_tbl.val[0] = s_lo[3]; // rows 3, 4, 5, 6
                samples_tbl.val[1] = s_lo[7]; // rows 7, 8, 9, 10
                s_lo[4] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[0]);
                s_lo[5] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[1]);
                s_lo[6] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[2]);
                samples_tbl.val[0] = s_hi[3]; // rows 3, 4, 5, 6
                samples_tbl.val[1] = s_hi[7]; // rows 7, 8, 9, 10
                s_hi[4] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[0]);
                s_hi[5] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[1]);
                s_hi[6] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[2]);

                int16x8_t d[4];
                d[0] = filter8_8_ps_partial(s_lo[0], s_hi[0], s_lo[4], s_hi[4],
                                            filter);
                d[1] = filter8_8_ps_partial(s_lo[1], s_hi[1], s_lo[5], s_hi[5],
                                            filter);
                d[2] = filter8_8_ps_partial(s_lo[2], s_hi[2], s_lo[6], s_hi[6],
                                            filter);
                d[3] = filter8_8_ps_partial(s_lo[3], s_hi[3], s_lo[7], s_hi[7],
                                            filter);

                store_s16x8xn<4>(dst_ptr, dstStride, d);

                s_lo[0] = s_lo[4];
                s_lo[1] = s_lo[5];
                s_lo[2] = s_lo[6];
                s_lo[3] = s_lo[7];
                s_hi[0] = s_hi[4];
                s_hi[1] = s_hi[5];
                s_hi[2] = s_hi[6];
                s_hi[3] = s_hi[7];

                src_ptr += 4 * srcStride;
                dst_ptr += 4 * dstStride;
            }

            src_ptr = src + 8;
            dst_ptr = dst + 8;
        }

        load_u8x8xn<7>(src_ptr, srcStride, s_u8);

        // Transform sample range from uint8_t to int8_t.
        s_s8[0] = vreinterpret_s8_u8(vsub_u8(s_u8[0], vdup_n_u8(128)));
        s_s8[1] = vreinterpret_s8_u8(vsub_u8(s_u8[1], vdup_n_u8(128)));
        s_s8[2] = vreinterpret_s8_u8(vsub_u8(s_u8[2], vdup_n_u8(128)));
        s_s8[3] = vreinterpret_s8_u8(vsub_u8(s_u8[3], vdup_n_u8(128)));
        s_s8[4] = vreinterpret_s8_u8(vsub_u8(s_u8[4], vdup_n_u8(128)));
        s_s8[5] = vreinterpret_s8_u8(vsub_u8(s_u8[5], vdup_n_u8(128)));
        s_s8[6] = vreinterpret_s8_u8(vsub_u8(s_u8[6], vdup_n_u8(128)));
        s_s8[7] = vdup_n_s8(0);
        s_s8[8] = vdup_n_s8(0);
        s_s8[9] = vdup_n_s8(0);

        transpose_concat_4x4(s_s8 + 0, s_lo[0]);
        transpose_concat_4x4(s_s8 + 1, s_lo[1]);
        transpose_concat_4x4(s_s8 + 2, s_lo[2]);
        transpose_concat_4x4(s_s8 + 3, s_lo[3]);
        transpose_concat_4x4(s_s8 + 4, s_lo[4]);
        transpose_concat_4x4(s_s8 + 5, s_lo[5]);
        transpose_concat_4x4(s_s8 + 6, s_lo[6]);

        src_ptr += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            load_u8x8xn<4>(src_ptr, srcStride, s_u8 + 7);

            // Transform sample range from uint8_t to int8_t.
            s_s8[7] = vreinterpret_s8_u8(vsub_u8(s_u8[7], vdup_n_u8(128)));
            s_s8[8] = vreinterpret_s8_u8(vsub_u8(s_u8[8], vdup_n_u8(128)));
            s_s8[9] = vreinterpret_s8_u8(vsub_u8(s_u8[9], vdup_n_u8(128)));
            s_s8[10] = vreinterpret_s8_u8(vsub_u8(s_u8[10], vdup_n_u8(128)));

            transpose_concat_4x4(s_s8 + 7, s_lo[7]);

            // Merge new data into block from previous iteration.
            samples_tbl.val[0] = s_lo[3]; // rows 3, 4, 5, 6
            samples_tbl.val[1] = s_lo[7]; // rows 7, 8, 9, 10
            s_lo[4] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[0]);
            s_lo[5] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[1]);
            s_lo[6] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[2]);

            int16x4_t d[4];
            d[0] = filter8_4_ps_partial(s_lo[0], s_lo[4], filter);
            d[1] = filter8_4_ps_partial(s_lo[1], s_lo[5], filter);
            d[2] = filter8_4_ps_partial(s_lo[2], s_lo[6], filter);
            d[3] = filter8_4_ps_partial(s_lo[3], s_lo[7], filter);

            store_s16x4xn<4>(dst_ptr, dstStride, d);

            s_lo[0] = s_lo[4];
            s_lo[1] = s_lo[5];
            s_lo[2] = s_lo[6];
            s_lo[3] = s_lo[7];

            src_ptr += 4 * srcStride;
            dst_ptr += 4 * dstStride;
        }
    }
    else
    {
        for (int col = 0; col < width; col += 8)
        {
            const uint8_t *src_ptr = src + col;
            int16_t *dst_ptr = dst + col;
            uint8x8_t s_u8[11];
            int8x8_t s_s8[11];
            int8x16x2_t samples_tbl;
            int8x16_t s_lo[8];
            int8x16_t s_hi[8];

            load_u8x8xn<7>(src_ptr, srcStride, s_u8);

            // Transform sample range from uint8_t to int8_t.
            s_s8[0] = vreinterpret_s8_u8(vsub_u8(s_u8[0], vdup_n_u8(128)));
            s_s8[1] = vreinterpret_s8_u8(vsub_u8(s_u8[1], vdup_n_u8(128)));
            s_s8[2] = vreinterpret_s8_u8(vsub_u8(s_u8[2], vdup_n_u8(128)));
            s_s8[3] = vreinterpret_s8_u8(vsub_u8(s_u8[3], vdup_n_u8(128)));
            s_s8[4] = vreinterpret_s8_u8(vsub_u8(s_u8[4], vdup_n_u8(128)));
            s_s8[5] = vreinterpret_s8_u8(vsub_u8(s_u8[5], vdup_n_u8(128)));
            s_s8[6] = vreinterpret_s8_u8(vsub_u8(s_u8[6], vdup_n_u8(128)));
            s_s8[7] = vdup_n_s8(0);
            s_s8[8] = vdup_n_s8(0);
            s_s8[9] = vdup_n_s8(0);

            transpose_concat_8x4(s_s8 + 0, s_lo[0], s_hi[0]);
            transpose_concat_8x4(s_s8 + 1, s_lo[1], s_hi[1]);
            transpose_concat_8x4(s_s8 + 2, s_lo[2], s_hi[2]);
            transpose_concat_8x4(s_s8 + 3, s_lo[3], s_hi[3]);
            transpose_concat_8x4(s_s8 + 4, s_lo[4], s_hi[4]);
            transpose_concat_8x4(s_s8 + 5, s_lo[5], s_hi[5]);
            transpose_concat_8x4(s_s8 + 6, s_lo[6], s_hi[6]);

            src_ptr += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u8x8xn<4>(src_ptr, srcStride, s_u8 + 7);

                // Transform sample range from uint8_t to int8_t.
                s_s8[7] = vreinterpret_s8_u8(vsub_u8(s_u8[7], vdup_n_u8(128)));
                s_s8[8] = vreinterpret_s8_u8(vsub_u8(s_u8[8], vdup_n_u8(128)));
                s_s8[9] = vreinterpret_s8_u8(vsub_u8(s_u8[9], vdup_n_u8(128)));
                s_s8[10] = vreinterpret_s8_u8(vsub_u8(s_u8[10],
                                                      vdup_n_u8(128)));

                transpose_concat_8x4(s_s8 + 7, s_lo[7], s_hi[7]);

                // Merge new data into block from previous iteration.
                samples_tbl.val[0] = s_lo[3]; // rows 3, 4, 5, 6
                samples_tbl.val[1] = s_lo[7]; // rows 7, 8, 9, 10
                s_lo[4] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[0]);
                s_lo[5] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[1]);
                s_lo[6] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[2]);
                samples_tbl.val[0] = s_hi[3]; // rows 3, 4, 5, 6
                samples_tbl.val[1] = s_hi[7]; // rows 7, 8, 9, 10
                s_hi[4] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[0]);
                s_hi[5] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[1]);
                s_hi[6] = vqtbl2q_s8(samples_tbl, merge_block_tbl.val[2]);

                int16x8_t d[4];
                d[0] = filter8_8_ps_partial(s_lo[0], s_hi[0], s_lo[4], s_hi[4],
                                            filter);
                d[1] = filter8_8_ps_partial(s_lo[1], s_hi[1], s_lo[5], s_hi[5],
                                            filter);
                d[2] = filter8_8_ps_partial(s_lo[2], s_hi[2], s_lo[6], s_hi[6],
                                            filter);
                d[3] = filter8_8_ps_partial(s_lo[3], s_hi[3], s_lo[7], s_hi[7],
                                            filter);

                store_s16x8xn<4>(dst_ptr, dstStride, d);

                s_lo[0] = s_lo[4];
                s_lo[1] = s_lo[5];
                s_lo[2] = s_lo[6];
                s_lo[3] = s_lo[7];
                s_hi[0] = s_hi[4];
                s_hi[1] = s_hi[5];
                s_hi[2] = s_hi[6];
                s_hi[3] = s_hi[7];

                src_ptr += 4 * srcStride;
                dst_ptr += 4 * dstStride;
            }
        }
    }
}

// Declaration for use in interp_hv_pp_dotprod().
template<int N, int width, int height>
void interp_vert_sp_neon(const int16_t *src, intptr_t srcStride, uint8_t *dst,
                         intptr_t dstStride, int coeffIdx);

// Implementation of luma_hvpp, using Neon DotProd implementation for the
// horizontal part, and Armv8.0 Neon implementation for the vertical part.
template<int width, int height>
void interp_hv_pp_dotprod(const pixel *src, intptr_t srcStride, pixel *dst,
                          intptr_t dstStride, int idxX, int idxY)
{
    const int N_TAPS = 8;
    ALIGN_VAR_32(int16_t, immed[width * (height + N_TAPS - 1)]);

    interp8_horiz_ps_dotprod<width, height>(src, srcStride, immed, width, idxX,
                                            1);
    interp_vert_sp_neon<N_TAPS, width, height>(immed + (N_TAPS / 2 - 1) * width,
                                               width, dst, dstStride, idxY);
}

#define LUMA_DOTPROD(W, H) \
        p.pu[LUMA_ ## W ## x ## H].luma_hpp = interp8_horiz_pp_dotprod<W, H>; \
        p.pu[LUMA_ ## W ## x ## H].luma_hps = interp8_horiz_ps_dotprod<W, H>; \
        p.pu[LUMA_ ## W ## x ## H].luma_vps = interp8_vert_ps_dotprod<W, H>;  \
        p.pu[LUMA_ ## W ## x ## H].luma_hvpp = interp_hv_pp_dotprod<W, H>;

#define CHROMA_420_DOTPROD(W, H) \
        p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_hpp = \
            interp4_horiz_pp_dotprod<W, H>; \
        p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_hps = \
            interp4_horiz_ps_dotprod<W, H>;

#define CHROMA_422_DOTPROD(W, H) \
        p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_hpp = \
            interp4_horiz_pp_dotprod<W, H>; \
        p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_hps = \
            interp4_horiz_ps_dotprod<W, H>;

#define CHROMA_444_DOTPROD(W, H) \
        p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_hpp = \
            interp4_horiz_pp_dotprod<W, H>; \
        p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_hps = \
            interp4_horiz_ps_dotprod<W, H>;

void setupFilterPrimitives_neon_dotprod(EncoderPrimitives &p)
{
    LUMA_DOTPROD(4, 4);
    LUMA_DOTPROD(4, 8);
    LUMA_DOTPROD(4, 16);
    LUMA_DOTPROD(12, 16);
    LUMA_DOTPROD(8, 4);
    LUMA_DOTPROD(8, 8);
    LUMA_DOTPROD(8, 16);
    LUMA_DOTPROD(8, 32);
    LUMA_DOTPROD(16, 4);
    LUMA_DOTPROD(16, 8);
    LUMA_DOTPROD(16, 12);
    LUMA_DOTPROD(16, 16);
    LUMA_DOTPROD(16, 32);
    LUMA_DOTPROD(16, 64);
    LUMA_DOTPROD(24, 32);
    LUMA_DOTPROD(32, 8);
    LUMA_DOTPROD(32, 16);
    LUMA_DOTPROD(32, 24);
    LUMA_DOTPROD(32, 32);
    LUMA_DOTPROD(32, 64);
    LUMA_DOTPROD(48, 64);
    LUMA_DOTPROD(64, 16);
    LUMA_DOTPROD(64, 32);
    LUMA_DOTPROD(64, 48);
    LUMA_DOTPROD(64, 64);

    CHROMA_420_DOTPROD(2, 4);
    CHROMA_420_DOTPROD(2, 8);
    CHROMA_420_DOTPROD(4, 2);
    CHROMA_420_DOTPROD(4, 4);
    CHROMA_420_DOTPROD(4, 8);
    CHROMA_420_DOTPROD(4, 16);
    CHROMA_420_DOTPROD(6, 8);
    CHROMA_420_DOTPROD(12, 16);
    CHROMA_420_DOTPROD(8, 2);
    CHROMA_420_DOTPROD(8, 4);
    CHROMA_420_DOTPROD(8, 6);
    CHROMA_420_DOTPROD(8, 8);
    CHROMA_420_DOTPROD(8, 16);
    CHROMA_420_DOTPROD(8, 32);
    CHROMA_420_DOTPROD(16, 4);
    CHROMA_420_DOTPROD(16, 8);
    CHROMA_420_DOTPROD(16, 12);
    CHROMA_420_DOTPROD(16, 16);
    CHROMA_420_DOTPROD(16, 32);
    CHROMA_420_DOTPROD(24, 32);
    CHROMA_420_DOTPROD(32, 8);
    CHROMA_420_DOTPROD(32, 16);
    CHROMA_420_DOTPROD(32, 24);
    CHROMA_420_DOTPROD(32, 32);

    CHROMA_422_DOTPROD(2, 8);
    CHROMA_422_DOTPROD(2, 16);
    CHROMA_422_DOTPROD(4, 4);
    CHROMA_422_DOTPROD(4, 8);
    CHROMA_422_DOTPROD(4, 16);
    CHROMA_422_DOTPROD(4, 32);
    CHROMA_422_DOTPROD(6, 16);
    CHROMA_422_DOTPROD(12, 32);
    CHROMA_422_DOTPROD(8, 4);
    CHROMA_422_DOTPROD(8, 8);
    CHROMA_422_DOTPROD(8, 12);
    CHROMA_422_DOTPROD(8, 16);
    CHROMA_422_DOTPROD(8, 32);
    CHROMA_422_DOTPROD(8, 64);
    CHROMA_422_DOTPROD(16, 8);
    CHROMA_422_DOTPROD(16, 16);
    CHROMA_422_DOTPROD(16, 24);
    CHROMA_422_DOTPROD(16, 32);
    CHROMA_422_DOTPROD(16, 64);
    CHROMA_422_DOTPROD(24, 64);
    CHROMA_422_DOTPROD(32, 16);
    CHROMA_422_DOTPROD(32, 32);
    CHROMA_422_DOTPROD(32, 48);
    CHROMA_422_DOTPROD(32, 64);

    CHROMA_444_DOTPROD(4, 4);
    CHROMA_444_DOTPROD(4, 8);
    CHROMA_444_DOTPROD(4, 16);
    CHROMA_444_DOTPROD(12, 16);
    CHROMA_444_DOTPROD(8, 4);
    CHROMA_444_DOTPROD(8, 8);
    CHROMA_444_DOTPROD(8, 16);
    CHROMA_444_DOTPROD(8, 32);
    CHROMA_444_DOTPROD(16, 4);
    CHROMA_444_DOTPROD(16, 8);
    CHROMA_444_DOTPROD(16, 12);
    CHROMA_444_DOTPROD(16, 16);
    CHROMA_444_DOTPROD(16, 32);
    CHROMA_444_DOTPROD(16, 64);
    CHROMA_444_DOTPROD(24, 32);
    CHROMA_444_DOTPROD(32, 8);
    CHROMA_444_DOTPROD(32, 16);
    CHROMA_444_DOTPROD(32, 24);
    CHROMA_444_DOTPROD(32, 32);
    CHROMA_444_DOTPROD(32, 64);
    CHROMA_444_DOTPROD(48, 64);
    CHROMA_444_DOTPROD(64, 16);
    CHROMA_444_DOTPROD(64, 32);
    CHROMA_444_DOTPROD(64, 48);
    CHROMA_444_DOTPROD(64, 64);
}
}

#else // !HIGH_BIT_DEPTH
namespace X265_NS {
void setupFilterPrimitives_neon_dotprod(EncoderPrimitives &)
{
}
}
#endif // !HIGH_BIT_DEPTH
