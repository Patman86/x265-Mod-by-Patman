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

#if defined(HAVE_NEON_I8MM)
#include "filter-neon-i8mm.h"
#if !HIGH_BIT_DEPTH

#include "mem-neon.h"

#include <arm_neon.h>

namespace {
static const uint8_t dotprod_permute_tbl[48] = {
    0, 1,  2,  3, 1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5, 6,
    4, 5,  6,  7, 5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10,
    8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14
};

static const uint8_t matmul_permute_tbl[2][32] = {
    // Permute for luma filter 3.
    { 0,  1,  2,  3,  4,  5,  6,  7,  2,  3,  4,  5,  6,  7,  8,  9,
      4,  5,  6,  7,  8,  9, 10, 11,  6,  7,  8,  9, 10, 11, 12, 13 },
    // Permute for luma filter 1.
    { 1,  2,  3,  4,  5,  6,  7,  8,  3,  4,  5,  6,  7,  8,  9, 10,
      5,  6,  7,  8,  9, 10, 11, 12,  7,  8,  9, 10, 11, 12, 13, 14 }
};

static const int8_t matmul_luma_filter[2][16] = {
    { -1, 4, -10, 58, 17, -5, 1, 0, 0, -1, 4, -10, 58, 17, -5, 1 },
    { 1, -5, 17, 58, -10, 4, -1, 0, 0, 1, -5, 17, 58, -10, 4, -1 }
};

uint8x8_t inline filter8_8_pp(uint8x16_t samples, const int8x8_t filter,
                              const uint8x16x3_t tbl)
{
    // Permute input samples for dot product.
    // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
    uint8x16_t perm_s0 = vqtbl1q_u8(samples, tbl.val[0]);
    // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
    uint8x16_t perm_s1 = vqtbl1q_u8(samples, tbl.val[1]);
    // { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 }
    uint8x16_t perm_S2 = vqtbl1q_u8(samples, tbl.val[2]);

    int32x4_t dotprod_lo = vusdotq_lane_s32(vdupq_n_s32(0), perm_s0, filter, 0);
    dotprod_lo = vusdotq_lane_s32(dotprod_lo, perm_s1, filter, 1);
    int32x4_t dotprod_hi = vusdotq_lane_s32(vdupq_n_s32(0), perm_s1, filter, 0);
    dotprod_hi = vusdotq_lane_s32(dotprod_hi, perm_S2, filter, 1);

    // Narrow and combine.
    int16x8_t dotprod = vcombine_s16(vmovn_s32(dotprod_lo),
                                     vmovn_s32(dotprod_hi));
    return vqrshrun_n_s16(dotprod, IF_FILTER_PREC);
}

void inline init_sample_permute(uint8x8_t *samples, const uint8x16x3_t tbl,
                                uint8x16_t *d)
{
    // Permute input samples for dot product.
    // { 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6 }
    d[0] = vqtbl1q_u8(vcombine_u8(samples[0], vdup_n_u8(0)), tbl.val[0]);
    d[1] = vqtbl1q_u8(vcombine_u8(samples[1], vdup_n_u8(0)), tbl.val[0]);
    d[2] = vqtbl1q_u8(vcombine_u8(samples[2], vdup_n_u8(0)), tbl.val[0]);
    d[3] = vqtbl1q_u8(vcombine_u8(samples[3], vdup_n_u8(0)), tbl.val[0]);
}

uint8x8_t inline filter8_8_pp_reuse(uint8x16_t samples, const int8x8_t filter,
                                    const uint8x16x3_t tbl, uint8x16_t &perm_s0)
{
    // Permute input samples for dot product.
    // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
    // Already in perm_s0.
    // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
    uint8x16_t perm_s1 = vqtbl1q_u8(samples, tbl.val[1]);
    // { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 }
    uint8x16_t perm_s2 = vqtbl1q_u8(samples, tbl.val[2]);

    int32x4_t dotprod_lo = vusdotq_lane_s32(vdupq_n_s32(0), perm_s0, filter, 0);
    dotprod_lo = vusdotq_lane_s32(dotprod_lo, perm_s1, filter, 1);
    int32x4_t dotprod_hi = vusdotq_lane_s32(vdupq_n_s32(0), perm_s1, filter, 0);
    dotprod_hi = vusdotq_lane_s32(dotprod_hi, perm_s2, filter, 1);

    // Save for re-use in next iteration.
    perm_s0 = perm_s2;

    // Narrow and combine.
    int16x8_t dotprod = vcombine_s16(vmovn_s32(dotprod_lo),
                                     vmovn_s32(dotprod_hi));
    return vqrshrun_n_s16(dotprod, IF_FILTER_PREC);
}

uint8x8_t inline filter8_8_pp_matmul(uint8x16_t samples, const int8x16_t filter,
                                     const uint8x16x2_t tbl)
{
    // Permute input samples for 8x2 by 2x8 matrix multiply.
    uint8x16_t perm_s0 = vqtbl1q_u8(samples, tbl.val[0]);
    uint8x16_t perm_s1 = vqtbl1q_u8(samples, tbl.val[1]);

    int32x4_t matmul_lo = vusmmlaq_s32(vdupq_n_s32(0), perm_s0, filter);
    int32x4_t matmul_hi = vusmmlaq_s32(vdupq_n_s32(0), perm_s1, filter);

    // Narrow and combine.
    int16x8_t matmul = vcombine_s16(vmovn_s32(matmul_lo), vmovn_s32(matmul_hi));
    return vqrshrun_n_s16(matmul, IF_FILTER_PREC);
}

int16x4_t inline filter8_4_ps(uint8x16_t samples, const int8x8_t filter,
                              const int16x8_t constant, const uint8x16x3_t tbl)
{
    // Permute input samples for dot product.
    // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
    uint8x16_t perm_s0 = vqtbl1q_u8(samples, tbl.val[0]);
    // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
    uint8x16_t perm_s1 = vqtbl1q_u8(samples, tbl.val[1]);

    int32x4_t dotprod = vusdotq_lane_s32(vdupq_n_s32(0), perm_s0, filter, 0);
    dotprod = vusdotq_lane_s32(dotprod, perm_s1, filter, 1);

    // Narrow.
    return vadd_s16(vmovn_s32(dotprod), vget_low_s16(constant));
}

int16x8_t inline filter8_8_ps(uint8x16_t samples, const int8x8_t filter,
                              const int16x8_t constant, const uint8x16x3_t tbl)
{
    // Permute input samples for dot product.
    // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
    uint8x16_t perm_s0 = vqtbl1q_u8(samples, tbl.val[0]);
    // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
    uint8x16_t perm_s1 = vqtbl1q_u8(samples, tbl.val[1]);
    // { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 }
    uint8x16_t perm_S2 = vqtbl1q_u8(samples, tbl.val[2]);

    int32x4_t dotprod_lo = vusdotq_lane_s32(vdupq_n_s32(0), perm_s0, filter, 0);
    dotprod_lo = vusdotq_lane_s32(dotprod_lo, perm_s1, filter, 1);
    int32x4_t dotprod_hi = vusdotq_lane_s32(vdupq_n_s32(0), perm_s1, filter, 0);
    dotprod_hi = vusdotq_lane_s32(dotprod_hi, perm_S2, filter, 1);

    // Narrow and combine.
    int16x8_t dotprod = vcombine_s16(vmovn_s32(dotprod_lo),
                                     vmovn_s32(dotprod_hi));
    return vaddq_s16(dotprod, constant);
}

int16x8_t inline filter8_8_ps_reuse(uint8x16_t samples, const int8x8_t filter,
                                    const int16x8_t constant,
                                    const uint8x16x3_t tbl, uint8x16_t &perm_s0)
{
    // Permute input samples for dot product.
    // { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 }
    // Already in perm_s0.
    // { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 }
    uint8x16_t perm_s1 = vqtbl1q_u8(samples, tbl.val[1]);
    // { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 }
    uint8x16_t perm_s2 = vqtbl1q_u8(samples, tbl.val[2]);

    int32x4_t dotprod_lo = vusdotq_lane_s32(vdupq_n_s32(0), perm_s0, filter, 0);
    dotprod_lo = vusdotq_lane_s32(dotprod_lo, perm_s1, filter, 1);
    int32x4_t dotprod_hi = vusdotq_lane_s32(vdupq_n_s32(0), perm_s1, filter, 0);
    dotprod_hi = vusdotq_lane_s32(dotprod_hi, perm_s2, filter, 1);

    // Save for re-use in next iteration.
    perm_s0 = perm_s2;

    // Narrow and combine.
    int16x8_t dotprod = vcombine_s16(vmovn_s32(dotprod_lo),
                                     vmovn_s32(dotprod_hi));
    return vaddq_s16(dotprod, constant);
}

int16x8_t inline filter8_8_ps_matmul(uint8x16_t samples, const int8x16_t filter,
                                     const int16x8_t constant,
                                     const uint8x16x2_t tbl)
{
    // Permute input samples for 8x2 by 2x8 matrix multiply.
    uint8x16_t perm_s0 = vqtbl1q_u8(samples, tbl.val[0]);
    uint8x16_t perm_s1 = vqtbl1q_u8(samples, tbl.val[1]);

    int32x4_t matmul_lo = vusmmlaq_s32(vdupq_n_s32(0), perm_s0, filter);
    int32x4_t matmul_hi = vusmmlaq_s32(vdupq_n_s32(0), perm_s1, filter);

    // Narrow and combine.
    int16x8_t matmul = vcombine_s16(vmovn_s32(matmul_lo), vmovn_s32(matmul_hi));
    return vaddq_s16(matmul, constant);
}

int16x4_t inline filter8_4_ps_matmul(uint8x16_t samples, const int8x16_t filter,
                                     const int16x8_t constant,
                                     const uint8x16x2_t tbl)
{
    // Permute input samples for 8x2 by 2x8 matrix multiply.
    uint8x16_t perm = vqtbl1q_u8(samples, tbl.val[0]);

    int32x4_t matmul = vusmmlaq_s32(vdupq_n_s32(0), perm, filter);

    return vadd_s16(vmovn_s32(matmul), vget_low_s16(constant));
}
} // Unnamed namespace.

namespace X265_NS {
template<int width, int height>
void inline interp8_horiz_pp_dotprod(const uint8_t *src, intptr_t srcStride,
                                     uint8_t *dst, intptr_t dstStride,
                                     int coeffIdx)
{
    const int N_TAPS = 8;
    src -= N_TAPS / 2 - 1;

    const uint8x16x3_t tbl = vld1q_u8_x3(dotprod_permute_tbl);
    const int8x8_t filter = vmovn_s16(vld1q_s16(g_lumaFilter[coeffIdx]));

    for (int row = 0; row < height; row += 4)
    {
        int col = 0;
        if (width >= 32)
        {
            // Peel first sample permute to enable passing between iterations.
            uint8x8_t s0[4];
            load_u8x8xn<4>(src, srcStride, s0);
            uint8x16_t ps0[4];
            init_sample_permute(s0, tbl, ps0);

            for (; (col + 16) <= width; col += 16)
            {
                uint8x16_t s_lo[4], s_hi[4];
                load_u8x16xn<4>(src + col + 0, srcStride, s_lo);
                load_u8x16xn<4>(src + col + 8, srcStride, s_hi);

                uint8x8_t d_lo[4];
                d_lo[0] = filter8_8_pp_reuse(s_lo[0], filter, tbl, ps0[0]);
                d_lo[1] = filter8_8_pp_reuse(s_lo[1], filter, tbl, ps0[1]);
                d_lo[2] = filter8_8_pp_reuse(s_lo[2], filter, tbl, ps0[2]);
                d_lo[3] = filter8_8_pp_reuse(s_lo[3], filter, tbl, ps0[3]);

                uint8x8_t d_hi[4];
                d_hi[0] = filter8_8_pp_reuse(s_hi[0], filter, tbl, ps0[0]);
                d_hi[1] = filter8_8_pp_reuse(s_hi[1], filter, tbl, ps0[1]);
                d_hi[2] = filter8_8_pp_reuse(s_hi[2], filter, tbl, ps0[2]);
                d_hi[3] = filter8_8_pp_reuse(s_hi[3], filter, tbl, ps0[3]);

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
                d[0] = filter8_8_pp(s[0], filter, tbl);
                d[1] = filter8_8_pp(s[1], filter, tbl);
                d[2] = filter8_8_pp(s[2], filter, tbl);
                d[3] = filter8_8_pp(s[3], filter, tbl);

                store_u8x8xn<4>(dst + col, dstStride, d);
            }
        }
        for (; col < width; col += 4)
        {
            uint8x16_t s[4];
            load_u8x16xn<4>(src + col, srcStride, s);

            uint8x8_t d[4];
            d[0] = filter8_8_pp(s[0], filter, tbl);
            d[1] = filter8_8_pp(s[1], filter, tbl);
            d[2] = filter8_8_pp(s[2], filter, tbl);
            d[3] = filter8_8_pp(s[3], filter, tbl);

            store_u8x4xn<4>(dst + col, dstStride, d);
        }

        src += 4 * srcStride;
        dst += 4 * dstStride;
    }
}

template<int coeffIdx, int width, int height>
void inline interp8_horiz_pp_matmul(const uint8_t *src, intptr_t srcStride,
                                    uint8_t *dst, intptr_t dstStride)
{
    const int N_TAPS = 8;
    src -= N_TAPS / 2 - 1;

    // coeffIdx is 1 or 3 for g_lumaFilter index.
    // Select filter and permute table from the first or second array indices.
    const int index = coeffIdx >> 1;
    const uint8x16x2_t tbl = vld1q_u8_x2(matmul_permute_tbl[index]);
    const int8x16_t filter = vld1q_s8(matmul_luma_filter[index]);

    for (int row = 0; row < height; row += 4)
    {
        int col = 0;
        if (width >= 32)
        {
            for (; (col + 16) <= width; col += 16)
            {
                uint8x16_t s_lo[4], s_hi[4];
                load_u8x16xn<4>(src + col + 0, srcStride, s_lo);
                load_u8x16xn<4>(src + col + 8, srcStride, s_hi);

                uint8x8_t d_lo[4];
                d_lo[0] = filter8_8_pp_matmul(s_lo[0], filter, tbl);
                d_lo[1] = filter8_8_pp_matmul(s_lo[1], filter, tbl);
                d_lo[2] = filter8_8_pp_matmul(s_lo[2], filter, tbl);
                d_lo[3] = filter8_8_pp_matmul(s_lo[3], filter, tbl);

                uint8x8_t d_hi[4];
                d_hi[0] = filter8_8_pp_matmul(s_hi[0], filter, tbl);
                d_hi[1] = filter8_8_pp_matmul(s_hi[1], filter, tbl);
                d_hi[2] = filter8_8_pp_matmul(s_hi[2], filter, tbl);
                d_hi[3] = filter8_8_pp_matmul(s_hi[3], filter, tbl);

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
                d[0] = filter8_8_pp_matmul(s[0], filter, tbl);
                d[1] = filter8_8_pp_matmul(s[1], filter, tbl);
                d[2] = filter8_8_pp_matmul(s[2], filter, tbl);
                d[3] = filter8_8_pp_matmul(s[3], filter, tbl);

                store_u8x8xn<4>(dst + col, dstStride, d);
            }
        }
        for (; col < width; col += 4)
        {
            uint8x16_t s[4];
            load_u8x16xn<4>(src + col, srcStride, s);

            uint8x8_t d[4];
            d[0] = filter8_8_pp_matmul(s[0], filter, tbl);
            d[1] = filter8_8_pp_matmul(s[1], filter, tbl);
            d[2] = filter8_8_pp_matmul(s[2], filter, tbl);
            d[3] = filter8_8_pp_matmul(s[3], filter, tbl);

            store_u8x4xn<4>(dst + col, dstStride, d);
        }

        src += 4 * srcStride;
        dst += 4 * dstStride;
    }
}

template<int width, int height>
void interp8_horiz_pp_i8mm(const uint8_t *src, intptr_t srcStride, uint8_t *dst,
                           intptr_t dstStride, int coeffIdx)
{
    switch (coeffIdx)
    {
    case 1:
        return interp8_horiz_pp_matmul<1, width, height>(src, srcStride, dst,
                                                         dstStride);
    case 2:
        return interp8_horiz_pp_dotprod<width, height>(src, srcStride, dst,
                                                       dstStride, coeffIdx);
    case 3:
        return interp8_horiz_pp_matmul<3, width, height>(src, srcStride, dst,
                                                         dstStride);
    }
}

template<int width, int height>
void inline interp8_horiz_ps_dotprod(const uint8_t *src, intptr_t srcStride,
                                     int16_t *dst, intptr_t dstStride,
                                     int coeffIdx, int isRowExt)
{
    const int offset = (unsigned)-IF_INTERNAL_OFFS;

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
    const int16x8_t c = vdupq_n_s16(offset);

    for (int row = 0; row + 4 <= blkheight; row += 4)
    {
        int col = 0;
        if (width >= 32)
        {
            // Peel first sample permute to enable passing between iterations.
            uint8x8_t s0[4];
            load_u8x8xn<4>(src, srcStride, s0);
            uint8x16_t ps0[4];
            init_sample_permute(s0, tbl, ps0);

            for (; col + 16 <= width; col += 16)
            {
                uint8x16_t s_lo[4], s_hi[4];
                load_u8x16xn<4>(src + col + 0, srcStride, s_lo);
                load_u8x16xn<4>(src + col + 8, srcStride, s_hi);

                int16x8_t d_lo[4];
                d_lo[0] = filter8_8_ps_reuse(s_lo[0], filter, c, tbl, ps0[0]);
                d_lo[1] = filter8_8_ps_reuse(s_lo[1], filter, c, tbl, ps0[1]);
                d_lo[2] = filter8_8_ps_reuse(s_lo[2], filter, c, tbl, ps0[2]);
                d_lo[3] = filter8_8_ps_reuse(s_lo[3], filter, c, tbl, ps0[3]);

                int16x8_t d_hi[4];
                d_hi[0] = filter8_8_ps_reuse(s_hi[0], filter, c, tbl, ps0[0]);
                d_hi[1] = filter8_8_ps_reuse(s_hi[1], filter, c, tbl, ps0[1]);
                d_hi[2] = filter8_8_ps_reuse(s_hi[2], filter, c, tbl, ps0[2]);
                d_hi[3] = filter8_8_ps_reuse(s_hi[3], filter, c, tbl, ps0[3]);

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
                d[0] = filter8_8_ps(s[0], filter, c, tbl);
                d[1] = filter8_8_ps(s[1], filter, c, tbl);
                d[2] = filter8_8_ps(s[2], filter, c, tbl);
                d[3] = filter8_8_ps(s[3], filter, c, tbl);

                store_s16x8xn<4>(dst + col, dstStride, d);
            }
        }
        for (; col < width; col += 4)
        {
            uint8x16_t s[4];
            load_u8x16xn<4>(src + col, srcStride, s);

            int16x4_t d[4];
            d[0] = filter8_4_ps(s[0], filter, c, tbl);
            d[1] = filter8_4_ps(s[1], filter, c, tbl);
            d[2] = filter8_4_ps(s[2], filter, c, tbl);
            d[3] = filter8_4_ps(s[3], filter, c, tbl);

            store_s16x4xn<4>(dst + col, dstStride, d);
        }

        src += 4 * srcStride;
        dst += 4 * dstStride;
    }

    if (isRowExt)
    {
        // process final 3 rows
        int col = 0;
        for (; (col + 8) <= width; col += 8)
        {
            uint8x16_t s[3];
            load_u8x16xn<3>(src + col, srcStride, s);

            int16x8_t d[3];
            d[0] = filter8_8_ps(s[0], filter, c, tbl);
            d[1] = filter8_8_ps(s[1], filter, c, tbl);
            d[2] = filter8_8_ps(s[2], filter, c, tbl);

            store_s16x8xn<3>(dst + col, dstStride, d);
        }

        for (; col < width; col += 4)
        {
            uint8x16_t s[3];
            load_u8x16xn<3>(src + col, srcStride, s);

            int16x4_t d[3];
            d[0] = filter8_4_ps(s[0], filter, c, tbl);
            d[1] = filter8_4_ps(s[1], filter, c, tbl);
            d[2] = filter8_4_ps(s[2], filter, c, tbl);

            store_s16x4xn<3>(dst + col, dstStride, d);
        }
    }
}

template<int coeffIdx, int width, int height>
void inline interp8_horiz_ps_matmul(const uint8_t *src, intptr_t srcStride,
                                    int16_t *dst, intptr_t dstStride,
                                    int isRowExt)
{
    const int offset = (unsigned)-IF_INTERNAL_OFFS;

    const int N_TAPS = 8;
    int blkheight = height;

    src -= N_TAPS / 2 - 1;
    if (isRowExt)
    {
        src -= (N_TAPS / 2 - 1) * srcStride;
        blkheight += N_TAPS - 1;
    }

    // coeffIdx is 1 or 3 for g_lumaFilter index.
    // Select filter and permute table from the first or second array indices.
    const int index = coeffIdx >> 1;
    const uint8x16x2_t tbl = vld1q_u8_x2(matmul_permute_tbl[index]);
    const int8x16_t filter = vld1q_s8(matmul_luma_filter[index]);

    const int16x8_t c = vdupq_n_s16(offset);

    for (int row = 0; row + 4 <= blkheight; row += 4)
    {
        int col = 0;
        if (width >= 32)
        {
            for (; col + 16 <= width; col += 16)
            {
                uint8x16_t s_lo[4], s_hi[4];
                load_u8x16xn<4>(src + col + 0, srcStride, s_lo);
                load_u8x16xn<4>(src + col + 8, srcStride, s_hi);

                int16x8_t d_lo[4];
                d_lo[0] = filter8_8_ps_matmul(s_lo[0], filter, c, tbl);
                d_lo[1] = filter8_8_ps_matmul(s_lo[1], filter, c, tbl);
                d_lo[2] = filter8_8_ps_matmul(s_lo[2], filter, c, tbl);
                d_lo[3] = filter8_8_ps_matmul(s_lo[3], filter, c, tbl);

                int16x8_t d_hi[4];
                d_hi[0] = filter8_8_ps_matmul(s_hi[0], filter, c, tbl);
                d_hi[1] = filter8_8_ps_matmul(s_hi[1], filter, c, tbl);
                d_hi[2] = filter8_8_ps_matmul(s_hi[2], filter, c, tbl);
                d_hi[3] = filter8_8_ps_matmul(s_hi[3], filter, c, tbl);

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
                d[0] = filter8_8_ps_matmul(s[0], filter, c, tbl);
                d[1] = filter8_8_ps_matmul(s[1], filter, c, tbl);
                d[2] = filter8_8_ps_matmul(s[2], filter, c, tbl);
                d[3] = filter8_8_ps_matmul(s[3], filter, c, tbl);

                store_s16x8xn<4>(dst + col, dstStride, d);
            }
        }
        for (; col < width; col += 4)
        {
            uint8x16_t s[4];
            load_u8x16xn<4>(src + col, srcStride, s);

            int16x4_t d[4];
            d[0] = filter8_4_ps_matmul(s[0], filter, c, tbl);
            d[1] = filter8_4_ps_matmul(s[1], filter, c, tbl);
            d[2] = filter8_4_ps_matmul(s[2], filter, c, tbl);
            d[3] = filter8_4_ps_matmul(s[3], filter, c, tbl);

            store_s16x4xn<4>(dst + col, dstStride, d);
        }

        src += 4 * srcStride;
        dst += 4 * dstStride;
    }

    if (isRowExt)
    {
        // process final 3 rows
        int col = 0;
        for (; (col + 8) <= width; col += 8)
        {
            uint8x16_t s[3];
            load_u8x16xn<3>(src + col, srcStride, s);

            int16x8_t d[3];
            d[0] = filter8_8_ps_matmul(s[0], filter, c, tbl);
            d[1] = filter8_8_ps_matmul(s[1], filter, c, tbl);
            d[2] = filter8_8_ps_matmul(s[2], filter, c, tbl);

            store_s16x8xn<3>(dst + col, dstStride, d);
        }

        for (; col < width; col += 4)
        {
            uint8x16_t s[3];
            load_u8x16xn<3>(src + col, srcStride, s);

            int16x4_t d[3];
            d[0] = filter8_4_ps_matmul(s[0], filter, c, tbl);
            d[1] = filter8_4_ps_matmul(s[1], filter, c, tbl);
            d[2] = filter8_4_ps_matmul(s[2], filter, c, tbl);

            store_s16x4xn<3>(dst + col, dstStride, d);
        }
    }
}

template<int width, int height>
void interp8_horiz_ps_i8mm(const uint8_t *src, intptr_t srcStride, int16_t *dst,
                           intptr_t dstStride, int coeffIdx, int isRowExt)
{
    switch (coeffIdx)
    {
    case 1:
        return interp8_horiz_ps_matmul<1, width, height>(src, srcStride, dst,
                                                         dstStride, isRowExt);
    case 2:
        return interp8_horiz_ps_dotprod<width, height>(src, srcStride, dst,
                                                       dstStride, coeffIdx,
                                                       isRowExt);
    case 3:
        return interp8_horiz_ps_matmul<3, width, height>(src, srcStride, dst,
                                                         dstStride, isRowExt);
    }
}

#define LUMA_I8MM(W, H) \
        p.pu[LUMA_ ## W ## x ## H].luma_hpp = interp8_horiz_pp_i8mm<W, H>; \
        p.pu[LUMA_ ## W ## x ## H].luma_hps = interp8_horiz_ps_i8mm<W, H>;

void setupFilterPrimitives_neon_i8mm(EncoderPrimitives &p)
{
    LUMA_I8MM(4, 4);
    LUMA_I8MM(4, 8);
    LUMA_I8MM(4, 16);
    LUMA_I8MM(12, 16);
    LUMA_I8MM(8, 4);
    LUMA_I8MM(8, 8);
    LUMA_I8MM(8, 16);
    LUMA_I8MM(8, 32);
    LUMA_I8MM(16, 4);
    LUMA_I8MM(16, 8);
    LUMA_I8MM(16, 12);
    LUMA_I8MM(16, 16);
    LUMA_I8MM(16, 32);
    LUMA_I8MM(16, 64);
    LUMA_I8MM(24, 32);
    LUMA_I8MM(32, 8);
    LUMA_I8MM(32, 16);
    LUMA_I8MM(32, 24);
    LUMA_I8MM(32, 32);
    LUMA_I8MM(32, 64);
    LUMA_I8MM(48, 64);
    LUMA_I8MM(64, 16);
    LUMA_I8MM(64, 32);
    LUMA_I8MM(64, 48);
    LUMA_I8MM(64, 64);
}
}

#else // if !HIGH_BIT_DEPTH
namespace X265_NS {
void setupFilterPrimitives_neon_i8mm(EncoderPrimitives &)
{
}
}
#endif // !HIGH_BIT_DEPTH

#endif // defined(HAVE_NEON_I8MM)
