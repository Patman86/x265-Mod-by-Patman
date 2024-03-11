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

#define LUMA_DOTPROD(W, H) \
        p.pu[LUMA_ ## W ## x ## H].luma_hpp = interp8_horiz_pp_dotprod<W, H>;

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
}
}

#else // !HIGH_BIT_DEPTH
namespace X265_NS {
void setupFilterPrimitives_neon_dotprod(EncoderPrimitives &)
{
}
}
#endif // !HIGH_BIT_DEPTH
