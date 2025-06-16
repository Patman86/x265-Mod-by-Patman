/*****************************************************************************
 * Copyright (C) 2021-2025 MulticoreWare, Inc
 *
 * Authors: Liwei Wang <liwei@multicorewareinc.com>
 *          Jonathan Swinney <jswinney@amazon.com>
 *          Hari Limaye <hari.limaye@arm.com>
 *          Gerda Zsejke More <gerdazsejke.more@arm.com>
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

#if HAVE_NEON

#include "filter-prim.h"
#include "mem-neon.h"

#include <arm_neon.h>

namespace {
#if !HIGH_BIT_DEPTH
// This is to use with vtbl2q_s32_s16.
// Extract the middle two bytes from each 32-bit element in a vector, using these byte
// indices.
static const uint8_t vert_shr_tbl[16] = {
    1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30
};
#endif

#if HIGH_BIT_DEPTH
#define SHIFT_INTERP_PS (IF_FILTER_PREC - (IF_INTERNAL_PREC - X265_DEPTH))
#endif

template<bool coeff4>
void inline filter4_s16x4_sum(const int16x4_t *s, const int16x4_t f,
                              const int32x4_t c, int32x4_t &sum)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        int16x4_t sum03 = vadd_s16(s[0], s[3]);
        int16x4_t sum12 = vadd_s16(s[1], s[2]);

        sum = vmlal_n_s16(c, sum12, 9);
        sum = vsubw_s16(sum, sum03);
    }
    else
    {
        sum = vmlal_lane_s16(c, s[0], f, 0);
        sum = vmlal_lane_s16(sum, s[1], f, 1);
        sum = vmlal_lane_s16(sum, s[2], f, 2);
        sum = vmlal_lane_s16(sum, s[3], f, 3);
    }
}

template<bool coeff4, int shift>
void inline filter4_ss_s16x4(const int16x4_t *s, const int16x4_t f,
                             const int32x4_t c, int16x4_t &d)
{
    int32x4_t sum;

    filter4_s16x4_sum<coeff4>(s, f, c, sum);

    // We divided filter values by 4 so subtract 2 from right shift in case of filter
    // coefficient 4.
    const int shift_offset = coeff4 ? shift - 2 : shift;

    d = vshrn_n_s32(sum, shift_offset);
}

template<bool coeff4, int shift>
void inline filter4x2_sp_s16x4(const int16x4_t *s0, const int16x4_t *s1,
                               const int16x4_t f, const int32x4_t c,
                               const uint8x16_t shr_tbl, uint8x8_t &d)
{
    int32x4_t sum0, sum1;

    filter4_s16x4_sum<coeff4>(s0, f, c, sum0);
    filter4_s16x4_sum<coeff4>(s1, f, c, sum1);
    int16x8_t sum = vtbl2q_s32_s16(sum0, sum1, shr_tbl);

    // We divided filter values by 4 so subtract 2 from right shift in case of filter
    // coefficient 4.
    const int shift_offset = coeff4 ? shift - 2 : shift;

    d = vqshrun_n_s16(sum, shift_offset);
}

template<bool coeff4>
void inline filter4_s16x8_sum(const int16x8_t *s, const int16x4_t f,
                              const int32x4_t c, int32x4_t &sum_lo, int32x4_t &sum_hi)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        int16x8_t sum03 = vaddq_s16(s[0], s[3]);
        int16x8_t sum12 = vaddq_s16(s[1], s[2]);

        sum_lo = vmlal_n_s16(c, vget_low_s16(sum12), 9);
        sum_hi = vmlal_n_s16(c, vget_high_s16(sum12), 9);

        sum_lo = vsubw_s16(sum_lo, vget_low_s16(sum03));
        sum_hi = vsubw_s16(sum_hi, vget_high_s16(sum03));
    }
    else
    {
        sum_lo = vmlal_lane_s16(c, vget_low_s16(s[0]), f, 0);
        sum_lo = vmlal_lane_s16(sum_lo, vget_low_s16(s[1]), f, 1);
        sum_lo = vmlal_lane_s16(sum_lo, vget_low_s16(s[2]), f, 2);
        sum_lo = vmlal_lane_s16(sum_lo, vget_low_s16(s[3]), f, 3);

        sum_hi = vmlal_lane_s16(c, vget_high_s16(s[0]), f, 0);
        sum_hi = vmlal_lane_s16(sum_hi, vget_high_s16(s[1]), f, 1);
        sum_hi = vmlal_lane_s16(sum_hi, vget_high_s16(s[2]), f, 2);
        sum_hi = vmlal_lane_s16(sum_hi, vget_high_s16(s[3]), f, 3);
    }
}

template<bool coeff4, int shift>
void inline filter4_ss_s16x8(const int16x8_t *s, const int16x4_t f,
                             const int32x4_t c, int16x8_t &d)
{
    int32x4_t sum_lo, sum_hi;

    filter4_s16x8_sum<coeff4>(s, f, c, sum_lo, sum_hi);

    // We divided filter values by 4 so subtract 2 from right shift in case of filter
    // coefficient 4.
    const int shift_offset = coeff4 ? shift - 2 : shift;

    d = vcombine_s16(vshrn_n_s32(sum_lo, shift_offset),
                     vshrn_n_s32(sum_hi, shift_offset));
}

template<bool coeff4, int shift>
void inline filter4_sp_s16x8(const int16x8_t *s, const int16x4_t f,
                             const int32x4_t c, const uint8x16_t shr_tbl, uint8x8_t &d)
{
    int32x4_t sum_lo, sum_hi;

    filter4_s16x8_sum<coeff4>(s, f, c, sum_lo, sum_hi);

    int16x8_t sum = vtbl2q_s32_s16(sum_lo, sum_hi, shr_tbl);

    // We divided filter values by 4 so subtract 2 from right shift in case of filter
    // coefficient 4.
    const int shift_offset = coeff4 ? shift - 2 : shift;

    d = vqshrun_n_s16(sum, shift_offset);
}

template<bool coeff4, int width, int height>
void interp4_vert_ss_neon(const int16_t *src, intptr_t srcStride, int16_t *dst,
                          intptr_t dstStride, int coeffIdx)
{
    const int N_TAPS = 4;
    const int shift = IF_FILTER_PREC;
    const int16x4_t filter = vld1_s16(X265_NS::g_chromaFilter[coeffIdx]);
    // Zero constant in order to use filter helper functions (optimised away).
    const int32x4_t c = vdupq_n_s32(0);

    src -= (N_TAPS / 2 - 1) * srcStride;

    if (width % 8 != 0)
    {
        if (width == 12 || width == 6)
        {
            const int n_store = width == 12 ? 8 : 6;
            const int16_t *s = src;
            int16_t *d = dst;

            int16x8_t in[7];
            load_s16x8xn<3>(s, srcStride, in);
            s += 3 * srcStride;

            for (int row = 0; row + 4 <= height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 3);

                int16x8_t res[4];
                filter4_ss_s16x8<coeff4, shift>(in + 0, filter, c, res[0]);
                filter4_ss_s16x8<coeff4, shift>(in + 1, filter, c, res[1]);
                filter4_ss_s16x8<coeff4, shift>(in + 2, filter, c, res[2]);
                filter4_ss_s16x8<coeff4, shift>(in + 3, filter, c, res[3]);

                store_s16xnxm<n_store, 4>(res, d, dstStride);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (width == 6)
            {
                return;
            }

            src += 8;
            dst += 8;
        }

        int16x4_t in[7];
        load_s16x4xn<3>(src, srcStride, in);
        src += 3 * srcStride;

        const int n_store = width > 4 ? 4 : width;
        for (int row = 0; row + 4 <= height; row += 4)
        {
            load_s16x4xn<4>(src, srcStride, in + 3);

            int16x4_t res[4];
            filter4_ss_s16x4<coeff4, shift>(in + 0, filter, c, res[0]);
            filter4_ss_s16x4<coeff4, shift>(in + 1, filter, c, res[1]);
            filter4_ss_s16x4<coeff4, shift>(in + 2, filter, c, res[2]);
            filter4_ss_s16x4<coeff4, shift>(in + 3, filter, c, res[3]);

            store_s16xnxm<n_store, 4>(res, dst, dstStride);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];

            src += 4 * srcStride;
            dst += 4 * dstStride;
        }

        if (height & 2)
        {
            load_s16x4xn<2>(src, srcStride, in + 3);

            int16x4_t res[2];
            filter4_ss_s16x4<coeff4, shift>(in + 0, filter, c, res[0]);
            filter4_ss_s16x4<coeff4, shift>(in + 1, filter, c, res[1]);

            store_s16xnxm<n_store, 2>(res, dst, dstStride);
        }
    }
    else
    {
        for (int col = 0; col < width; col += 8)
        {
            const int16_t *s = src;
            int16_t *d = dst;

            int16x8_t in[7];
            load_s16x8xn<3>(s, srcStride, in);
            s += 3 * srcStride;

            for (int row = 0; row + 4 <= height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 3);

                int16x8_t res[4];
                filter4_ss_s16x8<coeff4, shift>(in + 0, filter, c, res[0]);
                filter4_ss_s16x8<coeff4, shift>(in + 1, filter, c, res[1]);
                filter4_ss_s16x8<coeff4, shift>(in + 2, filter, c, res[2]);
                filter4_ss_s16x8<coeff4, shift>(in + 3, filter, c, res[3]);

                store_s16x8xn<4>(d, dstStride, res);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (height & 2)
            {
                load_s16x8xn<2>(s, srcStride, in + 3);

                int16x8_t res[2];
                filter4_ss_s16x8<coeff4, shift>(in + 0, filter, c, res[0]);
                filter4_ss_s16x8<coeff4, shift>(in + 1, filter, c, res[1]);

                store_s16x8xn<2>(d, dstStride, res);
            }

            src += 8;
            dst += 8;
        }
    }
}

template<int coeffIdx, int width, int height>
void interp8_vert_ss_neon(const int16_t *src, intptr_t srcStride, int16_t *dst,
                          intptr_t dstStride)
{
    const int N_TAPS = 8;
    const int16x8_t filter = vld1q_s16(X265_NS::g_lumaFilter[coeffIdx]);

    src -= (N_TAPS / 2 - 1) * srcStride;

    // Zero constant in order to use filter helper functions (optimised away).
    const int32x4_t c = vdupq_n_s32(0);

    if (width % 8 != 0)
    {
        const int16_t *s = src;
        int16_t *d = dst;
        if (width == 12)
        {
            int16x8_t in[11];
            load_s16x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 7);

                int32x4_t sum_lo[4];
                int32x4_t sum_hi[4];
                filter8_s16x8<coeffIdx>(in + 0, filter, c, sum_lo[0], sum_hi[0]);
                filter8_s16x8<coeffIdx>(in + 1, filter, c, sum_lo[1], sum_hi[1]);
                filter8_s16x8<coeffIdx>(in + 2, filter, c, sum_lo[2], sum_hi[2]);
                filter8_s16x8<coeffIdx>(in + 3, filter, c, sum_lo[3], sum_hi[3]);

                int16x8_t sum[4];
                sum[0] = vcombine_s16(vshrn_n_s32(sum_lo[0], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[0], IF_FILTER_PREC));
                sum[1] = vcombine_s16(vshrn_n_s32(sum_lo[1], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[1], IF_FILTER_PREC));
                sum[2] = vcombine_s16(vshrn_n_s32(sum_lo[2], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[2], IF_FILTER_PREC));
                sum[3] = vcombine_s16(vshrn_n_s32(sum_lo[3], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[3], IF_FILTER_PREC));

                store_s16x8xn<4>(d, dstStride, sum);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            s = src + 8;
            d = dst + 8;
        }

        int16x4_t in[11];
        load_s16x4xn<7>(s, srcStride, in);
        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            load_s16x4xn<4>(s, srcStride, in + 7);

            int32x4_t sum[4];
            filter8_s16x4<coeffIdx>(in + 0, filter, c, sum[0]);
            filter8_s16x4<coeffIdx>(in + 1, filter, c, sum[1]);
            filter8_s16x4<coeffIdx>(in + 2, filter, c, sum[2]);
            filter8_s16x4<coeffIdx>(in + 3, filter, c, sum[3]);

            int16x4_t sum_s16[4];
            sum_s16[0] = vshrn_n_s32(sum[0], IF_FILTER_PREC);
            sum_s16[1] = vshrn_n_s32(sum[1], IF_FILTER_PREC);
            sum_s16[2] = vshrn_n_s32(sum[2], IF_FILTER_PREC);
            sum_s16[3] = vshrn_n_s32(sum[3], IF_FILTER_PREC);

            store_s16x4xn<4>(d, dstStride, sum_s16);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];
            in[3] = in[7];
            in[4] = in[8];
            in[5] = in[9];
            in[6] = in[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }
    }
    else
    {
        for (int col = 0; col < width; col += 8)
        {
            const int16_t *s = src;
            int16_t *d = dst;

            int16x8_t in[11];
            load_s16x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 7);

                int32x4_t sum_lo[4];
                int32x4_t sum_hi[4];
                filter8_s16x8<coeffIdx>(in + 0, filter, c, sum_lo[0], sum_hi[0]);
                filter8_s16x8<coeffIdx>(in + 1, filter, c, sum_lo[1], sum_hi[1]);
                filter8_s16x8<coeffIdx>(in + 2, filter, c, sum_lo[2], sum_hi[2]);
                filter8_s16x8<coeffIdx>(in + 3, filter, c, sum_lo[3], sum_hi[3]);

                int16x8_t sum[4];
                sum[0] = vcombine_s16(vshrn_n_s32(sum_lo[0], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[0], IF_FILTER_PREC));
                sum[1] = vcombine_s16(vshrn_n_s32(sum_lo[1], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[1], IF_FILTER_PREC));
                sum[2] = vcombine_s16(vshrn_n_s32(sum_lo[2], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[2], IF_FILTER_PREC));
                sum[3] = vcombine_s16(vshrn_n_s32(sum_lo[3], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[3], IF_FILTER_PREC));

                store_s16x8xn<4>(d, dstStride, sum);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            src += 8;
            dst += 8;
        }
    }
}

#if !HIGH_BIT_DEPTH
// Element-wise ABS of g_chromaFilter
const uint8_t g_chromaFilterAbs8[8][NTAPS_CHROMA] =
{
    { 0, 64,  0, 0 },
    { 2, 58, 10, 2 },
    { 4, 54, 16, 2 },
    { 6, 46, 28, 4 },
    { 4, 36, 36, 4 },
    { 4, 28, 46, 6 },
    { 2, 16, 54, 4 },
    { 2, 10, 58, 2 }
};

template<int coeffIdx>
void inline filter8_u8x8(const uint8x8_t *s, const uint16x8_t c, int16x8_t &d)
{
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 },
        uint16x8_t t = vaddq_u16(c, vsubl_u8(s[6], s[0]));
        t = vmlal_u8(t, s[1], vdup_n_u8(4));
        t = vmlsl_u8(t, s[2], vdup_n_u8(10));
        t = vmlal_u8(t, s[3], vdup_n_u8(58));
        t = vmlal_u8(t, s[4], vdup_n_u8(17));
        t = vmlsl_u8(t, s[5], vdup_n_u8(5));
        d = vreinterpretq_s16_u16(t);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        int16x8_t t0 = vreinterpretq_s16_u16(vaddl_u8(s[3], s[4]));
        int16x8_t t1 = vreinterpretq_s16_u16(vaddl_u8(s[2], s[5]));
        int16x8_t t2 = vreinterpretq_s16_u16(vaddl_u8(s[1], s[6]));
        int16x8_t t3 = vreinterpretq_s16_u16(vaddl_u8(s[0], s[7]));

        d = vreinterpretq_s16_u16(c);
        d = vmlaq_n_s16(d, t0, 40);
        d = vmlaq_n_s16(d, t1, -11);
        d = vmlaq_n_s16(d, t2, 4);
        d = vmlaq_n_s16(d, t3, -1);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        uint16x8_t t = vaddq_u16(c, vsubl_u8(s[1], s[7]));
        t = vmlal_u8(t, s[6], vdup_n_u8(4));
        t = vmlsl_u8(t, s[5], vdup_n_u8(10));
        t = vmlal_u8(t, s[4], vdup_n_u8(58));
        t = vmlal_u8(t, s[3], vdup_n_u8(17));
        t = vmlsl_u8(t, s[2], vdup_n_u8(5));
        d = vreinterpretq_s16_u16(t);
    }
}

template<int coeffIdx>
void inline filter8_u8x16(const uint8x16_t *s, const uint16x8_t c,
                          int16x8_t &d0, int16x8_t &d1)
{
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        const uint8x16_t f0 = vdupq_n_u8(4);
        const uint8x16_t f1 = vdupq_n_u8(10);
        const uint8x16_t f2 = vdupq_n_u8(58);
        const uint8x16_t f3 = vdupq_n_u8(17);
        const uint8x16_t f4 = vdupq_n_u8(5);

        uint16x8_t t0 = vsubl_u8(vget_low_u8(s[6]), vget_low_u8(s[0]));
        t0 = vaddq_u16(c, t0);
        t0 = vmlal_u8(t0, vget_low_u8(s[1]), vget_low_u8(f0));
        t0 = vmlsl_u8(t0, vget_low_u8(s[2]), vget_low_u8(f1));
        t0 = vmlal_u8(t0, vget_low_u8(s[3]), vget_low_u8(f2));
        t0 = vmlal_u8(t0, vget_low_u8(s[4]), vget_low_u8(f3));
        t0 = vmlsl_u8(t0, vget_low_u8(s[5]), vget_low_u8(f4));
        d0 = vreinterpretq_s16_u16(t0);

        uint16x8_t t1 = vsubl_u8(vget_high_u8(s[6]), vget_high_u8(s[0]));
        t1 = vaddq_u16(c, t1);
        t1 = vmlal_u8(t1, vget_high_u8(s[1]), vget_high_u8(f0));
        t1 = vmlsl_u8(t1, vget_high_u8(s[2]), vget_high_u8(f1));
        t1 = vmlal_u8(t1, vget_high_u8(s[3]), vget_high_u8(f2));
        t1 = vmlal_u8(t1, vget_high_u8(s[4]), vget_high_u8(f3));
        t1 = vmlsl_u8(t1, vget_high_u8(s[5]), vget_high_u8(f4));
        d1 = vreinterpretq_s16_u16(t1);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        int16x8_t t0 = vreinterpretq_s16_u16(vaddl_u8(vget_low_u8(s[3]),
                                                      vget_low_u8(s[4])));
        int16x8_t t1 = vreinterpretq_s16_u16(vaddl_u8(vget_low_u8(s[2]),
                                                      vget_low_u8(s[5])));
        int16x8_t t2 = vreinterpretq_s16_u16(vaddl_u8(vget_low_u8(s[1]),
                                                      vget_low_u8(s[6])));
        int16x8_t t3 = vreinterpretq_s16_u16(vaddl_u8(vget_low_u8(s[0]),
                                                      vget_low_u8(s[7])));
        d0 = vreinterpretq_s16_u16(c);
        d0 = vmlaq_n_s16(d0, t0, 40);
        d0 = vmlaq_n_s16(d0, t1, -11);
        d0 = vmlaq_n_s16(d0, t2, 4);
        d0 = vmlaq_n_s16(d0, t3, -1);

        int16x8_t t4 = vreinterpretq_s16_u16(vaddl_u8(vget_high_u8(s[3]),
                                                      vget_high_u8(s[4])));
        int16x8_t t5 = vreinterpretq_s16_u16(vaddl_u8(vget_high_u8(s[2]),
                                                      vget_high_u8(s[5])));
        int16x8_t t6 = vreinterpretq_s16_u16(vaddl_u8(vget_high_u8(s[1]),
                                                      vget_high_u8(s[6])));
        int16x8_t t7 = vreinterpretq_s16_u16(vaddl_u8(vget_high_u8(s[0]),
                                                      vget_high_u8(s[7])));
        d1 = vreinterpretq_s16_u16(c);
        d1 = vmlaq_n_s16(d1, t4, 40);
        d1 = vmlaq_n_s16(d1, t5, -11);
        d1 = vmlaq_n_s16(d1, t6, 4);
        d1 = vmlaq_n_s16(d1, t7, -1);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        const uint8x16_t f0 = vdupq_n_u8(4);
        const uint8x16_t f1 = vdupq_n_u8(10);
        const uint8x16_t f2 = vdupq_n_u8(58);
        const uint8x16_t f3 = vdupq_n_u8(17);
        const uint8x16_t f4 = vdupq_n_u8(5);

        uint16x8_t t0 = vsubl_u8(vget_low_u8(s[1]), vget_low_u8(s[7]));
        t0 = vaddq_u16(c, t0);
        t0 = vmlal_u8(t0, vget_low_u8(s[6]), vget_low_u8(f0));
        t0 = vmlsl_u8(t0, vget_low_u8(s[5]), vget_low_u8(f1));
        t0 = vmlal_u8(t0, vget_low_u8(s[4]), vget_low_u8(f2));
        t0 = vmlal_u8(t0, vget_low_u8(s[3]), vget_low_u8(f3));
        t0 = vmlsl_u8(t0, vget_low_u8(s[2]), vget_low_u8(f4));
        d0 = vreinterpretq_s16_u16(t0);

        uint16x8_t t1 = vsubl_u8(vget_high_u8(s[1]), vget_high_u8(s[7]));
        t1 = vaddq_u16(c, t1);
        t1 = vmlal_u8(t1, vget_high_u8(s[6]), vget_high_u8(f0));
        t1 = vmlsl_u8(t1, vget_high_u8(s[5]), vget_high_u8(f1));
        t1 = vmlal_u8(t1, vget_high_u8(s[4]), vget_high_u8(f2));
        t1 = vmlal_u8(t1, vget_high_u8(s[3]), vget_high_u8(f3));
        t1 = vmlsl_u8(t1, vget_high_u8(s[2]), vget_high_u8(f4));
        d1 = vreinterpretq_s16_u16(t1);
    }
}

template<bool coeff4>
void inline filter4_u8x8(const uint8x8_t *s, const uint8x16x4_t f,
                         const uint16x8_t c, int16x8_t &d)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        uint16x8_t t0 = vaddl_u8(s[1], s[2]);
        uint16x8_t t1 = vaddl_u8(s[0], s[3]);
        d = vreinterpretq_s16_u16(vmlaq_n_u16(c, t0, 36));
        d = vmlsq_n_s16(d, vreinterpretq_s16_u16(t1), 4);
    }
    else
    {
        // All chroma filter taps have signs {-, +, +, -}, so we can use a
        // sequence of MLAL/MLSL with absolute filter values to avoid needing to
        // widen the input.
        uint16x8_t t = vmlal_u8(c, s[1], vget_low_u8(f.val[1]));
        t = vmlsl_u8(t, s[0], vget_low_u8(f.val[0]));
        t = vmlal_u8(t, s[2], vget_low_u8(f.val[2]));
        t = vmlsl_u8(t, s[3], vget_low_u8(f.val[3]));
        d = vreinterpretq_s16_u16(t);
    }
}

template<bool coeff4>
void inline filter4_u8x16(const uint8x16_t *s, const uint8x16x4_t f,
                          const uint16x8_t c, int16x8_t &d0, int16x8_t &d1)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        uint16x8_t t0 = vaddl_u8(vget_low_u8(s[1]), vget_low_u8(s[2]));
        uint16x8_t t1 = vaddl_u8(vget_low_u8(s[0]), vget_low_u8(s[3]));
        d0 = vreinterpretq_s16_u16(vmlaq_n_u16(c, t0, 36));
        d0 = vmlsq_n_s16(d0, vreinterpretq_s16_u16(t1), 4);

        uint16x8_t t2 = vaddl_u8(vget_high_u8(s[1]), vget_high_u8(s[2]));
        uint16x8_t t3 = vaddl_u8(vget_high_u8(s[0]), vget_high_u8(s[3]));
        d1 = vreinterpretq_s16_u16(vmlaq_n_u16(c, t2, 36));
        d1 = vmlsq_n_s16(d1, vreinterpretq_s16_u16(t3), 4);
    }
    else
    {
        // All chroma filter taps have signs {-, +, +, -}, so we can use a
        // sequence of MLAL/MLSL with absolute filter values to avoid needing to
        // widen the input.
        uint16x8_t t0 = vmlal_u8(c, vget_low_u8(s[1]), vget_low_u8(f.val[1]));
        t0 = vmlsl_u8(t0, vget_low_u8(s[0]), vget_low_u8(f.val[0]));
        t0 = vmlal_u8(t0, vget_low_u8(s[2]), vget_low_u8(f.val[2]));
        t0 = vmlsl_u8(t0, vget_low_u8(s[3]), vget_low_u8(f.val[3]));
        d0 = vreinterpretq_s16_u16(t0);

        uint16x8_t t1 = vmlal_u8(c, vget_high_u8(s[1]), vget_low_u8(f.val[1]));
        t1 = vmlsl_u8(t1, vget_high_u8(s[0]), vget_low_u8(f.val[0]));
        t1 = vmlal_u8(t1, vget_high_u8(s[2]), vget_low_u8(f.val[2]));
        t1 = vmlsl_u8(t1, vget_high_u8(s[3]), vget_low_u8(f.val[3]));
        d1 = vreinterpretq_s16_u16(t1);
    }
}

template<bool coeff4, int width, int height>
void interp4_horiz_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst,
                           intptr_t dstStride, int coeffIdx)
{
    const int N_TAPS = 4;
    src -= N_TAPS / 2 - 1;

    // Abs 8-bit filter taps to allow use of 8-bit MLAL/MLSL
    const uint8x16x4_t filter = vld4q_dup_u8(g_chromaFilterAbs8[coeffIdx]);

    // Zero constant in order to use filter helper functions (optimised away).
    const uint16x8_t c = vdupq_n_u16(0);

    if (width % 16 == 0)
    {
        for (int row = 0; row < height; row++)
        {
            int col = 0;
            for (; col + 32 <= width; col += 32)
            {
                uint8x16_t s0[N_TAPS], s1[N_TAPS];
                load_u8x16xn<4>(src + col + 0, 1, s0);
                load_u8x16xn<4>(src + col + 16, 1, s1);

                int16x8_t d0, d1, d2, d3;
                filter4_u8x16<coeff4>(s0, filter, c, d0, d1);
                filter4_u8x16<coeff4>(s1, filter, c, d2, d3);

                uint8x8_t d0_u8 = vqrshrun_n_s16(d0, IF_FILTER_PREC);
                uint8x8_t d1_u8 = vqrshrun_n_s16(d1, IF_FILTER_PREC);
                uint8x8_t d2_u8 = vqrshrun_n_s16(d2, IF_FILTER_PREC);
                uint8x8_t d3_u8 = vqrshrun_n_s16(d3, IF_FILTER_PREC);

                vst1q_u8(dst + col + 0, vcombine_u8(d0_u8, d1_u8));
                vst1q_u8(dst + col + 16, vcombine_u8(d2_u8, d3_u8));
            }

            for (; col + 16 <= width; col += 16)
            {
                uint8x16_t s[N_TAPS];
                load_u8x16xn<4>(src + col, 1, s);

                int16x8_t d0, d1;
                filter4_u8x16<coeff4>(s, filter, c, d0, d1);

                uint8x8_t d0_u8 = vqrshrun_n_s16(d0, IF_FILTER_PREC);
                uint8x8_t d1_u8 = vqrshrun_n_s16(d1, IF_FILTER_PREC);

                vst1q_u8(dst + col, vcombine_u8(d0_u8, d1_u8));
            }

            src += srcStride;
            dst += dstStride;
        }
    }
    else
    {
        for (int row = 0; row < height; row += 2)
        {
            int col = 0;
            for (; col + 8 <= width; col += 8)
            {
                uint8x8_t s0[N_TAPS], s1[N_TAPS];
                load_u8x8xn<4>(src + col + 0 * srcStride, 1, s0);
                load_u8x8xn<4>(src + col + 1 * srcStride, 1, s1);

                int16x8_t d0, d1;
                filter4_u8x8<coeff4>(s0, filter, c, d0);
                filter4_u8x8<coeff4>(s1, filter, c, d1);

                uint8x8_t d0_u8 = vqrshrun_n_s16(d0, IF_FILTER_PREC);
                uint8x8_t d1_u8 = vqrshrun_n_s16(d1, IF_FILTER_PREC);

                vst1_u8(dst + col + 0 * dstStride, d0_u8);
                vst1_u8(dst + col + 1 * dstStride, d1_u8);
            }

            if (width % 8 != 0)
            {
                uint8x8_t s0[N_TAPS], s1[N_TAPS];
                load_u8x8xn<4>(src + col + 0 * srcStride, 1, s0);
                load_u8x8xn<4>(src + col + 1 * srcStride, 1, s1);

                int16x8_t d0, d1;
                filter4_u8x8<coeff4>(s0, filter, c, d0);
                filter4_u8x8<coeff4>(s1, filter, c, d1);

                uint8x8_t d[2];
                d[0] = vqrshrun_n_s16(d0, IF_FILTER_PREC);
                d[1] = vqrshrun_n_s16(d1, IF_FILTER_PREC);

                if (width == 12 || width == 4)
                {
                    store_u8x4xn<2>(dst + col, dstStride, d);
                }
                if (width == 6)
                {
                    store_u8x6xn<2>(dst + col, dstStride, d);
                }
                if (width == 2)
                {
                    store_u8x2xn<2>(dst + col, dstStride, d);
                }
            }

            src += 2 * srcStride;
            dst += 2 * dstStride;
        }
    }
}

template<int coeffIdx, int width, int height>
void interp8_horiz_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst,
                           intptr_t dstStride)
{
    const int N_TAPS = 8;
    src -= N_TAPS / 2 - 1;

    // Zero constant in order to use filter helper functions (optimised away).
    const uint16x8_t c = vdupq_n_u16(0);

    if (width % 16 == 0)
    {
        for (int row = 0; row < height; row++)
        {
            int col = 0;
            for (; col + 32 <= width; col += 32)
            {
                uint8x16_t s0[N_TAPS], s1[N_TAPS];
                load_u8x16xn<8>(src + col + 0, 1, s0);
                load_u8x16xn<8>(src + col + 16, 1, s1);

                int16x8_t d0, d1, d2, d3;
                filter8_u8x16<coeffIdx>(s0, c, d0, d1);
                filter8_u8x16<coeffIdx>(s1, c, d2, d3);

                vst1_u8(dst + col + 0, vqrshrun_n_s16(d0, IF_FILTER_PREC));
                vst1_u8(dst + col + 8, vqrshrun_n_s16(d1, IF_FILTER_PREC));
                vst1_u8(dst + col + 16, vqrshrun_n_s16(d2, IF_FILTER_PREC));
                vst1_u8(dst + col + 24, vqrshrun_n_s16(d3, IF_FILTER_PREC));
            }

            for (; col + 16 <= width; col += 16)
            {
                uint8x16_t s[N_TAPS];
                load_u8x16xn<8>(src + col, 1, s);

                int16x8_t d0, d1;
                filter8_u8x16<coeffIdx>(s, c, d0, d1);

                uint8x8_t d0_u8 = vqrshrun_n_s16(d0, IF_FILTER_PREC);
                uint8x8_t d1_u8 = vqrshrun_n_s16(d1, IF_FILTER_PREC);

                vst1q_u8(dst + col, vcombine_u8(d0_u8, d1_u8));
            }

            for (; col + 8 <= width; col += 8)
            {
                uint8x8_t s[N_TAPS];
                load_u8x8xn<8>(src + col, 1, s);

                int16x8_t d;
                filter8_u8x8<coeffIdx>(s, c, d);

                vst1_u8(dst + col, vqrshrun_n_s16(d, IF_FILTER_PREC));
            }

            if (width % 8 != 0)
            {
                uint8x8_t s[N_TAPS];
                load_u8x8xn<8>(src + col, 1, s);

                int16x8_t d;
                filter8_u8x8<coeffIdx>(s, c, d);

                store_u8x4x1(dst + col, vqrshrun_n_s16(d, IF_FILTER_PREC));
            }

            src += srcStride;
            dst += dstStride;
        }
    }
    else
    {
        for (int row = 0; row < height; row += 2)
        {
            int col = 0;
            for (; col + 8 <= width; col += 8)
            {
                uint8x8_t s0[N_TAPS], s1[N_TAPS];
                load_u8x8xn<8>(src + col + 0 * srcStride, 1, s0);
                load_u8x8xn<8>(src + col + 1 * srcStride, 1, s1);

                int16x8_t d0, d1;
                filter8_u8x8<coeffIdx>(s0, c, d0);
                filter8_u8x8<coeffIdx>(s1, c, d1);

                uint8x8_t d0_u8 = vqrshrun_n_s16(d0, IF_FILTER_PREC);
                uint8x8_t d1_u8 = vqrshrun_n_s16(d1, IF_FILTER_PREC);

                vst1_u8(dst + col + 0 * dstStride, d0_u8);
                vst1_u8(dst + col + 1 * dstStride, d1_u8);
            }

            if (width % 8 != 0)
            {
                uint8x8_t s0[N_TAPS], s1[N_TAPS];
                load_u8x8xn<8>(src + col + 0 * srcStride, 1, s0);
                load_u8x8xn<8>(src + col + 1 * srcStride, 1, s1);

                int16x8_t d0, d1;
                filter8_u8x8<coeffIdx>(s0, c, d0);
                filter8_u8x8<coeffIdx>(s1, c, d1);

                uint8x8_t d[2];
                d[0] = vqrshrun_n_s16(d0, IF_FILTER_PREC);
                d[1] = vqrshrun_n_s16(d1, IF_FILTER_PREC);

                store_u8x4xn<2>(dst + col, dstStride, d);
            }

            src += 2 * srcStride;
            dst += 2 * dstStride;
        }
    }
}

template<bool coeff4, int width, int height>
void interp4_horiz_ps_neon(const uint8_t *src, intptr_t srcStride, int16_t *dst,
                           intptr_t dstStride, int coeffIdx,
                           int isRowExt)
{
    const int offset = (unsigned)-IF_INTERNAL_OFFS;

    int blkheight = height;
    const int N_TAPS = 4;
    src -= N_TAPS / 2 - 1;

    if (isRowExt)
    {
        src -= (N_TAPS / 2 - 1) * srcStride;
        blkheight += N_TAPS - 1;
    }

    const uint16x8_t c = vdupq_n_u16(offset);

    // Abs 8-bit filter taps to allow use of 8-bit MLAL/MLSL
    const uint8x16x4_t filter = vld4q_dup_u8(g_chromaFilterAbs8[coeffIdx]);

    for (int row = 0; row + 2 <= blkheight; row += 2)
    {
        int col = 0;
        for (; col + 8 <= width; col += 8)
        {
            uint8x8_t s0[N_TAPS], s1[N_TAPS];
            load_u8x8xn<4>(src + col + 0 * srcStride, 1, s0);
            load_u8x8xn<4>(src + col + 1 * srcStride, 1, s1);

            int16x8_t d0, d1;
            filter4_u8x8<coeff4>(s0, filter, c, d0);
            filter4_u8x8<coeff4>(s1, filter, c, d1);

            vst1q_s16(dst + col + 0 * dstStride, d0);
            vst1q_s16(dst + col + 1 * dstStride, d1);
        }

        if (width % 8 != 0)
        {
            uint8x8_t s0[N_TAPS], s1[N_TAPS];
            load_u8x8xn<4>(src + col + 0 * srcStride, 1, s0);
            load_u8x8xn<4>(src + col + 1 * srcStride, 1, s1);

            int16x8_t d[2];
            filter4_u8x8<coeff4>(s0, filter, c, d[0]);
            filter4_u8x8<coeff4>(s1, filter, c, d[1]);

            if (width == 12 || width == 4)
            {
                store_s16x4xn<2>(dst + col, dstStride, d);
            }
            if (width == 6)
            {
                store_s16x6xn<2>(dst + col, dstStride, d);
            }
            if (width == 2)
            {
                store_s16x2xn<2>(dst + col, dstStride, d);
            }
        }

        src += 2 * srcStride;
        dst += 2 * dstStride;
    }

    if (isRowExt)
    {
        int col = 0;
        for (; col + 8 <= width; col += 8)
        {
            uint8x8_t s[N_TAPS];
            load_u8x8xn<4>(src + col, 1, s);

            int16x8_t d;
            filter4_u8x8<coeff4>(s, filter, c, d);

            vst1q_s16(dst + col, d);
        }

        if (width % 8 != 0)
        {
            uint8x8_t s[N_TAPS];
            load_u8x8xn<4>(src + col, 1, s);

            int16x8_t d;
            filter4_u8x8<coeff4>(s, filter, c, d);

            if (width == 12 || width == 4)
            {
                store_s16x4xn<1>(dst + col, dstStride, &d);
            }
            if (width == 6)
            {
                store_s16x6xn<1>(dst + col, dstStride, &d);
            }
            if (width == 2)
            {
                store_s16x2xn<1>(dst + col, dstStride, &d);
            }
        }
    }
}

template<int coeffIdx, int width, int height>
void interp8_horiz_ps_neon(const uint8_t *src, intptr_t srcStride, int16_t *dst,
                           intptr_t dstStride, int isRowExt)
{
    const int offset = (unsigned)-IF_INTERNAL_OFFS;

    int blkheight = height;
    const int N_TAPS = 8;
    src -= N_TAPS / 2 - 1;

    if (isRowExt)
    {
        src -= (N_TAPS / 2 - 1) * srcStride;
        blkheight += N_TAPS - 1;
    }

    const uint16x8_t c = vdupq_n_u16(offset);

    for (int row = 0; row + 2 <= blkheight; row += 2)
    {
        int col = 0;
        for (; col + 16 <= width; col += 16)
        {
            uint8x16_t s0[N_TAPS], s1[N_TAPS];
            load_u8x16xn<8>(src + col + 0 * srcStride, 1, s0);
            load_u8x16xn<8>(src + col + 1 * srcStride, 1, s1);

            int16x8_t d0, d1, d2, d3;
            filter8_u8x16<coeffIdx>(s0, c, d0, d1);
            filter8_u8x16<coeffIdx>(s1, c, d2, d3);

            vst1q_s16(dst + col + 0 * dstStride + 0, d0);
            vst1q_s16(dst + col + 0 * dstStride + 8, d1);
            vst1q_s16(dst + col + 1 * dstStride + 0, d2);
            vst1q_s16(dst + col + 1 * dstStride + 8, d3);
        }

        for (; col + 8 <= width; col += 8)
        {
            uint8x8_t s0[N_TAPS], s1[N_TAPS];
            load_u8x8xn<8>(src + col + 0 * srcStride, 1, s0);
            load_u8x8xn<8>(src + col + 1 * srcStride, 1, s1);

            int16x8_t d0, d1;
            filter8_u8x8<coeffIdx>(s0, c, d0);
            filter8_u8x8<coeffIdx>(s1, c, d1);

            vst1q_s16(dst + col + 0 * dstStride, d0);
            vst1q_s16(dst + col + 1 * dstStride, d1);
        }

        if (width % 8 != 0)
        {
            uint8x8_t s0[N_TAPS], s1[N_TAPS];
            load_u8x8xn<8>(src + col + 0 * srcStride, 1, s0);
            load_u8x8xn<8>(src + col + 1 * srcStride, 1, s1);

            int16x8_t d0, d1;
            filter8_u8x8<coeffIdx>(s0, c, d0);
            filter8_u8x8<coeffIdx>(s1, c, d1);

            vst1_s16(dst + col + 0 * dstStride, vget_low_s16(d0));
            vst1_s16(dst + col + 1 * dstStride, vget_low_s16(d1));
        }

        src += 2 * srcStride;
        dst += 2 * dstStride;
    }

    if (isRowExt)
    {
        int col = 0;
        for (; col + 8 <= width; col += 8)
        {
            uint8x8_t s[N_TAPS];
            load_u8x8xn<8>(src + col, 1, s);

            int16x8_t d;
            filter8_u8x8<coeffIdx>(s, c, d);

            vst1q_s16(dst + col, d);
        }

        if (width % 8 != 0)
        {
            uint8x8_t s[N_TAPS];
            load_u8x8xn<8>(src + col, 1, s);

            int16x8_t d;
            filter8_u8x8<coeffIdx>(s, c, d);

            vst1_s16(dst + col, vget_low_s16(d));
        }
    }
}

template<bool coeff4, int width, int height>
void interp4_vert_pp_neon(const uint8_t *src, intptr_t srcStride, uint8_t *dst,
                          intptr_t dstStride, int coeffIdx)
{
    const int N_TAPS = 4;
    src -= (N_TAPS / 2 - 1) * srcStride;

    // Abs 8-bit filter taps to allow use of 8-bit MLAL/MLSL
    const uint8x16x4_t filter = vld4q_dup_u8(g_chromaFilterAbs8[coeffIdx]);

    // Zero constant in order to use filter helper functions (optimised away).
    const uint16x8_t c = vdupq_n_u16(0);

    if (width == 12)
    {
        const uint8_t *s = src;
        uint8_t *d = dst;

        uint8x8_t in[7];
        load_u8x8xn<3>(s, srcStride, in);
        s += 3 * srcStride;

        for (int row = 0; row + 4 <= height; row += 4)
        {
            load_u8x8xn<4>(s, srcStride, in + 3);

            int16x8_t sum[4];
            filter4_u8x8<coeff4>(in + 0, filter, c, sum[0]);
            filter4_u8x8<coeff4>(in + 1, filter, c, sum[1]);
            filter4_u8x8<coeff4>(in + 2, filter, c, sum[2]);
            filter4_u8x8<coeff4>(in + 3, filter, c, sum[3]);

            uint8x8_t sum_u8[4];
            sum_u8[0] = vqrshrun_n_s16(sum[0], IF_FILTER_PREC);
            sum_u8[1] = vqrshrun_n_s16(sum[1], IF_FILTER_PREC);
            sum_u8[2] = vqrshrun_n_s16(sum[2], IF_FILTER_PREC);
            sum_u8[3] = vqrshrun_n_s16(sum[3], IF_FILTER_PREC);

            store_u8x8xn<4>(d, dstStride, sum_u8);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        src += 8;
        dst += 8;
        s = src;
        d = dst;

        load_u8x8xn<3>(s, srcStride, in);
        s += 3 * srcStride;

        for (int row = 0; row + 4 <= height; row += 4)
        {
            load_u8x8xn<4>(s, srcStride, in + 3);

            int16x8_t sum[4];
            filter4_u8x8<coeff4>(in + 0, filter, c, sum[0]);
            filter4_u8x8<coeff4>(in + 1, filter, c, sum[1]);
            filter4_u8x8<coeff4>(in + 2, filter, c, sum[2]);
            filter4_u8x8<coeff4>(in + 3, filter, c, sum[3]);

            uint8x8_t sum_u8[4];
            sum_u8[0] = vqrshrun_n_s16(sum[0], IF_FILTER_PREC);
            sum_u8[1] = vqrshrun_n_s16(sum[1], IF_FILTER_PREC);
            sum_u8[2] = vqrshrun_n_s16(sum[2], IF_FILTER_PREC);
            sum_u8[3] = vqrshrun_n_s16(sum[3], IF_FILTER_PREC);

            store_u8x4xn<4>(d, dstStride, sum_u8);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }
    }
    else
    {
        const int n_store = (width < 8) ? width : 8;
        for (int col = 0; col < width; col += 8)
        {
            const uint8_t *s = src;
            uint8_t *d = dst;

            uint8x8_t in[7];
            load_u8x8xn<3>(s, srcStride, in);
            s += 3 * srcStride;

            for (int row = 0; row + 4 <= height; row += 4)
            {
                load_u8x8xn<4>(s, srcStride, in + 3);

                int16x8_t sum[4];
                filter4_u8x8<coeff4>(in + 0, filter, c, sum[0]);
                filter4_u8x8<coeff4>(in + 1, filter, c, sum[1]);
                filter4_u8x8<coeff4>(in + 2, filter, c, sum[2]);
                filter4_u8x8<coeff4>(in + 3, filter, c, sum[3]);

                uint8x8_t sum_u8[4];
                sum_u8[0] = vqrshrun_n_s16(sum[0], IF_FILTER_PREC);
                sum_u8[1] = vqrshrun_n_s16(sum[1], IF_FILTER_PREC);
                sum_u8[2] = vqrshrun_n_s16(sum[2], IF_FILTER_PREC);
                sum_u8[3] = vqrshrun_n_s16(sum[3], IF_FILTER_PREC);

                store_u8xnxm<n_store, 4>(d, dstStride, sum_u8);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (height & 2)
            {
                load_u8x8xn<2>(s, srcStride, in + 3);

                int16x8_t sum[2];
                filter4_u8x8<coeff4>(in + 0, filter, c, sum[0]);
                filter4_u8x8<coeff4>(in + 1, filter, c, sum[1]);

                uint8x8_t sum_u8[2];
                sum_u8[0] = vqrshrun_n_s16(sum[0], IF_FILTER_PREC);
                sum_u8[1] = vqrshrun_n_s16(sum[1], IF_FILTER_PREC);

                store_u8xnxm<n_store, 2>(d, dstStride, sum_u8);
            }

            src += 8;
            dst += 8;
        }
    }
}

template<int coeffIdx, int width, int height>
void interp8_vert_pp_neon(const uint8_t *src, intptr_t srcStride, uint8_t *dst,
                          intptr_t dstStride)
{
    const int N_TAPS = 8;
    src -= (N_TAPS / 2 - 1) * srcStride;

    // Zero constant in order to use filter helper functions (optimised away).
    const uint16x8_t c = vdupq_n_u16(0);

    if (width % 8 != 0)
    {
        uint8x8_t in[11];
        const uint8_t *s = src;
        uint8_t *d = dst;

        if (width == 12)
        {
            load_u8x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u8x8xn<4>(s, srcStride, in + 7);

                int16x8_t sum[4];
                filter8_u8x8<coeffIdx>(in + 0, c, sum[0]);
                filter8_u8x8<coeffIdx>(in + 1, c, sum[1]);
                filter8_u8x8<coeffIdx>(in + 2, c, sum[2]);
                filter8_u8x8<coeffIdx>(in + 3, c, sum[3]);

                uint8x8_t sum_u8[4];
                sum_u8[0] = vqrshrun_n_s16(sum[0], IF_FILTER_PREC);
                sum_u8[1] = vqrshrun_n_s16(sum[1], IF_FILTER_PREC);
                sum_u8[2] = vqrshrun_n_s16(sum[2], IF_FILTER_PREC);
                sum_u8[3] = vqrshrun_n_s16(sum[3], IF_FILTER_PREC);

                store_u8x8xn<4>(d, dstStride, sum_u8);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            s = src + 8;
            d = dst + 8;
        }

        load_u8x8xn<7>(s, srcStride, in);
        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            load_u8x8xn<4>(s, srcStride, in + 7);

            int16x8_t sum[4];
            filter8_u8x8<coeffIdx>(in + 0, c, sum[0]);
            filter8_u8x8<coeffIdx>(in + 1, c, sum[1]);
            filter8_u8x8<coeffIdx>(in + 2, c, sum[2]);
            filter8_u8x8<coeffIdx>(in + 3, c, sum[3]);

            uint8x8_t sum_u8[4];
            sum_u8[0] = vqrshrun_n_s16(sum[0], IF_FILTER_PREC);
            sum_u8[1] = vqrshrun_n_s16(sum[1], IF_FILTER_PREC);
            sum_u8[2] = vqrshrun_n_s16(sum[2], IF_FILTER_PREC);
            sum_u8[3] = vqrshrun_n_s16(sum[3], IF_FILTER_PREC);

            store_u8x4xn<4>(d, dstStride, sum_u8);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];
            in[3] = in[7];
            in[4] = in[8];
            in[5] = in[9];
            in[6] = in[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }
    }
    else if (width % 16 != 0)
    {
        for (int col = 0; col < width; col += 8)
        {
            const uint8_t *s = src;
            uint8_t *d = dst;

            uint8x8_t in[11];
            load_u8x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u8x8xn<4>(s, srcStride, in + 7);

                int16x8_t sum[4];
                filter8_u8x8<coeffIdx>(in + 0, c, sum[0]);
                filter8_u8x8<coeffIdx>(in + 1, c, sum[1]);
                filter8_u8x8<coeffIdx>(in + 2, c, sum[2]);
                filter8_u8x8<coeffIdx>(in + 3, c, sum[3]);

                uint8x8_t sum_u8[4];
                sum_u8[0] = vqrshrun_n_s16(sum[0], IF_FILTER_PREC);
                sum_u8[1] = vqrshrun_n_s16(sum[1], IF_FILTER_PREC);
                sum_u8[2] = vqrshrun_n_s16(sum[2], IF_FILTER_PREC);
                sum_u8[3] = vqrshrun_n_s16(sum[3], IF_FILTER_PREC);

                store_u8x8xn<4>(d, dstStride, sum_u8);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            src += 8;
            dst += 8;
        }
    }
    else
    {
        for (int col = 0; col < width; col += 16)
        {
            const uint8_t *s = src;
            uint8_t *d = dst;

            uint8x16_t in[11];
            load_u8x16xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u8x16xn<4>(s, srcStride, in + 7);

                int16x8_t sum_lo[4];
                int16x8_t sum_hi[4];
                filter8_u8x16<coeffIdx>(in + 0, c, sum_lo[0], sum_hi[0]);
                filter8_u8x16<coeffIdx>(in + 1, c, sum_lo[1], sum_hi[1]);
                filter8_u8x16<coeffIdx>(in + 2, c, sum_lo[2], sum_hi[2]);
                filter8_u8x16<coeffIdx>(in + 3, c, sum_lo[3], sum_hi[3]);

                uint8x16_t sum[4];
                sum[0] = vcombine_u8(vqrshrun_n_s16(sum_lo[0], IF_FILTER_PREC),
                                     vqrshrun_n_s16(sum_hi[0], IF_FILTER_PREC));
                sum[1] = vcombine_u8(vqrshrun_n_s16(sum_lo[1], IF_FILTER_PREC),
                                     vqrshrun_n_s16(sum_hi[1], IF_FILTER_PREC));
                sum[2] = vcombine_u8(vqrshrun_n_s16(sum_lo[2], IF_FILTER_PREC),
                                     vqrshrun_n_s16(sum_hi[2], IF_FILTER_PREC));
                sum[3] = vcombine_u8(vqrshrun_n_s16(sum_lo[3], IF_FILTER_PREC),
                                     vqrshrun_n_s16(sum_hi[3], IF_FILTER_PREC));

                store_u8x16xn<4>(d, dstStride, sum);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            src += 16;
            dst += 16;
        }
    }
}

template<bool coeff4, int width, int height>
void interp4_vert_ps_neon(const uint8_t *src, intptr_t srcStride, int16_t *dst,
                          intptr_t dstStride, int coeffIdx)
{
    const int offset = (unsigned)-IF_INTERNAL_OFFS;

    const int N_TAPS = 4;
    src -= (N_TAPS / 2 - 1) * srcStride;

    // Abs 8-bit filter taps to allow use of 8-bit MLAL/MLSL
    const uint8x16x4_t filter = vld4q_dup_u8(g_chromaFilterAbs8[coeffIdx]);

    const uint16x8_t c = vdupq_n_u16(offset);

    if (width == 12)
    {
        const uint8_t *s = src;
        int16_t *d = dst;

        uint8x8_t in[7];
        load_u8x8xn<3>(s, srcStride, in);
        s += 3 * srcStride;

        for (int row = 0; (row + 4) <= height; row += 4)
        {
            load_u8x8xn<4>(s, srcStride, in + 3);

            int16x8_t sum[4];
            filter4_u8x8<coeff4>(in + 0, filter, c, sum[0]);
            filter4_u8x8<coeff4>(in + 1, filter, c, sum[1]);
            filter4_u8x8<coeff4>(in + 2, filter, c, sum[2]);
            filter4_u8x8<coeff4>(in + 3, filter, c, sum[3]);

            store_s16x8xn<4>(d, dstStride, sum);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        src += 8;
        dst += 8;
        s = src;
        d = dst;

        load_u8x8xn<3>(s, srcStride, in);
        s += 3 * srcStride;

        for (int row = 0; (row + 4) <= height; row += 4)
        {
            load_u8x8xn<4>(s, srcStride, in + 3);

            int16x8_t sum[4];
            filter4_u8x8<coeff4>(in + 0, filter, c, sum[0]);
            filter4_u8x8<coeff4>(in + 1, filter, c, sum[1]);
            filter4_u8x8<coeff4>(in + 2, filter, c, sum[2]);
            filter4_u8x8<coeff4>(in + 3, filter, c, sum[3]);

            store_s16x4xn<4>(d, dstStride, sum);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }
    }
    else
    {
        const int n_store = (width < 8) ? width : 8;
        for (int col = 0; col < width; col += 8)
        {
            const uint8_t *s = src;
            int16_t *d = dst;

            uint8x8_t in[7];
            load_u8x8xn<3>(s, srcStride, in);
            s += 3 * srcStride;

            for (int row = 0; (row + 4) <= height; row += 4)
            {
                load_u8x8xn<4>(s, srcStride, in + 3);

                int16x8_t sum[4];
                filter4_u8x8<coeff4>(in + 0, filter, c, sum[0]);
                filter4_u8x8<coeff4>(in + 1, filter, c, sum[1]);
                filter4_u8x8<coeff4>(in + 2, filter, c, sum[2]);
                filter4_u8x8<coeff4>(in + 3, filter, c, sum[3]);

                store_s16xnxm<n_store, 4>(sum, d, dstStride);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (height & 2)
            {
                load_u8x8xn<2>(s, srcStride, in + 3);

                int16x8_t sum[2];
                filter4_u8x8<coeff4>(in + 0, filter, c, sum[0]);
                filter4_u8x8<coeff4>(in + 1, filter, c, sum[1]);

                store_s16xnxm<n_store, 2>(sum, d, dstStride);
            }

            src += 8;
            dst += 8;
        }
    }
}

template<int coeffIdx, int width, int height>
void interp8_vert_ps_neon(const uint8_t *src, intptr_t srcStride, int16_t *dst,
                          intptr_t dstStride)
{
    const int offset = (unsigned)-IF_INTERNAL_OFFS;

    const int N_TAPS = 8;
    src -= (N_TAPS / 2 - 1) * srcStride;

    const uint16x8_t c = vdupq_n_u16(offset);

    if (width % 8 != 0)
    {
        uint8x8_t in[11];
        const uint8_t *s = src;
        int16_t *d = dst;

        if (width == 12)
        {
            load_u8x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u8x8xn<4>(s, srcStride, in + 7);

                int16x8_t sum[4];
                filter8_u8x8<coeffIdx>(in + 0, c, sum[0]);
                filter8_u8x8<coeffIdx>(in + 1, c, sum[1]);
                filter8_u8x8<coeffIdx>(in + 2, c, sum[2]);
                filter8_u8x8<coeffIdx>(in + 3, c, sum[3]);

                store_s16x8xn<4>(d, dstStride, sum);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            s = src + 8;
            d = dst + 8;
        }

        load_u8x8xn<7>(s, srcStride, in);
        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            load_u8x8xn<4>(s, srcStride, in + 7);

            int16x8_t sum[4];
            filter8_u8x8<coeffIdx>(in + 0, c, sum[0]);
            filter8_u8x8<coeffIdx>(in + 1, c, sum[1]);
            filter8_u8x8<coeffIdx>(in + 2, c, sum[2]);
            filter8_u8x8<coeffIdx>(in + 3, c, sum[3]);

            store_s16x4xn<4>(d, dstStride, sum);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];
            in[3] = in[7];
            in[4] = in[8];
            in[5] = in[9];
            in[6] = in[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }
    }
    else if (width % 16 != 0)
    {
        for (int col = 0; col < width; col += 8)
        {
            const uint8_t *s = src;
            int16_t *d = dst;

            uint8x8_t in[11];
            load_u8x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u8x8xn<4>(s, srcStride, in + 7);

                int16x8_t sum[4];
                filter8_u8x8<coeffIdx>(in + 0, c, sum[0]);
                filter8_u8x8<coeffIdx>(in + 1, c, sum[1]);
                filter8_u8x8<coeffIdx>(in + 2, c, sum[2]);
                filter8_u8x8<coeffIdx>(in + 3, c, sum[3]);

                store_s16x8xn<4>(d, dstStride, sum);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            src += 8;
            dst += 8;
        }
    }
    else
    {
        for (int col = 0; col < width; col += 16)
        {
            const uint8_t *s = src;
            int16_t *d = dst;

            uint8x16_t in[11];
            load_u8x16xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u8x16xn<4>(s, srcStride, in + 7);

                int16x8_t sum_lo[4];
                int16x8_t sum_hi[4];
                filter8_u8x16<coeffIdx>(in + 0, c, sum_lo[0], sum_hi[0]);
                filter8_u8x16<coeffIdx>(in + 1, c, sum_lo[1], sum_hi[1]);
                filter8_u8x16<coeffIdx>(in + 2, c, sum_lo[2], sum_hi[2]);
                filter8_u8x16<coeffIdx>(in + 3, c, sum_lo[3], sum_hi[3]);

                store_s16x8xn<4>(d + 0, dstStride, sum_lo);
                store_s16x8xn<4>(d + 8, dstStride, sum_hi);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            src += 16;
            dst += 16;
        }
    }
}

template<bool coeff4, int width, int height>
void interp4_vert_sp_neon(const int16_t *src, intptr_t srcStride, uint8_t *dst,
                          intptr_t dstStride, int coeffIdx)
{
    assert(X265_DEPTH == 8);
    const int N_TAPS = 4;
    const int headRoom = IF_INTERNAL_PREC - X265_DEPTH;
    const int shift = IF_FILTER_PREC + headRoom;
    // Subtract 8 from shift since we account for that in table lookups.
    const int shift_offset = shift - 8;

    const int16x4_t filter = vld1_s16(X265_NS::g_chromaFilter[coeffIdx]);
    const uint8x16_t shr_tbl = vld1q_u8(vert_shr_tbl);
    int32x4_t offset;

    if (coeff4)
    {
        // The right shift by 2 is needed because we will divide the filter values by 4.
        offset = vdupq_n_s32(((1 << (shift - 1)) +
                              (IF_INTERNAL_OFFS << IF_FILTER_PREC)) >> 2);
    }
    else
    {
        offset = vdupq_n_s32((1 << (shift - 1)) +
                             (IF_INTERNAL_OFFS << IF_FILTER_PREC));
    }

    src -= (N_TAPS / 2 - 1) * srcStride;

    if (width % 8 != 0)
    {
        if (width == 12 || width == 6)
        {
            const int n_store = width == 12 ? 8 : 6;
            const int16_t *s = src;
            uint8_t *d = dst;

            int16x8_t in[7];
            load_s16x8xn<3>(s, srcStride, in);
            s += 3 * srcStride;

            for (int row = 0; row + 4 <= height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 3);

                uint8x8_t sum[4];
                filter4_sp_s16x8<coeff4, shift_offset>(in + 0, filter, offset, shr_tbl,
                                                       sum[0]);
                filter4_sp_s16x8<coeff4, shift_offset>(in + 1, filter, offset, shr_tbl,
                                                       sum[1]);
                filter4_sp_s16x8<coeff4, shift_offset>(in + 2, filter, offset, shr_tbl,
                                                       sum[2]);
                filter4_sp_s16x8<coeff4, shift_offset>(in + 3, filter, offset, shr_tbl,
                                                       sum[3]);

                store_u8xnxm<n_store, 4>(d, dstStride, sum);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (width == 6)
            {
                return;
            }

            src += 8;
            dst += 8;
        }

        const int n_store = width > 4 ? 4 : width;

        int16x4_t in[7];
        load_s16x4xn<3>(src, srcStride, in);
        src += 3 * srcStride;

        for (int row = 0; row + 4 <= height; row += 4)
        {
            load_s16x4xn<4>(src, srcStride, in + 3);

            uint8x8_t res[2];
            filter4x2_sp_s16x4<coeff4, shift_offset>(in + 0, in + 1, filter, offset,
                                                     shr_tbl, res[0]);
            filter4x2_sp_s16x4<coeff4, shift_offset>(in + 2, in + 3, filter, offset,
                                                     shr_tbl, res[1]);

            store_u8xnxm_strided<n_store, 4>(dst, dstStride, res);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];

            src += 4 * srcStride;
            dst += 4 * dstStride;
        }

        if (height & 2)
        {
            load_s16x4xn<2>(src, srcStride, in + 3);

            uint8x8_t res;
            filter4x2_sp_s16x4<coeff4, shift_offset>(in + 0, in + 1, filter, offset,
                                                     shr_tbl, res);

            store_u8xnxm_strided<n_store, 2>(dst, dstStride, &res);
        }
    }
    else
    {
        for (int col = 0; col < width; col += 8)
        {
            const int16_t *s = src;
            uint8_t *d = dst;

            int16x8_t in[7];
            load_s16x8xn<3>(s, srcStride, in);
            s += 3 * srcStride;

            for (int row = 0; row + 4 <= height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 3);

                uint8x8_t sum[4];
                filter4_sp_s16x8<coeff4, shift_offset>(in + 0, filter, offset, shr_tbl,
                                                       sum[0]);
                filter4_sp_s16x8<coeff4, shift_offset>(in + 1, filter, offset, shr_tbl,
                                                       sum[1]);
                filter4_sp_s16x8<coeff4, shift_offset>(in + 2, filter, offset, shr_tbl,
                                                       sum[2]);
                filter4_sp_s16x8<coeff4, shift_offset>(in + 3, filter, offset, shr_tbl,
                                                       sum[3]);

                store_u8x8xn<4>(d, dstStride, sum);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (height & 2)
            {
                load_s16x8xn<2>(s, srcStride, in + 3);

                uint8x8_t sum[2];
                filter4_sp_s16x8<coeff4, shift_offset>(in + 0, filter, offset, shr_tbl,
                                                       sum[0]);
                filter4_sp_s16x8<coeff4, shift_offset>(in + 1, filter, offset, shr_tbl,
                                                       sum[1]);

                store_u8x8xn<2>(d, dstStride, sum);
            }

            src += 8;
            dst += 8;
        }
    }
}

template<int coeffIdx, int width, int height>
void interp8_vert_sp_neon(const int16_t *src, intptr_t srcStride, pixel *dst,
                          intptr_t dstStride)
{
    assert(X265_DEPTH == 8);
    const int headRoom = IF_INTERNAL_PREC - X265_DEPTH;
    const int shift = IF_FILTER_PREC + headRoom;
    // Subtract 8 from shift since we account for that in table lookups.
    const int shift_offset = shift - 8;
    const int offset = (1 << (shift - 1)) + (IF_INTERNAL_OFFS << IF_FILTER_PREC);

    const int N_TAPS = 8;
    src -= (N_TAPS / 2 - 1) * srcStride;

    const int16x8_t filter = vld1q_s16(X265_NS::g_lumaFilter[coeffIdx]);
    const int32x4_t c = vdupq_n_s32(offset);
    const uint8x16_t shr_tbl = vld1q_u8(vert_shr_tbl);

    if (width % 8 != 0)
    {
        const int16_t *s = src;
        uint8_t *d = dst;

        if (width == 12)
        {
            int16x8_t in[11];
            load_s16x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 7);

                int32x4_t sum_lo[4], sum_hi[4];
                filter8_s16x8<coeffIdx>(in + 0, filter, c, sum_lo[0], sum_hi[0]);
                filter8_s16x8<coeffIdx>(in + 1, filter, c, sum_lo[1], sum_hi[1]);
                filter8_s16x8<coeffIdx>(in + 2, filter, c, sum_lo[2], sum_hi[2]);
                filter8_s16x8<coeffIdx>(in + 3, filter, c, sum_lo[3], sum_hi[3]);

                int16x8_t sum[4];
                sum[0] = vtbl2q_s32_s16(sum_lo[0], sum_hi[0], shr_tbl);
                sum[1] = vtbl2q_s32_s16(sum_lo[1], sum_hi[1], shr_tbl);
                sum[2] = vtbl2q_s32_s16(sum_lo[2], sum_hi[2], shr_tbl);
                sum[3] = vtbl2q_s32_s16(sum_lo[3], sum_hi[3], shr_tbl);

                uint8x8_t sum_u8[4];
                sum_u8[0] = vqshrun_n_s16(sum[0], shift_offset);
                sum_u8[1] = vqshrun_n_s16(sum[1], shift_offset);
                sum_u8[2] = vqshrun_n_s16(sum[2], shift_offset);
                sum_u8[3] = vqshrun_n_s16(sum[3], shift_offset);

                store_u8x8xn<4>(d, dstStride, sum_u8);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            s = src + 8;
            d = dst + 8;
        }

        int16x4_t in[11];
        load_s16x4xn<7>(s, srcStride, in);
        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            load_s16x4xn<4>(s, srcStride, in + 7);

            int32x4_t sum[4];
            filter8_s16x4<coeffIdx>(in + 0, filter, c, sum[0]);
            filter8_s16x4<coeffIdx>(in + 1, filter, c, sum[1]);
            filter8_s16x4<coeffIdx>(in + 2, filter, c, sum[2]);
            filter8_s16x4<coeffIdx>(in + 3, filter, c, sum[3]);

            int16x8_t sum_s16[2];
            sum_s16[0] = vtbl2q_s32_s16(sum[0], sum[1], shr_tbl);
            sum_s16[1] = vtbl2q_s32_s16(sum[2], sum[3], shr_tbl);

            uint8x8_t sum_u8[2];
            sum_u8[0] = vqshrun_n_s16(sum_s16[0], shift_offset);
            sum_u8[1] = vqshrun_n_s16(sum_s16[1], shift_offset);

            store_u8x4_strided_xN<4>(d, dstStride, sum_u8);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];
            in[3] = in[7];
            in[4] = in[8];
            in[5] = in[9];
            in[6] = in[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }
    }
    else
    {
        for (int col = 0; col < width; col += 8)
        {
            const int16_t *s = src;
            uint8_t *d = dst;

            int16x8_t in[11];
            load_s16x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 7);

                int32x4_t sum_lo[4], sum_hi[4];
                filter8_s16x8<coeffIdx>(in + 0, filter, c, sum_lo[0], sum_hi[0]);
                filter8_s16x8<coeffIdx>(in + 1, filter, c, sum_lo[1], sum_hi[1]);
                filter8_s16x8<coeffIdx>(in + 2, filter, c, sum_lo[2], sum_hi[2]);
                filter8_s16x8<coeffIdx>(in + 3, filter, c, sum_lo[3], sum_hi[3]);

                int16x8_t sum[4];
                sum[0] = vtbl2q_s32_s16(sum_lo[0], sum_hi[0], shr_tbl);
                sum[1] = vtbl2q_s32_s16(sum_lo[1], sum_hi[1], shr_tbl);
                sum[2] = vtbl2q_s32_s16(sum_lo[2], sum_hi[2], shr_tbl);
                sum[3] = vtbl2q_s32_s16(sum_lo[3], sum_hi[3], shr_tbl);

                uint8x8_t sum_u8[4];
                sum_u8[0] = vqshrun_n_s16(sum[0], shift_offset);
                sum_u8[1] = vqshrun_n_s16(sum[1], shift_offset);
                sum_u8[2] = vqshrun_n_s16(sum[2], shift_offset);
                sum_u8[3] = vqshrun_n_s16(sum[3], shift_offset);

                store_u8x8xn<4>(d, dstStride, sum_u8);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            src += 8;
            dst += 8;
        }
    }
}

template<int coeffIdx, int coeffIdy, int width, int height>
void interp8_hv_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst,
                        intptr_t dstStride)
{
    const int N_TAPS = 8;
    const int v_shift = IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH;
    // Subtract 8 from shift since we account for that in table lookups.
    const int v_shift_offset = v_shift - 8;
    const int16x8_t v_filter = vld1q_s16(X265_NS::g_lumaFilter[coeffIdy]);
    const uint16x8_t h_offset = vdupq_n_u16((uint16_t)-IF_INTERNAL_OFFS);
    const int32x4_t v_offset = vdupq_n_s32((1 << (v_shift - 1)) +
                                           (IF_INTERNAL_OFFS << IF_FILTER_PREC));
    const uint8x16_t shr_tbl = vld1q_u8(vert_shr_tbl);

    src -= (N_TAPS / 2 - 1) * srcStride + (N_TAPS / 2 - 1);

    int col = 0;
    for (; col + 16 <= width; col += 16)
    {
        const pixel *s = src;
        pixel *d = dst;

        uint8x16_t h_s[N_TAPS];
        int16x8_t v_s0[11], v_s1[11];

        load_u8x16xn<8>(s + 0 * srcStride, 1, h_s);
        filter8_u8x16<coeffIdx>(h_s, h_offset, v_s0[0], v_s1[0]);

        load_u8x16xn<8>(s + 1 * srcStride, 1, h_s);
        filter8_u8x16<coeffIdx>(h_s, h_offset, v_s0[1], v_s1[1]);

        load_u8x16xn<8>(s + 2 * srcStride, 1, h_s);
        filter8_u8x16<coeffIdx>(h_s, h_offset, v_s0[2], v_s1[2]);

        load_u8x16xn<8>(s + 3 * srcStride, 1, h_s);
        filter8_u8x16<coeffIdx>(h_s, h_offset, v_s0[3], v_s1[3]);

        load_u8x16xn<8>(s + 4 * srcStride, 1, h_s);
        filter8_u8x16<coeffIdx>(h_s, h_offset, v_s0[4], v_s1[4]);

        load_u8x16xn<8>(s + 5 * srcStride, 1, h_s);
        filter8_u8x16<coeffIdx>(h_s, h_offset, v_s0[5], v_s1[5]);

        load_u8x16xn<8>(s + 6 * srcStride, 1, h_s);
        filter8_u8x16<coeffIdx>(h_s, h_offset, v_s0[6], v_s1[6]);

        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            uint8x8_t res_lo[4], res_hi[4];
            int32x4_t sum_lo[8], sum_hi[8];

            load_u8x16xn<8>(s + 0 * srcStride, 1, h_s);
            filter8_u8x16<coeffIdx>(h_s, h_offset, v_s0[7], v_s1[7]);
            filter8_s16x8<coeffIdy>(v_s0 + 0, v_filter, v_offset, sum_lo[0], sum_hi[0]);
            filter8_s16x8<coeffIdy>(v_s1 + 0, v_filter, v_offset, sum_lo[1], sum_hi[1]);
            v_s0[0] = v_s0[4];
            v_s1[0] = v_s1[4];
            res_lo[0] = vqshrun_n_s16(vtbl2q_s32_s16(sum_lo[0], sum_hi[0], shr_tbl),
                                      v_shift_offset);
            res_hi[0] = vqshrun_n_s16(vtbl2q_s32_s16(sum_lo[1], sum_hi[1], shr_tbl),
                                      v_shift_offset);

            load_u8x16xn<8>(s + 1 * srcStride, 1, h_s);
            filter8_u8x16<coeffIdx>(h_s, h_offset, v_s0[8], v_s1[8]);
            filter8_s16x8<coeffIdy>(v_s0 + 1, v_filter, v_offset, sum_lo[2], sum_hi[2]);
            filter8_s16x8<coeffIdy>(v_s1 + 1, v_filter, v_offset, sum_lo[3], sum_hi[3]);
            v_s0[1] = v_s0[5];
            v_s1[1] = v_s1[5];
            res_lo[1] = vqshrun_n_s16(vtbl2q_s32_s16(sum_lo[2], sum_hi[2], shr_tbl),
                                      v_shift_offset);
            res_hi[1] = vqshrun_n_s16(vtbl2q_s32_s16(sum_lo[3], sum_hi[3], shr_tbl),
                                      v_shift_offset);

            load_u8x16xn<8>(s + 2 * srcStride, 1, h_s);
            filter8_u8x16<coeffIdx>(h_s, h_offset, v_s0[9], v_s1[9]);
            filter8_s16x8<coeffIdy>(v_s0 + 2, v_filter, v_offset, sum_lo[4], sum_hi[4]);
            filter8_s16x8<coeffIdy>(v_s1 + 2, v_filter, v_offset, sum_lo[5], sum_hi[5]);
            v_s0[2] = v_s0[6];
            v_s1[2] = v_s1[6];
            res_lo[2] = vqshrun_n_s16(vtbl2q_s32_s16(sum_lo[4], sum_hi[4], shr_tbl),
                                      v_shift_offset);
            res_hi[2] = vqshrun_n_s16(vtbl2q_s32_s16(sum_lo[5], sum_hi[5], shr_tbl),
                                      v_shift_offset);

            load_u8x16xn<8>(s + 3 * srcStride, 1, h_s);
            filter8_u8x16<coeffIdx>(h_s, h_offset, v_s0[10], v_s1[10]);
            filter8_s16x8<coeffIdy>(v_s0 + 3, v_filter, v_offset, sum_lo[6], sum_hi[6]);
            filter8_s16x8<coeffIdy>(v_s1 + 3, v_filter, v_offset, sum_lo[7], sum_hi[7]);
            v_s0[3] = v_s0[7];
            v_s1[3] = v_s1[7];
            res_lo[3] = vqshrun_n_s16(vtbl2q_s32_s16(sum_lo[6], sum_hi[6], shr_tbl),
                                      v_shift_offset);
            res_hi[3] = vqshrun_n_s16(vtbl2q_s32_s16(sum_lo[7], sum_hi[7], shr_tbl),
                                      v_shift_offset);

            vst1q_u8(d + 0 * dstStride, vcombine_u8(res_lo[0], res_hi[0]));
            vst1q_u8(d + 1 * dstStride, vcombine_u8(res_lo[1], res_hi[1]));
            vst1q_u8(d + 2 * dstStride, vcombine_u8(res_lo[2], res_hi[2]));
            vst1q_u8(d + 3 * dstStride, vcombine_u8(res_lo[3], res_hi[3]));

            v_s0[4] = v_s0[8];
            v_s1[4] = v_s1[8];
            v_s0[5] = v_s0[9];
            v_s1[5] = v_s1[9];
            v_s0[6] = v_s0[10];
            v_s1[6] = v_s1[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        src += 16;
        dst += 16;
    }

    for (; col + 8 <= width; col += 8)
    {
        const pixel *s = src;
        pixel *d = dst;

        uint8x8_t h_s[N_TAPS];
        int16x8_t v_s[11];

        load_u8x8xn<8>(s + 0 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, v_s[0]);

        load_u8x8xn<8>(s + 1 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, v_s[1]);

        load_u8x8xn<8>(s + 2 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, v_s[2]);

        load_u8x8xn<8>(s + 3 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, v_s[3]);

        load_u8x8xn<8>(s + 4 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, v_s[4]);

        load_u8x8xn<8>(s + 5 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, v_s[5]);

        load_u8x8xn<8>(s + 6 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, v_s[6]);

        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            uint8x8_t res[4];
            int32x4_t sum_lo[4], sum_hi[4];

            load_u8x8xn<8>(s + 0 * srcStride, 1, h_s);
            filter8_u8x8<coeffIdx>(h_s, h_offset, v_s[7]);
            filter8_s16x8<coeffIdy>(v_s + 0, v_filter, v_offset, sum_lo[0], sum_hi[0]);

            load_u8x8xn<8>(s + 1 * srcStride, 1, h_s);
            filter8_u8x8<coeffIdx>(h_s, h_offset, v_s[8]);
            filter8_s16x8<coeffIdy>(v_s + 1, v_filter, v_offset, sum_lo[1], sum_hi[1]);
            v_s[0] = v_s[4];
            v_s[1] = v_s[5];
            res[0] = vqshrun_n_s16(vtbl2q_s32_s16(sum_lo[0], sum_hi[0], shr_tbl),
                                   v_shift_offset);
            res[1] = vqshrun_n_s16(vtbl2q_s32_s16(sum_lo[1], sum_hi[1], shr_tbl),
                                   v_shift_offset);

            load_u8x8xn<8>(s + 2 * srcStride, 1, h_s);
            filter8_u8x8<coeffIdx>(h_s, h_offset, v_s[9]);
            filter8_s16x8<coeffIdy>(v_s + 2, v_filter, v_offset, sum_lo[2], sum_hi[2]);

            load_u8x8xn<8>(s + 3 * srcStride, 1, h_s);
            filter8_u8x8<coeffIdx>(h_s, h_offset, v_s[10]);
            filter8_s16x8<coeffIdy>(v_s + 3, v_filter, v_offset, sum_lo[3], sum_hi[3]);
            v_s[2] = v_s[6];
            v_s[3] = v_s[7];
            res[2] = vqshrun_n_s16(vtbl2q_s32_s16(sum_lo[2], sum_hi[2], shr_tbl),
                                   v_shift_offset);
            res[3] = vqshrun_n_s16(vtbl2q_s32_s16(sum_lo[3], sum_hi[3], shr_tbl),
                                   v_shift_offset);

            store_u8xnxm<8, 4>(d + 0, dstStride, res);

            v_s[4] = v_s[8];
            v_s[5] = v_s[9];
            v_s[6] = v_s[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        src += 8;
        dst += 8;
    }

    if (width % 8 != 0)
    {
        const pixel *s = src;
        pixel *d = dst;

        uint8x8_t h_s[N_TAPS];
        int16x8_t t_v_s[11];
        int16x4_t v_s[11];

        load_u8x8xn<8>(s + 0 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, t_v_s[0]);
        v_s[0] = vget_low_s16(t_v_s[0]);

        load_u8x8xn<8>(s + 1 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, t_v_s[1]);
        v_s[1] = vget_low_s16(t_v_s[1]);

        load_u8x8xn<8>(s + 2 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, t_v_s[2]);
        v_s[2] = vget_low_s16(t_v_s[2]);

        load_u8x8xn<8>(s + 3 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, t_v_s[3]);
        v_s[3] = vget_low_s16(t_v_s[3]);

        load_u8x8xn<8>(s + 4 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, t_v_s[4]);
        v_s[4] = vget_low_s16(t_v_s[4]);

        load_u8x8xn<8>(s + 5 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, t_v_s[5]);
        v_s[5] = vget_low_s16(t_v_s[5]);

        load_u8x8xn<8>(s + 6 * srcStride, 1, h_s);
        filter8_u8x8<coeffIdx>(h_s, h_offset, t_v_s[6]);
        v_s[6] = vget_low_s16(t_v_s[6]);

        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            int32x4_t sum[4];
            uint8x8_t res[2];

            load_u8x8xn<8>(s + 0 * srcStride, 1, h_s);
            filter8_u8x8<coeffIdx>(h_s, h_offset, t_v_s[7]);
            v_s[7] = vget_low_s16(t_v_s[7]);
            filter8_s16x4<coeffIdy>(v_s + 0, v_filter, v_offset, sum[0]);
            v_s[0] = v_s[4];

            load_u8x8xn<8>(s + 1 * srcStride, 1, h_s);
            filter8_u8x8<coeffIdx>(h_s, h_offset, t_v_s[8]);
            v_s[8] = vget_low_s16(t_v_s[8]);
            filter8_s16x4<coeffIdy>(v_s + 1, v_filter, v_offset, sum[1]);
            v_s[1] = v_s[5];

            load_u8x8xn<8>(s + 2 * srcStride, 1, h_s);
            filter8_u8x8<coeffIdx>(h_s, h_offset, t_v_s[9]);
            v_s[9] = vget_low_s16(t_v_s[9]);
            filter8_s16x4<coeffIdy>(v_s + 2, v_filter, v_offset, sum[2]);
            v_s[2] = v_s[6];

            load_u8x8xn<8>(s + 3 * srcStride, 1, h_s);
            filter8_u8x8<coeffIdx>(h_s, h_offset, t_v_s[10]);
            v_s[10] = vget_low_s16(t_v_s[10]);
            filter8_s16x4<coeffIdy>(v_s + 3, v_filter, v_offset, sum[3]);
            v_s[3] = v_s[7];

            res[0] = vqshrun_n_s16(vtbl2q_s32_s16(sum[0], sum[1], shr_tbl),
                                   v_shift_offset);
            res[1] = vqshrun_n_s16(vtbl2q_s32_s16(sum[2], sum[3], shr_tbl),
                                   v_shift_offset);

            store_u8x4_strided_xN<4>(d + 0 * dstStride, dstStride, res);

            v_s[4] = v_s[8];
            v_s[5] = v_s[9];
            v_s[6] = v_s[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }
    }
}

#else // !HIGH_BIT_DEPTH

#if X265_DEPTH == 10
template<bool coeff4>
void inline filter4_u16x4(const uint16x4_t *s, uint16x4_t f,
                          const uint16x8_t offset, const uint16x4_t maxVal,
                          uint16x4_t &d)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        // Filter values are divisible by 4, factor that out in order to only
        // need a multiplication by 9 and a subtraction (which is a
        // multiplication by -1).
        uint16x4_t sum03 = vadd_u16(s[0], s[3]);
        uint16x4_t sum12 = vadd_u16(s[1], s[2]);

        int16x4_t sum =
            vreinterpret_s16_u16(vmla_n_u16(vget_low_u16(offset), sum12, 9));
        sum = vsub_s16(sum, vreinterpret_s16_u16(sum03));

        // We divided filter values by 4 so -2 from right shift.
        sum = vshr_n_s16(sum, IF_FILTER_PREC - 2);

        d = vreinterpret_u16_s16(vmax_s16(sum, vdup_n_s16(0)));
        d = vmin_u16(d, maxVal);
    }
    else
    {
        // All chroma filter taps have signs {-, +, +, -}, so we can use a
        // sequence of MLA/MLS with absolute filter values to avoid needing to
        // widen the input.

        uint16x4_t sum01 = vmul_lane_u16(s[1], f, 1);
        sum01 = vmls_lane_u16(sum01, s[0], f, 0);

        uint16x4_t sum23 = vmla_lane_u16(vget_low_u16(offset), s[2], f, 2);
        sum23 = vmls_lane_u16(sum23, s[3], f, 3);

        int32x4_t sum = vaddl_s16(vreinterpret_s16_u16(sum01),
                                  vreinterpret_s16_u16(sum23));

        // We halved filter values so -1 from right shift.
        d = vqshrun_n_s32(sum, IF_FILTER_PREC - 1);
        d = vmin_u16(d, maxVal);
    }
}

template<bool coeff4>
void inline filter4_u16x8(const uint16x8_t *s, uint16x4_t f,
                          const uint16x8_t offset, const uint16x8_t maxVal,
                          uint16x8_t &d)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        // Filter values are divisible by 4, factor that out in order to only
        // need a multiplication by 9 and a subtraction (which is a
        // multiplication by -1).
        uint16x8_t sum03 = vaddq_u16(s[0], s[3]);
        uint16x8_t sum12 = vaddq_u16(s[1], s[2]);

        int16x8_t sum = vreinterpretq_s16_u16(vmlaq_n_u16(offset, sum12, 9));
        sum = vsubq_s16(sum, vreinterpretq_s16_u16(sum03));

        // We divided filter values by 4 so -2 from right shift.
        sum = vshrq_n_s16(sum, IF_FILTER_PREC - 2);

        d = vreinterpretq_u16_s16(vmaxq_s16(sum, vdupq_n_s16(0)));
        d = vminq_u16(d, maxVal);
    }
    else
    {
        // All chroma filter taps have signs {-, +, +, -}, so we can use a
        // sequence of MLA/MLS with absolute filter values to avoid needing to
        // widen the input.
        uint16x8_t sum01 = vmulq_lane_u16(s[1], f, 1);
        sum01 = vmlsq_lane_u16(sum01, s[0], f, 0);

        uint16x8_t sum23 = vmlaq_lane_u16(offset, s[2], f, 2);
        sum23 = vmlsq_lane_u16(sum23, s[3], f, 3);

        int32x4_t sum_lo = vaddl_s16(
            vreinterpret_s16_u16(vget_low_u16(sum01)),
            vreinterpret_s16_u16(vget_low_u16(sum23)));
        int32x4_t sum_hi = vaddl_s16(
            vreinterpret_s16_u16(vget_high_u16(sum01)),
            vreinterpret_s16_u16(vget_high_u16(sum23)));

        // We halved filter values so -1 from right shift.
        uint16x4_t d0 = vqshrun_n_s32(sum_lo, IF_FILTER_PREC - 1);
        uint16x4_t d1 = vqshrun_n_s32(sum_hi, IF_FILTER_PREC - 1);

        d = vminq_u16(vcombine_u16(d0, d1), maxVal);
    }
}

#else // X265_DEPTH == 12
template<bool coeff4>
void inline filter4_u16x4(const uint16x4_t *s, const uint16x4_t f,
                          const uint32x4_t offset, const uint16x4_t maxVal,
                          uint16x4_t &d)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        // Filter values are divisible by 4, factor that out in order to only
        // need a multiplication by 9 and a subtraction (which is a
        // multiplication by -1).
        uint16x4_t sum03 = vadd_u16(s[0], s[3]);
        uint16x4_t sum12 = vadd_u16(s[1], s[2]);

        int32x4_t sum = vreinterpretq_s32_u32(vmlal_n_u16(offset, sum12, 9));
        sum = vsubw_s16(sum, vreinterpret_s16_u16(sum03));

        // We divided filter values by 4 so -2 from right shift.
        d = vqshrun_n_s32(sum, IF_FILTER_PREC - 2);
        d = vmin_u16(d, maxVal);
    }
    else
    {
        uint32x4_t sum = vmlsl_lane_u16(offset, s[0], f, 0);
        sum = vmlal_lane_u16(sum, s[1], f, 1);
        sum = vmlal_lane_u16(sum, s[2], f, 2);
        sum = vmlsl_lane_u16(sum, s[3], f, 3);

        d = vqshrun_n_s32(vreinterpretq_s32_u32(sum), IF_FILTER_PREC);
        d = vmin_u16(d, maxVal);
    }
}

template<bool coeff4>
void inline filter4_u16x8(const uint16x8_t *s, const uint16x4_t f,
                          const uint32x4_t offset, const uint16x8_t maxVal,
                          uint16x8_t &d)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        // Filter values are divisible by 4, factor that out in order to only
        // need a multiplication by 9 and a subtraction (which is a
        // multiplication by -1).
        uint16x8_t sum03 = vaddq_u16(s[0], s[3]);
        uint16x8_t sum12 = vaddq_u16(s[1], s[2]);

        int32x4_t sum_lo = vreinterpretq_s32_u32(
            vmlal_n_u16(offset, vget_low_u16(sum12), 9));
        int32x4_t sum_hi = vreinterpretq_s32_u32(
            vmlal_n_u16(offset, vget_high_u16(sum12), 9));
        sum_lo = vsubw_s16(sum_lo, vreinterpret_s16_u16(vget_low_u16(sum03)));
        sum_hi = vsubw_s16(sum_hi, vreinterpret_s16_u16(vget_high_u16(sum03)));

        // We divided filter values by 4 so -2 from right shift.
        uint16x4_t d0 = vqshrun_n_s32(sum_lo, IF_FILTER_PREC - 2);
        uint16x4_t d1 = vqshrun_n_s32(sum_hi, IF_FILTER_PREC - 2);
        d = vminq_u16(vcombine_u16(d0, d1), maxVal);
    }
    else
    {
        uint32x4_t sum_lo = vmlsl_lane_u16(offset, vget_low_u16(s[0]), f, 0);
        sum_lo = vmlal_lane_u16(sum_lo, vget_low_u16(s[1]), f, 1);
        sum_lo = vmlal_lane_u16(sum_lo, vget_low_u16(s[2]), f, 2);
        sum_lo = vmlsl_lane_u16(sum_lo, vget_low_u16(s[3]), f, 3);

        uint32x4_t sum_hi = vmlsl_lane_u16(offset, vget_high_u16(s[0]), f, 0);
        sum_hi = vmlal_lane_u16(sum_hi, vget_high_u16(s[1]), f, 1);
        sum_hi = vmlal_lane_u16(sum_hi, vget_high_u16(s[2]), f, 2);
        sum_hi = vmlsl_lane_u16(sum_hi, vget_high_u16(s[3]), f, 3);

        uint16x4_t d0 = vqshrun_n_s32(vreinterpretq_s32_u32(sum_lo),
                                      IF_FILTER_PREC);
        uint16x4_t d1 = vqshrun_n_s32(vreinterpretq_s32_u32(sum_hi),
                                      IF_FILTER_PREC);
        d = vminq_u16(vcombine_u16(d0, d1), maxVal);
    }
}
#endif // X265_DEPTH == 10

template<bool coeff4, int width, int height>
void inline interp4_horiz_pp_neon(const pixel *src, intptr_t srcStride,
                                  pixel *dst, intptr_t dstStride,
                                  const int16_t coeffIdx)
{
    const int N_TAPS = 4;
    const uint16x8_t maxVal = vdupq_n_u16((1 << X265_DEPTH) - 1);
    uint16x4_t filter = vreinterpret_u16_s16(
        vabs_s16(vld1_s16(X265_NS::g_chromaFilter[coeffIdx])));

    uint16_t offset_u16;
    // A shim of 1 << (IF_FILTER_PREC - 1) enables us to use non-rounding
    // shifts - which are generally faster than rounding shifts on modern CPUs.
    if (coeff4)
    {
        // The outermost -2 is needed because we will divide the filter values by 4.
        offset_u16 = 1 << (IF_FILTER_PREC - 1 - 2);
    }
    else
    {
        offset_u16 = 1 << (IF_FILTER_PREC - 1);
    }

#if X265_DEPTH == 10
    if (!coeff4)
    {
        // All filter values are even, halve them to avoid needing to widen to
        // 32-bit elements in filter kernels.
        filter = vshr_n_u16(filter, 1);
        offset_u16 >>= 1;
    }

    const uint16x8_t offset = vdupq_n_u16(offset_u16);
#else
    const uint32x4_t offset = vdupq_n_u32(offset_u16);
#endif // X265_DEPTH == 10

    src -= N_TAPS / 2 - 1;

    for (int row = 0; row < height; row++)
    {
        if (width % 16 == 0)
        {
            for (int col = 0; col < width; col += 16)
            {
                uint16x8_t s0[N_TAPS], s1[N_TAPS];
                load_u16x8xn<4>(src + col + 0, 1, s0);
                load_u16x8xn<4>(src + col + 8, 1, s1);

                uint16x8_t d0, d1;
                filter4_u16x8<coeff4>(s0, filter, offset, maxVal, d0);
                filter4_u16x8<coeff4>(s1, filter, offset, maxVal, d1);

                vst1q_u16(dst + col + 0, d0);
                vst1q_u16(dst + col + 8, d1);
            }
        }
        else
        {
            int col = 0;
            for (; col + 8 <= width; col += 8)
            {
                uint16x8_t s0[N_TAPS];
                load_u16x8xn<4>(src + col, 1, s0);

                uint16x8_t d0;
                filter4_u16x8<coeff4>(s0, filter, offset, maxVal, d0);

                vst1q_u16(dst + col, d0);
            }

            if (width == 6)
            {
                uint16x8_t s0[N_TAPS];
                load_u16x8xn<4>(src, 1, s0);

                uint16x8_t d0;
                filter4_u16x8<coeff4>(s0, filter, offset, maxVal, d0);

                store_u16x6xn<1>(dst, dstStride, &d0);
            }
            else if (width % 8 != 0)
            {
                uint16x4_t s0[N_TAPS];
                load_u16x4xn<4>(src + col, 1, s0);

                uint16x4_t d0;
                filter4_u16x4<coeff4>(s0, filter, offset,
                                      vget_low_u16(maxVal), d0);

                if (width == 2)
                {
                    store_u16x2xn<1>(dst + col, dstStride, &d0);
                }
                else
                {
                    vst1_u16(dst + col, d0);
                }
            }
        }

        src += srcStride;
        dst += dstStride;
    }
}

#if X265_DEPTH == 10
template<int coeffIdx>
void inline filter8_u16x4(const uint16x4_t *s, uint16x4_t &d, uint16x8_t filter,
                          uint16x4_t maxVal)
{
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        uint16x4_t sum012456 = vsub_u16(s[6], s[0]);
        sum012456 = vmla_laneq_u16(sum012456, s[1], filter, 1);
        sum012456 = vmls_laneq_u16(sum012456, s[2], filter, 2);
        sum012456 = vmla_laneq_u16(sum012456, s[4], filter, 4);
        sum012456 = vmls_laneq_u16(sum012456, s[5], filter, 5);

        uint32x4_t sum3 = vmull_laneq_u16(s[3], filter, 3);

        int32x4_t d0 = vaddw_s16(vreinterpretq_s32_u32(sum3),
                                 vreinterpret_s16_u16(sum012456));

        d = vqrshrun_n_s32(d0, IF_FILTER_PREC);
        d = vmin_u16(d, maxVal);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        uint16x4_t sum07 = vadd_u16(s[0], s[7]);
        uint16x4_t sum16 = vadd_u16(s[1], s[6]);
        uint16x4_t sum25 = vadd_u16(s[2], s[5]);
        uint16x4_t sum34 = vadd_u16(s[3], s[4]);

        uint16x4_t sum0167 = vshl_n_u16(sum16, 2);
        sum0167 = vsub_u16(sum0167, sum07);

        uint32x4_t sum2345 = vmull_laneq_u16(sum34, filter, 3);
        sum2345 = vmlsl_laneq_u16(sum2345, sum25, filter, 2);

        int32x4_t sum = vaddw_s16(vreinterpretq_s32_u32(sum2345),
                                  vreinterpret_s16_u16(sum0167));

        d = vqrshrun_n_s32(sum, IF_FILTER_PREC);
        d = vmin_u16(d, maxVal);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        uint16x4_t sum123567 = vsub_u16(s[1], s[7]);
        sum123567 = vmls_laneq_u16(sum123567, s[2], filter, 2);
        sum123567 = vmla_laneq_u16(sum123567, s[3], filter, 3);
        sum123567 = vmla_laneq_u16(sum123567, s[6], filter, 6);
        sum123567 = vmls_laneq_u16(sum123567, s[5], filter, 5);

        uint32x4_t sum4 = vmull_laneq_u16(s[4], filter, 4);

        int32x4_t d0 = vaddw_s16(vreinterpretq_s32_u32(sum4),
                                 vreinterpret_s16_u16(sum123567));

        d = vqrshrun_n_s32(d0, IF_FILTER_PREC);
        d = vmin_u16(d, maxVal);
    }
}

template<int coeffIdx>
void inline filter8_u16x8(const uint16x8_t *s, uint16x8_t &d, uint16x8_t filter,
                          uint16x8_t maxVal)
{
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        uint16x8_t sum012456 = vsubq_u16(s[6], s[0]);
        sum012456 = vmlaq_laneq_u16(sum012456, s[1], filter, 1);
        sum012456 = vmlsq_laneq_u16(sum012456, s[2], filter, 2);
        sum012456 = vmlaq_laneq_u16(sum012456, s[4], filter, 4);
        sum012456 = vmlsq_laneq_u16(sum012456, s[5], filter, 5);

        uint32x4_t sum3_lo = vmull_laneq_u16(vget_low_u16(s[3]), filter, 3);
        uint32x4_t sum3_hi = vmull_laneq_u16(vget_high_u16(s[3]), filter, 3);

        int32x4_t sum_lo = vaddw_s16(vreinterpretq_s32_u32(sum3_lo),
                                     vget_low_s16(vreinterpretq_s16_u16(sum012456)));
        int32x4_t sum_hi = vaddw_s16(vreinterpretq_s32_u32(sum3_hi),
                                     vget_high_s16(vreinterpretq_s16_u16(sum012456)));

        uint16x4_t d_lo = vqrshrun_n_s32(sum_lo, IF_FILTER_PREC);
        uint16x4_t d_hi = vqrshrun_n_s32(sum_hi, IF_FILTER_PREC);
        d = vminq_u16(vcombine_u16(d_lo, d_hi), maxVal);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        uint16x8_t sum07 = vaddq_u16(s[0], s[7]);
        uint16x8_t sum16 = vaddq_u16(s[1], s[6]);
        uint16x8_t sum25 = vaddq_u16(s[2], s[5]);
        uint16x8_t sum34 = vaddq_u16(s[3], s[4]);

        uint16x8_t sum0167 = vshlq_n_u16(sum16, 2);
        sum0167 = vsubq_u16(sum0167, sum07);

        uint32x4_t sum2345_lo = vmull_laneq_u16(vget_low_u16(sum34),
                                                filter, 3);
        sum2345_lo = vmlsl_laneq_u16(sum2345_lo, vget_low_u16(sum25),
                                     filter, 2);

        uint32x4_t sum2345_hi = vmull_laneq_u16(vget_high_u16(sum34),
                                                filter, 3);
        sum2345_hi = vmlsl_laneq_u16(sum2345_hi, vget_high_u16(sum25),
                                     filter, 2);

        int32x4_t sum_lo = vaddw_s16(vreinterpretq_s32_u32(sum2345_lo),
                                     vget_low_s16(vreinterpretq_s16_u16(sum0167)));
        int32x4_t sum_hi = vaddw_s16(vreinterpretq_s32_u32(sum2345_hi),
                                     vget_high_s16(vreinterpretq_s16_u16(sum0167)));

        uint16x4_t d_lo = vqrshrun_n_s32(sum_lo, IF_FILTER_PREC);
        uint16x4_t d_hi = vqrshrun_n_s32(sum_hi, IF_FILTER_PREC);
        d = vminq_u16(vcombine_u16(d_lo, d_hi), maxVal);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        uint16x8_t sum1234567 = vsubq_u16(s[1], s[7]);
        sum1234567 = vmlsq_laneq_u16(sum1234567, s[2], filter, 2);
        sum1234567 = vmlaq_laneq_u16(sum1234567, s[3], filter, 3);
        sum1234567 = vmlsq_laneq_u16(sum1234567, s[5], filter, 5);
        sum1234567 = vmlaq_laneq_u16(sum1234567, s[6], filter, 6);

        uint32x4_t sum4_lo = vmull_laneq_u16(vget_low_u16(s[4]), filter, 4);
        uint32x4_t sum4_hi = vmull_laneq_u16(vget_high_u16(s[4]), filter, 4);

        int32x4_t sum_lo = vaddw_s16(vreinterpretq_s32_u32(sum4_lo),
                                     vget_low_s16(vreinterpretq_s16_u16(sum1234567)));
        int32x4_t sum_hi = vaddw_s16(vreinterpretq_s32_u32(sum4_hi),
                                     vget_high_s16(vreinterpretq_s16_u16(sum1234567)));

        uint16x4_t d_lo = vqrshrun_n_s32(sum_lo, IF_FILTER_PREC);
        uint16x4_t d_hi = vqrshrun_n_s32(sum_hi, IF_FILTER_PREC);
        d = vminq_u16(vcombine_u16(d_lo, d_hi), maxVal);
    }
}

#else // X265_DEPTH == 12
template<int coeffIdx>
void inline filter8_u16x4(const uint16x4_t *s, uint16x4_t &d,
                          uint16x8_t filter, uint16x4_t maxVal)
{
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        uint16x4_t sum0156 = vsub_u16(s[6], s[0]);
        sum0156 = vmla_laneq_u16(sum0156, s[1], filter, 1);
        sum0156 = vmls_laneq_u16(sum0156, s[5], filter, 5);

        uint32x4_t sum234 = vmull_laneq_u16(s[3], filter, 3);
        sum234 = vmlsl_laneq_u16(sum234, s[2], filter, 2);
        sum234 = vmlal_laneq_u16(sum234, s[4], filter, 4);

        int32x4_t sum = vaddw_s16(vreinterpretq_s32_u32(sum234),
                                  vreinterpret_s16_u16(sum0156));

        d = vqrshrun_n_s32(sum, IF_FILTER_PREC);
        d = vmin_u16(d, maxVal);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        uint16x4_t sum07 = vadd_u16(s[0], s[7]);
        uint16x4_t sum16 = vadd_u16(s[1], s[6]);
        uint16x4_t sum25 = vadd_u16(s[2], s[5]);
        uint16x4_t sum34 = vadd_u16(s[3], s[4]);

        uint16x4_t sum0167 = vshl_n_u16(sum16, 2);
        sum0167 = vsub_u16(sum0167, sum07);

        uint32x4_t sum2345 = vmull_laneq_u16(sum34, filter, 3);
        sum2345 = vmlsl_laneq_u16(sum2345, sum25, filter, 2);

        int32x4_t sum = vaddw_s16(vreinterpretq_s32_u32(sum2345),
                                  vreinterpret_s16_u16(sum0167));

        d = vqrshrun_n_s32(sum, IF_FILTER_PREC);
        d = vmin_u16(d, maxVal);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        uint16x4_t sum1267 = vsub_u16(s[1], s[7]);
        sum1267 = vmls_laneq_u16(sum1267, s[2], filter, 2);
        sum1267 = vmla_laneq_u16(sum1267, s[6], filter, 6);

        uint32x4_t sum345 = vmull_laneq_u16(s[3], filter, 3);
        sum345 = vmlal_laneq_u16(sum345, s[4], filter, 4);
        sum345 = vmlsl_laneq_u16(sum345, s[5], filter, 5);

        int32x4_t sum = vaddw_s16(vreinterpretq_s32_u32(sum345),
                                  vreinterpret_s16_u16(sum1267));

        d = vqrshrun_n_s32(sum, IF_FILTER_PREC);
        d = vmin_u16(d, maxVal);
    }
}

template<int coeffIdx>
void inline filter8_u16x8(const uint16x8_t *s, uint16x8_t &d, uint16x8_t filter,
                          uint16x8_t maxVal)
{
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        uint16x8_t sum0156 = vsubq_u16(s[6], s[0]);
        sum0156 = vmlaq_laneq_u16(sum0156, s[1], filter, 1);
        sum0156 = vmlsq_laneq_u16(sum0156, s[5], filter, 5);

        uint32x4_t sum234_lo = vmull_laneq_u16(vget_low_u16(s[3]), filter, 3);
        sum234_lo = vmlsl_laneq_u16(sum234_lo, vget_low_u16(s[2]), filter, 2);
        sum234_lo = vmlal_laneq_u16(sum234_lo, vget_low_u16(s[4]), filter, 4);

        uint32x4_t sum234_hi = vmull_laneq_u16(vget_high_u16(s[3]), filter, 3);
        sum234_hi = vmlsl_laneq_u16(sum234_hi, vget_high_u16(s[2]), filter, 2);
        sum234_hi = vmlal_laneq_u16(sum234_hi, vget_high_u16(s[4]), filter, 4);

        int32x4_t sum_lo = vaddw_s16(vreinterpretq_s32_u32(sum234_lo),
                                     vget_low_s16(vreinterpretq_s16_u16(sum0156)));
        int32x4_t sum_hi = vaddw_s16(vreinterpretq_s32_u32(sum234_hi),
                                     vget_high_s16(vreinterpretq_s16_u16(sum0156)));

        uint16x4_t d_lo = vqrshrun_n_s32(sum_lo, IF_FILTER_PREC);
        uint16x4_t d_hi = vqrshrun_n_s32(sum_hi, IF_FILTER_PREC);
        d = vminq_u16(vcombine_u16(d_lo, d_hi), maxVal);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        uint16x8_t sum07 = vaddq_u16(s[0], s[7]);
        uint16x8_t sum16 = vaddq_u16(s[1], s[6]);
        uint16x8_t sum25 = vaddq_u16(s[2], s[5]);
        uint16x8_t sum34 = vaddq_u16(s[3], s[4]);

        uint16x8_t sum0167 = vshlq_n_u16(sum16, 2);
        sum0167 = vsubq_u16(sum0167, sum07);

        uint32x4_t sum2345_lo = vmull_laneq_u16(vget_low_u16(sum34),
                                                filter, 3);
        sum2345_lo = vmlsl_laneq_u16(sum2345_lo, vget_low_u16(sum25),
                                     filter, 2);

        uint32x4_t sum2345_hi = vmull_laneq_u16(vget_high_u16(sum34),
                                                filter, 3);
        sum2345_hi = vmlsl_laneq_u16(sum2345_hi, vget_high_u16(sum25),
                                     filter, 2);

        int32x4_t sum_lo = vaddw_s16(vreinterpretq_s32_u32(sum2345_lo),
                                     vget_low_s16(vreinterpretq_s16_u16(sum0167)));
        int32x4_t sum_hi = vaddw_s16(vreinterpretq_s32_u32(sum2345_hi),
                                     vget_high_s16(vreinterpretq_s16_u16(sum0167)));

        uint16x4_t d_lo = vqrshrun_n_s32(sum_lo, IF_FILTER_PREC);
        uint16x4_t d_hi = vqrshrun_n_s32(sum_hi, IF_FILTER_PREC);
        d = vminq_u16(vcombine_u16(d_lo, d_hi), maxVal);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        uint16x8_t sum1267 = vsubq_u16(s[1], s[7]);
        sum1267 = vmlsq_laneq_u16(sum1267, s[2], filter, 2);
        sum1267 = vmlaq_laneq_u16(sum1267, s[6], filter, 6);

        uint32x4_t sum345_lo = vmull_laneq_u16(vget_low_u16(s[3]), filter, 3);
        sum345_lo = vmlal_laneq_u16(sum345_lo, vget_low_u16(s[4]), filter, 4);
        sum345_lo = vmlsl_laneq_u16(sum345_lo, vget_low_u16(s[5]), filter, 5);

        uint32x4_t sum345_hi = vmull_laneq_u16(vget_high_u16(s[3]), filter, 3);
        sum345_hi = vmlal_laneq_u16(sum345_hi, vget_high_u16(s[4]), filter, 4);
        sum345_hi = vmlsl_laneq_u16(sum345_hi, vget_high_u16(s[5]), filter, 5);

        int32x4_t sum_lo = vaddw_s16(vreinterpretq_s32_u32(sum345_lo),
                                     vget_low_s16(vreinterpretq_s16_u16(sum1267)));
        int32x4_t sum_hi = vaddw_s16(vreinterpretq_s32_u32(sum345_hi),
                                     vget_high_s16(vreinterpretq_s16_u16(sum1267)));

        uint16x4_t d_lo = vqrshrun_n_s32(sum_lo, IF_FILTER_PREC);
        uint16x4_t d_hi = vqrshrun_n_s32(sum_hi, IF_FILTER_PREC);

        d = vminq_u16(vcombine_u16(d_lo, d_hi), maxVal);
    }
}

#endif // X265_DEPTH == 10

template<int coeffIdx, int width, int height>
void inline interp8_horiz_pp_neon(const pixel *src, intptr_t srcStride,
                                  pixel *dst, intptr_t dstStride)
{
    const int N_TAPS = 8;
    const uint16x8_t maxVal = vdupq_n_u16((1 << X265_DEPTH) - 1);

    const uint16x8_t filter =
        vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(X265_NS::g_lumaFilter[coeffIdx])));

    src -= N_TAPS / 2 - 1;

    for (int row = 0; row < height; row++)
    {
        if (width % 16 == 0)
        {
            for (int col = 0; col < width; col += 16)
            {
                uint16x8_t s0[N_TAPS], s1[N_TAPS];
                load_u16x8xn<8>(src + col + 0, 1, s0);
                load_u16x8xn<8>(src + col + 8, 1, s1);

                uint16x8_t d0, d1;
                filter8_u16x8<coeffIdx>(s0, d0, filter, maxVal);
                filter8_u16x8<coeffIdx>(s1, d1, filter, maxVal);

                vst1q_u16(dst + col + 0, d0);
                vst1q_u16(dst + col + 8, d1);
            }
        }
        else
        {
            int col = 0;
            for (; col + 8 <= width; col += 8)
            {
                uint16x8_t s0[N_TAPS];
                load_u16x8xn<8>(src + col, 1, s0);

                uint16x8_t d0;
                filter8_u16x8<coeffIdx>(s0, d0, filter, maxVal);

                vst1q_u16(dst + col, d0);
            }

            if (width % 8 == 4)
            {
                uint16x4_t s0[N_TAPS];
                load_u16x4xn<8>(src + col, 1, s0);

                uint16x4_t d0;
                filter8_u16x4<coeffIdx>(s0, d0, filter, vget_low_u16(maxVal));

                vst1_u16(dst + col, d0);
            }
        }

        src += srcStride;
        dst += dstStride;
    }
}

#if X265_DEPTH == 10
template<int coeff4>
void inline filter4_ps_u16x4(const uint16x4_t *s, const uint16x4_t f,
                             const uint16x8_t offset, int16x4_t &d)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        // Filter values are divisible by 4, factor that out in order to only
        // need a multiplication by 9 and a subtraction (which is a
        // multiplication by -1).
        uint16x4_t sum03 = vadd_u16(s[0], s[3]);
        uint16x4_t sum12 = vadd_u16(s[1], s[2]);

        int16x4_t sum =
            vreinterpret_s16_u16(vmla_n_u16(vget_low_u16(offset), sum12, 9));
        d = vsub_s16(sum, vreinterpret_s16_u16(sum03));
    }
    else
    {
        uint16x4_t sum = vmls_lane_u16(vget_low_u16(offset), s[0], f, 0);
        sum = vmla_lane_u16(sum, s[1], f, 1);
        sum = vmla_lane_u16(sum, s[2], f, 2);
        sum = vmls_lane_u16(sum, s[3], f, 3);

        // We halved filter values so -1 from right shift.
        d = vshr_n_s16(vreinterpret_s16_u16(sum), SHIFT_INTERP_PS - 1);
    }
}

template<bool coeff4>
void inline filter4_ps_u16x8(const uint16x8_t *s, const uint16x4_t f,
                             const uint16x8_t offset, int16x8_t &d)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        // Filter values are divisible by 4, factor that out in order to only
        // need a multiplication by 9 and a subtraction (which is a
        // multiplication by -1).
        uint16x8_t sum03 = vaddq_u16(s[0], s[3]);
        uint16x8_t sum12 = vaddq_u16(s[1], s[2]);

        int16x8_t sum =
            vreinterpretq_s16_u16(vmlaq_n_u16(offset, sum12, 9));
        d = vsubq_s16(sum, vreinterpretq_s16_u16(sum03));
    }
    else
    {
        uint16x8_t sum = vmlsq_lane_u16(offset, s[0], f, 0);
        sum = vmlaq_lane_u16(sum, s[1], f, 1);
        sum = vmlaq_lane_u16(sum, s[2], f, 2);
        sum = vmlsq_lane_u16(sum, s[3], f, 3);

        // We halved filter values so -1 from right shift.
        d = vshrq_n_s16(vreinterpretq_s16_u16(sum), SHIFT_INTERP_PS - 1);
    }
}

#else // X265_DEPTH == 12
template<int coeff4>
void inline filter4_ps_u16x4(const uint16x4_t *s, const uint16x4_t f,
                             const uint32x4_t offset, int16x4_t &d)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        // Filter values are divisible by 4, factor that out in order to only
        // need a multiplication by 9 and a subtraction (which is a
        // multiplication by -1).
        uint16x4_t sum03 = vadd_u16(s[0], s[3]);
        uint16x4_t sum12 = vadd_u16(s[1], s[2]);

        int32x4_t sum = vreinterpretq_s32_u32(vmlal_n_u16(offset, sum12, 9));
        sum = vsubw_s16(sum, vreinterpret_s16_u16(sum03));

        // We divided filter values by 4 so -2 from right shift.
        d = vshrn_n_s32(sum, SHIFT_INTERP_PS - 2);
    }
    else
    {
        uint32x4_t sum = vmlsl_lane_u16(offset, s[0], f, 0);
        sum = vmlal_lane_u16(sum, s[1], f, 1);
        sum = vmlal_lane_u16(sum, s[2], f, 2);
        sum = vmlsl_lane_u16(sum, s[3], f, 3);

        d = vshrn_n_s32(vreinterpretq_s32_u32(sum), SHIFT_INTERP_PS);
    }
}

template<bool coeff4>
void inline filter4_ps_u16x8(const uint16x8_t *s, const uint16x4_t f,
                             const uint32x4_t offset, int16x8_t &d)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        // Filter values are divisible by 4, factor that out in order to only
        // need a multiplication by 9 and a subtraction (which is a
        // multiplication by -1).
        uint16x8_t sum03 = vaddq_u16(s[0], s[3]);
        uint16x8_t sum12 = vaddq_u16(s[1], s[2]);

        int32x4_t sum_lo = vreinterpretq_s32_u32(
            vmlal_n_u16(offset, vget_low_u16(sum12), 9));
        int32x4_t sum_hi = vreinterpretq_s32_u32(
            vmlal_n_u16(offset, vget_high_u16(sum12), 9));
        sum_lo = vsubw_s16(sum_lo, vreinterpret_s16_u16(vget_low_u16(sum03)));
        sum_hi = vsubw_s16(sum_hi, vreinterpret_s16_u16(vget_high_u16(sum03)));

        // We divided filter values by 4 so -2 from right shift.
        int16x4_t d0 = vshrn_n_s32(sum_lo, SHIFT_INTERP_PS - 2);
        int16x4_t d1 = vshrn_n_s32(sum_hi, SHIFT_INTERP_PS - 2);
        d = vcombine_s16(d0, d1);
    }
    else
    {
        uint32x4_t sum_lo = vmlsl_lane_u16(offset, vget_low_u16(s[0]), f, 0);
        sum_lo = vmlal_lane_u16(sum_lo, vget_low_u16(s[1]), f, 1);
        sum_lo = vmlal_lane_u16(sum_lo, vget_low_u16(s[2]), f, 2);
        sum_lo = vmlsl_lane_u16(sum_lo, vget_low_u16(s[3]), f, 3);

        uint32x4_t sum_hi = vmlsl_lane_u16(offset, vget_high_u16(s[0]), f, 0);
        sum_hi = vmlal_lane_u16(sum_hi, vget_high_u16(s[1]), f, 1);
        sum_hi = vmlal_lane_u16(sum_hi, vget_high_u16(s[2]), f, 2);
        sum_hi = vmlsl_lane_u16(sum_hi, vget_high_u16(s[3]), f, 3);

        int16x4_t d0 = vshrn_n_s32(vreinterpretq_s32_u32(sum_lo),
                                   SHIFT_INTERP_PS);
        int16x4_t d1 = vshrn_n_s32(vreinterpretq_s32_u32(sum_hi),
                                   SHIFT_INTERP_PS);
        d = vcombine_s16(d0, d1);
    }
}

#endif // X265_DEPTH == 10

template<int coeff4, int width, int height>
void interp4_horiz_ps_neon(const pixel *src, intptr_t srcStride, int16_t *dst,
                           intptr_t dstStride, int coeffIdx, int isRowExt)
{
    const int N_TAPS = 4;
    int blkheight = height;
    uint16x4_t filter = vreinterpret_u16_s16(
        vabs_s16(vld1_s16(X265_NS::g_chromaFilter[coeffIdx])));
    uint32_t offset_u32;

    if (coeff4)
    {
        // The -2 is needed because we will divide the filter values by 4.
        offset_u32 = (unsigned)-IF_INTERNAL_OFFS << (SHIFT_INTERP_PS - 2);
    }
    else
    {
        offset_u32 = (unsigned)-IF_INTERNAL_OFFS << SHIFT_INTERP_PS;
    }
#if X265_DEPTH == 10
    if (!coeff4)
    {
        // All filter values are even, halve them to avoid needing to widen to
        // 32-bit elements in filter kernels.
        filter = vshr_n_u16(filter, 1);
        offset_u32 >>= 1;
    }

    const uint16x8_t offset = vdupq_n_u16((uint16_t)offset_u32);
#else
    const uint32x4_t offset = vdupq_n_u32(offset_u32);
#endif // X265_DEPTH == 10

    if (isRowExt)
    {
        src -= (N_TAPS / 2 - 1) * srcStride;
        blkheight += N_TAPS - 1;
    }

    src -= N_TAPS / 2 - 1;

    for (int row = 0; row < blkheight; row++)
    {
        if (width % 16 == 0)
        {
            for (int col = 0; col < width; col += 16)
            {
                uint16x8_t s0[N_TAPS], s1[N_TAPS];
                load_u16x8xn<4>(src + col + 0, 1, s0);
                load_u16x8xn<4>(src + col + 8, 1, s1);

                int16x8_t d0, d1;
                filter4_ps_u16x8<coeff4>(s0, filter, offset, d0);
                filter4_ps_u16x8<coeff4>(s1, filter, offset, d1);

                vst1q_s16(dst + col + 0, d0);
                vst1q_s16(dst + col + 8, d1);
            }
        }
        else
        {
            int col = 0;
            for (; col + 8 <= width; col += 8)
            {
                uint16x8_t s0[N_TAPS];
                load_u16x8xn<4>(src + col, 1, s0);

                int16x8_t d0;
                filter4_ps_u16x8<coeff4>(s0, filter, offset, d0);

                vst1q_s16(dst + col, d0);
            }

            if (width == 6)
            {
                uint16x8_t s0[N_TAPS];
                load_u16x8xn<4>(src, 1, s0);

                int16x8_t d0;
                filter4_ps_u16x8<coeff4>(s0, filter, offset, d0);

                store_s16x6xn<1>(dst, dstStride, &d0);
            }
            else if (width % 8 != 0)
            {
                uint16x4_t s0[N_TAPS];
                load_u16x4xn<4>(src + col, 1, s0);

                int16x4_t d0;
                filter4_ps_u16x4<coeff4>(s0, filter, offset, d0);

                if (width == 2)
                {
                    store_s16x2xn<1>(dst + col, dstStride, &d0);
                }
                else
                {
                    vst1_s16(dst + col, d0);
                }
            }
        }

        src += srcStride;
        dst += dstStride;
    }
}

#if X265_DEPTH == 10
template<int coeffIdx>
void inline filter8_ps_u16x4(const uint16x4_t *s, int16x4_t &d,
                             uint32x4_t offset, uint16x8_t filter)
{
    uint16x4_t offset_u16 = vdup_n_u16((uint16_t)vgetq_lane_u32(offset, 0));

    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        uint16x4_t sum012456 = vsub_u16(s[6], s[0]);
        sum012456 = vmla_laneq_u16(sum012456, s[1], filter, 1);
        sum012456 = vmls_laneq_u16(sum012456, s[2], filter, 2);
        sum012456 = vmla_laneq_u16(sum012456, s[4], filter, 4);
        sum012456 = vmls_laneq_u16(sum012456, s[5], filter, 5);

        uint16x4_t sum3 =
            vmla_laneq_u16(offset_u16, s[3], filter, 3);

        int32x4_t sum = vaddl_s16(vreinterpret_s16_u16(sum3),
                                  vreinterpret_s16_u16(sum012456));

        d = vshrn_n_s32(sum, SHIFT_INTERP_PS);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        uint16x4_t sum07 = vadd_u16(s[0], s[7]);
        uint16x4_t sum16 = vadd_u16(s[1], s[6]);
        uint16x4_t sum25 = vadd_u16(s[2], s[5]);
        uint16x4_t sum34 = vadd_u16(s[3], s[4]);

        uint16x4_t sum0167 = vshl_n_u16(sum16, 2);
        sum0167 = vsub_u16(sum0167, sum07);

        uint32x4_t sum2345 = vmlal_laneq_u16(offset, sum34, filter, 3);
        sum2345 = vmlsl_laneq_u16(sum2345, sum25, filter, 2);

        int32x4_t sum = vaddw_s16(vreinterpretq_s32_u32(sum2345),
                                  vreinterpret_s16_u16(sum0167));

        d = vshrn_n_s32(sum, SHIFT_INTERP_PS);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        uint16x4_t sum123567 = vsub_u16(s[1], s[7]);
        sum123567 = vmls_laneq_u16(sum123567, s[2], filter, 2);
        sum123567 = vmla_laneq_u16(sum123567, s[3], filter, 3);
        sum123567 = vmla_laneq_u16(sum123567, s[6], filter, 6);
        sum123567 = vmls_laneq_u16(sum123567, s[5], filter, 5);

        uint16x4_t sum4 =
            vmla_laneq_u16(offset_u16, s[4], filter, 4);

        int32x4_t sum = vaddl_s16(vreinterpret_s16_u16(sum4),
                                  vreinterpret_s16_u16(sum123567));

        d = vshrn_n_s32(sum, SHIFT_INTERP_PS);
    }
}

template<int coeffIdx>
void inline filter8_ps_u16x8(const uint16x8_t *s, int16x8_t &d,
                             uint32x4_t offset, uint16x8_t filter)
{
    uint16x8_t offset_u16 = vdupq_n_u16((uint16_t)vgetq_lane_u32(offset, 0));

    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        uint16x8_t sum012456 = vsubq_u16(s[6], s[0]);
        sum012456 = vmlaq_laneq_u16(sum012456, s[1], filter, 1);
        sum012456 = vmlsq_laneq_u16(sum012456, s[2], filter, 2);
        sum012456 = vmlaq_laneq_u16(sum012456, s[4], filter, 4);
        sum012456 = vmlsq_laneq_u16(sum012456, s[5], filter, 5);

        uint16x8_t sum3 =
            vmlaq_laneq_u16(offset_u16, s[3], filter, 3);

        int32x4_t sum_lo = vaddl_s16(vget_low_s16(vreinterpretq_s16_u16(sum3)),
                                     vget_low_s16(vreinterpretq_s16_u16(sum012456)));
        int32x4_t sum_hi = vaddl_s16(vget_high_s16(vreinterpretq_s16_u16(sum3)),
                                     vget_high_s16(vreinterpretq_s16_u16(sum012456)));

        int16x4_t d_lo = vshrn_n_s32(sum_lo, SHIFT_INTERP_PS);
        int16x4_t d_hi = vshrn_n_s32(sum_hi, SHIFT_INTERP_PS);
        d = vcombine_s16(d_lo, d_hi);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        uint16x8_t sum07 = vaddq_u16(s[0], s[7]);
        uint16x8_t sum16 = vaddq_u16(s[1], s[6]);
        uint16x8_t sum25 = vaddq_u16(s[2], s[5]);
        uint16x8_t sum34 = vaddq_u16(s[3], s[4]);

        uint16x8_t sum0167 = vshlq_n_u16(sum16, 2);
        sum0167 = vsubq_u16(sum0167, sum07);

        uint32x4_t sum2345_lo = vmlal_laneq_u16(offset, vget_low_u16(sum34),
                                                filter, 3);
        sum2345_lo = vmlsl_laneq_u16(sum2345_lo, vget_low_u16(sum25),
                                     filter, 2);

        uint32x4_t sum2345_hi = vmlal_laneq_u16(offset, vget_high_u16(sum34),
                                                filter, 3);
        sum2345_hi = vmlsl_laneq_u16(sum2345_hi, vget_high_u16(sum25),
                                     filter, 2);

        int32x4_t sum_lo = vaddw_s16(vreinterpretq_s32_u32(sum2345_lo),
                                     vget_low_s16(vreinterpretq_s16_u16(sum0167)));
        int32x4_t sum_hi = vaddw_s16(vreinterpretq_s32_u32(sum2345_hi),
                                     vget_high_s16(vreinterpretq_s16_u16(sum0167)));

        int16x4_t d_lo = vshrn_n_s32(sum_lo, SHIFT_INTERP_PS);
        int16x4_t d_hi = vshrn_n_s32(sum_hi, SHIFT_INTERP_PS);
        d = vcombine_s16(d_lo, d_hi);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        uint16x8_t sum123567 = vsubq_u16(s[1], s[7]);
        sum123567 = vmlsq_laneq_u16(sum123567, s[2], filter, 2);
        sum123567 = vmlaq_laneq_u16(sum123567, s[3], filter, 3);
        sum123567 = vmlaq_laneq_u16(sum123567, s[6], filter, 6);
        sum123567 = vmlsq_laneq_u16(sum123567, s[5], filter, 5);

        uint16x8_t sum4 =
            vmlaq_laneq_u16(offset_u16, s[4], filter, 4);

        int32x4_t sum_lo = vaddl_s16(vget_low_s16(vreinterpretq_s16_u16(sum4)),
                                     vget_low_s16(vreinterpretq_s16_u16(sum123567)));
        int32x4_t sum_hi = vaddl_s16(vget_high_s16(vreinterpretq_s16_u16(sum4)),
                                     vget_high_s16(vreinterpretq_s16_u16(sum123567)));

        int16x4_t d_lo = vshrn_n_s32(sum_lo, SHIFT_INTERP_PS);
        int16x4_t d_hi = vshrn_n_s32(sum_hi, SHIFT_INTERP_PS);
        d = vcombine_s16(d_lo, d_hi);
    }
}

#else // X265_DEPTH == 12
template<int coeffIdx>
void inline filter8_ps_u16x4(const uint16x4_t *s, int16x4_t &d,
                             uint32x4_t offset, uint16x8_t filter)
{
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        uint16x4_t sum0156 = vsub_u16(s[6], s[0]);
        sum0156 = vmla_laneq_u16(sum0156, s[1], filter, 1);
        sum0156 = vmls_laneq_u16(sum0156, s[5], filter, 5);

        uint32x4_t sum234 = vmlal_laneq_u16(offset, s[3], filter, 3);
        sum234 = vmlsl_laneq_u16(sum234, s[2], filter, 2);
        sum234 = vmlal_laneq_u16(sum234, s[4], filter, 4);

        int32x4_t sum = vaddw_s16(vreinterpretq_s32_u32(sum234),
                                  vreinterpret_s16_u16(sum0156));

        d = vshrn_n_s32(sum, SHIFT_INTERP_PS);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        uint16x4_t sum07 = vadd_u16(s[0], s[7]);
        uint16x4_t sum16 = vadd_u16(s[1], s[6]);
        uint16x4_t sum25 = vadd_u16(s[2], s[5]);
        uint16x4_t sum34 = vadd_u16(s[3], s[4]);

        uint16x4_t sum0167 = vshl_n_u16(sum16, 2);
        sum0167 = vsub_u16(sum0167, sum07);

        uint32x4_t sum2345 = vmlal_laneq_u16(offset, sum34, filter, 3);
        sum2345 = vmlsl_laneq_u16(sum2345, sum25, filter, 2);

        int32x4_t sum = vaddw_s16(vreinterpretq_s32_u32(sum2345),
                                  vreinterpret_s16_u16(sum0167));

        d = vshrn_n_s32(sum, SHIFT_INTERP_PS);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        uint16x4_t sum1267 = vsub_u16(s[1], s[7]);
        sum1267 = vmls_laneq_u16(sum1267, s[2], filter, 2);
        sum1267 = vmla_laneq_u16(sum1267, s[6], filter, 6);

        uint32x4_t sum345 = vmlal_laneq_u16(offset, s[3], filter, 3);
        sum345 = vmlal_laneq_u16(sum345, s[4], filter, 4);
        sum345 = vmlsl_laneq_u16(sum345, s[5], filter, 5);

        int32x4_t sum = vaddw_s16(vreinterpretq_s32_u32(sum345),
                                  vreinterpret_s16_u16(sum1267));

        d = vshrn_n_s32(sum, SHIFT_INTERP_PS);
    }
}

template<int coeffIdx>
void inline filter8_ps_u16x8(const uint16x8_t *s, int16x8_t &d,
                             uint32x4_t offset, uint16x8_t filter)
{
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        uint16x8_t sum0156 = vsubq_u16(s[6], s[0]);
        sum0156 = vmlaq_laneq_u16(sum0156, s[1], filter, 1);
        sum0156 = vmlsq_laneq_u16(sum0156, s[5], filter, 5);

        uint32x4_t sum234_lo = vmlal_laneq_u16(offset, vget_low_u16(s[3]), filter, 3);
        sum234_lo = vmlsl_laneq_u16(sum234_lo, vget_low_u16(s[2]), filter, 2);
        sum234_lo = vmlal_laneq_u16(sum234_lo, vget_low_u16(s[4]), filter, 4);

        uint32x4_t sum234_hi = vmlal_laneq_u16(offset, vget_high_u16(s[3]), filter, 3);
        sum234_hi = vmlsl_laneq_u16(sum234_hi, vget_high_u16(s[2]), filter, 2);
        sum234_hi = vmlal_laneq_u16(sum234_hi, vget_high_u16(s[4]), filter, 4);

        int32x4_t sum_lo = vaddw_s16(vreinterpretq_s32_u32(sum234_lo),
                                     vget_low_s16(vreinterpretq_s16_u16(sum0156)));
        int32x4_t sum_hi = vaddw_s16(vreinterpretq_s32_u32(sum234_hi),
                                     vget_high_s16(vreinterpretq_s16_u16(sum0156)));

        int16x4_t d_lo = vshrn_n_s32(sum_lo, SHIFT_INTERP_PS);
        int16x4_t d_hi = vshrn_n_s32(sum_hi, SHIFT_INTERP_PS);
        d = vcombine_s16(d_lo, d_hi);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        uint16x8_t sum07 = vaddq_u16(s[0], s[7]);
        uint16x8_t sum16 = vaddq_u16(s[1], s[6]);
        uint16x8_t sum25 = vaddq_u16(s[2], s[5]);
        uint16x8_t sum34 = vaddq_u16(s[3], s[4]);

        uint16x8_t sum0167 = vshlq_n_u16(sum16, 2);
        sum0167 = vsubq_u16(sum0167, sum07);

        uint32x4_t sum2345_lo = vmlal_laneq_u16(offset, vget_low_u16(sum34),
                                                filter, 3);
        sum2345_lo = vmlsl_laneq_u16(sum2345_lo, vget_low_u16(sum25),
                                     filter, 2);

        uint32x4_t sum2345_hi = vmlal_laneq_u16(offset, vget_high_u16(sum34),
                                                filter, 3);
        sum2345_hi = vmlsl_laneq_u16(sum2345_hi, vget_high_u16(sum25),
                                     filter, 2);

        int32x4_t sum_lo = vaddw_s16(vreinterpretq_s32_u32(sum2345_lo),
                                     vget_low_s16(vreinterpretq_s16_u16(sum0167)));
        int32x4_t sum_hi = vaddw_s16(vreinterpretq_s32_u32(sum2345_hi),
                                     vget_high_s16(vreinterpretq_s16_u16(sum0167)));

        int16x4_t d_lo = vshrn_n_s32(sum_lo, SHIFT_INTERP_PS);
        int16x4_t d_hi = vshrn_n_s32(sum_hi, SHIFT_INTERP_PS);
        d = vcombine_s16(d_lo, d_hi);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        uint16x8_t sum1267 = vsubq_u16(s[1], s[7]);
        sum1267 = vmlsq_laneq_u16(sum1267, s[2], filter, 2);
        sum1267 = vmlaq_laneq_u16(sum1267, s[6], filter, 6);

        uint32x4_t sum345_lo = vmlal_laneq_u16(offset, vget_low_u16(s[3]), filter, 3);
        sum345_lo = vmlal_laneq_u16(sum345_lo, vget_low_u16(s[4]), filter, 4);
        sum345_lo = vmlsl_laneq_u16(sum345_lo, vget_low_u16(s[5]), filter, 5);

        uint32x4_t sum345_hi = vmlal_laneq_u16(offset, vget_high_u16(s[3]), filter, 3);
        sum345_hi = vmlal_laneq_u16(sum345_hi, vget_high_u16(s[4]), filter, 4);
        sum345_hi = vmlsl_laneq_u16(sum345_hi, vget_high_u16(s[5]), filter, 5);

        int32x4_t sum_lo = vaddw_s16(vreinterpretq_s32_u32(sum345_lo),
                                     vget_low_s16(vreinterpretq_s16_u16(sum1267)));
        int32x4_t sum_hi = vaddw_s16(vreinterpretq_s32_u32(sum345_hi),
                                     vget_high_s16(vreinterpretq_s16_u16(sum1267)));

        int16x4_t d_lo = vshrn_n_s32(sum_lo, SHIFT_INTERP_PS);
        int16x4_t d_hi = vshrn_n_s32(sum_hi, SHIFT_INTERP_PS);

        d = vcombine_s16(d_lo, d_hi);
    }
}

#endif // X265_DEPTH == 10

template<int coeffIdx, int width, int height>
void interp8_horiz_ps_neon(const pixel *src, intptr_t srcStride, int16_t *dst,
                           intptr_t dstStride, int isRowExt)
{
    const int N_TAPS = 8;
    int blkheight = height;
    const uint16x8_t filter =
        vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(X265_NS::g_lumaFilter[coeffIdx])));
    uint32x4_t offset =
        vdupq_n_u32((unsigned)-IF_INTERNAL_OFFS << SHIFT_INTERP_PS);

    if (isRowExt)
    {
        src -= (N_TAPS / 2 - 1) * srcStride;
        blkheight += N_TAPS - 1;
    }

    src -= N_TAPS / 2 - 1;

    for (int row = 0; row < blkheight; row++)
    {
        if (width % 16 == 0)
        {
            for (int col = 0; col < width; col += 16)
            {
                uint16x8_t s0[N_TAPS], s1[N_TAPS];
                load_u16x8xn<8>(src + col + 0, 1, s0);
                load_u16x8xn<8>(src + col + 8, 1, s1);

                int16x8_t d0, d1;
                filter8_ps_u16x8<coeffIdx>(s0, d0, offset, filter);
                filter8_ps_u16x8<coeffIdx>(s1, d1, offset, filter);

                vst1q_s16(dst + col + 0, d0);
                vst1q_s16(dst + col + 8, d1);
            }
        }
        else
        {
            int col = 0;
            for (; col + 8 <= width; col += 8)
            {
                uint16x8_t s0[N_TAPS];
                load_u16x8xn<8>(src + col, 1, s0);

                int16x8_t d0;
                filter8_ps_u16x8<coeffIdx>(s0, d0, offset, filter);

                vst1q_s16(dst + col, d0);
            }

            if (width % 8 == 4)
            {
                uint16x4_t s0[N_TAPS];
                load_u16x4xn<8>(src + col, 1, s0);

                int16x4_t d0;
                filter8_ps_u16x4<coeffIdx>(s0, d0, offset, filter);

                vst1_s16(dst + col, d0);
            }
        }

        src += srcStride;
        dst += dstStride;
    }
}

template<bool coeff4, int width, int height>
void inline interp4_vert_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst,
                                 intptr_t dstStride, const int16_t coeffIdx)
{
    const int N_TAPS = 4;
    const uint16x8_t maxVal = vdupq_n_u16((1 << X265_DEPTH) - 1);
    uint16x4_t filter = vreinterpret_u16_s16(
        vabs_s16(vld1_s16(X265_NS::g_chromaFilter[coeffIdx])));

    uint16_t offset_u16;

    // A shim of 1 << (IF_FILTER_PREC - 1) enables us to use non-rounding
    // shifts - which are generally faster than rounding shifts on modern CPUs.
    if (coeff4)
    {
        // The outermost -2 is needed because we will divide the filter values by 4.
        offset_u16 = 1 << (IF_FILTER_PREC - 1 - 2);
    }
    else
    {
        offset_u16 = 1 << (IF_FILTER_PREC - 1);
    }

#if X265_DEPTH == 10
    if (!coeff4)
    {
        // All filter values are even, halve them to avoid needing to widen to
        // 32-bit elements in filter kernels.
        filter = vshr_n_u16(filter, 1);
        offset_u16 >>= 1;
    }

    const uint16x8_t offset = vdupq_n_u16(offset_u16);
#else
    const uint32x4_t offset = vdupq_n_u32(offset_u16);
#endif // X265_DEPTH == 10

    src -= (N_TAPS / 2 - 1) * srcStride;

    if (width % 8 != 0)
    {
        if (width == 12 || width == 6)
        {
            const int n_store = width == 12 ? 8 : 6;
            const uint16_t *s = src;
            uint16_t *d = dst;

            uint16x8_t in0[7];
            load_u16x8xn<3>(s, srcStride, in0);
            s += 3 * srcStride;

            for (int row = 0; row + 4 <= height; row += 4)
            {
                load_u16x8xn<4>(s, srcStride, in0 + 3);

                uint16x8_t res[4];
                filter4_u16x8<coeff4>(in0 + 0, filter, offset, maxVal, res[0]);
                filter4_u16x8<coeff4>(in0 + 1, filter, offset, maxVal, res[1]);
                filter4_u16x8<coeff4>(in0 + 2, filter, offset, maxVal, res[2]);
                filter4_u16x8<coeff4>(in0 + 3, filter, offset, maxVal, res[3]);

                store_u16xnxm<n_store, 4>(d, dstStride, res);

                in0[0] = in0[4];
                in0[1] = in0[5];
                in0[2] = in0[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (width == 6)
            {
                return;
            }

            src += 8;
            dst += 8;
        }

        const int n_store = width > 4 ? 4 : width;

        uint16x4_t in1[7];
        load_u16x4xn<3>(src, srcStride, in1);
        src += 3 * srcStride;

        for (int row = 0; row + 4 <= height; row += 4)
        {
            load_u16x4xn<4>(src, srcStride, in1 + 3);

            uint16x4_t res[4];
            filter4_u16x4<coeff4>(in1 + 0, filter, offset, vget_low_u16(maxVal), res[0]);
            filter4_u16x4<coeff4>(in1 + 1, filter, offset, vget_low_u16(maxVal), res[1]);
            filter4_u16x4<coeff4>(in1 + 2, filter, offset, vget_low_u16(maxVal), res[2]);
            filter4_u16x4<coeff4>(in1 + 3, filter, offset, vget_low_u16(maxVal), res[3]);

            store_u16xnxm<n_store, 4>(dst, dstStride, res);

            in1[0] = in1[4];
            in1[1] = in1[5];
            in1[2] = in1[6];

            src += 4 * srcStride;
            dst += 4 * dstStride;
        }

        if (height & 2)
        {
            load_u16x4xn<2>(src, srcStride, in1 + 3);

            uint16x4_t res[2];
            filter4_u16x4<coeff4>(in1 + 0, filter, offset, vget_low_u16(maxVal), res[0]);
            filter4_u16x4<coeff4>(in1 + 1, filter, offset, vget_low_u16(maxVal), res[1]);

            store_u16xnxm<n_store, 2>(dst, dstStride, res);
        }
    }
    else
    {
        for (int col = 0; col < width; col += 8)
        {
            const uint16_t *s = src;
            uint16_t *d = dst;

            uint16x8_t in[7];
            load_u16x8xn<3>(s, srcStride, in);
            s += 3 * srcStride;

            for (int row = 0; row + 4 <= height; row += 4)
            {
                load_u16x8xn<4>(s, srcStride, in + 3);

                uint16x8_t res[4];
                filter4_u16x8<coeff4>(in + 0, filter, offset, maxVal, res[0]);
                filter4_u16x8<coeff4>(in + 1, filter, offset, maxVal, res[1]);
                filter4_u16x8<coeff4>(in + 2, filter, offset, maxVal, res[2]);
                filter4_u16x8<coeff4>(in + 3, filter, offset, maxVal, res[3]);

                store_u16x8xn<4>(d, dstStride, res);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (height & 2)
            {
                load_u16x8xn<2>(s, srcStride, in + 3);

                uint16x8_t res[2];
                filter4_u16x8<coeff4>(in + 0, filter, offset, maxVal, res[0]);
                filter4_u16x8<coeff4>(in + 1, filter,  offset, maxVal, res[1]);

                store_u16x8xn<2>(d, dstStride, res);
            }

            src += 8;
            dst += 8;
        }
    }
}

template<int coeffIdx, int width, int height>
void inline interp8_vert_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst,
                                 intptr_t dstStride)
{
    const int N_TAPS = 8;
    const uint16x8_t maxVal = vdupq_n_u16((1 << X265_DEPTH) - 1);
    const uint16x8_t filter =
        vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(X265_NS::g_lumaFilter[coeffIdx])));

    src -= (N_TAPS / 2 - 1) * srcStride;

    if (width % 8 != 0)
    {
        const uint16_t *s = src;
        uint16_t *d = dst;

        if (width == 12)
        {
            uint16x8_t in[11];
            load_u16x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u16x8xn<4>(s, srcStride, in + 7);

                uint16x8_t res[4];
                filter8_u16x8<coeffIdx>(in + 0, res[0], filter, maxVal);
                filter8_u16x8<coeffIdx>(in + 1, res[1], filter, maxVal);
                filter8_u16x8<coeffIdx>(in + 2, res[2], filter, maxVal);
                filter8_u16x8<coeffIdx>(in + 3, res[3], filter, maxVal);

                store_u16x8xn<4>(d, dstStride, res);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            s = src + 8;
            d = dst + 8;
        }

        uint16x4_t in[11];
        load_u16x4xn<7>(s, srcStride, in);
        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            load_u16x4xn<4>(s, srcStride, in + 7);

            uint16x4_t res[4];
            filter8_u16x4<coeffIdx>(in + 0, res[0], filter, vget_low_u16(maxVal));
            filter8_u16x4<coeffIdx>(in + 1, res[1], filter, vget_low_u16(maxVal));
            filter8_u16x4<coeffIdx>(in + 2, res[2], filter, vget_low_u16(maxVal));
            filter8_u16x4<coeffIdx>(in + 3, res[3], filter, vget_low_u16(maxVal));

            store_u16x4xn<4>(d, dstStride, res);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];
            in[3] = in[7];
            in[4] = in[8];
            in[5] = in[9];
            in[6] = in[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }
    }
    else if (width % 16 != 0)
    {
        for (int col = 0; col < width; col += 8)
        {
            const uint16_t *s = src;
            uint16_t *d = dst;

            uint16x8_t in[11];
            load_u16x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u16x8xn<4>(s, srcStride, in + 7);

                uint16x8_t res[4];
                filter8_u16x8<coeffIdx>(in + 0, res[0], filter, maxVal);
                filter8_u16x8<coeffIdx>(in + 1, res[1], filter, maxVal);
                filter8_u16x8<coeffIdx>(in + 2, res[2], filter, maxVal);
                filter8_u16x8<coeffIdx>(in + 3, res[3], filter, maxVal);

                store_u16x8xn<4>(d, dstStride, res);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            src += 8;
            dst += 8;
        }
    }
    else
    {
        for (int col = 0; col < width; col += 16)
        {
            const uint16_t *s = src;
            uint16_t *d = dst;

            uint16x8_t in0[11], in1[11];
            load_u16x8xn<7>(s + 0, srcStride, in0);
            load_u16x8xn<7>(s + 8, srcStride, in1);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u16x8xn<4>(s + 0, srcStride, in0 + 7);
                load_u16x8xn<4>(s + 8, srcStride, in1 + 7);

                uint16x8_t res0[4], res1[4];
                filter8_u16x8<coeffIdx>(in0 + 0, res0[0], filter, maxVal);
                filter8_u16x8<coeffIdx>(in0 + 1, res0[1], filter, maxVal);
                filter8_u16x8<coeffIdx>(in0 + 2, res0[2], filter, maxVal);
                filter8_u16x8<coeffIdx>(in0 + 3, res0[3], filter, maxVal);

                filter8_u16x8<coeffIdx>(in1 + 0, res1[0], filter, maxVal);
                filter8_u16x8<coeffIdx>(in1 + 1, res1[1], filter, maxVal);
                filter8_u16x8<coeffIdx>(in1 + 2, res1[2], filter, maxVal);
                filter8_u16x8<coeffIdx>(in1 + 3, res1[3], filter, maxVal);

                store_u16x8xn<4>(d + 0, dstStride, res0);
                store_u16x8xn<4>(d + 8, dstStride, res1);

                in0[0] = in0[4];
                in0[1] = in0[5];
                in0[2] = in0[6];
                in0[3] = in0[7];
                in0[4] = in0[8];
                in0[5] = in0[9];
                in0[6] = in0[10];

                in1[0] = in1[4];
                in1[1] = in1[5];
                in1[2] = in1[6];
                in1[3] = in1[7];
                in1[4] = in1[8];
                in1[5] = in1[9];
                in1[6] = in1[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            src += 16;
            dst += 16;
        }
    }
}

template<bool coeff4, int width, int height>
void inline interp4_vert_ps_neon(const pixel *src, intptr_t srcStride, int16_t *dst,
                                 intptr_t dstStride, const int16_t coeffIdx)
{
    const int N_TAPS = 4;
    uint16x4_t filter = vreinterpret_u16_s16(
        vabs_s16(vld1_s16(X265_NS::g_chromaFilter[coeffIdx])));
    uint32_t offset_u32;

    if (coeff4)
    {
        // The -2 is needed because we will divide the filter values by 4.
        offset_u32 = (unsigned)-IF_INTERNAL_OFFS << (SHIFT_INTERP_PS - 2);
    }
    else
    {
        offset_u32 = (unsigned)-IF_INTERNAL_OFFS << SHIFT_INTERP_PS;
    }
#if X265_DEPTH == 10
    if (!coeff4)
    {
        // All filter values are even, halve them to avoid needing to widen to
        // 32-bit elements in filter kernels.
        filter = vshr_n_u16(filter, 1);
        offset_u32 >>= 1;
    }

    const uint16x8_t offset = vdupq_n_u16((uint16_t)offset_u32);
#else
    const uint32x4_t offset = vdupq_n_u32(offset_u32);
#endif // X265_DEPTH == 10

    src -= (N_TAPS / 2 - 1) * srcStride;

    if (width % 8 != 0)
    {
        if (width == 12 || width == 6)
        {
            const int n_store = width == 12 ? 8 : 6;
            const uint16_t *s = src;
            int16_t *d = dst;

            uint16x8_t in0[7];
            load_u16x8xn<3>(s, srcStride, in0);
            s += 3 * srcStride;

            for (int row = 0; row + 4 <= height; row += 4)
            {
                load_u16x8xn<4>(s, srcStride, in0 + 3);

                int16x8_t res[4];
                filter4_ps_u16x8<coeff4>(in0 + 0, filter, offset, res[0]);
                filter4_ps_u16x8<coeff4>(in0 + 1, filter, offset, res[1]);
                filter4_ps_u16x8<coeff4>(in0 + 2, filter, offset, res[2]);
                filter4_ps_u16x8<coeff4>(in0 + 3, filter, offset, res[3]);

                store_s16xnxm<n_store, 4>(res, d, dstStride);

                in0[0] = in0[4];
                in0[1] = in0[5];
                in0[2] = in0[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (width == 6)
            {
                return;
            }

            src += 8;
            dst += 8;
        }

        const int n_store = width > 4 ? 4 : width;

        uint16x4_t in1[7];
        load_u16x4xn<3>(src, srcStride, in1);
        src += 3 * srcStride;

        for (int row = 0; row + 4 <= height; row += 4)
        {
            load_u16x4xn<4>(src, srcStride, in1 + 3);

            int16x4_t res[4];
            filter4_ps_u16x4<coeff4>(in1 + 0, filter, offset, res[0]);
            filter4_ps_u16x4<coeff4>(in1 + 1, filter, offset, res[1]);
            filter4_ps_u16x4<coeff4>(in1 + 2, filter, offset, res[2]);
            filter4_ps_u16x4<coeff4>(in1 + 3, filter, offset, res[3]);

            store_s16xnxm<n_store, 4>(res, dst, dstStride);

            in1[0] = in1[4];
            in1[1] = in1[5];
            in1[2] = in1[6];

            src += 4 * srcStride;
            dst += 4 * dstStride;
        }

        if (height & 2)
        {
            load_u16x4xn<2>(src, srcStride, in1 + 3);

            int16x4_t res[2];
            filter4_ps_u16x4<coeff4>(in1 + 0, filter, offset, res[0]);
            filter4_ps_u16x4<coeff4>(in1 + 1, filter, offset, res[1]);

            store_s16xnxm<n_store, 2>(res, dst, dstStride);
        }
    }
    else
    {
        for (int col = 0; col < width; col += 8)
        {
            const uint16_t *s = src;
            int16_t *d = dst;

            uint16x8_t in[7];
            load_u16x8xn<3>(s, srcStride, in);
            s += 3 * srcStride;

            for (int row = 0; row + 4 <= height; row += 4)
            {
                load_u16x8xn<4>(s, srcStride, in + 3);

                int16x8_t res[4];
                filter4_ps_u16x8<coeff4>(in + 0, filter, offset, res[0]);
                filter4_ps_u16x8<coeff4>(in + 1, filter, offset, res[1]);
                filter4_ps_u16x8<coeff4>(in + 2, filter, offset, res[2]);
                filter4_ps_u16x8<coeff4>(in + 3, filter, offset, res[3]);

                store_s16x8xn<4>(d, dstStride, res);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (height & 2)
            {
                load_u16x8xn<2>(s, srcStride, in + 3);

                int16x8_t res[2];
                filter4_ps_u16x8<coeff4>(in + 0, filter, offset, res[0]);
                filter4_ps_u16x8<coeff4>(in + 1, filter, offset, res[1]);

                store_s16x8xn<2>(d, dstStride, res);
            }

            src += 8;
            dst += 8;
        }
    }
}

template<int coeffIdx, int width, int height>
void inline interp8_vert_ps_neon(const pixel *src, intptr_t srcStride, int16_t *dst,
                                 intptr_t dstStride)
{
    const int N_TAPS = 8;
    const uint16x8_t filter =
        vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(X265_NS::g_lumaFilter[coeffIdx])));
    uint32x4_t offset =
        vdupq_n_u32((unsigned)-IF_INTERNAL_OFFS << SHIFT_INTERP_PS);

    src -= (N_TAPS / 2 - 1) * srcStride;

    if (width % 8 != 0)
    {
        const uint16_t *s = src;
        int16_t *d = dst;

        if (width == 12)
        {
            uint16x8_t in[11];
            load_u16x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u16x8xn<4>(s, srcStride, in + 7);

                int16x8_t res[4];
                filter8_ps_u16x8<coeffIdx>(in + 0, res[0], offset, filter);
                filter8_ps_u16x8<coeffIdx>(in + 1, res[1], offset, filter);
                filter8_ps_u16x8<coeffIdx>(in + 2, res[2], offset, filter);
                filter8_ps_u16x8<coeffIdx>(in + 3, res[3], offset, filter);

                store_s16x8xn<4>(d, dstStride, res);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            s = src + 8;
            d = dst + 8;
        }

        uint16x4_t in[11];
        load_u16x4xn<7>(s, srcStride, in);
        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            load_u16x4xn<4>(s, srcStride, in + 7);

            int16x4_t res[4];
            filter8_ps_u16x4<coeffIdx>(in + 0, res[0], offset, filter);
            filter8_ps_u16x4<coeffIdx>(in + 1, res[1], offset, filter);
            filter8_ps_u16x4<coeffIdx>(in + 2, res[2], offset, filter);
            filter8_ps_u16x4<coeffIdx>(in + 3, res[3], offset, filter);

            store_s16x4xn<4>(d, dstStride, res);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];
            in[3] = in[7];
            in[4] = in[8];
            in[5] = in[9];
            in[6] = in[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }
    }
    else if (width % 16 != 0)
    {
        for (int col = 0; col < width; col += 8)
        {
            const uint16_t *s = src;
            int16_t *d = dst;

            uint16x8_t in[11];
            load_u16x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u16x8xn<4>(s, srcStride, in + 7);

                int16x8_t res[4];
                filter8_ps_u16x8<coeffIdx>(in + 0, res[0], offset, filter);
                filter8_ps_u16x8<coeffIdx>(in + 1, res[1], offset, filter);
                filter8_ps_u16x8<coeffIdx>(in + 2, res[2], offset, filter);
                filter8_ps_u16x8<coeffIdx>(in + 3, res[3], offset, filter);

                store_s16x8xn<4>(d, dstStride, res);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            src += 8;
            dst += 8;
        }
    }
    else
    {
        for (int col = 0; col < width; col += 16)
        {
            const uint16_t *s = src;
            int16_t *d = dst;

            uint16x8_t in0[11], in1[11];
            load_u16x8xn<7>(s + 0, srcStride, in0);
            load_u16x8xn<7>(s + 8, srcStride, in1);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_u16x8xn<4>(s + 0, srcStride, in0 + 7);
                load_u16x8xn<4>(s + 8, srcStride, in1 + 7);

                int16x8_t res0[4], res1[4];
                filter8_ps_u16x8<coeffIdx>(in0 + 0, res0[0], offset, filter);
                filter8_ps_u16x8<coeffIdx>(in0 + 1, res0[1], offset, filter);
                filter8_ps_u16x8<coeffIdx>(in0 + 2, res0[2], offset, filter);
                filter8_ps_u16x8<coeffIdx>(in0 + 3, res0[3], offset, filter);

                filter8_ps_u16x8<coeffIdx>(in1 + 0, res1[0], offset, filter);
                filter8_ps_u16x8<coeffIdx>(in1 + 1, res1[1], offset, filter);
                filter8_ps_u16x8<coeffIdx>(in1 + 2, res1[2], offset, filter);
                filter8_ps_u16x8<coeffIdx>(in1 + 3, res1[3], offset, filter);

                store_s16x8xn<4>(d + 0, dstStride, res0);
                store_s16x8xn<4>(d + 8, dstStride, res1);

                in0[0] = in0[4];
                in0[1] = in0[5];
                in0[2] = in0[6];
                in0[3] = in0[7];
                in0[4] = in0[8];
                in0[5] = in0[9];
                in0[6] = in0[10];

                in1[0] = in1[4];
                in1[1] = in1[5];
                in1[2] = in1[6];
                in1[3] = in1[7];
                in1[4] = in1[8];
                in1[5] = in1[9];
                in1[6] = in1[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            src += 16;
            dst += 16;
        }
    }
}

template<bool coeff4>
void inline filter4_sp_s16x4(const int16x4_t *s, const int16x4_t f,
                             const int32x4_t offset, const uint16x4_t maxVal,
                             uint16x4_t &d)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        // Filter values are divisible by 4, factor that out in order to only
        // need a multiplication by 9 and a subtraction (which is a
        // multiplication by -1).
        int16x4_t sum03 = vadd_s16(s[0], s[3]);
        int16x4_t sum12 = vadd_s16(s[1], s[2]);

        int32x4_t sum = vmlal_n_s16(offset, sum12, 9);
        sum = vsubw_s16(sum, sum03);

        // We divided filter values by 4 so -2 from right shift.
        d = vqshrun_n_s32(sum, IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH - 2);
        d = vmin_u16(d, maxVal);
    }
    else
    {
        int32x4_t sum = vmlal_lane_s16(offset, s[0], f, 0);
        sum = vmlal_lane_s16(sum, s[1], f, 1);
        sum = vmlal_lane_s16(sum, s[2], f, 2);
        sum = vmlal_lane_s16(sum, s[3], f, 3);

        d = vqshrun_n_s32(sum, IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH);
        d = vmin_u16(d, maxVal);
    }
}

template<bool coeff4>
void inline filter4_sp_s16x8(const int16x8_t *s, const int16x4_t f,
                             const int32x4_t offset, const uint16x8_t maxVal,
                             uint16x8_t &d)
{
    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        // Filter values are divisible by 4, factor that out in order to only
        // need a multiplication by 9 and a subtraction (which is a
        // multiplication by -1).
        int16x8_t sum03 = vaddq_s16(s[0], s[3]);
        int16x8_t sum12 = vaddq_s16(s[1], s[2]);

        int32x4_t sum_lo = vmlal_n_s16(offset, vget_low_s16(sum12), 9);
        int32x4_t sum_hi = vmlal_n_s16(offset, vget_high_s16(sum12), 9);
        sum_lo = vsubw_s16(sum_lo, vget_low_s16(sum03));
        sum_hi = vsubw_s16(sum_hi, vget_high_s16(sum03));

        // We divided filter values by 4 so -2 from right shift.
        uint16x4_t d0 = vqshrun_n_s32(sum_lo,
                                      IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH - 2);
        uint16x4_t d1 = vqshrun_n_s32(sum_hi,
                                      IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH - 2);
        d = vminq_u16(vcombine_u16(d0, d1), maxVal);
    }
    else
    {
        int32x4_t sum_lo = vmlal_lane_s16(offset, vget_low_s16(s[0]), f, 0);
        sum_lo = vmlal_lane_s16(sum_lo, vget_low_s16(s[1]), f, 1);
        sum_lo = vmlal_lane_s16(sum_lo, vget_low_s16(s[2]), f, 2);
        sum_lo = vmlal_lane_s16(sum_lo, vget_low_s16(s[3]), f, 3);

        int32x4_t sum_hi = vmlal_lane_s16(offset, vget_high_s16(s[0]), f, 0);
        sum_hi = vmlal_lane_s16(sum_hi, vget_high_s16(s[1]), f, 1);
        sum_hi = vmlal_lane_s16(sum_hi, vget_high_s16(s[2]), f, 2);
        sum_hi = vmlal_lane_s16(sum_hi, vget_high_s16(s[3]), f, 3);

        uint16x4_t d0 = vqshrun_n_s32(sum_lo,
                                      IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH);
        uint16x4_t d1 = vqshrun_n_s32(sum_hi,
                                      IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH);
        d = vminq_u16(vcombine_u16(d0, d1), maxVal);
    }
}

template<bool coeff4, int width, int height>
void inline interp4_vert_sp_neon(const int16_t *src, intptr_t srcStride, pixel *dst,
                                 intptr_t dstStride, const int16_t coeffIdx)
{
    const int N_TAPS = 4;
    const int shift = IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH;
    const uint16x8_t maxVal = vdupq_n_u16((1 << X265_DEPTH) - 1);
    int16x4_t filter = vld1_s16(X265_NS::g_chromaFilter[coeffIdx]);
    int32x4_t offset;

    if (coeff4)
    {
        // The right shift by 2 is needed because we will divide the filter values by 4.
        offset = vdupq_n_s32(((1 << (shift - 1)) +
                              (IF_INTERNAL_OFFS << IF_FILTER_PREC)) >> 2);
    }
    else
    {
        offset = vdupq_n_s32((1 << (shift - 1)) +
                             (IF_INTERNAL_OFFS << IF_FILTER_PREC));
    }

    src -= (N_TAPS / 2 - 1) * srcStride;

    if (width % 8 != 0)
    {
        if (width == 12 || width == 6)
        {
            const int n_store = width == 12 ? 8 : 6;
            const int16_t *s = src;
            uint16_t *d = dst;

            int16x8_t in[7];
            load_s16x8xn<3>(s, srcStride, in);
            s += 3 * srcStride;

            for (int row = 0; row + 4 <= height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 3);

                uint16x8_t res[4];
                filter4_sp_s16x8<coeff4>(in + 0, filter, offset, maxVal, res[0]);
                filter4_sp_s16x8<coeff4>(in + 1, filter, offset, maxVal, res[1]);
                filter4_sp_s16x8<coeff4>(in + 2, filter, offset, maxVal, res[2]);
                filter4_sp_s16x8<coeff4>(in + 3, filter, offset, maxVal, res[3]);

                store_u16xnxm<n_store, 4>(d, dstStride, res);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (width == 6)
            {
                return;
            }

            src += 8;
            dst += 8;
        }
        const int n_store = width > 4 ? 4 : width;

        int16x4_t in[7];
        load_s16x4xn<3>(src, srcStride, in);
        src += 3 * srcStride;

        for (int row = 0; row + 4 <= height; row += 4)
        {
            load_s16x4xn<4>(src, srcStride, in + 3);

            uint16x4_t res[4];
            filter4_sp_s16x4<coeff4>(in + 0, filter, offset,
                                     vget_low_u16(maxVal), res[0]);
            filter4_sp_s16x4<coeff4>(in + 1, filter, offset,
                                     vget_low_u16(maxVal), res[1]);
            filter4_sp_s16x4<coeff4>(in + 2, filter, offset,
                                     vget_low_u16(maxVal), res[2]);
            filter4_sp_s16x4<coeff4>(in + 3, filter, offset,
                                     vget_low_u16(maxVal), res[3]);

            store_u16xnxm<n_store, 4>(dst, dstStride, res);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];

            src += 4 * srcStride;
            dst += 4 * dstStride;
        }

        if (height & 2)
        {
            load_s16x4xn<2>(src, srcStride, in + 3);

            uint16x4_t res[2];
            filter4_sp_s16x4<coeff4>(in + 0, filter, offset,
                                     vget_low_u16(maxVal), res[0]);
            filter4_sp_s16x4<coeff4>(in + 1, filter, offset,
                                     vget_low_u16(maxVal), res[1]);

            store_u16xnxm<n_store, 2>(dst, dstStride, res);
        }
    }
    else
    {
        for (int col = 0; col < width; col += 8)
        {
            const int16_t *s = src;
            uint16_t *d = dst;

            int16x8_t in[7];
            load_s16x8xn<3>(s, srcStride, in);
            s += 3 * srcStride;

            for (int row = 0; row + 4 <= height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 3);

                uint16x8_t res[4];
                filter4_sp_s16x8<coeff4>(in + 0, filter, offset, maxVal, res[0]);
                filter4_sp_s16x8<coeff4>(in + 1, filter, offset, maxVal, res[1]);
                filter4_sp_s16x8<coeff4>(in + 2, filter, offset, maxVal, res[2]);
                filter4_sp_s16x8<coeff4>(in + 3, filter, offset, maxVal, res[3]);

                store_u16x8xn<4>(d, dstStride, res);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (height & 2)
            {
                load_s16x8xn<2>(s, srcStride, in + 3);

                uint16x8_t res[2];
                filter4_sp_s16x8<coeff4>(in + 0, filter, offset, maxVal, res[0]);
                filter4_sp_s16x8<coeff4>(in + 1, filter, offset, maxVal, res[1]);

                store_u16x8xn<2>(d, dstStride, res);
            }

            src += 8;
            dst += 8;
        }
    }
}

template<int coeffIdx>
void inline filter8_sp_s16x4(const int16x4_t *s, uint16x4_t &d, int32x4_t offset,
                             int16x8_t filter, uint16x4_t maxVal)
{
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        int16x4_t sum06 = vsub_s16(s[6], s[0]);

        int32x4_t sum12345 = vmlal_laneq_s16(offset, s[1], filter, 1);
        sum12345 = vmlal_laneq_s16(sum12345, s[2], filter, 2);
        sum12345 = vmlal_laneq_s16(sum12345, s[3], filter, 3);
        sum12345 = vmlal_laneq_s16(sum12345, s[4], filter, 4);
        sum12345 = vmlal_laneq_s16(sum12345, s[5], filter, 5);

        int32x4_t sum = vaddw_s16(sum12345, sum06);

        d = vqshrun_n_s32(sum, IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH);
        d = vmin_u16(d, maxVal);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        int16x4_t sum07 = vadd_s16(s[0], s[7]);
        int16x4_t sum16 = vadd_s16(s[1], s[6]);
        int16x4_t sum25 = vadd_s16(s[2], s[5]);
        int16x4_t sum34 = vadd_s16(s[3], s[4]);

        int32x4_t sum12356 =  vmlal_laneq_s16(offset, sum16, filter, 1);
        sum12356 = vmlal_laneq_s16(sum12356, sum25, filter, 2);
        sum12356 = vmlal_laneq_s16(sum12356, sum34, filter, 3);

        int32x4_t sum = vsubw_s16(sum12356, sum07);

        d = vqshrun_n_s32(sum, IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH);
        d = vmin_u16(d, maxVal);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        int16x4_t sum17 = vsub_s16(s[1], s[7]);

        int32x4_t sum23456 = vmlal_laneq_s16(offset, s[2], filter, 2);
        sum23456 = vmlal_laneq_s16(sum23456, s[3], filter, 3);
        sum23456 = vmlal_laneq_s16(sum23456, s[4], filter, 4);
        sum23456 = vmlal_laneq_s16(sum23456, s[5], filter, 5);
        sum23456 = vmlal_laneq_s16(sum23456, s[6], filter, 6);

        int32x4_t sum = vaddw_s16(sum23456, sum17);

        d = vqshrun_n_s32(sum, IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH);
        d = vmin_u16(d, maxVal);
    }
}

template<int coeffIdx>
void inline filter8_sp_s16x8(const int16x8_t *s, uint16x8_t &d, int32x4_t offset,
                             int16x8_t filter, uint16x8_t maxVal)
{
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        int16x8_t sum06 =  vsubq_s16(s[6], s[0]);

        int32x4_t sum12345_lo = vmlal_laneq_s16(offset, vget_low_s16(s[1]), filter, 1);
        sum12345_lo = vmlal_laneq_s16(sum12345_lo, vget_low_s16(s[2]), filter, 2);
        sum12345_lo = vmlal_laneq_s16(sum12345_lo, vget_low_s16(s[3]), filter, 3);
        sum12345_lo = vmlal_laneq_s16(sum12345_lo, vget_low_s16(s[4]), filter, 4);
        sum12345_lo = vmlal_laneq_s16(sum12345_lo, vget_low_s16(s[5]), filter, 5);

        int32x4_t sum12345_hi = vmlal_laneq_s16(offset, vget_high_s16(s[1]), filter, 1);
        sum12345_hi = vmlal_laneq_s16(sum12345_hi, vget_high_s16(s[2]), filter, 2);
        sum12345_hi = vmlal_laneq_s16(sum12345_hi, vget_high_s16(s[3]), filter, 3);
        sum12345_hi = vmlal_laneq_s16(sum12345_hi, vget_high_s16(s[4]), filter, 4);
        sum12345_hi = vmlal_laneq_s16(sum12345_hi, vget_high_s16(s[5]), filter, 5);

        int32x4_t sum_lo = vaddw_s16(sum12345_lo, vget_low_s16(sum06));
        int32x4_t sum_hi = vaddw_s16(sum12345_hi, vget_high_s16(sum06));

        uint16x4_t d_lo = vqshrun_n_s32(sum_lo,
                                        IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH);
        uint16x4_t d_hi = vqshrun_n_s32(sum_hi,
                                        IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH);

        d = vminq_u16(vcombine_u16(d_lo, d_hi), maxVal);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        int16x8_t sum07 = vaddq_s16(s[0], s[7]);
        int16x8_t sum16 = vaddq_s16(s[1], s[6]);
        int16x8_t sum25 = vaddq_s16(s[2], s[5]);
        int16x8_t sum34 = vaddq_s16(s[3], s[4]);

        int32x4_t sum123456_lo = vmlal_laneq_s16(offset, vget_low_s16(sum16), filter, 1);
        sum123456_lo = vmlal_laneq_s16(sum123456_lo, vget_low_s16(sum25), filter, 2);
        sum123456_lo = vmlal_laneq_s16(sum123456_lo, vget_low_s16(sum34), filter, 3);

        int32x4_t sum123456_hi = vmlal_laneq_s16(offset, vget_high_s16(sum16), filter, 1);
        sum123456_hi = vmlal_laneq_s16(sum123456_hi, vget_high_s16(sum25), filter, 2);
        sum123456_hi = vmlal_laneq_s16(sum123456_hi, vget_high_s16(sum34), filter, 3);

        int32x4_t sum_lo = vsubw_s16(sum123456_lo, vget_low_s16(sum07));
        int32x4_t sum_hi = vsubw_s16(sum123456_hi, vget_high_s16(sum07));

        uint16x4_t d_lo = vqshrun_n_s32(sum_lo,
                                        IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH);
        uint16x4_t d_hi = vqshrun_n_s32(sum_hi,
                                        IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH);

        d = vminq_u16(vcombine_u16(d_lo, d_hi), maxVal);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        int16x8_t sum17 = vsubq_s16(s[1], s[7]);

        int32x4_t sum23456_lo = vmlal_laneq_s16(offset, vget_low_s16(s[2]), filter, 2);
        sum23456_lo = vmlal_laneq_s16(sum23456_lo, vget_low_s16(s[3]), filter, 3);
        sum23456_lo = vmlal_laneq_s16(sum23456_lo, vget_low_s16(s[4]), filter, 4);
        sum23456_lo = vmlal_laneq_s16(sum23456_lo, vget_low_s16(s[5]), filter, 5);
        sum23456_lo = vmlal_laneq_s16(sum23456_lo, vget_low_s16(s[6]), filter, 6);

        int32x4_t sum23456_hi = vmlal_laneq_s16(offset, vget_high_s16(s[2]), filter, 2);
        sum23456_hi = vmlal_laneq_s16(sum23456_hi, vget_high_s16(s[3]), filter, 3);
        sum23456_hi = vmlal_laneq_s16(sum23456_hi, vget_high_s16(s[4]), filter, 4);
        sum23456_hi = vmlal_laneq_s16(sum23456_hi, vget_high_s16(s[5]), filter, 5);
        sum23456_hi = vmlal_laneq_s16(sum23456_hi, vget_high_s16(s[6]), filter, 6);

        int32x4_t sum_lo = vaddw_s16(sum23456_lo, vget_low_s16(sum17));
        int32x4_t sum_hi = vaddw_s16(sum23456_hi, vget_high_s16(sum17));

        uint16x4_t d_lo = vqshrun_n_s32(sum_lo,
                                        IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH);
        uint16x4_t d_hi = vqshrun_n_s32(sum_hi,
                                        IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH);

        d = vminq_u16(vcombine_u16(d_lo, d_hi), maxVal);
    }
}

template<int coeffIdx, int width, int height>
void inline interp8_vert_sp_neon(const int16_t *src, intptr_t srcStride, pixel *dst,
                                 intptr_t dstStride)
{
    const int N_TAPS = 8;
    int shift = IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH;
    const uint16x8_t maxVal = vdupq_n_u16((1 << X265_DEPTH) - 1);
    const int16x8_t filter = vld1q_s16(X265_NS::g_lumaFilter[coeffIdx]);
    const int32x4_t offset = vdupq_n_s32((1 << (shift - 1)) +
                                         (IF_INTERNAL_OFFS << IF_FILTER_PREC));

    src -= (N_TAPS / 2 - 1) * srcStride;

    if (width % 8 != 0)
    {
        const int16_t *s = src;
        uint16_t *d = dst;

        if (width == 12)
        {
            int16x8_t in[11];
            load_s16x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 7);

                uint16x8_t res[4];
                filter8_sp_s16x8<coeffIdx>(in + 0, res[0], offset, filter, maxVal);
                filter8_sp_s16x8<coeffIdx>(in + 1, res[1], offset, filter, maxVal);
                filter8_sp_s16x8<coeffIdx>(in + 2, res[2], offset, filter, maxVal);
                filter8_sp_s16x8<coeffIdx>(in + 3, res[3], offset, filter, maxVal);

                store_u16x8xn<4>(d, dstStride, res);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            s = src + 8;
            d = dst + 8;
        }

        int16x4_t in[11];
        load_s16x4xn<7>(s, srcStride, in);
        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            load_s16x4xn<4>(s, srcStride, in + 7);

            uint16x4_t res[4];
            filter8_sp_s16x4<coeffIdx>(in + 0, res[0], offset, filter,
                                       vget_low_u16(maxVal));
            filter8_sp_s16x4<coeffIdx>(in + 1, res[1], offset, filter,
                                       vget_low_u16(maxVal));
            filter8_sp_s16x4<coeffIdx>(in + 2, res[2], offset, filter,
                                       vget_low_u16(maxVal));
            filter8_sp_s16x4<coeffIdx>(in + 3, res[3], offset, filter,
                                       vget_low_u16(maxVal));

            store_u16x4xn<4>(d, dstStride, res);

            in[0] = in[4];
            in[1] = in[5];
            in[2] = in[6];
            in[3] = in[7];
            in[4] = in[8];
            in[5] = in[9];
            in[6] = in[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }
    }
    else if (width % 16 != 0)
    {
        const int16_t *s2 = src;
        uint16_t *d2 = dst;
        for (int col = 0; col < width; col += 8)
        {
            const int16_t *s = s2;
            uint16_t *d = d2;

            int16x8_t in[11];
            load_s16x8xn<7>(s, srcStride, in);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 7);

                uint16x8_t res[4];
                filter8_sp_s16x8<coeffIdx>(in + 0, res[0], offset, filter, maxVal);
                filter8_sp_s16x8<coeffIdx>(in + 1, res[1], offset, filter, maxVal);
                filter8_sp_s16x8<coeffIdx>(in + 2, res[2], offset, filter, maxVal);
                filter8_sp_s16x8<coeffIdx>(in + 3, res[3], offset, filter, maxVal);

                store_u16x8xn<4>(d, dstStride, res);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];
                in[3] = in[7];
                in[4] = in[8];
                in[5] = in[9];
                in[6] = in[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            s2 += 8;
            d2 += 8;
        }
    }
    else
    {
        for (int col = 0; col < width; col += 16)
        {
            const int16_t *s = src;
            uint16_t *d = dst;

            int16x8_t in0[11], in1[11];
            load_s16x8xn<7>(s + 0, srcStride, in0);
            load_s16x8xn<7>(s + 8, srcStride, in1);
            s += 7 * srcStride;

            for (int row = 0; row < height; row += 4)
            {
                load_s16x8xn<4>(s + 0, srcStride, in0 + 7);
                load_s16x8xn<4>(s + 8, srcStride, in1 + 7);

                uint16x8_t res0[4], res1[4];
                filter8_sp_s16x8<coeffIdx>(in0 + 0, res0[0], offset, filter, maxVal);
                filter8_sp_s16x8<coeffIdx>(in0 + 1, res0[1], offset, filter, maxVal);
                filter8_sp_s16x8<coeffIdx>(in0 + 2, res0[2], offset, filter, maxVal);
                filter8_sp_s16x8<coeffIdx>(in0 + 3, res0[3], offset, filter, maxVal);

                filter8_sp_s16x8<coeffIdx>(in1 + 0, res1[0], offset, filter, maxVal);
                filter8_sp_s16x8<coeffIdx>(in1 + 1, res1[1], offset, filter, maxVal);
                filter8_sp_s16x8<coeffIdx>(in1 + 2, res1[2], offset, filter, maxVal);
                filter8_sp_s16x8<coeffIdx>(in1 + 3, res1[3], offset, filter, maxVal);

                store_u16x8xn<4>(d + 0, dstStride, res0);
                store_u16x8xn<4>(d + 8, dstStride, res1);

                in0[0] = in0[4];
                in0[1] = in0[5];
                in0[2] = in0[6];
                in0[3] = in0[7];
                in0[4] = in0[8];
                in0[5] = in0[9];
                in0[6] = in0[10];

                in1[0] = in1[4];
                in1[1] = in1[5];
                in1[2] = in1[6];
                in1[3] = in1[7];
                in1[4] = in1[8];
                in1[5] = in1[9];
                in1[6] = in1[10];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            src += 16;
            dst += 16;
        }
    }
}

template<int coeffIdx, int coeffIdy, int width, int height>
void interp8_hv_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst, intptr_t dstStride)
{
    const int N_TAPS = 8;
    const uint16x8_t h_filter =
        vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(X265_NS::g_lumaFilter[coeffIdx])));
    const int16x8_t v_filter = vld1q_s16(X265_NS::g_lumaFilter[coeffIdy]);
    const uint32x4_t h_offset =
        vdupq_n_u32((unsigned)-IF_INTERNAL_OFFS << SHIFT_INTERP_PS);
    int shift = IF_FILTER_PREC + IF_INTERNAL_PREC - X265_DEPTH;
    const int32x4_t v_offset = vdupq_n_s32((1 << (shift - 1)) +
                                           (IF_INTERNAL_OFFS << IF_FILTER_PREC));
    const uint16x8_t maxVal = vdupq_n_u16((1 << X265_DEPTH) - 1);

    src -= (N_TAPS / 2 - 1) * srcStride + (N_TAPS / 2 - 1);

    for (int col = 0; col + 8 <= width; col += 8)
    {
        const pixel *s = src;
        pixel *d = dst;

        uint16x8_t h_s[N_TAPS];
        int16x8_t v_s[16];

        load_u16x8xn<8>(s + 0 * srcStride, 1, h_s);
        filter8_ps_u16x8<coeffIdx>(h_s, v_s[0], h_offset, h_filter);

        load_u16x8xn<8>(s + 1 * srcStride, 1, h_s);
        filter8_ps_u16x8<coeffIdx>(h_s, v_s[1], h_offset, h_filter);

        load_u16x8xn<8>(s + 2 * srcStride, 1, h_s);
        filter8_ps_u16x8<coeffIdx>(h_s, v_s[2], h_offset, h_filter);

        load_u16x8xn<8>(s + 3 * srcStride, 1, h_s);
        filter8_ps_u16x8<coeffIdx>(h_s, v_s[3], h_offset, h_filter);

        load_u16x8xn<8>(s + 4 * srcStride, 1, h_s);
        filter8_ps_u16x8<coeffIdx>(h_s, v_s[4], h_offset, h_filter);

        load_u16x8xn<8>(s + 5 * srcStride, 1, h_s);
        filter8_ps_u16x8<coeffIdx>(h_s, v_s[5], h_offset, h_filter);

        load_u16x8xn<8>(s + 6 * srcStride, 1, h_s);
        filter8_ps_u16x8<coeffIdx>(h_s, v_s[6], h_offset, h_filter);

        s += 7 * srcStride;

        int row = 0;
        if (coeffIdy == 1)
        {
            for (; row + 8 <= height; row += 8)
            {
                uint16x8_t res[8];

                load_u16x8xn<8>(s + 0 * srcStride, 1, h_s);
                filter8_ps_u16x8<coeffIdx>(h_s, v_s[7], h_offset, h_filter);
                filter8_sp_s16x8<coeffIdy>(v_s + 0, res[0], v_offset, v_filter, maxVal);

                load_u16x8xn<8>(s + 1 * srcStride, 1, h_s);
                filter8_ps_u16x8<coeffIdx>(h_s, v_s[8], h_offset, h_filter);
                filter8_sp_s16x8<coeffIdy>(v_s + 1, res[1], v_offset, v_filter, maxVal);
                v_s[0] = v_s[8];

                load_u16x8xn<8>(s + 2 * srcStride, 1, h_s);
                filter8_ps_u16x8<coeffIdx>(h_s, v_s[9], h_offset, h_filter);
                filter8_sp_s16x8<coeffIdy>(v_s + 2, res[2], v_offset, v_filter, maxVal);
                v_s[1] = v_s[9];

                load_u16x8xn<8>(s + 3 * srcStride, 1, h_s);
                filter8_ps_u16x8<coeffIdx>(h_s, v_s[10], h_offset, h_filter);
                filter8_sp_s16x8<coeffIdy>(v_s + 3, res[3], v_offset, v_filter, maxVal);
                v_s[2] = v_s[10];

                load_u16x8xn<8>(s + 4 * srcStride, 1, h_s);
                filter8_ps_u16x8<coeffIdx>(h_s, v_s[11], h_offset, h_filter);
                filter8_sp_s16x8<coeffIdy>(v_s + 4, res[4], v_offset, v_filter, maxVal);
                v_s[3] = v_s[11];

                load_u16x8xn<8>(s + 5 * srcStride, 1, h_s);
                filter8_ps_u16x8<coeffIdx>(h_s, v_s[12], h_offset, h_filter);
                filter8_sp_s16x8<coeffIdy>(v_s + 5, res[5], v_offset, v_filter, maxVal);
                v_s[4] = v_s[12];

                load_u16x8xn<8>(s + 6 * srcStride, 1, h_s);
                filter8_ps_u16x8<coeffIdx>(h_s, v_s[13], h_offset, h_filter);
                filter8_sp_s16x8<coeffIdy>(v_s + 6, res[6], v_offset, v_filter, maxVal);
                v_s[5] = v_s[13];

                load_u16x8xn<8>(s + 7 * srcStride, 1, h_s);
                filter8_ps_u16x8<coeffIdx>(h_s, v_s[14], h_offset, h_filter);
                filter8_sp_s16x8<coeffIdy>(v_s + 7, res[7], v_offset, v_filter, maxVal);
                v_s[6] = v_s[14];

                store_u16xnxm<8, 8>(d, dstStride, res);

                s += 8 * srcStride;
                d += 8 * dstStride;
            }
        }

        for (; row < height; row += 4)
        {
            uint16x8_t res[4];

            load_u16x8xn<8>(s + 0 * srcStride, 1, h_s);
            filter8_ps_u16x8<coeffIdx>(h_s, v_s[7], h_offset, h_filter);
            filter8_sp_s16x8<coeffIdy>(v_s + 0, res[0], v_offset, v_filter, maxVal);
            v_s[0] = v_s[4];

            load_u16x8xn<8>(s + 1 * srcStride, 1, h_s);
            filter8_ps_u16x8<coeffIdx>(h_s, v_s[8], h_offset, h_filter);
            filter8_sp_s16x8<coeffIdy>(v_s + 1, res[1], v_offset, v_filter, maxVal);
            v_s[1] = v_s[5];

            load_u16x8xn<8>(s + 2 * srcStride, 1, h_s);
            filter8_ps_u16x8<coeffIdx>(h_s, v_s[9], h_offset, h_filter);
            filter8_sp_s16x8<coeffIdy>(v_s + 2, res[2], v_offset, v_filter, maxVal);
            v_s[2] = v_s[6];

            load_u16x8xn<8>(s + 3 * srcStride, 1, h_s);
            filter8_ps_u16x8<coeffIdx>(h_s, v_s[10], h_offset, h_filter);
            filter8_sp_s16x8<coeffIdy>(v_s + 3, res[3], v_offset, v_filter, maxVal);
            v_s[3] = v_s[7];

            store_u16xnxm<8, 4>(d, dstStride, res);

            v_s[4] = v_s[8];
            v_s[5] = v_s[9];
            v_s[6] = v_s[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        src += 8;
        dst += 8;
    }

    if (width % 8 != 0)
    {
        uint16x4_t h_s0[N_TAPS], h_s1[N_TAPS];
        int16x4_t v_s[16];

        load_u16x4xn<8>(src + 0 * srcStride, 1, h_s0);
        filter8_ps_u16x4<coeffIdx>(h_s0, v_s[0], h_offset, h_filter);

        load_u16x4xn<8>(src + 1 * srcStride, 1, h_s1);
        load_u16x4xn<8>(src + 2 * srcStride, 1, h_s0);

        filter8_ps_u16x4<coeffIdx>(h_s1, v_s[1], h_offset, h_filter);
        filter8_ps_u16x4<coeffIdx>(h_s0, v_s[2], h_offset, h_filter);

        load_u16x4xn<8>(src + 3 * srcStride, 1, h_s1);
        load_u16x4xn<8>(src + 4 * srcStride, 1, h_s0);

        filter8_ps_u16x4<coeffIdx>(h_s1, v_s[3], h_offset, h_filter);
        filter8_ps_u16x4<coeffIdx>(h_s0, v_s[4], h_offset, h_filter);

        load_u16x4xn<8>(src + 5 * srcStride, 1, h_s1);
        load_u16x4xn<8>(src + 6 * srcStride, 1, h_s0);

        filter8_ps_u16x4<coeffIdx>(h_s1, v_s[5], h_offset, h_filter);
        filter8_ps_u16x4<coeffIdx>(h_s0, v_s[6], h_offset, h_filter);

        src += 7 * srcStride;

        int row = 0;
        for (; row + 8 <= height; row += 8)
        {
            uint16x4_t res[8];

            load_u16x4xn<8>(src + 0 * srcStride, 1, h_s1);
            load_u16x4xn<8>(src + 1 * srcStride, 1, h_s0);

            filter8_ps_u16x4<coeffIdx>(h_s1, v_s[7], h_offset, h_filter);
            filter8_sp_s16x4<coeffIdy>(v_s + 0, res[0], v_offset, v_filter,
                                       vget_low_u16(maxVal));

            filter8_ps_u16x4<coeffIdx>(h_s0, v_s[8], h_offset, h_filter);
            filter8_sp_s16x4<coeffIdy>(v_s + 1, res[1], v_offset, v_filter,
                                       vget_low_u16(maxVal));
            v_s[0] = v_s[8];

            load_u16x4xn<8>(src + 2 * srcStride, 1, h_s1);
            load_u16x4xn<8>(src + 3 * srcStride, 1, h_s0);

            filter8_ps_u16x4<coeffIdx>(h_s1, v_s[9], h_offset, h_filter);
            filter8_sp_s16x4<coeffIdy>(v_s + 2, res[2], v_offset, v_filter,
                                       vget_low_u16(maxVal));
            v_s[1] = v_s[9];

            filter8_ps_u16x4<coeffIdx>(h_s0, v_s[10], h_offset, h_filter);
            filter8_sp_s16x4<coeffIdy>(v_s + 3, res[3], v_offset, v_filter,
                                       vget_low_u16(maxVal));
            v_s[2] = v_s[10];

            load_u16x4xn<8>(src + 4 * srcStride, 1, h_s1);
            load_u16x4xn<8>(src + 5 * srcStride, 1, h_s0);

            filter8_ps_u16x4<coeffIdx>(h_s1, v_s[11], h_offset, h_filter);
            filter8_sp_s16x4<coeffIdy>(v_s + 4, res[4], v_offset, v_filter,
                                       vget_low_u16(maxVal));
            v_s[3] = v_s[11];

            filter8_ps_u16x4<coeffIdx>(h_s0, v_s[12], h_offset, h_filter);
            filter8_sp_s16x4<coeffIdy>(v_s + 5, res[5], v_offset, v_filter,
                                       vget_low_u16(maxVal));
            v_s[4] = v_s[12];

            load_u16x4xn<8>(src + 6 * srcStride, 1, h_s1);
            load_u16x4xn<8>(src + 7 * srcStride, 1, h_s0);

            filter8_ps_u16x4<coeffIdx>(h_s1, v_s[13], h_offset, h_filter);
            filter8_sp_s16x4<coeffIdy>(v_s + 6, res[6], v_offset, v_filter,
                                       vget_low_u16(maxVal));
            v_s[5] = v_s[13];

            filter8_ps_u16x4<coeffIdx>(h_s0, v_s[14], h_offset, h_filter);
            filter8_sp_s16x4<coeffIdy>(v_s + 7, res[7], v_offset, v_filter,
                                       vget_low_u16(maxVal));
            v_s[6] = v_s[14];

            store_u16xnxm<4, 8>(dst, dstStride, res);

            src += 8 * srcStride;
            dst += 8 * dstStride;
        }

        for (; row < height; row += 4)
        {
            uint16x4_t res[4];

            load_u16x4xn<8>(src + 0 * srcStride, 1, h_s1);
            load_u16x4xn<8>(src + 1 * srcStride, 1, h_s0);

            filter8_ps_u16x4<coeffIdx>(h_s1, v_s[7], h_offset, h_filter);
            filter8_sp_s16x4<coeffIdy>(v_s + 0, res[0], v_offset, v_filter,
                                       vget_low_u16(maxVal));
            v_s[0] = v_s[4];

            filter8_ps_u16x4<coeffIdx>(h_s0, v_s[8], h_offset, h_filter);
            filter8_sp_s16x4<coeffIdy>(v_s + 1, res[1], v_offset, v_filter,
                                       vget_low_u16(maxVal));
            v_s[1] = v_s[5];

            load_u16x4xn<8>(src + 2 * srcStride, 1, h_s1);
            load_u16x4xn<8>(src + 3 * srcStride, 1, h_s0);

            filter8_ps_u16x4<coeffIdx>(h_s1, v_s[9], h_offset, h_filter);
            filter8_sp_s16x4<coeffIdy>(v_s + 2, res[2], v_offset, v_filter,
                                       vget_low_u16(maxVal));
            v_s[2] = v_s[6];

            filter8_ps_u16x4<coeffIdx>(h_s0, v_s[10], h_offset, h_filter);
            filter8_sp_s16x4<coeffIdy>(v_s + 3, res[3], v_offset, v_filter,
                                       vget_low_u16(maxVal));
            v_s[3] = v_s[7];

            store_u16xnxm<4, 4>(dst, dstStride, res);

            v_s[4] = v_s[8];
            v_s[5] = v_s[9];
            v_s[6] = v_s[10];

            src += 4 * srcStride;
            dst += 4 * dstStride;
        }
    }
}

#endif // !HIGH_BIT_DEPTH
}

namespace X265_NS
{

template<int N, int width, int height>
void interp_horiz_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst,
                          intptr_t dstStride, int coeffIdx)
{
    if (N == 8)
    {
        switch (coeffIdx)
        {
        case 1:
            return interp8_horiz_pp_neon<1, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 2:
            return interp8_horiz_pp_neon<2, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 3:
            return interp8_horiz_pp_neon<3, width, height>(src, srcStride, dst,
                                                           dstStride);
        }
    }
    else
    {
        switch (coeffIdx)
        {
        case 4:
            return interp4_horiz_pp_neon<true, width, height>(src, srcStride,
                                                              dst, dstStride,
                                                              coeffIdx);
        default:
            return interp4_horiz_pp_neon<false, width, height>(src, srcStride,
                                                               dst, dstStride,
                                                               coeffIdx);
        }
    }
}

template<int N, int width, int height>
void interp_horiz_ps_neon(const pixel *src, intptr_t srcStride, int16_t *dst,
                          intptr_t dstStride, int coeffIdx, int isRowExt)
{
    if (N == 8)
    {
        switch (coeffIdx)
        {
        case 1:
            return interp8_horiz_ps_neon<1, width, height>(src, srcStride, dst,
                                                           dstStride, isRowExt);
        case 2:
            return interp8_horiz_ps_neon<2, width, height>(src, srcStride, dst,
                                                           dstStride, isRowExt);
        case 3:
            return interp8_horiz_ps_neon<3, width, height>(src, srcStride, dst,
                                                           dstStride, isRowExt);
        }
    }
    else
    {
        switch (coeffIdx)
        {
        case 4:
            return interp4_horiz_ps_neon<true, width, height>(src, srcStride,
                                                              dst, dstStride,
                                                              coeffIdx,
                                                              isRowExt);
        default:
            return interp4_horiz_ps_neon<false, width, height>(src, srcStride,
                                                               dst, dstStride,
                                                               coeffIdx,
                                                               isRowExt);
        }
    }
}

template<int N, int width, int height>
void interp_vert_ss_neon(const int16_t *src, intptr_t srcStride, int16_t *dst, intptr_t dstStride, int coeffIdx)
{
    if (N == 8)
    {
        switch (coeffIdx)
        {
        case 1:
            return interp8_vert_ss_neon<1, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 2:
            return interp8_vert_ss_neon<2, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 3:
            return interp8_vert_ss_neon<3, width, height>(src, srcStride, dst,
                                                          dstStride);
        }
    }
    else
    {
        switch (coeffIdx)
        {
        case 4:
            return interp4_vert_ss_neon<true, width, height>(src, srcStride, dst,
                                                             dstStride, coeffIdx);
        default:
            return interp4_vert_ss_neon<false, width, height>(src, srcStride, dst,
                                                              dstStride, coeffIdx);
        }
    }
}

template<int N, int width, int height>
void interp_vert_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst,
                         intptr_t dstStride, int coeffIdx)
{
    if (N == 8)
    {
        switch (coeffIdx)
        {
        case 1:
            return interp8_vert_pp_neon<1, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 2:
            return interp8_vert_pp_neon<2, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 3:
            return interp8_vert_pp_neon<3, width, height>(src, srcStride, dst,
                                                          dstStride);
        }
    }
    else
    {
        switch (coeffIdx)
        {
        case 4:
            return interp4_vert_pp_neon<true, width, height>(src, srcStride,
                                                             dst, dstStride,
                                                             coeffIdx);
        default:
            return interp4_vert_pp_neon<false, width, height>(src, srcStride,
                                                              dst, dstStride,
                                                              coeffIdx);
        }
    }
}

template<int N, int width, int height>
void interp_vert_ps_neon(const pixel *src, intptr_t srcStride, int16_t *dst,
                         intptr_t dstStride, int coeffIdx)
{
    if (N == 8)
    {
        switch (coeffIdx)
        {
        case 1:
            return interp8_vert_ps_neon<1, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 2:
            return interp8_vert_ps_neon<2, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 3:
            return interp8_vert_ps_neon<3, width, height>(src, srcStride, dst,
                                                          dstStride);
        }
    }
    else
    {
        switch (coeffIdx)
        {
        case 4:
            return interp4_vert_ps_neon<true, width, height>(src, srcStride,
                                                             dst, dstStride,
                                                             coeffIdx);
        default:
            return interp4_vert_ps_neon<false, width, height>(src, srcStride,
                                                              dst, dstStride,
                                                              coeffIdx);
        }
    }
}

template<int N, int width, int height>
void interp_vert_sp_neon(const int16_t *src, intptr_t srcStride, pixel *dst,
                         intptr_t dstStride, int coeffIdx)
{
    if (N == 8)
    {
        switch (coeffIdx)
        {
        case 1:
            return interp8_vert_sp_neon<1, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 2:
            return interp8_vert_sp_neon<2, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 3:
            return interp8_vert_sp_neon<3, width, height>(src, srcStride, dst,
                                                          dstStride);
        }
    }
    else
    {
        switch (coeffIdx)
        {
        case 4:
            return interp4_vert_sp_neon<true, width, height>(src, srcStride, dst,
                                                             dstStride, coeffIdx);
        default:
            return interp4_vert_sp_neon<false, width, height>(src, srcStride, dst,
                                                              dstStride, coeffIdx);
        }
    }
}

#if HIGH_BIT_DEPTH
template<int N, int width, int height>
void interp_hv_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst,
                       intptr_t dstStride, int idxX, int idxY)
{
// Use the merged hv paths with Clang only as performance with GCC is worse than the
// existing approach of doing horizontal and vertical interpolation separately.
#ifdef __clang__
    switch (idxX)
    {
    case 1:
    {
        switch (idxY)
        {
        case 1:
            return interp8_hv_pp_neon<1, 1, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 2:
            return interp8_hv_pp_neon<1, 2, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 3:
            return interp8_hv_pp_neon<1, 3, width, height>(src, srcStride, dst,
                                                           dstStride);
        }

        break;
    }
    case 2:
    {
        switch (idxY)
        {
        case 1:
            return interp8_hv_pp_neon<2, 1, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 2:
            return interp8_hv_pp_neon<2, 2, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 3:
            return interp8_hv_pp_neon<2, 3, width, height>(src, srcStride, dst,
                                                           dstStride);
        }

        break;
    }
    case 3:
    {
        switch (idxY)
        {
        case 1:
            return interp8_hv_pp_neon<3, 1, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 2:
            return interp8_hv_pp_neon<3, 2, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 3:
            return interp8_hv_pp_neon<3, 3, width, height>(src, srcStride, dst,
                                                           dstStride);
        }

        break;
    }
    }

#else // __clang__

    ALIGN_VAR_32(int16_t, immed[width * (height + N - 1)]);

    interp_horiz_ps_neon<N, width, height>(src, srcStride, immed, width, idxX, 1);
    interp_vert_sp_neon<N, width, height>(immed + (N / 2 - 1) * width, width, dst,
                                          dstStride, idxY);
#endif // __clang__
}

#else // HIGH_BIT_DEPTH

template<int N, int width, int height>
void interp_hv_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst,
                       intptr_t dstStride, int idxX, int idxY)
{
// Use the merged hv paths with Clang only as performance with GCC is worse than the
// existing approach of doing horizontal and vertical interpolation separately.
#ifdef __clang__
    switch (idxX)
    {
    case 1:
    {
        switch (idxY)
        {
        case 1:
            return interp8_hv_pp_neon<1, 1, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 2:
            return interp8_hv_pp_neon<1, 2, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 3:
            return interp8_hv_pp_neon<1, 3, width, height>(src, srcStride, dst,
                                                           dstStride);
        }

        break;
    }
    case 2:
    {
        switch (idxY)
        {
        case 1:
            return interp8_hv_pp_neon<2, 1, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 2:
            return interp8_hv_pp_neon<2, 2, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 3:
            return interp8_hv_pp_neon<2, 3, width, height>(src, srcStride, dst,
                                                           dstStride);
        }

        break;
    }
    case 3:
    {
        switch (idxY)
        {
        case 1:
            return interp8_hv_pp_neon<3, 1, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 2:
            return interp8_hv_pp_neon<3, 2, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 3:
            return interp8_hv_pp_neon<3, 3, width, height>(src, srcStride, dst,
                                                           dstStride);
        }

        break;
    }
    }

#else // __clang__
    ALIGN_VAR_32(int16_t, immed[width * (height + N - 1)]);

    interp_horiz_ps_neon<N, width, height>(src, srcStride, immed, width, idxX, 1);
    interp_vert_sp_neon<N, width, height>(immed + (N / 2 - 1) * width, width, dst,
                                          dstStride, idxY);
#endif // __clang__
}

#endif // HIGH_BIT_DEPTH

template<int width, int height>
void filterPixelToShort_neon(const pixel *src, intptr_t srcStride, int16_t *dst, intptr_t dstStride)
{
    const int shift = IF_INTERNAL_PREC - X265_DEPTH;
    const int16x8_t off = vdupq_n_s16(IF_INTERNAL_OFFS);

    for (int row = 0; row < height; row++)
    {
        int col = 0;
        for (; col + 8 <= width; col += 8)
        {
            uint16x8_t in;

#if HIGH_BIT_DEPTH
            in = vld1q_u16(src + col);
#else
            in = vmovl_u8(vld1_u8(src + col));
#endif

            int16x8_t tmp = vreinterpretq_s16_u16(vshlq_n_u16(in, shift));
            tmp = vsubq_s16(tmp, off);
            vst1q_s16(dst + col, tmp);
        }

        for (; col + 4 <= width; col += 4)
        {
            uint16x4_t in;

#if HIGH_BIT_DEPTH
            in = vld1_u16(src + col);
#else
            in = vget_low_u16(vmovl_u8(vld1_u8(src + col)));
#endif

            int16x4_t tmp = vreinterpret_s16_u16(vshl_n_u16(in, shift));
            tmp = vsub_s16(tmp, vget_low_s16(off));
            vst1_s16(dst + col, tmp);
        }

        for (; col < width; col += 2)
        {
            uint16x4_t in;

#if HIGH_BIT_DEPTH
            in = vld1_u16(src + col);
#else
            in = vget_low_u16(vmovl_u8(vld1_u8(src + col)));
#endif

            int16x4_t tmp = vreinterpret_s16_u16(vshl_n_u16(in, shift));
            tmp = vsub_s16(tmp, vget_low_s16(off));
            store_s16x2xn<1>(dst + col, dstStride, &tmp);
        }

        src += srcStride;
        dst += dstStride;
    }
}

#define CHROMA_420(W, H) \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_hpp = interp_horiz_pp_neon<4, W, H>; \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_hps = interp_horiz_ps_neon<4, W, H>; \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_vpp = interp_vert_pp_neon<4, W, H>;  \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_vps = interp_vert_ps_neon<4, W, H>;  \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_vsp = interp_vert_sp_neon<4, W, H>;  \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_vss = interp_vert_ss_neon<4, W, H>;  \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].p2s[NONALIGNED] = filterPixelToShort_neon<W, H>;\
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].p2s[ALIGNED] = filterPixelToShort_neon<W, H>;

#define CHROMA_422(W, H) \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_hpp = interp_horiz_pp_neon<4, W, H>; \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_hps = interp_horiz_ps_neon<4, W, H>; \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_vpp = interp_vert_pp_neon<4, W, H>;  \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_vps = interp_vert_ps_neon<4, W, H>;  \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_vsp = interp_vert_sp_neon<4, W, H>;  \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_vss = interp_vert_ss_neon<4, W, H>;  \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].p2s[NONALIGNED] = filterPixelToShort_neon<W, H>;\
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].p2s[ALIGNED] = filterPixelToShort_neon<W, H>;

#define CHROMA_444(W, H) \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_hpp = interp_horiz_pp_neon<4, W, H>; \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_hps = interp_horiz_ps_neon<4, W, H>; \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_vpp = interp_vert_pp_neon<4, W, H>;  \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_vps = interp_vert_ps_neon<4, W, H>;  \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_vsp = interp_vert_sp_neon<4, W, H>;  \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_vss = interp_vert_ss_neon<4, W, H>;  \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].p2s[NONALIGNED] = filterPixelToShort_neon<W, H>;\
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].p2s[ALIGNED] = filterPixelToShort_neon<W, H>;

#define LUMA(W, H) \
    p.pu[LUMA_ ## W ## x ## H].luma_hpp     = interp_horiz_pp_neon<8, W, H>; \
    p.pu[LUMA_ ## W ## x ## H].luma_hps     = interp_horiz_ps_neon<8, W, H>; \
    p.pu[LUMA_ ## W ## x ## H].luma_vpp     = interp_vert_pp_neon<8, W, H>;  \
    p.pu[LUMA_ ## W ## x ## H].luma_vps     = interp_vert_ps_neon<8, W, H>;  \
    p.pu[LUMA_ ## W ## x ## H].luma_vsp     = interp_vert_sp_neon<8, W, H>;  \
    p.pu[LUMA_ ## W ## x ## H].luma_vss     = interp_vert_ss_neon<8, W, H>;  \
    p.pu[LUMA_ ## W ## x ## H].luma_hvpp    = interp_hv_pp_neon<8, W, H>; \
    p.pu[LUMA_ ## W ## x ## H].convert_p2s[NONALIGNED] = filterPixelToShort_neon<W, H>;\
    p.pu[LUMA_ ## W ## x ## H].convert_p2s[ALIGNED] = filterPixelToShort_neon<W, H>;

void setupFilterPrimitives_neon(EncoderPrimitives &p)
{
    LUMA(4, 4);
    LUMA(4, 8);
    LUMA(4, 16);
    LUMA(12, 16);
    LUMA(8, 4);
    LUMA(8, 8);
    LUMA(8, 16);
    LUMA(8, 32);
    LUMA(16, 4);
    LUMA(16, 8);
    LUMA(16, 12);
    LUMA(16, 16);
    LUMA(16, 32);
    LUMA(16, 64);
    LUMA(24, 32);
    LUMA(32, 8);
    LUMA(32, 16);
    LUMA(32, 24);
    LUMA(32, 32);
    LUMA(32, 64);
    LUMA(48, 64);
    LUMA(64, 16);
    LUMA(64, 32);
    LUMA(64, 48);
    LUMA(64, 64);

    CHROMA_420(2, 4);
    CHROMA_420(2, 8);
    CHROMA_420(4, 2);
    CHROMA_420(4, 4);
    CHROMA_420(4, 8);
    CHROMA_420(4, 16);
    CHROMA_420(6, 8);
    CHROMA_420(12, 16);
    CHROMA_420(8, 2);
    CHROMA_420(8, 4);
    CHROMA_420(8, 6);
    CHROMA_420(8, 8);
    CHROMA_420(8, 16);
    CHROMA_420(8, 32);
    CHROMA_420(16, 4);
    CHROMA_420(16, 8);
    CHROMA_420(16, 12);
    CHROMA_420(16, 16);
    CHROMA_420(16, 32);
    CHROMA_420(24, 32);
    CHROMA_420(32, 8);
    CHROMA_420(32, 16);
    CHROMA_420(32, 24);
    CHROMA_420(32, 32);

    CHROMA_422(2, 8);
    CHROMA_422(2, 16);
    CHROMA_422(4, 4);
    CHROMA_422(4, 8);
    CHROMA_422(4, 16);
    CHROMA_422(4, 32);
    CHROMA_422(6, 16);
    CHROMA_422(12, 32);
    CHROMA_422(8, 4);
    CHROMA_422(8, 8);
    CHROMA_422(8, 12);
    CHROMA_422(8, 16);
    CHROMA_422(8, 32);
    CHROMA_422(8, 64);
    CHROMA_422(16, 8);
    CHROMA_422(16, 16);
    CHROMA_422(16, 24);
    CHROMA_422(16, 32);
    CHROMA_422(16, 64);
    CHROMA_422(24, 64);
    CHROMA_422(32, 16);
    CHROMA_422(32, 32);
    CHROMA_422(32, 48);
    CHROMA_422(32, 64);

    CHROMA_444(4, 4);
    CHROMA_444(4, 8);
    CHROMA_444(4, 16);
    CHROMA_444(12, 16);
    CHROMA_444(8, 4);
    CHROMA_444(8, 8);
    CHROMA_444(8, 16);
    CHROMA_444(8, 32);
    CHROMA_444(16, 4);
    CHROMA_444(16, 8);
    CHROMA_444(16, 12);
    CHROMA_444(16, 16);
    CHROMA_444(16, 32);
    CHROMA_444(16, 64);
    CHROMA_444(24, 32);
    CHROMA_444(32, 8);
    CHROMA_444(32, 16);
    CHROMA_444(32, 24);
    CHROMA_444(32, 32);
    CHROMA_444(32, 64);
    CHROMA_444(48, 64);
    CHROMA_444(64, 16);
    CHROMA_444(64, 32);
    CHROMA_444(64, 48);
    CHROMA_444(64, 64);
}

};


#endif


