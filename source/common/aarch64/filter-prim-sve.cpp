/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Gerda Zsejke More <gerdazsejke.more@arm.com>
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

#include "filter-prim-sve.h"
#include "mem-neon.h"
#include "neon-sve-bridge.h"

#include <arm_neon.h>

#if HIGH_BIT_DEPTH
#define SHIFT_INTERP_PS (IF_FILTER_PREC - (IF_INTERNAL_PREC - X265_DEPTH))

static const uint16_t dotprod_h_permute_tbl[32] = {
    // clang-format off
    0, 1, 2, 3, 1, 2, 3, 4,
    2, 3, 4, 5, 3, 4, 5, 6,
    3, 2, 1, 0, 4, 3, 2, 1,
    5, 4, 3, 2, 6, 5, 4, 3,
    // clang-format on
};

template<bool coeff2>
void inline filter8_u16x4(const uint16x8_t *s, uint16x4_t &d, int16x8_t filter,
                          uint16x4_t maxVal)
{
    if (coeff2)
    {
        int16x8_t sum01 = vreinterpretq_s16_u16(vaddq_u16(s[0], s[1]));
        int16x8_t sum23 = vreinterpretq_s16_u16(vaddq_u16(s[2], s[3]));

        int64x2_t sum_lo = x265_sdotq_lane_s16(vdupq_n_s64(0), sum01, filter, 0);
        int64x2_t sum_hi = x265_sdotq_lane_s16(vdupq_n_s64(0), sum23, filter, 0);

        int32x4_t sum = vcombine_s32(vmovn_s64(sum_lo), vmovn_s64(sum_hi));

        d = vqrshrun_n_s32(sum, IF_FILTER_PREC);
        d = vmin_u16(d, maxVal);
    }
    else
    {
        int64x2_t sum_lo =
            x265_sdotq_lane_s16(vdupq_n_s64(0), vreinterpretq_s16_u16(s[0]), filter, 0);
        int64x2_t sum_hi =
            x265_sdotq_lane_s16(vdupq_n_s64(0), vreinterpretq_s16_u16(s[2]), filter, 0);

        sum_lo = x265_sdotq_lane_s16(sum_lo, vreinterpretq_s16_u16(s[1]), filter, 1);
        sum_hi = x265_sdotq_lane_s16(sum_hi, vreinterpretq_s16_u16(s[3]), filter, 1);

        int32x4_t sum = vcombine_s32(vmovn_s64(sum_lo), vmovn_s64(sum_hi));

        d = vqrshrun_n_s32(sum, IF_FILTER_PREC);
        d = vmin_u16(d, maxVal);
    }
}

template<bool coeff2>
void inline filter8_u16x8(uint16x8_t *s, uint16x8_t &d, int16x8_t filter,
                          uint16x8_t maxVal)
{
    if (coeff2)
    {
        int16x8_t sum01 = vreinterpretq_s16_u16(vaddq_u16(s[0], s[1]));
        int16x8_t sum23 = vreinterpretq_s16_u16(vaddq_u16(s[2], s[3]));
        int16x8_t sum45 = vreinterpretq_s16_u16(vaddq_u16(s[4], s[5]));
        int16x8_t sum67 = vreinterpretq_s16_u16(vaddq_u16(s[6], s[7]));

        int64x2_t sum0 = x265_sdotq_lane_s16(vdupq_n_s64(0), sum01, filter, 0);
        int64x2_t sum1 = x265_sdotq_lane_s16(vdupq_n_s64(0), sum23, filter, 0);
        int64x2_t sum2 = x265_sdotq_lane_s16(vdupq_n_s64(0), sum45, filter, 0);
        int64x2_t sum3 = x265_sdotq_lane_s16(vdupq_n_s64(0), sum67, filter, 0);

        int32x4_t sum_lo = vcombine_s32(vmovn_s64(sum0), vmovn_s64(sum1));
        int32x4_t sum_hi = vcombine_s32(vmovn_s64(sum2), vmovn_s64(sum3));

        uint16x4_t d_lo = vqrshrun_n_s32(sum_lo, IF_FILTER_PREC);
        uint16x4_t d_hi = vqrshrun_n_s32(sum_hi, IF_FILTER_PREC);

        d = vminq_u16(vcombine_u16(d_lo, d_hi), maxVal);
    }
    else
    {
        int64x2_t sum0 =
            x265_sdotq_lane_s16(vdupq_n_s64(0), vreinterpretq_s16_u16(s[0]), filter, 0);
        int64x2_t sum1 =
            x265_sdotq_lane_s16(vdupq_n_s64(0), vreinterpretq_s16_u16(s[1]), filter, 0);
        int64x2_t sum2 =
            x265_sdotq_lane_s16(vdupq_n_s64(0), vreinterpretq_s16_u16(s[2]), filter, 0);
        int64x2_t sum3 =
            x265_sdotq_lane_s16(vdupq_n_s64(0), vreinterpretq_s16_u16(s[3]), filter, 0);

        sum0 = x265_sdotq_lane_s16(sum0, vreinterpretq_s16_u16(s[4]), filter, 1);
        sum1 = x265_sdotq_lane_s16(sum1, vreinterpretq_s16_u16(s[5]), filter, 1);
        sum2 = x265_sdotq_lane_s16(sum2, vreinterpretq_s16_u16(s[6]), filter, 1);
        sum3 = x265_sdotq_lane_s16(sum3, vreinterpretq_s16_u16(s[7]), filter, 1);

        int32x4_t sum_lo = vcombine_s32(vmovn_s64(sum0), vmovn_s64(sum2));
        int32x4_t sum_hi = vcombine_s32(vmovn_s64(sum1), vmovn_s64(sum3));

        uint16x4_t d_lo = vqrshrun_n_s32(sum_lo, IF_FILTER_PREC);
        uint16x4_t d_hi = vqrshrun_n_s32(sum_hi, IF_FILTER_PREC);

        d = vminq_u16(vcombine_u16(d_lo, d_hi), maxVal);
    }
}

template<bool coeff2>
void inline setup_s_hpp_x4(uint16x8_t *d, uint16x8_t s0, uint16x8_t s1, uint16x8_t *idx)
{
    if (coeff2)
    {
        d[0] = x265_tblq_u16(s0, idx[0]);
        d[1] = x265_tblq_u16(s1, idx[2]);
        d[2] = x265_tblq_u16(s0, idx[1]);
        d[3] = x265_tblq_u16(s1, idx[3]);
    }
    else
    {
        d[0] = x265_tblq_u16(s0, idx[0]);
        d[1] = x265_tblq_u16(s1, idx[0]);
        d[2] = x265_tblq_u16(s0, idx[1]);
        d[3] = x265_tblq_u16(s1, idx[1]);
    }
}

template<bool coeff2>
void inline setup_s_hpp_x8(uint16x8_t *d, uint16x8_t s0, uint16x8_t s1, uint16x8_t s2,
                           uint16x8_t *idx)
{
    if (coeff2)
    {
        d[0] = x265_tblq_u16(s0, idx[0]);
        d[1] = x265_tblq_u16(s1, idx[2]);
        d[2] = x265_tblq_u16(s0, idx[1]);
        d[3] = x265_tblq_u16(s1, idx[3]);
        d[4] = x265_tblq_u16(s1, idx[0]);
        d[5] = x265_tblq_u16(s2, idx[2]);
        d[6] = x265_tblq_u16(s1, idx[1]);
        d[7] = x265_tblq_u16(s2, idx[3]);
    }
    else
    {
        d[0] = x265_tblq_u16(s0, idx[0]);
        d[1] = x265_tblq_u16(s1, idx[0]);
        d[2] = x265_tblq_u16(s0, idx[1]);
        d[3] = x265_tblq_u16(s1, idx[1]);
        d[4] = d[1];
        d[5] = x265_tblq_u16(s2, idx[0]);
        d[6] = d[3];
        d[7] = x265_tblq_u16(s2, idx[1]);
    }
}

template<bool coeff2, int width, int height>
void inline interp8_hpp_sve(const pixel *src, intptr_t srcStride,
                            pixel *dst, intptr_t dstStride, int coeffIdx)
{
    const int N_TAPS = 8;
    const uint16x8_t maxVal = vdupq_n_u16((1 << X265_DEPTH) - 1);
    const int16x8_t filter = vld1q_s16(X265_NS::g_lumaFilter[coeffIdx]);
    uint16x8_t idx[4];

    idx[0] = vld1q_u16(dotprod_h_permute_tbl + 0);
    idx[1] = vld1q_u16(dotprod_h_permute_tbl + 8);
    if (coeff2)
    {
        idx[2] = vld1q_u16(dotprod_h_permute_tbl + 16);
        idx[3] = vld1q_u16(dotprod_h_permute_tbl + 24);
    }

    src -= N_TAPS / 2 - 1;

    for (int row = 0; row < height; row++)
    {
        if (width % 16 == 0 || width == 24)
        {
            int col = 0;
            for (; col <= width - 16; col += 16)
            {
                uint16x8_t s[5];
                load_u16x8xn<5>(src + col, 4, s);

                uint16x8_t s0[N_TAPS], s1[N_TAPS];
                setup_s_hpp_x8<coeff2>(s0, s[0], s[1], s[2], idx);
                setup_s_hpp_x8<coeff2>(s1, s[2], s[3], s[4], idx);

                uint16x8_t d0, d1;
                filter8_u16x8<coeff2>(s0, d0, filter, maxVal);
                filter8_u16x8<coeff2>(s1, d1, filter, maxVal);

                vst1q_u16(dst + col + 0, d0);
                vst1q_u16(dst + col + 8, d1);
            }

            if (width == 24)
            {
                uint16x8_t s[3];
                load_u16x8xn<3>(src + col, 4, s);

                uint16x8_t s0[N_TAPS];
                setup_s_hpp_x8<coeff2>(s0, s[0], s[1], s[2], idx);

                uint16x8_t d0;
                filter8_u16x8<coeff2>(s0, d0, filter, maxVal);

                vst1q_u16(dst + col, d0);
            }
        }
        else if (width == 4)
        {
            uint16x8_t s[2];
            load_u16x8xn<2>(src, 4, s);

            uint16x8_t s0[N_TAPS];
            setup_s_hpp_x4<coeff2>(s0, s[0], s[1], idx);

            uint16x4_t d0;
            filter8_u16x4<coeff2>(s0, d0, filter, vget_low_u16(maxVal));

            vst1_u16(dst, d0);
        }

        src += srcStride;
        dst += dstStride;
    }
}

void inline filter8_ps_u16x4(const uint16x8_t *s, int16x4_t &d, int16x8_t filter,
                             int64x2_t offset)
{
    int16x8_t sum01 = vreinterpretq_s16_u16(vaddq_u16(s[0], s[1]));
    int16x8_t sum23 = vreinterpretq_s16_u16(vaddq_u16(s[2], s[3]));

    int64x2_t sum_lo = x265_sdotq_lane_s16(offset, sum01, filter, 0);
    int64x2_t sum_hi = x265_sdotq_lane_s16(offset, sum23, filter, 0);

    int32x4_t sum = vcombine_s32(vmovn_s64(sum_lo), vmovn_s64(sum_hi));

    d = vshrn_n_s32(sum, SHIFT_INTERP_PS);
}

template<int width, int height>
void inline interp8_hps_sve(const pixel *src, intptr_t srcStride,
                            int16_t *dst, intptr_t dstStride, int coeffIdx, int isRowExt)
{
    const int N_TAPS = 8;
    int blkheight = height;
    const int16x8_t filter = vld1q_s16(X265_NS::g_lumaFilter[coeffIdx]);
    const int64x2_t offset =
        vdupq_n_s64((unsigned)-IF_INTERNAL_OFFS << SHIFT_INTERP_PS);

    uint16x8_t idx[4];

    idx[0] = vld1q_u16(dotprod_h_permute_tbl + 0);
    idx[1] = vld1q_u16(dotprod_h_permute_tbl + 8);
    idx[2] = vld1q_u16(dotprod_h_permute_tbl + 16);
    idx[3] = vld1q_u16(dotprod_h_permute_tbl + 24);

    if (isRowExt)
    {
        src -= (N_TAPS / 2 - 1) * srcStride;
        blkheight += N_TAPS - 1;
    }

    src -= N_TAPS / 2 - 1;

    for (int row = 0; row < blkheight; row++)
    {
        uint16x8_t s[2];
        s[0] = vld1q_u16(src);
        s[1] = vld1q_u16(src + 4);

        uint16x8_t s0[N_TAPS];
        setup_s_hpp_x4<true>(s0, s[0], s[1], idx);

        int16x4_t d0;
        filter8_ps_u16x4(s0, d0, filter, offset);

        vst1_s16(dst, d0);

        src += srcStride;
        dst += dstStride;
    }
}

namespace X265_NS {
// Declaration for use in interp8_horiz_pp_sve().
template<int N, int width, int height>
void interp_horiz_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst,
                          intptr_t dstStride, int coeffIdx);

template<int width, int height>
void interp8_horiz_pp_sve(const pixel *src, intptr_t srcStride, pixel *dst,
                          intptr_t dstStride, int coeffIdx)
{
    switch (coeffIdx)
    {
    case 1:
        if (width <= 16)
        {
            return interp_horiz_pp_neon<8, width, height>(src, srcStride, dst,
                                                          dstStride, coeffIdx);
        }
        else
        {
            return interp8_hpp_sve<false, width, height>(src, srcStride, dst,
                                                         dstStride, coeffIdx);
        }
    case 2:
        if (width > 4)
        {
            return interp_horiz_pp_neon<8, width, height>(src, srcStride, dst,
                                                          dstStride, coeffIdx);
        }
        else
        {
            return interp8_hpp_sve<true, width, height>(src, srcStride, dst,
                                                        dstStride, coeffIdx);
        }
    case 3:
        return interp_horiz_pp_neon<8, width, height>(src, srcStride, dst,
                                                      dstStride, coeffIdx);
    }
}

// Declaration for use in interp8_horiz_ps_sve().
template<int N, int width, int height>
void interp_horiz_ps_neon(const pixel *src, intptr_t srcStride, int16_t *dst,
                          intptr_t dstStride, int coeffIdx, int isRowExt);

template<int width, int height>
void interp8_horiz_ps_sve(const pixel *src, intptr_t srcStride, int16_t *dst,
                          intptr_t dstStride, int coeffIdx, int isRowExt)
{
    switch (coeffIdx)
    {
    case 1:
        return interp_horiz_ps_neon<8, width, height>(src, srcStride, dst, dstStride,
                                                      coeffIdx, isRowExt);
    case 2:
        return interp8_hps_sve<width, height>(src, srcStride, dst, dstStride,
                                              coeffIdx, isRowExt);
    case 3:
        return interp_horiz_ps_neon<8, width, height>(src, srcStride, dst, dstStride,
                                                      coeffIdx, isRowExt);
    }
}

void setupFilterPrimitives_sve(EncoderPrimitives &p)
{
    p.pu[LUMA_4x4].luma_hpp    = interp8_horiz_pp_sve<4, 4>;
    p.pu[LUMA_4x8].luma_hpp    = interp8_horiz_pp_sve<4, 8>;
    p.pu[LUMA_4x16].luma_hpp   = interp8_horiz_pp_sve<4, 16>;
#if X265_DEPTH == 12
    p.pu[LUMA_16x4].luma_hpp   = interp8_horiz_pp_sve<16, 4>;
    p.pu[LUMA_16x8].luma_hpp   = interp8_horiz_pp_sve<16, 8>;
    p.pu[LUMA_16x12].luma_hpp  = interp8_horiz_pp_sve<16, 12>;
    p.pu[LUMA_16x16].luma_hpp  = interp8_horiz_pp_sve<16, 16>;
    p.pu[LUMA_16x32].luma_hpp  = interp8_horiz_pp_sve<16, 32>;
    p.pu[LUMA_16x64].luma_hpp  = interp8_horiz_pp_sve<16, 64>;
    p.pu[LUMA_24x32].luma_hpp  = interp8_horiz_pp_sve<24, 32>;
    p.pu[LUMA_32x8].luma_hpp   = interp8_horiz_pp_sve<32, 8>;
    p.pu[LUMA_32x16].luma_hpp  = interp8_horiz_pp_sve<32, 16>;
    p.pu[LUMA_32x24].luma_hpp  = interp8_horiz_pp_sve<32, 24>;
    p.pu[LUMA_32x32].luma_hpp  = interp8_horiz_pp_sve<32, 32>;
    p.pu[LUMA_32x64].luma_hpp  = interp8_horiz_pp_sve<32, 64>;
    p.pu[LUMA_48x64].luma_hpp  = interp8_horiz_pp_sve<48, 64>;
    p.pu[LUMA_64x16].luma_hpp  = interp8_horiz_pp_sve<64, 16>;
    p.pu[LUMA_64x32].luma_hpp  = interp8_horiz_pp_sve<64, 32>;
    p.pu[LUMA_64x48].luma_hpp  = interp8_horiz_pp_sve<64, 48>;
    p.pu[LUMA_64x64].luma_hpp  = interp8_horiz_pp_sve<64, 64>;
#endif // X265_DEPTH == 12

    p.pu[LUMA_4x4].luma_hps   = interp8_horiz_ps_sve<4, 4>;
    p.pu[LUMA_4x8].luma_hps   = interp8_horiz_ps_sve<4, 8>;
    p.pu[LUMA_4x16].luma_hps  = interp8_horiz_ps_sve<4, 16>;
}
} // namespace X265_NS
#else // !HIGH_BIT_DEPTH
namespace X265_NS {
void setupFilterPrimitives_sve(EncoderPrimitives &)
{
}
}
#endif // HIGH_BIT_DEPTH
