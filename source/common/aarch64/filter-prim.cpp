#if HAVE_NEON

#include "filter-prim.h"
#include "mem-neon.h"

#include <arm_neon.h>

namespace {
void inline filter4_s16x8(int coeffIdx, const int16x8_t *s, const int16x4_t f,
                          const int32x4_t c, int32x4_t &d0, int32x4_t &d1)
{
    if (coeffIdx == 4)
    {
        // { -4, 36, 36, -4 }
        int16x8_t t0 = vaddq_s16(s[1], s[2]);
        int16x8_t t1 = vaddq_s16(s[0], s[3]);
        d0 = vmlal_n_s16(c, vget_low_s16(t0), 36);
        d0 = vmlsl_n_s16(d0, vget_low_s16(t1), 4);

        d1 = vmlal_n_s16(c, vget_high_s16(t0), 36);
        d1 = vmlsl_n_s16(d1, vget_high_s16(t1), 4);
    }
    else
    {
        d0 = vmlal_lane_s16(c, vget_low_s16(s[0]), f, 0);
        d0 = vmlal_lane_s16(d0, vget_low_s16(s[1]), f, 1);
        d0 = vmlal_lane_s16(d0, vget_low_s16(s[2]), f, 2);
        d0 = vmlal_lane_s16(d0, vget_low_s16(s[3]), f, 3);

        d1 = vmlal_lane_s16(c, vget_high_s16(s[0]), f, 0);
        d1 = vmlal_lane_s16(d1, vget_high_s16(s[1]), f, 1);
        d1 = vmlal_lane_s16(d1, vget_high_s16(s[2]), f, 2);
        d1 = vmlal_lane_s16(d1, vget_high_s16(s[3]), f, 3);
    }
}

template<int coeffIdx>
void inline filter8_s16x4(const int16x4_t *s, const int32x4_t c, int32x4_t &d)
{
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        d = vsubl_s16(s[6], s[0]);
        d = vaddq_s32(d, c);
        d = vmlal_n_s16(d, s[1], 4);
        d = vmlsl_n_s16(d, s[2], 10);
        d = vmlal_n_s16(d, s[3], 58);
        d = vmlal_n_s16(d, s[4], 17);
        d = vmlsl_n_s16(d, s[5], 5);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        int32x4_t t0 = vaddl_s16(s[3], s[4]);
        int32x4_t t1 = vaddl_s16(s[2], s[5]);
        int32x4_t t2 = vaddl_s16(s[1], s[6]);
        int32x4_t t3 = vaddl_s16(s[0], s[7]);

        d = vmlaq_n_s32(c, t0, 40);
        d = vmlaq_n_s32(d, t1, -11);
        d = vmlaq_n_s32(d, t2, 4);
        d = vmlaq_n_s32(d, t3, -1);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        d = vsubl_s16(s[1], s[7]);
        d = vaddq_s32(d, c);
        d = vmlal_n_s16(d, s[6], 4);
        d = vmlsl_n_s16(d, s[5], 10);
        d = vmlal_n_s16(d, s[4], 58);
        d = vmlal_n_s16(d, s[3], 17);
        d = vmlsl_n_s16(d, s[2], 5);
    }
}

template<int coeffIdx>
void inline filter8_s16x8(const int16x8_t *s, const int32x4_t c, int32x4_t &d0,
                          int32x4_t &d1)
{
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        d0 = vsubl_s16(vget_low_s16(s[6]), vget_low_s16(s[0]));
        d0 = vaddq_s32(d0, c);
        d0 = vmlal_n_s16(d0, vget_low_s16(s[1]), 4);
        d0 = vmlsl_n_s16(d0, vget_low_s16(s[2]), 10);
        d0 = vmlal_n_s16(d0, vget_low_s16(s[3]), 58);
        d0 = vmlal_n_s16(d0, vget_low_s16(s[4]), 17);
        d0 = vmlsl_n_s16(d0, vget_low_s16(s[5]), 5);

        d1 = vsubl_s16(vget_high_s16(s[6]), vget_high_s16(s[0]));
        d1 = vaddq_s32(d1, c);
        d1 = vmlal_n_s16(d1, vget_high_s16(s[1]), 4);
        d1 = vmlsl_n_s16(d1, vget_high_s16(s[2]), 10);
        d1 = vmlal_n_s16(d1, vget_high_s16(s[3]), 58);
        d1 = vmlal_n_s16(d1, vget_high_s16(s[4]), 17);
        d1 = vmlsl_n_s16(d1, vget_high_s16(s[5]), 5);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        int32x4_t t0 = vaddl_s16(vget_low_s16(s[3]), vget_low_s16(s[4]));
        int32x4_t t1 = vaddl_s16(vget_low_s16(s[2]), vget_low_s16(s[5]));
        int32x4_t t2 = vaddl_s16(vget_low_s16(s[1]), vget_low_s16(s[6]));
        int32x4_t t3 = vaddl_s16(vget_low_s16(s[0]), vget_low_s16(s[7]));

        d0 = vmlaq_n_s32(c, t0, 40);
        d0 = vmlaq_n_s32(d0, t1, -11);
        d0 = vmlaq_n_s32(d0, t2, 4);
        d0 = vmlaq_n_s32(d0, t3, -1);

        int32x4_t t4 = vaddl_s16(vget_high_s16(s[3]), vget_high_s16(s[4]));
        int32x4_t t5 = vaddl_s16(vget_high_s16(s[2]), vget_high_s16(s[5]));
        int32x4_t t6 = vaddl_s16(vget_high_s16(s[1]), vget_high_s16(s[6]));
        int32x4_t t7 = vaddl_s16(vget_high_s16(s[0]), vget_high_s16(s[7]));

        d1 = vmlaq_n_s32(c, t4, 40);
        d1 = vmlaq_n_s32(d1, t5, -11);
        d1 = vmlaq_n_s32(d1, t6, 4);
        d1 = vmlaq_n_s32(d1, t7, -1);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        d0 = vsubl_s16(vget_low_s16(s[1]), vget_low_s16(s[7]));
        d0 = vaddq_s32(d0, c);
        d0 = vmlal_n_s16(d0, vget_low_s16(s[6]), 4);
        d0 = vmlsl_n_s16(d0, vget_low_s16(s[5]), 10);
        d0 = vmlal_n_s16(d0, vget_low_s16(s[4]), 58);
        d0 = vmlal_n_s16(d0, vget_low_s16(s[3]), 17);
        d0 = vmlsl_n_s16(d0, vget_low_s16(s[2]), 5);

        d1 = vsubl_s16(vget_high_s16(s[1]), vget_high_s16(s[7]));
        d1 = vaddq_s32(d1, c);
        d1 = vmlal_n_s16(d1, vget_high_s16(s[6]), 4);
        d1 = vmlsl_n_s16(d1, vget_high_s16(s[5]), 10);
        d1 = vmlal_n_s16(d1, vget_high_s16(s[4]), 58);
        d1 = vmlal_n_s16(d1, vget_high_s16(s[3]), 17);
        d1 = vmlsl_n_s16(d1, vget_high_s16(s[2]), 5);
    }
}

template<int width, int height>
void interp4_vert_ss_neon(const int16_t *src, intptr_t srcStride, int16_t *dst,
                          intptr_t dstStride, int coeffIdx)
{
    const int N_TAPS = 4;
    src -= (N_TAPS / 2 - 1) * srcStride;

    const int16x4_t filter = vld1_s16(X265_NS::g_chromaFilter[coeffIdx]);

    // Zero constant in order to use filter helper functions (optimised away).
    const int32x4_t c = vdupq_n_s32(0);

    if (width == 12)
    {
        const int16_t *s = src;
        int16_t *d = dst;

        int16x8_t in[7];
        load_s16x8xn<3>(s, srcStride, in);
        s += 3 * srcStride;

        for (int row = 0; (row + 4) <= height; row += 4)
        {
            load_s16x8xn<4>(s, srcStride, in + 3);

            int32x4_t sum_lo[4];
            int32x4_t sum_hi[4];
            filter4_s16x8(coeffIdx, in + 0, filter, c, sum_lo[0], sum_hi[0]);
            filter4_s16x8(coeffIdx, in + 1, filter, c, sum_lo[1], sum_hi[1]);
            filter4_s16x8(coeffIdx, in + 2, filter, c, sum_lo[2], sum_hi[2]);
            filter4_s16x8(coeffIdx, in + 3, filter, c, sum_lo[3], sum_hi[3]);

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

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        src += 8;
        dst += 8;
        s = src;
        d = dst;

        load_s16x8xn<3>(s, srcStride, in);
        s += 3 * srcStride;

        for (int row = 0; (row + 4) <= height; row += 4)
        {
            load_s16x8xn<4>(s, srcStride, in + 3);

            int32x4_t sum_lo[4];
            int32x4_t sum_hi[4];
            filter4_s16x8(coeffIdx, in + 0, filter, c, sum_lo[0], sum_hi[0]);
            filter4_s16x8(coeffIdx, in + 1, filter, c, sum_lo[1], sum_hi[1]);
            filter4_s16x8(coeffIdx, in + 2, filter, c, sum_lo[2], sum_hi[2]);
            filter4_s16x8(coeffIdx, in + 3, filter, c, sum_lo[3], sum_hi[3]);

            int16x8_t sum[4];
            sum[0] = vcombine_s16(vshrn_n_s32(sum_lo[0], IF_FILTER_PREC),
                                  vshrn_n_s32(sum_hi[0], IF_FILTER_PREC));
            sum[1] = vcombine_s16(vshrn_n_s32(sum_lo[1], IF_FILTER_PREC),
                                  vshrn_n_s32(sum_hi[1], IF_FILTER_PREC));
            sum[2] = vcombine_s16(vshrn_n_s32(sum_lo[2], IF_FILTER_PREC),
                                  vshrn_n_s32(sum_hi[2], IF_FILTER_PREC));
            sum[3] = vcombine_s16(vshrn_n_s32(sum_lo[3], IF_FILTER_PREC),
                                  vshrn_n_s32(sum_hi[3], IF_FILTER_PREC));

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
            const int16_t *s = src;
            int16_t *d = dst;

            int16x8_t in[7];
            load_s16x8xn<3>(s, srcStride, in);
            s += 3 * srcStride;

            for (int row = 0; (row + 4) <= height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 3);

                int32x4_t sum_lo[4];
                int32x4_t sum_hi[4];
                filter4_s16x8(coeffIdx, in + 0, filter, c, sum_lo[0],
                              sum_hi[0]);
                filter4_s16x8(coeffIdx, in + 1, filter, c, sum_lo[1],
                              sum_hi[1]);
                filter4_s16x8(coeffIdx, in + 2, filter, c, sum_lo[2],
                              sum_hi[2]);
                filter4_s16x8(coeffIdx, in + 3, filter, c, sum_lo[3],
                              sum_hi[3]);

                int16x8_t sum[4];
                sum[0] = vcombine_s16(vshrn_n_s32(sum_lo[0], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[0], IF_FILTER_PREC));
                sum[1] = vcombine_s16(vshrn_n_s32(sum_lo[1], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[1], IF_FILTER_PREC));
                sum[2] = vcombine_s16(vshrn_n_s32(sum_lo[2], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[2], IF_FILTER_PREC));
                sum[3] = vcombine_s16(vshrn_n_s32(sum_lo[3], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[3], IF_FILTER_PREC));

                store_s16xnxm<n_store, 4>(sum, d, dstStride);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (height & 2)
            {
                load_s16x8xn<2>(s, srcStride, in + 3);

                int32x4_t sum_lo[2];
                int32x4_t sum_hi[2];
                filter4_s16x8(coeffIdx, in + 0, filter, c, sum_lo[0],
                              sum_hi[0]);
                filter4_s16x8(coeffIdx, in + 1, filter, c, sum_lo[1],
                              sum_hi[1]);

                int16x8_t sum[2];
                sum[0] = vcombine_s16(vshrn_n_s32(sum_lo[0], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[0], IF_FILTER_PREC));
                sum[1] = vcombine_s16(vshrn_n_s32(sum_lo[1], IF_FILTER_PREC),
                                      vshrn_n_s32(sum_hi[1], IF_FILTER_PREC));

                store_s16xnxm<n_store, 2>(sum, d, dstStride);
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
                filter8_s16x8<coeffIdx>(in + 0, c, sum_lo[0], sum_hi[0]);
                filter8_s16x8<coeffIdx>(in + 1, c, sum_lo[1], sum_hi[1]);
                filter8_s16x8<coeffIdx>(in + 2, c, sum_lo[2], sum_hi[2]);
                filter8_s16x8<coeffIdx>(in + 3, c, sum_lo[3], sum_hi[3]);

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
            filter8_s16x4<coeffIdx>(in + 0, c, sum[0]);
            filter8_s16x4<coeffIdx>(in + 1, c, sum[1]);
            filter8_s16x4<coeffIdx>(in + 2, c, sum[2]);
            filter8_s16x4<coeffIdx>(in + 3, c, sum[3]);

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
                filter8_s16x8<coeffIdx>(in + 0, c, sum_lo[0], sum_hi[0]);
                filter8_s16x8<coeffIdx>(in + 1, c, sum_lo[1], sum_hi[1]);
                filter8_s16x8<coeffIdx>(in + 2, c, sum_lo[2], sum_hi[2]);
                filter8_s16x8<coeffIdx>(in + 3, c, sum_lo[3], sum_hi[3]);

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

template<int width, int height>
void interp4_vert_sp_neon(const int16_t *src, intptr_t srcStride, uint8_t *dst,
                          intptr_t dstStride, int coeffIdx)
{
    assert(X265_DEPTH == 8);
    const int headRoom = IF_INTERNAL_PREC - X265_DEPTH;
    const int shift = IF_FILTER_PREC + headRoom;
    const int offset = (1 << (shift - 1)) + (IF_INTERNAL_OFFS <<
        IF_FILTER_PREC);

    const int N_TAPS = 4;
    src -= (N_TAPS / 2 - 1) * srcStride;

    const int16x4_t filter = vld1_s16(X265_NS::g_chromaFilter[coeffIdx]);
    const int32x4_t c = vdupq_n_s32(offset);

    if (width == 12)
    {
        const int16_t *s = src;
        uint8_t *d = dst;

        int16x8_t in[7];
        load_s16x8xn<3>(s, srcStride, in);
        s += 3 * srcStride;

        for (int row = 0; (row + 4) <= height; row += 4)
        {
            load_s16x8xn<4>(s, srcStride, in + 3);

            int32x4_t sum_lo[4];
            int32x4_t sum_hi[4];
            filter4_s16x8(coeffIdx, in + 0, filter, c, sum_lo[0], sum_hi[0]);
            filter4_s16x8(coeffIdx, in + 1, filter, c, sum_lo[1], sum_hi[1]);
            filter4_s16x8(coeffIdx, in + 2, filter, c, sum_lo[2], sum_hi[2]);
            filter4_s16x8(coeffIdx, in + 3, filter, c, sum_lo[3], sum_hi[3]);

            int16x8_t sum[4];
            sum[0] = vcombine_s16(vshrn_n_s32(sum_lo[0], shift),
                                  vshrn_n_s32(sum_hi[0], shift));
            sum[1] = vcombine_s16(vshrn_n_s32(sum_lo[1], shift),
                                  vshrn_n_s32(sum_hi[1], shift));
            sum[2] = vcombine_s16(vshrn_n_s32(sum_lo[2], shift),
                                  vshrn_n_s32(sum_hi[2], shift));
            sum[3] = vcombine_s16(vshrn_n_s32(sum_lo[3], shift),
                                  vshrn_n_s32(sum_hi[3], shift));

            uint8x8_t sum_u8[4];
            sum_u8[0] = vqmovun_s16(sum[0]);
            sum_u8[1] = vqmovun_s16(sum[1]);
            sum_u8[2] = vqmovun_s16(sum[2]);
            sum_u8[3] = vqmovun_s16(sum[3]);

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

        load_s16x8xn<3>(s, srcStride, in);
        s += 3 * srcStride;

        for (int row = 0; (row + 4) <= height; row += 4)
        {
            load_s16x8xn<4>(s, srcStride, in + 3);

            int32x4_t sum_lo[4];
            int32x4_t sum_hi[4];
            filter4_s16x8(coeffIdx, in + 0, filter, c, sum_lo[0], sum_hi[0]);
            filter4_s16x8(coeffIdx, in + 1, filter, c, sum_lo[1], sum_hi[1]);
            filter4_s16x8(coeffIdx, in + 2, filter, c, sum_lo[2], sum_hi[2]);
            filter4_s16x8(coeffIdx, in + 3, filter, c, sum_lo[3], sum_hi[3]);

            int16x8_t sum[4];
            sum[0] = vcombine_s16(vshrn_n_s32(sum_lo[0], shift),
                                  vshrn_n_s32(sum_hi[0], shift));
            sum[1] = vcombine_s16(vshrn_n_s32(sum_lo[1], shift),
                                  vshrn_n_s32(sum_hi[1], shift));
            sum[2] = vcombine_s16(vshrn_n_s32(sum_lo[2], shift),
                                  vshrn_n_s32(sum_hi[2], shift));
            sum[3] = vcombine_s16(vshrn_n_s32(sum_lo[3], shift),
                                  vshrn_n_s32(sum_hi[3], shift));

            uint8x8_t sum_u8[4];
            sum_u8[0] = vqmovun_s16(sum[0]);
            sum_u8[1] = vqmovun_s16(sum[1]);
            sum_u8[2] = vqmovun_s16(sum[2]);
            sum_u8[3] = vqmovun_s16(sum[3]);

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
            const int16_t *s = src;
            uint8_t *d = dst;

            int16x8_t in[7];
            load_s16x8xn<3>(s, srcStride, in);
            s += 3 * srcStride;

            for (int row = 0; (row + 4) <= height; row += 4)
            {
                load_s16x8xn<4>(s, srcStride, in + 3);

                int32x4_t sum_lo[4];
                int32x4_t sum_hi[4];
                filter4_s16x8(coeffIdx, in + 0, filter, c, sum_lo[0],
                              sum_hi[0]);
                filter4_s16x8(coeffIdx, in + 1, filter, c, sum_lo[1],
                              sum_hi[1]);
                filter4_s16x8(coeffIdx, in + 2, filter, c, sum_lo[2],
                              sum_hi[2]);
                filter4_s16x8(coeffIdx, in + 3, filter, c, sum_lo[3],
                              sum_hi[3]);

                int16x8_t sum[4];
                sum[0] = vcombine_s16(vshrn_n_s32(sum_lo[0], shift),
                                      vshrn_n_s32(sum_hi[0], shift));
                sum[1] = vcombine_s16(vshrn_n_s32(sum_lo[1], shift),
                                      vshrn_n_s32(sum_hi[1], shift));
                sum[2] = vcombine_s16(vshrn_n_s32(sum_lo[2], shift),
                                      vshrn_n_s32(sum_hi[2], shift));
                sum[3] = vcombine_s16(vshrn_n_s32(sum_lo[3], shift),
                                      vshrn_n_s32(sum_hi[3], shift));

                uint8x8_t sum_u8[4];
                sum_u8[0] = vqmovun_s16(sum[0]);
                sum_u8[1] = vqmovun_s16(sum[1]);
                sum_u8[2] = vqmovun_s16(sum[2]);
                sum_u8[3] = vqmovun_s16(sum[3]);

                store_u8xnxm<n_store, 4>(d, dstStride, sum_u8);

                in[0] = in[4];
                in[1] = in[5];
                in[2] = in[6];

                s += 4 * srcStride;
                d += 4 * dstStride;
            }

            if (height & 2)
            {
                load_s16x8xn<2>(s, srcStride, in + 3);

                int32x4_t sum_lo[2];
                int32x4_t sum_hi[2];
                filter4_s16x8(coeffIdx, in + 0, filter, c, sum_lo[0],
                              sum_hi[0]);
                filter4_s16x8(coeffIdx, in + 1, filter, c, sum_lo[1],
                              sum_hi[1]);

                int16x8_t sum[2];
                sum[0] = vcombine_s16(vshrn_n_s32(sum_lo[0], shift),
                                      vshrn_n_s32(sum_hi[0], shift));
                sum[1] = vcombine_s16(vshrn_n_s32(sum_lo[1], shift),
                                      vshrn_n_s32(sum_hi[1], shift));

                uint8x8_t sum_u8[2];
                sum_u8[0] = vqmovun_s16(sum[0]);
                sum_u8[1] = vqmovun_s16(sum[1]);

                store_u8xnxm<n_store, 2>(d, dstStride, sum_u8);
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
    const int offset = (1 << (shift - 1)) + (IF_INTERNAL_OFFS <<
        IF_FILTER_PREC);

    const int N_TAPS = 8;
    src -= (N_TAPS / 2 - 1) * srcStride;

    const int32x4_t c = vdupq_n_s32(offset);

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

                int32x4_t sum_lo[4];
                int32x4_t sum_hi[4];
                filter8_s16x8<coeffIdx>(in + 0, c, sum_lo[0], sum_hi[0]);
                filter8_s16x8<coeffIdx>(in + 1, c, sum_lo[1], sum_hi[1]);
                filter8_s16x8<coeffIdx>(in + 2, c, sum_lo[2], sum_hi[2]);
                filter8_s16x8<coeffIdx>(in + 3, c, sum_lo[3], sum_hi[3]);

                int16x8_t sum[4];
                sum[0] = vcombine_s16(vshrn_n_s32(sum_lo[0], shift),
                                      vshrn_n_s32(sum_hi[0], shift));
                sum[1] = vcombine_s16(vshrn_n_s32(sum_lo[1], shift),
                                      vshrn_n_s32(sum_hi[1], shift));
                sum[2] = vcombine_s16(vshrn_n_s32(sum_lo[2], shift),
                                      vshrn_n_s32(sum_hi[2], shift));
                sum[3] = vcombine_s16(vshrn_n_s32(sum_lo[3], shift),
                                      vshrn_n_s32(sum_hi[3], shift));

                uint8x8_t sum_u8[4];
                sum_u8[0] = vqmovun_s16(sum[0]);
                sum_u8[1] = vqmovun_s16(sum[1]);
                sum_u8[2] = vqmovun_s16(sum[2]);
                sum_u8[3] = vqmovun_s16(sum[3]);

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
            filter8_s16x4<coeffIdx>(in + 0, c, sum[0]);
            filter8_s16x4<coeffIdx>(in + 1, c, sum[1]);
            filter8_s16x4<coeffIdx>(in + 2, c, sum[2]);
            filter8_s16x4<coeffIdx>(in + 3, c, sum[3]);

            int16x4_t sum_s16[4];
            sum_s16[0] = vshrn_n_s32(sum[0], shift);
            sum_s16[1] = vshrn_n_s32(sum[1], shift);
            sum_s16[2] = vshrn_n_s32(sum[2], shift);
            sum_s16[3] = vshrn_n_s32(sum[3], shift);

            uint8x8_t sum_u8[4];
            sum_u8[0] = vqmovun_s16(vcombine_s16(sum_s16[0], vdup_n_s16(0)));
            sum_u8[1] = vqmovun_s16(vcombine_s16(sum_s16[1], vdup_n_s16(0)));
            sum_u8[2] = vqmovun_s16(vcombine_s16(sum_s16[2], vdup_n_s16(0)));
            sum_u8[3] = vqmovun_s16(vcombine_s16(sum_s16[3], vdup_n_s16(0)));

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

                int32x4_t sum_lo[4];
                int32x4_t sum_hi[4];
                filter8_s16x8<coeffIdx>(in + 0, c, sum_lo[0], sum_hi[0]);
                filter8_s16x8<coeffIdx>(in + 1, c, sum_lo[1], sum_hi[1]);
                filter8_s16x8<coeffIdx>(in + 2, c, sum_lo[2], sum_hi[2]);
                filter8_s16x8<coeffIdx>(in + 3, c, sum_lo[3], sum_hi[3]);

                int16x8_t sum[4];
                sum[0] = vcombine_s16(vshrn_n_s32(sum_lo[0], shift),
                                      vshrn_n_s32(sum_hi[0], shift));
                sum[1] = vcombine_s16(vshrn_n_s32(sum_lo[1], shift),
                                      vshrn_n_s32(sum_hi[1], shift));
                sum[2] = vcombine_s16(vshrn_n_s32(sum_lo[2], shift),
                                      vshrn_n_s32(sum_hi[2], shift));
                sum[3] = vcombine_s16(vshrn_n_s32(sum_lo[3], shift),
                                      vshrn_n_s32(sum_hi[3], shift));

                uint8x8_t sum_u8[4];
                sum_u8[0] = vqmovun_s16(sum[0]);
                sum_u8[1] = vqmovun_s16(sum[1]);
                sum_u8[2] = vqmovun_s16(sum[2]);
                sum_u8[3] = vqmovun_s16(sum[3]);

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

#endif // !HIGH_BIT_DEPTH
}

namespace X265_NS
{

#if HIGH_BIT_DEPTH
#define SHIFT_INTERP_PS (IF_FILTER_PREC - (IF_INTERNAL_PREC - X265_DEPTH))
#endif

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

#if HIGH_BIT_DEPTH
template<int N, int width, int height>
void interp_horiz_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst, intptr_t dstStride, int coeffIdx)
{
    const int16_t *coeff = (N == 4) ? g_chromaFilter[coeffIdx] : g_lumaFilter[coeffIdx];
    int headRoom = IF_FILTER_PREC;
    int offset = (1 << (headRoom - 1));
    uint16_t maxVal = (1 << X265_DEPTH) - 1;
    int cStride = 1;

    src -= (N / 2 - 1) * cStride;
    int16x8_t vc = vld1q_s16(coeff);
    int16x4_t low_vc = vget_low_s16(vc);
    int16x4_t high_vc = vget_high_s16(vc);

    const int32x4_t voffset = vdupq_n_s32(offset);
    const int32x4_t vhr = vdupq_n_s32(-headRoom);

    int row, col;
    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col += 8)
        {
            int32x4_t vsum1, vsum2;

            int16x8_t input[N];

            for (int i = 0; i < N; i++)
            {
                input[i] = vreinterpretq_s16_u16(vld1q_u16(src + col + i));
            }
            vsum1 = voffset;
            vsum2 = voffset;

            vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[0]), low_vc, 0);
            vsum2 = vmlal_high_lane_s16(vsum2, input[0], low_vc, 0);

            vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[1]), low_vc, 1);
            vsum2 = vmlal_high_lane_s16(vsum2, input[1], low_vc, 1);

            vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[2]), low_vc, 2);
            vsum2 = vmlal_high_lane_s16(vsum2, input[2], low_vc, 2);

            vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[3]), low_vc, 3);
            vsum2 = vmlal_high_lane_s16(vsum2, input[3], low_vc, 3);

            if (N == 8)
            {
                vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[4]), high_vc, 0);
                vsum2 = vmlal_high_lane_s16(vsum2, input[4], high_vc, 0);
                vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[5]), high_vc, 1);
                vsum2 = vmlal_high_lane_s16(vsum2, input[5], high_vc, 1);
                vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[6]), high_vc, 2);
                vsum2 = vmlal_high_lane_s16(vsum2, input[6], high_vc, 2);
                vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[7]), high_vc, 3);
                vsum2 = vmlal_high_lane_s16(vsum2, input[7], high_vc, 3);

            }

            vsum1 = vshlq_s32(vsum1, vhr);
            vsum2 = vshlq_s32(vsum2, vhr);

            int16x8_t vsum = vuzp1q_s16(vreinterpretq_s16_s32(vsum1),
                                        vreinterpretq_s16_s32(vsum2));
            vsum = vminq_s16(vsum, vdupq_n_s16(maxVal));
            vsum = vmaxq_s16(vsum, vdupq_n_s16(0));
            vst1q_u16(dst + col, vreinterpretq_u16_s16(vsum));
        }

        src += srcStride;
        dst += dstStride;
    }
}

#else // HIGH_BIT_DEPTH
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

#endif // HIGH_BIT_DEPTH

#if HIGH_BIT_DEPTH

template<int N, int width, int height>
void interp_horiz_ps_neon(const uint16_t *src, intptr_t srcStride, int16_t *dst, intptr_t dstStride, int coeffIdx,
                          int isRowExt)
{
    const int16_t *coeff = (N == 4) ? g_chromaFilter[coeffIdx] : g_lumaFilter[coeffIdx];
    const int offset = (unsigned) - IF_INTERNAL_OFFS << SHIFT_INTERP_PS;

    int blkheight = height;
    src -= N / 2 - 1;

    if (isRowExt)
    {
        src -= (N / 2 - 1) * srcStride;
        blkheight += N - 1;
    }
    int16x8_t vc3 = vld1q_s16(coeff);
    const int32x4_t voffset = vdupq_n_s32(offset);

    int row, col;
    for (row = 0; row < blkheight; row++)
    {
        for (col = 0; col < width; col += 8)
        {
            int32x4_t vsum, vsum2;

            int16x8_t input[N];
            for (int i = 0; i < N; i++)
            {
                input[i] = vreinterpretq_s16_u16(vld1q_u16(src + col + i));
            }

            vsum = voffset;
            vsum2 = voffset;

            vsum = vmlal_lane_s16(vsum, vget_low_s16(input[0]),
                                  vget_low_s16(vc3), 0);
            vsum2 = vmlal_high_lane_s16(vsum2, input[0], vget_low_s16(vc3), 0);

            vsum = vmlal_lane_s16(vsum, vget_low_s16(input[1]),
                                  vget_low_s16(vc3), 1);
            vsum2 = vmlal_high_lane_s16(vsum2, input[1], vget_low_s16(vc3), 1);

            vsum = vmlal_lane_s16(vsum, vget_low_s16(input[2]),
                                  vget_low_s16(vc3), 2);
            vsum2 = vmlal_high_lane_s16(vsum2, input[2], vget_low_s16(vc3), 2);

            vsum = vmlal_lane_s16(vsum, vget_low_s16(input[3]),
                                  vget_low_s16(vc3), 3);
            vsum2 = vmlal_high_lane_s16(vsum2, input[3], vget_low_s16(vc3), 3);

            if (N == 8)
            {
                vsum = vmlal_lane_s16(vsum, vget_low_s16(input[4]), vget_high_s16(vc3), 0);
                vsum2 = vmlal_high_lane_s16(vsum2, input[4], vget_high_s16(vc3), 0);

                vsum = vmlal_lane_s16(vsum, vget_low_s16(input[5]), vget_high_s16(vc3), 1);
                vsum2 = vmlal_high_lane_s16(vsum2, input[5], vget_high_s16(vc3), 1);

                vsum = vmlal_lane_s16(vsum, vget_low_s16(input[6]), vget_high_s16(vc3), 2);
                vsum2 = vmlal_high_lane_s16(vsum2, input[6], vget_high_s16(vc3), 2);

                vsum = vmlal_lane_s16(vsum, vget_low_s16(input[7]), vget_high_s16(vc3), 3);
                vsum2 = vmlal_high_lane_s16(vsum2, input[7], vget_high_s16(vc3), 3);
            }

            int16x4_t res_lo = vshrn_n_s32(vsum, SHIFT_INTERP_PS);
            int16x4_t res_hi = vshrn_n_s32(vsum2, SHIFT_INTERP_PS);
            vst1q_s16(dst + col, vcombine_s16(res_lo, res_hi));
        }

        src += srcStride;
        dst += dstStride;
    }
}

#else // HIGH_BIT_DEPTH
template<int N, int width, int height>
void interp_horiz_ps_neon(const uint8_t *src, intptr_t srcStride, int16_t *dst, intptr_t dstStride, int coeffIdx,
                          int isRowExt)
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

#endif // HIGH_BIT_DEPTH

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
        return interp4_vert_ss_neon<width, height>(src, srcStride, dst,
                                                   dstStride, coeffIdx);
    }
}

#if HIGH_BIT_DEPTH

template<int N, int width, int height>
void interp_vert_pp_neon(const uint16_t *src, intptr_t srcStride, uint16_t *dst, intptr_t dstStride, int coeffIdx)
{

    const int16_t *c = (N == 4) ? g_chromaFilter[coeffIdx] : g_lumaFilter[coeffIdx];
    int offset = 1 << (IF_FILTER_PREC - 1);
    const uint16_t maxVal = (1 << X265_DEPTH) - 1;

    src -= (N / 2 - 1) * srcStride;
    int16x8_t vc = vld1q_s16(c);
    int32x4_t low_vc = vmovl_s16(vget_low_s16(vc));
    int32x4_t high_vc = vmovl_s16(vget_high_s16(vc));

    const int32x4_t voffset = vdupq_n_s32(offset);

    int row, col;
    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col += 4)
        {
            int32x4_t vsum;

            int32x4_t input[N];

            for (int i = 0; i < N; i++)
            {
                uint16x4_t in_tmp = vld1_u16(src + col + i * srcStride);
                input[i] = vreinterpretq_s32_u32(vmovl_u16(in_tmp));
            }
            vsum = voffset;

            vsum = vmlaq_laneq_s32(vsum, (input[0]), low_vc, 0);
            vsum = vmlaq_laneq_s32(vsum, (input[1]), low_vc, 1);
            vsum = vmlaq_laneq_s32(vsum, (input[2]), low_vc, 2);
            vsum = vmlaq_laneq_s32(vsum, (input[3]), low_vc, 3);

            if (N == 8)
            {
                vsum = vmlaq_laneq_s32(vsum, (input[4]), high_vc, 0);
                vsum = vmlaq_laneq_s32(vsum, (input[5]), high_vc, 1);
                vsum = vmlaq_laneq_s32(vsum, (input[6]), high_vc, 2);
                vsum = vmlaq_laneq_s32(vsum, (input[7]), high_vc, 3);
            }

            uint16x4_t res = vqshrun_n_s32(vsum, IF_FILTER_PREC);
            res = vmin_u16(res, vdup_n_u16(maxVal));
            vst1_u16(dst + col, res);
        }
        src += srcStride;
        dst += dstStride;
    }
}




#else

template<int N, int width, int height>
void interp_vert_pp_neon(const uint8_t *src, intptr_t srcStride, uint8_t *dst, intptr_t dstStride, int coeffIdx)
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

#endif


#if HIGH_BIT_DEPTH

template<int N, int width, int height>
void interp_vert_ps_neon(const uint16_t *src, intptr_t srcStride, int16_t *dst, intptr_t dstStride, int coeffIdx)
{
    const int16_t *c = (N == 4) ? g_chromaFilter[coeffIdx] : g_lumaFilter[coeffIdx];
    int offset = (unsigned) - IF_INTERNAL_OFFS << SHIFT_INTERP_PS;
    src -= (N / 2 - 1) * srcStride;

    int16x8_t vc = vld1q_s16(c);
    int32x4_t low_vc = vmovl_s16(vget_low_s16(vc));
    int32x4_t high_vc = vmovl_s16(vget_high_s16(vc));

    const int32x4_t voffset = vdupq_n_s32(offset);

    int row, col;
    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col += 4)
        {
            int32x4_t vsum;

            int32x4_t input[N];

            for (int i = 0; i < N; i++)
            {
                uint16x4_t in_tmp = vld1_u16(src + col + i * srcStride);
                input[i] = vreinterpretq_s32_u32(vmovl_u16(in_tmp));
            }
            vsum = voffset;

            vsum = vmlaq_laneq_s32(vsum, input[0], low_vc, 0);
            vsum = vmlaq_laneq_s32(vsum, input[1], low_vc, 1);
            vsum = vmlaq_laneq_s32(vsum, input[2], low_vc, 2);
            vsum = vmlaq_laneq_s32(vsum, input[3], low_vc, 3);

            if (N == 8)
            {
                int32x4_t vsum1 = vmulq_laneq_s32(input[4], high_vc, 0);
                vsum1 = vmlaq_laneq_s32(vsum1, input[5], high_vc, 1);
                vsum1 = vmlaq_laneq_s32(vsum1, input[6], high_vc, 2);
                vsum1 = vmlaq_laneq_s32(vsum1, input[7], high_vc, 3);
                vsum = vaddq_s32(vsum, vsum1);
            }

            vst1_s16(dst + col, vshrn_n_s32(vsum, SHIFT_INTERP_PS));
        }

        src += srcStride;
        dst += dstStride;
    }
}

#else

template<int N, int width, int height>
void interp_vert_ps_neon(const uint8_t *src, intptr_t srcStride, int16_t *dst, intptr_t dstStride, int coeffIdx)
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

#endif



#if HIGH_BIT_DEPTH
template<int N, int width, int height>
void interp_vert_sp_neon(const int16_t *src, intptr_t srcStride, pixel *dst, intptr_t dstStride, int coeffIdx)
{
    int headRoom = IF_INTERNAL_PREC - X265_DEPTH;
    int shift = IF_FILTER_PREC + headRoom;
    int offset = (1 << (shift - 1)) + (IF_INTERNAL_OFFS << IF_FILTER_PREC);
    uint16_t maxVal = (1 << X265_DEPTH) - 1;
    const int16_t *coeff = (N == 8 ? g_lumaFilter[coeffIdx] : g_chromaFilter[coeffIdx]);

    src -= (N / 2 - 1) * srcStride;

    int16x8_t vc = vld1q_s16(coeff);
    int16x4_t low_vc = vget_low_s16(vc);
    int16x4_t high_vc = vget_high_s16(vc);

    const int32x4_t voffset = vdupq_n_s32(offset);
    const int32x4_t vhr = vdupq_n_s32(-shift);

    int row, col;
    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col += 8)
        {
            int32x4_t vsum1, vsum2;

            int16x8_t input[N];

            for (int i = 0; i < N; i++)
            {
                input[i] = vld1q_s16(src + col + i * srcStride);
            }
            vsum1 = voffset;
            vsum2 = voffset;

            vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[0]), low_vc, 0);
            vsum2 = vmlal_high_lane_s16(vsum2, input[0], low_vc, 0);

            vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[1]), low_vc, 1);
            vsum2 = vmlal_high_lane_s16(vsum2, input[1], low_vc, 1);

            vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[2]), low_vc, 2);
            vsum2 = vmlal_high_lane_s16(vsum2, input[2], low_vc, 2);

            vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[3]), low_vc, 3);
            vsum2 = vmlal_high_lane_s16(vsum2, input[3], low_vc, 3);

            if (N == 8)
            {
                vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[4]), high_vc, 0);
                vsum2 = vmlal_high_lane_s16(vsum2, input[4], high_vc, 0);

                vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[5]), high_vc, 1);
                vsum2 = vmlal_high_lane_s16(vsum2, input[5], high_vc, 1);

                vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[6]), high_vc, 2);
                vsum2 = vmlal_high_lane_s16(vsum2, input[6], high_vc, 2);

                vsum1 = vmlal_lane_s16(vsum1, vget_low_s16(input[7]), high_vc, 3);
                vsum2 = vmlal_high_lane_s16(vsum2, input[7], high_vc, 3);
            }

            vsum1 = vshlq_s32(vsum1, vhr);
            vsum2 = vshlq_s32(vsum2, vhr);

            int16x8_t vsum = vuzp1q_s16(vreinterpretq_s16_s32(vsum1),
                                        vreinterpretq_s16_s32(vsum2));
            vsum = vminq_s16(vsum, vdupq_n_s16(maxVal));
            vsum = vmaxq_s16(vsum, vdupq_n_s16(0));
            vst1q_u16(dst + col, vreinterpretq_u16_s16(vsum));
        }

        src += srcStride;
        dst += dstStride;
    }
}

#else // if HIGH_BIT_DEPTH

template<int N, int width, int height>
void interp_vert_sp_neon(const int16_t *src, intptr_t srcStride, uint8_t *dst,
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
        return interp4_vert_sp_neon<width, height>(src, srcStride, dst,
                                                   dstStride, coeffIdx);
    }
}

#endif // if HIGH_BIT_DEPTH

template<int N, int width, int height>
void interp_hv_pp_neon(const pixel *src, intptr_t srcStride, pixel *dst, intptr_t dstStride, int idxX, int idxY)
{
    ALIGN_VAR_32(int16_t, immed[width * (height + N - 1)]);

    interp_horiz_ps_neon<N, width, height>(src, srcStride, immed, width, idxX, 1);
    interp_vert_sp_neon<N, width, height>(immed + (N / 2 - 1) * width, width, dst, dstStride, idxY);
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
#if !HIGH_BIT_DEPTH
    LUMA(4, 4);
    LUMA(4, 8);
    LUMA(4, 16);
    LUMA(12, 16);
#endif
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

#if !HIGH_BIT_DEPTH
    CHROMA_420(2, 4);
    CHROMA_420(2, 8);
    CHROMA_420(4, 2);
    CHROMA_420(4, 4);
    CHROMA_420(4, 8);
    CHROMA_420(4, 16);
    CHROMA_420(6, 8);
    CHROMA_420(12, 16);
#endif
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

#if !HIGH_BIT_DEPTH
    CHROMA_422(2, 8);
    CHROMA_422(2, 16);
    CHROMA_422(4, 4);
    CHROMA_422(4, 8);
    CHROMA_422(4, 16);
    CHROMA_422(4, 32);
    CHROMA_422(6, 16);
    CHROMA_422(12, 32);
#endif
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

#if !HIGH_BIT_DEPTH
    CHROMA_444(4, 4);
    CHROMA_444(4, 8);
    CHROMA_444(4, 16);
    CHROMA_444(12, 16);
#endif
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


