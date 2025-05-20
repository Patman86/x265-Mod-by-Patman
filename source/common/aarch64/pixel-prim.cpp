#include "common.h"
#include "slicetype.h"      // LOWRES_COST_MASK
#include "primitives.h"
#include "x265.h"

#include "pixel-prim.h"
#include "arm64-utils.h"
#if HAVE_NEON

#include "mem-neon.h"

#include <arm_neon.h>

using namespace X265_NS;



namespace
{


static inline void sumsubq_s16(int16x8_t *sum, int16x8_t *sub, const int16x8_t a, const int16x8_t b)
{
    *sum = vaddq_s16(a, b);
    *sub = vsubq_s16(a, b);
}

static inline void transpose_s16_s16x2(int16x8_t *t1, int16x8_t *t2,
                                       const int16x8_t s1, const int16x8_t s2)
{
    *t1 = vtrn1q_s16(s1, s2);
    *t2 = vtrn2q_s16(s1, s2);
}

static inline void transpose_s16_s32x2(int16x8_t *t1, int16x8_t *t2,
                                       const int16x8_t s1, const int16x8_t s2)
{
    int32x4_t tmp1 = vreinterpretq_s32_s16(s1);
    int32x4_t tmp2 = vreinterpretq_s32_s16(s2);

    *t1 = vreinterpretq_s16_s32(vtrn1q_s32(tmp1, tmp2));
    *t2 = vreinterpretq_s16_s32(vtrn2q_s32(tmp1, tmp2));
}

static inline void transpose_s16_s64x2(int16x8_t *t1, int16x8_t *t2,
                                       const int16x8_t s1, const int16x8_t s2)
{
    int64x2_t tmp1 = vreinterpretq_s64_s16(s1);
    int64x2_t tmp2 = vreinterpretq_s64_s16(s2);

    *t1 = vreinterpretq_s16_s64(vtrn1q_s64(tmp1, tmp2));
    *t2 = vreinterpretq_s16_s64(vtrn2q_s64(tmp1, tmp2));
}

static inline uint16x8_t max_abs_s16(const int16x8_t a, const int16x8_t b)
{
    uint16x8_t abs0 = vreinterpretq_u16_s16(vabsq_s16(a));
    uint16x8_t abs1 = vreinterpretq_u16_s16(vabsq_s16(b));

    return vmaxq_u16(abs0, abs1);
}

#if X265_DEPTH == 12
static inline void sumsubq_s32(int32x4_t *sum, int32x4_t *sub, const int32x4_t a, const int32x4_t b)
{
    *sum = vaddq_s32(a, b);
    *sub = vsubq_s32(a, b);
}

static inline void sumsublq_s16(int32x4_t *sum_lo, int32x4_t *sum_hi,
                                int32x4_t *sub_lo, int32x4_t *sub_hi,
                                const int16x8_t a, const int16x8_t b)
{
    *sum_lo = vaddl_s16(vget_low_s16(a), vget_low_s16(b));
    *sub_lo = vsubl_s16(vget_low_s16(a), vget_low_s16(b));
    *sum_hi = vaddl_s16(vget_high_s16(a), vget_high_s16(b));
    *sub_hi = vsubl_s16(vget_high_s16(a), vget_high_s16(b));
}

static inline void transpose_inplace_s32_s64x2(int32x4_t *t1, int32x4_t *t2)
{
    int64x2_t tmp1 = vreinterpretq_s64_s32(*t1);
    int64x2_t tmp2 = vreinterpretq_s64_s32(*t2);

    *t1 = vreinterpretq_s32_s64(vtrn1q_s64(tmp1, tmp2));
    *t2 = vreinterpretq_s32_s64(vtrn2q_s64(tmp1, tmp2));
}

static inline uint32x4_t max_abs_s32(int32x4_t a, int32x4_t b)
{
    uint32x4_t abs0 = vreinterpretq_u32_s32(vabsq_s32(a));
    uint32x4_t abs1 = vreinterpretq_u32_s32(vabsq_s32(b));

    return vmaxq_u32(abs0, abs1);
}

#endif // X265_DEPTH == 12

#if HIGH_BIT_DEPTH
static inline void load_diff_u16x8x4(const uint16_t *pix1, intptr_t stride_pix1,
                                     const uint16_t *pix2, intptr_t stride_pix2, int16x8_t diff[4])
{
    uint16x8_t r[4], t[4];
    load_u16x8xn<4>(pix1, stride_pix1, r);
    load_u16x8xn<4>(pix2, stride_pix2, t);

    diff[0] = vreinterpretq_s16_u16(vsubq_u16(r[0], t[0]));
    diff[1] = vreinterpretq_s16_u16(vsubq_u16(r[1], t[1]));
    diff[2] = vreinterpretq_s16_u16(vsubq_u16(r[2], t[2]));
    diff[3] = vreinterpretq_s16_u16(vsubq_u16(r[3], t[3]));
}

static inline void load_diff_u16x8x4_dual(const uint16_t *pix1, intptr_t stride_pix1,
                                          const uint16_t *pix2, intptr_t stride_pix2, int16x8_t diff[8])
{
    load_diff_u16x8x4(pix1, stride_pix1, pix2, stride_pix2, diff);
    load_diff_u16x8x4(pix1 + 4 * stride_pix1, stride_pix1,
                      pix2 + 4 * stride_pix2, stride_pix2, diff + 4);
}

static inline void load_diff_u16x8x8(const uint16_t *pix1, intptr_t stride_pix1,
                                     const uint16_t *pix2, intptr_t stride_pix2, int16x8_t diff[8])
{
    uint16x8_t r[8], t[8];
    load_u16x8xn<8>(pix1, stride_pix1, r);
    load_u16x8xn<8>(pix2, stride_pix2, t);

    diff[0] = vreinterpretq_s16_u16(vsubq_u16(r[0], t[0]));
    diff[1] = vreinterpretq_s16_u16(vsubq_u16(r[1], t[1]));
    diff[2] = vreinterpretq_s16_u16(vsubq_u16(r[2], t[2]));
    diff[3] = vreinterpretq_s16_u16(vsubq_u16(r[3], t[3]));
    diff[4] = vreinterpretq_s16_u16(vsubq_u16(r[4], t[4]));
    diff[5] = vreinterpretq_s16_u16(vsubq_u16(r[5], t[5]));
    diff[6] = vreinterpretq_s16_u16(vsubq_u16(r[6], t[6]));
    diff[7] = vreinterpretq_s16_u16(vsubq_u16(r[7], t[7]));
}

#else // !HIGH_BIT_DEPTH
static inline void load_diff_u8x8x4(const uint8_t *pix1, intptr_t stride_pix1,
                                    const uint8_t *pix2, intptr_t stride_pix2, int16x8_t diff[4])
{
    uint8x8_t r[4], t[4];
    load_u8x8xn<4>(pix1, stride_pix1, r);
    load_u8x8xn<4>(pix2, stride_pix2, t);

    diff[0] = vreinterpretq_s16_u16(vsubl_u8(r[0], t[0]));
    diff[1] = vreinterpretq_s16_u16(vsubl_u8(r[1], t[1]));
    diff[2] = vreinterpretq_s16_u16(vsubl_u8(r[2], t[2]));
    diff[3] = vreinterpretq_s16_u16(vsubl_u8(r[3], t[3]));
}

static inline void load_diff_u8x8x8(const uint8_t *pix1, intptr_t stride_pix1,
                                    const uint8_t *pix2, intptr_t stride_pix2, int16x8_t diff[8])
{
    load_diff_u8x8x4(pix1, stride_pix1, pix2, stride_pix2, diff);
    load_diff_u8x8x4(pix1 + 4 * stride_pix1, stride_pix1,
                     pix2 + 4 * stride_pix2, stride_pix2, diff + 4);
}

static inline void load_diff_u8x16x4(const uint8_t *pix1, intptr_t stride_pix1,
                                     const uint8_t *pix2, intptr_t stride_pix2, int16x8_t diff[8])
{
    uint8x16_t s1[4], s2[4];
    load_u8x16xn<4>(pix1, stride_pix1, s1);
    load_u8x16xn<4>(pix2, stride_pix2, s2);

    diff[0] = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s1[0]), vget_low_u8(s2[0])));
    diff[1] = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s1[1]), vget_low_u8(s2[1])));
    diff[2] = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s1[2]), vget_low_u8(s2[2])));
    diff[3] = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s1[3]), vget_low_u8(s2[3])));
    diff[4] = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s1[0]), vget_high_u8(s2[0])));
    diff[5] = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s1[1]), vget_high_u8(s2[1])));
    diff[6] = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s1[2]), vget_high_u8(s2[2])));
    diff[7] = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s1[3]), vget_high_u8(s2[3])));
}

#endif // HIGH_BIT_DEPTH

// 4 way hadamard vertical pass.
static inline void hadamard_4_v(const int16x8_t in_coefs[4], int16x8_t out_coefs[4])
{
    int16x8_t s0, s1, d0, d1;

    sumsubq_s16(&s0, &d0, in_coefs[0], in_coefs[1]);
    sumsubq_s16(&s1, &d1, in_coefs[2], in_coefs[3]);

    sumsubq_s16(&out_coefs[0], &out_coefs[2], s0, s1);
    sumsubq_s16(&out_coefs[1], &out_coefs[3], d0, d1);
}

// 8 way hadamard vertical pass.
static inline void hadamard_8_v(const int16x8_t in_coefs[8], int16x8_t out_coefs[8])
{
    int16x8_t temp[8];

    hadamard_4_v(in_coefs, temp);
    hadamard_4_v(in_coefs + 4, temp + 4);

    sumsubq_s16(&out_coefs[0], &out_coefs[4], temp[0], temp[4]);
    sumsubq_s16(&out_coefs[1], &out_coefs[5], temp[1], temp[5]);
    sumsubq_s16(&out_coefs[2], &out_coefs[6], temp[2], temp[6]);
    sumsubq_s16(&out_coefs[3], &out_coefs[7], temp[3], temp[7]);
}

// 4 way hadamard horizontal pass.
static inline void hadamard_4_h(const int16x8_t in_coefs[4], int16x8_t out_coefs[4])
{
    int16x8_t s0, s1, d0, d1, t0, t1, t2, t3;

    transpose_s16_s16x2(&t0, &t1, in_coefs[0], in_coefs[1]);
    transpose_s16_s16x2(&t2, &t3, in_coefs[2], in_coefs[3]);

    sumsubq_s16(&s0, &d0, t0, t1);
    sumsubq_s16(&s1, &d1, t2, t3);

    transpose_s16_s32x2(&out_coefs[0], &out_coefs[1], s0, s1);
    transpose_s16_s32x2(&out_coefs[2], &out_coefs[3], d0, d1);
}

#if X265_DEPTH != 12
// 8 way hadamard horizontal pass.
static inline void hadamard_8_h(int16x8_t coefs[8], uint16x8_t out[4])
{
    int16x8_t s0, s1, s2, s3, d0, d1, d2, d3;
    int16x8_t temp[8];

    hadamard_4_h(coefs, temp);
    hadamard_4_h(coefs + 4, temp + 4);

    sumsubq_s16(&s0, &d0, temp[0], temp[1]);
    sumsubq_s16(&s1, &d1, temp[2], temp[3]);
    sumsubq_s16(&s2, &d2, temp[4], temp[5]);
    sumsubq_s16(&s3, &d3, temp[6], temp[7]);

    transpose_s16_s64x2(&temp[0], &temp[1], s0, s2);
    transpose_s16_s64x2(&temp[2], &temp[3], s1, s3);
    transpose_s16_s64x2(&temp[4], &temp[5], d0, d2);
    transpose_s16_s64x2(&temp[6], &temp[7], d1, d3);

    out[0] = max_abs_s16(temp[0], temp[1]);
    out[1] = max_abs_s16(temp[2], temp[3]);
    out[2] = max_abs_s16(temp[4], temp[5]);
    out[3] = max_abs_s16(temp[6], temp[7]);
}

#else // X265_DEPTH == 12
static inline void hadamard_8_h(int16x8_t coefs[8], uint32x4_t out[4])
{
    int16x8_t a[8];

    transpose_s16_s16x2(&a[0], &a[1], coefs[0], coefs[1]);
    transpose_s16_s16x2(&a[2], &a[3], coefs[2], coefs[3]);
    transpose_s16_s16x2(&a[4], &a[5], coefs[4], coefs[5]);
    transpose_s16_s16x2(&a[6], &a[7], coefs[6], coefs[7]);

    int32x4_t a_lo[8], a_hi[8], b_lo[8], b_hi[8];

    sumsublq_s16(&a_lo[0], &a_hi[0], &a_lo[4], &a_hi[4], a[0], a[1]);
    sumsublq_s16(&a_lo[1], &a_hi[1], &a_lo[5], &a_hi[5], a[2], a[3]);
    sumsublq_s16(&a_lo[2], &a_hi[2], &a_lo[6], &a_hi[6], a[4], a[5]);
    sumsublq_s16(&a_lo[3], &a_hi[3], &a_lo[7], &a_hi[7], a[6], a[7]);

    transpose_inplace_s32_s64x2(&a_lo[0], &a_lo[1]);
    transpose_inplace_s32_s64x2(&a_lo[2], &a_lo[3]);
    transpose_inplace_s32_s64x2(&a_lo[4], &a_lo[5]);
    transpose_inplace_s32_s64x2(&a_lo[6], &a_lo[7]);

    transpose_inplace_s32_s64x2(&a_hi[0], &a_hi[1]);
    transpose_inplace_s32_s64x2(&a_hi[2], &a_hi[3]);
    transpose_inplace_s32_s64x2(&a_hi[4], &a_hi[5]);
    transpose_inplace_s32_s64x2(&a_hi[6], &a_hi[7]);

    sumsubq_s32(&b_lo[0], &b_lo[1], a_lo[0], a_lo[1]);
    sumsubq_s32(&b_lo[2], &b_lo[3], a_lo[2], a_lo[3]);
    sumsubq_s32(&b_lo[4], &b_lo[5], a_lo[4], a_lo[5]);
    sumsubq_s32(&b_lo[6], &b_lo[7], a_lo[6], a_lo[7]);

    sumsubq_s32(&b_hi[0], &b_hi[1], a_hi[0], a_hi[1]);
    sumsubq_s32(&b_hi[2], &b_hi[3], a_hi[2], a_hi[3]);
    sumsubq_s32(&b_hi[4], &b_hi[5], a_hi[4], a_hi[5]);
    sumsubq_s32(&b_hi[6], &b_hi[7], a_hi[6], a_hi[7]);

    uint32x4_t max0_lo = max_abs_s32(b_lo[0], b_hi[0]);
    uint32x4_t max1_lo = max_abs_s32(b_lo[1], b_hi[1]);
    uint32x4_t max2_lo = max_abs_s32(b_lo[2], b_hi[2]);
    uint32x4_t max3_lo = max_abs_s32(b_lo[3], b_hi[3]);
    uint32x4_t max0_hi = max_abs_s32(b_lo[4], b_hi[4]);
    uint32x4_t max1_hi = max_abs_s32(b_lo[5], b_hi[5]);
    uint32x4_t max2_hi = max_abs_s32(b_lo[6], b_hi[6]);
    uint32x4_t max3_hi = max_abs_s32(b_lo[7], b_hi[7]);

    out[0] = vaddq_u32(max0_lo, max0_hi);
    out[1] = vaddq_u32(max1_lo, max1_hi);
    out[2] = vaddq_u32(max2_lo, max2_hi);
    out[3] = vaddq_u32(max3_lo, max3_hi);
}

#endif // X265_DEPTH != 12

static inline int hadamard_4x4(int16x8_t a0, int16x8_t a1)
{
    int16x8_t sum, dif, t0, t1;
    sumsubq_s16(&sum, &dif, a0, a1);

    transpose_s16_s64x2(&t0, &t1, sum, dif);
    sumsubq_s16(&sum, &dif, t0, t1);

    transpose_s16_s16x2(&t0, &t1, sum, dif);
    sumsubq_s16(&sum, &dif, t0, t1);

    transpose_s16_s32x2(&t0, &t1, sum, dif);

    uint16x8_t max = max_abs_s16(t0, t1);

    return vaddlvq_u16(max);
}

// Calculate 2 4x4 hadamard transformation.
static void hadamard_4x4_dual(int16x8_t diff[4], uint16x8_t *out)
{
    int16x8_t temp[4];

    hadamard_4_v(diff, temp);
    hadamard_4_h(temp, diff);

    uint16x8_t sum0 = max_abs_s16(diff[0], diff[1]);
    uint16x8_t sum1 = max_abs_s16(diff[2], diff[3]);

    *out = vaddq_u16(sum0, sum1);
}

// Calculate 4 4x4 hadamard transformation.
static inline void hadamard_4x4_quad(int16x8_t diff[8], uint16x8_t out[2])
{
    int16x8_t temp[8];

    hadamard_4_v(diff, temp);
    hadamard_4_v(diff + 4, temp + 4);

    hadamard_4_h(temp, diff);
    hadamard_4_h(temp + 4, diff + 4);

    uint16x8_t sum0 = max_abs_s16(diff[0], diff[1]);
    uint16x8_t sum1 = max_abs_s16(diff[2], diff[3]);
    uint16x8_t sum2 = max_abs_s16(diff[4], diff[5]);
    uint16x8_t sum3 = max_abs_s16(diff[6], diff[7]);

    out[0] = vaddq_u16(sum0, sum1);
    out[1] = vaddq_u16(sum2, sum3);
}

#if X265_DEPTH == 8
static inline void hadamard_8x8(int16x8_t diff[8], uint16x8_t out[2])
{
    int16x8_t temp[8];
    uint16x8_t sum[4];

    hadamard_8_v(diff, temp);
    hadamard_8_h(temp, sum);

    out[0] = vaddq_u16(sum[0], sum[1]);
    out[1] = vaddq_u16(sum[2], sum[3]);
}

#elif X265_DEPTH == 10
static inline void hadamard_8x8(int16x8_t diff[8], uint32x4_t out[2])
{
    int16x8_t temp[8];
    uint16x8_t sum[4];

    hadamard_8_v(diff, temp);
    hadamard_8_h(temp, sum);

    out[0] = vpaddlq_u16(sum[0]);
    out[1] = vpaddlq_u16(sum[1]);
    out[0] = vpadalq_u16(out[0], sum[2]);
    out[1] = vpadalq_u16(out[1], sum[3]);
}

#elif X265_DEPTH == 12
static inline void hadamard_8x8(int16x8_t diff[8], uint32x4_t out[2])
{
    int16x8_t temp[8];
    uint32x4_t sum[4];

    hadamard_8_v(diff, temp);
    hadamard_8_h(temp, sum);

    out[0] = vaddq_u32(sum[0], sum[1]);
    out[1] = vaddq_u32(sum[2], sum[3]);
}

#endif // X265_DEPTH == 8

#if HIGH_BIT_DEPTH
static inline int pixel_satd_4x4_neon(const uint16_t *pix1, intptr_t stride_pix1,
                                      const uint16_t *pix2, intptr_t stride_pix2)
{
    uint16x4_t s[4], r[4];
    load_u16x4xn<4>(pix1, stride_pix1, s);
    load_u16x4xn<4>(pix2, stride_pix2, r);

    uint16x8_t s0 = vcombine_u16(s[0], s[2]);
    uint16x8_t s1 = vcombine_u16(s[1], s[3]);
    uint16x8_t r0 = vcombine_u16(r[0], r[2]);
    uint16x8_t r1 = vcombine_u16(r[1], r[3]);

    int16x8_t diff0 = vreinterpretq_s16_u16(vsubq_u16(s0, r0));
    int16x8_t diff1 = vreinterpretq_s16_u16(vsubq_u16(r1, s1));

    return hadamard_4x4(diff0, diff1);
}

static inline int pixel_satd_4x8_neon(const uint16_t *pix1, intptr_t stride_pix1,
                                      const uint16_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[4];

    uint16x4_t s[8], r[8];
    load_u16x4xn<8>(pix1, stride_pix1, s);
    load_u16x4xn<8>(pix2, stride_pix2, r);

    uint16x8_t s0 = vcombine_u16(s[0], s[4]);
    uint16x8_t s1 = vcombine_u16(s[1], s[5]);
    uint16x8_t s2 = vcombine_u16(s[2], s[6]);
    uint16x8_t s3 = vcombine_u16(s[3], s[7]);
    uint16x8_t r0 = vcombine_u16(r[0], r[4]);
    uint16x8_t r1 = vcombine_u16(r[1], r[5]);
    uint16x8_t r2 = vcombine_u16(r[2], r[6]);
    uint16x8_t r3 = vcombine_u16(r[3], r[7]);

    diff[0] = vreinterpretq_s16_u16(vsubq_u16(s0, r0));
    diff[1] = vreinterpretq_s16_u16(vsubq_u16(r1, s1));
    diff[2] = vreinterpretq_s16_u16(vsubq_u16(s2, r2));
    diff[3] = vreinterpretq_s16_u16(vsubq_u16(r3, s3));

    uint16x8_t out;
    hadamard_4x4_dual(diff, &out);

    return vaddlvq_u16(out);
}

static inline int pixel_satd_8x4_neon(const uint16_t *pix1, intptr_t stride_pix1,
                                      const uint16_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[4];
    load_diff_u16x8x4(pix1, stride_pix1, pix2, stride_pix2, diff);

    uint16x8_t out;
    hadamard_4x4_dual(diff, &out);

    return vaddlvq_u16(out);
}

static inline int pixel_satd_8x8_neon(const uint16_t *pix1, intptr_t stride_pix1,
                                      const uint16_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[8];
    uint16x8_t out[2];

    load_diff_u16x8x4_dual(pix1, stride_pix1, pix2, stride_pix2, diff);
    hadamard_4x4_quad(diff, out);

    uint32x4_t res = vpaddlq_u16(out[0]);
    res = vpadalq_u16(res, out[1]);

    return vaddvq_u32(res);
}

static inline int pixel_satd_8x16_neon(const uint16_t *pix1, intptr_t stride_pix1,
                                       const uint16_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[16];
    uint16x8_t out[4];

    load_diff_u16x8x4_dual(pix1, stride_pix1, pix2, stride_pix2, diff);
    load_diff_u16x8x4_dual(pix1 + 8 * stride_pix1, stride_pix1,
                           pix2 + 8 * stride_pix2, stride_pix2, diff + 8);

    hadamard_4x4_quad(diff, out);
    hadamard_4x4_quad(diff + 8, out + 2);

    uint16x8_t sum0 = vaddq_u16(out[0], out[1]);
    uint16x8_t sum1 = vaddq_u16(out[2], out[3]);

    uint32x4_t res = vpaddlq_u16(sum0);
    res = vpadalq_u16(res, sum1);

    return vaddvq_u32(res);
}

static inline int pixel_satd_16x4_neon(const uint16_t *pix1, intptr_t stride_pix1,
                                       const uint16_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[8];

    load_diff_u16x8x4(pix1, stride_pix1, pix2, stride_pix2, diff);
    load_diff_u16x8x4(pix1 + 8, stride_pix1, pix2 + 8, stride_pix2, diff + 4);

    uint16x8_t sum0, sum1;
    hadamard_4x4_dual(diff, &sum0);
    hadamard_4x4_dual(diff + 4, &sum1);

    sum0 = vaddq_u16(sum0, sum1);

    return vaddlvq_u16(sum0);
}

static inline int pixel_satd_16x8_neon(const uint16_t *pix1, intptr_t stride_pix1,
                                       const uint16_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[16];
    uint16x8_t out[4];

    load_diff_u16x8x4_dual(pix1, stride_pix1, pix2, stride_pix2, diff);
    load_diff_u16x8x4_dual(pix1 + 8, stride_pix1,  pix2 + 8, stride_pix2, diff + 8);

    hadamard_4x4_quad(diff, out);
    hadamard_4x4_quad(diff + 8, out + 2);

#if X265_DEPTH == 10
    uint16x8_t sum0 = vaddq_u16(out[0], out[1]);
    uint16x8_t sum1 = vaddq_u16(out[2], out[3]);

    sum0 = vaddq_u16(sum0, sum1);

    return vaddlvq_u16(sum0);
#else // X265_DEPTH == 12
    uint32x4_t sum0 = vpaddlq_u16(out[0]);
    uint32x4_t sum1 = vpaddlq_u16(out[1]);
    sum0 = vpadalq_u16(sum0, out[2]);
    sum1 = vpadalq_u16(sum1, out[3]);

    sum0 = vaddq_u32(sum0, sum1);

    return vaddvq_u32(sum0);
#endif // X265_DEPTH == 10
}

static inline int pixel_satd_16x16_neon(const uint16_t *pix1, intptr_t stride_pix1,
                                        const uint16_t *pix2, intptr_t stride_pix2)
{
    uint32x4_t sum[2]= { vdupq_n_u32(0), vdupq_n_u32(0) };
    int16x8_t diff[8];
    uint16x8_t out[2];

    for (int i = 0; i < 4; ++i)
    {
        load_diff_u16x8x4(pix1, stride_pix1, pix2, stride_pix2, diff);
        load_diff_u16x8x4(pix1 + 8, stride_pix1, pix2 + 8, stride_pix2, diff + 4);

        hadamard_4x4_quad(diff, out);

        sum[0] = vpadalq_u16(sum[0], out[0]);
        sum[1] = vpadalq_u16(sum[1], out[1]);

        pix1 += 4 * stride_pix1;
        pix2 += 4 * stride_pix2;
    }

    return vaddvq_u32(vaddq_u32(sum[0], sum[1]));
}

static inline int pixel_sa8d_8x8_neon(const uint16_t *pix1, intptr_t stride_pix1,
                                      const uint16_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[8];
    uint32x4_t res[2];

    load_diff_u16x8x4_dual(pix1, stride_pix1, pix2, stride_pix2, diff);
    hadamard_8x8(diff, res);

    uint32x4_t s = vaddq_u32(res[0], res[1]);

    return (vaddvq_u32(s) + 1) >> 1;
}

static inline int pixel_sa8d_16x16_neon(const uint16_t *pix1, intptr_t stride_pix1,
                                        const uint16_t *pix2, intptr_t stride_pix2)
{
    uint32x4_t sum0, sum1;

    int16x8_t diff[8];
    uint32x4_t res[2];

    load_diff_u16x8x8(pix1, stride_pix1, pix2, stride_pix2, diff);
    hadamard_8x8(diff, res);
    sum0 = vaddq_u32(res[0], res[1]);

    load_diff_u16x8x8(pix1 + 8, stride_pix1, pix2 + 8, stride_pix2, diff);
    hadamard_8x8(diff, res);
    sum1 = vaddq_u32(res[0], res[1]);

    load_diff_u16x8x8(pix1 + 8 * stride_pix1, stride_pix1,
                      pix2 + 8 * stride_pix2, stride_pix2, diff);
    hadamard_8x8(diff, res);
    sum0 = vaddq_u32(sum0, res[0]);
    sum1 = vaddq_u32(sum1, res[1]);

    load_diff_u16x8x8(pix1 + 8 * stride_pix1 + 8, stride_pix1,
                      pix2 + 8 * stride_pix2 + 8, stride_pix2, diff);
    hadamard_8x8(diff, res);
    sum0 = vaddq_u32(sum0, res[0]);
    sum1 = vaddq_u32(sum1, res[1]);

    sum0 = vaddq_u32(sum0, sum1);

    return (vaddvq_u32(sum0) + 1) >> 1;
}

#else // !HIGH_BIT_DEPTH
static inline int pixel_satd_4x4_neon(const uint8_t *pix1, intptr_t stride_pix1,
                                      const uint8_t *pix2, intptr_t stride_pix2)
{
    uint8x8_t s0 = load_u8x4x2(pix1, 2 * stride_pix1);
    uint8x8_t s1 = load_u8x4x2(pix1 + stride_pix1, 2 * stride_pix1);

    uint8x8_t r0 = load_u8x4x2(pix2, 2 * stride_pix2);
    uint8x8_t r1 = load_u8x4x2(pix2 + stride_pix2, 2 * stride_pix2);

    int16x8_t diff0 = vreinterpretq_s16_u16(vsubl_u8(s0, r0));
    int16x8_t diff1 = vreinterpretq_s16_u16(vsubl_u8(r1, s1));

    return hadamard_4x4(diff0, diff1);
}

static inline int pixel_satd_4x8_neon(const uint8_t *pix1, intptr_t stride_pix1,
                                      const uint8_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[4];

    uint8x8_t s0 = load_u8x4x2(pix1 + 0 * stride_pix1, 4 * stride_pix1);
    uint8x8_t s1 = load_u8x4x2(pix1 + 1 * stride_pix1, 4 * stride_pix1);
    uint8x8_t s2 = load_u8x4x2(pix1 + 2 * stride_pix1, 4 * stride_pix1);
    uint8x8_t s3 = load_u8x4x2(pix1 + 3 * stride_pix1, 4 * stride_pix1);
    uint8x8_t r0 = load_u8x4x2(pix2 + 0 * stride_pix2, 4 * stride_pix2);
    uint8x8_t r1 = load_u8x4x2(pix2 + 1 * stride_pix2, 4 * stride_pix2);
    uint8x8_t r2 = load_u8x4x2(pix2 + 2 * stride_pix2, 4 * stride_pix2);
    uint8x8_t r3 = load_u8x4x2(pix2 + 3 * stride_pix2, 4 * stride_pix2);

    diff[0] = vreinterpretq_s16_u16(vsubl_u8(s0, r0));
    diff[1] = vreinterpretq_s16_u16(vsubl_u8(r1, s1));
    diff[2] = vreinterpretq_s16_u16(vsubl_u8(s2, r2));
    diff[3] = vreinterpretq_s16_u16(vsubl_u8(r3, s3));

    uint16x8_t out;
    hadamard_4x4_dual(diff, &out);

    return vaddlvq_u16(out);
}

static inline int pixel_satd_8x4_neon(const uint8_t *pix1, intptr_t stride_pix1,
                                      const uint8_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[4];

    load_diff_u8x8x4(pix1, stride_pix1, pix2, stride_pix2, diff);

    uint16x8_t out;
    hadamard_4x4_dual(diff, &out);

    return vaddlvq_u16(out);
}

static inline int pixel_satd_8x8_neon(const uint8_t *pix1, intptr_t stride_pix1,
                                      const uint8_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[8];
    uint16x8_t out[2];

    load_diff_u8x8x8(pix1, stride_pix1, pix2, stride_pix2, diff);
    hadamard_4x4_quad(diff, out);

    out[0] = vaddq_u16(out[0], out[1]);

    return vaddlvq_u16(out[0]);
}

static inline int pixel_satd_8x16_neon(const uint8_t *pix1, intptr_t stride_pix1,
                                       const uint8_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[16];
    uint16x8_t out[4];

    load_diff_u8x8x8(pix1, stride_pix1, pix2, stride_pix2, diff);
    load_diff_u8x8x8(pix1 + 8 * stride_pix1, stride_pix1,
                     pix2 + 8 * stride_pix2, stride_pix2, diff + 8);

    hadamard_4x4_quad(diff, out);
    hadamard_4x4_quad(diff + 8, out + 2);

    uint16x8_t sum0 = vaddq_u16(out[0], out[1]);
    uint16x8_t sum1 = vaddq_u16(out[2], out[3]);

    sum0 = vaddq_u16(sum0, sum1);

    return vaddlvq_u16(sum0);
}

static inline int pixel_satd_16x4_neon(const uint8_t *pix1, intptr_t stride_pix1,
                                       const uint8_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[8];

    load_diff_u8x8x4(pix1, stride_pix1, pix2, stride_pix2, diff);
    load_diff_u8x8x4(pix1 + 8, stride_pix1, pix2 + 8, stride_pix2, diff + 4);

    uint16x8_t out[2];
    hadamard_4x4_dual(diff, &out[0]);
    hadamard_4x4_dual(diff + 4, &out[1]);

    out[0] = vaddq_u16(out[0], out[1]);

    return vaddlvq_u16(out[0]);
}

static inline int pixel_satd_16x8_neon(const uint8_t *pix1, intptr_t stride_pix1,
                                       const uint8_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[16];
    uint16x8_t out[4];

    load_diff_u8x8x8(pix1, stride_pix1, pix2, stride_pix2, diff);
    load_diff_u8x8x8(pix1 + 8, stride_pix1,  pix2 + 8, stride_pix2, diff + 8);

    hadamard_4x4_quad(diff, out);
    hadamard_4x4_quad(diff + 8, out + 2);

    uint16x8_t sum0 = vaddq_u16(out[0], out[1]);
    uint16x8_t sum1 = vaddq_u16(out[2], out[3]);

    sum0 = vaddq_u16(sum0, sum1);

    return vaddlvq_u16(sum0);
}

static inline int pixel_satd_16x16_neon(const uint8_t *pix1, intptr_t stride_pix1,
                                        const uint8_t *pix2, intptr_t stride_pix2)
{
    uint16x8_t sum[2], out[2];
    int16x8_t diff[8];

    load_diff_u8x16x4(pix1, stride_pix1, pix2, stride_pix2, diff);
    hadamard_4x4_quad(diff, out);
    sum[0] = out[0];
    sum[1] = out[1];

    load_diff_u8x16x4(pix1 + 4 * stride_pix1, stride_pix1,
                      pix2 + 4 * stride_pix2, stride_pix2, diff);
    hadamard_4x4_quad(diff, out);
    sum[0] = vaddq_u16(sum[0], out[0]);
    sum[1] = vaddq_u16(sum[1], out[1]);

    load_diff_u8x16x4(pix1 + 8 * stride_pix1, stride_pix1,
                      pix2 + 8 * stride_pix2, stride_pix2, diff);
    hadamard_4x4_quad(diff, out);
    sum[0] = vaddq_u16(sum[0], out[0]);
    sum[1] = vaddq_u16(sum[1], out[1]);

    load_diff_u8x16x4(pix1 + 12 * stride_pix1, stride_pix1,
                      pix2 + 12 * stride_pix2, stride_pix2, diff);
    hadamard_4x4_quad(diff, out);
    sum[0] = vaddq_u16(sum[0], out[0]);
    sum[1] = vaddq_u16(sum[1], out[1]);

    uint32x4_t sum0 = vpaddlq_u16(sum[0]);
    uint32x4_t sum1 = vpaddlq_u16(sum[1]);

    sum0 = vaddq_u32(sum0, sum1);

    return vaddvq_u32(sum0);
}

static inline int pixel_sa8d_8x8_neon(const uint8_t *pix1, intptr_t stride_pix1,
                                      const uint8_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[8];
    uint16x8_t res[2];

    load_diff_u8x8x8(pix1, stride_pix1, pix2, stride_pix2, diff);
    hadamard_8x8(diff, res);

    return (vaddlvq_u16(vaddq_u16(res[0], res[1])) + 1) >> 1;
}

static inline int pixel_sa8d_16x16_neon(const uint8_t *pix1, intptr_t stride_pix1,
                                        const uint8_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[8];
    uint16x8_t res[2];
    uint32x4_t sum0, sum1;

    load_diff_u8x8x8(pix1, stride_pix1, pix2, stride_pix2, diff);
    hadamard_8x8(diff, res);
    sum0 = vpaddlq_u16(res[0]);
    sum1 = vpaddlq_u16(res[1]);

    load_diff_u8x8x8(pix1 + 8, stride_pix1, pix2 + 8, stride_pix2, diff);
    hadamard_8x8(diff, res);
    sum0 = vpadalq_u16(sum0, res[0]);
    sum1 = vpadalq_u16(sum1, res[1]);

    load_diff_u8x8x8(pix1 + 8 * stride_pix1, stride_pix1,
                     pix2 + 8 * stride_pix2, stride_pix2, diff);
    hadamard_8x8(diff, res);
    sum0 = vpadalq_u16(sum0, res[0]);
    sum1 = vpadalq_u16(sum1, res[1]);

    load_diff_u8x8x8(pix1 + 8 * stride_pix1 + 8, stride_pix1,
                     pix2 + 8 * stride_pix2 + 8, stride_pix2, diff);
    hadamard_8x8(diff, res);
    sum0 = vpadalq_u16(sum0, res[0]);
    sum1 = vpadalq_u16(sum1, res[1]);

    sum0 = vaddq_u32(sum0, sum1);

    return (vaddvq_u32(sum0) + 1) >> 1;
}

#endif // HIGH_BIT_DEPTH

template<int lx, int ly>
int sad_pp_neon(const pixel *pix1, intptr_t stride_pix1, const pixel *pix2, intptr_t stride_pix2)
{
    int sum = 0;


    for (int y = 0; y < ly; y++)
    {
#if HIGH_BIT_DEPTH
        int x = 0;
        uint16x8_t vsum16_1 = vdupq_n_u16(0);
        for (; (x + 8) <= lx; x += 8)
        {
            uint16x8_t p1 = vld1q_u16(pix1 + x);
            uint16x8_t p2 = vld1q_u16(pix2 + x);
            vsum16_1 = vabaq_u16(vsum16_1, p1, p2);
        }
        if (lx & 4)
        {
            uint16x4_t p1 = vld1_u16(pix1 + x);
            uint16x4_t p2 = vld1_u16(pix2 + x);
            sum += vaddlv_u16(vaba_u16(vdup_n_u16(0), p1, p2));
            x += 4;
        }
        if (lx >= 4)
        {
            sum += vaddlvq_u16(vsum16_1);
        }

#else

        int x = 0;
        uint16x8_t vsum16_1 = vdupq_n_u16(0);
        uint16x8_t vsum16_2 = vdupq_n_u16(0);

        for (; (x + 16) <= lx; x += 16)
        {
            uint8x16_t p1 = vld1q_u8(pix1 + x);
            uint8x16_t p2 = vld1q_u8(pix2 + x);
            vsum16_1 = vabal_u8(vsum16_1, vget_low_u8(p1), vget_low_u8(p2));
            vsum16_2 = vabal_high_u8(vsum16_2, p1, p2);
        }
        if (lx & 8)
        {
            uint8x8_t p1 = vld1_u8(pix1 + x);
            uint8x8_t p2 = vld1_u8(pix2 + x);
            vsum16_1 = vabal_u8(vsum16_1, p1, p2);
            x += 8;
        }
        if (lx & 4)
        {
            uint8x8_t p1 = load_u8x4x1(pix1 + x);
            uint8x8_t p2 = load_u8x4x1(pix2 + x);
            vsum16_1 = vabal_u8(vsum16_1, p1, p2);
            x += 4;
        }
        if (lx >= 16)
        {
            vsum16_1 = vaddq_u16(vsum16_1, vsum16_2);
        }
        if (lx >= 4)
        {
            sum += vaddvq_u16(vsum16_1);
        }

#endif
        if (lx & 3) for (; x < lx; x++)
            {
                sum += abs(pix1[x] - pix2[x]);
            }

        pix1 += stride_pix1;
        pix2 += stride_pix2;
    }

    return sum;
}

template<int size>
void blockfill_s_neon(int16_t *dst, intptr_t dstride, int16_t val)
{
    for (int h = 0; h < size; h++)
    {
        for (int w = 0; w + 16 <= size; w += 16)
        {
            vst1q_s16(dst + h * dstride + w, vdupq_n_s16(val));
            vst1q_s16(dst + h * dstride + w + 8, vdupq_n_s16(val));
        }
        if (size == 8)
        {
            vst1q_s16(dst + h * dstride, vdupq_n_s16(val));
        }
        if (size == 4)
        {
            vst1_s16(dst + h * dstride, vdup_n_s16(val));
        }
    }
}

#if !HIGH_BIT_DEPTH
template<int width, int height>
void blockcopy_ps_neon(int16_t *dst, intptr_t dst_stride, const pixel *src,
                       intptr_t src_stride)
{
    for (int h = 0; h < height; h++)
    {
        int w = 0;
        for (; w + 16 <= width; w += 16)
        {
            uint8x16_t s = vld1q_u8(src + w);
            uint8x16x2_t t = vzipq_u8(s, vdupq_n_u8(0));
            int16x8x2_t s_s16;
            s_s16.val[0] = vreinterpretq_s16_u8(t.val[0]);
            s_s16.val[1] = vreinterpretq_s16_u8(t.val[1]);
            vst1q_s16_x2(dst + w, s_s16);
        }
        if (width & 8)
        {
            uint8x8_t s = vld1_u8(src + w);
            uint16x8_t s_u16 = vmovl_u8(s);
            vst1q_s16(dst + w, vreinterpretq_s16_u16(s_u16));
            w += 8;
        }
        if (width & 4)
        {
            uint8x8_t s = load_u8x4x1(src + w);
            uint16x4_t s_u16 = vget_low_u16(vmovl_u8(s));
            vst1_s16(dst + w, vreinterpret_s16_u16(s_u16));
        }

        dst += dst_stride;
        src += src_stride;
    }
}
#endif // !HIGH_BIT_DEPTH

template<int width, int height>
void blockcopy_pp_neon(pixel *dst, intptr_t dst_stride, const pixel *src,
                       intptr_t src_stride)
{
    for (int h = 0; h < height; h++)
    {
        int w = 0;
#if HIGH_BIT_DEPTH
        for (; w + 16 <= width; w += 16)
        {
            uint16x8_t s0_lo = vld1q_u16(src + w);
            uint16x8_t s0_hi = vld1q_u16(src + w + 8);
            vst1q_u16(dst + w, s0_lo);
            vst1q_u16(dst + w + 8, s0_hi);
        }
        if (width & 8)
        {
            uint16x8_t s0 = vld1q_u16(src + w);
            vst1q_u16(dst + w, s0);
            w += 8;
        }
        if (width & 4)
        {
            uint16x4_t s0 = vld1_u16(src + w);
            vst1_u16(dst + w, s0);
            w += 4;
        }
#else
        for (; w + 32 <= width; w += 32)
        {
            uint8x16_t s0_lo = vld1q_u8(src + w);
            uint8x16_t s0_hi = vld1q_u8(src + w + 16);
            vst1q_u8(dst + w, s0_lo);
            vst1q_u8(dst + w + 16, s0_hi);
        }
        if (width & 16)
        {
            uint8x16_t s0 = vld1q_u8(src + w);
            vst1q_u8(dst + w, s0);
            w += 16;
        }
        if (width & 8)
        {
            uint8x8_t s0 = vld1_u8(src + w);
            vst1_u8(dst + w, s0);
            w += 8;
        }
        if (width & 4)
        {
            uint8x8_t s0 = load_u8x4x1(src + w);
            store_u8x4x1(dst + w, s0);
            w += 4;
        }
#endif
        for (; w < width; w++)
        {
            dst[w] = src[w];
        }

        src += src_stride;
        dst += dst_stride;
    }
}

template<int width, int height>
void blockcopy_ss_neon(int16_t *dst, intptr_t dst_stride, const int16_t *src,
                       intptr_t src_stride)
{
    for (int h = 0; h < height; h++)
    {
        int w = 0;
        for (; w + 16 <= width; w += 16)
        {
            int16x8_t a0 = vld1q_s16(src + w + 0);
            int16x8_t a1 = vld1q_s16(src + w + 8);
            vst1q_s16(dst + w + 0, a0);
            vst1q_s16(dst + w + 8, a1);
        }
        if (width & 8)
        {
            vst1q_s16(dst + w, vld1q_s16(src + w));
            w += 8;
        }
        if (width & 4)
        {
            vst1_s16(dst + w, vld1_s16(src + w));
        }

        dst += dst_stride;
        src += src_stride;
    }
}

#if !HIGH_BIT_DEPTH
template<int width, int height>
void blockcopy_sp_neon(pixel *dst, intptr_t dst_stride, const int16_t *src,
                       intptr_t src_stride)
{
    for (int h = 0; h < height; h++)
    {
        int w = 0;
        for (; w + 16 <= width; w += 16) {
            int16x8_t s0 = vld1q_s16(src + w + 0);
            int16x8_t s1 = vld1q_s16(src + w + 8);
            int8x16_t s01 = vcombine_s8(vmovn_s16(s0), vmovn_s16(s1));
            vst1q_u8(dst + w, vreinterpretq_u8_s8(s01));
        }
        if (width & 8)
        {
            int16x8_t s0 = vld1q_s16(src + w);
            int8x8_t s0_s8 = vmovn_s16(s0);
            vst1_u8(dst + w, vreinterpret_u8_s8(s0_s8));
            w += 8;
        }
        if (width & 4)
        {
            int16x4_t s0 = vld1_s16(src + w);
            int8x8_t s0_s8 = vmovn_s16(vcombine_s16(s0, vdup_n_s16(0)));
            store_u8x4x1(dst + w, vreinterpret_u8_s8(s0_s8));
        }

        dst += dst_stride;
        src += src_stride;
    }
}
#endif // !HIGH_BIT_DEPTH

template<int bx, int by>
void pixel_sub_ps_neon(int16_t *a, intptr_t dstride, const pixel *b0, const pixel *b1, intptr_t sstride0,
                       intptr_t sstride1)
{
    for (int y = 0; y < by; y++)
    {
        int x = 0;
        for (; (x + 8) <= bx; x += 8)
        {
#if HIGH_BIT_DEPTH
            uint16x8_t diff = vsubq_u16(vld1q_u16(b0 + x), vld1q_u16(b1 + x));
            vst1q_s16(a + x, vreinterpretq_s16_u16(diff));
#else
            uint16x8_t diff = vsubl_u8(vld1_u8(b0 + x), vld1_u8(b1 + x));
            vst1q_s16(a + x, vreinterpretq_s16_u16(diff));
#endif
        }
        for (; x < bx; x++)
        {
            a[x] = (int16_t)(b0[x] - b1[x]);
        }

        b0 += sstride0;
        b1 += sstride1;
        a += dstride;
    }
}

template<int bx, int by>
void pixel_add_ps_neon(pixel *a, intptr_t dstride, const pixel *b0, const int16_t *b1, intptr_t sstride0,
                       intptr_t sstride1)
{
    for (int y = 0; y < by; y++)
    {
        int x = 0;
        for (; (x + 8) <= bx; x += 8)
        {
            int16x8_t t;
            int16x8_t b1e = vld1q_s16(b1 + x);
            int16x8_t b0e;
#if HIGH_BIT_DEPTH
            b0e = vreinterpretq_s16_u16(vld1q_u16(b0 + x));
            t = vaddq_s16(b0e, b1e);
            t = vminq_s16(t, vdupq_n_s16((1 << X265_DEPTH) - 1));
            t = vmaxq_s16(t, vdupq_n_s16(0));
            vst1q_u16(a + x, vreinterpretq_u16_s16(t));
#else
            b0e = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(b0 + x)));
            t = vaddq_s16(b0e, b1e);
            vst1_u8(a + x, vqmovun_s16(t));
#endif
        }
        for (; x < bx; x++)
        {
            a[x] = (int16_t)x265_clip(b0[x] + b1[x]);
        }

        b0 += sstride0;
        b1 += sstride1;
        a += dstride;
    }
}

template<int bx, int by>
void addAvg_neon(const int16_t *src0, const int16_t *src1, pixel *dst, intptr_t src0Stride, intptr_t src1Stride,
                 intptr_t dstStride)
{

    const int shiftNum = IF_INTERNAL_PREC + 1 - X265_DEPTH;
    const int offset = (1 << (shiftNum - 1)) + 2 * IF_INTERNAL_OFFS;

    const int32x4_t addon = vdupq_n_s32(offset);
    for (int y = 0; y < by; y++)
    {
        int x = 0;

        for (; (x + 8) <= bx; x += 8)
        {
            int16x8_t in0 = vld1q_s16(src0 + x);
            int16x8_t in1 = vld1q_s16(src1 + x);
            int32x4_t t1 = vaddl_s16(vget_low_s16(in0), vget_low_s16(in1));
            int32x4_t t2 = vaddl_high_s16(in0, in1);
            t1 = vaddq_s32(t1, addon);
            t2 = vaddq_s32(t2, addon);
            t1 = vshrq_n_s32(t1, shiftNum);
            t2 = vshrq_n_s32(t2, shiftNum);
            int16x8_t t = vuzp1q_s16(vreinterpretq_s16_s32(t1),
                                     vreinterpretq_s16_s32(t2));
#if HIGH_BIT_DEPTH
            t = vminq_s16(t, vdupq_n_s16((1 << X265_DEPTH) - 1));
            t = vmaxq_s16(t, vdupq_n_s16(0));
            vst1q_u16(dst + x, vreinterpretq_u16_s16(t));
#else
            vst1_u8(dst + x, vqmovun_s16(t));
#endif
        }
        for (; x < bx; x += 2)
        {
            dst[x + 0] = x265_clip((src0[x + 0] + src1[x + 0] + offset) >> shiftNum);
            dst[x + 1] = x265_clip((src0[x + 1] + src1[x + 1] + offset) >> shiftNum);
        }

        src0 += src0Stride;
        src1 += src1Stride;
        dst  += dstStride;
    }
}

void planecopy_cp_neon(const uint8_t *src, intptr_t srcStride, pixel *dst,
                       intptr_t dstStride, int width, int height, int shift)
{
    X265_CHECK(width >= 16, "width length error\n");
    X265_CHECK(height >= 1, "height length error\n");
    X265_CHECK(shift == X265_DEPTH - 8, "shift value error\n");

    (void)shift;

    do
    {
#if HIGH_BIT_DEPTH
        for (int w = 0; w < width - 16; w += 16)
        {
            uint8x16_t in = vld1q_u8(src + w);
            uint16x8_t t0 = vshll_n_u8(vget_low_u8(in), X265_DEPTH - 8);
            uint16x8_t t1 = vshll_n_u8(vget_high_u8(in), X265_DEPTH - 8);
            vst1q_u16(dst + w + 0, t0);
            vst1q_u16(dst + w + 8, t1);
        }
        // Tail - src must be different from dst for this to work.
        {
            uint8x16_t in = vld1q_u8(src + width - 16);
            uint16x8_t t0 = vshll_n_u8(vget_low_u8(in), X265_DEPTH - 8);
            uint16x8_t t1 = vshll_n_u8(vget_high_u8(in), X265_DEPTH - 8);
            vst1q_u16(dst + width - 16, t0);
            vst1q_u16(dst + width - 8, t1);
        }
#else
        int w;
        for (w = 0; w < width - 32; w += 32)
        {
            uint8x16_t in0 = vld1q_u8(src + w + 0);
            uint8x16_t in1 = vld1q_u8(src + w + 16);
            vst1q_u8(dst + w + 0, in0);
            vst1q_u8(dst + w + 16, in1);
        }
        if (w < width - 16)
        {
            uint8x16_t in = vld1q_u8(src + w);
            vst1q_u8(dst + w, in);
        }
        // Tail - src must be different from dst for this to work.
        {
            uint8x16_t in = vld1q_u8(src + width - 16);
            vst1q_u8(dst + width - 16, in);
        }
#endif
        dst += dstStride;
        src += srcStride;
    }
    while (--height != 0);
}

void weight_pp_neon(const pixel *src, pixel *dst, intptr_t stride, int width, int height,
                    int w0, int round, int shift, int offset)
{
    const int correction = IF_INTERNAL_PREC - X265_DEPTH;

    X265_CHECK(height >= 1, "height length error\n");
    X265_CHECK(width >= 16, "width length error\n");
    X265_CHECK(!(width & 15), "width alignment error\n");
    X265_CHECK(w0 >= 0, "w0 should be min 0\n");
    X265_CHECK(w0 < 128, "w0 should be max 127\n");
    X265_CHECK(shift >= correction, "shift must include factor correction\n");
    X265_CHECK((round & ((1 << correction) - 1)) == 0,
               "round must include factor correction\n");

    (void)round;

#if HIGH_BIT_DEPTH
    int32x4_t corrected_shift = vdupq_n_s32(correction - shift);

    do
    {
        int w = 0;
        do
        {
            int16x8_t s0 = vreinterpretq_s16_u16(vld1q_u16(src + w + 0));
            int16x8_t s1 = vreinterpretq_s16_u16(vld1q_u16(src + w + 8));
            int32x4_t weighted_s0_lo = vmull_n_s16(vget_low_s16(s0), w0);
            int32x4_t weighted_s0_hi = vmull_n_s16(vget_high_s16(s0), w0);
            int32x4_t weighted_s1_lo = vmull_n_s16(vget_low_s16(s1), w0);
            int32x4_t weighted_s1_hi = vmull_n_s16(vget_high_s16(s1), w0);
            weighted_s0_lo = vrshlq_s32(weighted_s0_lo, corrected_shift);
            weighted_s0_hi = vrshlq_s32(weighted_s0_hi, corrected_shift);
            weighted_s1_lo = vrshlq_s32(weighted_s1_lo, corrected_shift);
            weighted_s1_hi = vrshlq_s32(weighted_s1_hi, corrected_shift);
            weighted_s0_lo = vaddq_s32(weighted_s0_lo, vdupq_n_s32(offset));
            weighted_s0_hi = vaddq_s32(weighted_s0_hi, vdupq_n_s32(offset));
            weighted_s1_lo = vaddq_s32(weighted_s1_lo, vdupq_n_s32(offset));
            weighted_s1_hi = vaddq_s32(weighted_s1_hi, vdupq_n_s32(offset));
            uint16x4_t t0_lo = vqmovun_s32(weighted_s0_lo);
            uint16x4_t t0_hi = vqmovun_s32(weighted_s0_hi);
            uint16x4_t t1_lo = vqmovun_s32(weighted_s1_lo);
            uint16x4_t t1_hi = vqmovun_s32(weighted_s1_hi);
            uint16x8_t d0 = vminq_u16(vcombine_u16(t0_lo, t0_hi), vdupq_n_u16(PIXEL_MAX));
            uint16x8_t d1 = vminq_u16(vcombine_u16(t1_lo, t1_hi), vdupq_n_u16(PIXEL_MAX));

            vst1q_u16(dst + w + 0, d0);
            vst1q_u16(dst + w + 8, d1);
            w += 16;
        }
        while (w != width);

        src += stride;
        dst += stride;
    }
    while (--height != 0);

#else
    // Re-arrange the shift operations.
    // Then, hoist the right shift out of the loop if BSF(w0) >= shift - correction.
    // Orig: (((src[x] << correction) * w0 + round) >> shift) + offset.
    // New: (src[x] * (w0 >> shift - correction)) + (round >> shift) + offset.
    // (round >> shift) is always zero since round = 1 << (shift - 1).

    unsigned long id;
    BSF(id, w0);

    if ((int)id >= shift - correction)
    {
        w0 >>= shift - correction;

        do
        {
            int w = 0;
            do
            {
                uint8x16_t s = vld1q_u8(src + w);
                int16x8_t weighted_s0 = vreinterpretq_s16_u16(
                    vmlal_u8(vdupq_n_u16(offset), vget_low_u8(s), vdup_n_u8(w0)));
                int16x8_t weighted_s1 = vreinterpretq_s16_u16(
                    vmlal_u8(vdupq_n_u16(offset), vget_high_u8(s), vdup_n_u8(w0)));
                uint8x8_t d0 = vqmovun_s16(weighted_s0);
                uint8x8_t d1 = vqmovun_s16(weighted_s1);

                vst1q_u8(dst + w, vcombine_u8(d0, d1));
                w += 16;
            }
            while (w != width);

            src += stride;
            dst += stride;
        }
        while (--height != 0);
    }
    else // Keep rounding shifts within the loop.
    {
        int16x8_t corrected_shift = vdupq_n_s16(correction - shift);

        do
        {
            int w = 0;
            do
            {
                uint8x16_t s = vld1q_u8(src + w);
                int16x8_t weighted_s0 =
                    vreinterpretq_s16_u16(vmull_u8(vget_low_u8(s), vdup_n_u8(w0)));
                int16x8_t weighted_s1 =
                    vreinterpretq_s16_u16(vmull_u8(vget_high_u8(s), vdup_n_u8(w0)));
                weighted_s0 = vrshlq_s16(weighted_s0, corrected_shift);
                weighted_s1 = vrshlq_s16(weighted_s1, corrected_shift);
                weighted_s0 = vaddq_s16(weighted_s0, vdupq_n_s16(offset));
                weighted_s1 = vaddq_s16(weighted_s1, vdupq_n_s16(offset));
                uint8x8_t d0 = vqmovun_s16(weighted_s0);
                uint8x8_t d1 = vqmovun_s16(weighted_s1);

                vst1q_u8(dst + w, vcombine_u8(d0, d1));
                w += 16;
            }
            while (w != width);

            src += stride;
            dst += stride;
        }
        while (--height != 0);
    }
#endif
}

template<int lx, int ly>
void pixelavg_pp_neon(pixel *dst, intptr_t dstride, const pixel *src0, intptr_t sstride0, const pixel *src1,
                      intptr_t sstride1, int)
{
    for (int y = 0; y < ly; y++)
    {
        int x = 0;
        for (; (x + 8) <= lx; x += 8)
        {
#if HIGH_BIT_DEPTH
            uint16x8_t in0 = vld1q_u16(src0 + x);
            uint16x8_t in1 = vld1q_u16(src1 + x);
            uint16x8_t t = vrhaddq_u16(in0, in1);
            vst1q_u16(dst + x, t);
#else
            uint16x8_t in0 = vmovl_u8(vld1_u8(src0 + x));
            uint16x8_t in1 = vmovl_u8(vld1_u8(src1 + x));
            uint16x8_t t = vrhaddq_u16(in0, in1);
            vst1_u8(dst + x, vmovn_u16(t));
#endif
        }
        for (; x < lx; x++)
        {
            dst[x] = (src0[x] + src1[x] + 1) >> 1;
        }

        src0 += sstride0;
        src1 += sstride1;
        dst += dstride;
    }
}


template<int size>
void cpy1Dto2D_shl_neon(int16_t *dst, const int16_t *src, intptr_t dstStride, int shift)
{
    X265_CHECK((((intptr_t)dst | (dstStride * sizeof(*dst))) & 15) == 0 || size == 4, "dst alignment error\n");
    X265_CHECK(((intptr_t)src & 15) == 0, "src alignment error\n");
    X265_CHECK(shift >= 0, "invalid shift\n");

    for (int h = 0; h < size; h++)
    {
        for (int w = 0; w + 16 <= size; w += 16)
        {
            int16x8_t s0_lo = vld1q_s16(src + w);
            int16x8_t s0_hi = vld1q_s16(src + w + 8);
            int16x8_t d0_lo = vshlq_s16(s0_lo, vdupq_n_s16(shift));
            int16x8_t d0_hi = vshlq_s16(s0_hi, vdupq_n_s16(shift));
            vst1q_s16(dst + w, d0_lo);
            vst1q_s16(dst + w + 8, d0_hi);
        }
        if (size == 8)
        {
            int16x8_t s0 = vld1q_s16(src);
            int16x8_t d0 = vshlq_s16(s0, vdupq_n_s16(shift));
            vst1q_s16(dst, d0);
        }
        if (size == 4)
        {
            int16x4_t s0 = vld1_s16(src);
            int16x4_t d0 = vshl_s16(s0, vdup_n_s16(shift));
            vst1_s16(dst, d0);
        }

        src += size;
        dst += dstStride;
    }
}

template<int size>
void cpy1Dto2D_shr_neon(int16_t* dst, const int16_t* src, intptr_t dstStride, int shift)
{
    X265_CHECK((((intptr_t)dst | (dstStride * sizeof(*dst))) & 15) == 0 || size == 4, "dst alignment error\n");
    X265_CHECK(((intptr_t)src & 15) == 0, "src alignment error\n");
    X265_CHECK(shift > 0, "invalid shift\n");

    for (int h = 0; h < size; h++)
    {
        for (int w = 0; w + 16 <= size; w += 16)
        {
            int16x8_t s0_lo = vld1q_s16(src + w);
            int16x8_t s0_hi = vld1q_s16(src + w + 8);
            int16x8_t d0_lo = vrshlq_s16(s0_lo, vdupq_n_s16(-shift));
            int16x8_t d0_hi = vrshlq_s16(s0_hi, vdupq_n_s16(-shift));
            vst1q_s16(dst + w, d0_lo);
            vst1q_s16(dst + w + 8, d0_hi);
        }
        if (size == 8)
        {
            int16x8_t s0 = vld1q_s16(src);
            int16x8_t d0 = vrshlq_s16(s0, vdupq_n_s16(-shift));
            vst1q_s16(dst, d0);
        }
        if (size == 4)
        {
            int16x4_t s0 = vld1_s16(src);
            int16x4_t d0 = vrshl_s16(s0, vdup_n_s16(-shift));
            vst1_s16(dst, d0);
        }

        src += size;
        dst += dstStride;
    }
}

template<int size>
uint64_t pixel_var_neon(const uint8_t *pix, intptr_t i_stride)
{
    uint32_t sum = 0, sqr = 0;

    uint32x4_t vsqr = vdupq_n_u32(0);

    for (int y = 0; y < size; y++)
    {
        int x = 0;
        uint16x8_t vsum = vdupq_n_u16(0);
        for (; (x + 8) <= size; x += 8)
        {
            uint16x8_t in;
            in = vmovl_u8(vld1_u8(pix + x));
            vsum = vaddq_u16(vsum, in);
            vsqr = vmlal_u16(vsqr, vget_low_u16(in), vget_low_u16(in));
            vsqr = vmlal_high_u16(vsqr, in, in);
        }
        for (; x < size; x++)
        {
            sum += pix[x];
            sqr += pix[x] * pix[x];
        }

        sum += vaddvq_u16(vsum);

        pix += i_stride;
    }
    sqr += vaddvq_u32(vsqr);
    return sum + ((uint64_t)sqr << 32);
}

template<int blockSize>
void getResidual_neon(const pixel *fenc, const pixel *pred, int16_t *residual, intptr_t stride)
{
    for (int y = 0; y < blockSize; y++)
    {
        int x = 0;
        for (; (x + 8) < blockSize; x += 8)
        {
            uint16x8_t vfenc, vpred;
#if HIGH_BIT_DEPTH
            vfenc = vld1q_u16(fenc + x);
            vpred = vld1q_u16(pred + x);
#else
            vfenc = vmovl_u8(vld1_u8(fenc + x));
            vpred = vmovl_u8(vld1_u8(pred + x));
#endif
            int16x8_t res = vreinterpretq_s16_u16(vsubq_u16(vfenc, vpred));
            vst1q_s16(residual + x, res);
        }
        for (; x < blockSize; x++)
        {
            residual[x] = static_cast<int16_t>(fenc[x]) - static_cast<int16_t>(pred[x]);
        }
        fenc += stride;
        residual += stride;
        pred += stride;
    }
}

#if HIGH_BIT_DEPTH
static inline int calc_energy_8x8(const uint16_t *source, intptr_t sstride)
{
    uint16x8_t s[8];
    load_u16x8xn<8>(source, sstride, s);

    int16x8_t in[8], temp[8];

    in[0] = vreinterpretq_s16_u16(vaddq_u16(s[0], s[1]));
    in[1] = vreinterpretq_s16_u16(vaddq_u16(s[2], s[3]));
    in[2] = vreinterpretq_s16_u16(vaddq_u16(s[4], s[5]));
    in[3] = vreinterpretq_s16_u16(vaddq_u16(s[6], s[7]));
    in[4] = vreinterpretq_s16_u16(vsubq_u16(s[0], s[1]));
    in[5] = vreinterpretq_s16_u16(vsubq_u16(s[2], s[3]));
    in[6] = vreinterpretq_s16_u16(vsubq_u16(s[4], s[5]));
    in[7] = vreinterpretq_s16_u16(vsubq_u16(s[6], s[7]));

    hadamard_4_v(in, temp);
    hadamard_4_v(in + 4, temp + 4);

    // The first line after the vertical hadamard transform contains the sum of coefficients.
    int sum = vaddlvq_s16(temp[0]) >> 2;

#if X265_DEPTH == 10
    uint16x8_t sa8_out[4];

    hadamard_8_h(temp, sa8_out);

    uint32x4_t res = vpaddlq_u16(sa8_out[0]);
    res = vpadalq_u16(res, sa8_out[1]);
    res = vpadalq_u16(res, sa8_out[2]);
    res = vpadalq_u16(res, sa8_out[3]);
#else // X265_DEPTH == 12
    uint32x4_t sa8_out[4];

    hadamard_8_h(temp, sa8_out);

    sa8_out[0] = vaddq_u32(sa8_out[0], sa8_out[1]);
    sa8_out[2] = vaddq_u32(sa8_out[2], sa8_out[3]);
    uint32x4_t res = vaddq_u32(sa8_out[0], sa8_out[2]);
#endif // X265_DEPTH == 10

    int sa8 = (vaddvq_u32(res) + 1) >> 1;

    return sa8 - sum;
}

#else // !HIGH_BIT_DEPTH
static inline int calc_energy_8x8(const uint8_t *source, intptr_t sstride)
{
    uint8x8_t s[8];
    load_u8x8xn<8>(source, sstride, s);

    int16x8_t in[8], temp[8];

    in[0] = vreinterpretq_s16_u16(vaddl_u8(s[0], s[1]));
    in[1] = vreinterpretq_s16_u16(vaddl_u8(s[2], s[3]));
    in[2] = vreinterpretq_s16_u16(vaddl_u8(s[4], s[5]));
    in[3] = vreinterpretq_s16_u16(vaddl_u8(s[6], s[7]));
    in[4] = vreinterpretq_s16_u16(vsubl_u8(s[0], s[1]));
    in[5] = vreinterpretq_s16_u16(vsubl_u8(s[2], s[3]));
    in[6] = vreinterpretq_s16_u16(vsubl_u8(s[4], s[5]));
    in[7] = vreinterpretq_s16_u16(vsubl_u8(s[6], s[7]));

    hadamard_4_v(in, temp);
    hadamard_4_v(in + 4, temp + 4);

    // The first line after the vertical hadamard transform contains the sum of coefficients.
    int sum = vaddvq_s16(temp[0]) >> 2;

    uint16x8_t sa8_out[4];
    hadamard_8_h(temp, sa8_out);

    uint16x8_t res = vaddq_u16(sa8_out[0], sa8_out[1]);
    res = vaddq_u16(res, sa8_out[2]);
    res = vaddq_u16(res, sa8_out[3]);

    int sa8 = (vaddlvq_u16(res) + 1) >> 1;

    return sa8 - sum;
}

#endif // HIGH_BIT_DEPTH

static inline int calc_energy_4x4(const pixel *source, intptr_t sstride)
{
#if HIGH_BIT_DEPTH
    uint16x4_t s[4];
    load_u16x4xn<4>(source, sstride, s);

    uint16x8_t s01 = vcombine_u16(s[0], s[1]);
    uint16x8_t s23 = vcombine_u16(s[2], s[3]);

    int16x8_t s01_23 = vreinterpretq_s16_u16(vaddq_u16(s01, s23));
    int16x8_t d01_23 = vreinterpretq_s16_u16(vsubq_u16(s01, s23));
#else
    uint8x8_t s[2];
    s[0] = load_u8x4x2(source + 0 * sstride, sstride);
    s[1] = load_u8x4x2(source + 2 * sstride, sstride);

    int16x8_t s01_23 = vreinterpretq_s16_u16(vaddl_u8(s[0], s[1]));
    int16x8_t d01_23 = vreinterpretq_s16_u16(vsubl_u8(s[0], s[1]));
#endif

    // The first line after the vertical hadamard transform contains the sum of coefficients.
    int sum = vaddvq_u16(vreinterpretq_u16_s16(s01_23)) >> 2;

    int16x8_t t0, t1;

    transpose_s16_s64x2(&t0, &t1, s01_23, d01_23);
    sumsubq_s16(&s01_23, &d01_23, t0, t1);

    transpose_s16_s16x2(&t0, &t1, s01_23, d01_23);
    sumsubq_s16(&s01_23, &d01_23, t0, t1);

    transpose_s16_s32x2(&t0, &t1, s01_23, d01_23);

    int sat = vaddvq_u16(max_abs_s16(t0, t1));

    return sat - sum;
}

template<int size>
int psyCost_pp_neon(const pixel *source, intptr_t sstride, const pixel *recon, intptr_t rstride)
{
    if (size)
    {
        int dim = 1 << (size + 2);
        uint32_t totEnergy = 0;
        for (int i = 0; i < dim; i += 8)
        {
            for (int j = 0; j < dim; j += 8)
            {
                int sourceEnergy = calc_energy_8x8(source + i * sstride + j, sstride);
                int reconEnergy = calc_energy_8x8(recon + i * rstride + j, rstride);

                totEnergy += abs(sourceEnergy - reconEnergy);
            }
        }
        return totEnergy;
    }
    else
    {
        int sourceEnergy = calc_energy_4x4(source, sstride);
        int reconEnergy = calc_energy_4x4(recon, rstride);

        return abs(sourceEnergy - reconEnergy);
    }
}


template<int w, int h>
// Calculate sa8d in blocks of 8x8
int sa8d8_neon(const pixel *pix1, intptr_t i_pix1, const pixel *pix2, intptr_t i_pix2)
{
    int cost = 0;

    for (int y = 0; y < h; y += 8)
        for (int x = 0; x < w; x += 8)
        {
            cost += pixel_sa8d_8x8_neon(pix1 + i_pix1 * y + x, i_pix1, pix2 + i_pix2 * y + x, i_pix2);
        }

    return cost;
}

template<int w, int h>
// Calculate sa8d in blocks of 16x16
int sa8d16_neon(const pixel *pix1, intptr_t i_pix1, const pixel *pix2, intptr_t i_pix2)
{
    int cost = 0;

    for (int y = 0; y < h; y += 16)
        for (int x = 0; x < w; x += 16)
        {
            cost += pixel_sa8d_16x16_neon(pix1 + i_pix1 * y + x, i_pix1, pix2 + i_pix2 * y + x, i_pix2);
        }

    return cost;
}

template<int size>
void cpy2Dto1D_shl_neon(int16_t *dst, const int16_t *src, intptr_t srcStride, int shift)
{
    X265_CHECK(((intptr_t)dst & 15) == 0, "dst alignment error\n");
    X265_CHECK((((intptr_t)src | (srcStride * sizeof(*src))) & 15) == 0 || size == 4, "src alignment error\n");
    X265_CHECK(shift >= 0, "invalid shift\n");

    for (int h = 0; h < size; h++)
    {
        int w = 0;
        for (; w + 16 <= size; w += 16)
        {
            int16x8_t a0_lo = vld1q_s16(src + w);
            int16x8_t a0_hi = vld1q_s16(src + w + 8);
            int16x8_t d0_lo = vshlq_s16(a0_lo, vdupq_n_s16(shift));
            int16x8_t d0_hi = vshlq_s16(a0_hi, vdupq_n_s16(shift));
            vst1q_s16(dst + w, d0_lo);
            vst1q_s16(dst + w + 8, d0_hi);
        }
        if (size == 8)
        {
            int16x8_t a0 = vld1q_s16(src + w);
            int16x8_t d0 = vshlq_s16(a0, vdupq_n_s16(shift));
            vst1q_s16(dst + w, d0);
        }
        if (size == 4)
        {
            int16x4_t a0 = vld1_s16(src + w);
            int16x4_t d0 = vshl_s16(a0, vdup_n_s16(shift));
            vst1_s16(dst + w, d0);
        }

        src += srcStride;
        dst += size;
    }
}

template<int size>
void cpy2Dto1D_shr_neon(int16_t* dst, const int16_t* src, intptr_t srcStride, int shift)
{
    X265_CHECK(((intptr_t)dst & 15) == 0, "dst alignment error\n");
    X265_CHECK((((intptr_t)src | (srcStride * sizeof(*src))) & 15) == 0 || size == 4, "src alignment error\n");
    X265_CHECK(shift > 0, "invalid shift\n");

    for (int h = 0; h < size; h++)
    {
        for (int w = 0; w + 16 <= size; w += 16)
        {
            int16x8_t s0_lo = vld1q_s16(src + w);
            int16x8_t s0_hi = vld1q_s16(src + w + 8);
            int16x8_t d0_lo = vrshlq_s16(s0_lo, vdupq_n_s16(-shift));
            int16x8_t d0_hi = vrshlq_s16(s0_hi, vdupq_n_s16(-shift));
            vst1q_s16(dst + w, d0_lo);
            vst1q_s16(dst + w + 8, d0_hi);
        }
        if (size == 8)
        {
            int16x8_t s0 = vld1q_s16(src);
            int16x8_t d0 = vrshlq_s16(s0, vdupq_n_s16(-shift));
            vst1q_s16(dst, d0);
        }
        if (size == 4)
        {
            int16x4_t s0 = vld1_s16(src);
            int16x4_t d0 = vrshl_s16(s0, vdup_n_s16(-shift));
            vst1_s16(dst, d0);
        }

        src += srcStride;
        dst += size;
    }
}

template<int w, int h>
int satd4_neon(const pixel *pix1, intptr_t stride_pix1, const pixel *pix2, intptr_t stride_pix2)
{
    int satd = 0;

    if (w == 4 && h == 4) {
        satd = pixel_satd_4x4_neon(pix1, stride_pix1, pix2, stride_pix2);
    } else {
        for (int row = 0; row < h; row += 8)
            for (int col = 0; col < w; col += 4)
                satd += pixel_satd_4x8_neon(pix1 + row * stride_pix1 + col, stride_pix1,
                                            pix2 + row * stride_pix2 + col, stride_pix2);
    }

    return satd;
}

template<int w, int h>
int satd8_neon(const pixel *pix1, intptr_t stride_pix1, const pixel *pix2, intptr_t stride_pix2)
{
    int satd = 0;

    if (w % 16 == 0 && h % 16 == 0)
    {
        for (int row = 0; row < h; row += 16)
            for (int col = 0; col < w; col += 16)
                satd += pixel_satd_16x16_neon(pix1 + row * stride_pix1 + col, stride_pix1,
                                              pix2 + row * stride_pix2 + col, stride_pix2);
    }
    else if (w % 8 == 0 && h % 16 == 0)
    {
        for (int row = 0; row < h; row += 16)
            for (int col = 0; col < w; col += 8)
                satd += pixel_satd_8x16_neon(pix1 + row * stride_pix1 + col, stride_pix1,
                                             pix2 + row * stride_pix2 + col, stride_pix2);
    }
    else if (w % 16 == 0 && h % 8 == 0)
    {
        for (int row = 0; row < h; row += 8)
            for (int col = 0; col < w; col += 16)
                satd += pixel_satd_16x8_neon(pix1 + row * stride_pix1 + col, stride_pix1,
                                             pix2 + row * stride_pix2 + col, stride_pix2);
    }
    else if (w % 16 == 0 && h % 4 == 0)
    {
        for (int row = 0; row < h; row += 4)
            for (int col = 0; col < w; col += 16)
                satd += pixel_satd_16x4_neon(pix1 + row * stride_pix1 + col, stride_pix1,
                                             pix2 + row * stride_pix2 + col, stride_pix2);
    }
    else if (w % 8 == 0 && h % 8 == 0)
    {
        for (int row = 0; row < h; row += 8)
            for (int col = 0; col < w; col += 8)
                satd += pixel_satd_8x8_neon(pix1 + row * stride_pix1 + col, stride_pix1,
                                            pix2 + row * stride_pix2 + col, stride_pix2);
    }
    else // w multiple of 8, h multiple of 4
    {
        for (int row = 0; row < h; row += 4)
            for (int col = 0; col < w; col += 8)
                satd += pixel_satd_8x4_neon(pix1 + row * stride_pix1 + col, stride_pix1,
                                            pix2 + row * stride_pix2 + col, stride_pix2);
    }

    return satd;
}


template<int blockSize>
void transpose_neon(pixel *dst, const pixel *src, intptr_t stride)
{
    for (int k = 0; k < blockSize; k++)
        for (int l = 0; l < blockSize; l++)
        {
            dst[k * blockSize + l] = src[l * stride + k];
        }
}


template<>
void transpose_neon<8>(pixel *dst, const pixel *src, intptr_t stride)
{
    transpose8x8(dst, src, 8, stride);
}

template<>
void transpose_neon<16>(pixel *dst, const pixel *src, intptr_t stride)
{
    transpose16x16(dst, src, 16, stride);
}

template<>
void transpose_neon<32>(pixel *dst, const pixel *src, intptr_t stride)
{
    transpose32x32(dst, src, 32, stride);
}


template<>
void transpose_neon<64>(pixel *dst, const pixel *src, intptr_t stride)
{
    transpose32x32(dst, src, 64, stride);
    transpose32x32(dst + 32 * 64 + 32, src + 32 * stride + 32, 64, stride);
    transpose32x32(dst + 32 * 64, src + 32, 64, stride);
    transpose32x32(dst + 32, src + 32 * stride, 64, stride);
}



};




namespace X265_NS
{


void setupPixelPrimitives_neon(EncoderPrimitives &p)
{
#define LUMA_PU(W, H) \
    p.pu[LUMA_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].addAvg[NONALIGNED] = addAvg_neon<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].addAvg[ALIGNED] = addAvg_neon<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].pixelavg_pp[NONALIGNED] = pixelavg_pp_neon<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].pixelavg_pp[ALIGNED] = pixelavg_pp_neon<W, H>;

#if !(HIGH_BIT_DEPTH)
#define LUMA_PU_S(W, H) \
    p.pu[LUMA_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].addAvg[NONALIGNED] = addAvg_neon<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].addAvg[ALIGNED] = addAvg_neon<W, H>;
#else // !(HIGH_BIT_DEPTH)
#define LUMA_PU_S(W, H) \
    p.pu[LUMA_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].addAvg[NONALIGNED] = addAvg_neon<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].addAvg[ALIGNED] = addAvg_neon<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].pixelavg_pp[NONALIGNED] = pixelavg_pp_neon<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].pixelavg_pp[ALIGNED] = pixelavg_pp_neon<W, H>;
#endif // !(HIGH_BIT_DEPTH)

#if HIGH_BIT_DEPTH
#define LUMA_CU(W, H) \
    p.cu[BLOCK_ ## W ## x ## H].sub_ps        = pixel_sub_ps_neon<W, H>; \
    p.cu[BLOCK_ ## W ## x ## H].add_ps[NONALIGNED]    = pixel_add_ps_neon<W, H>; \
    p.cu[BLOCK_ ## W ## x ## H].add_ps[ALIGNED] = pixel_add_ps_neon<W, H>; \
    p.cu[BLOCK_ ## W ## x ## H].blockfill_s[NONALIGNED] = blockfill_s_neon<W>; \
    p.cu[BLOCK_ ## W ## x ## H].blockfill_s[ALIGNED]    = blockfill_s_neon<W>; \
    p.cu[BLOCK_ ## W ## x ## H].copy_pp       = blockcopy_pp_neon<W, H>; \
    p.cu[BLOCK_ ## W ## x ## H].copy_ss       = blockcopy_ss_neon<W, H>; \
    p.cu[BLOCK_ ## W ## x ## H].cpy2Dto1D_shl = cpy2Dto1D_shl_neon<W>; \
    p.cu[BLOCK_ ## W ## x ## H].cpy1Dto2D_shl[NONALIGNED] = cpy1Dto2D_shl_neon<W>; \
    p.cu[BLOCK_ ## W ## x ## H].cpy1Dto2D_shl[ALIGNED] = cpy1Dto2D_shl_neon<W>; \
    p.cu[BLOCK_ ## W ## x ## H].psy_cost_pp   = psyCost_pp_neon<BLOCK_ ## W ## x ## H>; \
    p.cu[BLOCK_ ## W ## x ## H].transpose     = transpose_neon<W>;
#else  // !HIGH_BIT_DEPTH
#define LUMA_CU(W, H) \
    p.cu[BLOCK_ ## W ## x ## H].sub_ps        = pixel_sub_ps_neon<W, H>; \
    p.cu[BLOCK_ ## W ## x ## H].add_ps[NONALIGNED]    = pixel_add_ps_neon<W, H>; \
    p.cu[BLOCK_ ## W ## x ## H].add_ps[ALIGNED] = pixel_add_ps_neon<W, H>; \
    p.cu[BLOCK_ ## W ## x ## H].blockfill_s[NONALIGNED] = blockfill_s_neon<W>; \
    p.cu[BLOCK_ ## W ## x ## H].blockfill_s[ALIGNED]    = blockfill_s_neon<W>; \
    p.cu[BLOCK_ ## W ## x ## H].copy_pp       = blockcopy_pp_neon<W, H>; \
    p.cu[BLOCK_ ## W ## x ## H].copy_ps       = blockcopy_ps_neon<W, H>; \
    p.cu[BLOCK_ ## W ## x ## H].copy_ss       = blockcopy_ss_neon<W, H>; \
    p.cu[BLOCK_ ## W ## x ## H].copy_sp       = blockcopy_sp_neon<W, H>; \
    p.cu[BLOCK_ ## W ## x ## H].cpy2Dto1D_shl = cpy2Dto1D_shl_neon<W>; \
    p.cu[BLOCK_ ## W ## x ## H].cpy2Dto1D_shr = cpy2Dto1D_shr_neon<W>; \
    p.cu[BLOCK_ ## W ## x ## H].cpy1Dto2D_shl[NONALIGNED] = cpy1Dto2D_shl_neon<W>; \
    p.cu[BLOCK_ ## W ## x ## H].cpy1Dto2D_shl[ALIGNED] = cpy1Dto2D_shl_neon<W>; \
    p.cu[BLOCK_ ## W ## x ## H].cpy1Dto2D_shr = cpy1Dto2D_shr_neon<W>; \
    p.cu[BLOCK_ ## W ## x ## H].psy_cost_pp   = psyCost_pp_neon<BLOCK_ ## W ## x ## H>; \
    p.cu[BLOCK_ ## W ## x ## H].transpose     = transpose_neon<W>;
#endif // HIGH_BIT_DEPTH

    LUMA_PU_S(4, 4);
    LUMA_PU_S(8, 8);
    LUMA_PU(16, 16);
    LUMA_PU(32, 32);
    LUMA_PU(64, 64);
    LUMA_PU_S(4, 8);
    LUMA_PU_S(8, 4);
    LUMA_PU(16,  8);
    LUMA_PU_S(8, 16);
    LUMA_PU(16, 12);
    LUMA_PU(12, 16);
    LUMA_PU(16,  4);
    LUMA_PU_S(4, 16);
    LUMA_PU(32, 16);
    LUMA_PU(16, 32);
    LUMA_PU(32, 24);
    LUMA_PU(24, 32);
    LUMA_PU(32,  8);
    LUMA_PU_S(8, 32);
    LUMA_PU(64, 32);
    LUMA_PU(32, 64);
    LUMA_PU(64, 48);
    LUMA_PU(48, 64);
    LUMA_PU(64, 16);
    LUMA_PU(16, 64);

    p.pu[LUMA_4x4].satd   = satd4_neon<4, 4>;
    p.pu[LUMA_4x8].satd   = satd4_neon<4, 8>;
    p.pu[LUMA_4x16].satd  = satd4_neon<4, 16>;
    p.pu[LUMA_8x4].satd   = satd8_neon<8, 4>;
    p.pu[LUMA_8x8].satd   = satd8_neon<8, 8>;
    p.pu[LUMA_8x16].satd  = satd8_neon<8, 16>;
    p.pu[LUMA_8x32].satd  = satd8_neon<8, 32>;
    p.pu[LUMA_12x16].satd = satd4_neon<12, 16>;
    p.pu[LUMA_16x4].satd  = satd8_neon<16, 4>;
    p.pu[LUMA_16x8].satd  = satd8_neon<16, 8>;
    p.pu[LUMA_16x12].satd = satd8_neon<16, 12>;
    p.pu[LUMA_16x16].satd = satd8_neon<16, 16>;
    p.pu[LUMA_16x32].satd = satd8_neon<16, 32>;
    p.pu[LUMA_16x64].satd = satd8_neon<16, 64>;
    p.pu[LUMA_24x32].satd = satd8_neon<24, 32>;
    p.pu[LUMA_32x8].satd  = satd8_neon<32, 8>;
    p.pu[LUMA_32x16].satd = satd8_neon<32, 16>;
    p.pu[LUMA_32x24].satd = satd8_neon<32, 24>;
    p.pu[LUMA_32x32].satd = satd8_neon<32, 32>;
    p.pu[LUMA_32x64].satd = satd8_neon<32, 64>;
    p.pu[LUMA_48x64].satd = satd8_neon<48, 64>;
    p.pu[LUMA_64x16].satd = satd8_neon<64, 16>;
    p.pu[LUMA_64x32].satd = satd8_neon<64, 32>;
    p.pu[LUMA_64x48].satd = satd8_neon<64, 48>;
    p.pu[LUMA_64x64].satd = satd8_neon<64, 64>;


    LUMA_CU(4, 4);
    LUMA_CU(8, 8);
    LUMA_CU(16, 16);
    LUMA_CU(32, 32);
    LUMA_CU(64, 64);

#if !(HIGH_BIT_DEPTH)
    p.cu[BLOCK_8x8].var   = pixel_var_neon<8>;
    p.cu[BLOCK_16x16].var = pixel_var_neon<16>;
    p.cu[BLOCK_32x32].var = pixel_var_neon<32>;
    p.cu[BLOCK_64x64].var = pixel_var_neon<64>;
#endif // !(HIGH_BIT_DEPTH)


    p.cu[BLOCK_4x4].calcresidual[NONALIGNED]    = getResidual_neon<4>;
    p.cu[BLOCK_4x4].calcresidual[ALIGNED]       = getResidual_neon<4>;
    p.cu[BLOCK_8x8].calcresidual[NONALIGNED]    = getResidual_neon<8>;
    p.cu[BLOCK_8x8].calcresidual[ALIGNED]       = getResidual_neon<8>;
    p.cu[BLOCK_16x16].calcresidual[NONALIGNED]  = getResidual_neon<16>;
    p.cu[BLOCK_16x16].calcresidual[ALIGNED]     = getResidual_neon<16>;
    p.cu[BLOCK_32x32].calcresidual[NONALIGNED]  = getResidual_neon<32>;
    p.cu[BLOCK_32x32].calcresidual[ALIGNED]     = getResidual_neon<32>;

    p.cu[BLOCK_4x4].sa8d   = satd4_neon<4, 4>;
    p.cu[BLOCK_8x8].sa8d   = sa8d8_neon<8, 8>;
    p.cu[BLOCK_16x16].sa8d = sa8d16_neon<16, 16>;
    p.cu[BLOCK_32x32].sa8d = sa8d16_neon<32, 32>;
    p.cu[BLOCK_64x64].sa8d = sa8d16_neon<64, 64>;


#define CHROMA_PU_420(W, H) \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].addAvg[NONALIGNED]  = addAvg_neon<W, H>;         \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].addAvg[ALIGNED]  = addAvg_neon<W, H>;         \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \


    CHROMA_PU_420(4, 4);
    CHROMA_PU_420(8, 8);
    CHROMA_PU_420(16, 16);
    CHROMA_PU_420(32, 32);
    CHROMA_PU_420(4, 2);
    CHROMA_PU_420(8, 4);
    CHROMA_PU_420(4, 8);
    CHROMA_PU_420(8, 6);
    CHROMA_PU_420(6, 8);
    CHROMA_PU_420(8, 2);
    CHROMA_PU_420(2, 8);
    CHROMA_PU_420(16, 8);
    CHROMA_PU_420(8,  16);
    CHROMA_PU_420(16, 12);
    CHROMA_PU_420(12, 16);
    CHROMA_PU_420(16, 4);
    CHROMA_PU_420(4,  16);
    CHROMA_PU_420(32, 16);
    CHROMA_PU_420(16, 32);
    CHROMA_PU_420(32, 24);
    CHROMA_PU_420(24, 32);
    CHROMA_PU_420(32, 8);
    CHROMA_PU_420(8,  32);



    p.chroma[X265_CSP_I420].pu[CHROMA_420_2x2].satd   = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_2x4].satd   = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_2x8].satd   = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x2].satd   = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x4].satd   = satd4_neon<4, 4>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x8].satd   = satd4_neon<4, 8>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x16].satd  = satd4_neon<4, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_6x8].satd   = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x2].satd   = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x4].satd   = satd8_neon<8, 4>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x6].satd   = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x8].satd   = satd8_neon<8, 8>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x16].satd  = satd8_neon<8, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x32].satd  = satd8_neon<8, 32>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_12x16].satd = satd4_neon<12, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x4].satd  = satd8_neon<16, 4>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x8].satd  = satd8_neon<16, 8>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x12].satd = satd8_neon<16, 12>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x16].satd = satd8_neon<16, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x32].satd = satd8_neon<16, 32>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_24x32].satd = satd8_neon<24, 32>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x8].satd  = satd8_neon<32, 8>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x16].satd = satd8_neon<32, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x24].satd = satd8_neon<32, 24>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x32].satd = satd8_neon<32, 32>;

#if HIGH_BIT_DEPTH
#define CHROMA_CU_420(W, H) \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_ss = blockcopy_ss_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].sub_ps = pixel_sub_ps_neon<W, H>;  \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].add_ps[NONALIGNED] = pixel_add_ps_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].add_ps[ALIGNED] = pixel_add_ps_neon<W, H>;

#define CHROMA_CU_S_420(W, H) \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_ss = blockcopy_ss_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].sub_ps = pixel_sub_ps_neon<W, H>;  \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].add_ps[NONALIGNED] = pixel_add_ps_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].add_ps[ALIGNED] = pixel_add_ps_neon<W, H>;
#else // !HIGH_BIT_DEPTH
#define CHROMA_CU_420(W, H) \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_ps = blockcopy_ps_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_ss = blockcopy_ss_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_sp = blockcopy_sp_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].sub_ps = pixel_sub_ps_neon<W, H>;  \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].add_ps[NONALIGNED] = pixel_add_ps_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].add_ps[ALIGNED] = pixel_add_ps_neon<W, H>;

#define CHROMA_CU_S_420(W, H) \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_ps = blockcopy_ps_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_ss = blockcopy_ss_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].copy_sp = blockcopy_sp_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].sub_ps = pixel_sub_ps_neon<W, H>;  \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].add_ps[NONALIGNED] = pixel_add_ps_neon<W, H>; \
    p.chroma[X265_CSP_I420].cu[BLOCK_420_ ## W ## x ## H].add_ps[ALIGNED] = pixel_add_ps_neon<W, H>;
#endif // HIGH_BIT_DEPTH

    CHROMA_CU_S_420(4, 4)
    CHROMA_CU_420(8, 8)
    CHROMA_CU_420(16, 16)
    CHROMA_CU_420(32, 32)


    p.chroma[X265_CSP_I420].cu[BLOCK_8x8].sa8d   = p.chroma[X265_CSP_I420].pu[CHROMA_420_4x4].satd;
    p.chroma[X265_CSP_I420].cu[BLOCK_16x16].sa8d = sa8d8_neon<8, 8>;
    p.chroma[X265_CSP_I420].cu[BLOCK_32x32].sa8d = sa8d16_neon<16, 16>;
    p.chroma[X265_CSP_I420].cu[BLOCK_64x64].sa8d = sa8d16_neon<32, 32>;


#define CHROMA_PU_422(W, H) \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].addAvg[NONALIGNED]  = addAvg_neon<W, H>;         \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].addAvg[ALIGNED]  = addAvg_neon<W, H>;         \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \


    CHROMA_PU_422(4, 8);
    CHROMA_PU_422(8, 16);
    CHROMA_PU_422(16, 32);
    CHROMA_PU_422(32, 64);
    CHROMA_PU_422(4, 4);
    CHROMA_PU_422(2, 8);
    CHROMA_PU_422(8, 8);
    CHROMA_PU_422(4, 16);
    CHROMA_PU_422(8, 12);
    CHROMA_PU_422(6, 16);
    CHROMA_PU_422(8, 4);
    CHROMA_PU_422(2, 16);
    CHROMA_PU_422(16, 16);
    CHROMA_PU_422(8, 32);
    CHROMA_PU_422(16, 24);
    CHROMA_PU_422(12, 32);
    CHROMA_PU_422(16, 8);
    CHROMA_PU_422(4,  32);
    CHROMA_PU_422(32, 32);
    CHROMA_PU_422(16, 64);
    CHROMA_PU_422(32, 48);
    CHROMA_PU_422(24, 64);
    CHROMA_PU_422(32, 16);
    CHROMA_PU_422(8,  64);


    p.chroma[X265_CSP_I422].pu[CHROMA_422_2x4].satd   = NULL;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_2x8].satd   = NULL;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_2x16].satd  = NULL;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x4].satd   = satd4_neon<4, 4>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x8].satd   = satd4_neon<4, 8>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x16].satd  = satd4_neon<4, 16>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x32].satd  = satd4_neon<4, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_6x16].satd  = NULL;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x4].satd   = satd8_neon<8, 4>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x8].satd   = satd8_neon<8, 8>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x12].satd  = satd8_neon<8, 12>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x16].satd  = satd8_neon<8, 16>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x32].satd  = satd8_neon<8, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x64].satd  = satd8_neon<8, 64>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_12x32].satd = satd4_neon<12, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x8].satd  = satd8_neon<16, 8>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x16].satd = satd8_neon<16, 16>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x24].satd = satd8_neon<16, 24>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x32].satd = satd8_neon<16, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x64].satd = satd8_neon<16, 64>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_24x64].satd = satd8_neon<24, 64>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x16].satd = satd8_neon<32, 16>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x32].satd = satd8_neon<32, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x48].satd = satd8_neon<32, 48>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x64].satd = satd8_neon<32, 64>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_12x32].satd = satd4_neon<12, 32>;


#if HIGH_BIT_DEPTH
#define CHROMA_CU_422(W, H) \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_ss = blockcopy_ss_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].sub_ps = pixel_sub_ps_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].add_ps[NONALIGNED] = pixel_add_ps_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].add_ps[ALIGNED] = pixel_add_ps_neon<W, H>;

#define CHROMA_CU_S_422(W, H) \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_ss = blockcopy_ss_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].sub_ps = pixel_sub_ps_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].add_ps[NONALIGNED] = pixel_add_ps_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].add_ps[ALIGNED] = pixel_add_ps_neon<W, H>;
#else // !HIGH_BIT_DEPTH
#define CHROMA_CU_422(W, H) \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_ps = blockcopy_ps_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_ss = blockcopy_ss_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_sp = blockcopy_sp_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].sub_ps = pixel_sub_ps_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].add_ps[NONALIGNED] = pixel_add_ps_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].add_ps[ALIGNED] = pixel_add_ps_neon<W, H>;

#define CHROMA_CU_S_422(W, H) \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_pp = blockcopy_pp_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_ps = blockcopy_ps_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_ss = blockcopy_ss_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].copy_sp = blockcopy_sp_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].sub_ps = pixel_sub_ps_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].add_ps[NONALIGNED] = pixel_add_ps_neon<W, H>; \
    p.chroma[X265_CSP_I422].cu[BLOCK_422_ ## W ## x ## H].add_ps[ALIGNED] = pixel_add_ps_neon<W, H>;
#endif // HIGH_BIT_DEPTH


    CHROMA_CU_S_422(4, 8)
    CHROMA_CU_422(8, 16)
    CHROMA_CU_422(16, 32)
    CHROMA_CU_422(32, 64)

    p.chroma[X265_CSP_I422].cu[BLOCK_8x8].sa8d       = p.chroma[X265_CSP_I422].pu[CHROMA_422_4x8].satd;
    p.chroma[X265_CSP_I422].cu[BLOCK_16x16].sa8d     = sa8d8_neon<8, 16>;
    p.chroma[X265_CSP_I422].cu[BLOCK_32x32].sa8d     = sa8d16_neon<16, 32>;
    p.chroma[X265_CSP_I422].cu[BLOCK_64x64].sa8d     = sa8d16_neon<32, 64>;

    p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].sa8d  = sa8d8_neon<8, 16>;
    p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].sa8d = sa8d16_neon<16, 32>;
    p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].sa8d = sa8d16_neon<32, 64>;

    p.weight_pp = weight_pp_neon;

    p.planecopy_cp = planecopy_cp_neon;
}


}


#endif

