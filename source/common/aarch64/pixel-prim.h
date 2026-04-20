#ifndef PIXEL_PRIM_NEON_H__
#define PIXEL_PRIM_NEON_H__

#include "common.h"
#include "primitives.h"
#include "slicetype.h" // LOWRES_COST_MASK
#include "x265.h"

#include "mem-neon.h"

#include <arm_neon.h>

namespace X265_NS
{

static inline void sumsubq_s16(int16x8_t *sum, int16x8_t *sub, const int16x8_t a,
                               const int16x8_t b)
{
    *sum = vaddq_s16(a, b);
    *sub = vsubq_s16(a, b);
}

static inline void abssumsubq_s16(int16x8_t *sum, int16x8_t *sub, const int16x8_t a,
                                  const int16x8_t b)
{
    *sum = vabsq_s16(vaddq_s16(a, b));
    *sub = vabdq_s16(a, b);
}

static inline void transpose_s16_s16x2(int16x8_t *t1, int16x8_t *t2, const int16x8_t s1,
                                       const int16x8_t s2)
{
    *t1 = vtrn1q_s16(s1, s2);
    *t2 = vtrn2q_s16(s1, s2);
}

static inline void transpose_s16_s32x2(int16x8_t *t1, int16x8_t *t2, const int16x8_t s1,
                                       const int16x8_t s2)
{
    int32x4_t tmp1 = vreinterpretq_s32_s16(s1);
    int32x4_t tmp2 = vreinterpretq_s32_s16(s2);

    *t1 = vreinterpretq_s16_s32(vtrn1q_s32(tmp1, tmp2));
    *t2 = vreinterpretq_s16_s32(vtrn2q_s32(tmp1, tmp2));
}

static inline void transpose_s16_s64x2(int16x8_t *t1, int16x8_t *t2, const int16x8_t s1,
                                       const int16x8_t s2)
{
    int64x2_t tmp1 = vreinterpretq_s64_s16(s1);
    int64x2_t tmp2 = vreinterpretq_s64_s16(s2);

    *t1 = vreinterpretq_s16_s64(vtrn1q_s64(tmp1, tmp2));
    *t2 = vreinterpretq_s16_s64(vtrn2q_s64(tmp1, tmp2));
}

#if X265_DEPTH == 12

static inline void abssumsubq_s32(int32x4_t *sum, int32x4_t *sub, const int32x4_t a,
                                  const int32x4_t b)
{
    *sum = vabsq_s32(vaddq_s32(a, b));
    *sub = vabdq_s32(a, b);
}

static inline void sumsublq_s16(int32x4_t *sum_lo, int32x4_t *sum_hi, int32x4_t *sub_lo,
                                int32x4_t *sub_hi, const int16x8_t a, const int16x8_t b)
{
    *sum_lo = vaddl_s16(vget_low_s16(a), vget_low_s16(b));
    *sub_lo = vsubl_s16(vget_low_s16(a), vget_low_s16(b));
    *sum_hi = vaddl_s16(vget_high_s16(a), vget_high_s16(b));
    *sub_hi = vsubl_s16(vget_high_s16(a), vget_high_s16(b));
}

#endif // X265_DEPTH == 12

#if HIGH_BIT_DEPTH
static inline void load_diff_u16x8x4(const uint16_t *pix1, intptr_t stride_pix1,
                                     const uint16_t *pix2, intptr_t stride_pix2,
                                     int16x8_t diff[4])
{
    uint16x8_t s[4], r[4];
    load_u16x8xn<4>(pix1, stride_pix1, s);
    load_u16x8xn<4>(pix2, stride_pix2, r);

    diff[0] = vreinterpretq_s16_u16(vsubq_u16(s[0], r[0]));
    diff[1] = vreinterpretq_s16_u16(vsubq_u16(s[1], r[1]));
    diff[2] = vreinterpretq_s16_u16(vsubq_u16(s[2], r[2]));
    diff[3] = vreinterpretq_s16_u16(vsubq_u16(s[3], r[3]));
}

static inline void load_diff_u16x8x4_dual(const uint16_t *pix1, intptr_t stride_pix1,
                                          const uint16_t *pix2, intptr_t stride_pix2,
                                          int16x8_t diff[8])
{
    load_diff_u16x8x4(pix1 + 0 * stride_pix1, stride_pix1, pix2 + 0 * stride_pix2,
                      stride_pix2, diff);
    load_diff_u16x8x4(pix1 + 4 * stride_pix1, stride_pix1, pix2 + 4 * stride_pix2,
                      stride_pix2, diff + 4);
}

static inline void load_diff_u16x8x8(const uint16_t *pix1, intptr_t stride_pix1,
                                     const uint16_t *pix2, intptr_t stride_pix2,
                                     int16x8_t diff[8])
{
    uint16x8_t s[8], r[8];
    load_u16x8xn<8>(pix1, stride_pix1, s);
    load_u16x8xn<8>(pix2, stride_pix2, r);

    diff[0] = vreinterpretq_s16_u16(vsubq_u16(s[0], r[0]));
    diff[1] = vreinterpretq_s16_u16(vsubq_u16(s[1], r[1]));
    diff[2] = vreinterpretq_s16_u16(vsubq_u16(s[2], r[2]));
    diff[3] = vreinterpretq_s16_u16(vsubq_u16(s[3], r[3]));
    diff[4] = vreinterpretq_s16_u16(vsubq_u16(s[4], r[4]));
    diff[5] = vreinterpretq_s16_u16(vsubq_u16(s[5], r[5]));
    diff[6] = vreinterpretq_s16_u16(vsubq_u16(s[6], r[6]));
    diff[7] = vreinterpretq_s16_u16(vsubq_u16(s[7], r[7]));
}

#else // !HIGH_BIT_DEPTH
static inline void load_diff_u8x8x4(const uint8_t *pix1, intptr_t stride_pix1,
                                    const uint8_t *pix2, intptr_t stride_pix2,
                                    int16x8_t diff[4])
{
    uint8x8_t s[4], r[4];
    load_u8x8xn<4>(pix1, stride_pix1, s);
    load_u8x8xn<4>(pix2, stride_pix2, r);

    diff[0] = vreinterpretq_s16_u16(vsubl_u8(s[0], r[0]));
    diff[1] = vreinterpretq_s16_u16(vsubl_u8(s[1], r[1]));
    diff[2] = vreinterpretq_s16_u16(vsubl_u8(s[2], r[2]));
    diff[3] = vreinterpretq_s16_u16(vsubl_u8(s[3], r[3]));
}

static inline void load_diff_u8x8x8(const uint8_t *pix1, intptr_t stride_pix1,
                                    const uint8_t *pix2, intptr_t stride_pix2,
                                    int16x8_t diff[8])
{
    load_diff_u8x8x4(pix1 + 0 * stride_pix1, stride_pix1, pix2 + 0 * stride_pix2,
                     stride_pix2, diff);
    load_diff_u8x8x4(pix1 + 4 * stride_pix1, stride_pix1, pix2 + 4 * stride_pix2,
                     stride_pix2, diff + 4);
}

static inline void load_diff_u8x16x4(const uint8_t *pix1, intptr_t stride_pix1,
                                     const uint8_t *pix2, intptr_t stride_pix2,
                                     int16x8_t diff[8])
{
    uint8x16_t s[4], r[4];
    load_u8x16xn<4>(pix1, stride_pix1, s);
    load_u8x16xn<4>(pix2, stride_pix2, r);

    diff[0] = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s[0]), vget_low_u8(r[0])));
    diff[1] = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s[1]), vget_low_u8(r[1])));
    diff[2] = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s[2]), vget_low_u8(r[2])));
    diff[3] = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s[3]), vget_low_u8(r[3])));
    diff[4] = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s[0]), vget_high_u8(r[0])));
    diff[5] = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s[1]), vget_high_u8(r[1])));
    diff[6] = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s[2]), vget_high_u8(r[2])));
    diff[7] = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s[3]), vget_high_u8(r[3])));
}

#endif // HIGH_BIT_DEPTH

void setupPixelPrimitives_neon(EncoderPrimitives &p);

#if defined(HAVE_NEON_DOTPROD)
void setupPixelPrimitives_neon_dotprod(EncoderPrimitives &p);
#endif

#if defined(HAVE_SVE) && HAVE_SVE_BRIDGE
void setupPixelPrimitives_sve(EncoderPrimitives &p);
#endif

#if defined(HAVE_SVE2) && HAVE_SVE_BRIDGE
void setupPixelPrimitives_sve2(EncoderPrimitives &p);
#endif
} // namespace X265_NS

#endif
