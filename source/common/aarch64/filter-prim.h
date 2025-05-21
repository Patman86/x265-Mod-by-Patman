#ifndef _FILTER_PRIM_ARM64_H__
#define _FILTER_PRIM_ARM64_H__


#include "common.h"
#include "slicetype.h"      // LOWRES_COST_MASK
#include "primitives.h"
#include "x265.h"

#include <arm_neon.h>

static inline int16x8_t vtbl2q_s32_s16(int32x4_t a, int32x4_t b, uint8x16_t index)
{
    uint8x16x2_t ab;

    ab.val[0] = vreinterpretq_u8_s32(a);
    ab.val[1] = vreinterpretq_u8_s32(b);

    return vreinterpretq_s16_u8(vqtbl2q_u8(ab, index));
}

template<int coeffIdx>
void inline filter8_s16x4(const int16x4_t *s, const int16x8_t filter,
                          const int32x4_t c, int32x4_t &d)
{
    if (coeffIdx == 1)
    {
        d = vsubl_s16(s[6], s[0]);
        d = vaddq_s32(d, c);
        d = vmlal_laneq_s16(d, s[1], filter, 1);
        d = vmlal_laneq_s16(d, s[2], filter, 2);
        d = vmlal_laneq_s16(d, s[3], filter, 3);
        d = vmlal_laneq_s16(d, s[4], filter, 4);
        d = vmlal_laneq_s16(d, s[5], filter, 5);
    }
    else if (coeffIdx == 2)
    {
        int16x4_t sum07 = vadd_s16(s[0], s[7]);
        int16x4_t sum16 = vadd_s16(s[1], s[6]);
        int16x4_t sum25 = vadd_s16(s[2], s[5]);
        int16x4_t sum34 = vadd_s16(s[3], s[4]);

        int32x4_t sum12356 =  vmlal_laneq_s16(c, sum16, filter, 1);
        sum12356 = vmlal_laneq_s16(sum12356, sum25, filter, 2);
        sum12356 = vmlal_laneq_s16(sum12356, sum34, filter, 3);

        d = vsubw_s16(sum12356, sum07);
    }
    else
    {
        d = vsubl_s16(s[1], s[7]);
        d = vaddq_s32(d, c);
        d = vmlal_laneq_s16(d, s[2], filter, 2);
        d = vmlal_laneq_s16(d, s[3], filter, 3);
        d = vmlal_laneq_s16(d, s[4], filter, 4);
        d = vmlal_laneq_s16(d, s[5], filter, 5);
        d = vmlal_laneq_s16(d, s[6], filter, 6);
    }
}

template<int coeffIdx>
void inline filter8_s16x8(const int16x8_t *s, const int16x8_t filter,
                          const int32x4_t c, int32x4_t &d0, int32x4_t &d1)
{
    if (coeffIdx == 1)
    {
        d0 = vsubl_s16(vget_low_s16(s[6]), vget_low_s16(s[0]));
        d0 = vaddq_s32(d0, c);
        d0 = vmlal_laneq_s16(d0, vget_low_s16(s[1]), filter, 1);
        d0 = vmlal_laneq_s16(d0, vget_low_s16(s[2]), filter, 2);
        d0 = vmlal_laneq_s16(d0, vget_low_s16(s[3]), filter, 3);
        d0 = vmlal_laneq_s16(d0, vget_low_s16(s[4]), filter, 4);
        d0 = vmlal_laneq_s16(d0, vget_low_s16(s[5]), filter, 5);

        d1 = vsubl_s16(vget_high_s16(s[6]), vget_high_s16(s[0]));
        d1 = vaddq_s32(d1, c);
        d1 = vmlal_laneq_s16(d1, vget_high_s16(s[1]), filter, 1);
        d1 = vmlal_laneq_s16(d1, vget_high_s16(s[2]), filter, 2);
        d1 = vmlal_laneq_s16(d1, vget_high_s16(s[3]), filter, 3);
        d1 = vmlal_laneq_s16(d1, vget_high_s16(s[4]), filter, 4);
        d1 = vmlal_laneq_s16(d1, vget_high_s16(s[5]), filter, 5);
    }
    else if (coeffIdx == 2)
    {
        int16x8_t sum07 = vaddq_s16(s[0], s[7]);
        int16x8_t sum16 = vaddq_s16(s[1], s[6]);
        int16x8_t sum25 = vaddq_s16(s[2], s[5]);
        int16x8_t sum34 = vaddq_s16(s[3], s[4]);

        int32x4_t sum123456_lo = vmlal_laneq_s16(c, vget_low_s16(sum16), filter, 1);
        sum123456_lo = vmlal_laneq_s16(sum123456_lo, vget_low_s16(sum25), filter, 2);
        sum123456_lo = vmlal_laneq_s16(sum123456_lo, vget_low_s16(sum34), filter, 3);

        int32x4_t sum123456_hi = vmlal_laneq_s16(c, vget_high_s16(sum16), filter, 1);
        sum123456_hi = vmlal_laneq_s16(sum123456_hi, vget_high_s16(sum25), filter, 2);
        sum123456_hi = vmlal_laneq_s16(sum123456_hi, vget_high_s16(sum34), filter, 3);

        d0 = vsubw_s16(sum123456_lo, vget_low_s16(sum07));
        d1 = vsubw_s16(sum123456_hi, vget_high_s16(sum07));
    }
    else
    {
        int16x8_t sum17 = vsubq_s16(s[1], s[7]);
        d0 = vaddw_s16(c, vget_low_s16(sum17));
        d1 = vaddw_s16(c, vget_high_s16(sum17));

        d0 = vmlal_laneq_s16(d0, vget_low_s16(s[2]), filter, 2);
        d0 = vmlal_laneq_s16(d0, vget_low_s16(s[3]), filter, 3);
        d0 = vmlal_laneq_s16(d0, vget_low_s16(s[4]), filter, 4);
        d0 = vmlal_laneq_s16(d0, vget_low_s16(s[5]), filter, 5);
        d0 = vmlal_laneq_s16(d0, vget_low_s16(s[6]), filter, 6);

        d1 = vmlal_laneq_s16(d1, vget_high_s16(s[2]), filter, 2);
        d1 = vmlal_laneq_s16(d1, vget_high_s16(s[3]), filter, 3);
        d1 = vmlal_laneq_s16(d1, vget_high_s16(s[4]), filter, 4);
        d1 = vmlal_laneq_s16(d1, vget_high_s16(s[5]), filter, 5);
        d1 = vmlal_laneq_s16(d1, vget_high_s16(s[6]), filter, 6);
    }
}

namespace X265_NS
{


void setupFilterPrimitives_neon(EncoderPrimitives &p);

};


#endif

