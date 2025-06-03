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

namespace X265_NS
{


void setupFilterPrimitives_neon(EncoderPrimitives &p);

};


#endif

