/*****************************************************************************
 * Copyright (C) 2026 MulticoreWare, Inc
 *
 * Authors: Alex Davicenko <alex.davicenko@arm.com>
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

#include "common.h"
#include "mem-neon.h"
#include "neon-sve-bridge.h"
#include "neon-sve2-bridge.h"
#include "pixel-prim.h"
#include "primitives.h"

#include <arm_neon.h>

#if defined(HAVE_SVE2) && HAVE_SVE_BRIDGE

static inline int16x8_t vtbl2q_s16(int16x8_t a, int16x8_t b, uint8x16_t index)
{
    uint8x16x2_t ab = {vreinterpretq_u8_s16(a), vreinterpretq_u8_s16(b)};

    return vreinterpretq_s16_u8(vqtbl2q_u8(ab, index));
}

static inline int16x8_t vqtbl1q_s16(int16x8_t a, uint8x16_t index)
{
    return vreinterpretq_s16_s8(vqtbl1q_s8(vreinterpretq_s8_s16(a), index));
}

// Swap 16-bit values in the following way:
// {1, 3, 0, 2,  9, 11,  8, 10}
// {5, 7, 4, 6, 13, 15, 12, 14}
const static uint8_t kSwapTransposeTbl[] = { 2,  3,  6,  7,  0,  1,  4,  5,
                                            18, 19, 22, 23, 16, 17, 20, 21,
                                            10, 11, 14, 15,  8,  9, 12, 13,
                                            26, 27, 30, 31, 24, 25, 28, 29};

static inline void hadamard_4x4(int16x8_t diff[2], uint64x2_t *sum)
{
    int16x8_t a0 = x265_caddq_s16<90>(diff[0], diff[0]);
    int16x8_t a1 = x265_caddq_s16<90>(diff[1], diff[1]);

    int16x8_t b0, b1;
    sumsubq_s16(&b0, &b1, a0, a1);

    const uint8x16_t idx0 = vld1q_u8(kSwapTransposeTbl);
    const uint8x16_t idx1 = vld1q_u8(kSwapTransposeTbl + 16);
    // Re-order input ready for CADD instruction.
    a0 = vtbl2q_s16(b0, b1, idx0);
    a1 = vtbl2q_s16(b0, b1, idx1);

    b0 = x265_caddq_s16<90>(a0, a0);
    b1 = x265_caddq_s16<90>(a1, a1);

    a0 = vabsq_s16(b0);
    a1 = vabsq_s16(b1);

    uint16x8_t max = vmaxq_u16(vreinterpretq_u16_s16(a0), vreinterpretq_u16_s16(a1));
    *sum = x265_udotq_u16(*sum, max, vdupq_n_u16(1));
}

// Swap 16-bit values in the following way:
// {0, 2, 4, 6, 1, 3, 5, 7}
const static uint8_t kHADPermuteTbl[] = {0, 1, 4, 5, 8,  9,  12, 13,
                                         2, 3, 6, 7, 10, 11, 14, 15};

// Calculate 2 4x4 hadamard transformations.
static inline void hadamard_4x4_dual(int16x8_t diff[4], uint64x2_t *sum)
{
    int16x8_t a0 = x265_caddq_s16<90>(diff[0], diff[0]);
    int16x8_t a1 = x265_caddq_s16<90>(diff[1], diff[1]);
    int16x8_t a2 = x265_caddq_s16<90>(diff[2], diff[2]);
    int16x8_t a3 = x265_caddq_s16<90>(diff[3], diff[3]);

    const uint8x16_t idx = vld1q_u8(kHADPermuteTbl);
    // Re-order input ready for CADD instruction.
    int16x8_t b0 = vqtbl1q_s16(a0, idx);
    int16x8_t b1 = vqtbl1q_s16(a1, idx);
    int16x8_t b2 = vqtbl1q_s16(a2, idx);
    int16x8_t b3 = vqtbl1q_s16(a3, idx);

    a0 = x265_caddq_s16<90>(b0, b0);
    a1 = x265_caddq_s16<90>(b1, b1);
    a2 = x265_caddq_s16<90>(b2, b2);
    a3 = x265_caddq_s16<90>(b3, b3);

    abssumsubq_s16(&b0, &b1, a0, a1);
    abssumsubq_s16(&b2, &b3, a2, a3);

    uint16x8_t max0 = vmaxq_u16(vreinterpretq_u16_s16(b0), vreinterpretq_u16_s16(b2));
    uint16x8_t max1 = vmaxq_u16(vreinterpretq_u16_s16(b1), vreinterpretq_u16_s16(b3));

    *sum = x265_udotq_u16(*sum, vaddq_u16(max0, max1), vdupq_n_u16(1));
}

// Calculate 4 4x4 hadamard transformations.
static inline void hadamard_4x4_quad(int16x8_t diff[8], uint64x2_t *sum)
{
    int16x8_t a[8], b[8];

    a[0] = x265_caddq_s16<90>(diff[0], diff[0]);
    a[1] = x265_caddq_s16<90>(diff[1], diff[1]);
    a[2] = x265_caddq_s16<90>(diff[2], diff[2]);
    a[3] = x265_caddq_s16<90>(diff[3], diff[3]);
    a[4] = x265_caddq_s16<90>(diff[4], diff[4]);
    a[5] = x265_caddq_s16<90>(diff[5], diff[5]);
    a[6] = x265_caddq_s16<90>(diff[6], diff[6]);
    a[7] = x265_caddq_s16<90>(diff[7], diff[7]);

    const uint8x16_t idx = vld1q_u8(kHADPermuteTbl);
    // Re-order input ready for CADD instruction.
    b[0] = vqtbl1q_s16(a[0], idx);
    b[1] = vqtbl1q_s16(a[1], idx);
    b[2] = vqtbl1q_s16(a[2], idx);
    b[3] = vqtbl1q_s16(a[3], idx);
    b[4] = vqtbl1q_s16(a[4], idx);
    b[5] = vqtbl1q_s16(a[5], idx);
    b[6] = vqtbl1q_s16(a[6], idx);
    b[7] = vqtbl1q_s16(a[7], idx);

    a[0] = x265_caddq_s16<90>(b[0], b[0]);
    a[1] = x265_caddq_s16<90>(b[1], b[1]);
    a[2] = x265_caddq_s16<90>(b[2], b[2]);
    a[3] = x265_caddq_s16<90>(b[3], b[3]);
    a[4] = x265_caddq_s16<90>(b[4], b[4]);
    a[5] = x265_caddq_s16<90>(b[5], b[5]);
    a[6] = x265_caddq_s16<90>(b[6], b[6]);
    a[7] = x265_caddq_s16<90>(b[7], b[7]);

    abssumsubq_s16(&b[0], &b[1], a[0], a[1]);
    abssumsubq_s16(&b[2], &b[3], a[2], a[3]);
    abssumsubq_s16(&b[4], &b[5], a[4], a[5]);
    abssumsubq_s16(&b[6], &b[7], a[6], a[7]);

    uint16x8_t max0 = vmaxq_u16(vreinterpretq_u16_s16(b[0]), vreinterpretq_u16_s16(b[2]));
    uint16x8_t max1 = vmaxq_u16(vreinterpretq_u16_s16(b[1]), vreinterpretq_u16_s16(b[3]));
    uint16x8_t max2 = vmaxq_u16(vreinterpretq_u16_s16(b[4]), vreinterpretq_u16_s16(b[6]));
    uint16x8_t max3 = vmaxq_u16(vreinterpretq_u16_s16(b[5]), vreinterpretq_u16_s16(b[7]));

    uint16x8_t sum0 = vaddq_u16(max0, max1);
    uint16x8_t sum1 = vaddq_u16(max2, max3);
    *sum = x265_udotq_u16(*sum, sum0, vdupq_n_u16(1));
    *sum = x265_udotq_u16(*sum, sum1, vdupq_n_u16(1));
}

#if X265_DEPTH == 8
static inline void hadamard_8x8(int16x8_t diff[8], uint64x2_t *sum)
{
    int16x8_t a[8], b[8];

    a[0] = x265_caddq_s16<90>(diff[0], diff[0]);
    a[1] = x265_caddq_s16<90>(diff[1], diff[1]);
    a[2] = x265_caddq_s16<90>(diff[2], diff[2]);
    a[3] = x265_caddq_s16<90>(diff[3], diff[3]);
    a[4] = x265_caddq_s16<90>(diff[4], diff[4]);
    a[5] = x265_caddq_s16<90>(diff[5], diff[5]);
    a[6] = x265_caddq_s16<90>(diff[6], diff[6]);
    a[7] = x265_caddq_s16<90>(diff[7], diff[7]);

    const uint8x16_t idx = vld1q_u8(kHADPermuteTbl);
    // Re-order input ready for CADD instruction.
    b[0] = vqtbl1q_s16(a[0], idx);
    b[1] = vqtbl1q_s16(a[1], idx);
    b[2] = vqtbl1q_s16(a[2], idx);
    b[3] = vqtbl1q_s16(a[3], idx);
    b[4] = vqtbl1q_s16(a[4], idx);
    b[5] = vqtbl1q_s16(a[5], idx);
    b[6] = vqtbl1q_s16(a[6], idx);
    b[7] = vqtbl1q_s16(a[7], idx);

    a[0] = x265_caddq_s16<90>(b[0], b[0]);
    a[1] = x265_caddq_s16<90>(b[1], b[1]);
    a[2] = x265_caddq_s16<90>(b[2], b[2]);
    a[3] = x265_caddq_s16<90>(b[3], b[3]);
    a[4] = x265_caddq_s16<90>(b[4], b[4]);
    a[5] = x265_caddq_s16<90>(b[5], b[5]);
    a[6] = x265_caddq_s16<90>(b[6], b[6]);
    a[7] = x265_caddq_s16<90>(b[7], b[7]);

    // Re-order input ready for CADD instruction.
    b[0] = vqtbl1q_s16(a[0], idx);
    b[1] = vqtbl1q_s16(a[1], idx);
    b[2] = vqtbl1q_s16(a[2], idx);
    b[3] = vqtbl1q_s16(a[3], idx);
    b[4] = vqtbl1q_s16(a[4], idx);
    b[5] = vqtbl1q_s16(a[5], idx);
    b[6] = vqtbl1q_s16(a[6], idx);
    b[7] = vqtbl1q_s16(a[7], idx);

    a[0] = x265_caddq_s16<90>(b[0], b[0]);
    a[1] = x265_caddq_s16<90>(b[1], b[1]);
    a[2] = x265_caddq_s16<90>(b[2], b[2]);
    a[3] = x265_caddq_s16<90>(b[3], b[3]);
    a[4] = x265_caddq_s16<90>(b[4], b[4]);
    a[5] = x265_caddq_s16<90>(b[5], b[5]);
    a[6] = x265_caddq_s16<90>(b[6], b[6]);
    a[7] = x265_caddq_s16<90>(b[7], b[7]);

    sumsubq_s16(&b[0], &b[1], a[0], a[1]);
    sumsubq_s16(&b[2], &b[3], a[2], a[3]);
    sumsubq_s16(&b[4], &b[5], a[4], a[5]);
    sumsubq_s16(&b[6], &b[7], a[6], a[7]);

    abssumsubq_s16(&a[0], &a[2], b[0], b[2]);
    abssumsubq_s16(&a[1], &a[3], b[1], b[3]);
    abssumsubq_s16(&a[4], &a[6], b[4], b[6]);
    abssumsubq_s16(&a[5], &a[7], b[5], b[7]);

    uint16x8_t max0 = vmaxq_u16(vreinterpretq_u16_s16(a[0]), vreinterpretq_u16_s16(a[4]));
    uint16x8_t max1 = vmaxq_u16(vreinterpretq_u16_s16(a[1]), vreinterpretq_u16_s16(a[5]));
    uint16x8_t max2 = vmaxq_u16(vreinterpretq_u16_s16(a[2]), vreinterpretq_u16_s16(a[6]));
    uint16x8_t max3 = vmaxq_u16(vreinterpretq_u16_s16(a[3]), vreinterpretq_u16_s16(a[7]));

    uint16x8_t sum0 = vaddq_u16(max0, max1);
    uint16x8_t sum1 = vaddq_u16(max2, max3);

    *sum = x265_udotq_u16(*sum, sum0, vdupq_n_u16(1));
    *sum = x265_udotq_u16(*sum, sum1, vdupq_n_u16(1));
}

#elif X265_DEPTH == 10
static inline void hadamard_8x8(int16x8_t diff[8], uint64x2_t *sum)
{
    int16x8_t a[8], b[8];

    a[0] = x265_caddq_s16<90>(diff[0], diff[0]);
    a[1] = x265_caddq_s16<90>(diff[1], diff[1]);
    a[2] = x265_caddq_s16<90>(diff[2], diff[2]);
    a[3] = x265_caddq_s16<90>(diff[3], diff[3]);
    a[4] = x265_caddq_s16<90>(diff[4], diff[4]);
    a[5] = x265_caddq_s16<90>(diff[5], diff[5]);
    a[6] = x265_caddq_s16<90>(diff[6], diff[6]);
    a[7] = x265_caddq_s16<90>(diff[7], diff[7]);

    const uint8x16_t idx = vld1q_u8(kHADPermuteTbl);
    // Re-order input ready for CADD instruction.
    b[0] = vqtbl1q_s16(a[0], idx);
    b[1] = vqtbl1q_s16(a[1], idx);
    b[2] = vqtbl1q_s16(a[2], idx);
    b[3] = vqtbl1q_s16(a[3], idx);
    b[4] = vqtbl1q_s16(a[4], idx);
    b[5] = vqtbl1q_s16(a[5], idx);
    b[6] = vqtbl1q_s16(a[6], idx);
    b[7] = vqtbl1q_s16(a[7], idx);

    a[0] = x265_caddq_s16<90>(b[0], b[0]);
    a[1] = x265_caddq_s16<90>(b[1], b[1]);
    a[2] = x265_caddq_s16<90>(b[2], b[2]);
    a[3] = x265_caddq_s16<90>(b[3], b[3]);
    a[4] = x265_caddq_s16<90>(b[4], b[4]);
    a[5] = x265_caddq_s16<90>(b[5], b[5]);
    a[6] = x265_caddq_s16<90>(b[6], b[6]);
    a[7] = x265_caddq_s16<90>(b[7], b[7]);

    // Re-order input ready for CADD instruction.
    b[0] = vqtbl1q_s16(a[0], idx);
    b[1] = vqtbl1q_s16(a[1], idx);
    b[2] = vqtbl1q_s16(a[2], idx);
    b[3] = vqtbl1q_s16(a[3], idx);
    b[4] = vqtbl1q_s16(a[4], idx);
    b[5] = vqtbl1q_s16(a[5], idx);
    b[6] = vqtbl1q_s16(a[6], idx);
    b[7] = vqtbl1q_s16(a[7], idx);

    a[0] = x265_caddq_s16<90>(b[0], b[0]);
    a[1] = x265_caddq_s16<90>(b[1], b[1]);
    a[2] = x265_caddq_s16<90>(b[2], b[2]);
    a[3] = x265_caddq_s16<90>(b[3], b[3]);
    a[4] = x265_caddq_s16<90>(b[4], b[4]);
    a[5] = x265_caddq_s16<90>(b[5], b[5]);
    a[6] = x265_caddq_s16<90>(b[6], b[6]);
    a[7] = x265_caddq_s16<90>(b[7], b[7]);

    sumsubq_s16(&b[0], &b[1], a[0], a[1]);
    sumsubq_s16(&b[2], &b[3], a[2], a[3]);
    sumsubq_s16(&b[4], &b[5], a[4], a[5]);
    sumsubq_s16(&b[6], &b[7], a[6], a[7]);

    abssumsubq_s16(&a[0], &a[2], b[0], b[2]);
    abssumsubq_s16(&a[1], &a[3], b[1], b[3]);
    abssumsubq_s16(&a[4], &a[6], b[4], b[6]);
    abssumsubq_s16(&a[5], &a[7], b[5], b[7]);

    uint16x8_t max0 = vmaxq_u16(vreinterpretq_u16_s16(a[0]), vreinterpretq_u16_s16(a[4]));
    uint16x8_t max1 = vmaxq_u16(vreinterpretq_u16_s16(a[1]), vreinterpretq_u16_s16(a[5]));
    uint16x8_t max2 = vmaxq_u16(vreinterpretq_u16_s16(a[2]), vreinterpretq_u16_s16(a[6]));
    uint16x8_t max3 = vmaxq_u16(vreinterpretq_u16_s16(a[3]), vreinterpretq_u16_s16(a[7]));

    *sum = x265_udotq_u16(*sum, max0, vdupq_n_u16(1));
    *sum = x265_udotq_u16(*sum, max1, vdupq_n_u16(1));
    *sum = x265_udotq_u16(*sum, max2, vdupq_n_u16(1));
    *sum = x265_udotq_u16(*sum, max3, vdupq_n_u16(1));
}

#elif X265_DEPTH == 12
static inline void hadamard_8x8(int16x8_t diff[8], uint64x2_t *sum)
{
    int16x8_t a[8], b[8];
    int32x4_t c[16], d[16];

    a[0] = x265_caddq_s16<90>(diff[0], diff[0]);
    a[1] = x265_caddq_s16<90>(diff[1], diff[1]);
    a[2] = x265_caddq_s16<90>(diff[2], diff[2]);
    a[3] = x265_caddq_s16<90>(diff[3], diff[3]);
    a[4] = x265_caddq_s16<90>(diff[4], diff[4]);
    a[5] = x265_caddq_s16<90>(diff[5], diff[5]);
    a[6] = x265_caddq_s16<90>(diff[6], diff[6]);
    a[7] = x265_caddq_s16<90>(diff[7], diff[7]);

    const uint8x16_t idx = vld1q_u8(kHADPermuteTbl);
    // Re-order input ready for CADD instruction.
    b[0] = vqtbl1q_s16(a[0], idx);
    b[1] = vqtbl1q_s16(a[1], idx);
    b[2] = vqtbl1q_s16(a[2], idx);
    b[3] = vqtbl1q_s16(a[3], idx);
    b[4] = vqtbl1q_s16(a[4], idx);
    b[5] = vqtbl1q_s16(a[5], idx);
    b[6] = vqtbl1q_s16(a[6], idx);
    b[7] = vqtbl1q_s16(a[7], idx);

    a[0] = x265_caddq_s16<90>(b[0], b[0]);
    a[1] = x265_caddq_s16<90>(b[1], b[1]);
    a[2] = x265_caddq_s16<90>(b[2], b[2]);
    a[3] = x265_caddq_s16<90>(b[3], b[3]);
    a[4] = x265_caddq_s16<90>(b[4], b[4]);
    a[5] = x265_caddq_s16<90>(b[5], b[5]);
    a[6] = x265_caddq_s16<90>(b[6], b[6]);
    a[7] = x265_caddq_s16<90>(b[7], b[7]);

    // Re-order input ready for CADD instruction.
    b[0] = vqtbl1q_s16(a[0], idx);
    b[1] = vqtbl1q_s16(a[1], idx);
    b[2] = vqtbl1q_s16(a[2], idx);
    b[3] = vqtbl1q_s16(a[3], idx);
    b[4] = vqtbl1q_s16(a[4], idx);
    b[5] = vqtbl1q_s16(a[5], idx);
    b[6] = vqtbl1q_s16(a[6], idx);
    b[7] = vqtbl1q_s16(a[7], idx);

    a[0] = x265_caddq_s16<90>(b[0], b[0]);
    a[1] = x265_caddq_s16<90>(b[1], b[1]);
    a[2] = x265_caddq_s16<90>(b[2], b[2]);
    a[3] = x265_caddq_s16<90>(b[3], b[3]);
    a[4] = x265_caddq_s16<90>(b[4], b[4]);
    a[5] = x265_caddq_s16<90>(b[5], b[5]);
    a[6] = x265_caddq_s16<90>(b[6], b[6]);
    a[7] = x265_caddq_s16<90>(b[7], b[7]);

    sumsublq_s16(&c[0], &c[1], &c[2], &c[3], a[0], a[1]);
    sumsublq_s16(&c[4], &c[5], &c[6], &c[7], a[2], a[3]);
    sumsublq_s16(&c[8], &c[9], &c[10], &c[11], a[4], a[5]);
    sumsublq_s16(&c[12], &c[13], &c[14], &c[15], a[6], a[7]);

    abssumsubq_s32(&d[0], &d[4], c[0], c[4]);
    abssumsubq_s32(&d[1], &d[5], c[1], c[5]);
    abssumsubq_s32(&d[2], &d[6], c[2], c[6]);
    abssumsubq_s32(&d[3], &d[7], c[3], c[7]);
    abssumsubq_s32(&d[8], &d[12], c[8], c[12]);
    abssumsubq_s32(&d[9], &d[13], c[9], c[13]);
    abssumsubq_s32(&d[10], &d[14], c[10], c[14]);
    abssumsubq_s32(&d[11], &d[15], c[11], c[15]);

    uint32x4_t sum0 = vmaxq_u32(vreinterpretq_u32_s32(d[0]), vreinterpretq_u32_s32(d[8]));
    uint32x4_t sum1 = vmaxq_u32(vreinterpretq_u32_s32(d[1]), vreinterpretq_u32_s32(d[9]));
    uint32x4_t sum2 =
        vmaxq_u32(vreinterpretq_u32_s32(d[2]), vreinterpretq_u32_s32(d[10]));
    uint32x4_t sum3 =
        vmaxq_u32(vreinterpretq_u32_s32(d[3]), vreinterpretq_u32_s32(d[11]));
    uint32x4_t sum4 =
        vmaxq_u32(vreinterpretq_u32_s32(d[4]), vreinterpretq_u32_s32(d[12]));
    uint32x4_t sum5 =
        vmaxq_u32(vreinterpretq_u32_s32(d[5]), vreinterpretq_u32_s32(d[13]));
    uint32x4_t sum6 =
        vmaxq_u32(vreinterpretq_u32_s32(d[6]), vreinterpretq_u32_s32(d[14]));
    uint32x4_t sum7 =
        vmaxq_u32(vreinterpretq_u32_s32(d[7]), vreinterpretq_u32_s32(d[15]));

    uint32x4_t sum01 = vaddq_u32(sum0, sum1);
    uint32x4_t sum23 = vaddq_u32(sum2, sum3);
    uint32x4_t sum45 = vaddq_u32(sum4, sum5);
    uint32x4_t sum67 = vaddq_u32(sum6, sum7);

    uint32x4_t sum0123 = vaddq_u32(sum01, sum23);
    uint32x4_t sum4567 = vaddq_u32(sum45, sum67);
    *sum = vpadalq_u32(*sum, sum0123);
    *sum = vpadalq_u32(*sum, sum4567);
}

#endif // X265_DEPTH == 8

#if HIGH_BIT_DEPTH
static inline void pixel_satd_4x4_sve2(const uint16_t *pix1, intptr_t stride_pix1,
                                       const uint16_t *pix2, intptr_t stride_pix2,
                                       uint64x2_t *sum)
{
    uint16x4_t s[4], r[4];
    int16x8_t diff[2];

    load_u16x4xn<4>(pix1, stride_pix1, s);
    load_u16x4xn<4>(pix2, stride_pix2, r);

    uint16x8_t s0 = vcombine_u16(s[0], s[2]);
    uint16x8_t s1 = vcombine_u16(s[1], s[3]);
    uint16x8_t r0 = vcombine_u16(r[0], r[2]);
    uint16x8_t r1 = vcombine_u16(r[1], r[3]);

    diff[0] = vreinterpretq_s16_u16(vsubq_u16(s0, r0));
    diff[1] = vreinterpretq_s16_u16(vsubq_u16(r1, s1));

    hadamard_4x4(diff, sum);
}

static inline void pixel_satd_4x8_sve2(const uint16_t *pix1, intptr_t stride_pix1,
                                       const uint16_t *pix2, intptr_t stride_pix2,
                                       uint64x2_t *sum)
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

    hadamard_4x4_dual(diff, sum);
}

static inline void pixel_satd_8x4_sve2(const uint16_t *pix1, intptr_t stride_pix1,
                                       const uint16_t *pix2, intptr_t stride_pix2,
                                       uint64x2_t *sum)
{
    int16x8_t diff[4];

    load_diff_u16x8x4(pix1, stride_pix1, pix2, stride_pix2, diff);

    hadamard_4x4_dual(diff, sum);
}

static inline void pixel_satd_8x8_sve2(const uint16_t *pix1, intptr_t stride_pix1,
                                       const uint16_t *pix2, intptr_t stride_pix2,
                                       uint64x2_t *sum)
{
    int16x8_t diff[8];

    load_diff_u16x8x4_dual(pix1, stride_pix1, pix2, stride_pix2, diff);

    hadamard_4x4_quad(diff, sum);
}

static inline void pixel_satd_8x16_sve2(const uint16_t *pix1, intptr_t stride_pix1,
                                        const uint16_t *pix2, intptr_t stride_pix2,
                                        uint64x2_t *sum)
{
    int16x8_t diff[16];

    load_diff_u16x8x4_dual(pix1 + 0 * stride_pix1, stride_pix1, pix2 + 0 * stride_pix2,
                           stride_pix2, diff + 0);
    load_diff_u16x8x4_dual(pix1 + 8 * stride_pix1, stride_pix1, pix2 + 8 * stride_pix2,
                           stride_pix2, diff + 8);

    hadamard_4x4_quad(diff + 0, sum);
    hadamard_4x4_quad(diff + 8, sum);
}

static inline void pixel_satd_16x4_sve2(const uint16_t *pix1, intptr_t stride_pix1,
                                        const uint16_t *pix2, intptr_t stride_pix2,
                                        uint64x2_t *sum)
{
    int16x8_t diff[8];

    load_diff_u16x8x4(pix1 + 0, stride_pix1, pix2 + 0, stride_pix2, diff + 0);
    load_diff_u16x8x4(pix1 + 8, stride_pix1, pix2 + 8, stride_pix2, diff + 4);

    hadamard_4x4_quad(diff, sum);
}

static inline void pixel_satd_16x8_sve2(const uint16_t *pix1, intptr_t stride_pix1,
                                        const uint16_t *pix2, intptr_t stride_pix2,
                                        uint64x2_t *sum)
{
    int16x8_t diff[16];

    load_diff_u16x8x4_dual(pix1 + 0, stride_pix1, pix2 + 0, stride_pix2, diff + 0);
    load_diff_u16x8x4_dual(pix1 + 8, stride_pix1, pix2 + 8, stride_pix2, diff + 8);

    hadamard_4x4_quad(diff + 0, sum);
    hadamard_4x4_quad(diff + 8, sum);
}

static inline void pixel_satd_16x16_sve2(const uint16_t *pix1, intptr_t stride_pix1,
                                         const uint16_t *pix2, intptr_t stride_pix2,
                                         uint64x2_t *sum)
{
    int16x8_t diff[8];

    for (int i = 0; i < 4; ++i)
    {
        load_diff_u16x8x4(pix1 + 0, stride_pix1, pix2 + 0, stride_pix2, diff + 0);
        load_diff_u16x8x4(pix1 + 8, stride_pix1, pix2 + 8, stride_pix2, diff + 4);

        hadamard_4x4_quad(diff, sum);

        pix1 += 4 * stride_pix1;
        pix2 += 4 * stride_pix2;
    }
}

static inline int pixel_sa8d_8x8_sve2(const uint16_t *pix1, intptr_t stride_pix1,
                                      const uint16_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[8];
    uint64x2_t sum = vdupq_n_u64(0);

    load_diff_u16x8x4_dual(pix1, stride_pix1, pix2, stride_pix2, diff);
    hadamard_8x8(diff, &sum);

    return (vaddvq_u64(sum) + 1) >> 1;
}

static inline int pixel_sa8d_16x16_sve2(const uint16_t *pix1, intptr_t stride_pix1,
                                        const uint16_t *pix2, intptr_t stride_pix2)
{
    uint64x2_t sum = vdupq_n_u64(0);
    int16x8_t diff[8];

    load_diff_u16x8x8(pix1 + 0 * stride_pix1 + 0, stride_pix1, pix2 + 0 * stride_pix2 + 0,
                      stride_pix2, diff);
    hadamard_8x8(diff, &sum);

    load_diff_u16x8x8(pix1 + 0 * stride_pix1 + 8, stride_pix1, pix2 + 0 * stride_pix2 + 8,
                      stride_pix2, diff);
    hadamard_8x8(diff, &sum);

    load_diff_u16x8x8(pix1 + 8 * stride_pix1 + 0, stride_pix1, pix2 + 8 * stride_pix2 + 0,
                      stride_pix2, diff);
    hadamard_8x8(diff, &sum);

    load_diff_u16x8x8(pix1 + 8 * stride_pix1 + 8, stride_pix1, pix2 + 8 * stride_pix2 + 8,
                      stride_pix2, diff);
    hadamard_8x8(diff, &sum);

    return (vaddvq_u64(sum) + 1) >> 1;
}

static inline int pixel_sa8d_16x32_sve2(const uint16_t *pix1, intptr_t stride_pix1,
                                        const uint16_t *pix2, intptr_t stride_pix2)
{
    uint64x2_t sum0 = vdupq_n_u64(0);
    uint64x2_t sum1 = vdupq_n_u64(0);
    int16x8_t diff[8];

    load_diff_u16x8x8(pix1 + 0 * stride_pix1 + 0, stride_pix1, pix2 + 0 * stride_pix2 + 0,
                      stride_pix2, diff);
    hadamard_8x8(diff, &sum0);
    load_diff_u16x8x8(pix1 + 8 * stride_pix1 + 0, stride_pix1, pix2 + 8 * stride_pix2 + 0,
                      stride_pix2, diff);
    hadamard_8x8(diff, &sum0);
    load_diff_u16x8x8(pix1 + 0 * stride_pix1 + 8, stride_pix1, pix2 + 0 * stride_pix2 + 8,
                      stride_pix2, diff);
    hadamard_8x8(diff, &sum0);
    load_diff_u16x8x8(pix1 + 8 * stride_pix1 + 8, stride_pix1, pix2 + 8 * stride_pix2 + 8,
                      stride_pix2, diff);
    hadamard_8x8(diff, &sum0);

    load_diff_u16x8x8(pix1 + 16 * stride_pix1 + 0, stride_pix1,
                      pix2 + 16 * stride_pix2 + 0, stride_pix2, diff);
    hadamard_8x8(diff, &sum1);
    load_diff_u16x8x8(pix1 + 24 * stride_pix1 + 0, stride_pix1,
                      pix2 + 24 * stride_pix2 + 0, stride_pix2, diff);
    hadamard_8x8(diff, &sum1);
    load_diff_u16x8x8(pix1 + 16 * stride_pix1 + 8, stride_pix1,
                      pix2 + 16 * stride_pix2 + 8, stride_pix2, diff);
    hadamard_8x8(diff, &sum1);
    load_diff_u16x8x8(pix1 + 24 * stride_pix1 + 8, stride_pix1,
                      pix2 + 24 * stride_pix2 + 8, stride_pix2, diff);
    hadamard_8x8(diff, &sum1);

    uint64x2_t sum = vpaddq_u64(sum0, sum1);
    uint64x2_t sa8d = vrshrq_n_u64(sum, 1);
    return vaddvq_u64(sa8d);
}

#else

static inline void pixel_satd_4x4_sve2(const uint8_t *pix1, intptr_t stride_pix1,
                                       const uint8_t *pix2, intptr_t stride_pix2,
                                       uint64x2_t *sum)
{
    int16x8_t diff[2];

    uint8x8_t s0 = load_u8x4x2(pix1 + 0 * stride_pix1, 2 * stride_pix1);
    uint8x8_t s1 = load_u8x4x2(pix1 + 1 * stride_pix1, 2 * stride_pix1);

    uint8x8_t r0 = load_u8x4x2(pix2 + 0 * stride_pix2, 2 * stride_pix2);
    uint8x8_t r1 = load_u8x4x2(pix2 + 1 * stride_pix2, 2 * stride_pix2);

    diff[0] = vreinterpretq_s16_u16(vsubl_u8(s0, r0));
    diff[1] = vreinterpretq_s16_u16(vsubl_u8(r1, s1));

    hadamard_4x4(diff, sum);
}

static inline void pixel_satd_4x8_sve2(const uint8_t *pix1, intptr_t stride_pix1,
                                       const uint8_t *pix2, intptr_t stride_pix2,
                                       uint64x2_t *sum)
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

    hadamard_4x4_dual(diff, sum);
}

static inline void pixel_satd_8x4_sve2(const uint8_t *pix1, intptr_t stride_pix1,
                                       const uint8_t *pix2, intptr_t stride_pix2,
                                       uint64x2_t *sum)
{
    int16x8_t diff[4];

    load_diff_u8x8x4(pix1, stride_pix1, pix2, stride_pix2, diff);

    hadamard_4x4_dual(diff, sum);
}

static inline void pixel_satd_8x8_sve2(const uint8_t *pix1, intptr_t stride_pix1,
                                       const uint8_t *pix2, intptr_t stride_pix2,
                                       uint64x2_t *sum)
{
    int16x8_t diff[8];

    load_diff_u8x8x8(pix1, stride_pix1, pix2, stride_pix2, diff);

    hadamard_4x4_quad(diff, sum);
}

static inline void pixel_satd_8x16_sve2(const uint8_t *pix1, intptr_t stride_pix1,
                                        const uint8_t *pix2, intptr_t stride_pix2,
                                        uint64x2_t *sum)
{
    int16x8_t diff[16];

    load_diff_u8x8x8(pix1 + 0 * stride_pix1, stride_pix1, pix2 + 0 * stride_pix2,
                     stride_pix2, diff + 0);
    load_diff_u8x8x8(pix1 + 8 * stride_pix1, stride_pix1, pix2 + 8 * stride_pix2,
                     stride_pix2, diff + 8);

    hadamard_4x4_quad(diff + 0, sum);
    hadamard_4x4_quad(diff + 8, sum);
}

static inline void pixel_satd_16x4_sve2(const uint8_t *pix1, intptr_t stride_pix1,
                                        const uint8_t *pix2, intptr_t stride_pix2,
                                        uint64x2_t *sum)
{
    int16x8_t diff[8];

    load_diff_u8x8x4(pix1 + 0, stride_pix1, pix2 + 0, stride_pix2, diff + 0);
    load_diff_u8x8x4(pix1 + 8, stride_pix1, pix2 + 8, stride_pix2, diff + 4);

    hadamard_4x4_quad(diff, sum);
}

static inline void pixel_satd_16x8_sve2(const uint8_t *pix1, intptr_t stride_pix1,
                                        const uint8_t *pix2, intptr_t stride_pix2,
                                        uint64x2_t *sum)
{
    int16x8_t diff[16];

    load_diff_u8x8x8(pix1 + 0, stride_pix1, pix2 + 0, stride_pix2, diff + 0);
    load_diff_u8x8x8(pix1 + 8, stride_pix1, pix2 + 8, stride_pix2, diff + 8);

    hadamard_4x4_quad(diff + 0, sum);
    hadamard_4x4_quad(diff + 8, sum);
}

static inline void pixel_satd_16x16_sve2(const uint8_t *pix1, intptr_t stride_pix1,
                                         const uint8_t *pix2, intptr_t stride_pix2,
                                         uint64x2_t *sum)
{
    int16x8_t diff[8];

    load_diff_u8x16x4(pix1 + 0 * stride_pix1, stride_pix1, pix2 + 0 * stride_pix2,
                      stride_pix2, diff);
    hadamard_4x4_quad(diff, sum);

    load_diff_u8x16x4(pix1 + 4 * stride_pix1, stride_pix1, pix2 + 4 * stride_pix2,
                      stride_pix2, diff);
    hadamard_4x4_quad(diff, sum);

    load_diff_u8x16x4(pix1 + 8 * stride_pix1, stride_pix1, pix2 + 8 * stride_pix2,
                      stride_pix2, diff);
    hadamard_4x4_quad(diff, sum);

    load_diff_u8x16x4(pix1 + 12 * stride_pix1, stride_pix1, pix2 + 12 * stride_pix2,
                      stride_pix2, diff);
    hadamard_4x4_quad(diff, sum);
}

static inline int pixel_sa8d_8x8_sve2(const uint8_t *pix1, intptr_t stride_pix1,
                                      const uint8_t *pix2, intptr_t stride_pix2)
{
    int16x8_t diff[8];
    uint64x2_t sum = vdupq_n_u64(0);

    load_diff_u8x8x8(pix1, stride_pix1, pix2, stride_pix2, diff);
    hadamard_8x8(diff, &sum);

    return (vaddvq_u64(sum) + 1) >> 1;
}

static inline int pixel_sa8d_16x16_sve2(const uint8_t *pix1, intptr_t stride_pix1,
                                        const uint8_t *pix2, intptr_t stride_pix2)
{
    uint64x2_t sum = vdupq_n_u64(0);
    int16x8_t diff[8];

    load_diff_u8x8x8(pix1 + 0 * stride_pix1 + 0, stride_pix1, pix2 + 0 * stride_pix2 + 0,
                     stride_pix2, diff);
    hadamard_8x8(diff, &sum);

    load_diff_u8x8x8(pix1 + 0 * stride_pix1 + 8, stride_pix1, pix2 + 0 * stride_pix2 + 8,
                     stride_pix2, diff);
    hadamard_8x8(diff, &sum);

    load_diff_u8x8x8(pix1 + 8 * stride_pix1 + 0, stride_pix1, pix2 + 8 * stride_pix2 + 0,
                     stride_pix2, diff);
    hadamard_8x8(diff, &sum);

    load_diff_u8x8x8(pix1 + 8 * stride_pix1 + 8, stride_pix1, pix2 + 8 * stride_pix2 + 8,
                     stride_pix2, diff);
    hadamard_8x8(diff, &sum);

    return (vaddvq_u64(sum) + 1) >> 1;
}

static inline int pixel_sa8d_16x32_sve2(const uint8_t *pix1, intptr_t stride_pix1,
                                        const uint8_t *pix2, intptr_t stride_pix2)
{
    uint64x2_t sum0 = vdupq_n_u64(0);
    uint64x2_t sum1 = vdupq_n_u64(0);
    int16x8_t diff[8];

    load_diff_u8x8x8(pix1 + 0 * stride_pix1 + 0, stride_pix1, pix2 + 0 * stride_pix2 + 0,
                     stride_pix2, diff);
    hadamard_8x8(diff, &sum0);
    load_diff_u8x8x8(pix1 + 8 * stride_pix1 + 0, stride_pix1, pix2 + 8 * stride_pix2 + 0,
                     stride_pix2, diff);
    hadamard_8x8(diff, &sum0);
    load_diff_u8x8x8(pix1 + 0 * stride_pix1 + 8, stride_pix1, pix2 + 0 * stride_pix2 + 8,
                     stride_pix2, diff);
    hadamard_8x8(diff, &sum0);
    load_diff_u8x8x8(pix1 + 8 * stride_pix1 + 8, stride_pix1, pix2 + 8 * stride_pix2 + 8,
                     stride_pix2, diff);
    hadamard_8x8(diff, &sum0);

    load_diff_u8x8x8(pix1 + 16 * stride_pix1 + 0, stride_pix1,
                     pix2 + 16 * stride_pix2 + 0, stride_pix2, diff);
    hadamard_8x8(diff, &sum1);
    load_diff_u8x8x8(pix1 + 24 * stride_pix1 + 0, stride_pix1,
                     pix2 + 24 * stride_pix2 + 0, stride_pix2, diff);
    hadamard_8x8(diff, &sum1);
    load_diff_u8x8x8(pix1 + 16 * stride_pix1 + 8, stride_pix1,
                     pix2 + 16 * stride_pix2 + 8, stride_pix2, diff);
    hadamard_8x8(diff, &sum1);
    load_diff_u8x8x8(pix1 + 24 * stride_pix1 + 8, stride_pix1,
                     pix2 + 24 * stride_pix2 + 8, stride_pix2, diff);
    hadamard_8x8(diff, &sum1);

    uint64x2_t sum = vpaddq_u64(sum0, sum1);
    uint64x2_t sa8d = vrshrq_n_u64(sum, 1);
    return vaddvq_u64(sa8d);
}

#endif

namespace X265_NS
{

template<int w, int h>
int satd4_sve2(const pixel *pix1, intptr_t stride_pix1, const pixel *pix2,
               intptr_t stride_pix2)
{
    uint64x2_t sum = vdupq_n_u64(0);

    if (w == 4 && h == 4)
    {
        pixel_satd_4x4_sve2(pix1, stride_pix1, pix2, stride_pix2, &sum);
    }
    else
    {
        for (int row = 0; row < h; row += 8)
        {
            for (int col = 0; col < w; col += 4)
            {
                pixel_satd_4x8_sve2(pix1 + row * stride_pix1 + col, stride_pix1,
                                    pix2 + row * stride_pix2 + col, stride_pix2, &sum);
            }
        }
    }

    return (int)vaddvq_u64(sum);
}

template<int w, int h>
int satd8_sve2(const pixel *pix1, intptr_t stride_pix1, const pixel *pix2,
               intptr_t stride_pix2)
{
    uint64x2_t sum = vdupq_n_u64(0);

    if (w % 16 == 0 && h % 16 == 0)
    {
        for (int row = 0; row < h; row += 16)
        {
            for (int col = 0; col < w; col += 16)
            {
                pixel_satd_16x16_sve2(pix1 + row * stride_pix1 + col, stride_pix1,
                                      pix2 + row * stride_pix2 + col, stride_pix2, &sum);
            }
        }
    }
    else if (w % 8 == 0 && h % 16 == 0)
    {
        for (int row = 0; row < h; row += 16)
        {
            for (int col = 0; col < w; col += 8)
            {
                pixel_satd_8x16_sve2(pix1 + row * stride_pix1 + col, stride_pix1,
                                     pix2 + row * stride_pix2 + col, stride_pix2, &sum);
            }
        }
    }
    else if (w % 16 == 0 && h % 8 == 0)
    {
        for (int row = 0; row < h; row += 8)
        {
            for (int col = 0; col < w; col += 16)
            {
                pixel_satd_16x8_sve2(pix1 + row * stride_pix1 + col, stride_pix1,
                                     pix2 + row * stride_pix2 + col, stride_pix2, &sum);
            }
        }
    }
    else if (w % 16 == 0 && h % 4 == 0)
    {
        for (int row = 0; row < h; row += 4)
        {
            for (int col = 0; col < w; col += 16)
            {
                pixel_satd_16x4_sve2(pix1 + row * stride_pix1 + col, stride_pix1,
                                     pix2 + row * stride_pix2 + col, stride_pix2, &sum);
            }
        }
    }
    else if (w % 8 == 0 && h % 8 == 0)
    {
        for (int row = 0; row < h; row += 8)
        {
            for (int col = 0; col < w; col += 8)
            {
                pixel_satd_8x8_sve2(pix1 + row * stride_pix1 + col, stride_pix1,
                                    pix2 + row * stride_pix2 + col, stride_pix2, &sum);
            }
        }
    }
    else // w multiple of 8, h multiple of 4
    {
        for (int row = 0; row < h; row += 4)
        {
            for (int col = 0; col < w; col += 8)
            {
                pixel_satd_8x4_sve2(pix1 + row * stride_pix1 + col, stride_pix1,
                                    pix2 + row * stride_pix2 + col, stride_pix2, &sum);
            }
        }
    }

    return (int)vaddvq_u64(sum);
}

template<int w, int h>
// Calculate sa8d in blocks of 8x8
int sa8d8_sve2(const pixel *pix1, intptr_t i_pix1, const pixel *pix2, intptr_t i_pix2)
{
    int cost = 0;

    for (int y = 0; y < h; y += 8)
    {
        for (int x = 0; x < w; x += 8)
        {
            cost += pixel_sa8d_8x8_sve2(pix1 + i_pix1 * y + x, i_pix1,
                                        pix2 + i_pix2 * y + x, i_pix2);
        }
    }
    return cost;
}

template<int w, int h>
// Calculate sa8d in blocks of 16x16
int sa8d16_sve2(const pixel *pix1, intptr_t i_pix1, const pixel *pix2, intptr_t i_pix2)
{
    int cost = 0;

    for (int y = 0; y < h; y += 16)
    {
        for (int x = 0; x < w; x += 16)
        {
            cost += pixel_sa8d_16x16_sve2(pix1 + i_pix1 * y + x, i_pix1,
                                          pix2 + i_pix2 * y + x, i_pix2);
        }
    }
    return cost;
}

template<int w, int h>
// Calculate sa8d in blocks of 16x32
int sa8d16x32_sve2(const pixel *pix1, intptr_t i_pix1, const pixel *pix2, intptr_t i_pix2)
{
    int cost = 0;

    for (int y = 0; y < h; y += 32)
    {
        for (int x = 0; x < w; x += 16)
        {
            cost += pixel_sa8d_16x32_sve2(pix1 + i_pix1 * y + x, i_pix1,
                                          pix2 + i_pix2 * y + x, i_pix2);
        }
    }
    return cost;
}

void setupPixelPrimitives_sve2(EncoderPrimitives &p)
{
    p.pu[LUMA_4x4].satd = satd4_sve2<4, 4>;
    p.pu[LUMA_4x8].satd = satd4_sve2<4, 8>;
    p.pu[LUMA_4x16].satd = satd4_sve2<4, 16>;
    p.pu[LUMA_8x4].satd = satd8_sve2<8, 4>;
    p.pu[LUMA_8x8].satd = satd8_sve2<8, 8>;
    p.pu[LUMA_8x16].satd = satd8_sve2<8, 16>;
    p.pu[LUMA_8x32].satd = satd8_sve2<8, 32>;
    p.pu[LUMA_12x16].satd = satd4_sve2<12, 16>;
    p.pu[LUMA_16x4].satd = satd8_sve2<16, 4>;
    p.pu[LUMA_16x8].satd = satd8_sve2<16, 8>;
    p.pu[LUMA_16x12].satd = satd8_sve2<16, 12>;
    p.pu[LUMA_16x16].satd = satd8_sve2<16, 16>;
    p.pu[LUMA_16x32].satd = satd8_sve2<16, 32>;
    p.pu[LUMA_16x64].satd = satd8_sve2<16, 64>;
    p.pu[LUMA_24x32].satd = satd8_sve2<24, 32>;
    p.pu[LUMA_32x8].satd = satd8_sve2<32, 8>;
    p.pu[LUMA_32x16].satd = satd8_sve2<32, 16>;
    p.pu[LUMA_32x24].satd = satd8_sve2<32, 24>;
    p.pu[LUMA_32x32].satd = satd8_sve2<32, 32>;
    p.pu[LUMA_32x64].satd = satd8_sve2<32, 64>;
    p.pu[LUMA_48x64].satd = satd8_sve2<48, 64>;
    p.pu[LUMA_64x16].satd = satd8_sve2<64, 16>;
    p.pu[LUMA_64x32].satd = satd8_sve2<64, 32>;
    p.pu[LUMA_64x48].satd = satd8_sve2<64, 48>;
    p.pu[LUMA_64x64].satd = satd8_sve2<64, 64>;

    p.cu[BLOCK_4x4].sa8d = satd4_sve2<4, 4>;
    p.cu[BLOCK_8x8].sa8d = sa8d8_sve2<8, 8>;
    p.cu[BLOCK_16x16].sa8d = sa8d16_sve2<16, 16>;
    p.cu[BLOCK_32x32].sa8d = sa8d16x32_sve2<32, 32>;
    p.cu[BLOCK_64x64].sa8d = sa8d16x32_sve2<64, 64>;

    p.chroma[X265_CSP_I420].pu[CHROMA_420_2x2].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_2x4].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_2x8].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x2].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x4].satd = satd4_sve2<4, 4>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x8].satd = satd4_sve2<4, 8>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x16].satd = satd4_sve2<4, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_6x8].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x2].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x4].satd = satd8_sve2<8, 4>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x6].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x8].satd = satd8_sve2<8, 8>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x16].satd = satd8_sve2<8, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x32].satd = satd8_sve2<8, 32>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_12x16].satd = satd4_sve2<12, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x4].satd = satd8_sve2<16, 4>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x8].satd = satd8_sve2<16, 8>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x12].satd = satd8_sve2<16, 12>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x16].satd = satd8_sve2<16, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x32].satd = satd8_sve2<16, 32>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_24x32].satd = satd8_sve2<24, 32>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x8].satd = satd8_sve2<32, 8>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x16].satd = satd8_sve2<32, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x24].satd = satd8_sve2<32, 24>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x32].satd = satd8_sve2<32, 32>;

    p.chroma[X265_CSP_I420].cu[BLOCK_8x8].sa8d =
        p.chroma[X265_CSP_I420].pu[CHROMA_420_4x4].satd;
    p.chroma[X265_CSP_I420].cu[BLOCK_16x16].sa8d = sa8d8_sve2<8, 8>;
    p.chroma[X265_CSP_I420].cu[BLOCK_32x32].sa8d = sa8d16_sve2<16, 16>;
    p.chroma[X265_CSP_I420].cu[BLOCK_64x64].sa8d = sa8d16x32_sve2<32, 32>;

    p.chroma[X265_CSP_I422].pu[CHROMA_422_2x4].satd = NULL;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_2x8].satd = NULL;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_2x16].satd = NULL;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x4].satd = satd4_sve2<4, 4>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x8].satd = satd4_sve2<4, 8>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x16].satd = satd4_sve2<4, 16>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x32].satd = satd4_sve2<4, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_6x16].satd = NULL;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x4].satd = satd8_sve2<8, 4>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x8].satd = satd8_sve2<8, 8>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x12].satd = satd8_sve2<8, 12>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x16].satd = satd8_sve2<8, 16>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x32].satd = satd8_sve2<8, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x64].satd = satd8_sve2<8, 64>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_12x32].satd = satd4_sve2<12, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x8].satd = satd8_sve2<16, 8>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x16].satd = satd8_sve2<16, 16>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x24].satd = satd8_sve2<16, 24>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x32].satd = satd8_sve2<16, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x64].satd = satd8_sve2<16, 64>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_24x64].satd = satd8_sve2<24, 64>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x16].satd = satd8_sve2<32, 16>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x32].satd = satd8_sve2<32, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x48].satd = satd8_sve2<32, 48>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x64].satd = satd8_sve2<32, 64>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_12x32].satd = satd4_sve2<12, 32>;

    p.chroma[X265_CSP_I422].cu[BLOCK_8x8].sa8d =
        p.chroma[X265_CSP_I422].pu[CHROMA_422_4x8].satd;
    p.chroma[X265_CSP_I422].cu[BLOCK_16x16].sa8d = sa8d8_sve2<8, 16>;
    p.chroma[X265_CSP_I422].cu[BLOCK_32x32].sa8d = sa8d16x32_sve2<16, 32>;
    p.chroma[X265_CSP_I422].cu[BLOCK_64x64].sa8d = sa8d16x32_sve2<32, 64>;

    p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].sa8d = sa8d8_sve2<8, 16>;
    p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].sa8d = sa8d16x32_sve2<16, 32>;
    p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].sa8d = sa8d16x32_sve2<32, 64>;
}
} // namespace X265_NS
#endif // defined(HAVE_SVE2) && HAVE_SVE_BRIDGE
