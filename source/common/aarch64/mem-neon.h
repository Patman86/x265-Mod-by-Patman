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

#ifndef X265_COMMON_AARCH64_MEM_NEON_H
#define X265_COMMON_AARCH64_MEM_NEON_H

#include <arm_neon.h>
#include <cassert>
#include <stdint.h>

// Load 4 bytes into the low half of a uint8x8_t, zero the upper half.
static uint8x8_t inline load_u8x4x1(const uint8_t *s)
{
    uint8x8_t ret = vdup_n_u8(0);

    ret = vreinterpret_u8_u32(vld1_lane_u32((const uint32_t*)s,
                                            vreinterpret_u32_u8(ret), 0));
    return ret;
}

static uint8x8_t inline load_u8x4x2(const uint8_t *s, intptr_t stride)
{
    uint8x8_t ret = vdup_n_u8(0);

    ret = vreinterpret_u8_u32(vld1_lane_u32((const uint32_t*)s,
                                            vreinterpret_u32_u8(ret), 0));
    s += stride;
    ret = vreinterpret_u8_u32(vld1_lane_u32((const uint32_t*)s,
                                            vreinterpret_u32_u8(ret), 1));

    return ret;
}

// Store 4 bytes from the low half of a uint8x8_t.
static void inline store_u8x4x1(uint8_t *d, const uint8x8_t s)
{
    vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(s), 0);
}

// Store N blocks of 32-bits from (N / 2) D-Registers.
template<int N>
static void inline store_u8x4_strided_xN(uint8_t *d, intptr_t stride,
                                         const uint8x8_t *s)
{
    assert(N % 2 == 0);
    for (int i = 0; i < N / 2; ++i)
    {
        vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(s[i]), 0);
        d += stride;
        vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(s[i]), 1);
        d += stride;
    }
}

template<int N>
static void inline load_u8x8xn(const uint8_t *src, const intptr_t stride,
                               uint8x8_t *dst)
{
    for (int i = 0; i < N; ++i)
    {
        dst[i] = vld1_u8(src);
        src += stride;
    }
}

template<int N>
static void inline load_u8x16xn(const uint8_t *src, const intptr_t stride,
                                uint8x16_t *dst)
{
    for (int i = 0; i < N; ++i)
    {
        dst[i] = vld1q_u8(src);
        src += stride;
    }
}

template<int N>
static void inline store_u8x2xn(uint8_t *dst, intptr_t dst_stride,
                                const uint8x8_t *src)
{
    for (int i = 0; i < N; ++i)
    {
        vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(src[i]), 0);
        dst += dst_stride;
    }
}

template<int N>
static void inline store_u8x4xn(uint8_t *dst, intptr_t dst_stride,
                                const uint8x8_t *src)
{
    for (int i = 0; i < N; ++i)
    {
        vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(src[i]), 0);
        dst += dst_stride;
    }
}

template<int N>
static void inline store_u8x6xn(uint8_t *dst, intptr_t dst_stride,
                                const uint8x8_t *src)
{
    for (int i = 0; i < N; ++i)
    {
        vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(src[i]), 0);
        vst1_lane_u16((uint16_t *)(dst + 4), vreinterpret_u16_u8(src[i]), 2);
        dst += dst_stride;
    }
}

template<int N>
static void inline store_u8x8xn(uint8_t *dst, intptr_t dst_stride,
                                const uint8x8_t *src)
{
    for (int i = 0; i < N; ++i)
    {
        vst1_u8(dst, src[i]);
        dst += dst_stride;
    }
}

template<int N, int M>
static void inline store_u8xnxm(uint8_t *dst, intptr_t dst_stride,
                                const uint8x8_t *src)
{
    switch (N)
    {
    case 2: return store_u8x2xn<M>(dst, dst_stride, src);
    case 4: return store_u8x4xn<M>(dst, dst_stride, src);
    case 6: return store_u8x6xn<M>(dst, dst_stride, src);
    case 8: return store_u8x8xn<M>(dst, dst_stride, src);
    }
}

template<int N>
static void inline store_u8x16xn(uint8_t *dst, intptr_t dst_stride,
                                 const uint8x16_t *src)
{
    for (int i = 0; i < N; ++i)
    {
        vst1q_u8(dst, src[i]);
        dst += dst_stride;
    }
}

template<int N>
static void inline load_s16x4xn(const int16_t *src, const intptr_t stride,
                                int16x4_t *dst)
{
    for (int i = 0; i < N; ++i)
    {
        dst[i] = vld1_s16(src);
        src += stride;
    }
}

template<int N>
static void inline load_s16x8xn(const int16_t *src, const intptr_t stride,
                                int16x8_t *dst)
{
    for (int i = 0; i < N; ++i)
    {
        dst[i] = vld1q_s16(src);
        src += stride;
    }
}

template<int N>
static void inline store_s16x2xn(int16_t *dst, intptr_t dst_stride,
                                 const int16x4_t *src)
{
    for (int i = 0; i < N; ++i)
    {
        vst1_lane_s32((int32_t*)dst, vreinterpret_s32_s16(src[i]), 0);
        dst += dst_stride;
    }
}

template<int N>
static void inline store_s16x2xn(int16_t *dst, intptr_t dst_stride,
                                 const int16x8_t *src)
{
    for (int i = 0; i < N; ++i)
    {
        vst1q_lane_s32((int32_t *)dst, vreinterpretq_s32_s16(src[i]), 0);
        dst += dst_stride;
    }
}

template<int N>
static void inline store_s16x4xn(int16_t *dst, intptr_t dst_stride,
                                 const int16x4_t *src)
{
    for (int i = 0; i < N; ++i)
    {
        vst1_s16(dst, src[i]);
        dst += dst_stride;
    }
}

template<int N>
static void inline store_s16x4xn(int16_t *dst, intptr_t dst_stride,
                                 const int16x8_t *src)
{
    for (int i = 0; i < N; ++i)
    {
        vst1_s16(dst, vget_low_s16(src[i]));
        dst += dst_stride;
    }
}

template<int N>
static void inline store_s16x6xn(int16_t *dst, intptr_t dst_stride,
                                 const int16x8_t *src)
{
    for (int i = 0; i < N; ++i)
    {
        vst1_s16(dst, vget_low_s16(src[i]));
        vst1q_lane_s32((int32_t*)(dst + 4), vreinterpretq_s32_s16(src[i]), 2);
        dst += dst_stride;
    }
}

template<int N>
static void inline store_s16x8xn(int16_t *dst, intptr_t dst_stride,
                                 const int16x8_t *src)
{
    for (int i = 0; i < N; ++i)
    {
        vst1q_s16(dst, src[i]);
        dst += dst_stride;
    }
}

template<int N, int M>
static void inline store_s16xnxm(const int16x8_t *src, int16_t *dst,
                                 intptr_t dst_stride)
{
    switch (N)
    {
    case 2: return store_s16x2xn<M>(dst, dst_stride, src);
    case 4: return store_s16x4xn<M>(dst, dst_stride, src);
    case 6: return store_s16x6xn<M>(dst, dst_stride, src);
    case 8: return store_s16x8xn<M>(dst, dst_stride, src);
    }
}

#endif // X265_COMMON_AARCH64_MEM_NEON_H
