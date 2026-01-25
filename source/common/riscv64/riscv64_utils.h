/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Jia Yuan <yuan.jia@sanechips.com.cn>
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

#ifndef __RISCV64_UTILS_H__
#define __RISCV64_UTILS_H__

#include <stdint.h>

extern "C" {
    void transpose4x4_rvv_u8(uint8_t* dst, const uint8_t* src, intptr_t dstride, intptr_t sstride);
    void transpose8x8_rvv_u8(uint8_t* dst, const uint8_t* src, intptr_t dstride, intptr_t sstride);

    void transpose4x4_rvv_u16(uint16_t* dst, const uint16_t* src, intptr_t dstride, intptr_t sstride);
    void transpose8x8_rvv_u16(uint16_t* dst, const uint16_t* src, intptr_t dstride, intptr_t sstride);

    void transpose16x16_rvv_u8(uint8_t* dst, const uint8_t* src, intptr_t dstride, intptr_t sstride);
}

namespace X265_NS
{

// overload
inline void transpose4x4_rvv(uint8_t* dst, const uint8_t* src, intptr_t dstride, intptr_t sstride) {
    transpose4x4_rvv_u8(dst, src, dstride, sstride);
}

inline void transpose4x4_rvv(uint16_t* dst, const uint16_t* src, intptr_t dstride, intptr_t sstride) {
    transpose4x4_rvv_u16(dst, src, dstride, sstride);
}

inline void transpose8x8_rvv(uint8_t* dst, const uint8_t* src, intptr_t dstride, intptr_t sstride) {
    transpose8x8_rvv_u8(dst, src, dstride, sstride);
}

inline void transpose8x8_rvv(uint16_t* dst, const uint16_t* src, intptr_t dstride, intptr_t sstride) {
    transpose8x8_rvv_u16(dst, src, dstride, sstride);
}

inline void transpose16x16_rvv(uint8_t* dst, const uint8_t* src, intptr_t dstride, intptr_t sstride) {
    transpose16x16_rvv_u8(dst, src, dstride, sstride);
}

void transpose32x32_rvv(uint8_t *dst, const uint8_t *src, intptr_t dstride, intptr_t sstride);

void transpose16x16_rvv(uint16_t *dst, const uint16_t *src, intptr_t dstride, intptr_t sstride);
void transpose32x32_rvv(uint16_t *dst, const uint16_t *src, intptr_t dstride, intptr_t sstride);
}

#endif