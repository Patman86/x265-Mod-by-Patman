
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

#include "common.h"
#include "x265.h"
#include "riscv64_utils.h"
#include <riscv_vector.h>

namespace X265_NS
{
#if !HIGH_BIT_DEPTH
void transpose8x8_rvv(uint8_t *dst, const uint8_t *src, intptr_t dstride, intptr_t sstride)
{
    size_t vl = 8;

    vuint8m1_t a0 = __riscv_vle8_v_u8m1(src + 0 * sstride, vl);
    vuint8m1_t a1 = __riscv_vle8_v_u8m1(src + 1 * sstride, vl);
    vuint8m1_t a2 = __riscv_vle8_v_u8m1(src + 2 * sstride, vl);
    vuint8m1_t a3 = __riscv_vle8_v_u8m1(src + 3 * sstride, vl);
    vuint8m1_t a4 = __riscv_vle8_v_u8m1(src + 4 * sstride, vl);
    vuint8m1_t a5 = __riscv_vle8_v_u8m1(src + 5 * sstride, vl);
    vuint8m1_t a6 = __riscv_vle8_v_u8m1(src + 6 * sstride, vl);
    vuint8m1_t a7 = __riscv_vle8_v_u8m1(src + 7 * sstride, vl);

    vuint8m1x2_t v0 = __riscv_vcreate_v_u8m1x2(a0, a1);
    vuint8m1x2_t v2 = __riscv_vcreate_v_u8m1x2(a2, a3);
    vuint8m1x2_t v4 = __riscv_vcreate_v_u8m1x2(a4, a5);
    vuint8m1x2_t v6 = __riscv_vcreate_v_u8m1x2(a6, a7);

    __riscv_vssseg2e8_v_u8m1x2(dst, dstride, v0, vl);
    __riscv_vssseg2e8_v_u8m1x2(dst + 2, dstride, v2, vl);
    __riscv_vssseg2e8_v_u8m1x2(dst + 4, dstride, v4, vl);
    __riscv_vssseg2e8_v_u8m1x2(dst + 6, dstride, v6, vl);
}

void transpose16x16_rvv(uint8_t *dst, const uint8_t *src, intptr_t dstride, intptr_t sstride)
{
    size_t vl = 16;
    vuint8m1_t a0 = __riscv_vle8_v_u8m1(src + 0 * sstride, vl);
    vuint8m1_t a1 = __riscv_vle8_v_u8m1(src + 1 * sstride, vl);
    vuint8m1_t a2 = __riscv_vle8_v_u8m1(src + 2 * sstride, vl);
    vuint8m1_t a3 = __riscv_vle8_v_u8m1(src + 3 * sstride, vl);
    vuint8m1_t a4 = __riscv_vle8_v_u8m1(src + 4 * sstride, vl);
    vuint8m1_t a5 = __riscv_vle8_v_u8m1(src + 5 * sstride, vl);
    vuint8m1_t a6 = __riscv_vle8_v_u8m1(src + 6 * sstride, vl);
    vuint8m1_t a7 = __riscv_vle8_v_u8m1(src + 7 * sstride, vl);

    vuint8m1_t a8 = __riscv_vle8_v_u8m1(src + 8 * sstride, vl);
    vuint8m1_t a9 = __riscv_vle8_v_u8m1(src + 9 * sstride, vl);
    vuint8m1_t aA = __riscv_vle8_v_u8m1(src + 10 * sstride, vl);
    vuint8m1_t aB = __riscv_vle8_v_u8m1(src + 11 * sstride, vl);
    vuint8m1_t aC = __riscv_vle8_v_u8m1(src + 12 * sstride, vl);
    vuint8m1_t aD = __riscv_vle8_v_u8m1(src + 13 * sstride, vl);
    vuint8m1_t aE = __riscv_vle8_v_u8m1(src + 14 * sstride, vl);
    vuint8m1_t aF = __riscv_vle8_v_u8m1(src + 15 * sstride, vl);

    vuint8m1x2_t v0 = __riscv_vcreate_v_u8m1x2(a0, a1);
    vuint8m1x2_t v2 = __riscv_vcreate_v_u8m1x2(a2, a3);
    vuint8m1x2_t v4 = __riscv_vcreate_v_u8m1x2(a4, a5);
    vuint8m1x2_t v6 = __riscv_vcreate_v_u8m1x2(a6, a7);
    __riscv_vssseg2e8_v_u8m1x2(dst, dstride, v0, vl);
    __riscv_vssseg2e8_v_u8m1x2(dst + 2, dstride, v2, vl);
    __riscv_vssseg2e8_v_u8m1x2(dst + 4, dstride, v4, vl);
    __riscv_vssseg2e8_v_u8m1x2(dst + 6, dstride, v6, vl);

    vuint8m1x2_t v8 = __riscv_vcreate_v_u8m1x2(a8, a9);
    vuint8m1x2_t v10 = __riscv_vcreate_v_u8m1x2(aA, aB);
    vuint8m1x2_t v12 = __riscv_vcreate_v_u8m1x2(aC, aD);
    vuint8m1x2_t v14 = __riscv_vcreate_v_u8m1x2(aE, aF);
    __riscv_vssseg2e8_v_u8m1x2(dst + 8,  dstride, v8, vl);
    __riscv_vssseg2e8_v_u8m1x2(dst + 10, dstride, v10, vl);
    __riscv_vssseg2e8_v_u8m1x2(dst + 12, dstride, v12, vl);
    __riscv_vssseg2e8_v_u8m1x2(dst + 14, dstride, v14, vl);
}

void transpose32x32_rvv(uint8_t *dst, const uint8_t *src, intptr_t dstride, intptr_t sstride)
{
    transpose16x16_rvv(dst, src, dstride, sstride);
    transpose16x16_rvv(dst + 16 * dstride + 16, src + 16 * sstride + 16, dstride, sstride);
    if (dst == src)
    {
        size_t vl = 16;
        uint8_t tmp[16 * 16] __attribute__((aligned(64)));
        transpose16x16_rvv(tmp, src + 16, 16, sstride);
        transpose16x16_rvv(dst + 16, src + 16 * sstride, dstride, sstride);
        for (int i = 0; i < 16; i++)
        {
            __riscv_vse8_v_u8m1(dst + (16 + i) * dstride, __riscv_vle8_v_u8m1(tmp + 16 * i, vl), vl);
        }
    }
    else
    {
        transpose16x16_rvv(dst + 16 * dstride, src + 16, dstride, sstride);
        transpose16x16_rvv(dst + 16, src + 16 * sstride, dstride, sstride);
    }

}
#else
void transpose8x8_rvv(uint16_t *dst, const uint16_t *src, intptr_t dstride, intptr_t sstride)
{
    size_t vl = 8;

    vuint16m1_t a0 = __riscv_vle16_v_u16m1(src + 0 * sstride, vl);
    vuint16m1_t a1 = __riscv_vle16_v_u16m1(src + 1 * sstride, vl);
    vuint16m1_t a2 = __riscv_vle16_v_u16m1(src + 2 * sstride, vl);
    vuint16m1_t a3 = __riscv_vle16_v_u16m1(src + 3 * sstride, vl);

    vuint16m1_t a4 = __riscv_vle16_v_u16m1(src + 4 * sstride, vl);
    vuint16m1_t a5 = __riscv_vle16_v_u16m1(src + 5 * sstride, vl);
    vuint16m1_t a6 = __riscv_vle16_v_u16m1(src + 6 * sstride, vl);
    vuint16m1_t a7 = __riscv_vle16_v_u16m1(src + 7 * sstride, vl);

    vuint16m1x2_t v0 = __riscv_vcreate_v_u16m1x2 (a0, a1);
    vuint16m1x2_t v2 = __riscv_vcreate_v_u16m1x2 (a2, a3);
    vuint16m1x2_t v4 = __riscv_vcreate_v_u16m1x2 (a4, a5);
    vuint16m1x2_t v6 = __riscv_vcreate_v_u16m1x2 (a6, a7);

    // Notice: uint16_t* seg_addr = (uint16_t*) (bstride*i + (char*)base);
    __riscv_vssseg2e16_v_u16m1x2(dst, 2 * dstride, v0, vl);
    __riscv_vssseg2e16_v_u16m1x2(dst + 2, 2 * dstride, v2, vl);
    __riscv_vssseg2e16_v_u16m1x2(dst + 4, 2 * dstride, v4, vl);
    __riscv_vssseg2e16_v_u16m1x2(dst + 6, 2 * dstride, v6, vl);
}

void transpose16x16_rvv(uint16_t *dst, const uint16_t *src, intptr_t dstride, intptr_t sstride)
{
    transpose8x8_rvv(dst, src, dstride, sstride);
    transpose8x8_rvv(dst + 8 * dstride + 8, src + 8 * sstride + 8, dstride, sstride);

    if (dst == src)
    {
        uint16_t tmp[8 * 8];
        size_t vl = 8;
        transpose8x8_rvv(tmp, src + 8, 8, sstride);
        transpose8x8_rvv(dst + 8, src + 8 * sstride, dstride, sstride);
        for (int i = 0; i < 8; i++)
        {
            __riscv_vse16_v_u16m1(dst + (8 + i) * dstride, __riscv_vle16_v_u16m1(tmp + 8 * i, vl), vl);
        }
    }
    else
    {
        transpose8x8_rvv(dst + 8 * dstride, src + 8, dstride, sstride);
        transpose8x8_rvv(dst + 8, src + 8 * sstride, dstride, sstride);
    }

}

void transpose32x32_rvv(uint16_t *dst, const uint16_t *src, intptr_t dstride, intptr_t sstride)
{
    //assumption: there is no partial overlap
    for (int i = 0; i < 4; i++)
    {
        transpose8x8_rvv(dst + i * 8 * (1 + dstride), src + i * 8 * (1 + sstride), dstride, sstride);
        for (int j = i + 1; j < 4; j++)
        {
            if (dst == src)
            {
                uint16_t tmp[8 * 8] __attribute__((aligned(64)));
                size_t vl = 8;
                transpose8x8_rvv(tmp, src + 8 * i + 8 * j * sstride, 8, sstride);
                transpose8x8_rvv(dst + 8 * i + 8 * j * dstride, src + 8 * j + 8 * i * sstride, dstride, sstride);
                for (int k = 0; k < 8; k++)
                {
                    __riscv_vse16_v_u16m1(dst + 8 * j + (8 * i + k) * dstride,
                              __riscv_vle16_v_u16m1(tmp + 8 * k, vl), vl);
                }
            }
            else
            {
                transpose8x8_rvv(dst + 8 * (j + i * dstride), src + 8 * (i + j * sstride), dstride, sstride);
                transpose8x8_rvv(dst + 8 * (i + j * dstride), src + 8 * (j + i * sstride), dstride, sstride);
            }

        }
    }
}
#endif // !HIGH_BIT_DEPTH
}
