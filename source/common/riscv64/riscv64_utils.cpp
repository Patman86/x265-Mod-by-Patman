
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
                    __riscv_vse16_v_u16m1(dst + 8 * j + (8 * i + k) * dstride, __riscv_vle16_v_u16m1(tmp + 8 * k, vl), vl);
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