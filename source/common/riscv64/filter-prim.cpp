/*****************************************************************************
 * Copyright (C) 2021-2025 MulticoreWare, Inc
 *
 * Authors: Yujiao He <he.yujiao@sanechips.com.cn>
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

#include "filter-prim.h"
#include <stdint.h>

namespace X265_NS
{
#if !HIGH_BIT_DEPTH

template<int coeffIdx, int width, int height>
void interp8_horiz_pp_rvv(const pixel *src, intptr_t srcStride, pixel *dst,
                           intptr_t dstStride)
{
    const int N_TAPS = 8;
    src -= (N_TAPS / 2 - 1);
    size_t vl = 32;

    if (width > 8)
    {
        const vuint8m1_t shift = __riscv_vmv_v_x_u8m1(IF_FILTER_PREC, vl);
        const vint16m2_t zero = __riscv_vmv_v_x_i16m2(0, vl);
        vint16m2_t t0, t1, t2, t3, t4, t5, t6, t7;
        vint16m2_t r0, r1, r2, r3, r4, r5, r6, r7;
        vint16m2_t *s0[8] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7};
        vint16m2_t *s1[8] = {&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7};

        for (int row = 0; row < height; row +=2)
        {
            for (int col = 0; col < width; col += vl)
            {
                vl = __riscv_vsetvl_e16m2(width - col);

                load_u8x16xn<8>(src + col + 0 * srcStride, s0, 1, vl);
                load_u8x16xn<8>(src + col + 1 * srcStride, s1, 1, vl);

                vint16m2_t d0 = filter8_s16x16<coeffIdx>(s0, zero, vl);
                vint16m2_t d1 = filter8_s16x16<coeffIdx>(s1, zero, vl);

                vuint16m2_t d0_u16 = __riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax_vv_i16m2(d0, zero, vl));
                vuint16m2_t d1_u16 = __riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax_vv_i16m2(d1, zero, vl));

                vuint8m1_t d0_u8 = __riscv_vnclipu_wv_u8m1(d0_u16, shift, 0, vl);
                vuint8m1_t d1_u8 = __riscv_vnclipu_wv_u8m1(d1_u16, shift, 0, vl);

                __riscv_vse8_v_u8m1(dst + col + 0 * dstStride, d0_u8, vl);
                __riscv_vse8_v_u8m1(dst + col + 1 * dstStride, d1_u8, vl);
            }

            src += 2 * srcStride;
            dst += 2 * dstStride;
        }
    }
    else
    {
        vuint8mf2_t shift = __riscv_vmv_v_x_u8mf2(IF_FILTER_PREC, vl);
        vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t t0, t1, t2, t3, t4, t5, t6, t7;
        vint16m1_t r0, r1, r2, r3, r4, r5, r6, r7;
        vint16m1_t *s0[8] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7};
        vint16m1_t *s1[8] = {&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7};

        for (int row = 0; row < height; row += 2)
        {
            int col = 0;
            for (; col < width; col += vl)
            {
                vl = __riscv_vsetvl_e16m1(width - col);
                load_u8x8xn<8>(src + col + 0 * srcStride, s0, 1, vl);
                load_u8x8xn<8>(src + col + 1 * srcStride, s1, 1, vl);

                vint16m1_t d0 = filter8_s16x8<coeffIdx>(s0, zero, vl);
                vint16m1_t d1 = filter8_s16x8<coeffIdx>(s1, zero, vl);

                vuint16m1_t d0_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d0, zero, vl));
                vuint16m1_t d1_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d1, zero, vl));

                vuint8mf2_t d0_u8 = __riscv_vnclipu_wv_u8mf2(d0_u16, shift, 0, vl);
                vuint8mf2_t d1_u8 = __riscv_vnclipu_wv_u8mf2(d1_u16, shift, 0, vl);

                __riscv_vse8_v_u8mf2(dst + col + 0 * dstStride, d0_u8, vl);
                __riscv_vse8_v_u8mf2(dst + col + 1 * dstStride, d1_u8, vl);
            }

            src += 2 * srcStride;
            dst += 2 * dstStride;
        }
    }
}

template<int coeffIdx, int width, int height>
void interp8_vert_pp_rvv(const pixel *src, intptr_t srcStride, pixel *dst,
                           intptr_t dstStride)
{
    const int N_TAPS = 8;
    src -= (N_TAPS / 2 - 1) * srcStride;

    size_t vl = 16;

    vuint8mf2_t shift = __riscv_vmv_v_x_u8mf2(IF_FILTER_PREC, vl);
    vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, vl);
    vint16m1_t t0, t1, t2, t3, t4, t5, t6;
    vint16m1_t r0, r1, r2, r3;
    vint16m1_t *s0[11] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6, &r0, &r1, &r2, &r3};

    for (int col = 0; col < width; col += vl)
    {
        vl = __riscv_vsetvl_e16m1(width - col);
        const uint8_t *s = src;
        uint8_t *d = dst;

        load_u8x8xn<7>(s, s0, srcStride, vl);
        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            load_u8x8xn<4>(s, s0 + 7, srcStride, vl);
            vint16m1_t d0 = filter8_s16x8<coeffIdx>(s0, zero, vl);
            vint16m1_t d1 = filter8_s16x8<coeffIdx>(s0 + 1, zero, vl);
            vint16m1_t d2 = filter8_s16x8<coeffIdx>(s0 + 2, zero, vl);
            vint16m1_t d3 = filter8_s16x8<coeffIdx>(s0 + 3, zero, vl);

            vuint16m1_t d0_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d0, zero, vl));
            vuint16m1_t d1_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d1, zero, vl));
            vuint16m1_t d2_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d2, zero, vl));
            vuint16m1_t d3_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d3, zero, vl));

            vuint8mf2_t d0_u8 = __riscv_vnclipu_wv_u8mf2(d0_u16, shift, 0, vl);
            vuint8mf2_t d1_u8 = __riscv_vnclipu_wv_u8mf2(d1_u16, shift, 0, vl);
            vuint8mf2_t d2_u8 = __riscv_vnclipu_wv_u8mf2(d2_u16, shift, 0, vl);
            vuint8mf2_t d3_u8 = __riscv_vnclipu_wv_u8mf2(d3_u16, shift, 0, vl);

            __riscv_vse8_v_u8mf2(d + 0 * dstStride, d0_u8, vl);
            __riscv_vse8_v_u8mf2(d + 1 * dstStride, d1_u8, vl);
            __riscv_vse8_v_u8mf2(d + 2 * dstStride, d2_u8, vl);
            __riscv_vse8_v_u8mf2(d + 3 * dstStride, d3_u8, vl);

            *s0[0] = *s0[4];
            *s0[1] = *s0[5];
            *s0[2] = *s0[6];
            *s0[3] = *s0[7];
            *s0[4] = *s0[8];
            *s0[5] = *s0[9];
            *s0[6] = *s0[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        src += vl;
        dst += vl;
    }
}

// Element-wise of g_chromaFilter
const int16_t g_chromaFilter8[8][NTAPS_CHROMA] =
{
    { 0, 64,  0, 0 },
    { -2, 58, 10, -2 },
    { -4, 54, 16, -2 },
    { -6, 46, 28, -4 },
    { -4, 36, 36, -4 },
    { -4, 28, 46, -6 },
    { -2, 16, 54, -4 },
    { -2, 10, 58, -2 }
};

template<bool coeff4, int width, int height>
void interp4_horiz_pp_rvv(const pixel *src, intptr_t srcStride, pixel *dst,
                           intptr_t dstStride, int coeffIdx)
{
    const int N_TAPS = 4;
    src -= (N_TAPS / 2 - 1) * 1;

    const int16_t* filter = g_chromaFilter8[coeffIdx];

    size_t vl = 32;

    if (width > 8)
    {
        vuint8m1_t shift = __riscv_vmv_v_x_u8m1(IF_FILTER_PREC, vl);
        vint16m2_t zero = __riscv_vmv_v_x_i16m2(0, vl);
        vint16m2_t t0, t1, t2, t3, r0, r1, r2, r3;
        vint16m2_t *s0[4] = {&t0, &t1, &t2, &t3};
        vint16m2_t *s1[4] = {&r0, &r1, &r2, &r3};

        for (int row = 0; row < height; row +=2)
        {
            for (int col = 0; col < width; col += vl)
            {
                vl = __riscv_vsetvl_e16m2(width - col);

                load_u8x16xn<4>(src + col + 0 * srcStride, s0, 1, vl);
                load_u8x16xn<4>(src + col + 1 * srcStride, s1, 1, vl);
                vint16m2_t d0 = filter4_s16x16<coeff4>(s0, zero, filter, vl);
                vint16m2_t d1 = filter4_s16x16<coeff4>(s1, zero, filter, vl);

                vuint16m2_t d0_u16 = __riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax_vv_i16m2(d0, zero, vl));
                vuint16m2_t d1_u16 = __riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax_vv_i16m2(d1, zero, vl));

                vuint8m1_t d0_u8 = __riscv_vnclipu_wv_u8m1(d0_u16, shift, 0, vl);
                vuint8m1_t d1_u8 = __riscv_vnclipu_wv_u8m1(d1_u16, shift, 0, vl);

                __riscv_vse8_v_u8m1(dst + col + 0 * dstStride, d0_u8, vl);
                __riscv_vse8_v_u8m1(dst + col + 1 * dstStride, d1_u8, vl);
            }

            src += 2 * srcStride;
            dst += 2 * dstStride;
        }
    }
    else
    {
        vuint8mf2_t shift = __riscv_vmv_v_x_u8mf2(IF_FILTER_PREC, vl);
        vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t t0, t1, t2, t3, r0, r1, r2, r3;
        vint16m1_t *s0[4] = {&t0, &t1, &t2, &t3};
        vint16m1_t *s1[4] = {&r0, &r1, &r2, &r3};

        for (int row = 0; row < height; row += 2)
        {
            int col = 0;
            for (; col < width; col += vl)
            {
                vl = __riscv_vsetvl_e16m1(width - col);

                load_u8x8xn<4>(src + col + 0 * srcStride, s0, 1, vl);
                load_u8x8xn<4>(src + col + 1 * srcStride, s1, 1, vl);
                vint16m1_t d0 = filter4_s16x8<coeff4>(s0, zero, filter, vl);
                vint16m1_t d1 = filter4_s16x8<coeff4>(s1, zero, filter, vl);

                vuint16m1_t d0_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d0, zero, vl));
                vuint16m1_t d1_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d1, zero, vl));

                vuint8mf2_t d0_u8 = __riscv_vnclipu_wv_u8mf2(d0_u16, shift, 0, vl);
                vuint8mf2_t d1_u8 = __riscv_vnclipu_wv_u8mf2(d1_u16, shift, 0, vl);

                __riscv_vse8_v_u8mf2(dst + col + 0 * dstStride, d0_u8, vl);
                __riscv_vse8_v_u8mf2(dst + col + 1 * dstStride, d1_u8, vl);
            }

            src += 2 * srcStride;
            dst += 2 * dstStride;
        }
    }
}

template<bool coeff4, int width, int height>
void interp4_vert_pp_rvv(const pixel *src, intptr_t srcStride, pixel *dst,
                           intptr_t dstStride, int coeffIdx)
{
    const int N_TAPS = 4;
    src -= (N_TAPS / 2 - 1) * srcStride;

    const int16_t* filter = g_chromaFilter8[coeffIdx];

    size_t vl = 16;
    vuint8mf2_t shift = __riscv_vmv_v_x_u8mf2(IF_FILTER_PREC, vl);
    vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, vl);
    vint16m1_t t0, t1, t2, t3, t4, t5, t6;
    vint16m1_t *s0[7] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6};

    for (int col = 0; col < width; col += vl)
    {
        vl = __riscv_vsetvl_e16m1(width - col);
        const uint8_t *s = src;
        uint8_t *d = dst;
        load_u8x8xn<3>(s, s0, srcStride, vl);
        s += 3 * srcStride;

        for (int row = 0; row + 4 <= height; row += 4)
        {
            load_u8x8xn<4>(s, s0 + 3, srcStride, vl);
            vint16m1_t d0 = filter4_s16x8<coeff4>(s0, zero, filter, vl);
            vint16m1_t d1 = filter4_s16x8<coeff4>(s0 + 1, zero, filter, vl);
            vint16m1_t d2 = filter4_s16x8<coeff4>(s0 + 2, zero, filter, vl);
            vint16m1_t d3 = filter4_s16x8<coeff4>(s0 + 3, zero, filter, vl);

            vuint16m1_t d0_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d0, zero, vl));
            vuint16m1_t d1_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d1, zero, vl));
            vuint16m1_t d2_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d2, zero, vl));
            vuint16m1_t d3_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d3, zero, vl));

            vuint8mf2_t d0_u8 = __riscv_vnclipu_wv_u8mf2(d0_u16, shift, 0, vl);
            vuint8mf2_t d1_u8 = __riscv_vnclipu_wv_u8mf2(d1_u16, shift, 0, vl);
            vuint8mf2_t d2_u8 = __riscv_vnclipu_wv_u8mf2(d2_u16, shift, 0, vl);
            vuint8mf2_t d3_u8 = __riscv_vnclipu_wv_u8mf2(d3_u16, shift, 0, vl);

            __riscv_vse8_v_u8mf2(d + 0 * dstStride, d0_u8, vl);
            __riscv_vse8_v_u8mf2(d + 1 * dstStride, d1_u8, vl);
            __riscv_vse8_v_u8mf2(d + 2 * dstStride, d2_u8, vl);
            __riscv_vse8_v_u8mf2(d + 3 * dstStride, d3_u8, vl);

            *s0[0] = *s0[4];
            *s0[1] = *s0[5];
            *s0[2] = *s0[6];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        if (height & 2)
        {
            load_u8x8xn<2>(s, s0 + 3, srcStride, vl);
            vint16m1_t d0 = filter4_s16x8<coeff4>(s0, zero, filter, vl);
            vint16m1_t d1 = filter4_s16x8<coeff4>(s0 + 1, zero, filter, vl);

            vuint16m1_t d0_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d0, zero, vl));
            vuint16m1_t d1_u16 = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vmax_vv_i16m1(d1, zero, vl));

            vuint8mf2_t d0_u8 = __riscv_vnclipu_wv_u8mf2(d0_u16, shift, 0, vl);
            vuint8mf2_t d1_u8 = __riscv_vnclipu_wv_u8mf2(d1_u16, shift, 0, vl);

            __riscv_vse8_v_u8mf2(d + 0 * dstStride, d0_u8, vl);
            __riscv_vse8_v_u8mf2(d + 1 * dstStride, d1_u8, vl);
        }
        src += vl;
        dst += vl;
    }
}


template<int N, int width, int height>
void interp_horiz_pp_rvv(const pixel *src, intptr_t srcStride, pixel *dst,
                          intptr_t dstStride, int coeffIdx)
{
    if (N == 8)
    {
        switch (coeffIdx)
        {
        case 1:
            return interp8_horiz_pp_rvv<1, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 2:
            return interp8_horiz_pp_rvv<2, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 3:
            return interp8_horiz_pp_rvv<3, width, height>(src, srcStride, dst,
                                                           dstStride);
        }
    }
    else
    {
        switch (coeffIdx)
        {
        case 4:
            return interp4_horiz_pp_rvv<true, width, height>(src, srcStride,
                                                              dst, dstStride,
                                                              coeffIdx);
        default:
            return interp4_horiz_pp_rvv<false, width, height>(src, srcStride,
                                                               dst, dstStride,
                                                               coeffIdx);
        }
    }
}

template<int coeffIdx, int width, int height>
void interp8_horiz_ps_rvv(const uint8_t *src, intptr_t srcStride, int16_t *dst,
                           intptr_t dstStride, int isRowExt)
{
    int blkheight = height;
    const int N_TAPS = 8;
    src -= (N_TAPS / 2 - 1);

    if (isRowExt)
    {
        src -= (N_TAPS / 2 - 1) * srcStride;
        blkheight += N_TAPS - 1;
    }

    size_t vl = 16;

    vint16m2_t c = __riscv_vmv_v_x_i16m2( -IF_INTERNAL_OFFS, vl);
    vint16m1_t c2 = __riscv_vmv_v_x_i16m1(-IF_INTERNAL_OFFS, vl);

    for (int row = 0; row + 2 <= blkheight; row += 2)
    {
        int col = 0;
        for (; col + 16 <= width; col += 16)
        {
            vint16m2_t t0, t1, t2, t3, t4, t5, t6, t7;
            vint16m2_t r0, r1, r2, r3, r4, r5, r6, r7;
            vint16m2_t *s0[8] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7};
            vint16m2_t *s1[8] = {&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7};

            load_u8x16xn<8>(src + col + 0 * srcStride, s0, 1, vl);
            load_u8x16xn<8>(src + col + 1 * srcStride, s1, 1, vl);
            vint16m2_t d0 = filter8_s16x16<coeffIdx>(s0, c, vl);
            vint16m2_t d1 = filter8_s16x16<coeffIdx>(s1, c, vl);

            __riscv_vse16_v_i16m2(dst + col + 0 * dstStride, d0, vl);
            __riscv_vse16_v_i16m2(dst + col + 1 * dstStride, d1, vl);
        }

        if(col < width){
            vint16m1_t t0, t1, t2, t3, t4, t5, t6, t7;
            vint16m1_t r0, r1, r2, r3, r4, r5, r6, r7;
            vint16m1_t *s0[8] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7};
            vint16m1_t *s1[8] = {&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7};

            for (; col + 8 <= width; col += 8)
            {
                load_u8x8xn<8>(src + col + 0 * srcStride, s0, 1, 8);
                load_u8x8xn<8>(src + col + 1 * srcStride, s1, 1, 8);

                vint16m1_t d0 = filter8_s16x8<coeffIdx>(s0, c2, 8);
                vint16m1_t d1 = filter8_s16x8<coeffIdx>(s1, c2, 8);

                __riscv_vse16_v_i16m1(dst + col + 0 * dstStride, d0, 8);
                __riscv_vse16_v_i16m1(dst + col + 1 * dstStride, d1, 8);
            }

            if (width % 8 != 0)
            {
                load_u8x8xn<8>(src + col + 0 * srcStride, s0, 1, 4);
                load_u8x8xn<8>(src + col + 1 * srcStride, s1, 1, 4);

                vint16m1_t d0 = filter8_s16x8<coeffIdx>(s0, c2, 4);
                vint16m1_t d1 = filter8_s16x8<coeffIdx>(s1, c2, 4);

                __riscv_vse16_v_i16m1(dst + col + 0 * dstStride, d0, 4);
                __riscv_vse16_v_i16m1(dst + col + 1 * dstStride, d1, 4);
            }
        }

        src += 2 * srcStride;
        dst += 2 * dstStride;
    }

    if (isRowExt)
    {
        int col = 0;
        vint16m1_t t0, t1, t2, t3, t4, t5, t6, t7;
        vint16m1_t *s0[8] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7};

        for (; col + 8 <= width; col += 8)
        {
            load_u8x8xn<8>(src + col, s0, 1, 8);
            vint16m1_t d0 = filter8_s16x8<coeffIdx>(s0, c2, 8);
            __riscv_vse16_v_i16m1(dst + col, d0, 8);
        }

        if (width % 8 != 0)
        {
            load_u8x8xn<8>(src + col, s0, 1, 8);
            vint16m1_t d0 = filter8_s16x8<coeffIdx>(s0, c2, 8);
            __riscv_vse16_v_i16m1(dst + col, d0, 4);
        }
    }
}

template<int coeffIdx, int width, int height>
void interp8_vert_ps_rvv(const uint8_t *src, intptr_t srcStride, int16_t *dst,
                           intptr_t dstStride)
{
    const int N_TAPS = 8;
    src -= (N_TAPS / 2 - 1) * srcStride;

    size_t vl = 16;
    vint16m1_t c = __riscv_vmv_v_x_i16m1(-IF_INTERNAL_OFFS, vl);

    vint16m1_t t0, t1, t2, t3, t4, t5, t6;
    vint16m1_t r0, r1, r2, r3;
    vint16m1_t *s0[11] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6, &r0, &r1, &r2, &r3};

    for (int col = 0; col < width; col += vl)
    {
        vl = __riscv_vsetvl_e16m1(width - col);
        const uint8_t *s = src;
        int16_t *d = dst;

        load_u8x8xn<7>(s, s0, srcStride, vl);
        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            load_u8x8xn<4>(s, s0 + 7, srcStride, vl);

            vint16m1_t d0 = filter8_s16x8<coeffIdx>(s0, c, vl);
            vint16m1_t d1 = filter8_s16x8<coeffIdx>(s0 + 1, c, vl);
            vint16m1_t d2 = filter8_s16x8<coeffIdx>(s0 + 2, c, vl);
            vint16m1_t d3 = filter8_s16x8<coeffIdx>(s0 + 3, c, vl);

            __riscv_vse16_v_i16m1(d + 0 * dstStride, d0, vl);
            __riscv_vse16_v_i16m1(d + 1 * dstStride, d1, vl);
            __riscv_vse16_v_i16m1(d + 2 * dstStride, d2, vl);
            __riscv_vse16_v_i16m1(d + 3 * dstStride, d3, vl);

            *s0[0] = *s0[4];
            *s0[1] = *s0[5];
            *s0[2] = *s0[6];
            *s0[3] = *s0[7];
            *s0[4] = *s0[8];
            *s0[5] = *s0[9];
            *s0[6] = *s0[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        src += vl;
        dst += vl;
    }
}

template<bool coeff4, int width, int height>
void interp4_horiz_ps_rvv(const uint8_t *src, intptr_t srcStride, int16_t *dst,
                           intptr_t dstStride, int coeffIdx,
                           int isRowExt)
{
    int blkheight = height;
    const int N_TAPS = 4;
    src -= (N_TAPS / 2 - 1);

    if (isRowExt)
    {
        src -= (N_TAPS / 2 - 1) * srcStride;
        blkheight += N_TAPS - 1;
    }

    const int16_t* filter = g_chromaFilter8[coeffIdx];

    size_t vl = 8;
    const vint16m1_t c = __riscv_vmv_v_x_i16m1(-IF_INTERNAL_OFFS, vl);
    size_t store_vl = width % 8;
    vint16m1_t t0, t1, t2, t3, r0, r1, r2, r3;
    vint16m1_t *s0[4] = {&t0, &t1, &t2, &t3};
    vint16m1_t *s1[4] = {&r0, &r1, &r2, &r3};

    for (int row = 0; row + 2 <= blkheight; row += 2)
    {
        int col = 0;
        for (; col + 8 <= width; col += 8)
        {
            load_u8x8xn<4>(src + col + 0 * srcStride, s0, 1, vl);
            load_u8x8xn<4>(src + col + 1 * srcStride, s1, 1, vl);

            vint16m1_t d0 = filter4_s16x8<coeff4>(s0, c, filter, vl);
            vint16m1_t d1 = filter4_s16x8<coeff4>(s1, c, filter, vl);

            __riscv_vse16_v_i16m1(dst + col + 0 * dstStride, d0, vl);
            __riscv_vse16_v_i16m1(dst + col + 1 * dstStride, d1, vl);
        }

        if (width % 8 != 0)
        {
            load_u8x8xn<4>(src + col + 0 * srcStride, s0, 1, store_vl);
            load_u8x8xn<4>(src + col + 1 * srcStride, s1, 1, store_vl);

            vint16m1_t d0 = filter4_s16x8<coeff4>(s0, c, filter, store_vl);
            vint16m1_t d1 = filter4_s16x8<coeff4>(s1, c, filter, store_vl);

            __riscv_vse16_v_i16m1(dst + col + 0 * dstStride, d0, store_vl);
            __riscv_vse16_v_i16m1(dst + col + 1 * dstStride, d1, store_vl);
        }

        src += 2 * srcStride;
        dst += 2 * dstStride;
    }

    if (isRowExt)
    {
        int col = 0;
        for (; col + 8 <= width; col += 8)
        {
            load_u8x8xn<4>(src + col, s0, 1, vl);
            vint16m1_t d0 = filter4_s16x8<coeff4>(s0, c, filter, vl);
            __riscv_vse16_v_i16m1(dst + col, d0, vl);
        }

        if (width % 8 != 0)
        {
            load_u8x8xn<4>(src + col, s0, 1, store_vl);
            vint16m1_t d0 = filter4_s16x8<coeff4>(s0, c, filter, store_vl);
            __riscv_vse16_v_i16m1(dst + col, d0, store_vl);
        }
    }
}

template<bool coeff4, int width, int height>
void interp4_vert_ps_rvv(const uint8_t *src, intptr_t srcStride, int16_t *dst,
                           intptr_t dstStride, int coeffIdx)
{
    const int N_TAPS = 4;
    src -= (N_TAPS / 2 - 1) * srcStride;

    const int16_t* filter = g_chromaFilter8[coeffIdx];

    size_t vl = 16;
    const vint16m1_t c = __riscv_vmv_v_x_i16m1(-IF_INTERNAL_OFFS, vl);
    vint16m1_t t0, t1, t2, t3, t4, t5, t6;
    vint16m1_t *s0[7] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6};

    for (int col = 0; col < width; col += vl)
    {
        vl = __riscv_vsetvl_e16m1(width - col);
        const uint8_t *s = src;
        int16_t *d = dst;

        load_u8x8xn<3>(s, s0, srcStride, vl);
        s += 3 * srcStride;

        for (int row = 0; row + 4 <= height; row += 4)
        {
            load_u8x8xn<4>(s, s0 + 3, srcStride, vl);

            vint16m1_t d0 = filter4_s16x8<coeff4>(s0, c, filter, vl);
            vint16m1_t d1 = filter4_s16x8<coeff4>(s0 + 1, c, filter, vl);
            vint16m1_t d2 = filter4_s16x8<coeff4>(s0 + 2, c, filter, vl);
            vint16m1_t d3 = filter4_s16x8<coeff4>(s0 + 3, c, filter, vl);

            __riscv_vse16_v_i16m1(d + 0 * dstStride, d0, vl);
            __riscv_vse16_v_i16m1(d + 1 * dstStride, d1, vl);
            __riscv_vse16_v_i16m1(d + 2 * dstStride, d2, vl);
            __riscv_vse16_v_i16m1(d + 3 * dstStride, d3, vl);

            *s0[0] = *s0[4];
            *s0[1] = *s0[5];
            *s0[2] = *s0[6];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        if(height & 2)
        {
            load_u8x8xn<2>(s, s0 + 3, srcStride, vl);

            vint16m1_t d0 = filter4_s16x8<coeff4>(s0, c, filter, vl);
            vint16m1_t d1 = filter4_s16x8<coeff4>(s0 + 1, c, filter, vl);

            __riscv_vse16_v_i16m1(d + 0 * dstStride, d0, vl);
            __riscv_vse16_v_i16m1(d + 1 * dstStride, d1, vl);
        }

        src += vl;
        dst += vl;
    }
}

template<int N, int width, int height>
void interp_horiz_ps_rvv(const pixel *src, intptr_t srcStride, int16_t *dst,
                          intptr_t dstStride, int coeffIdx, int isRowExt)
{
    if (N == 8)
    {
        switch (coeffIdx)
        {
        case 1:
            return interp8_horiz_ps_rvv<1, width, height>(src, srcStride, dst,
                                                           dstStride, isRowExt);
        case 2:
            return interp8_horiz_ps_rvv<2, width, height>(src, srcStride, dst,
                                                           dstStride, isRowExt);
        case 3:
            return interp8_horiz_ps_rvv<3, width, height>(src, srcStride, dst,
                                                           dstStride, isRowExt);
        }
    }
    else
    {
        switch (coeffIdx)
        {
        case 4:
            return interp4_horiz_ps_rvv<true, width, height>(src, srcStride,
                                                              dst, dstStride,
                                                              coeffIdx,
                                                              isRowExt);
        default:
            return interp4_horiz_ps_rvv<false, width, height>(src, srcStride,
                                                               dst, dstStride,
                                                               coeffIdx,
                                                               isRowExt);
        }
    }
}

template<int N, int width, int height>
void interp_vert_pp_rvv(const pixel *src, intptr_t srcStride, pixel *dst,
                          intptr_t dstStride, int coeffIdx)
{
    if (N == 8)
    {
        switch (coeffIdx)
        {
        case 1:
            return interp8_vert_pp_rvv<1, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 2:
            return interp8_vert_pp_rvv<2, width, height>(src, srcStride, dst,
                                                           dstStride);
        case 3:
            return interp8_vert_pp_rvv<3, width, height>(src, srcStride, dst,
                                                           dstStride);
        }
    }
    else
    {
        switch (coeffIdx)
        {
        case 4:
            return interp4_vert_pp_rvv<true, width, height>(src, srcStride,
                                                              dst, dstStride,
                                                              coeffIdx);
        default:
            return interp4_vert_pp_rvv<false, width, height>(src, srcStride,
                                                               dst, dstStride,
                                                               coeffIdx);
        }
    }
}

template<int N, int width, int height>
void interp_vert_ps_rvv(const pixel *src, intptr_t srcStride, int16_t *dst,
                         intptr_t dstStride, int coeffIdx)
{
    if (N == 8)
    {
        switch (coeffIdx)
        {
        case 1:
            return interp8_vert_ps_rvv<1, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 2:
            return interp8_vert_ps_rvv<2, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 3:
            return interp8_vert_ps_rvv<3, width, height>(src, srcStride, dst,
                                                          dstStride);
        }
    }
    else
    {
        switch (coeffIdx)
        {
        case 4:
            return interp4_vert_ps_rvv<true, width, height>(src, srcStride,
                                                             dst, dstStride,
                                                             coeffIdx);
        default:
            return interp4_vert_ps_rvv<false, width, height>(src, srcStride,
                                                              dst, dstStride,
                                                              coeffIdx);
        }
    }
}

template<int coeffIdx, int width, int height>
void interp8_vert_sp_rvv(const int16_t *src, intptr_t srcStride, pixel *dst,
                          intptr_t dstStride)
{
    const int headRoom = IF_INTERNAL_PREC - X265_DEPTH;
    const int shift0 = IF_FILTER_PREC + headRoom;
    const int offset = (1 << (shift0 - 1)) + (IF_INTERNAL_OFFS << IF_FILTER_PREC);
    const int N_TAPS = 8;

    src -= (N_TAPS / 2 - 1) * srcStride;
    size_t vl = 16;

    vuint16m1_t shift = __riscv_vmv_v_x_u16m1(shift0, vl);
    vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, vl);
    vint32m2_t c = __riscv_vmv_v_x_i32m2(offset, vl);
    vint16m1_t maxVal = __riscv_vmv_v_x_i16m1((1 << X265_DEPTH) - 1, vl);

    vint32m2_t t0, t1, t2, t3, t4, t5, t6;
    vint32m2_t r0, r1, r2, r3;
    vint32m2_t *s0[11] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6, &r0, &r1, &r2, &r3};

    for (int col = 0; col < width; col += vl)
    {
        vl = __riscv_vsetvl_e32m2(width - col);
        const int16_t *s = src;
        uint8_t *d = dst;

        load_s16x8xn<7>(s, s0, srcStride, vl);

        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            load_s16x8xn<4>(s, s0 + 7, srcStride, vl);

            vint32m2_t d0 = filter8_s32x8<coeffIdx>(s0, c, vl);
            vint32m2_t d1 = filter8_s32x8<coeffIdx>(s0 + 1, c, vl);
            vint32m2_t d2 = filter8_s32x8<coeffIdx>(s0 + 2, c, vl);
            vint32m2_t d3 = filter8_s32x8<coeffIdx>(s0 + 3, c, vl);

            vint16m1_t d0_i16 = __riscv_vnclip_wv_i16m1(d0, shift, 2, vl);
            vint16m1_t d1_i16 = __riscv_vnclip_wv_i16m1(d1, shift, 2, vl);
            vint16m1_t d2_i16 = __riscv_vnclip_wv_i16m1(d2, shift, 2, vl);
            vint16m1_t d3_i16 = __riscv_vnclip_wv_i16m1(d3, shift, 2, vl);

            d0_i16 = __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(d0_i16, zero, vl), maxVal, vl);
            d1_i16 = __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(d1_i16, zero, vl), maxVal, vl);
            d2_i16 = __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(d2_i16, zero, vl), maxVal, vl);
            d3_i16 = __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(d3_i16, zero, vl), maxVal, vl);

            vuint8mf2_t d0_u8 = __riscv_vnsrl_wx_u8mf2(__riscv_vreinterpret_v_i16m1_u16m1(d0_i16), 0, vl);
            vuint8mf2_t d1_u8 = __riscv_vnsrl_wx_u8mf2(__riscv_vreinterpret_v_i16m1_u16m1(d1_i16), 0, vl);
            vuint8mf2_t d2_u8 = __riscv_vnsrl_wx_u8mf2(__riscv_vreinterpret_v_i16m1_u16m1(d2_i16), 0, vl);
            vuint8mf2_t d3_u8 = __riscv_vnsrl_wx_u8mf2(__riscv_vreinterpret_v_i16m1_u16m1(d3_i16), 0, vl);

            __riscv_vse8_v_u8mf2(d + 0 * dstStride, d0_u8, vl);
            __riscv_vse8_v_u8mf2(d + 1 * dstStride, d1_u8, vl);
            __riscv_vse8_v_u8mf2(d + 2 * dstStride, d2_u8, vl);
            __riscv_vse8_v_u8mf2(d + 3 * dstStride, d3_u8, vl);

            *s0[0] = *s0[4];
            *s0[1] = *s0[5];
            *s0[2] = *s0[6];
            *s0[3] = *s0[7];
            *s0[4] = *s0[8];
            *s0[5] = *s0[9];
            *s0[6] = *s0[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        src += vl;
        dst += vl;
    }
}

template<bool coeff4, int width, int height>
void interp4_vert_sp_rvv(const int16_t *src, intptr_t srcStride, uint8_t *dst,
                          intptr_t dstStride, int coeffIdx)
{
    assert(X265_DEPTH == 8);
    const int headRoom = IF_INTERNAL_PREC - X265_DEPTH;
    const int shift0 = IF_FILTER_PREC + headRoom;
    const int offset = (1 << (shift0 - 1)) + (IF_INTERNAL_OFFS << IF_FILTER_PREC);
    const int N_TAPS = 4;
    src -= (N_TAPS / 2 - 1) * srcStride;

    size_t vl = 16;
    const int16_t* filter = g_chromaFilter8[coeffIdx];

    vuint16m1_t shift = __riscv_vmv_v_x_u16m1(shift0, vl);
    vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, vl);
    vint32m2_t c = __riscv_vmv_v_x_i32m2(offset, vl);
    vint16m1_t maxVal = __riscv_vmv_v_x_i16m1((1 << X265_DEPTH) - 1, vl);

    vint32m2_t t0, t1, t2, t3, t4, t5, t6;
    vint32m2_t *s0[7] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6};

    for (int col = 0; col < width; col += vl)
    {
        vl = __riscv_vsetvl_e32m2(width - col);
        const int16_t *s = src;
        uint8_t *d = dst;
        load_s16x8xn<3>(s, s0, srcStride, vl);
        s += 3 * srcStride;

        for (int row = 0; row + 4 <= height; row += 4)
        {
           load_s16x8xn<4>(s, s0 + 3, srcStride, vl);

            vint32m2_t d0 = filter4_s32x8<coeff4>(s0, c, filter, vl);
            vint32m2_t d1 = filter4_s32x8<coeff4>(s0 + 1, c, filter, vl);
            vint32m2_t d2 = filter4_s32x8<coeff4>(s0 + 2, c, filter, vl);
            vint32m2_t d3 = filter4_s32x8<coeff4>(s0 + 3, c, filter, vl);

            vint16m1_t d0_i16 = __riscv_vnclip_wv_i16m1(d0, shift, 2, vl);
            vint16m1_t d1_i16 = __riscv_vnclip_wv_i16m1(d1, shift, 2, vl);
            vint16m1_t d2_i16 = __riscv_vnclip_wv_i16m1(d2, shift, 2, vl);
            vint16m1_t d3_i16 = __riscv_vnclip_wv_i16m1(d3, shift, 2, vl);

            d0_i16 = __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(d0_i16, zero, vl), maxVal, vl);
            d1_i16 = __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(d1_i16, zero, vl), maxVal, vl);
            d2_i16 = __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(d2_i16, zero, vl), maxVal, vl);
            d3_i16 = __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(d3_i16, zero, vl), maxVal, vl);

            vuint8mf2_t d0_u8 = __riscv_vnsrl_wx_u8mf2(__riscv_vreinterpret_v_i16m1_u16m1(d0_i16), 0, vl);
            vuint8mf2_t d1_u8 = __riscv_vnsrl_wx_u8mf2(__riscv_vreinterpret_v_i16m1_u16m1(d1_i16), 0, vl);
            vuint8mf2_t d2_u8 = __riscv_vnsrl_wx_u8mf2(__riscv_vreinterpret_v_i16m1_u16m1(d2_i16), 0, vl);
            vuint8mf2_t d3_u8 = __riscv_vnsrl_wx_u8mf2(__riscv_vreinterpret_v_i16m1_u16m1(d3_i16), 0, vl);

            __riscv_vse8_v_u8mf2(d + 0 * dstStride, d0_u8, vl);
            __riscv_vse8_v_u8mf2(d + 1 * dstStride, d1_u8, vl);
            __riscv_vse8_v_u8mf2(d + 2 * dstStride, d2_u8, vl);
            __riscv_vse8_v_u8mf2(d + 3 * dstStride, d3_u8, vl);

            *s0[0] = *s0[4];
            *s0[1] = *s0[5];
            *s0[2] = *s0[6];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        if(height & 2)
        {
            load_s16x8xn<2>(s, s0 + 3, srcStride, vl);

            vint32m2_t d0 = filter4_s32x8<coeff4>(s0, c, filter, vl);
            vint32m2_t d1 = filter4_s32x8<coeff4>(s0 + 1, c, filter, vl);

            vint16m1_t d0_i16 = __riscv_vnclip_wv_i16m1(d0, shift, 2, vl);
            vint16m1_t d1_i16 = __riscv_vnclip_wv_i16m1(d1, shift, 2, vl);

            d0_i16 = __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(d0_i16, zero, vl), maxVal, vl);
            d1_i16 = __riscv_vmin_vv_i16m1(__riscv_vmax_vv_i16m1(d1_i16, zero, vl), maxVal, vl);

            vuint8mf2_t d0_u8 = __riscv_vnsrl_wx_u8mf2(__riscv_vreinterpret_v_i16m1_u16m1(d0_i16), 0, vl);
            vuint8mf2_t d1_u8 = __riscv_vnsrl_wx_u8mf2(__riscv_vreinterpret_v_i16m1_u16m1(d1_i16), 0, vl);

            __riscv_vse8_v_u8mf2(d + 0 * dstStride, d0_u8, vl);
            __riscv_vse8_v_u8mf2(d + 1 * dstStride, d1_u8, vl);
        }

        src += vl;
        dst += vl;
    }
}

template<int N, int width, int height>
void interp_vert_sp_rvv(const int16_t *src, intptr_t srcStride, pixel *dst,
                         intptr_t dstStride, int coeffIdx)
{
    if (N == 8)
    {
        switch (coeffIdx)
        {
        case 1:
            return interp8_vert_sp_rvv<1, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 2:
            return interp8_vert_sp_rvv<2, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 3:
            return interp8_vert_sp_rvv<3, width, height>(src, srcStride, dst,
                                                          dstStride);
        }
    }
    else
    {
        switch (coeffIdx)
        {
        case 4:
            return interp4_vert_sp_rvv<true, width, height>(src, srcStride, dst,
                                                             dstStride, coeffIdx);
        default:
            return interp4_vert_sp_rvv<false, width, height>(src, srcStride, dst,
                                                              dstStride, coeffIdx);
        }
    }
}


template<int coeffIdx, int width, int height>
void interp8_vert_ss_rvv(const int16_t *src, intptr_t srcStride, int16_t *dst,
                          intptr_t dstStride)
{
    const int N_TAPS = 8;
    src -= (N_TAPS / 2 - 1) * srcStride;
    size_t vl = 16;

    vuint16m1_t shift = __riscv_vmv_v_x_u16m1(IF_FILTER_PREC, vl);
    vint32m2_t c = __riscv_vmv_v_x_i32m2(0, vl);

    vint32m2_t t0, t1, t2, t3, t4, t5, t6;
    vint32m2_t r0, r1, r2, r3;
    vint32m2_t *s0[11] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6, &r0, &r1, &r2, &r3};

    for (int col = 0; col < width; col += vl)
    {
        vl = __riscv_vsetvl_e32m2(width - col);
        const int16_t *s = src;
        int16_t *d = dst;

        load_s16x8xn<7>(s, s0, srcStride, vl);
        s += 7 * srcStride;

        for (int row = 0; row < height; row += 4)
        {
            load_s16x8xn<4>(s, s0 + 7, srcStride, vl);

            vint32m2_t d0 = filter8_s32x8<coeffIdx>(s0, c, vl);
            vint32m2_t d1 = filter8_s32x8<coeffIdx>(s0 + 1, c, vl);
            vint32m2_t d2 = filter8_s32x8<coeffIdx>(s0 + 2, c, vl);
            vint32m2_t d3 = filter8_s32x8<coeffIdx>(s0 + 3, c, vl);

            vint16m1_t d0_i16 = __riscv_vnclip_wv_i16m1(d0, shift, 2, vl);
            vint16m1_t d1_i16 = __riscv_vnclip_wv_i16m1(d1, shift, 2, vl);
            vint16m1_t d2_i16 = __riscv_vnclip_wv_i16m1(d2, shift, 2, vl);
            vint16m1_t d3_i16 = __riscv_vnclip_wv_i16m1(d3, shift, 2, vl);

            __riscv_vse16_v_i16m1(d + 0 * dstStride, d0_i16, vl);
            __riscv_vse16_v_i16m1(d + 1 * dstStride, d1_i16, vl);
            __riscv_vse16_v_i16m1(d + 2 * dstStride, d2_i16, vl);
            __riscv_vse16_v_i16m1(d + 3 * dstStride, d3_i16, vl);

            *s0[0] = *s0[4];
            *s0[1] = *s0[5];
            *s0[2] = *s0[6];
            *s0[3] = *s0[7];
            *s0[4] = *s0[8];
            *s0[5] = *s0[9];
            *s0[6] = *s0[10];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        src += vl;
        dst += vl;
    }
}

template<bool coeff4, int width, int height>
void interp4_vert_ss_rvv(const int16_t *src, intptr_t srcStride, int16_t *dst,
                          intptr_t dstStride, int coeffIdx)
{
    const int N_TAPS = 4;

    src -= (N_TAPS / 2 - 1) * srcStride;

    size_t vl = 16;
    const int16_t* filter = g_chromaFilter8[coeffIdx];

    vuint16m1_t shift = __riscv_vmv_v_x_u16m1(IF_FILTER_PREC, vl);
    vint32m2_t c = __riscv_vmv_v_x_i32m2(0, vl);

    vint32m2_t t0, t1, t2, t3, t4, t5, t6;
    vint32m2_t *s0[7] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6};

    for (int col = 0; col < width; col += vl)
    {
        vl = __riscv_vsetvl_e32m2(width - col);
        const int16_t *s = src;
        int16_t *d = dst;
        load_s16x8xn<3>(s, s0, srcStride, vl);
        s += 3 * srcStride;

        for (int row = 0; row + 4 <= height; row += 4)
        {
            load_s16x8xn<4>(s, s0 + 3, srcStride, vl);

            vint32m2_t d0 = filter4_s32x8<coeff4>(s0, c, filter, vl);
            vint32m2_t d1 = filter4_s32x8<coeff4>(s0 + 1, c, filter, vl);
            vint32m2_t d2 = filter4_s32x8<coeff4>(s0 + 2, c, filter, vl);
            vint32m2_t d3 = filter4_s32x8<coeff4>(s0 + 3, c, filter, vl);

            vint16m1_t d0_i16 = __riscv_vnclip_wv_i16m1(d0, shift, 2, vl);
            vint16m1_t d1_i16 = __riscv_vnclip_wv_i16m1(d1, shift, 2, vl);
            vint16m1_t d2_i16 = __riscv_vnclip_wv_i16m1(d2, shift, 2, vl);
            vint16m1_t d3_i16 = __riscv_vnclip_wv_i16m1(d3, shift, 2, vl);

            __riscv_vse16_v_i16m1(d + 0 * dstStride, d0_i16, vl);
            __riscv_vse16_v_i16m1(d + 1 * dstStride, d1_i16, vl);
            __riscv_vse16_v_i16m1(d + 2 * dstStride, d2_i16, vl);
            __riscv_vse16_v_i16m1(d + 3 * dstStride, d3_i16, vl);

            *s0[0] = *s0[4];
            *s0[1] = *s0[5];
            *s0[2] = *s0[6];

            s += 4 * srcStride;
            d += 4 * dstStride;
        }

        if(height & 2)
        {
            load_s16x8xn<2>(s, s0 + 3, srcStride, vl);

            vint32m2_t d0 = filter4_s32x8<coeff4>(s0, c, filter, vl);
            vint32m2_t d1 = filter4_s32x8<coeff4>(s0 + 1, c, filter, vl);

            vint16m1_t d0_i16 = __riscv_vnclip_wv_i16m1(d0, shift, 2, vl);
            vint16m1_t d1_i16 = __riscv_vnclip_wv_i16m1(d1, shift, 2, vl);

            __riscv_vse16_v_i16m1(d + 0 * dstStride, d0_i16, vl);
            __riscv_vse16_v_i16m1(d + 1 * dstStride, d1_i16, vl);
        }

        src += vl;
        dst += vl;
    }
}

template<int N, int width, int height>
void interp_vert_ss_rvv(const int16_t *src, intptr_t srcStride, int16_t *dst, intptr_t dstStride, int coeffIdx)
{
    if (N == 8)
    {
        switch (coeffIdx)
        {
        case 1:
            return interp8_vert_ss_rvv<1, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 2:
            return interp8_vert_ss_rvv<2, width, height>(src, srcStride, dst,
                                                          dstStride);
        case 3:
            return interp8_vert_ss_rvv<3, width, height>(src, srcStride, dst,
                                                          dstStride);
        }
    }
    else
    {
        switch (coeffIdx)
        {
        case 4:
            return interp4_vert_ss_rvv<true, width, height>(src, srcStride, dst,
                                                             dstStride, coeffIdx);
        default:
            return interp4_vert_ss_rvv<false, width, height>(src, srcStride, dst,
                                                              dstStride, coeffIdx);
        }
    }
}

template<int N, int width, int height>
void interp_hv_pp_rvv(const pixel *src, intptr_t srcStride, pixel *dst,
                       intptr_t dstStride, int idxX, int idxY)
{
    ALIGN_VAR_32(int16_t, immed[width * (height + N - 1)]);

    interp_horiz_ps_rvv<N, width, height>(src, srcStride, immed, width, idxX, 1);
    interp_vert_sp_rvv<N, width, height>(immed + (N / 2 - 1) * width, width, dst,
                                          dstStride, idxY);
}

#define CHROMA_420(W, H) \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_hpp = interp_horiz_pp_rvv<4, W, H>; \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_hps = interp_horiz_ps_rvv<4, W, H>; \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_vpp = interp_vert_pp_rvv<4, W, H>; \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_vps = interp_vert_ps_rvv<4, W, H>; \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_vsp = interp_vert_sp_rvv<4, W, H>; \
    p.chroma[X265_CSP_I420].pu[CHROMA_420_ ## W ## x ## H].filter_vss = interp_vert_ss_rvv<4, W, H>;

#define CHROMA_422(W, H) \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_hpp = interp_horiz_pp_rvv<4, W, H>; \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_hps = interp_horiz_ps_rvv<4, W, H>; \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_vpp = interp_vert_pp_rvv<4, W, H>; \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_vps = interp_vert_ps_rvv<4, W, H>; \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_vsp = interp_vert_sp_rvv<4, W, H>; \
    p.chroma[X265_CSP_I422].pu[CHROMA_422_ ## W ## x ## H].filter_vss = interp_vert_ss_rvv<4, W, H>;

#define CHROMA_444(W, H) \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_hpp = interp_horiz_pp_rvv<4, W, H>; \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_hps = interp_horiz_ps_rvv<4, W, H>; \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_vpp = interp_vert_pp_rvv<4, W, H>; \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_vps = interp_vert_ps_rvv<4, W, H>; \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_vsp = interp_vert_sp_rvv<4, W, H>; \
    p.chroma[X265_CSP_I444].pu[LUMA_ ## W ## x ## H].filter_vss = interp_vert_ss_rvv<4, W, H>;

#define LUMA(W, H) \
    p.pu[LUMA_ ## W ## x ## H].luma_hpp     = interp_horiz_pp_rvv<8, W, H>; \
    p.pu[LUMA_ ## W ## x ## H].luma_hps     = interp_horiz_ps_rvv<8, W, H>; \
    p.pu[LUMA_ ## W ## x ## H].luma_vpp     = interp_vert_pp_rvv<8, W, H>; \
    p.pu[LUMA_ ## W ## x ## H].luma_vps     = interp_vert_ps_rvv<8, W, H>; \
    p.pu[LUMA_ ## W ## x ## H].luma_vsp     = interp_vert_sp_rvv<8, W, H>; \
    p.pu[LUMA_ ## W ## x ## H].luma_vss     = interp_vert_ss_rvv<8, W, H>; \
    p.pu[LUMA_ ## W ## x ## H].luma_hvpp    = interp_hv_pp_rvv<8, W, H>;

void setupFilterPrimitives_rvv(EncoderPrimitives &p)
{
    LUMA(4, 4);
    LUMA(4, 8);
    LUMA(4, 16);
    LUMA(12, 16);
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

    CHROMA_420(4, 2);
    CHROMA_420(4, 4);
    CHROMA_420(4, 8);
    CHROMA_420(4, 16);
    CHROMA_420(6, 8);
    CHROMA_420(12, 16);
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

    CHROMA_422(4, 4);
    CHROMA_422(4, 8);
    CHROMA_422(4, 16);
    CHROMA_422(4, 32);
    CHROMA_422(6, 16);
    CHROMA_422(12, 32);
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

    CHROMA_444(4, 4);
    CHROMA_444(4, 8);
    CHROMA_444(4, 16);
    CHROMA_444(12, 16);
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
#endif //!HIGH_BIT_DEPTH
}; // namespace X265_NS
