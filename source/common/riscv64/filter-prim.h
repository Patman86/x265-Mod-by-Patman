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

#include <riscv_vector.h>
#include "primitives.h"

namespace X265_NS
{
template<int N>
static void inline load_u8x8xn(const uint8_t *s, vint16m1_t **d, const intptr_t stride, const size_t vl)
{
    for (int i = 0; i < N; ++i)
    {
        *d[i] = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(__riscv_vle8_v_u8mf2(s + i * stride, vl),vl));
    }
}

template<int N>
static void inline load_u8x16xn(const uint8_t *s, vint16m2_t **d, const intptr_t stride, const size_t vl)
{
    for (int i = 0; i < N; ++i)
    {
        *d[i] = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(s + i * stride, vl),vl));
    }
}

template<int N>
static void inline load_s16x8xn(const int16_t *s, vint32m2_t **d, const intptr_t stride, const size_t vl)
{
    for (int i = 0; i < N; ++i)
    {
        *d[i] = __riscv_vsext_vf2_i32m2(__riscv_vle16_v_i16m1(s + i * stride, vl), vl);
    }
}

#if !HIGH_BIT_DEPTH

/* N_TAPS = 8 */
template<int coeffIdx>
vint16m1_t inline filter8_s16x8(vint16m1_t **s, const vint16m1_t c, const size_t vl)
{
    vint16m1_t d0;
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        d0 = __riscv_vsub_vv_i16m1(*s[6], *s[0], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, 4, *s[1], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, -10, *s[2], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, 58, *s[3], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, 17, *s[4], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, -5, *s[5], vl);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        d0 = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t t0 = __riscv_vadd_vv_i16m1(*s[3], *s[4], vl);
        vint16m1_t t1 = __riscv_vadd_vv_i16m1(*s[2], *s[5], vl);
        vint16m1_t t2 = __riscv_vadd_vv_i16m1(*s[1], *s[6], vl);
        vint16m1_t t3 = __riscv_vadd_vv_i16m1(*s[0], *s[7], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, 40, t0, vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, -11, t1, vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, 4, t2, vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, -1, t3, vl);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        d0 = __riscv_vsub_vv_i16m1(*s[1], *s[7], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, -5, *s[2], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, 17, *s[3], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, 58, *s[4], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, -10, *s[5], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, 4, *s[6], vl);
    }
    return __riscv_vadd_vv_i16m1(d0, c, vl);
}

template<int coeffIdx>
vint16m2_t inline filter8_s16x16(vint16m2_t **s, const vint16m2_t c, size_t vl)
{
    vint16m2_t d0;
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        d0 = __riscv_vsub_vv_i16m2(*s[6], *s[0], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, 4, *s[1], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, -10, *s[2], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, 58, *s[3], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, 17, *s[4], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, -5, *s[5], vl);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        d0 = __riscv_vmv_v_x_i16m2(0, vl);
        vint16m2_t t0 = __riscv_vadd_vv_i16m2(*s[3], *s[4], vl);
        vint16m2_t t1 = __riscv_vadd_vv_i16m2(*s[2], *s[5], vl);
        vint16m2_t t2 = __riscv_vadd_vv_i16m2(*s[1], *s[6], vl);
        vint16m2_t t3 = __riscv_vadd_vv_i16m2(*s[0], *s[7], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, 40, t0, vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, -11, t1, vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, 4, t2, vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, -1, t3, vl);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        d0 = __riscv_vsub_vv_i16m2(*s[1], *s[7], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, -5, *s[2], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, 17, *s[3], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, 58, *s[4], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, -10, *s[5], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, 4, *s[6], vl);
    }
    return __riscv_vadd_vv_i16m2(d0, c, vl);
}

template<int coeffIdx>
vint32m2_t inline filter8_s32x8(vint32m2_t **s, const vint32m2_t c, size_t vl)
{
    vint32m2_t d0;
    if (coeffIdx == 1)
    {
        // { -1, 4, -10, 58, 17, -5, 1, 0 }
        d0 = __riscv_vsub_vv_i32m2(*s[6], *s[0], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, 4, *s[1], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, -10, *s[2], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, 58, *s[3], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, 17, *s[4], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, -5, *s[5], vl);
    }
    else if (coeffIdx == 2)
    {
        // { -1, 4, -11, 40, 40, -11, 4, -1 }
        d0 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t t0 = __riscv_vadd_vv_i32m2(*s[3], *s[4], vl);
        vint32m2_t t1 = __riscv_vadd_vv_i32m2(*s[2], *s[5], vl);
        vint32m2_t t2 = __riscv_vadd_vv_i32m2(*s[1], *s[6], vl);
        vint32m2_t t3 = __riscv_vadd_vv_i32m2(*s[0], *s[7], vl);

        d0 = __riscv_vmacc_vx_i32m2(d0, 40, t0, vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, -11, t1, vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, 4, t2, vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, -1, t3, vl);
    }
    else
    {
        // { 0, 1, -5, 17, 58, -10, 4, -1 }
        d0 = __riscv_vsub_vv_i32m2(*s[1], *s[7], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, -5, *s[2], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, 17, *s[3], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, 58, *s[4], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, -10, *s[5], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, 4, *s[6], vl);
    }
    return __riscv_vadd_vv_i32m2(d0, c, vl);
}

/* N_TAPS = 4 */
template<bool coeff4>
vint16m1_t inline filter4_s16x8(vint16m1_t **s, const vint16m1_t c, const int16_t* f, size_t vl)
{
    vint16m1_t d0 = __riscv_vmv_v_x_i16m1(0, vl);

    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        vint16m1_t t1 = __riscv_vadd_vv_i16m1(*s[1], *s[2], vl);
        vint16m1_t t2 = __riscv_vadd_vv_i16m1(*s[0], *s[3], vl);

        d0 = __riscv_vmacc_vx_i16m1(d0, 36, t1, vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, -4, t2, vl);
    }
    else
    {
        d0 = __riscv_vmacc_vx_i16m1(d0, f[0], *s[0], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, f[1], *s[1], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, f[2], *s[2], vl);
        d0 = __riscv_vmacc_vx_i16m1(d0, f[3], *s[3], vl);
    }
    return __riscv_vadd_vv_i16m1(d0, c, vl);
}

template<bool coeff4>
vint16m2_t inline filter4_s16x16(vint16m2_t **s, const vint16m2_t c, const int16_t* f, size_t vl)
{
    vint16m2_t d0 = __riscv_vmv_v_x_i16m2(0, vl);

    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        vint16m2_t t1 = __riscv_vadd_vv_i16m2(*s[1], *s[2], vl);
        vint16m2_t t2 = __riscv_vadd_vv_i16m2(*s[0], *s[3], vl);

        d0 = __riscv_vmacc_vx_i16m2(d0, 36, t1, vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, -4, t2, vl);
    }
    else
    {
        d0 = __riscv_vmacc_vx_i16m2(d0, f[0], *s[0], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, f[1], *s[1], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, f[2], *s[2], vl);
        d0 = __riscv_vmacc_vx_i16m2(d0, f[3], *s[3], vl);
    }
    return __riscv_vadd_vv_i16m2(d0, c, vl);
}

template<bool coeff4>
vint32m2_t inline filter4_s32x8(vint32m2_t **s, vint32m2_t c, const int16_t* f, size_t vl)
{
    vint32m2_t d0 = __riscv_vmv_v_x_i32m2(0, vl);

    if (coeff4)
    {
        // { -4, 36, 36, -4 }
        vint32m2_t t1 = __riscv_vadd_vv_i32m2(*s[1], *s[2], vl);
        vint32m2_t t2 = __riscv_vadd_vv_i32m2(*s[0], *s[3], vl);

        d0 = __riscv_vmacc_vx_i32m2(d0, 36, t1, vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, -4, t2, vl);
    }
    else
    {
        d0 = __riscv_vmacc_vx_i32m2(d0, f[0], *s[0], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, f[1], *s[1], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, f[2], *s[2], vl);
        d0 = __riscv_vmacc_vx_i32m2(d0, f[3], *s[3], vl);
    }
    return __riscv_vadd_vv_i32m2(d0, c, vl);
    return d0;
}

#endif //!HIGH_BIT_DEPTH

}; // namespace X265_NS
