/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Jia Yuan <yuan.jia@sanechips.com.cn>
 *          foolgry <wang.zhiyong11@sanechips.com.cn>
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
#include "slicetype.h" // LOWRES_COST_MASK
#include "primitives.h"
#include "x265.h"
#include "riscv64_utils.h"

#include <riscv_vector.h>
#include <stdint.h>

#define SUMSUB_AB(sum, diff, a, b, vl)                     \
    do {                                                   \
        vuint16m1_t _a = (a);                              \
        vuint16m1_t _b = (b);                              \
        (sum) = __riscv_vadd_vv_u16m1(_a, _b, (vl));       \
        (diff) = __riscv_vsub_vv_u16m1(_a, _b, (vl));      \
    } while (0)

namespace {

using namespace X265_NS;

#if HIGH_BIT_DEPTH
// todo

#else // !HIGH_BIT_DEPTH

typedef struct {
    vuint8mf2_t *m0;
    vuint8mf2_t *m1;
    vuint8mf2_t *m2;
    vuint8mf2_t *m3;
} vectors_u8_mf2_t;

typedef struct {
    vuint8mf2_t *m0;
    vuint8mf2_t *m1;
    vuint8mf2_t *m2;
    vuint8mf2_t *m3;
    vuint8mf2_t *m4;
    vuint8mf2_t *m5;
    vuint8mf2_t *m6;
    vuint8mf2_t *m7;
} vectors_u8_mf2_8_t;

typedef struct {
    vuint8m1_t *m0;
    vuint8m1_t *m1;
    vuint8m1_t *m2;
    vuint8m1_t *m3;
} vectors_u8_m1_t;

typedef struct {
    vuint16m1_t *m0;
    vuint16m1_t *m1;
    vuint16m1_t *m2;
    vuint16m1_t *m3;
} vectors_u16_m1_t;

typedef struct {
    vuint16m1_t *m0;
    vuint16m1_t *m1;
    vuint16m1_t *m2;
    vuint16m1_t *m3;
    vuint16m1_t *m4;
    vuint16m1_t *m5;
    vuint16m1_t *m6;
    vuint16m1_t *m7;
} vectors_u16_m1_8_t;

// 4 times vle8 performs better than one time vlsseg4e8.
// 1. array elements cannot have RVV type, such as 'vuint8mf2_t';
// 2. The member variables in the structure cannot be RVV type, so use the structure pointer version.
static void inline vload_u8x8x4_mf2(const uint8_t **pix, const intptr_t stride_pix,
                               vectors_u8_mf2_t *d, size_t vl)
{
    *(d->m0) = __riscv_vle8_v_u8mf2(*pix, vl);
    *(d->m1) = __riscv_vle8_v_u8mf2(*pix + stride_pix, vl);
    *(d->m2) = __riscv_vle8_v_u8mf2(*pix + 2 * stride_pix, vl);
    *(d->m3) = __riscv_vle8_v_u8mf2(*pix + 3 * stride_pix, vl);
}

static inline void load_diff_u8x8x8(const uint8_t *pix1, intptr_t stride_pix1,
                                    const uint8_t *pix2, intptr_t stride_pix2, vectors_u16_m1_8_t *diff)
{
    const uint8_t *pix1_ptr = pix1;
    const uint8_t *pix2_ptr = pix2;

    size_t vl = __riscv_vsetvl_e8mf2(8);
    vuint8mf2_t r0, r1, r2, r3, t0, t1, t2, t3;
    vectors_u8_mf2_t r = {&r0, &r1, &r2, &r3};
    vectors_u8_mf2_t t = {&t0, &t1, &t2, &t3};

    //row 0~3
    vload_u8x8x4_mf2(&pix1_ptr, stride_pix1, &r, vl);
    vload_u8x8x4_mf2(&pix2_ptr, stride_pix2, &t, vl);
    *(diff->m0) = __riscv_vwsubu_vv_u16m1(*(r.m0), *(t.m0), vl);
    *(diff->m1) = __riscv_vwsubu_vv_u16m1(*(r.m1), *(t.m1), vl);
    *(diff->m2) = __riscv_vwsubu_vv_u16m1(*(r.m2), *(t.m2), vl);
    *(diff->m3) = __riscv_vwsubu_vv_u16m1(*(r.m3), *(t.m3), vl);
    //row4~7

    pix1_ptr += 4 * stride_pix1;
    pix2_ptr += 4 * stride_pix2;
    vload_u8x8x4_mf2(&pix1_ptr, stride_pix1, &r, vl);
    vload_u8x8x4_mf2(&pix2_ptr, stride_pix2, &t, vl);
    *(diff->m4) = __riscv_vwsubu_vv_u16m1(*(r.m0), *(t.m0), vl);
    *(diff->m5) = __riscv_vwsubu_vv_u16m1(*(r.m1), *(t.m1), vl);
    *(diff->m6) = __riscv_vwsubu_vv_u16m1(*(r.m2), *(t.m2), vl);
    *(diff->m7) = __riscv_vwsubu_vv_u16m1(*(r.m3), *(t.m3), vl);
}

static inline void vload_u8mf2x4(vectors_u8_mf2_t *d, const uint8_t **pix, intptr_t stride_pix, size_t vl) {
    *(d->m0) = __riscv_vle8_v_u8mf2(*pix, vl);
    *(d->m1) = __riscv_vle8_v_u8mf2(*pix + stride_pix, vl);
    *(d->m2) = __riscv_vle8_v_u8mf2(*pix + 2 * stride_pix, vl);
    *(d->m3) = __riscv_vle8_v_u8mf2(*pix + 3 * stride_pix, vl);
    *pix += 4 * stride_pix;
}

static inline void vload_u8m1x4(vectors_u8_m1_t *d, const uint8_t **pix, intptr_t stride_pix, size_t vl) {
    *(d->m0) = __riscv_vle8_v_u8m1(*pix, vl);
    *(d->m1) = __riscv_vle8_v_u8m1(*pix + stride_pix, vl);
    *(d->m2) = __riscv_vle8_v_u8m1(*pix + 2 * stride_pix, vl);
    *(d->m3) = __riscv_vle8_v_u8m1(*pix + 3 * stride_pix, vl);
    *pix += 4 * stride_pix;
}

static inline void vslide_combine_u8(vuint8mf2_t *d0, vuint8mf2_t *d1, vectors_u8_mf2_t s) {
    *d0 = __riscv_vslideup_vx_u8mf2(*(s.m0), *(s.m2), 4, 8);
    *d1 = __riscv_vslideup_vx_u8mf2(*(s.m1), *(s.m3), 4, 8);
}

static inline void vslidedown_u8x4(vectors_u8_m1_t *d0, vectors_u8_m1_t *d1,
                                   const vectors_u8_m1_t s0, const vectors_u8_m1_t s1, size_t vl) {
    const size_t offset = 8;
    *(d0->m0) = __riscv_vslidedown_vx_u8m1(*(s0.m0), offset, vl);
    *(d0->m1) = __riscv_vslidedown_vx_u8m1(*(s0.m1), offset, vl);
    *(d0->m2) = __riscv_vslidedown_vx_u8m1(*(s0.m2), offset, vl);
    *(d0->m3) = __riscv_vslidedown_vx_u8m1(*(s0.m3), offset, vl);

    *(d1->m0) = __riscv_vslidedown_vx_u8m1(*(s1.m0), offset, vl);
    *(d1->m1) = __riscv_vslidedown_vx_u8m1(*(s1.m1), offset, vl);
    *(d1->m2) = __riscv_vslidedown_vx_u8m1(*(s1.m2), offset, vl);
    *(d1->m3) = __riscv_vslidedown_vx_u8m1(*(s1.m3), offset, vl);
}

static inline void vwsubu_u8x4(vectors_u16_m1_t *diff, const vectors_u8_mf2_t s0, const vectors_u8_mf2_t s1, size_t vl) {
    *(diff->m0) = __riscv_vwsubu_vv_u16m1(*(s0.m0), *(s1.m0), vl);
    *(diff->m1) = __riscv_vwsubu_vv_u16m1(*(s0.m1), *(s1.m1), vl);
    *(diff->m2) = __riscv_vwsubu_vv_u16m1(*(s0.m2), *(s1.m2), vl);
    *(diff->m3) = __riscv_vwsubu_vv_u16m1(*(s0.m3), *(s1.m3), vl);
}

static inline void vget_first8_u8m1(vectors_u8_mf2_t *d, const vectors_u8_m1_t s) {
    *(d->m0) = __riscv_vlmul_trunc_v_u8m1_u8mf2(*(s.m0));
    *(d->m1) = __riscv_vlmul_trunc_v_u8m1_u8mf2(*(s.m1));
    *(d->m2) = __riscv_vlmul_trunc_v_u8m1_u8mf2(*(s.m2));
    *(d->m3) = __riscv_vlmul_trunc_v_u8m1_u8mf2(*(s.m3));
}

static inline vint16m1_t vabs_u16(vuint16m1_t s0, size_t vl) {
    vint16m1_t tmp = __riscv_vreinterpret_v_u16m1_i16m1(s0);
    vint16m1_t t0 = __riscv_vrsub_vx_i16m1(tmp, 0, vl);  // t0 = -s0
    return __riscv_vmax_vv_i16m1(tmp, t0, vl);           // d0 = max(s0, t0)
}

static inline vuint16m1_t vmax_abs_u16(vuint16m1_t s0, vuint16m1_t s1, size_t vl) {
    vuint16m1_t b0 = __riscv_vreinterpret_v_i16m1_u16m1(vabs_u16(s0, vl));
    vuint16m1_t b1 = __riscv_vreinterpret_v_i16m1_u16m1(vabs_u16(s1, vl));
    return __riscv_vmaxu_vv_u16m1(b0, b1, vl);
}

static inline int vredsum_u16(const vuint16m1_t src, size_t vl) {
    vuint16m1_t v_sum = __riscv_vmv_v_x_u16m1(0, vl);
    v_sum = __riscv_vredsum_vs_u16m1_u16m1(src, v_sum, vl);
    return __riscv_vmv_x_s_u16m1_u16(v_sum);
}

static inline int vredsum_u32(const vuint32m2_t src, size_t vl) {
    vuint32m1_t v_sum = __riscv_vmv_v_x_u32m1(0, 4);
    v_sum = __riscv_vredsum_vs_u32m2_u32m1(src, v_sum, vl);
    return __riscv_vmv_x_s_u32m1_u32(v_sum);
}

static inline void vtrn_8h(vuint16m1_t *d0, vuint16m1_t *d1, vuint16m1_t s0, vuint16m1_t s1) {
    size_t vl = __riscv_vsetvl_e32m1(4);
    const size_t shift = 16;

    vuint32m1_t s0_32 = __riscv_vreinterpret_v_u16m1_u32m1(s0);
    vuint32m1_t s1_32 = __riscv_vreinterpret_v_u16m1_u32m1(s1);

    vuint32m1_t t2 = __riscv_vsll_vx_u32m1(s0_32, shift, vl);
    vuint32m1_t t0 = __riscv_vsrl_vx_u32m1(s0_32, shift, vl);

    vuint32m1_t d0_32 = __riscv_vsll_vx_u32m1(s1_32, shift, vl);
    vuint32m1_t t1 = __riscv_vsrl_vx_u32m1(s1_32, shift, vl);

    vuint32m1_t d1_32 = __riscv_vsll_vx_u32m1(t1, shift, vl);
    t2 = __riscv_vsrl_vx_u32m1(t2, shift, vl);

    vl = __riscv_vsetvl_e16m1(8);
    *d0 = __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vor_vv_u32m1(d0_32, t2, vl / 2));
    *d1 = __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vor_vv_u32m1(d1_32, t0, vl / 2));
}

static inline void vtrn_4s(vuint16m1_t *d0, vuint16m1_t *d1, vuint16m1_t s0, vuint16m1_t s1) {
    size_t vl = __riscv_vsetvl_e64m1(2);
    const size_t shift = 32;

    vuint64m1_t s0_64 = __riscv_vreinterpret_v_u16m1_u64m1(s0);
    vuint64m1_t s1_64 = __riscv_vreinterpret_v_u16m1_u64m1(s1);

    vuint64m1_t t2 = __riscv_vsll_vx_u64m1(s0_64, shift, vl);
    vuint64m1_t t0 =  __riscv_vsrl_vx_u64m1(s0_64, shift, vl);

    vuint64m1_t d0_64 = __riscv_vsll_vx_u64m1(s1_64, shift, vl);
    vuint64m1_t t1 =  __riscv_vsrl_vx_u64m1(s1_64, shift, vl);

    vuint64m1_t d1_64 = __riscv_vsll_vx_u64m1(t1, shift, vl);
    t2 = __riscv_vsrl_vx_u64m1(t2, shift, vl);

    vl = __riscv_vsetvl_e32m1(4);
    *d0 = __riscv_vreinterpret_v_u64m1_u16m1(__riscv_vor_vv_u64m1(d0_64, t2, vl / 2));
    *d1 = __riscv_vreinterpret_v_u64m1_u16m1(__riscv_vor_vv_u64m1(d1_64, t0, vl / 2));
}

static inline void vtrn_16s(vuint16m1_t *d0, vuint16m1_t *d1, vuint16m1_t s0, vuint16m1_t s1) {
    size_t vl = __riscv_vsetvl_e16m1(8);
    const size_t offset = 4;
    vuint16m1_t v1_slide = s1;

    *d0 = __riscv_vslideup_vx_u16m1(s0, s1, offset, vl);

    vl = __riscv_vsetvl_e16m1(4);
    *d1 = __riscv_vslidedown_vx_u16m1_tu(v1_slide, s0, offset, vl);
}

// 4 way hadamard vertical pass.
static inline void hadamard_vx4(vectors_u16_m1_t *out, const vectors_u16_m1_t *in, size_t vl) {
    vuint16m1_t s0, s1, d0, d1;

    SUMSUB_AB(s0, d0, *(in->m0), *(in->m1), vl);
    SUMSUB_AB(s1, d1, *(in->m2), *(in->m3), vl);

    SUMSUB_AB(*(out->m0), *(out->m2), s0, s1, vl);
    SUMSUB_AB(*(out->m1), *(out->m3), d0, d1, vl);
}

// 8 way hadamard vertical pass.
static inline void hadamard_vx8(vectors_u16_m1_8_t *out, const vectors_u16_m1_8_t *in, size_t vl)
{
    vuint16m1_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    vectors_u16_m1_t t0={&tmp0, &tmp1, &tmp2, &tmp3};
    vectors_u16_m1_t t1={&tmp4, &tmp5, &tmp6, &tmp7};
    vectors_u16_m1_t in0={in->m0, in->m1, in->m2, in->m3};
    vectors_u16_m1_t in1={in->m4, in->m5, in->m6, in->m7};

    hadamard_vx4(&t0, &in0 ,vl);
    hadamard_vx4(&t1, &in1, vl);

    SUMSUB_AB(*(out->m0), *(out->m4), *(t0.m0), *(t1.m0), vl);
    SUMSUB_AB(*(out->m1), *(out->m5), *(t0.m1), *(t1.m1), vl);
    SUMSUB_AB(*(out->m2), *(out->m6), *(t0.m2), *(t1.m2), vl);
    SUMSUB_AB(*(out->m3), *(out->m7), *(t0.m3), *(t1.m3), vl);
}

// 4 way hadamard horizontal pass.
static inline void hadamard_hx4(vectors_u16_m1_t *out, const vectors_u16_m1_t *in, size_t vl) {
    vuint16m1_t s0, s1, t0, t1, t2, t3, d0, d1;

    vtrn_8h(&t0, &t1, *(in->m0), *(in->m1));
    vtrn_8h(&t2, &t3, *(in->m2), *(in->m3));

    SUMSUB_AB(s0, d0, t0, t1, vl);
    SUMSUB_AB(s1, d1, t2, t3, vl);

    vtrn_4s(out->m0, out->m1, s0, s1);
    vtrn_4s(out->m2, out->m3, d0, d1);
}

// 8 way hadamard horizontal pass.
static inline void hadamard_hx8(vectors_u16_m1_t *out, const vectors_u16_m1_8_t *in, size_t vl)
{
    vuint16m1_t s0, s1, s2, s3, d0, d1, d2, d3;
    vuint16m1_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    vectors_u16_m1_t t0={&tmp0, &tmp1, &tmp2, &tmp3};
    vectors_u16_m1_t t1={&tmp4, &tmp5, &tmp6, &tmp7};
    vectors_u16_m1_t in0={in->m0, in->m1, in->m2, in->m3};
    vectors_u16_m1_t in1={in->m4, in->m5, in->m6, in->m7};

    hadamard_hx4(&t0, &in0 ,vl);
    hadamard_hx4(&t1, &in1 ,vl);

    SUMSUB_AB(s0, d0, *(t0.m0), *(t0.m1), vl);
    SUMSUB_AB(s1, d1, *(t0.m2), *(t0.m3), vl);
    SUMSUB_AB(s2, d2, *(t1.m0), *(t1.m1), vl);
    SUMSUB_AB(s3, d3, *(t1.m2), *(t1.m3), vl);

    vtrn_16s((t0.m0), (t0.m1), s0, s2);
    vtrn_16s((t0.m2), (t0.m3), s1, s3);
    vtrn_16s((t1.m0), (t1.m1), d0, d2);
    vtrn_16s((t1.m2), (t1.m3), d1, d3);

    *(out->m0) = vmax_abs_u16(*(t0.m0), *(t0.m1), vl);
    *(out->m1) = vmax_abs_u16(*(t0.m2), *(t0.m3), vl);
    *(out->m2) = vmax_abs_u16(*(t1.m0), *(t1.m1), vl);
    *(out->m3) = vmax_abs_u16(*(t1.m2), *(t1.m3), vl);
}

// Calculate 2 4x4 hadamard transformation.
static inline void hadamard_4x4x2(vuint16m1_t *out, vectors_u16_m1_t *s, size_t vl) {
    vuint16m1_t tmp0, tmp1, tmp2, tmp3;
    vectors_u16_m1_t tmp = {&tmp0, &tmp1, &tmp2, &tmp3};

    hadamard_vx4(&tmp, s, vl);
    hadamard_hx4(s, &tmp, vl);

    vuint16m1_t b0 = vmax_abs_u16(*(s->m0), *(s->m1), vl);
    vuint16m1_t b1 = vmax_abs_u16(*(s->m2), *(s->m3), vl);
    *out = __riscv_vadd_vv_u16m1(b0, b1, vl);
}

// Calculate 4 4x4 hadamard transformation.
static inline void hadamard_4x4x4(vuint16m1_t *out0, vuint16m1_t *out1, vectors_u16_m1_t *s0,
                                  vectors_u16_m1_t *s1, size_t vl) {
    vuint16m1_t tmp0_0, tmp0_1, tmp0_2, tmp0_3, tmp1_0, tmp1_1, tmp1_2, tmp1_3;
    vectors_u16_m1_t tmp0 = {&tmp0_0, &tmp0_1, &tmp0_2, &tmp0_3};
    vectors_u16_m1_t tmp1 = {&tmp1_0, &tmp1_1, &tmp1_2, &tmp1_3};
    hadamard_vx4(&tmp0, s0, vl);
    hadamard_vx4(&tmp1, s1, vl);

    hadamard_hx4(s0, &tmp0, vl);
    hadamard_hx4(s1, &tmp1, vl);

    vuint16m1_t b0 = vmax_abs_u16(*(s0->m0), *(s0->m1), vl);
    vuint16m1_t b1 = vmax_abs_u16(*(s0->m2), *(s0->m3), vl);
    vuint16m1_t b2 = vmax_abs_u16(*(s1->m0), *(s1->m1), vl);
    vuint16m1_t b3 = vmax_abs_u16(*(s1->m2), *(s1->m3), vl);
    *out0 = __riscv_vadd_vv_u16m1(b0, b1, vl);
    *out1 = __riscv_vadd_vv_u16m1(b2, b3, vl);
}

// Calculate 8x8 hadamard transformation.
static inline void hadamard_8x8(vuint16m1_t *out0, vuint16m1_t *out1, vectors_u16_m1_8_t *diff, size_t vl)
{
    vuint16m1_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    vectors_u16_m1_8_t tmp={&tmp0, &tmp1, &tmp2, &tmp3, &tmp4, &tmp5, &tmp6, &tmp7};
    vuint16m1_t sum0, sum1, sum2, sum3;
    vectors_u16_m1_t sum={&sum0, &sum1, &sum2, &sum3};

    hadamard_vx8(&tmp, diff, vl);
    hadamard_hx8(&sum, &tmp, vl);

    *out0 = __riscv_vadd_vv_u16m1(*(sum.m0), *(sum.m1), vl);
    *out1 = __riscv_vadd_vv_u16m1(*(sum.m2), *(sum.m3), vl);
}

static inline int pixel_satd_4x4_rvv(const uint8_t *pix1, intptr_t stride_pix1,
                                     const uint8_t *pix2, intptr_t stride_pix2) {
    // Load 4x4 blocks
    size_t vl = __riscv_vsetvl_e8mf2(4);
    const uint8_t *pix1_ptr = pix1;
    const uint8_t *pix2_ptr = pix2;
    vuint8mf2_t s0, s1, s2, s3, r0, r1, r2, r3;
    vectors_u8_mf2_t s = {&s0, &s1, &s2, &s3};
    vectors_u8_mf2_t r = {&r0, &r1, &r2, &r3};
    vload_u8mf2x4(&s, &pix1_ptr, stride_pix1, vl);
    vload_u8mf2x4(&r, &pix2_ptr, stride_pix2, vl);

    // Slide upper parts
    vuint8mf2_t s_combined0, s_combined1, r_combined0, r_combined1;
    vslide_combine_u8(&s_combined0, &s_combined1, s);
    vslide_combine_u8(&r_combined0, &r_combined1, r);

    // Convert to 16-bit and compute differences (sign extension, not zero extension)
    vl = __riscv_vsetvl_e8mf2(8);
    vuint16m1_t diff0 = __riscv_vwsubu_vv_u16m1(s_combined0, r_combined0, vl);
    vuint16m1_t diff1 = __riscv_vwsubu_vv_u16m1(s_combined1, r_combined1, vl);

    vuint16m1_t v0, v1, v2, v3;
    vl = __riscv_vsetvl_e16m1(8);
    SUMSUB_AB(v2, v3, diff0, diff1, vl);

    // Slide and combine
    vuint16m1_t v0_slide = __riscv_vslideup_vx_u16m1(v2, v3, 4, vl);
    vuint16m1_t v1_slide = v3;
    vl = __riscv_vsetvl_e16m1(4);
    //v1_slide = __riscv_vslidedown_vx_u16m1(v2, 4, vl);
    // Protect the tail of the destination register
    v1_slide = __riscv_vslidedown_vx_u16m1_tu(v1_slide, v2, 4, vl);
    vl = __riscv_vsetvl_e16m1(8);
    SUMSUB_AB(v2, v3, v0_slide, v1_slide, vl);

    vtrn_8h(&v0, &v1, v2, v3);
    SUMSUB_AB(v2, v3, v0, v1, vl);

    vtrn_4s(&v0, &v1, v2, v3);

    // Absolute values and reduction
    vl = __riscv_vsetvl_e16m1(8);
    vuint16m1_t vmax = vmax_abs_u16(v0, v1, vl);
    int sum = vredsum_u16(vmax, vl);
    return sum;
}

static inline int pixel_satd_4x8_rvv(const uint8_t *pix1, intptr_t stride_pix1,
                                     const uint8_t *pix2, intptr_t stride_pix2) {
    // Load 4x8 blocks
    size_t vl = __riscv_vsetvl_e8mf2(4);
    const uint8_t *pix1_ptr = pix1;
    const uint8_t *pix2_ptr = pix2;

    // row: 0-3
    vuint8mf2_t s0_0, s0_1, s0_2, s0_3;
    vuint8mf2_t r0_0, r0_1, r0_2, r0_3;
    vectors_u8_mf2_t s0 = {&s0_0, &s0_1, &s0_2, &s0_3};
    vectors_u8_mf2_t r0 = {&r0_0, &r0_1, &r0_2, &r0_3};
    vload_u8mf2x4(&s0, &pix1_ptr, stride_pix1, vl);
    vload_u8mf2x4(&r0, &pix2_ptr, stride_pix2, vl);

    // row: 4-7
    vuint8mf2_t s1_0, s1_1, s1_2, s1_3;
    vuint8mf2_t r1_0, r1_1, r1_2, r1_3;
    vectors_u8_mf2_t s1 = {&s1_0, &s1_1, &s1_2, &s1_3};
    vectors_u8_mf2_t r1 = {&r1_0, &r1_1, &r1_2, &r1_3};
    vload_u8mf2x4(&s1, &pix1_ptr, stride_pix1, vl);
    vload_u8mf2x4(&r1, &pix2_ptr, stride_pix2, vl);

    // Slide upper parts
    vuint8mf2_t s_combined0, s_combined1, s_combined2, s_combined3;
    vuint8mf2_t r_combined0, r_combined1, r_combined2, r_combined3;
    vectors_u8_mf2_t tmp_s0 = {s0.m0, s0.m1, s1.m0, s1.m1};
    vectors_u8_mf2_t tmp_s1 = {s0.m2, s0.m3, s1.m2, s1.m3};
    vectors_u8_mf2_t tmp_r0 = {r0.m0, r0.m1, r1.m0, r1.m1};
    vectors_u8_mf2_t tmp_r1 = {r0.m2, r0.m3, r1.m2, r1.m3};
    vslide_combine_u8(&s_combined0, &s_combined1, tmp_s0);
    vslide_combine_u8(&s_combined2, &s_combined3, tmp_s1);
    vslide_combine_u8(&r_combined0, &r_combined1, tmp_r0);
    vslide_combine_u8(&r_combined2, &r_combined3, tmp_r1);

    // Convert to 16-bit and compute differences
    vectors_u8_mf2_t tmp_s_combined = {&s_combined0, &s_combined1, &s_combined2, &s_combined3};
    vectors_u8_mf2_t tmp_r_combined = {&r_combined0, &r_combined1, &r_combined2, &r_combined3};
    vuint16m1_t diff0, diff1, diff2, diff3, out;
    vectors_u16_m1_t diff = {&diff0, &diff1, &diff2, &diff3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff, tmp_s_combined, tmp_r_combined, vl);

    vl = __riscv_vsetvl_e16m1(8);
    hadamard_4x4x2(&out, &diff, vl);
    int sum = vredsum_u16(out, vl);

    return sum;
}

static inline int pixel_satd_8x4_rvv(const uint8_t *pix1, intptr_t stride_pix1,
                                     const uint8_t *pix2, intptr_t stride_pix2) {
    // Load 8x4 blocks, convert to 16-bit and compute differences
    size_t vl = __riscv_vsetvl_e8mf2(8);
    const uint8_t *pix1_ptr = pix1;
    const uint8_t *pix2_ptr = pix2;
    vuint8mf2_t s0, s1, s2, s3, r0, r1, r2, r3;
    vectors_u8_mf2_t s = {&s0, &s1, &s2, &s3};
    vectors_u8_mf2_t r = {&r0, &r1, &r2, &r3};
    vload_u8mf2x4(&s, &pix1_ptr, stride_pix1, vl);
    vload_u8mf2x4(&r, &pix2_ptr, stride_pix2, vl);
    vuint16m1_t diff0, diff1, diff2, diff3, out;
    vectors_u16_m1_t diff = {&diff0, &diff1, &diff2, &diff3};
    vwsubu_u8x4(&diff, s, r, vl);

    vl = __riscv_vsetvl_e16m1(8);
    hadamard_4x4x2(&out, &diff, vl);
    int sum = vredsum_u16(out, vl);

    return sum;
}

static inline int pixel_satd_8x8_rvv(const uint8_t *pix1, intptr_t stride_pix1,
                                     const uint8_t *pix2, intptr_t stride_pix2) {
    // Load 8x8 blocks; convert to 16-bit and compute differences
    size_t vl = __riscv_vsetvl_e8mf2(8);
    const uint8_t *pix1_ptr = pix1;
    const uint8_t *pix2_ptr = pix2;
    // row: 0-3
    vuint8mf2_t s0_0, s0_1, s0_2, s0_3;
    vuint8mf2_t r0_0, r0_1, r0_2, r0_3;
    vuint16m1_t diff0_0, diff0_1, diff0_2, diff0_3;
    vectors_u8_mf2_t s0 = {&s0_0, &s0_1, &s0_2, &s0_3};
    vectors_u8_mf2_t r0 = {&r0_0, &r0_1, &r0_2, &r0_3};
    vectors_u16_m1_t diff0 = {&diff0_0, &diff0_1, &diff0_2, &diff0_3};
    vload_u8mf2x4(&s0, &pix1_ptr, stride_pix1, vl);
    vload_u8mf2x4(&r0, &pix2_ptr, stride_pix2, vl);
    vwsubu_u8x4(&diff0, s0, r0, vl);

    // row: 4-7
    vuint8mf2_t s1_0, s1_1, s1_2, s1_3;
    vuint8mf2_t r1_0, r1_1, r1_2, r1_3;
    vectors_u8_mf2_t s1 = {&s1_0, &s1_1, &s1_2, &s1_3};
    vectors_u8_mf2_t r1 = {&r1_0, &r1_1, &r1_2, &r1_3};
    vuint16m1_t diff1_0, diff1_1, diff1_2, diff1_3;
    vectors_u16_m1_t diff1 = {&diff1_0, &diff1_1, &diff1_2, &diff1_3};
    vload_u8mf2x4(&s1, &pix1_ptr, stride_pix1, vl);
    vload_u8mf2x4(&r1, &pix2_ptr, stride_pix2, vl);
    vwsubu_u8x4(&diff1, s1, r1, vl);

    vuint16m1_t out0, out1;
    vl = __riscv_vsetvl_e16m1(8);
    hadamard_4x4x4(&out0, &out1, &diff0, &diff1, vl);

    // Absolute values and reduction
    out0 = __riscv_vadd_vv_u16m1(out0, out1, vl);
    int sum = vredsum_u16(out0, vl);

    return sum;
}

static inline int pixel_satd_8x16_rvv(const uint8_t *pix1, intptr_t stride_pix1,
                                      const uint8_t *pix2, intptr_t stride_pix2) {
    // Load 8x16 blocks
    size_t vl = __riscv_vsetvl_e8mf2(8);
    const uint8_t *pix1_ptr = pix1;
    const uint8_t *pix2_ptr = pix2;
    // row: 0-3
    vuint8mf2_t s0_0, s0_1, s0_2, s0_3;
    vuint8mf2_t r0_0, r0_1, r0_2, r0_3;
    vuint16m1_t diff0_0, diff0_1, diff0_2, diff0_3;
    vectors_u8_mf2_t s0 = {&s0_0, &s0_1, &s0_2, &s0_3};
    vectors_u8_mf2_t r0 = {&r0_0, &r0_1, &r0_2, &r0_3};
    vectors_u16_m1_t diff0 = {&diff0_0, &diff0_1, &diff0_2, &diff0_3};
    vload_u8mf2x4(&s0, &pix1_ptr, stride_pix1, vl);
    vload_u8mf2x4(&r0, &pix2_ptr, stride_pix2, vl);
    vwsubu_u8x4(&diff0, s0, r0, vl);
    // row: 4-7
    vuint8mf2_t s1_0, s1_1, s1_2, s1_3;
    vuint8mf2_t r1_0, r1_1, r1_2, r1_3;
    vectors_u8_mf2_t s1 = {&s1_0, &s1_1, &s1_2, &s1_3};
    vectors_u8_mf2_t r1 = {&r1_0, &r1_1, &r1_2, &r1_3};
    vuint16m1_t diff1_0, diff1_1, diff1_2, diff1_3;
    vectors_u16_m1_t diff1 = {&diff1_0, &diff1_1, &diff1_2, &diff1_3};
    vload_u8mf2x4(&s1, &pix1_ptr, stride_pix1, vl);
    vload_u8mf2x4(&r1, &pix2_ptr, stride_pix2, vl);
    vwsubu_u8x4(&diff1, s1, r1, vl);

    vuint16m1_t out0, out1;
    vl = __riscv_vsetvl_e16m1(8);
    hadamard_4x4x4(&out0, &out1, &diff0, &diff1, vl);

    // row: 8-11
    vl = __riscv_vsetvl_e8mf2(8);
    vuint8mf2_t s2_0, s2_1, s2_2, s2_3;
    vuint8mf2_t r2_0, r2_1, r2_2, r2_3;
    vuint16m1_t diff2_0, diff2_1, diff2_2, diff2_3;
    vectors_u8_mf2_t s2 = {&s2_0, &s2_1, &s2_2, &s2_3};
    vectors_u8_mf2_t r2 = {&r2_0, &r2_1, &r2_2, &r2_3};
    vectors_u16_m1_t diff2 = {&diff2_0, &diff2_1, &diff2_2, &diff2_3};
    vload_u8mf2x4(&s2, &pix1_ptr, stride_pix1, vl);
    vload_u8mf2x4(&r2, &pix2_ptr, stride_pix2, vl);
    vwsubu_u8x4(&diff2, s2, r2, vl);
    // row: 12-15
    vuint8mf2_t s3_0, s3_1, s3_2, s3_3;
    vuint8mf2_t r3_0, r3_1, r3_2, r3_3;
    vectors_u8_mf2_t s3 = {&s3_0, &s3_1, &s3_2, &s3_3};
    vectors_u8_mf2_t r3 = {&r3_0, &r3_1, &r3_2, &r3_3};
    vuint16m1_t diff3_0, diff3_1, diff3_2, diff3_3;
    vectors_u16_m1_t diff3 = {&diff3_0, &diff3_1, &diff3_2, &diff3_3};
    vload_u8mf2x4(&s3, &pix1_ptr, stride_pix1, vl);
    vload_u8mf2x4(&r3, &pix2_ptr, stride_pix2, vl);
    vwsubu_u8x4(&diff3, s3, r3, vl);

    vuint16m1_t out2, out3;
    vl = __riscv_vsetvl_e16m1(8);
    hadamard_4x4x4(&out2, &out3, &diff2, &diff3, vl);

    vl = __riscv_vsetvl_e16m1(8);
    vuint16m1_t sum0 = __riscv_vadd_vv_u16m1(out0, out1, vl);
    vuint16m1_t sum1 = __riscv_vadd_vv_u16m1(out2, out3, vl);
    vuint16m1_t sum2 = __riscv_vadd_vv_u16m1(sum0, sum1, vl);
    int sum = vredsum_u16(sum2, vl);

    return sum;
}

// To be optimized
static inline int pixel_satd_16x4_rvv(const uint8_t *pix1, intptr_t stride_pix1,
                                      const uint8_t *pix2, intptr_t stride_pix2) {
/*
|       8 element     |    |       8 element      |
|       diff0[0]      |    |       diff1[0]       |
|       diff0[1]      |    |       diff1[1]       |
|       diff0[2]      |    |       diff1[2]       |
|       diff0[3]      |    |       diff1[3]       |
*/
    size_t vl = __riscv_vsetvl_e8m1(16);
    const uint8_t *pix1_ptr = pix1;
    const uint8_t *pix2_ptr = pix2;
    vuint8m1_t s0_0, s0_1, s0_2, s0_3;
    vuint8m1_t r0_0, r0_1, r0_2, r0_3;
    vectors_u8_m1_t s0 = {&s0_0, &s0_1, &s0_2, &s0_3};
    vectors_u8_m1_t r0 = {&r0_0, &r0_1, &r0_2, &r0_3};
    vload_u8m1x4(&s0, &pix1_ptr, stride_pix1, vl);
    vload_u8m1x4(&r0, &pix2_ptr, stride_pix2, vl);

    // Low part
    vuint8mf2_t s0_h_0, s0_h_1, s0_h_2, s0_h_3;
    vuint8mf2_t r0_h_0, r0_h_1, r0_h_2, r0_h_3;
    vectors_u8_mf2_t s0_half = {&s0_h_0, &s0_h_1, &s0_h_2, &s0_h_3};
    vectors_u8_mf2_t r0_half = {&r0_h_0, &r0_h_1, &r0_h_2, &r0_h_3};
    vget_first8_u8m1(&s0_half, s0);
    vget_first8_u8m1(&r0_half, r0);
    vuint16m1_t diff0_0, diff0_1, diff0_2, diff0_3;
    vectors_u16_m1_t diff0 = {&diff0_0, &diff0_1, &diff0_2, &diff0_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff0, s0_half, r0_half, vl);

    // High part
    vl = __riscv_vsetvl_e8m1(16);
    vuint8m1_t s1_0, s1_1, s1_2, s1_3;
    vuint8m1_t r1_0, r1_1, r1_2, r1_3;
    vectors_u8_m1_t s1 = {&s1_0, &s1_1, &s1_2, &s1_3};
    vectors_u8_m1_t r1 = {&r1_0, &r1_1, &r1_2, &r1_3};
    vslidedown_u8x4(&s1, &r1, s0, r0, vl);
    vuint8mf2_t s1_h_0, s1_h_1, s1_h_2, s1_h_3;
    vuint8mf2_t r1_h_0, r1_h_1, r1_h_2, r1_h_3;
    vectors_u8_mf2_t s1_half = {&s1_h_0, &s1_h_1, &s1_h_2, &s1_h_3};
    vectors_u8_mf2_t r1_half = {&r1_h_0, &r1_h_1, &r1_h_2, &r1_h_3};
    vget_first8_u8m1(&s1_half, s1);
    vget_first8_u8m1(&r1_half, r1);
    vuint16m1_t diff1_0, diff1_1, diff1_2, diff1_3;
    vectors_u16_m1_t diff1 = {&diff1_0, &diff1_1, &diff1_2, &diff1_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff1, s1_half, r1_half, vl);

    vuint16m1_t out0, out1;
    vl = __riscv_vsetvl_e16m1(8);
    hadamard_4x4x2(&out0, &diff0, vl);
    hadamard_4x4x2(&out1, &diff1, vl);

    out0 = __riscv_vadd_vv_u16m1(out0, out1, vl);
    int sum = vredsum_u16(out0, vl);

    return sum;
}

static inline int pixel_satd_16x8_rvv(const uint8_t *pix1, intptr_t stride_pix1,
                                       const uint8_t *pix2, intptr_t stride_pix2) {
/*
|       8 element     |    |       8 element      |
|       diff0[0]      |    |       diff1[0]       |
|       diff0[1]      |    |       diff1[1]       |
|       diff0[2]      |    |       diff1[2]       |
|       diff0[3]      |    |       diff1[3]       |
-------------------------------------------------------
|       diff2[0]      |    |       diff3[0]       |
|       diff2[1]      |    |       diff3[1]       |
|       diff2[2]      |    |       diff3[2]       |
|       diff2[3]      |    |       diff3[3]       |
*/
    // row 0-3
    size_t vl = __riscv_vsetvl_e8m1(16);
    const uint8_t *pix1_ptr = pix1;
    const uint8_t *pix2_ptr = pix2;
    vuint8m1_t s0_0, s0_1, s0_2, s0_3;
    vuint8m1_t r0_0, r0_1, r0_2, r0_3;
    vectors_u8_m1_t s0 = {&s0_0, &s0_1, &s0_2, &s0_3};
    vectors_u8_m1_t r0 = {&r0_0, &r0_1, &r0_2, &r0_3};
    vload_u8m1x4(&s0, &pix1_ptr, stride_pix1, vl);
    vload_u8m1x4(&r0, &pix2_ptr, stride_pix2, vl);

    // Low part
    vuint8mf2_t s0_h_0, s0_h_1, s0_h_2, s0_h_3;
    vuint8mf2_t r0_h_0, r0_h_1, r0_h_2, r0_h_3;
    vectors_u8_mf2_t s0_half = {&s0_h_0, &s0_h_1, &s0_h_2, &s0_h_3};
    vectors_u8_mf2_t r0_half = {&r0_h_0, &r0_h_1, &r0_h_2, &r0_h_3};
    vget_first8_u8m1(&s0_half, s0);
    vget_first8_u8m1(&r0_half, r0);
    vuint16m1_t diff0_0, diff0_1, diff0_2, diff0_3;
    vectors_u16_m1_t diff0 = {&diff0_0, &diff0_1, &diff0_2, &diff0_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff0, s0_half, r0_half, vl);

    // High part
    vl = __riscv_vsetvl_e8m1(16);
    vuint8m1_t s1_0, s1_1, s1_2, s1_3;
    vuint8m1_t r1_0, r1_1, r1_2, r1_3;
    vectors_u8_m1_t s1 = {&s1_0, &s1_1, &s1_2, &s1_3};
    vectors_u8_m1_t r1 = {&r1_0, &r1_1, &r1_2, &r1_3};
    vslidedown_u8x4(&s1, &r1, s0, r0, vl);
    vuint8mf2_t s1_h_0, s1_h_1, s1_h_2, s1_h_3;
    vuint8mf2_t r1_h_0, r1_h_1, r1_h_2, r1_h_3;
    vectors_u8_mf2_t s1_half = {&s1_h_0, &s1_h_1, &s1_h_2, &s1_h_3};
    vectors_u8_mf2_t r1_half = {&r1_h_0, &r1_h_1, &r1_h_2, &r1_h_3};
    vget_first8_u8m1(&s1_half, s1);
    vget_first8_u8m1(&r1_half, r1);
    vuint16m1_t diff1_0, diff1_1, diff1_2, diff1_3;
    vectors_u16_m1_t diff1 = {&diff1_0, &diff1_1, &diff1_2, &diff1_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff1, s1_half, r1_half, vl);

    vuint16m1_t out0, out1;
    vl = __riscv_vsetvl_e16m1(8);
    hadamard_4x4x4(&out0, &out1, &diff0, &diff1, vl);

    // row 4-7
    vl = __riscv_vsetvl_e8m1(16);
    vuint8m1_t s2_0, s2_1, s2_2, s2_3;
    vuint8m1_t r2_0, r2_1, r2_2, r2_3;
    vectors_u8_m1_t s2 = {&s2_0, &s2_1, &s2_2, &s2_3};
    vectors_u8_m1_t r2 = {&r2_0, &r2_1, &r2_2, &r2_3};
    vload_u8m1x4(&s2, &pix1_ptr, stride_pix1, vl);
    vload_u8m1x4(&r2, &pix2_ptr, stride_pix2, vl);

    // Low part
    vuint8mf2_t s2_h_0, s2_h_1, s2_h_2, s2_h_3;
    vuint8mf2_t r2_h_0, r2_h_1, r2_h_2, r2_h_3;
    vectors_u8_mf2_t s2_half = {&s2_h_0, &s2_h_1, &s2_h_2, &s2_h_3};
    vectors_u8_mf2_t r2_half = {&r2_h_0, &r2_h_1, &r2_h_2, &r2_h_3};
    vget_first8_u8m1(&s2_half, s2);
    vget_first8_u8m1(&r2_half, r2);
    vuint16m1_t diff2_0, diff2_1, diff2_2, diff2_3;
    vectors_u16_m1_t diff2 = {&diff2_0, &diff2_1, &diff2_2, &diff2_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff2, s2_half, r2_half, vl);

    // High part
    vl = __riscv_vsetvl_e8m1(16);
    vuint8m1_t s3_0, s3_1, s3_2, s3_3;
    vuint8m1_t r3_0, r3_1, r3_2, r3_3;
    vectors_u8_m1_t s3 = {&s3_0, &s3_1, &s3_2, &s3_3};
    vectors_u8_m1_t r3 = {&r3_0, &r3_1, &r3_2, &r3_3};
    vslidedown_u8x4(&s3, &r3, s2, r2, vl);
    vuint8mf2_t s3_h_0, s3_h_1, s3_h_2, s3_h_3;
    vuint8mf2_t r3_h_0, r3_h_1, r3_h_2, r3_h_3;
    vectors_u8_mf2_t s3_half = {&s3_h_0, &s3_h_1, &s3_h_2, &s3_h_3};
    vectors_u8_mf2_t r3_half = {&r3_h_0, &r3_h_1, &r3_h_2, &r3_h_3};
    vget_first8_u8m1(&s3_half, s3);
    vget_first8_u8m1(&r3_half, r3);
    vuint16m1_t diff3_0, diff3_1, diff3_2, diff3_3;
    vectors_u16_m1_t diff3 = {&diff3_0, &diff3_1, &diff3_2, &diff3_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff3, s3_half, r3_half, vl);

    vuint16m1_t out2, out3;
    vl = __riscv_vsetvl_e16m1(8);
    hadamard_4x4x4(&out2, &out3, &diff2, &diff3, vl);

    vuint16m1_t sum0 = __riscv_vadd_vv_u16m1(out0, out1, vl);
    vuint16m1_t sum1 = __riscv_vadd_vv_u16m1(out2, out3, vl);
    vuint16m1_t sum2 = __riscv_vadd_vv_u16m1(sum0, sum1, vl);
    int sum = vredsum_u16(sum2, vl);

    return sum;
}

static inline int pixel_satd_16x16_rvv(const uint8_t *pix1, intptr_t stride_pix1,
                                       const uint8_t *pix2, intptr_t stride_pix2) {
/*
|       8 element     |    |       8 element      |
|       diff0[0]      |    |       diff1[0]       |
|       diff0[1]      |    |       diff1[1]       |
|       diff0[2]      |    |       diff1[2]       |
|       diff0[3]      |    |       diff1[3]       |
                        ...
*/
    // Load 16x16 blocks four times
    // 1:
    size_t vl = __riscv_vsetvl_e8m1(16);
    const uint8_t *pix1_ptr = pix1;
    const uint8_t *pix2_ptr = pix2;
    vuint8m1_t s0_0, s0_1, s0_2, s0_3;
    vuint8m1_t r0_0, r0_1, r0_2, r0_3;
    vectors_u8_m1_t s0 = {&s0_0, &s0_1, &s0_2, &s0_3};
    vectors_u8_m1_t r0 = {&r0_0, &r0_1, &r0_2, &r0_3};
    vload_u8m1x4(&s0, &pix1_ptr, stride_pix1, vl);
    vload_u8m1x4(&r0, &pix2_ptr, stride_pix2, vl);

    // Low part
    vuint8mf2_t s0_h_0, s0_h_1, s0_h_2, s0_h_3;
    vuint8mf2_t r0_h_0, r0_h_1, r0_h_2, r0_h_3;
    vectors_u8_mf2_t s0_half = {&s0_h_0, &s0_h_1, &s0_h_2, &s0_h_3};
    vectors_u8_mf2_t r0_half = {&r0_h_0, &r0_h_1, &r0_h_2, &r0_h_3};
    vget_first8_u8m1(&s0_half, s0);
    vget_first8_u8m1(&r0_half, r0);
    vuint16m1_t diff0_0, diff0_1, diff0_2, diff0_3;
    vectors_u16_m1_t diff0 = {&diff0_0, &diff0_1, &diff0_2, &diff0_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff0, s0_half, r0_half, vl);

    // High part
    vl = __riscv_vsetvl_e8m1(16);
    vuint8m1_t s1_0, s1_1, s1_2, s1_3;
    vuint8m1_t r1_0, r1_1, r1_2, r1_3;
    vectors_u8_m1_t s1 = {&s1_0, &s1_1, &s1_2, &s1_3};
    vectors_u8_m1_t r1 = {&r1_0, &r1_1, &r1_2, &r1_3};
    vslidedown_u8x4(&s1, &r1, s0, r0, vl);
    vuint8mf2_t s1_h_0, s1_h_1, s1_h_2, s1_h_3;
    vuint8mf2_t r1_h_0, r1_h_1, r1_h_2, r1_h_3;
    vectors_u8_mf2_t s1_half = {&s1_h_0, &s1_h_1, &s1_h_2, &s1_h_3};
    vectors_u8_mf2_t r1_half = {&r1_h_0, &r1_h_1, &r1_h_2, &r1_h_3};
    vget_first8_u8m1(&s1_half, s1);
    vget_first8_u8m1(&r1_half, r1);
    vuint16m1_t diff1_0, diff1_1, diff1_2, diff1_3;
    vectors_u16_m1_t diff1 = {&diff1_0, &diff1_1, &diff1_2, &diff1_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff1, s1_half, r1_half, vl);

    vuint16m1_t out0, out1, sum0, sum1;
    vl = __riscv_vsetvl_e16m1(8);
    hadamard_4x4x4(&out0, &out1, &diff0, &diff1, vl);
    sum0 = out0;
    sum1 = out1;

    // 2:
    vl = __riscv_vsetvl_e8m1(16);
    vuint8m1_t s2_0, s2_1, s2_2, s2_3;
    vuint8m1_t r2_0, r2_1, r2_2, r2_3;
    vectors_u8_m1_t s2 = {&s2_0, &s2_1, &s2_2, &s2_3};
    vectors_u8_m1_t r2 = {&r2_0, &r2_1, &r2_2, &r2_3};
    vload_u8m1x4(&s2, &pix1_ptr, stride_pix1, vl);
    vload_u8m1x4(&r2, &pix2_ptr, stride_pix2, vl);

    // Low part
    vuint8mf2_t s2_h_0, s2_h_1, s2_h_2, s2_h_3;
    vuint8mf2_t r2_h_0, r2_h_1, r2_h_2, r2_h_3;
    vectors_u8_mf2_t s2_half = {&s2_h_0, &s2_h_1, &s2_h_2, &s2_h_3};
    vectors_u8_mf2_t r2_half = {&r2_h_0, &r2_h_1, &r2_h_2, &r2_h_3};
    vget_first8_u8m1(&s2_half, s2);
    vget_first8_u8m1(&r2_half, r2);
    vuint16m1_t diff2_0, diff2_1, diff2_2, diff2_3;
    vectors_u16_m1_t diff2 = {&diff2_0, &diff2_1, &diff2_2, &diff2_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff2, s2_half, r2_half, vl);

    // High part
    vl = __riscv_vsetvl_e8m1(16);
    vuint8m1_t s3_0, s3_1, s3_2, s3_3;
    vuint8m1_t r3_0, r3_1, r3_2, r3_3;
    vectors_u8_m1_t s3 = {&s3_0, &s3_1, &s3_2, &s3_3};
    vectors_u8_m1_t r3 = {&r3_0, &r3_1, &r3_2, &r3_3};
    vslidedown_u8x4(&s3, &r3, s2, r2, vl);
    vuint8mf2_t s3_h_0, s3_h_1, s3_h_2, s3_h_3;
    vuint8mf2_t r3_h_0, r3_h_1, r3_h_2, r3_h_3;
    vectors_u8_mf2_t s3_half = {&s3_h_0, &s3_h_1, &s3_h_2, &s3_h_3};
    vectors_u8_mf2_t r3_half = {&r3_h_0, &r3_h_1, &r3_h_2, &r3_h_3};
    vget_first8_u8m1(&s3_half, s3);
    vget_first8_u8m1(&r3_half, r3);
    vuint16m1_t diff3_0, diff3_1, diff3_2, diff3_3;
    vectors_u16_m1_t diff3 = {&diff3_0, &diff3_1, &diff3_2, &diff3_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff3, s3_half, r3_half, vl);

    vl = __riscv_vsetvl_e16m1(8);
    hadamard_4x4x4(&out0, &out1, &diff2, &diff3, vl);
    sum0 = __riscv_vadd_vv_u16m1(sum0, out0, vl);
    sum1 = __riscv_vadd_vv_u16m1(sum1, out1, vl);

    // 3:
    vl = __riscv_vsetvl_e8m1(16);
    vuint8m1_t s4_0, s4_1, s4_2, s4_3;
    vuint8m1_t r4_0, r4_1, r4_2, r4_3;
    vectors_u8_m1_t s4 = {&s4_0, &s4_1, &s4_2, &s4_3};
    vectors_u8_m1_t r4 = {&r4_0, &r4_1, &r4_2, &r4_3};
    vload_u8m1x4(&s4, &pix1_ptr, stride_pix1, vl);
    vload_u8m1x4(&r4, &pix2_ptr, stride_pix2, vl);

    // Low part
    vuint8mf2_t s4_h_0, s4_h_1, s4_h_2, s4_h_3;
    vuint8mf2_t r4_h_0, r4_h_1, r4_h_2, r4_h_3;
    vectors_u8_mf2_t s4_half = {&s4_h_0, &s4_h_1, &s4_h_2, &s4_h_3};
    vectors_u8_mf2_t r4_half = {&r4_h_0, &r4_h_1, &r4_h_2, &r4_h_3};
    vget_first8_u8m1(&s4_half, s4);
    vget_first8_u8m1(&r4_half, r4);
    vuint16m1_t diff4_0, diff4_1, diff4_2, diff4_3;
    vectors_u16_m1_t diff4 = {&diff4_0, &diff4_1, &diff4_2, &diff4_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff4, s4_half, r4_half, vl);

    // High part
    vl = __riscv_vsetvl_e8m1(16);
    vuint8m1_t s5_0, s5_1, s5_2, s5_3;
    vuint8m1_t r5_0, r5_1, r5_2, r5_3;
    vectors_u8_m1_t s5 = {&s5_0, &s5_1, &s5_2, &s5_3};
    vectors_u8_m1_t r5 = {&r5_0, &r5_1, &r5_2, &r5_3};
    vslidedown_u8x4(&s5, &r5, s4, r4, vl);
    vuint8mf2_t s5_h_0, s5_h_1, s5_h_2, s5_h_3;
    vuint8mf2_t r5_h_0, r5_h_1, r5_h_2, r5_h_3;
    vectors_u8_mf2_t s5_half = {&s5_h_0, &s5_h_1, &s5_h_2, &s5_h_3};
    vectors_u8_mf2_t r5_half = {&r5_h_0, &r5_h_1, &r5_h_2, &r5_h_3};
    vget_first8_u8m1(&s5_half, s5);
    vget_first8_u8m1(&r5_half, r5);
    vuint16m1_t diff5_0, diff5_1, diff5_2, diff5_3;
    vectors_u16_m1_t diff5 = {&diff5_0, &diff5_1, &diff5_2, &diff5_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff5, s5_half, r5_half, vl);

    vl = __riscv_vsetvl_e16m1(8);
    hadamard_4x4x4(&out0, &out1, &diff4, &diff5, vl);
    sum0 = __riscv_vadd_vv_u16m1(sum0, out0, vl);
    sum1 = __riscv_vadd_vv_u16m1(sum1, out1, vl);

    // 4:
    vl = __riscv_vsetvl_e8m1(16);
    vuint8m1_t s6_0, s6_1, s6_2, s6_3;
    vuint8m1_t r6_0, r6_1, r6_2, r6_3;
    vectors_u8_m1_t s6 = {&s6_0, &s6_1, &s6_2, &s6_3};
    vectors_u8_m1_t r6 = {&r6_0, &r6_1, &r6_2, &r6_3};
    vload_u8m1x4(&s6, &pix1_ptr, stride_pix1, vl);
    vload_u8m1x4(&r6, &pix2_ptr, stride_pix2, vl);

    // Low part
    vuint8mf2_t s6_h_0, s6_h_1, s6_h_2, s6_h_3;
    vuint8mf2_t r6_h_0, r6_h_1, r6_h_2, r6_h_3;
    vectors_u8_mf2_t s6_half = {&s6_h_0, &s6_h_1, &s6_h_2, &s6_h_3};
    vectors_u8_mf2_t r6_half = {&r6_h_0, &r6_h_1, &r6_h_2, &r6_h_3};
    vget_first8_u8m1(&s6_half, s6);
    vget_first8_u8m1(&r6_half, r6);
    vuint16m1_t diff6_0, diff6_1, diff6_2, diff6_3;
    vectors_u16_m1_t diff6 = {&diff6_0, &diff6_1, &diff6_2, &diff6_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff6, s6_half, r6_half, vl);

    // High part
    vl = __riscv_vsetvl_e8m1(16);
    vuint8m1_t s7_0, s7_1, s7_2, s7_3;
    vuint8m1_t r7_0, r7_1, r7_2, r7_3;
    vectors_u8_m1_t s7 = {&s7_0, &s7_1, &s7_2, &s7_3};
    vectors_u8_m1_t r7 = {&r7_0, &r7_1, &r7_2, &r7_3};
    vslidedown_u8x4(&s7, &r7, s6, r6, vl);
    vuint8mf2_t s7_h_0, s7_h_1, s7_h_2, s7_h_3;
    vuint8mf2_t r7_h_0, r7_h_1, r7_h_2, r7_h_3;
    vectors_u8_mf2_t s7_half = {&s7_h_0, &s7_h_1, &s7_h_2, &s7_h_3};
    vectors_u8_mf2_t r7_half = {&r7_h_0, &r7_h_1, &r7_h_2, &r7_h_3};
    vget_first8_u8m1(&s7_half, s7);
    vget_first8_u8m1(&r7_half, r7);
    vuint16m1_t diff7_0, diff7_1, diff7_2, diff7_3;
    vectors_u16_m1_t diff7 = {&diff7_0, &diff7_1, &diff7_2, &diff7_3};
    vl = __riscv_vsetvl_e8mf2(8);
    vwsubu_u8x4(&diff7, s7_half, r7_half, vl);

    vl = __riscv_vsetvl_e16m1(8);
    hadamard_4x4x4(&out0, &out1, &diff6, &diff7, vl);
    sum0 = __riscv_vadd_vv_u16m1(sum0, out0, vl);
    sum1 = __riscv_vadd_vv_u16m1(sum1, out1, vl);

    // Reduced sum
    vl = __riscv_vsetvl_e32m2(8);
    vuint32m2_t sum_ext0 = __riscv_vzext_vf2_u32m2(sum0, vl);
    vuint32m2_t sum_ext1 = __riscv_vzext_vf2_u32m2(sum1, vl);

    vuint32m1_t v_sum = __riscv_vmv_v_x_u32m1(0, vl);
    v_sum = __riscv_vredsum_vs_u32m2_u32m1(sum_ext0, v_sum, vl);
    v_sum = __riscv_vredsum_vs_u32m2_u32m1(sum_ext1, v_sum, vl);
    return __riscv_vmv_x_s_u32m1_u32(v_sum);
}

static inline int pixel_sa8d_8x8_rvv(const uint8_t *pix1, intptr_t stride_pix1,
                                      const uint8_t *pix2, intptr_t stride_pix2)
{
    const uint8_t *pix1_ptr = pix1;
    const uint8_t *pix2_ptr = pix2;
    vuint16m1_t diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7;
    vectors_u16_m1_8_t diff = {&diff0, &diff1, &diff2, &diff3, &diff4, &diff5, &diff6, &diff7};
    vuint16m1_t res0, res1;
    vuint16m1_t out;

    load_diff_u8x8x8(pix1_ptr, stride_pix1, pix2_ptr, stride_pix2, &diff);

    size_t vl = __riscv_vsetvl_e16m1(8);
    hadamard_8x8(&res0, &res1, &diff, vl);
    out = __riscv_vadd_vv_u16m1(res0, res1, vl);
    int sum = vredsum_u16(out, vl);
    return (sum + 1) >> 1;
}

static inline int pixel_sa8d_16x16_rvv(const uint8_t *pix1, intptr_t stride_pix1,
                                        const uint8_t *pix2, intptr_t stride_pix2)
{
    const uint8_t *pix1_ptr = pix1;
    const uint8_t *pix2_ptr = pix2;
    vuint16m1_t diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7;
    vectors_u16_m1_8_t diff = {&diff0, &diff1, &diff2, &diff3, &diff4, &diff5, &diff6, &diff7};
    vuint16m1_t res0, res1;
    vuint32m2_t sum, tmp;

    load_diff_u8x8x8(pix1_ptr, stride_pix1, pix2, stride_pix2, &diff);
    size_t vl = __riscv_vsetvl_e16m1(8);
    hadamard_8x8(&res0, &res1, &diff, vl);
    sum = __riscv_vwaddu_vv_u32m2(res0, res1, vl);

    load_diff_u8x8x8(pix1_ptr + 8, stride_pix1, pix2_ptr + 8, stride_pix2, &diff);
    hadamard_8x8(&res0, &res1, &diff, vl);
    tmp = __riscv_vwaddu_vv_u32m2(res0, res1, vl);
    sum = __riscv_vadd_vv_u32m2(tmp, sum, vl);

    load_diff_u8x8x8(pix1_ptr + 8 * stride_pix1, stride_pix1, pix2_ptr + 8 * stride_pix2, stride_pix2, &diff);
    hadamard_8x8(&res0, &res1, &diff, vl);
    tmp = __riscv_vwaddu_vv_u32m2(res0, res1, vl);
    sum = __riscv_vadd_vv_u32m2(tmp, sum, vl);

    load_diff_u8x8x8(pix1_ptr + 8 * stride_pix1 + 8, stride_pix1, pix2_ptr + 8 * stride_pix2 + 8, stride_pix2, &diff);
    hadamard_8x8(&res0, &res1, &diff, vl);
    tmp = __riscv_vwaddu_vv_u32m2(res0, res1, vl);
    sum = __riscv_vadd_vv_u32m2(tmp, sum, vl);

    int sum_int = vredsum_u32(sum, vl);

    return (sum_int + 1) >> 1;
}

#endif // HIGH_BIT_DEPTH

// To be optimized
template<int lx, int ly>
void pixelavg_pp_rvv(pixel *dst, intptr_t dstride, const pixel *src0, intptr_t sstride0, const pixel *src1,
                     intptr_t sstride1, int)
{
    // Use rnu rounding mode
    const unsigned int vxrm = 0;
    for (int y = 0; y < ly; y++)
    {
        int x = 0;
        size_t vl;
        for (; x < lx; x += vl) {
#if HIGH_BIT_DEPTH
            vl = __riscv_vsetvl_e16m1(lx - x);
            vuint16m1_t in0 = __riscv_vle16_v_u16m1((const uint16_t*)(src0 + x), vl);
            vuint16m1_t in1 = __riscv_vle16_v_u16m1((const uint16_t*)(src1 + x), vl);
            // compute (a + b + 1) >> 1
            vuint16m1_t avg = __riscv_vaaddu_vv_u16m1(in0, in1, vxrm, vl);
            __riscv_vse16_v_u16m1((uint16_t*)(dst + x), avg, vl);
#else
            vl = __riscv_vsetvl_e8m1(lx - x);
            vuint8m1_t in0 = __riscv_vle8_v_u8m1((const uint8_t*)(src0 + x), vl);
            vuint8m1_t in1 = __riscv_vle8_v_u8m1((const uint8_t*)(src1 + x), vl);
            // zero-extended
            vuint16m2_t w_in0 = __riscv_vzext_vf2_u16m2(in0, vl);
            vuint16m2_t w_in1 = __riscv_vzext_vf2_u16m2(in1, vl);
            vuint16m2_t sum = __riscv_vadd_vv_u16m2(w_in0, w_in1, vl);
            vuint8m1_t avg = __riscv_vnclipu_wx_u8m1(sum, 1, vxrm, vl);
            __riscv_vse8_v_u8m1((uint8_t*)(dst + x), avg, vl);
#endif
        }
        src0 += sstride0;
        src1 += sstride1;
        dst += dstride;
    }
}

#if !(HIGH_BIT_DEPTH)
template<int w, int h>
int satd4_rvv(const pixel *pix1, intptr_t stride_pix1, const pixel *pix2, intptr_t stride_pix2)
{
    int satd = 0;

    if (w == 4 && h == 4) {
        satd = pixel_satd_4x4_rvv(pix1, stride_pix1, pix2, stride_pix2);
    } else {
        for (int row = 0; row < h; row += 8)
            for (int col = 0; col < w; col += 4)
                satd += pixel_satd_4x8_rvv(pix1 + row * stride_pix1 + col, stride_pix1,
                                           pix2 + row * stride_pix2 + col, stride_pix2);
    }

    return satd;
}

template<int w, int h>
int satd8_rvv(const pixel *pix1, intptr_t stride_pix1, const pixel *pix2, intptr_t stride_pix2)
{
    int satd = 0;

    if (w % 16 == 0 && h % 16 == 0)
    {
        for (int row = 0; row < h; row += 16)
            for (int col = 0; col < w; col += 16)
                satd += pixel_satd_16x16_rvv(pix1 + row * stride_pix1 + col, stride_pix1,
                                              pix2 + row * stride_pix2 + col, stride_pix2);
    }
    else if (w % 8 == 0 && h % 16 == 0)
    {
        for (int row = 0; row < h; row += 16)
            for (int col = 0; col < w; col += 8)
                satd += pixel_satd_8x16_rvv(pix1 + row * stride_pix1 + col, stride_pix1,
                                             pix2 + row * stride_pix2 + col, stride_pix2);
    }
    else if (w % 16 == 0 && h % 8 == 0)
    {
        for (int row = 0; row < h; row += 8)
            for (int col = 0; col < w; col += 16)
                satd += pixel_satd_16x8_rvv(pix1 + row * stride_pix1 + col, stride_pix1,
                                             pix2 + row * stride_pix2 + col, stride_pix2);
    }
    else if (w % 16 == 0 && h % 4 == 0)
    {
        for (int row = 0; row < h; row += 4)
            for (int col = 0; col < w; col += 16)
                satd += pixel_satd_16x4_rvv(pix1 + row * stride_pix1 + col, stride_pix1,
                                             pix2 + row * stride_pix2 + col, stride_pix2);
    }
    else if (w % 8 == 0 && h % 8 == 0)
    {
        for (int row = 0; row < h; row += 8)
            for (int col = 0; col < w; col += 8)
                satd += pixel_satd_8x8_rvv(pix1 + row * stride_pix1 + col, stride_pix1,
                                            pix2 + row * stride_pix2 + col, stride_pix2);
    }
    else // w multiple of 8, h multiple of 4
    {
        for (int row = 0; row < h; row += 4)
            for (int col = 0; col < w; col += 8)
                satd += pixel_satd_8x4_rvv(pix1 + row * stride_pix1 + col, stride_pix1,
                                            pix2 + row * stride_pix2 + col, stride_pix2);
    }

    return satd;
}

// Calculate sa8d in blocks of 8x8
template<int w, int h>
int sa8d8_rvv(const pixel *pix1, intptr_t i_pix1, const pixel *pix2, intptr_t i_pix2)
{
    int cost = 0;

    for (int y = 0; y < h; y += 8)
        for (int x = 0; x < w; x += 8)
        {
            cost += pixel_sa8d_8x8_rvv(pix1 + i_pix1 * y + x, i_pix1, pix2 + i_pix2 * y + x, i_pix2);
        }

    return cost;
}

// Calculate sa8d in blocks of 16x16
template<int w, int h>
int sa8d16_rvv(const pixel *pix1, intptr_t i_pix1, const pixel *pix2, intptr_t i_pix2)
{
    int cost = 0;

    for (int y = 0; y < h; y += 16)
        for (int x = 0; x < w; x += 16)
        {
            cost += pixel_sa8d_16x16_rvv(pix1 + i_pix1 * y + x, i_pix1, pix2 + i_pix2 * y + x, i_pix2);
        }

    return cost;
}
#endif

#if HIGH_BIT_DEPTH
// todo
#else // !HIGH_BIT_DEPTH
static inline int calc_energy_8x8(const uint8_t *source, intptr_t sstride)
{
    const uint8_t *pix_ptr = source;

    vuint8mf2_t s0, s1, s2, s3, s4, s5, s6, s7;
    vectors_u8_mf2_t s_matrix_low = {&s0, &s1, &s2, &s3};
    vectors_u8_mf2_t s_matrix_high = {&s4, &s5, &s6, &s7};

    size_t vl = __riscv_vsetvl_e8mf2(8);
    vload_u8x8x4_mf2(&pix_ptr, sstride, &s_matrix_low, vl);
    pix_ptr += 4 * sstride;
    vload_u8x8x4_mf2(&pix_ptr, sstride, &s_matrix_high, vl);

    vuint16m1_t diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7;
    diff0 = __riscv_vwaddu_vv_u16m1(*(s_matrix_low.m0), *(s_matrix_low.m1), vl);
    diff1 = __riscv_vwaddu_vv_u16m1(*(s_matrix_low.m2), *(s_matrix_low.m3), vl);
    diff2 = __riscv_vwaddu_vv_u16m1(*(s_matrix_high.m0), *(s_matrix_high.m1), vl);
    diff3 = __riscv_vwaddu_vv_u16m1(*(s_matrix_high.m2), *(s_matrix_high.m3), vl);
    diff4 = __riscv_vwsubu_vv_u16m1(*(s_matrix_low.m0), *(s_matrix_low.m1), vl);
    diff5 = __riscv_vwsubu_vv_u16m1(*(s_matrix_low.m2), *(s_matrix_low.m3), vl);
    diff6 = __riscv_vwsubu_vv_u16m1(*(s_matrix_high.m0), *(s_matrix_high.m1), vl);
    diff7 = __riscv_vwsubu_vv_u16m1(*(s_matrix_high.m2), *(s_matrix_high.m3), vl);

    vectors_u16_m1_t diff_0 = {&diff0, &diff1, &diff2, &diff3};
    vectors_u16_m1_t diff_1 = {&diff4, &diff5, &diff6, &diff7};

    vuint16m1_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    vectors_u16_m1_t tmp_0 = {&tmp0, &tmp1, &tmp2, &tmp3};
    vectors_u16_m1_t tmp_1 = {&tmp4, &tmp5, &tmp6, &tmp7};
    hadamard_vx4(&tmp_0, &diff_0, vl);
    hadamard_vx4(&tmp_1, &diff_1, vl);

    // The first line after the vertical hadamard transform contains the sum of coefficients.
    int sum = vredsum_u16(*(tmp_0.m0), vl) >> 2;

    vuint16m1_t sum0, sum1, sum2, sum3;
    vectors_u16_m1_t sum_sa8d={&sum0, &sum1, &sum2, &sum3};
    vectors_u16_m1_8_t tmp={tmp_0.m0, tmp_0.m1, tmp_0.m2, tmp_0.m3, tmp_1.m0, tmp_1.m1, tmp_1.m2, tmp_1.m3};
    hadamard_hx8(&sum_sa8d, &tmp, vl);

    vuint16m1_t out, out0, out1;
    out0 = __riscv_vadd_vv_u16m1(*(sum_sa8d.m0), *(sum_sa8d.m1), vl);
    out1 = __riscv_vadd_vv_u16m1(*(sum_sa8d.m2), *(sum_sa8d.m3), vl);
    out = __riscv_vadd_vv_u16m1(out0, out1, vl);
    int sa8 = (vredsum_u16(out, vl) + 1) >> 1;

    return sa8 - sum;
}


static inline int calc_energy_4x4(const pixel *source, intptr_t sstride)
{
    const uint8_t *pix_ptr = source;
    vuint8mf2_t s0, s1, s2, s3;
    vectors_u8_mf2_t s = {&s0, &s1, &s2, &s3};
    size_t vl = __riscv_vsetvl_e8mf2(4);
    vload_u8mf2x4(&s, &pix_ptr, sstride, vl);
    vuint16m1_t tmp0, tmp1, tmp2, tmp3;
    tmp0 = __riscv_vwaddu_vx_u16m1(*(s.m0), 0, vl);
    tmp1 = __riscv_vwaddu_vx_u16m1(*(s.m1), 0, vl);
    tmp2 = __riscv_vwaddu_vx_u16m1(*(s.m2), 0, vl);
    tmp3 = __riscv_vwaddu_vx_u16m1(*(s.m3), 0, vl);

    vuint16m1_t s01_23, d01_23, d0, d1;
    d0 = __riscv_vslideup_vx_u16m1(tmp0, tmp1, 4, 8);
    d1 = __riscv_vslideup_vx_u16m1(tmp2, tmp3, 4, 8);

    vl = __riscv_vsetvl_e16m1(8);
    SUMSUB_AB(s01_23, d01_23, d0, d1, vl);

    // The first line after the vertical hadamard transform contains the sum of coefficients.
    int sum = vredsum_u16(s01_23, vl) >> 2;

    vtrn_16s(&d0, &d1, s01_23, d01_23);
    SUMSUB_AB(s01_23, d01_23, d0, d1, vl);

    vtrn_8h(&d0, &d1, s01_23, d01_23);
    SUMSUB_AB(s01_23, d01_23, d0, d1, vl);

    vtrn_4s(&d0, &d1, s01_23, d01_23);

    vuint16m1_t vmax = vmax_abs_u16(d0, d1, vl);
    int sat = vredsum_u16(vmax, vl);

    return sat - sum;
}

template<int size>
int psyCost_pp_rvv(const pixel *source, intptr_t sstride, const pixel *recon, intptr_t rstride)
{
    if (size)
    {
        int dim = 1 << (size + 2);
        uint32_t totEnergy = 0;
        for (int i = 0; i < dim; i += 8)
        {
            for (int j = 0; j < dim; j += 8)
            {
                int sourceEnergy = calc_energy_8x8(source + i * sstride + j, sstride);
                int reconEnergy = calc_energy_8x8(recon + i * rstride + j, rstride);

                totEnergy += abs(sourceEnergy - reconEnergy);
            }
        }
        return totEnergy;
    }
    else
    {
        int sourceEnergy = calc_energy_4x4(source, sstride);
        int reconEnergy = calc_energy_4x4(recon, rstride);

        return abs(sourceEnergy - reconEnergy);
    }
}
#endif // HIGH_BIT_DEPTH

template<int blockSize>
void transpose_rvv(pixel *dst, const pixel *src, intptr_t stride)
{
    for (int k = 0; k < blockSize; k++)
        for (int l = 0; l < blockSize; l++)
        {
            dst[k * blockSize + l] = src[l * stride + k];
        }
}

template<>
__attribute__((unused))
void transpose_rvv<8>(pixel *dst, const pixel *src, intptr_t stride)
{
    transpose8x8_rvv(dst, src, 8, stride);
}

template<>
__attribute__((unused))
void transpose_rvv<16>(pixel *dst, const pixel *src, intptr_t stride)
{
    transpose16x16_rvv(dst, src, 16, stride);
}

template<>
__attribute__((unused))
void transpose_rvv<32>(pixel *dst, const pixel *src, intptr_t stride)
{
    transpose32x32_rvv(dst, src, 32, stride);
}

template<>
__attribute__((unused))
void transpose_rvv<64>(pixel *dst, const pixel *src, intptr_t stride)
{
    transpose32x32_rvv(dst, src, 64, stride);
    transpose32x32_rvv(dst + 32 * 64 + 32, src + 32 * stride + 32, 64, stride);
    transpose32x32_rvv(dst + 32 * 64, src + 32, 64, stride);
    transpose32x32_rvv(dst + 32, src + 32 * stride, 64, stride);
}

};

namespace X265_NS {
void setupPixelPrimitives_rvv(EncoderPrimitives &p) {
#define LUMA_PU(W, H) \
    p.pu[LUMA_ ## W ## x ## H].pixelavg_pp[NONALIGNED] = pixelavg_pp_rvv<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].pixelavg_pp[ALIGNED] = pixelavg_pp_rvv<W, H>;

#if !(HIGH_BIT_DEPTH)
#define LUMA_PU_S(W, H)
#else // (HIGH_BIT_DEPTH)
#define LUMA_PU_S(W, H) \
    p.pu[LUMA_ ## W ## x ## H].pixelavg_pp[NONALIGNED] = pixelavg_pp_rvv<W, H>; \
    p.pu[LUMA_ ## W ## x ## H].pixelavg_pp[ALIGNED] = pixelavg_pp_rvv<W, H>;
#endif // !(HIGH_BIT_DEPTH)

#define LUMA_CU(W, H) \
    p.cu[BLOCK_ ## W ## x ## H].psy_cost_pp   = psyCost_pp_rvv<BLOCK_ ## W ## x ## H>;
    //p.cu[BLOCK_ ## W ## x ## H].transpose = transpose_rvv<W>;

    LUMA_PU_S(4, 4);
    LUMA_PU_S(8, 8);
    LUMA_PU(16, 16);
    LUMA_PU(32, 32);
    LUMA_PU(64, 64);
    LUMA_PU_S(4, 8);
    LUMA_PU_S(8, 4);
    LUMA_PU(16,  8);
    LUMA_PU_S(8, 16);
    LUMA_PU(16, 12);
    LUMA_PU(12, 16);
    LUMA_PU(16,  4);
    LUMA_PU_S(4, 16);
    LUMA_PU(32, 16);
    LUMA_PU(16, 32);
    LUMA_PU(32, 24);
    LUMA_PU(24, 32);
    LUMA_PU(32,  8);
    LUMA_PU_S(8, 32);
    LUMA_PU(64, 32);
    LUMA_PU(32, 64);
    LUMA_PU(64, 48);
    LUMA_PU(48, 64);
    LUMA_PU(64, 16);
    LUMA_PU(16, 64);

#if !(HIGH_BIT_DEPTH)
    LUMA_CU(4, 4);
    LUMA_CU(8, 8);
    LUMA_CU(16, 16);
    LUMA_CU(32, 32);
    LUMA_CU(64, 64);

    p.pu[LUMA_4x4].satd = satd4_rvv<4, 4>;
    p.pu[LUMA_4x8].satd = satd4_rvv<4, 8>;
    p.pu[LUMA_4x16].satd = satd4_rvv<4, 16>;
    p.pu[LUMA_8x4].satd = satd8_rvv<8, 4>;
    p.pu[LUMA_8x8].satd = satd8_rvv<8, 8>;
    p.pu[LUMA_8x16].satd = satd8_rvv<8, 16>;
    p.pu[LUMA_8x32].satd = satd8_rvv<8, 32>;
    p.pu[LUMA_12x16].satd = satd4_rvv<12, 16>;
    p.pu[LUMA_16x4].satd = satd8_rvv<16, 4>;
    p.pu[LUMA_16x8].satd = satd8_rvv<16, 8>;
    p.pu[LUMA_16x12].satd = satd8_rvv<16, 12>;
    p.pu[LUMA_16x16].satd = satd8_rvv<16, 16>;
    p.pu[LUMA_16x32].satd = satd8_rvv<16, 32>;
    p.pu[LUMA_16x64].satd = satd8_rvv<16, 64>;
    p.pu[LUMA_24x32].satd = satd8_rvv<24, 32>;
    p.pu[LUMA_32x8].satd = satd8_rvv<32, 8>;
    p.pu[LUMA_32x16].satd = satd8_rvv<32, 16>;
    p.pu[LUMA_32x24].satd = satd8_rvv<32, 24>;
    p.pu[LUMA_32x32].satd = satd8_rvv<32, 32>;
    p.pu[LUMA_32x64].satd = satd8_rvv<32, 64>;
    p.pu[LUMA_48x64].satd = satd8_rvv<48, 64>;
    p.pu[LUMA_64x16].satd = satd8_rvv<64, 16>;
    p.pu[LUMA_64x32].satd = satd8_rvv<64, 32>;
    p.pu[LUMA_64x48].satd = satd8_rvv<64, 48>;
    p.pu[LUMA_64x64].satd = satd8_rvv<64, 64>;

    p.cu[BLOCK_4x4].sa8d   = satd4_rvv<4, 4>;
    p.cu[BLOCK_8x8].sa8d   = sa8d8_rvv<8, 8>;
    p.cu[BLOCK_16x16].sa8d = sa8d16_rvv<16, 16>;
    p.cu[BLOCK_32x32].sa8d = sa8d16_rvv<32, 32>;
    p.cu[BLOCK_64x64].sa8d = sa8d16_rvv<64, 64>;

    p.chroma[X265_CSP_I420].pu[CHROMA_420_2x2].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_2x4].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_2x8].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x2].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x4].satd = satd4_rvv<4, 4>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x8].satd = satd4_rvv<4, 8>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_4x16].satd = satd4_rvv<4, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_6x8].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x2].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x4].satd = satd8_rvv<8, 4>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x6].satd = NULL;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x8].satd = satd8_rvv<8, 8>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x16].satd = satd8_rvv<8, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_8x32].satd = satd8_rvv<8, 32>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_12x16].satd = satd4_rvv<12, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x4].satd = satd8_rvv<16, 4>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x8].satd = satd8_rvv<16, 8>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x12].satd = satd8_rvv<16, 12>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x16].satd = satd8_rvv<16, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_16x32].satd = satd8_rvv<16, 32>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_24x32].satd = satd8_rvv<24, 32>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x8].satd = satd8_rvv<32, 8>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x16].satd = satd8_rvv<32, 16>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x24].satd = satd8_rvv<32, 24>;
    p.chroma[X265_CSP_I420].pu[CHROMA_420_32x32].satd = satd8_rvv<32, 32>;

    p.chroma[X265_CSP_I422].pu[CHROMA_422_2x4].satd = NULL;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_2x8].satd = NULL;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_2x16].satd = NULL;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x4].satd = satd4_rvv<4, 4>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x8].satd = satd4_rvv<4, 8>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x16].satd = satd4_rvv<4, 16>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_4x32].satd = satd4_rvv<4, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_6x16].satd = NULL;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x4].satd = satd8_rvv<8, 4>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x8].satd = satd8_rvv<8, 8>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x12].satd = satd8_rvv<8, 12>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x16].satd = satd8_rvv<8, 16>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x32].satd = satd8_rvv<8, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_8x64].satd = satd8_rvv<8, 64>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_12x32].satd = satd4_rvv<12, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x8].satd = satd8_rvv<16, 8>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x16].satd = satd8_rvv<16, 16>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x24].satd = satd8_rvv<16, 24>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x32].satd = satd8_rvv<16, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_16x64].satd = satd8_rvv<16, 64>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_24x64].satd = satd8_rvv<24, 64>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x16].satd = satd8_rvv<32, 16>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x32].satd = satd8_rvv<32, 32>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x48].satd = satd8_rvv<32, 48>;
    p.chroma[X265_CSP_I422].pu[CHROMA_422_32x64].satd = satd8_rvv<32, 64>;

    p.chroma[X265_CSP_I420].cu[BLOCK_8x8].sa8d   = p.chroma[X265_CSP_I420].pu[CHROMA_420_4x4].satd;
    p.chroma[X265_CSP_I420].cu[BLOCK_16x16].sa8d = sa8d8_rvv<8, 8>;
    p.chroma[X265_CSP_I420].cu[BLOCK_32x32].sa8d = sa8d16_rvv<16, 16>;
    p.chroma[X265_CSP_I420].cu[BLOCK_64x64].sa8d = sa8d16_rvv<32, 32>;

    p.chroma[X265_CSP_I422].cu[BLOCK_8x8].sa8d       = p.chroma[X265_CSP_I422].pu[CHROMA_422_4x8].satd;
    p.chroma[X265_CSP_I422].cu[BLOCK_16x16].sa8d     = sa8d8_rvv<8, 16>;
    p.chroma[X265_CSP_I422].cu[BLOCK_32x32].sa8d     = sa8d16_rvv<16, 32>;
    p.chroma[X265_CSP_I422].cu[BLOCK_64x64].sa8d     = sa8d16_rvv<32, 64>;

    p.chroma[X265_CSP_I422].cu[BLOCK_422_8x16].sa8d  = sa8d8_rvv<8, 16>;
    p.chroma[X265_CSP_I422].cu[BLOCK_422_16x32].sa8d = sa8d16_rvv<16, 32>;
    p.chroma[X265_CSP_I422].cu[BLOCK_422_32x64].sa8d = sa8d16_rvv<32, 64>;

#else
    (void)p;
#endif
}

} // namespace X265_NS
