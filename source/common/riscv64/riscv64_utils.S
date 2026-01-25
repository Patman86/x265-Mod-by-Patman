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

#include "asm.S"

.align 4

.text

.macro CALC_STRIDED_ADDRS src, stride, reg1, reg2, reg3, reg4
    slli                 \reg4, \stride, 1          # reg4 = stride * 2
    add                  \reg1, \src, \stride       # reg1 = src + stride
    add                  \reg2, \src, \reg4         # reg2 = src + 2*stride
    add                  \reg3, \reg2, \stride      # reg3 = src + 3*stride
.endm

.macro LOAD_4_LINE src, es, sstride, d1, d2, d3, d4, reg1, reg2, reg3, reg4
    CALC_STRIDED_ADDRS   \src, \sstride, \reg1, \reg2, \reg3, \reg4
    vl\es\().v           \d1, (\src)
    vl\es\().v           \d2, (\reg1)
    vl\es\().v           \d3, (\reg2)
    vl\es\().v           \d4, (\reg3)
.endm

.macro STORE_4_LINE dst, es, dstride, d1, d2, d3, d4, reg1, reg2, reg3, reg4
    CALC_STRIDED_ADDRS   \dst, \dstride, \reg1, \reg2, \reg3, \reg4
    vs\es\().v           \d1, (\dst)
    vs\es\().v           \d2, (\reg1)
    vs\es\().v           \d3, (\reg2)
    vs\es\().v           \d4, (\reg3)
.endm

.macro TRANSPOSE_HELPER a, b, c, d, e, f, g, h, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8
    vslideup.vi         \tmp1, \b, 1
    vslidedown.vi       \tmp2, \a, 1
    vslideup.vi         \tmp3, \d, 1
    vslidedown.vi       \tmp4, \c, 1
    vslideup.vi         \tmp5, \f, 1
    vslidedown.vi       \tmp6, \e, 1
    vslideup.vi         \tmp7, \h, 1
    vslidedown.vi       \tmp8, \g, 1

    vmerge.vvm          \a, \tmp1, \a, v0
    vmerge.vvm          \b, \b, \tmp2, v0

    vmerge.vvm          \c, \tmp3, \c, v0
    vmerge.vvm          \d, \d, \tmp4, v0

    vmerge.vvm          \e, \tmp5, \e, v0
    vmerge.vvm          \f, \f, \tmp6, v0

    vmerge.vvm          \g, \tmp7, \g, v0
    vmerge.vvm          \h, \h, \tmp8, v0
.endm

#if !HIGH_BIT_DEPTH
# void transpose4x4_rvv(uint8_t *dst, const uint8_t *src, intptr_t dstride, intptr_t sstride)
function transpose4x4_rvv_u8
    li                    t4, 0x5
    vsetivli              zero, 4, e8, mf2, ta, ma
    vmv.v.x               v0, t4

    LOAD_4_LINE           a1, e8, a3, v4, v5, v6, v7, t0, t1, t2, t3
    TRANSPOSE_4x4         8, v16, v17, v18, v19, v4, v5, v6, v7

    vsetivli              zero, 4, e8, mf2, ta, ma
    STORE_4_LINE          a0, e8, a2, v16, v17, v18, v19, t0, t1, t2, t3

    ret
endfunc

function transpose8x8_rvv_u8
    li                    t4, 0x55
    vsetivli              zero, 8, e8, m1, ta, ma
    vmv.v.x               v0, t4

    LOAD_4_LINE           a1, e8, a3, v4, v5, v6, v7, t0, t1, t2, t3
    add                   t4, t2, a3
    LOAD_4_LINE           t4, e8, a3, v8, v9, v10 v11, t0, t1, t2, t3

    TRANSPOSE_8x8         8, v16, v17, v18, v19, v20, v21, v22, v23, v4, v5, v6, v7, v8, v9, v10, v11

    vsetivli              zero, 8, e8, m1, ta, ma
    STORE_4_LINE          a0, e8, a2, v16, v17, v18, v19, t0, t1, t2, t3
    add                   t4, t2, a2
    STORE_4_LINE          t4, e8, a2, v20, v21, v22, v23, t0, t1, t2, t3

    ret
endfunc

function transpose16x16_rvv_u8
    # v0 must be 0x5555 as mask
    li                   t4, 0x5555
    vsetivli             zero, 16, e8, m1, ta, ma
    vmv.v.x              v0, t4

    # load: v1-v16
    LOAD_4_LINE           a1, e8, a3, v1, v2, v3, v4, t0, t1, t2, t3
    add                   t4, t2, a3
    LOAD_4_LINE           t4, e8, a3, v5, v6, v7, v8, t0, t1, t2, t3
    add                   t4, t2, a3
    LOAD_4_LINE           t4, e8, a3, v9, v10, v11, v12, t0, t1, t2, t3
    add                   t4, t2, a3
    LOAD_4_LINE           t4, e8, a3, v13, v14, v15, v16, t0, t1, t2, t3

    # ------------------------------------------------------------
    # Stage 1: e8, VL=16
    # ------------------------------------------------------------
    TRANSPOSE_HELPER      v1, v2, v3, v4, v5, v6, v7, v8, v17, v18, v19, v20, v21, v22, v23, v24
    TRANSPOSE_HELPER      v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24

    # ------------------------------------------------------------
    # Stage 2:  e16, VL=8
    # ------------------------------------------------------------
    vsetivli             zero, 8, e16, m1, ta, ma

    TRANSPOSE_HELPER     v1, v3, v2, v4, v5, v7, v6, v8, v17, v18, v19, v20, v21, v22, v23, v24
    TRANSPOSE_HELPER     v9, v11, v10, v12, v13, v15, v14, v16, v17, v18, v19, v20, v21, v22, v23, v24


    # ------------------------------------------------------------
    # Stage 3: e32, VL=4
    # ------------------------------------------------------------
    vsetivli             zero, 4, e32, m1, ta, ma

    TRANSPOSE_HELPER     v1, v5, v2, v6, v3, v7, v4, v8, v17, v18, v19, v20, v21, v22, v23, v24
    TRANSPOSE_HELPER     v9, v13, v10, v14, v11, v15, v12, v16, v17, v18, v19, v20, v21, v22, v23, v24


    # ------------------------------------------------------------
    # Stage 4: e64, VL=2
    # ------------------------------------------------------------
    vsetivli             zero, 2, e64, m1, ta, ma

    TRANSPOSE_HELPER     v1, v9, v2, v10, v3, v11, v4, v12, v17, v18, v19, v20, v21, v22, v23, v24
    TRANSPOSE_HELPER     v5, v13, v6, v14, v7, v15, v8, v16, v17, v18, v19, v20, v21, v22, v23, v24


    # store: v1-v16
    vsetivli              zero, 16, e8, m1, ta, ma
    STORE_4_LINE          a0, e8, a2, v1, v2, v3, v4, t0, t1, t2, t3
    add                   t4, t2, a2
    STORE_4_LINE          t4, e8, a2, v5, v6, v7, v8, t0, t1, t2, t3
    add                   t4, t2, a2
    STORE_4_LINE          t4, e8, a2, v9, v10, v11, v12, t0, t1, t2, t3
    add                   t4, t2, a2
    STORE_4_LINE          t4, e8, a2, v13, v14, v15, v16, t0, t1, t2, t3

    ret
endfunc

#else
# void transpose4x4_rvv(uint16_t *dst, const uint16_t *src, intptr_t dstride, intptr_t sstride)
function transpose4x4_rvv_u16
    li                     t4, 0x5
    vsetivli               zero, 4, e16, mf2, ta, ma
    vmv.v.x                v0, t4

    slli                   t5, a3, 1
    slli                   t6, a2, 1

    LOAD_4_LINE            a1, e16, t5, v4, v5, v6, v7, t0, t1, t2, t3
    TRANSPOSE_4x4          16, v16, v17, v18, v19, v4, v5, v6, v7

    vsetivli               zero, 4, e16, mf2, ta, ma
    STORE_4_LINE           a0, e16, t6, v16, v17, v18, v19, t0, t1, t2, t3

    ret
endfunc

function transpose8x8_rvv_u16
    li                    t4, 0x55
    vsetivli              zero, 8, e16, m1, ta, ma
    vmv.v.x               v0, t4

    slli                  t5, a3, 1
    slli                  t6, a2, 1

    LOAD_4_LINE           a1, e16, t5, v4, v5, v6, v7, t0, t1, t2, t3
    add                   t4, t2, t5
    LOAD_4_LINE           t4, e16, t5, v8, v9, v10 v11, t0, t1, t2, t3
    TRANSPOSE_8x8         16, v16, v17, v18, v19, v20, v21, v22, v23, v4, v5, v6, v7, v8, v9, v10, v11

    vsetivli              zero, 8, e16, m1, ta, ma
    STORE_4_LINE          a0, e16, t6, v16, v17, v18, v19, t0, t1, t2, t3
    add                   t4, t2, t6
    STORE_4_LINE          t4, e16, t6, v20, v21, v22, v23, t0, t1, t2, t3

    ret
endfunc

#endif