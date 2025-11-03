/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Changsheng Wu <wu.changsheng@sanechips.com.cn>
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

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4
pelFilterLumaStrong_coeff_16:
    .hword 0, 2, 1, 1, 2, 3, 3, 7
    .hword 100, 4, 2, 5, 6, 4, 4, 100
    .hword 100, 1, 3, 2, 3, 5, 5, 100
    .hword 100, 0, 4, 3, 4, 6, 6, 100
    .hword 100, 3, 100, 4, 5, 100, 7, 100

    .hword 0, 3, 1, 2, 2, 1, 1, 0
    .hword 0, 2, 1, 2, 2, 1, 3, 0
    .hword 0, 1, 1, 2, 2, 1, 2, 0

    .hword 0, 4, 2, 4, 4, 2, 4, 0
    .hword 0, 3, 2, 3, 3, 2, 3, 0

.align 4
.text

//void processSaoCUE0(pixel * rec, int8_t * offsetEo, int width, int8_t* signLeft, intptr_t stride)
function PFX(processSaoCUE0_rvv)
#if HIGH_BIT_DEPTH
    slli            a4, a4, 1
#endif
    li              a5, (1 << BIT_DEPTH) - 1
    lb              t0, (a3)
    lb              t1, 1(a3)
    neg             t0, t0
    neg             t1, t1
    add             t3, a0, a4
    vsetvli         zero, a2, e16, m2, ta, ma
    vmv.v.i         v30, 0
    vmv.v.x         v28, a5
    vsetvli         zero, a2, e8, m1, ta, ma
    vmv.s.x         v8, t0
    vmv.s.x         v9, t1
loop_processSaoCUE0:
#if HIGH_BIT_DEPTH
    vsetvli         t2, a2, e16, m2, ta, ma
    addi            t4, a0, 2
    addi            t5, t3, 2
    vle16.v         v2, (a0)
    vle16.v         v20, (t4)
    vle16.v         v4, (t3)
    vle16.v         v22, (t5)
    SIGNOF_RVV      v24, v2, v20
    SIGNOF_RVV      v26, v4, v22
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v6, v24, 0
    vnsrl.wi        v7, v26, 0
#else
    vsetvli         t2, a2, e8, m1, ta, ma
    addi            t4, a0, 1
    addi            t5, t3, 1
    vle8.v          v2, (a0)
    vle8.v          v20, (t4)
    vle8.v          v4, (t3)
    vle8.v          v22, (t5)
    SIGNOF_RVV      v6, v2, v20
    SIGNOF_RVV      v7, v4, v22
#endif
    addi            t6, t2, -1
    vslideup.vi     v8, v6, 1
    vslideup.vi     v9, v7, 1
    vsub.vv         v10, v6, v8
    vsub.vv         v11, v7, v9
    vadd.vi         v10, v10, 2
    vadd.vi         v11, v11, 2
    vslidedown.vx   v8, v6, t6
    vslidedown.vx   v9, v7, t6

#if HIGH_BIT_DEPTH
    vluxei8.v       v20, (a1), v10
    vluxei8.v       v22, (a1), v11
    vwadd.wv        v2, v2, v20
    vwadd.wv        v4, v4, v22
    vsetvli         zero, t2, e16, m2, ta, ma
    vmax.vv         v2, v2, v30
    vmax.vv         v4, v4, v30
    vmin.vv         v2, v2, v28
    vmin.vv         v4, v4, v28
    vse16.v         v2, (a0)
    vse16.v         v4, (t3)
#else
    vluxei8.v       v20, (a1), v10
    vluxei8.v       v22, (a1), v11
    vsetvli         zero, t2, e16, m2, ta, ma
    vzext.vf2       v12, v2
    vzext.vf2       v14, v4
    vsetvli         zero, t2, e8, m1, ta, ma
    vwadd.wv        v12, v12, v20
    vwadd.wv        v14, v14, v22
    vsetvli         zero, t2, e16, m2, ta, ma
    vmax.vv         v12, v12, v30
    vmax.vv         v14, v14, v30
    vmin.vv         v12, v12, v28
    vmin.vv         v14, v14, v28
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v16, v12, 0
    vnsrl.wi        v17, v14, 0
    vse8.v          v16, (a0)
    vse8.v          v17, (t3)
#endif

#if HIGH_BIT_DEPTH
    slli            t1, t2, 1
    add             a0, a0, t1
    add             t3, t3, t1
#else
    add             a0, a0, t2
    add             t3, t3, t2
#endif
    sub             a2, a2, t2
    bgtz            a2, loop_processSaoCUE0
    ret
endfunc

//void processSaoCUE1(pixel* rec, int8_t* upBuff1, int8_t* offsetEo, intptr_t stride, int width)
function PFX(processSaoCUE1_rvv)
#if HIGH_BIT_DEPTH
    slli            a3, a3, 1
#endif
    li              a5, (1 << BIT_DEPTH) - 1
    add             t0, a0, a3
    vsetvli         zero, a4, e16, m2, ta, ma
    vmv.v.i         v30, 0
    vmv.v.x         v28, a5
loop_processSaoCUE1:
#if HIGH_BIT_DEPTH
    vsetvli         t2, a4, e16, m2, ta, ma
    vle16.v         v14, (a0)
    vle16.v         v4, (t0)
    vle8.v          v6, (a1)
    SIGNOF_RVV      v20, v14, v4
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v8, v20, 0
#else
    vsetvli         t2, a4, e8, m1, ta, ma
    vle8.v          v2, (a0)
    vle8.v          v4, (t0)
    vle8.v          v6, (a1)
    SIGNOF_RVV      v8, v2, v4
#endif
    vadd.vv         v10, v8, v6
    vadd.vi         v10, v10, 2
    vrsub.vi        v8, v8, 0
    vse8.v          v8, (a1)
    vluxei8.v       v12, (a2), v10

#if !HIGH_BIT_DEPTH
    vsetvli         zero, t2, e16, m2, ta, ma
    vzext.vf2       v14, v2
    vsetvli         zero, t2, e8, m1, ta, ma
#endif
    vwadd.wv        v14, v14, v12
    vsetvli         zero, t2, e16, m2, ta, ma
    vmax.vv         v14, v14, v30
    vmin.vv         v14, v14, v28
#if HIGH_BIT_DEPTH
    vse16.v         v14, (a0)
#else
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v16, v14, 0
    vse8.v          v16, (a0)
#endif

#if HIGH_BIT_DEPTH
    slli            t3, t2, 1
    add             a0, a0, t3
    add             t0, t0, t3
#else
    add             a0, a0, t2
    add             t0, t0, t2
#endif
    add             a1, a1, t2
    sub             a4, a4, t2
    bgtz            a4, loop_processSaoCUE1
    ret
endfunc

//void processSaoCUE1_2Rows(pixel* rec, int8_t* upBuff1, int8_t* offsetEo, intptr_t stride, int width)
function PFX(processSaoCUE1_2Rows_rvv)
#if HIGH_BIT_DEPTH
    slli            a3, a3, 1
#endif
    li              a5, (1 << BIT_DEPTH) - 1
    add             t0, a0, a3
    add             t4, t0, a3
    vsetvli         zero, a4, e16, m2, ta, ma
    vmv.v.i         v30, 0
    vmv.v.x         v18, a5
loop_processSaoCUE1_2Rows:
#if HIGH_BIT_DEPTH
    vsetvli         t2, a4, e16, m2, ta, ma
    vle16.v         v14, (a0)
    vle16.v         v28, (t0)
    vle16.v         v2, (t4)
    vle8.v          v6, (a1)
    SIGNOF_RVV      v20, v14, v28
    SIGNOF_RVV      v4, v28, v2
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v8, v20, 0
    vnsrl.wi        v22, v4, 0
#else
    vsetvli         t2, a4, e8, m1, ta, ma
    vle8.v          v2, (a0)
    vle8.v          v4, (t0)
    vle8.v          v20, (t4)
    vle8.v          v6, (a1)
    SIGNOF_RVV      v8, v2, v4
    SIGNOF_RVV      v22, v4, v20
#endif
    vadd.vv         v10, v8, v6
    vadd.vi         v10, v10, 2
    vrsub.vi        v8, v8, 0
    vadd.vv         v24, v22, v8
    vadd.vi         v24, v24, 2
    vrsub.vi        v8, v22, 0
    vse8.v          v8, (a1)
    vluxei8.v       v12, (a2), v10
    vluxei8.v       v26, (a2), v24

#if !HIGH_BIT_DEPTH
    vsetvli         zero, t2, e16, m2, ta, ma
    vzext.vf2       v14, v2
    vzext.vf2       v28, v4
    vsetvli         zero, t2, e8, m1, ta, ma
#endif
    vwadd.wv        v14, v14, v12
    vwadd.wv        v28, v28, v26
    vsetvli         zero, t2, e16, m2, ta, ma
    vmax.vv         v14, v14, v30
    vmax.vv         v28, v28, v30
    vmin.vv         v14, v14, v18
    vmin.vv         v28, v28, v18
#if HIGH_BIT_DEPTH
    vse16.v         v14, (a0)
    vse16.v         v28, (t0)
#else
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v16, v14, 0
    vnsrl.wi        v12, v28, 0
    vse8.v          v16, (a0)
    vse8.v          v12, (t0)
#endif

#if HIGH_BIT_DEPTH
    slli            t3, t2, 1
    add             a0, a0, t3
    add             t0, t0, t3
    add             t4, t4, t3
#else
    add             a0, a0, t2
    add             t0, t0, t2
    add             t4, t4, t2
#endif
    add             a1, a1, t2
    sub             a4, a4, t2
    bgtz            a4, loop_processSaoCUE1_2Rows
    ret
endfunc

//void processSaoCUE2(pixel * rec, int8_t * bufft, int8_t * buff1, int8_t * offsetEo, int width, intptr_t stride)
function PFX(processSaoCUE2_rvv)
#if HIGH_BIT_DEPTH
    slli            a5, a5, 1
    add             t0, a0, a5
    addi            t0, t0, 2
#else
    add             t0, a0, a5
    addi            t0, t0, 1
#endif
    li              a6, (1 << BIT_DEPTH) - 1
    addi            a1, a1, 1
    vsetvli         zero, a4, e16, m2, ta, ma
    vmv.v.i         v30, 0
    vmv.v.x         v28, a6

loo_processSaoCUE2:
#if HIGH_BIT_DEPTH
    vsetvli         t2, a4, e16, m2, ta, ma
    vle16.v         v16, (a0)
    vle16.v         v4, (t0)
    vle8.v          v6, (a2)
    SIGNOF_RVV      v10, v16, v4
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v8, v10, 0
#else
    vsetvli         t2, a4, e8, m1, ta, ma
    vle8.v          v2, (a0)
    vle8.v          v4, (t0)
    vle8.v          v6, (a2)
    SIGNOF_RVV      v8, v2, v4
#endif
    vadd.vv         v12, v8, v6
    vadd.vi         v12, v12, 2
    vrsub.vi        v8, v8, 0
    vse8.v          v8, (a1)
    vluxei8.v       v14, (a3), v12

#if !HIGH_BIT_DEPTH
    vsetvli         zero, t2, e16, m2, ta, ma
    vzext.vf2       v16, v2
    vsetvli         zero, t2, e8, m1, ta, ma
#endif
    vwadd.wv        v16, v16, v14
    vsetvli         zero, t2, e16, m2, ta, ma
    vmax.vv         v16, v16, v30
    vmin.vv         v16, v16, v28
#if HIGH_BIT_DEPTH
    vse16.v         v16, (a0)
#else
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v18, v16, 0
    vse8.v          v18, (a0)
#endif

#if HIGH_BIT_DEPTH
    slli            t3, t2, 1
    add             a0, a0, t3
    add             t0, t0, t3
#else
    add             a0, a0, t2
    add             t0, t0, t2
#endif
    add             a1, a1, t2
    add             a2, a2, t2
    sub             a4, a4, t2
    bgtz            a4, loo_processSaoCUE2
    ret
endfunc

//void processSaoCUE3(pixel *rec, int8_t *upBuff1, int8_t *offsetEo, intptr_t stride, int startX, int endX)
function PFX(processSaoCUE3_rvv)
#if HIGH_BIT_DEPTH
    slli            t1, a4, 1
    add             a0, a0, t1
    addi            a0, a0, 2
    slli            a3, a3, 1
    add             t0, a0, a3
#else
    add             a0, a0, a4
    addi            a0, a0, 1
    add             t0, a0, a3
#endif
    add             a1, a1, a4
    addi            a1, a1, 1
    sub             a4, a5, a4
    addi            a4, a4, -1
    addi            t5, a1, -1
    bltz            a4, processSaoCUE3_finish
    li              a6, (1 << BIT_DEPTH) - 1
    vsetvli         zero, a4, e16, m2, ta, ma
    vmv.v.i         v30, 0
    vmv.v.x         v28, a6

loo_processSaoCUE3:
#if HIGH_BIT_DEPTH
    vsetvli         t2, a4, e16, m2, ta, ma
    vle16.v         v16, (a0)
    vle16.v         v4, (t0)
    vle8.v          v6, (a1)
    SIGNOF_RVV      v10, v16, v4
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v8, v10, 0
#else
    vsetvli         t2, a4, e8, m1, ta, ma
    vle8.v          v2, (a0)
    vle8.v          v4, (t0)
    vle8.v          v6, (a1)
    SIGNOF_RVV      v8, v2, v4
#endif
    vadd.vv         v12, v8, v6
    vadd.vi         v12, v12, 2
    vrsub.vi        v8, v8, 0
    vse8.v          v8, (t5)
    vluxei8.v       v14, (a2), v12

#if !HIGH_BIT_DEPTH
    vsetvli         zero, t2, e16, m2, ta, ma
    vzext.vf2       v16, v2
    vsetvli         zero, t2, e8, m1, ta, ma
#endif
    vwadd.wv        v16, v16, v14
    vsetvli         zero, t2, e16, m2, ta, ma
    vmax.vv         v16, v16, v30
    vmin.vv         v16, v16, v28
#if HIGH_BIT_DEPTH
    vse16.v         v16, (a0)
#else
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v18, v16, 0
    vse8.v          v18, (a0)
#endif

#if HIGH_BIT_DEPTH
    slli            t3, t2, 1
    add             a0, a0, t3
    add             t0, t0, t3
#else
    add             a0, a0, t2
    add             t0, t0, t2
#endif
    add             t5, t5, t2
    add             a1, a1, t2
    sub             a4, a4, t2
    bgtz            a4, loo_processSaoCUE3
processSaoCUE3_finish:
    ret
endfunc

//void processSaoCUB0(pixel* rec, const int8_t* offset, int ctuWidth, int ctuHeight, intptr_t stride)
function PFX(processSaoCUB0_rvv)
#if HIGH_BIT_DEPTH
    slli            a4, a4, 1
#endif
    li              a6, (1 << BIT_DEPTH) - 1
    vsetvli         zero, a2, e16, m2, ta, ma
    vmv.v.i         v30, 0
    vmv.v.x         v28, a6
loop_processSaoCUB0_y:
    mv              t0, a0
    mv              t1, a2
loop_processSaoCUB0_x:
#if HIGH_BIT_DEPTH
    vsetvli         t2, t1, e16, m2, ta, ma
    vle16.v         v16, (t0)
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v4, v16, BIT_DEPTH - 5
#else
    vsetvli         t2, t1, e8, m1, ta, ma
    vle8.v          v2, (t0)
    vsrl.vi         v4, v2, BIT_DEPTH - 5
#endif
    vluxei8.v       v6, (a1), v4

#if !HIGH_BIT_DEPTH
    vsetvli         zero, t2, e16, m2, ta, ma
    vzext.vf2       v16, v2
    vsetvli         zero, t2, e8, m1, ta, ma
#endif
    vwadd.wv        v16, v16, v6
    vsetvli         zero, t2, e16, m2, ta, ma
    vmax.vv         v16, v16, v30
    vmin.vv         v16, v16, v28
#if HIGH_BIT_DEPTH
    vse16.v         v16, (t0)
#else
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v18, v16, 0
    vse8.v          v18, (t0)
#endif

#if HIGH_BIT_DEPTH
    slli            t3, t2, 1
    add             t0, t0, t3
#else
    add             t0, t0, t2
#endif
    sub             t1, t1, t2
    bgtz            t1, loop_processSaoCUB0_x
    addi            a3, a3, -1
    add             a0, a0, a4
    bgtz            a3, loop_processSaoCUB0_y
    ret
endfunc

//void calSign(int8_t *dst, const pixel *src1, const pixel *src2, const int endX)
function PFX(calSign_rvv)
loop_calSign:
#if HIGH_BIT_DEPTH
    vsetvli         t2, a3, e16, m2, ta, ma
    vle16.v         v2, (a1)
    vle16.v         v4, (a2)
    SIGNOF_RVV      v6, v2, v4
    vsetvli         zero, t2, e8, m1, ta, ma
    vnsrl.wi        v8, v6, 0
    vse8.v          v8, (a0)
    slli            t3, t2, 1
    add             a0, a0, t2
    add             a1, a1, t3
    add             a2, a2, t3
#else
    vsetvli         t2, a3, e8, m1, ta, ma
    vle8.v          v2, (a1)
    vle8.v          v4, (a2)
    SIGNOF_RVV      v6, v2, v4
    vse8.v          v6, (a0)
    add             a0, a0, t2
    add             a1, a1, t2
    add             a2, a2, t2
#endif
    sub             a3, a3, t2
    bgtz            a3, loop_calSign
    ret
endfunc


//void pelFilterLumaStrong_c(pixel* src, intptr_t srcStep, intptr_t offset, int32_t tcP, int32_t tcQ)
function PFX(pelFilterLumaStrong_v_rvv)
#if HIGH_BIT_DEPTH
    slli            a1, a1, 1
    addi            a0, a0, -8
#else
    addi            a0, a0, -4
#endif
    slli            a6, a1, 1
    vsetivli        zero, 8, e16, m1, ta, ma
    la              t0, pelFilterLumaStrong_coeff_16
    addi            t1, t0, 16
    addi            t2, t0, 32
    addi            t3, t0, 48
    addi            t4, t0, 64
    vle16.v         v6, (t0)
    vle16.v         v7, (t1)
    vle16.v         v8, (t2)
    vle16.v         v9, (t3)
    vle16.v         v10, (t4)
    addi            t1, t0, 80
    addi            t2, t0, 96
    addi            t3, t0, 112
    addi            t4, t0, 128
    addi            t5, t0, 144
    vle16.v         v12, (t1)
    vle16.v         v13, (t2)
    vle16.v         v14, (t3)
    vle16.v         v15, (t4)
    vle16.v         v16, (t5)
    vmv.v.x         v4, a3
    vmv.v.x         v5, a4
    vslideup.vi     v4, v5, 4
    vrsub.vi        v5, v4, 0

.rept 2
    add             a7, a0, a1
#if HIGH_BIT_DEPTH
    vle16.v         v22, (a0)
    vle16.v         v28, (a7)
#else
    vsetivli        zero, 8, e8, m1, ta, ma
    vle8.v          v0, (a0)
    vle8.v          v1, (a7)
    vsetivli        zero, 8, e16, m1, ta, ma
    vzext.vf2       v22, v0
    vzext.vf2       v28, v1
#endif

    vrgather.vv     v17, v22, v6
    vrgather.vv     v18, v22, v7
    vrgather.vv     v19, v22, v8
    vrgather.vv     v20, v22, v9
    vrgather.vv     v21, v22, v10
    vrgather.vv     v23, v28, v6
    vrgather.vv     v24, v28, v7
    vrgather.vv     v25, v28, v8
    vrgather.vv     v26, v28, v9
    vrgather.vv     v27, v28, v10

    vmacc.vv        v17, v19, v12
    vmacc.vv        v23, v25, v12
    vmacc.vv        v18, v20, v13
    vmacc.vv        v24, v26, v13
    vmacc.vv        v17, v21, v14
    vmacc.vv        v23, v27, v14
    vadd.vv         v17, v17, v18
    vadd.vv         v23, v23, v24
    vadd.vv         v17, v17, v15
    vadd.vv         v23, v23, v15
    vsra.vv         v17, v17, v16
    vsra.vv         v23, v23, v16
    vsub.vv         v17, v17, v22
    vsub.vv         v23, v23, v28

    vmin.vv         v17, v17, v4
    vmin.vv         v23, v23, v4
    vmax.vv         v17, v17, v5
    vmax.vv         v23, v23, v5

    vadd.vv         v18, v17, v22
    vadd.vv         v24, v23, v28

#if HIGH_BIT_DEPTH
    vse16.v         v18, (a0)
    vse16.v         v24, (a7)
#else
    vsetivli        zero, 8, e8, m1, ta, ma
    vnsrl.wi        v17, v18, 0
    vnsrl.wi        v23, v24, 0
    vse8.v          v17, (a0)
    vse8.v          v23, (a7)
    vsetivli        zero, 8, e16, m1, ta, ma
#endif

    add             a0, a0, a6
.endr
    ret
endfunc

//void pelFilterLumaStrong_c(pixel* src, intptr_t srcStep, intptr_t offset, int32_t tcP, int32_t tcQ)
function PFX(pelFilterLumaStrong_h_rvv)
#if HIGH_BIT_DEPTH
    slli            a2, a2, 1
#endif
    slli            t0, a2, 2
    sub             a0, a0, t0
    add             t0, a0, a2
    add             t1, t0, a2
    add             t2, t1, a2
    add             t3, t2, a2
    add             t4, t3, a2
    add             t5, t4, a2
    add             t6, t5, a2
    vsetivli        zero, 4, e16, m1, ta, ma
    vmv.v.i         v30, 2

#if HIGH_BIT_DEPTH
    vsetivli        zero, 4, e16, m1, ta, ma
    vle16.v         v0, (a0)
    vle16.v         v1, (t0)
    vle16.v         v2, (t1)
    vle16.v         v3, (t2)
    vle16.v         v4, (t3)
    vle16.v         v5, (t4)
    vle16.v         v6, (t5)
    vle16.v         v7, (t6)
#else
    vsetivli        zero, 4, e8, m1, ta, ma
    vle8.v          v8, (a0)
    vle8.v          v9, (t0)
    vle8.v          v10, (t1)
    vle8.v          v11, (t2)
    vle8.v          v12, (t3)
    vle8.v          v13, (t4)
    vle8.v          v14, (t5)
    vle8.v          v15, (t6)
    vsetivli        zero, 4, e16, m1, ta, ma
    vzext.vf2       v0, v8
    vzext.vf2       v1, v9
    vzext.vf2       v2, v10
    vzext.vf2       v3, v11
    vzext.vf2       v4, v12
    vzext.vf2       v5, v13
    vzext.vf2       v6, v14
    vzext.vf2       v7, v15
#endif
    vadd.vv         v8, v3, v4
    vadd.vv         v22, v1, v5
    vadd.vv         v20, v0, v1
    vadd.vv         v9, v8, v2
    vadd.vv         v10, v8, v5
    vadd.vv         v28, v2, v6
    vmacc.vv        v22, v9, v30
    vadd.vv         v26, v9, v1
    vadd.vv         v24, v10, v6
    vadd.vv         v18, v6, v7
    vmadd.vv        v20, v30, v26
    vmacc.vv        v28, v10, v30
    vmadd.vv        v18, v30, v24

    vadd.vi         v20, v20, 4
    vadd.vi         v26, v26, 2
    vadd.vi         v22, v22, 4
    vadd.vi         v28, v28, 4
    vadd.vi         v24, v24, 2
    vadd.vi         v18, v18, 4
    vsra.vi         v20, v20, 3
    vsra.vi         v26, v26, 2
    vsra.vi         v22, v22, 3
    vsra.vi         v28, v28, 3
    vsra.vi         v24, v24, 2
    vsra.vi         v18, v18, 3
    vsub.vv         v20, v20, v1
    vsub.vv         v26, v26, v2
    vsub.vv         v22, v22, v3
    vsub.vv         v28, v28, v4
    vsub.vv         v24, v24, v5
    vsub.vv         v18, v18, v6

    vmv.v.x         v10, a3
    vmv.v.x         v12, a4
    vrsub.vi        v11, v10, 0
    vrsub.vi        v13, v12, 0
    vmax.vv         v20, v20, v11
    vmax.vv         v26, v26, v11
    vmax.vv         v22, v22, v11
    vmax.vv         v28, v28, v13
    vmax.vv         v24, v24, v13
    vmax.vv         v18, v18, v13
    vmin.vv         v20, v20, v10
    vmin.vv         v26, v26, v10
    vmin.vv         v22, v22, v10
    vmin.vv         v28, v28, v12
    vmin.vv         v24, v24, v12
    vmin.vv         v18, v18, v12
    vadd.vv         v20, v20, v1
    vadd.vv         v26, v26, v2
    vadd.vv         v22, v22, v3
    vadd.vv         v28, v28, v4
    vadd.vv         v24, v24, v5
    vadd.vv         v18, v18, v6

#if HIGH_BIT_DEPTH
    vse16.v         v20, (t0)
    vse16.v         v26, (t1)
    vse16.v         v22, (t2)
    vse16.v         v28, (t3)
    vse16.v         v24, (t4)
    vse16.v         v18, (t5)
#else
    vsetivli        zero, 4, e8, m1, ta, ma
    vnsrl.wi        v0, v20, 0
    vnsrl.wi        v1, v26, 0
    vnsrl.wi        v2, v22, 0
    vnsrl.wi        v3, v28, 0
    vnsrl.wi        v4, v24, 0
    vnsrl.wi        v5, v18, 0
    vse8.v          v0, (t0)
    vse8.v          v1, (t1)
    vse8.v          v2, (t2)
    vse8.v          v3, (t3)
    vse8.v          v4, (t4)
    vse8.v          v5, (t5)
#endif
    ret
endfunc

//void pelFilterChroma_V_c(pixel *src, intptr_t srcStep, intptr_t offset, int32_t tc, int32_t maskP, int32_t maskQ)
function PFX(pelFilterChroma_V_rvv)
#if HIGH_BIT_DEPTH
    slli            a1, a1, 1
    addi            a0, a0, -4
    addi            t0, a0, 2
    addi            t1, a0, 4
    addi            t2, a0, 6
    vsetivli        zero, 4, e16, m1, ta, ma
    vlse16.v        v2, (a0), a1
    vlse16.v        v3, (t0), a1
    vlse16.v        v4, (t1), a1
    vlse16.v        v5, (t2), a1
#else
    addi            a0, a0, -2
    addi            t0, a0, 1
    addi            t1, a0, 2
    addi            t2, a0, 3
    vsetivli        zero, 4, e8, m1, ta, ma
    vlse8.v         v6, (a0), a1
    vlse8.v         v7, (t0), a1
    vlse8.v         v8, (t1), a1
    vlse8.v         v9, (t2), a1
    vsetivli        zero, 4, e16, m1, ta, ma
    vzext.vf2       v2, v6
    vzext.vf2       v3, v7
    vzext.vf2       v4, v8
    vzext.vf2       v5, v9
#endif
    li              a6, (1 << BIT_DEPTH) - 1
    neg             t3, a3
    vmv.v.i         v27, 4
    vmv.v.i         v25, 0
    vmv.v.x         v28, a3
    vmv.v.x         v29, t3
    vmv.v.x         v30, a4
    vmv.v.x         v31, a5
    vmv.v.x         v26, a6
    vsub.vv         v6, v4, v3
    vsub.vv         v7, v2, v5
    vmacc.vv        v7, v6, v27
    vadd.vi         v7, v7, 4
    vsra.vi         v7, v7, 3
    vmax.vv         v7, v7, v29
    vmin.vv         v7, v7, v28

    vand.vv         v8, v7, v30
    vand.vv         v10, v7, v31
    vadd.vv         v8, v8, v3
    vsub.vv         v10, v4, v10
    vmax.vv         v8, v8, v25
    vmax.vv         v10, v10, v25
    vmin.vv         v8, v8, v26
    vmin.vv         v10, v10, v26

#if HIGH_BIT_DEPTH
    vsse16.v        v8, (t0), a1
    vsse16.v        v10, (t1), a1
#else
    vsetivli        zero, 4, e8, m1, ta, ma
    vnsrl.wi        v0, v8, 0
    vnsrl.wi        v1, v10, 0
    vsse8.v         v0, (t0), a1
    vsse8.v         v1, (t1), a1
#endif
    ret
endfunc

//void pelFilterChroma_H_c(pixel *src, intptr_t srcStep, intptr_t offset, int32_t tc, int32_t maskP, int32_t maskQ)
function PFX(pelFilterChroma_H_rvv)
#if HIGH_BIT_DEPTH
    slli            a2, a2, 1
#endif
    slli            t3, a2, 1
    sub             t0, a0, t3
    sub             t1, a0, a2
    add             t2, a0, a2
#if HIGH_BIT_DEPTH
    vsetivli        zero, 4, e16, m1, ta, ma
    vle16.v         v2, (t0)
    vle16.v         v3, (t1)
    vle16.v         v4, (a0)
    vle16.v         v5, (t2)
#else
    vsetivli        zero, 4, e8, m1, ta, ma
    vle8.v          v6, (t0)
    vle8.v          v7, (t1)
    vle8.v          v8, (a0)
    vle8.v          v9, (t2)
    vsetivli        zero, 4, e16, m1, ta, ma
    vzext.vf2       v2, v6
    vzext.vf2       v3, v7
    vzext.vf2       v4, v8
    vzext.vf2       v5, v9
#endif
    li              a6, (1 << BIT_DEPTH) - 1
    neg             t3, a3
    vmv.v.i         v27, 4
    vmv.v.i         v25, 0
    vmv.v.x         v28, a3
    vmv.v.x         v29, t3
    vmv.v.x         v30, a4
    vmv.v.x         v31, a5
    vmv.v.x         v26, a6
    vsub.vv         v6, v4, v3
    vsub.vv         v7, v2, v5
    vmacc.vv        v7, v6, v27
    vadd.vi         v7, v7, 4
    vsra.vi         v7, v7, 3
    vmax.vv         v7, v7, v29
    vmin.vv         v7, v7, v28

    vand.vv         v8, v7, v30
    vand.vv         v10, v7, v31
    vadd.vv         v8, v8, v3
    vsub.vv         v10, v4, v10
    vmax.vv         v8, v8, v25
    vmax.vv         v10, v10, v25
    vmin.vv         v8, v8, v26
    vmin.vv         v10, v10, v26

#if HIGH_BIT_DEPTH
    vse16.v         v8, (t1)
    vse16.v         v10, (a0)
#else
    vsetivli        zero, 4, e8, m1, ta, ma
    vnsrl.wi        v0, v8, 0
    vnsrl.wi        v1, v10, 0
    vse8.v          v0, (t1)
    vse8.v          v1, (a0)
#endif
    ret
endfunc
