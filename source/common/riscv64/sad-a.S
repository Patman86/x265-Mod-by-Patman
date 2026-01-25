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

.macro push
    addi          sp, sp, -48
    sd            ra, 16(sp)    // save return address
    sd            s0, 32(sp)    // save the frame pointer
    addi          s0, sp, 48    // set the new frame pointer

    sd            x9, 0(sp)
.endm

.macro  pop
    ld           x9, 0(sp)

    // restore the frame pointer and return address
    ld           s0, 32(sp)
    ld           ra, 16(sp)
    addi         sp, sp, 48
.endm

#if !HIGH_BIT_DEPTH
.macro SAD_FUNC_SMALL w, h, lmul1, lmul2
function PFX(pixel_sad_\w\()x\h\()_rvv)
    li            t0, \h
    li            t1, \w
    // prevent overflow when there is a bigger \h
    vsetvli       zero, t1, e32, m\()\lmul2, ta, ma
    vmv.v.i       v28, 0

1:
    addi          t0, t0, -4
    vsetvli       zero, t1, e8, m\()\lmul1, ta, ma

    vle8.v        v0, (a0)
    vle8.v        v1, (a2)
    add           a0, a0, a1
    add           a2, a2, a3
    VABDU         v8, v0, v1, v24

    vle8.v        v2, (a0)
    vle8.v        v3, (a2)
    add           a0, a0, a1
    add           a2, a2, a3
    VABDU         v9, v2, v3, v25

    vle8.v        v4, (a0)
    vle8.v        v5, (a2)
    add           a0, a0, a1
    add           a2, a2, a3
    VABDU         v10, v4, v5, v26

    vle8.v        v6, (a0)
    vle8.v        v7, (a2)
    add           a0, a0, a1
    add           a2, a2, a3
    VABDU         v11, v6, v7, v27

    vsetvli       zero, t1, e32, m\()\lmul2, ta, ma

    vzext.vf4     v0, v8
    vzext.vf4     v4, v9
    vadd.vv       v28, v28, v0
    vadd.vv       v28, v28, v4

    vzext.vf4     v12, v10
    vzext.vf4     v16, v11
    vadd.vv       v28, v28, v12
    vadd.vv       v28, v28, v16

    bnez          t0, 1b

    vmv.v.i       v20, 0
    vredsum.vs    v20, v28, v20
    vmv.x.s       a0, v20
    ret
endfunc
.endm

SAD_FUNC_SMALL 4, 4, 1, 1
SAD_FUNC_SMALL 4, 8, 1, 1
SAD_FUNC_SMALL 4, 16, 1, 1

SAD_FUNC_SMALL 8, 4, 1, 2
SAD_FUNC_SMALL 8, 8, 1, 2
SAD_FUNC_SMALL 8, 16, 1, 2
SAD_FUNC_SMALL 8, 32, 1, 2

SAD_FUNC_SMALL 12, 16, 1, 4

SAD_FUNC_SMALL 16, 4, 1, 4
SAD_FUNC_SMALL 16, 8, 1, 4
SAD_FUNC_SMALL 16, 12, 1, 4
SAD_FUNC_SMALL 16, 16, 1, 4
SAD_FUNC_SMALL 16, 32, 1, 4
SAD_FUNC_SMALL 16, 64, 1, 4

.macro SAD_FUNC_MEDIUM w, h, lmul1, lmul2
function PFX(pixel_sad_\w\()x\h\()_rvv)
    li           t0, \h
    li           t1, \w

    vsetvli      zero, t1, e32, m\()\lmul2, ta, ma
    vmv.v.i      v24, 0

1:
    addi          t0, t0, -2
    vsetvli       zero, t1, e8, m\()\lmul1, ta, ma

    vle8.v        v0, (a0)
    vle8.v        v2, (a2)
    add           a0, a0, a1
    add           a2, a2, a3
    VABDU         v8, v0, v2, v12

    vle8.v        v4, (a0)
    vle8.v        v6, (a2)
    add           a0, a0, a1
    add           a2, a2, a3
    VABDU         v10, v4, v6, v14

    vsetvli       zero, t1, e32, m\()\lmul2, ta, ma

    vzext.vf4     v0, v8
    vadd.vv       v24, v24, v0

    vzext.vf4     v16, v10
    vadd.vv       v24, v24, v16

    bnez          t0, 1b

    vmv.v.i       v0, 0
    vredsum.vs    v0, v24, v0
    vmv.x.s       a0, v0
    ret
endfunc
.endm

SAD_FUNC_MEDIUM  24, 32, 2, 8

SAD_FUNC_MEDIUM  32, 8, 2, 8
SAD_FUNC_MEDIUM  32, 16, 2, 8
SAD_FUNC_MEDIUM  32, 24, 2, 8
SAD_FUNC_MEDIUM  32, 32, 2, 8
SAD_FUNC_MEDIUM  32, 64, 2, 8

.macro SAD_FUNC_COMMON w, h, lmul1, lmul2
function PFX(pixel_sad_\w\()x\h\()_rvv)
    li             t0, \h
    li             t1, \w

    vsetvli        zero, t1, e32, m\()\lmul2, ta, ma
    vmv.v.i        v24, 0

1:  // row loop
    mv             t3, a0
    mv             t4, a2
    addi           t0, t0, -1

2:  // inline segment loop
    vsetvli         t5, t1, e8, m\()\lmul1, ta, ma

    vle8.v          v0, (t3)
    vle8.v          v4, (t4)
    add             t3, t3, t5
    add             t4, t4, t5
    VABDU           v8, v0, v4, v12

    vsetvli         zero, t1, e32,m\()\lmul2, tu, ma
    vzext.vf4       v16, v8
    vadd.vv         v24, v24, v16

    sub             t1, t1, t5
    bgtz            t1, 2b

    // update row pointer
    add             a0, a0, a1
    add             a2, a2, a3

    li              t1, \w      // reset the inline element counter
    bnez            t0, 1b

    li              t1, \w
    vsetvli         zero, t1, e32, m\()\lmul2, ta, ma
    vmv.v.i         v0, 0
    vredsum.vs      v0, v24, v0
    vmv.x.s         a0, v0
    ret
endfunc
.endm
SAD_FUNC_COMMON  48, 64, 1, 4

.macro SAD_FUNC_LARGE w, h, lmul1, lmul2
function PFX(pixel_sad_\w\()x\h\()_rvv)
    li             t0, \h
    li             t1, \w
    li             t2, 0

1:
    addi           t0, t0, -1
    vsetvli        zero, t1, e8, m\()\lmul1, ta, ma

    vle8.v         v0, (a0)
    vle8.v         v4, (a2)
    add            a0, a0, a1
    add            a2, a2, a3
    UWABDU         v8, v0, v4, v12, v28

    vsetvli        zero, t1, e16, m\()\lmul2, ta, ma
    vmv.v.i        v24, 0
    vredsum.vs     v24, v8, v24
    vmv.x.s        t3, v24
    add            t2, t2, t3

    bgtz           t0, 1b
    mv             a0, t2
    ret
endfunc
.endm

SAD_FUNC_LARGE  64, 16, 4, 8
SAD_FUNC_LARGE  64, 32, 4, 8
SAD_FUNC_LARGE  64, 48, 4, 8
SAD_FUNC_LARGE  64, 64, 4, 8

//============= SAD_X3 and SAD_X4 code start========================
// static void x264_pixel_sad_x3_##size(pixel *fenc, pixel *pix0, pixel *pix1, pixel *pix2, intptr_t i_stride, int scores[3])
// static void x264_pixel_sad_x4_##size(pixel *fenc, pixel *pix0, pixel *pix1,pixel *pix2, pixel *pix3, intptr_t i_stride, int scores[4])
.macro SAD_X_FUNC_SMALL x, w, h, lmul1, lmul2
function PFX(sad_x\x\()_\w\()x\h\()_rvv)
// Make function arguments for x == 3 look like x == 4.
.if \x == 3
    mv            t0, a4
    mv            t1, a5
    mv            a5, t0                // a5 = i_stride
    mv            a6, t1                // a6 = scores
.endif
    li            t2, FENC_STRIDE
    li            t3, \w
    li            t4, \h

    vsetvli       zero, t3, e32, m\()\lmul2, ta, ma
    vmv.v.i       v16, 0
    vmv.v.i       v20, 0
    vmv.v.i       v24, 0
.if \x == 4
    vmv.v.i       v28, 0
.endif

1:
    addi           t4, t4, -1
    vsetvli        zero, t3, e8, m\()\lmul1, ta, ma

    vle8.v         v0, (a0)
    vle8.v         v1, (a1)
    add            a0, a0, t2
    add            a1, a1, a5
    VABDU          v5, v0, v1, v9

    vle8.v         v2, (a2)
    vle8.v         v3, (a3)
    add            a2, a2, a5
    add            a3, a3, a5
    VABDU          v6, v0, v2, v10
    VABDU          v7, v0, v3, v11
.if \x == 4
    vle8.v         v4, (a4)
    add            a4, a4, a5
    VABDU          v8, v0, v4, v12
.endif

    vsetvli        zero, t3, e32, m\()\lmul2, ta, ma

    vzext.vf4      v12, v5
    vadd.vv        v16, v16, v12

    vzext.vf4      v0, v6
    vadd.vv        v20, v20, v0

    vzext.vf4      v4, v7
    vadd.vv        v24, v24, v4
.if \x == 4
    vzext.vf4      v12, v8
    vadd.vv        v28, v28, v12
.endif

    bnez           t4, 1b

    // reduce and store the result
    vsetvli        zero, t3, e32, m\()\lmul2, ta, ma
    vmv.v.i        v0, 0
    vredsum.vs     v0, v16, v0
    vmv.x.s        t0, v0
    sw             t0, 0(a6)      // scores[0]

    vmv.v.i        v4, 0
    vredsum.vs     v4, v20, v4
    vmv.x.s        t1, v4
    sw             t1, 4(a6)      // scores[1]

    vmv.v.i        v8, 0
    vredsum.vs     v8, v24, v8
    vmv.x.s        t5, v8
    sw             t5, 8(a6)      // scores[2]

.if \x == 4
    vmv.v.i       v12, 0
    vredsum.vs    v12, v28, v12
    vmv.x.s       t6, v12

    sw            t6, 12(a6)      // scores[3]
.endif
    ret
endfunc
.endm

SAD_X_FUNC_SMALL 3, 4, 4, 1, 1
SAD_X_FUNC_SMALL 3, 4, 8, 1, 1
SAD_X_FUNC_SMALL 3, 4, 16, 1, 1

SAD_X_FUNC_SMALL 3, 8, 4, 1, 2
SAD_X_FUNC_SMALL 3, 8, 8, 1, 2
SAD_X_FUNC_SMALL 3, 8, 16, 1, 2
SAD_X_FUNC_SMALL 3, 8, 32, 1, 2

SAD_X_FUNC_SMALL 4, 4, 4, 1, 1
SAD_X_FUNC_SMALL 4, 4, 8, 1, 1
SAD_X_FUNC_SMALL 4, 4, 16, 1, 1
SAD_X_FUNC_SMALL 4, 8, 4, 1, 2

SAD_X_FUNC_SMALL 4, 8, 8, 1, 2
SAD_X_FUNC_SMALL 4, 8, 16, 1, 2
SAD_X_FUNC_SMALL 4, 8, 32, 1, 2

SAD_X_FUNC_SMALL 3, 12, 16, 1, 4

SAD_X_FUNC_SMALL 3, 16, 4, 1, 4
SAD_X_FUNC_SMALL 3, 16, 8, 1, 4
SAD_X_FUNC_SMALL 3, 16, 12, 1, 4
SAD_X_FUNC_SMALL 3, 16, 16, 1, 4
SAD_X_FUNC_SMALL 3, 16, 32, 1, 4
SAD_X_FUNC_SMALL 3, 16, 64, 1, 4

SAD_X_FUNC_SMALL 4, 12, 16, 1, 4

SAD_X_FUNC_SMALL 4, 16, 4, 1, 4
SAD_X_FUNC_SMALL 4, 16, 8, 1, 4
SAD_X_FUNC_SMALL 4, 16, 12, 1, 4
SAD_X_FUNC_SMALL 4, 16, 16, 1, 4
SAD_X_FUNC_SMALL 4, 16, 32, 1, 4
SAD_X_FUNC_SMALL 4, 16, 64, 1, 4

.macro SAD_X_FUNC_COMMON, x, w, h, lmul1, lmul2
function PFX(sad_x\x\()_\w\()x\h\()_rvv)
// make function arguments for x == 3 look like x == 4.
.if \x == 3
    mv             t0, a4
    mv             t1, a5
    mv             a5, t0
    mv             a6, t1
.endif

    li             t5, \w
    li             t6, \h

    vsetvli       zero, t5, e32, m\()\lmul2, ta, ma
    vmv.v.i       v16, 0
    vmv.v.i       v20, 0
    vmv.v.i       v24, 0
.if \x == 4
    vmv.v.i       v28, 0
.endif

1:  // row loop
    mv             t0, a0   // current fenc
    mv             t1, a1   // current pix0
    mv             t2, a2   // current pix1
    mv             t3, a3   // current pix2
.if \x == 4
    mv             t4, a4   // current pix3
.endif
    addi           t6, t6, -1

2:  // inline segment loop
    vsetvli        a7, t5, e8, m\()\lmul1, ta, ma

    vle8.v         v0, (t0)
    vle8.v         v1, (t1)
    add            t0, t0, a7
    add            t1, t1, a7
    VABDU          v5, v0, v1, v9

    vle8.v         v2, (t2)
    vle8.v         v3, (t3)
    add            t2, t2, a7
    add            t3, t3, a7
    VABDU          v6, v0, v2, v10
    VABDU          v7, v0, v3, v11
.if \x == 4
    vle8.v         v4, (t4)
    add            t4, t4, a7
    VABDU          v8, v0, v4, v12
.endif

    vsetvli        zero, a7, e32, m\()\lmul2, tu, ma

    vzext.vf4      v12, v5
    vadd.vv        v16, v16, v12

    vzext.vf4      v0, v6
    vadd.vv        v20, v20, v0

    vzext.vf4      v4, v7
    vadd.vv        v24, v24, v4
.if \x == 4
    vzext.vf4      v12, v8
    vadd.vv        v28, v28, v12
.endif

    sub             t5, t5, a7
    bgtz            t5, 2b

    // update row pointer
    addi            a0, a0, FENC_STRIDE
    add             a1, a1, a5
    add             a2, a2, a5
    add             a3, a3, a5
.if \x == 4
    add             a4, a4, a5
.endif

    li              t5, \w          // reset the inline element counter
    bgtz            t6, 1b

    // reduce and store the result
    li             t5, \w
    vsetvli        zero, t5, e32, m\()\lmul2, ta, ma
    vmv.v.i        v0, 0
    vredsum.vs     v0, v16, v0
    vmv.x.s        t0, v0
    sw             t0, 0(a6)      // scores[0]

    vmv.v.i        v4, 0
    vredsum.vs     v4, v20, v4
    vmv.x.s        t1, v4
    sw             t1, 4(a6)      // scores[1]

    vmv.v.i        v8, 0
    vredsum.vs     v8, v24, v8
    vmv.x.s        t5, v8
    sw             t5, 8(a6)      // scores[2]

.if \x == 4
    vmv.v.i       v12, 0
    vredsum.vs    v12, v28, v12
    vmv.x.s       t6, v12

    sw            t6, 12(a6)      // scores[3]
.endif
    ret
endfunc
.endm

SAD_X_FUNC_COMMON 3, 24, 32, 1, 4

SAD_X_FUNC_COMMON 3, 32, 8, 1, 4
SAD_X_FUNC_COMMON 3, 32, 16, 1, 4
SAD_X_FUNC_COMMON 3, 32, 24, 1, 4
SAD_X_FUNC_COMMON 3, 32, 32, 1, 4
SAD_X_FUNC_COMMON 3, 32, 64, 1, 4

SAD_X_FUNC_COMMON 3, 48, 64, 1, 4

SAD_X_FUNC_COMMON 3, 64, 16, 1, 4
SAD_X_FUNC_COMMON 3, 64, 32, 1, 4
SAD_X_FUNC_COMMON 3, 64, 48, 1, 4
SAD_X_FUNC_COMMON 3, 64, 64, 1, 4

SAD_X_FUNC_COMMON 4, 24, 32, 1, 4

SAD_X_FUNC_COMMON 4, 32, 8, 1, 4
SAD_X_FUNC_COMMON 4, 32, 16, 1, 4
SAD_X_FUNC_COMMON 4, 32, 24, 1, 4
SAD_X_FUNC_COMMON 4, 32, 32, 1, 4
SAD_X_FUNC_COMMON 4, 32, 64, 1, 4

SAD_X_FUNC_COMMON 4, 48, 64, 1, 4

SAD_X_FUNC_COMMON 4, 64, 16, 1, 4
SAD_X_FUNC_COMMON 4, 64, 32, 1, 4
SAD_X_FUNC_COMMON 4, 64, 48, 1, 4
SAD_X_FUNC_COMMON 4, 64, 64, 1, 4

#else  // HIGH_BIT_DEPTH

.macro SAD_FUNC_SMALL w, h, lmul1, lmul2
function PFX(pixel_sad_\w\()x\h\()_rvv)
    li            t0, \h
    li            t1, \w

    vsetvli       zero, t1, e32, m\()\lmul2, ta, ma
    vmv.v.i       v28, 0

    STRIDE_SHIFT  t2, t3, a1, a3, 1, 1

1:
    addi          t0, t0, -4
    vsetvli       zero, t1, e16, m\()\lmul1, ta, ma

    vle16.v       v0, (a0)
    vle16.v       v1, (a2)
    add           a0, a0, t2
    add           a2, a2, t3
    UWADBDACCU    v28, v0, v1, v16, v17

    vle16.v       v2, (a0)
    vle16.v       v3, (a2)
    add           a0, a0, t2
    add           a2, a2, t3
    UWADBDACCU    v28, v2, v3, v18, v19

    vle16.v       v4, (a0)
    vle16.v       v5, (a2)
    add           a0, a0, t2
    add           a2, a2, t3
    UWADBDACCU    v28, v4, v5, v20, v21

    vle16.v       v6, (a0)
    vle16.v       v7, (a2)
    add           a0, a0, t2
    add           a2, a2, t3
    UWADBDACCU    v28, v6, v7, v22, v23

    bnez          t0, 1b

    vsetvli       zero, t1, e32, m\()\lmul2, ta, ma
    vmv.v.i       v24, 0
    vredsum.vs    v24, v28, v24
    vmv.x.s       a0, v24
    ret
endfunc
.endm

SAD_FUNC_SMALL 4, 4, 1, 1
SAD_FUNC_SMALL 4, 8, 1, 1
SAD_FUNC_SMALL 4, 16, 1, 1

SAD_FUNC_SMALL 8, 4, 1, 2
SAD_FUNC_SMALL 8, 8, 1, 2
SAD_FUNC_SMALL 8, 16, 1, 2
SAD_FUNC_SMALL 8, 32, 1, 2

.macro SAD_FUNC_MEDIUM w, h, lmul1, lmul2
function PFX(pixel_sad_\w\()x\h\()_rvv)
    li           t0, \h
    li           t1, \w

    vsetvli      zero, t1, e32, m\()\lmul2, ta, ma
    vmv.v.i      v24, 0

    STRIDE_SHIFT t2, t3, a1, a3, 1, 1

1:
    addi          t0, t0, -2
    vsetvli       zero, t1, e16, m\()\lmul1, ta, ma

    vle16.v       v0, (a0)
    vle16.v       v2, (a2)
    add           a0, a0, t2
    add           a2, a2, t3
    UWADBDACCU    v24, v0, v2, v12, v14

    vle16.v       v4, (a0)
    vle16.v       v6, (a2)
    add           a0, a0, t2
    add           a2, a2, t3
    UWADBDACCU    v24, v4, v6, v18, v20

    bnez          t0, 1b

    vsetvli       zero, t1, e32, m\()\lmul2, ta, ma
    vmv.v.i       v0, 0
    vredsum.vs    v0, v24, v0
    vmv.x.s       a0, v0
    ret
endfunc
.endm

SAD_FUNC_MEDIUM 12, 16, 2, 8

SAD_FUNC_MEDIUM 16, 4, 2, 8
SAD_FUNC_MEDIUM 16, 8, 2, 8
SAD_FUNC_MEDIUM 16, 12, 2, 8
SAD_FUNC_MEDIUM 16, 16, 2, 8
SAD_FUNC_MEDIUM 16, 32, 2, 8
SAD_FUNC_MEDIUM 16, 64, 2, 8

.macro SAD_FUNC_COMMON w, h, lmul1, lmul2
function PFX(pixel_sad_\w\()x\h\()_rvv)
    li             t0, \h
    li             t1, \w

    vsetvli        zero, t1, e32, m\()\lmul2, ta, ma
    vmv.v.i        v24, 0

1:  // row loop
    mv             t3, a0
    mv             t4, a2
    addi           t0, t0, -1

2:  // inline segment loop
    vsetvli         t5, t1, e16, m\()\lmul1, ta, ma

    vle16.v         v0, (t3)
    vle16.v         v4, (t4)
    ADDPTR          a6, a7, t3, t4, t5, t5, 1, 1
    UWABDU          v16, v0, v4, v8, v12

    vsetvli         zero, t5, e32, m\()\lmul2, tu, ma
    vadd.vv         v24, v24, v16

    sub             t1, t1, t5
    bgtz            t1, 2b

    // update row pointer
    ADDPTR          a6, a7, a0, a2, a1, a3, 1, 1

    li              t1, \w
    bgtz            t0, 1b

    li              t1, \w
    vsetvli         zero, t1, e32, m\()\lmul2, ta, ma
    vmv.v.i         v0, 0
    vredsum.vs      v0, v24, v0
    vmv.x.s         a0, v0
    ret
endfunc
.endm

SAD_FUNC_COMMON  24, 32, 1 , 2

SAD_FUNC_COMMON  32, 8, 1 , 2
SAD_FUNC_COMMON  32, 16, 1 , 2
SAD_FUNC_COMMON  32, 24, 1 , 2
SAD_FUNC_COMMON  32, 32, 1 , 2
SAD_FUNC_COMMON  32, 64, 1 , 2

SAD_FUNC_COMMON  48, 64, 1 , 2

SAD_FUNC_COMMON  64, 16, 1 , 2
SAD_FUNC_COMMON  64, 32, 1 , 2
SAD_FUNC_COMMON  64, 48, 1 , 2
SAD_FUNC_COMMON  64, 64, 1 , 2

//============= SAD_X3 and SAD_X4 code start========================
// static void x264_pixel_sad_x3_##size(pixel *fenc, pixel *pix0, pixel *pix1, pixel *pix2, intptr_t i_stride, int scores[3])
// static void x264_pixel_sad_x4_##size(pixel *fenc, pixel *pix0, pixel *pix1,pixel *pix2, pixel *pix3, intptr_t i_stride, int scores[4])
.macro SAD_X_FUNC_SMALL x, w, h, lmul1, lmul2
function PFX(sad_x\x\()_\w\()x\h\()_rvv)
// Make function arguments for x == 3 look like x == 4.
.if \x == 3
    mv             t0, a4
    mv             t1, a5
    mv             a5, t0                // a5 = i_stride
    mv             a6, t1                // a6 = scores
.endif

    // Stride is given in terms of pixel channel size, so double to get number of bytes.
    add            a5, a5, a5
    li             t2, FDNC_STRIDE

    li             t3, \w
    li             t4, \h

    vsetvli        zero, t3, e32, m\()\lmul2, ta, ma
    vmv.v.i        v24, 0
    vmv.v.i        v26, 0
    vmv.v.i        v28, 0
.if \x == 4
    vmv.v.i        v30, 0
.endif

1:
    addi           t4, t4, -1
    vsetvli        zero, t3, e16, m\()\lmul1, ta, ma

    vle16.v        v0, (a0)
    vle16.v        v1, (a1)
    add            a0, a0, t2
    add            a1, a1, a5
    UWADBDACCU     v24, v0, v1, v14, v15

    vle16.v        v2, (a2)
    vle16.v        v3, (a3)
    add            a2, a2, a5
    add            a3, a3, a5
    UWADBDACCU     v26, v0, v2, v16, v17
    UWADBDACCU     v28, v0, v3, v18, v19
.if \x == 4
    vle16.v        v4, (a4)
    add            a4, a4, a5
    UWADBDACCU     v30, v0, v4, v20, v21
.endif

    bnez           t4, 1b

    // reduce and store the result
    vsetvli        zero, t3, e32, m\()\lmul2, ta, ma
    vmv.v.i        v8, 0
    vredsum.vs     v8, v24, v8
    vmv.x.s        t0, v8
    sw             t0, 0(a6)      // scores[0]

    vmv.v.i        v12, 0
    vredsum.vs     v12, v26, v12
    vmv.x.s        t1, v12
    sw             t1, 4(a6)      // scores[1]

    vmv.v.i        v4, 0
    vredsum.vs     v4, v28, v4
    vmv.x.s        t5, v4
    sw             t5, 8(a6)      // scores[2]

.if \x == 4
    vmv.v.i       v0, 0
    vredsum.vs    v0, v30, v0
    vmv.x.s       t6, v0

    sw            t6, 12(a6)      // scores[3]
.endif
    ret
endfunc
.endm

SAD_X_FUNC_SMALL 3, 4, 4, 1, 1
SAD_X_FUNC_SMALL 3, 4, 8, 1, 1
SAD_X_FUNC_SMALL 3, 4, 16, 1, 1

SAD_X_FUNC_SMALL 3, 8, 4, 1, 2
SAD_X_FUNC_SMALL 3, 8, 8, 1, 2
SAD_X_FUNC_SMALL 3, 8, 16, 1, 2
SAD_X_FUNC_SMALL 3, 8, 32, 1, 2

SAD_X_FUNC_SMALL 4, 4, 4, 1, 1
SAD_X_FUNC_SMALL 4, 4, 8, 1, 1
SAD_X_FUNC_SMALL 4, 4, 16, 1, 1
SAD_X_FUNC_SMALL 4, 8, 4, 1, 2

SAD_X_FUNC_SMALL 4, 8, 8, 1, 2
SAD_X_FUNC_SMALL 4, 8, 16, 1, 2
SAD_X_FUNC_SMALL 4, 8, 32, 1, 2

.macro SAD_X_FUNC_COMMON, x, w, h, lmul1, lmul2
function PFX(sad_x\x\()_\w\()x\h\()_rvv)
    push

// Make function arguments for x == 3 look like x == 4.
.if \x == 3
    mv             t0, a4
    mv             t1, a5
    mv             a5, t0
    mv             a6, t1
.endif

    // Stride is given in terms of pixel channel size, so double to get number of bytes.
    add            a5, a5, a5

    li             t5, \w
    li             t6, \h

    vsetvli       zero, t5, e32, m\()\lmul2, ta, ma
    vmv.v.i       v22, 0
    vmv.v.i       v24, 0
    vmv.v.i       v26, 0
.if \x == 4
    vmv.v.i       v28, 0
.endif

1:  // row loop
    mv             t0, a0   // current fenc
    mv             t1, a1   // current pix0
    mv             t2, a2   // current pix1
    mv             t3, a3   // current pix2
.if \x == 4
    mv             t4, a4   // current pix3
.endif
    addi           t6, t6, -1

2:  // inline segment loop
    vsetvli        a7, t5, e16, m\()\lmul1, ta, ma
    slli           x9, a7, 1

    vle16.v        v0, (t0)
    vle16.v        v1, (t1)
    add            t0, t0, x9
    add            t1, t1, x9
    UWABDU         v6, v0, v1, v14, v16

    vle16.v        v2, (t2)
    vle16.v        v3, (t3)
    add            t2, t2, x9
    add            t3, t3, x9
    UWABDU         v8, v0, v2, v18, v20
    UWABDU         v10, v0, v3, v14, v16
.if \x == 4
    vle16.v        v4, (t4)
    add            t4, t4, x9
    UWABDU         v12, v0, v4, v18, v20
.endif

    vsetvli        zero, a7, e32,  m\()\lmul2, tu, ma

    vadd.vv        v22, v22, v6
    vadd.vv        v24, v24, v8
    vadd.vv        v26, v26, v10
.if \x == 4
    vadd.vv        v28, v28, v12
.endif

    sub             t5, t5, a7
    bgtz            t5, 2b

    // update row pointer
    addi            a0, a0, FDNC_STRIDE
    add             a1, a1, a5
    add             a2, a2, a5
    add             a3, a3, a5
.if \x == 4
    add             a4, a4, a5
.endif

    li              t5, \w          // reset the inline element counter
    bgtz            t6, 1b

    // reduce and store the result
    li             t5, \w
    vsetvli        zero, t5, e32, m\()\lmul2, ta, ma
    vmv.v.i        v0, 0
    vredsum.vs     v0, v22, v0
    vmv.x.s        t0, v0
    sw             t0, 0(a6)      // scores[0]

    vmv.v.i        v4, 0
    vredsum.vs     v4, v24, v4
    vmv.x.s        t1, v4
    sw             t1, 4(a6)      // scores[1]

    vmv.v.i        v8, 0
    vredsum.vs     v8, v26, v8
    vmv.x.s        t5, v8
    sw             t5, 8(a6)      // scores[2]

.if \x == 4
    vmv.v.i       v12, 0
    vredsum.vs    v12, v28, v12
    vmv.x.s       t6, v12

    sw            t6, 12(a6)      // scores[3]
.endif
    pop
    ret
endfunc
.endm

SAD_X_FUNC_COMMON 3, 12, 16, 1, 2

SAD_X_FUNC_COMMON 3, 16, 4, 1, 2
SAD_X_FUNC_COMMON 3, 16, 8, 1, 2
SAD_X_FUNC_COMMON 3, 16, 12, 1, 2
SAD_X_FUNC_COMMON 3, 16, 16, 1, 2
SAD_X_FUNC_COMMON 3, 16, 32, 1, 2
SAD_X_FUNC_COMMON 3, 16, 64, 1, 2

SAD_X_FUNC_COMMON 3, 24, 32, 1, 2

SAD_X_FUNC_COMMON 3, 32, 8, 1, 2
SAD_X_FUNC_COMMON 3, 32, 16, 1, 2
SAD_X_FUNC_COMMON 3, 32, 24, 1, 2
SAD_X_FUNC_COMMON 3, 32, 32, 1, 2
SAD_X_FUNC_COMMON 3, 32, 64, 1, 2

SAD_X_FUNC_COMMON 3, 48, 64, 1, 2

SAD_X_FUNC_COMMON 3, 64, 16, 1, 2
SAD_X_FUNC_COMMON 3, 64, 32, 1, 2
SAD_X_FUNC_COMMON 3, 64, 48, 1, 2
SAD_X_FUNC_COMMON 3, 64, 64, 1, 2


SAD_X_FUNC_COMMON 4, 12, 16, 1, 2

SAD_X_FUNC_COMMON 4, 16, 4, 1, 2
SAD_X_FUNC_COMMON 4, 16, 8, 1, 2
SAD_X_FUNC_COMMON 4, 16, 12, 1, 2
SAD_X_FUNC_COMMON 4, 16, 16, 1, 2
SAD_X_FUNC_COMMON 4, 16, 32, 1, 2
SAD_X_FUNC_COMMON 4, 16, 64, 1, 2

SAD_X_FUNC_COMMON 4, 24, 32, 1, 2

SAD_X_FUNC_COMMON 4, 32, 8, 1, 2
SAD_X_FUNC_COMMON 4, 32, 16, 1, 2
SAD_X_FUNC_COMMON 4, 32, 24, 1, 2
SAD_X_FUNC_COMMON 4, 32, 32, 1, 2
SAD_X_FUNC_COMMON 4, 32, 64, 1, 2

SAD_X_FUNC_COMMON 4, 48, 64, 1, 2

SAD_X_FUNC_COMMON 4, 64, 16, 1, 2
SAD_X_FUNC_COMMON 4, 64, 32, 1, 2
SAD_X_FUNC_COMMON 4, 64, 48, 1, 2
SAD_X_FUNC_COMMON 4, 64, 64, 1, 2

#endif // !HIGH_BIT_DEPTH