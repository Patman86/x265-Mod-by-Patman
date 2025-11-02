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

#if BIT_DEPTH == 8

// To be optimized
.macro SAD_FUNC w, h
function PFX(pixel_sad_\w\()x\h\()_rvv)
    li t1,        \w
    li t2,        \h
    // prevent overflow when there is a bigger \h
    vsetvli      zero, t1, e32, m2, ta, ma
    vmv.v.i       v16, 0  // v16-v17

1:
    vsetvli      zero, t1, e8, m1, ta, ma
    vle8.v        v0, (a0)
    vle8.v        v1, (a2)

    //vssubu.vv      v2, v0, v1   // v2 = max(pix1 - pix2, 0)
    //vssubu.vv      v3, v1, v0   // v3 = max(pix2 - pix1, 0)
    //vadd.vv        v4, v2, v3   // v4 = |pix1 - pix2|
    vmaxu.vv      v2, v0, v1      // max(pix1, pix2)
    vminu.vv      v3, v0, v1      // min(pix1, pix2)
    vsub.vv       v4, v2, v3      // |pix1 - pix2|

    vsetvli      zero, t1, e32, m2, ta, ma
    vzext.vf4     v6, v4
    vadd.vv       v16, v16, v6

    add           a0, a0, a1
    add           a2, a2, a3
    addi          t2, t2, -1
    bnez          t2, 1b

    vmv.v.i       v8, 0
    vredsum.vs    v8, v16, v8
    vmv.x.s       a0, v8
    ret
endfunc
.endm

.macro SAD_FUNC_LOOP_16x_OPT h
    mv      t0, a0
    mv      t1, a2
    li      t2, \h / 4
    li      a5, 0

    // initialize 32-bit accumulator (to prevent overflow)
    vsetivli      zero, 16, e32, m4, ta, ma
    vmv.v.i       v24, 0   // v24-v27

1:
    // set 8-bit processing mode (fixed processing of 16 elements at a time)
    vsetivli    zero, 16, e8, m1, ta, ma
    // row 1
    vle8.v      v0, (t0)
    add         t3, t0, a1
    vle8.v      v1, (t1)
    add         t4, t1, a3

    vmaxu.vv    v9, v0, v1
    vminu.vv    v10, v0, v1
    vsub.vv     v8, v9, v10  // |pix1 - pix2|

    // row 2
    vle8.v      v2, (t3)
    add         t3, t3, a1
    vle8.v      v3, (t4)
    add         t4, t4, a3

    vmaxu.vv    v11, v2, v3
    vminu.vv    v13, v2, v3
    vsub.vv     v12, v11, v13

    // row 3
    vle8.v      v4, (t3)
    add         t3, t3, a1
    vle8.v      v5, (t4)
    add         t4, t4, a3

    vmaxu.vv    v14, v4, v5
    vminu.vv    v15, v4, v5
    vsub.vv     v16, v14, v15

    // row 4
    vle8.v      v6, (t3)
    vle8.v      v7, (t4)

    vmaxu.vv    v17, v6, v7
    vminu.vv    v18, v6, v7
    vsub.vv     v4, v17, v18

    // expand the 8-bit difference to 32-bit and accumulate
    vsetivli    zero, 16, e32, m4, ta, ma

    vzext.vf4   v20, v8
    vadd.vv     v24, v24, v20

    vzext.vf4   v28, v12
    vadd.vv     v24, v24, v28

    vzext.vf4   v20, v16
    vadd.vv     v24, v24, v20

    vzext.vf4   v28, v4
    vadd.vv     v24, v24, v28

    // update pointer
    //add      t0, t0, a1, lsl 2  // pix1 += stride*4
    //add      t1, t1, a3, lsl 2  // pix2 += stride*4
    slli     t6, a1, 2
    add      t0, t0, t6
    slli     t6, a3, 2
    add      t1, t1, t6

    addi      t2, t2, -1
    bgtz      t2, 1b

    vsetivli      zero, 16, e32, m4, ta, ma
    vmv.v.i       v0, 0
    vredsum.vs    v0, v24, v0
    vmv.x.s       a0, v0
    ret
.endm

.macro SAD_FUNC_LOOP w, h
function PFX(pixel_sad_\w\()x\h\()_rvv)
.if \w == 16
    SAD_FUNC_LOOP_16x_OPT \h  // loop unrolling
.else
    li           t2, \h  // height
    li           t3, \w  // weight
    li           t6, 0

1:  // row loop
    // inline segmentation processing
    mv      t4, a0   // current pix1
    mv      t5, a2   // current pix2

2:  // inline segment loop
    vsetvli          a7, t3, e8, m1, ta, ma  // the max number of elements processed at one time.

    vle8.v           v0, (t4)
    add              t4, t4, a7
    vle8.v           v1, (t5)
    add              t5, t5, a7

    vmaxu.vv         v2, v0, v1
    vminu.vv         v3, v0, v1
    vsub.vv          v4, v2, v3

    vsetvli         zero, a7, e16, m2, ta, ma
    vzext.vf2       v8, v4  // zero-extend to 16-bit
    vmv.v.i         v12, 0
    vredsum.vs      v12, v8, v12
    vmv.x.s         t0, v12
    add             t6, t6, t0  // add the result to the return register

    sub             t3, t3, a7
    bgtz            t3, 2b

    // update row pointer
    add      a0, a0, a1  // pix1 += stride_pix1
    add      a2, a2, a3  // pix2 += stride_pix2

    li        t3, \w    // reset the inline element counter
    addi      t2, t2, -1
    bgtz      t2, 1b
    mv        a0, t6
    ret
.endif
endfunc
.endm

SAD_FUNC  4,  4
SAD_FUNC  4,  8
SAD_FUNC  4,  16
SAD_FUNC  8,  4
SAD_FUNC  8,  8
SAD_FUNC  8,  16
SAD_FUNC  8,  32

SAD_FUNC_LOOP  16, 4
SAD_FUNC_LOOP  16, 8
SAD_FUNC_LOOP  16, 12
SAD_FUNC_LOOP  16, 16
SAD_FUNC_LOOP  16, 32
SAD_FUNC_LOOP  16, 64
SAD_FUNC_LOOP  32, 8
SAD_FUNC_LOOP  32, 16
SAD_FUNC_LOOP  32, 24
SAD_FUNC_LOOP  32, 32
SAD_FUNC_LOOP  32, 64
SAD_FUNC_LOOP  64, 16
SAD_FUNC_LOOP  64, 32
SAD_FUNC_LOOP  64, 48
SAD_FUNC_LOOP  64, 64
SAD_FUNC_LOOP  12, 16
SAD_FUNC_LOOP  24, 32
SAD_FUNC_LOOP  48, 64

//============= SAD_X3 and SAD_X4 code start========================
// static void x264_pixel_sad_x3_##size(pixel *fenc, pixel *pix0, pixel *pix1, pixel *pix2, intptr_t i_stride, int scores[3])
// static void x264_pixel_sad_x4_##size(pixel *fenc, pixel *pix0, pixel *pix1,pixel *pix2, pixel *pix3, intptr_t i_stride, int scores[4])
.macro SAD_X_FUNC x, w, h
function PFX(sad_x\x\()_\w\()x\h\()_rvv)
// Make function arguments for x == 3 look like x == 4.
.if \x == 3
    mv      t0, a4
    mv      t1, a5
    mv      a5, t0                // a5 = i_stride
    mv      a6, t1                // a6 = scores
.endif
    li      t2, FENC_STRIDE       // FENC_STRIDE = 64
    li      t3, \w
    li      t4, \h

    vsetvli      zero, t3, e32, m2, ta, ma
    vmv.v.i       v24, 0
    vmv.v.i       v26, 0
    vmv.v.i       v28, 0
.if \x == 4
    vmv.v.i       v30, 0
.endif

// main loop (processing each row)
1:
    vsetvli       zero, t3, e8, m1, ta, ma
    vle8.v         v0, (a0)
    vle8.v         v1, (a1)
    vle8.v         v2, (a2)
    vle8.v         v3, (a3)
.if \x == 4
    vle8.v         v4, (a4)
.endif

   // |fenc - pix0|
    vmaxu.vv      v5, v0, v1  // max(fenc, pix0)
    vminu.vv      v6, v0, v1  // min(fenc, pix0)
    vsub.vv       v8, v5, v6  // |fenc - pix0|

    vmaxu.vv      v7, v0, v2
    vminu.vv      v9, v0, v2
    vsub.vv       v10, v7, v9

    vmaxu.vv      v11, v0, v3
    vminu.vv      v13, v0, v3
    vsub.vv       v12, v11, v13

.if \x == 4
    vmaxu.vv      v14, v0, v4
    vminu.vv      v15, v0, v4
    vsub.vv       v16, v14, v15
.endif

    vsetvli       zero, t3, e32, m2, ta, ma
    vzext.vf4     v18, v8
    vadd.vv       v24, v24, v18

    vzext.vf4     v20, v10
    vadd.vv       v26, v26, v20

    vzext.vf4     v22, v12
    vadd.vv       v28, v28, v22

.if \x == 4
    vzext.vf4     v2, v16  // register reuse
    vadd.vv       v30, v30, v2
.endif

    add     a0, a0, t2  // fenc += FENC_STRIDE
    add     a1, a1, a5  // pix0 += i_stride
    add     a2, a2, a5  // pix1 += i_stride
    add     a3, a3, a5  // pix2 += i_stride
.if \x == 4
    add     a4, a4, a5  // pix3 += i_stride
.endif
    addi    t4, t4, -1
    bnez    t4, 1b

    // reduce and store the result
    vsetvli        zero, t3, e32, m2, ta, ma    // 32 bit = 4 byte
    vmv.v.i        v0, 0
    vredsum.vs     v0, v24, v0
    vmv.x.s        t0, v0
    sw             t0, 0(a6)      // scores[0]

    vmv.v.i        v2, 0
    vredsum.vs     v2, v26, v2
    vmv.x.s        t1, v2
    sw             t1, 4(a6)      // scores[1]

    vmv.v.i        v4, 0
    vredsum.vs     v4, v28, v4
    vmv.x.s        t5, v4
    sw             t5, 8(a6)      // scores[2]

.if \x == 4
    vmv.v.i       v6, 0
    vredsum.vs    v6, v30, v6
    vmv.x.s       t6, v6
    sw            t6, 12(a6)     // scores[3]
.endif
    ret
endfunc
.endm

.macro SAD_X_LOOP x, w, h
function PFX(sad_x\x\()_\w\()x\h\()_rvv)

    // push
    addi      sp, sp, -176
    sd        ra, 144(sp)    // save return address
    sd        s0, 160(sp)    // save the frame pointer
    addi      s0, sp, 176    // set the new frame pointer

    sd        x9, 0(sp)
    sd        x20, 16(sp)
    sd        x21, 32(sp)
    sd        x22, 48(sp)
    sd        x23, 64(sp)
    sd        x24, 80(sp)
    sd        x25, 96(sp)
    sd        x26, 112(sp)
    sd        x27, 128(sp)

// make function arguments for x == 3 look like x == 4.
.if \x == 3
    mv            t0, a4
    mv            t1, a5
    mv            a5, t0                  // a5 = i_stride
    mv            a6, t1                  // a6 = scores
.endif
    li            x9, FENC_STRIDE         // FENC_STRIDE = 64
    li            t5, \w
    li            t6, \h
    // store the tmp result
    li            x20, 0
    li            x21, 0
    li            x22, 0
.if \x == 4
    li            x23, 0
.endif

1:  // row loop
    // inline segmentation processing
    mv           t0, a0   // current fenc
    mv           t1, a1   // current pix0
    mv           t2, a2   // current pix1
    mv           t3, a3   // current pix2
.if \x == 4
    mv           t4, a4   // current pix3
.endif
    //li           x18, 0   // inline offset

2:  // inline segment loop
    vsetvli      a7, t5, e8, m1, ta, ma  // the max number of elements processed at one time.

    vle8.v       v0, (t0)
    vle8.v       v1, (t1)
    add          t0, t0, a7
    add          t1, t1, a7
    vle8.v       v2, (t2)
    vle8.v       v3, (t3)
    add          t2, t2, a7
    add          t3, t3, a7
.if \x == 4
    vle8.v       v4, (t4)
    add          t4, t4, a7
.endif

   // |fenc - pix0|
    vmaxu.vv      v5, v0, v1  // max(fenc, pix0)
    vminu.vv      v6, v0, v1  // min(fenc, pix0)
    vsub.vv       v8, v5, v6  // |fenc - pix0|

    vmaxu.vv      v7, v0, v2
    vminu.vv      v9, v0, v2
    vsub.vv       v10, v7, v9

    vmaxu.vv      v11, v0, v3
    vminu.vv      v13, v0, v3
    vsub.vv       v12, v11, v13

.if \x == 4
    vmaxu.vv      v14, v0, v4
    vminu.vv      v15, v0, v4
    vsub.vv       v16, v14, v15
.endif

    vsetvli         zero, a7, e16, m2, ta, ma
    vzext.vf2       v30, v8  // zero-extend to 16-bit
    vmv.v.i         v18, 0
    vredsum.vs      v18, v30, v18

    vzext.vf2       v28, v10
    vmv.v.i         v20, 0
    vredsum.vs      v20, v28, v20

    vzext.vf2       v30, v12
    vmv.v.i         v22, 0
    vredsum.vs      v22, v30, v22
.if \x == 4
    vzext.vf2       v28, v16
    vmv.v.i         v26, 0
    vredsum.vs      v26, v28, v26
.endif

    vmv.x.s         x24, v18
    add             x20, x20, x24  // add the result to the return register
    vmv.x.s         x25, v20
    add             x21, x21, x25
    vmv.x.s         x26, v22
    add             x22, x22, x26
.if \x == 4
    vmv.x.s         x27, v26
    add             x23, x23, x27
.endif

    sub             t5, t5, a7
    bgtz            t5, 2b

    // update row pointer
    add      a0, a0, x9    // fenc += FENC_STRIDE
    add      a1, a1, a5    // pix0 += i_stride
    add      a2, a2, a5    // pix1 += i_stride
    add      a3, a3, a5    // pix2 += i_stride
.if \x == 4
    add      a4, a4, a5    // pix3 += i_stride
.endif

    li        t5, \w       // reset the inline element counter
    addi      t6, t6, -1
    bgtz      t6, 1b

    sw        x20, 0(a6)      // scores[0]
    sw        x21, 4(a6)      // scores[1]
    sw        x22, 8(a6)      // scores[2]
.if \x == 4
    sw        x23, 12(a6)     // scores[3]
.endif

    // pop
    ld      x27, 128(sp)
    ld      x26, 112(sp)
    ld      x25, 96(sp)
    ld      x24, 80(sp)
    ld      x23, 64(sp)
    ld      x22, 48(sp)
    ld      x21, 32(sp)
    ld      x20, 16(sp)
    ld      x9, 0(sp)

    // restore the frame pointer and return address
    ld      s0, 160(sp)
    ld      ra, 144(sp)
    addi    sp, sp, 176

    ret
endfunc
.endm

SAD_X_FUNC  3, 4,  4
SAD_X_FUNC  3, 4,  8
SAD_X_FUNC  3, 4,  16
SAD_X_FUNC  3, 8,  4
SAD_X_FUNC  3, 8,  8
SAD_X_FUNC  3, 8,  16
SAD_X_FUNC  3, 8,  32
SAD_X_LOOP  3, 12, 16
SAD_X_LOOP  3, 16, 4
SAD_X_LOOP  3, 16, 8
SAD_X_LOOP  3, 16, 12
SAD_X_LOOP  3, 16, 16
SAD_X_LOOP  3, 16, 32
SAD_X_LOOP  3, 16, 64
SAD_X_LOOP  3, 24, 32
SAD_X_LOOP  3, 32, 8
SAD_X_LOOP  3, 32, 16
SAD_X_LOOP  3, 32, 24
SAD_X_LOOP  3, 32, 32
SAD_X_LOOP  3, 32, 64
SAD_X_LOOP  3, 48, 64
SAD_X_LOOP  3, 64, 16
SAD_X_LOOP  3, 64, 32
SAD_X_LOOP  3, 64, 48
SAD_X_LOOP  3, 64, 64

SAD_X_FUNC  4, 4,  4
SAD_X_FUNC  4, 4,  8
SAD_X_FUNC  4, 4,  16
SAD_X_FUNC  4, 8,  4
SAD_X_FUNC  4, 8,  8
SAD_X_FUNC  4, 8,  16
SAD_X_FUNC  4, 8,  32
SAD_X_LOOP  4, 12, 16
SAD_X_LOOP  4, 16, 4
SAD_X_LOOP  4, 16, 8
SAD_X_LOOP  4, 16, 12
SAD_X_LOOP  4, 16, 16
SAD_X_LOOP  4, 16, 32
SAD_X_LOOP  4, 16, 64
SAD_X_LOOP  4, 24, 32
SAD_X_LOOP  4, 32, 8
SAD_X_LOOP  4, 32, 16
SAD_X_LOOP  4, 32, 24
SAD_X_LOOP  4, 32, 32
SAD_X_LOOP  4, 32, 64
SAD_X_LOOP  4, 48, 64
SAD_X_LOOP  4, 64, 16
SAD_X_LOOP  4, 64, 32
SAD_X_LOOP  4, 64, 48
SAD_X_LOOP  4, 64, 64

#endif // !HIGH_BIT_DEPTH
