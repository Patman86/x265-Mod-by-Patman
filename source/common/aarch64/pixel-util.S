/*****************************************************************************
 * Copyright (C) 2020-2021 MulticoreWare, Inc
 *
 * Authors: Yimeng Su <yimeng.su@huawei.com>
 *          Hongbin Liu <liuhongbin1@huawei.com>
 *          Sebastian Pop <spop@amazon.com>
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
#include "pixel-util-common.S"

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4

.text

// void getResidual4_neon(const pixel* fenc, const pixel* pred, int16_t* residual, intptr_t stride)
function PFX(getResidual4_neon)
    lsl             x4, x3, #1
.rept 2
    ld1             {v0.8b}, [x0], x3
    ld1             {v1.8b}, [x1], x3
    ld1             {v2.8b}, [x0], x3
    ld1             {v3.8b}, [x1], x3
    usubl           v4.8h, v0.8b, v1.8b
    usubl           v5.8h, v2.8b, v3.8b
    st1             {v4.8b}, [x2], x4
    st1             {v5.8b}, [x2], x4
.endr
    ret
endfunc

function PFX(getResidual8_neon)
    lsl             x4, x3, #1
.rept 4
    ld1             {v0.8b}, [x0], x3
    ld1             {v1.8b}, [x1], x3
    ld1             {v2.8b}, [x0], x3
    ld1             {v3.8b}, [x1], x3
    usubl           v4.8h, v0.8b, v1.8b
    usubl           v5.8h, v2.8b, v3.8b
    st1             {v4.16b}, [x2], x4
    st1             {v5.16b}, [x2], x4
.endr
    ret
endfunc

function PFX(getResidual16_neon)
    lsl             x4, x3, #1
.rept 8
    ld1             {v0.16b}, [x0], x3
    ld1             {v1.16b}, [x1], x3
    ld1             {v2.16b}, [x0], x3
    ld1             {v3.16b}, [x1], x3
    usubl           v4.8h, v0.8b, v1.8b
    usubl2          v5.8h, v0.16b, v1.16b
    usubl           v6.8h, v2.8b, v3.8b
    usubl2          v7.8h, v2.16b, v3.16b
    st1             {v4.8h-v5.8h}, [x2], x4
    st1             {v6.8h-v7.8h}, [x2], x4
.endr
    ret
endfunc

function PFX(getResidual32_neon)
    lsl             x4, x3, #1
    mov             w12, #4
.Loop_residual_32:
    sub             w12, w12, #1
.rept 4
    ld1             {v0.16b-v1.16b}, [x0], x3
    ld1             {v2.16b-v3.16b}, [x1], x3
    ld1             {v4.16b-v5.16b}, [x0], x3
    ld1             {v6.16b-v7.16b}, [x1], x3
    usubl           v16.8h, v0.8b, v2.8b
    usubl2          v17.8h, v0.16b, v2.16b
    usubl           v18.8h, v1.8b, v3.8b
    usubl2          v19.8h, v1.16b, v3.16b
    usubl           v20.8h, v4.8b, v6.8b
    usubl2          v21.8h, v4.16b, v6.16b
    usubl           v22.8h, v5.8b, v7.8b
    usubl2          v23.8h, v5.16b, v7.16b
    st1             {v16.8h-v19.8h}, [x2], x4
    st1             {v20.8h-v23.8h}, [x2], x4
.endr
    cbnz            w12, .Loop_residual_32
    ret
endfunc

// void pixel_sub_ps_neon(int16_t* a, intptr_t dstride, const pixel* b0, const pixel* b1, intptr_t sstride0, intptr_t sstride1)
function PFX(pixel_sub_ps_4x4_neon)
    lsl             x1, x1, #1
.rept 2
    ld1             {v0.8b}, [x2], x4
    ld1             {v1.8b}, [x3], x5
    ld1             {v2.8b}, [x2], x4
    ld1             {v3.8b}, [x3], x5
    usubl           v4.8h, v0.8b, v1.8b
    usubl           v5.8h, v2.8b, v3.8b
    st1             {v4.4h}, [x0], x1
    st1             {v5.4h}, [x0], x1
.endr
    ret
endfunc

function PFX(pixel_sub_ps_8x8_neon)
    lsl             x1, x1, #1
.rept 4
    ld1             {v0.8b}, [x2], x4
    ld1             {v1.8b}, [x3], x5
    ld1             {v2.8b}, [x2], x4
    ld1             {v3.8b}, [x3], x5
    usubl           v4.8h, v0.8b, v1.8b
    usubl           v5.8h, v2.8b, v3.8b
    st1             {v4.8h}, [x0], x1
    st1             {v5.8h}, [x0], x1
.endr
    ret
endfunc

function PFX(pixel_sub_ps_16x16_neon)
    lsl             x1, x1, #1
.rept 8
    ld1             {v0.16b}, [x2], x4
    ld1             {v1.16b}, [x3], x5
    ld1             {v2.16b}, [x2], x4
    ld1             {v3.16b}, [x3], x5
    usubl           v4.8h, v0.8b, v1.8b
    usubl2          v5.8h, v0.16b, v1.16b
    usubl           v6.8h, v2.8b, v3.8b
    usubl2          v7.8h, v2.16b, v3.16b
    st1             {v4.8h-v5.8h}, [x0], x1
    st1             {v6.8h-v7.8h}, [x0], x1
.endr
    ret
endfunc

function PFX(pixel_sub_ps_32x32_neon)
    lsl             x1, x1, #1
    mov             w12, #4
.Loop_sub_ps_32:
    sub             w12, w12, #1
.rept 4
    ld1             {v0.16b-v1.16b}, [x2], x4
    ld1             {v2.16b-v3.16b}, [x3], x5
    ld1             {v4.16b-v5.16b}, [x2], x4
    ld1             {v6.16b-v7.16b}, [x3], x5
    usubl           v16.8h, v0.8b, v2.8b
    usubl2          v17.8h, v0.16b, v2.16b
    usubl           v18.8h, v1.8b, v3.8b
    usubl2          v19.8h, v1.16b, v3.16b
    usubl           v20.8h, v4.8b, v6.8b
    usubl2          v21.8h, v4.16b, v6.16b
    usubl           v22.8h, v5.8b, v7.8b
    usubl2          v23.8h, v5.16b, v7.16b
    st1             {v16.8h-v19.8h}, [x0], x1
    st1             {v20.8h-v23.8h}, [x0], x1
.endr
    cbnz            w12, .Loop_sub_ps_32
    ret
endfunc

function PFX(pixel_sub_ps_64x64_neon)
    lsl             x1, x1, #1
    sub             x1, x1, #64
    mov             w12, #16
.Loop_sub_ps_64:
    sub             w12, w12, #1
.rept 4
    ld1             {v0.16b-v3.16b}, [x2], x4
    ld1             {v4.16b-v7.16b}, [x3], x5
    usubl           v16.8h, v0.8b, v4.8b
    usubl2          v17.8h, v0.16b, v4.16b
    usubl           v18.8h, v1.8b, v5.8b
    usubl2          v19.8h, v1.16b, v5.16b
    usubl           v20.8h, v2.8b, v6.8b
    usubl2          v21.8h, v2.16b, v6.16b
    usubl           v22.8h, v3.8b, v7.8b
    usubl2          v23.8h, v3.16b, v7.16b
    st1             {v16.8h-v19.8h}, [x0], #64
    st1             {v20.8h-v23.8h}, [x0], x1
.endr
    cbnz            w12, .Loop_sub_ps_64
    ret
endfunc

// chroma sub_ps
function PFX(pixel_sub_ps_4x8_neon)
    lsl             x1, x1, #1
.rept 4
    ld1             {v0.8b}, [x2], x4
    ld1             {v1.8b}, [x3], x5
    ld1             {v2.8b}, [x2], x4
    ld1             {v3.8b}, [x3], x5
    usubl           v4.8h, v0.8b, v1.8b
    usubl           v5.8h, v2.8b, v3.8b
    st1             {v4.4h}, [x0], x1
    st1             {v5.4h}, [x0], x1
.endr
    ret
endfunc

function PFX(pixel_sub_ps_8x16_neon)
    lsl             x1, x1, #1
.rept 8
    ld1             {v0.8b}, [x2], x4
    ld1             {v1.8b}, [x3], x5
    ld1             {v2.8b}, [x2], x4
    ld1             {v3.8b}, [x3], x5
    usubl           v4.8h, v0.8b, v1.8b
    usubl           v5.8h, v2.8b, v3.8b
    st1             {v4.8h}, [x0], x1
    st1             {v5.8h}, [x0], x1
.endr
    ret
endfunc

function PFX(pixel_sub_ps_16x32_neon)
    lsl             x1, x1, #1
.rept 16
    ld1             {v0.16b}, [x2], x4
    ld1             {v1.16b}, [x3], x5
    ld1             {v2.16b}, [x2], x4
    ld1             {v3.16b}, [x3], x5
    usubl           v4.8h, v0.8b, v1.8b
    usubl2          v5.8h, v0.16b, v1.16b
    usubl           v6.8h, v2.8b, v3.8b
    usubl2          v7.8h, v2.16b, v3.16b
    st1             {v4.8h-v5.8h}, [x0], x1
    st1             {v6.8h-v7.8h}, [x0], x1
.endr
    ret
endfunc

function PFX(pixel_sub_ps_32x64_neon)
    lsl             x1, x1, #1
    mov             w12, #8
.Loop_sub_ps_32x64:
    sub             w12, w12, #1
.rept 4
    ld1             {v0.16b-v1.16b}, [x2], x4
    ld1             {v2.16b-v3.16b}, [x3], x5
    ld1             {v4.16b-v5.16b}, [x2], x4
    ld1             {v6.16b-v7.16b}, [x3], x5
    usubl           v16.8h, v0.8b, v2.8b
    usubl2          v17.8h, v0.16b, v2.16b
    usubl           v18.8h, v1.8b, v3.8b
    usubl2          v19.8h, v1.16b, v3.16b
    usubl           v20.8h, v4.8b, v6.8b
    usubl2          v21.8h, v4.16b, v6.16b
    usubl           v22.8h, v5.8b, v7.8b
    usubl2          v23.8h, v5.16b, v7.16b
    st1             {v16.8h-v19.8h}, [x0], x1
    st1             {v20.8h-v23.8h}, [x0], x1
.endr
    cbnz            w12, .Loop_sub_ps_32x64
    ret
endfunc

// void scale1D_128to64(pixel *dst, const pixel *src)
function PFX(scale1D_128to64_neon)
.rept 2
    ld2             {v0.16b, v1.16b}, [x1], #32
    ld2             {v2.16b, v3.16b}, [x1], #32
    ld2             {v4.16b, v5.16b}, [x1], #32
    ld2             {v6.16b, v7.16b}, [x1], #32
    urhadd          v0.16b, v0.16b, v1.16b
    urhadd          v1.16b, v2.16b, v3.16b
    urhadd          v2.16b, v4.16b, v5.16b
    urhadd          v3.16b, v6.16b, v7.16b
    st1             {v0.16b-v3.16b}, [x0], #64
.endr
    ret
endfunc

.macro scale2D_1  v0, v1
    uaddlp          \v0\().8h, \v0\().16b
    uaddlp          \v1\().8h, \v1\().16b
    add             \v0\().8h, \v0\().8h, \v1\().8h
.endm

// void scale2D_64to32(pixel* dst, const pixel* src, intptr_t stride)
function PFX(scale2D_64to32_neon)
    mov             w12, #32
.Loop_scale2D:
    ld1             {v0.16b-v3.16b}, [x1], x2
    sub             w12, w12, #1
    ld1             {v4.16b-v7.16b}, [x1], x2
    scale2D_1       v0, v4
    scale2D_1       v1, v5
    scale2D_1       v2, v6
    scale2D_1       v3, v7
    uqrshrn         v0.8b, v0.8h, #2
    uqrshrn2        v0.16b, v1.8h, #2
    uqrshrn         v1.8b, v2.8h, #2
    uqrshrn2        v1.16b, v3.8h, #2
    st1             {v0.16b-v1.16b}, [x0], #32
    cbnz            w12, .Loop_scale2D
    ret
endfunc

/***** dequant_scaling*****/
// void dequant_scaling_c(const int16_t* quantCoef, const int32_t* deQuantCoef, int16_t* coef, int num, int per, int shift)
function PFX(dequant_scaling_neon)
    add             x5, x5, #4              // shift + 4
    lsr             x3, x3, #3              // num / 8
    cmp             x5, x4
    blt             .dequant_skip

    mov             x12, #1
    sub             x6, x5, x4          // shift - per
    sub             x6, x6, #1          // shift - per - 1
    lsl             x6, x12, x6         // 1 << shift - per - 1 (add)
    dup             v0.4s, w6
    sub             x7, x4, x5          // per - shift
    dup             v3.4s, w7

.dequant_loop1:
    ld1             {v19.8h}, [x0], #16 // quantCoef
    ld1             {v2.4s}, [x1], #16  // deQuantCoef
    ld1             {v20.4s}, [x1], #16
    sub             x3, x3, #1
    sxtl            v1.4s, v19.4h
    sxtl2           v19.4s, v19.8h

    mul             v1.4s, v1.4s, v2.4s // quantCoef * deQuantCoef
    mul             v19.4s, v19.4s, v20.4s
    add             v1.4s, v1.4s, v0.4s // quantCoef * deQuantCoef + add
    add             v19.4s, v19.4s, v0.4s

    sshl            v1.4s, v1.4s, v3.4s
    sshl            v19.4s, v19.4s, v3.4s
    sqxtn           v16.4h, v1.4s       // x265_clip3
    sqxtn2          v16.8h, v19.4s
    st1             {v16.8h}, [x2], #16
    cbnz            x3, .dequant_loop1
    ret

.dequant_skip:
    sub             x6, x4, x5          // per - shift
    dup             v0.8h, w6

.dequant_loop2:
    ld1             {v19.8h}, [x0], #16 // quantCoef
    ld1             {v2.4s}, [x1], #16  // deQuantCoef
    ld1             {v20.4s}, [x1], #16
    sub             x3, x3, #1
    sxtl            v1.4s, v19.4h
    sxtl2           v19.4s, v19.8h

    mul             v1.4s, v1.4s, v2.4s // quantCoef * deQuantCoef
    mul             v19.4s, v19.4s, v20.4s
    sqxtn           v16.4h, v1.4s       // x265_clip3
    sqxtn2          v16.8h, v19.4s

    sqshl           v16.8h, v16.8h, v0.8h // coefQ << per - shift
    st1             {v16.8h}, [x2], #16
    cbnz            x3, .dequant_loop2
    ret
endfunc

// void dequant_normal_c(const int16_t* quantCoef, int16_t* coef, int num, int scale, int shift)
function PFX(dequant_normal_neon)
//  X265_CHECK(num >= 4 * 4, "dequant num %d too small\n", num);
//  X265_CHECK(num <= 32 * 32, "dequant num %d too large\n", num);
//  X265_CHECK((num % 16) == 0, "dequant num %d not multiple of 16\n", num);
#if HIGH_BIT_DEPTH
    cmp             w3, #32768
    blt             .dqn_skip
    lsr             w3, w3, #(BIT_DEPTH - 8)
    sub             w4, w4, #(BIT_DEPTH - 8)
.dqn_skip:
#endif
    neg             w4, w4
    dup             v0.8h, w3
    dup             v1.4s, w4

.dqn_loop1:
    ld1             {v2.8h, v3.8h}, [x0], #32
    smull           v16.4s, v2.4h, v0.4h
    smull2          v17.4s, v2.8h, v0.8h
    smull           v18.4s, v3.4h, v0.4h
    smull2          v19.4s, v3.8h, v0.8h

    srshl           v16.4s, v16.4s, v1.4s
    srshl           v17.4s, v17.4s, v1.4s
    srshl           v18.4s, v18.4s, v1.4s
    srshl           v19.4s, v19.4s, v1.4s

    sqxtn           v2.4h, v16.4s
    sqxtn2          v2.8h, v17.4s
    sqxtn           v3.4h, v18.4s
    sqxtn2          v3.8h, v19.4s

    sub             w2, w2, #16
    st1             {v2.8h, v3.8h}, [x1], #32
    cbnz            w2, .dqn_loop1
    ret
endfunc

/********* ssim ***********/
// void ssim_4x4x2_core(const pixel* pix1, intptr_t stride1, const pixel* pix2, intptr_t stride2, int sums[2][4])
function PFX(ssim_4x4x2_core_neon)
    ld1             {v0.8b}, [x0], x1
    ld1             {v1.8b}, [x0], x1
    ld1             {v2.8b}, [x0], x1
    ld1             {v3.8b}, [x0], x1

    ld1             {v4.8b}, [x2], x3
    ld1             {v5.8b}, [x2], x3
    ld1             {v6.8b}, [x2], x3
    ld1             {v7.8b}, [x2], x3

    umull           v16.8h, v0.8b, v0.8b
    umull           v17.8h, v1.8b, v1.8b
    umull           v18.8h, v2.8b, v2.8b
    uaddlp          v30.4s, v16.8h
    umull           v19.8h, v3.8b, v3.8b
    umull           v20.8h, v4.8b, v4.8b
    umull           v21.8h, v5.8b, v5.8b
    uadalp          v30.4s, v17.8h
    umull           v22.8h, v6.8b, v6.8b
    umull           v23.8h, v7.8b, v7.8b

    umull           v24.8h, v0.8b, v4.8b
    uadalp          v30.4s, v18.8h
    umull           v25.8h, v1.8b, v5.8b
    umull           v26.8h, v2.8b, v6.8b
    umull           v27.8h, v3.8b, v7.8b
    uadalp          v30.4s, v19.8h

    uaddl           v28.8h, v0.8b, v1.8b
    uaddl           v29.8h, v4.8b, v5.8b
    uadalp          v30.4s, v20.8h
    uaddlp          v31.4s, v24.8h

    uaddw           v28.8h, v28.8h, v2.8b
    uaddw           v29.8h, v29.8h, v6.8b
    uadalp          v30.4s, v21.8h
    uadalp          v31.4s, v25.8h

    uaddw           v28.8h, v28.8h, v3.8b
    uaddw           v29.8h, v29.8h, v7.8b
    uadalp          v30.4s, v22.8h
    uadalp          v31.4s, v26.8h

    uaddlp          v28.4s, v28.8h
    uaddlp          v29.4s, v29.8h
    uadalp          v30.4s, v23.8h
    uadalp          v31.4s, v27.8h

    addp            v28.4s, v28.4s, v28.4s
    addp            v29.4s, v29.4s, v29.4s
    addp            v30.4s, v30.4s, v30.4s
    addp            v31.4s, v31.4s, v31.4s

    st4             {v28.2s, v29.2s, v30.2s, v31.2s}, [x4]
    ret
endfunc

// uint32_t quant_c(const int16_t* coef, const int32_t* quantCoeff, int32_t* deltaU, int16_t* qCoef, int qBits, int add, int numCoeff)
function PFX(quant_neon)
    mov             w9, #1
    lsl             w9, w9, w4
    dup             v0.4s, w9
    neg             w9, w4
    dup             v1.4s, w9
    add             w9, w9, #8
    dup             v2.4s, w9
    dup             v3.4s, w5
    movi            v31.2d, #0

    lsr             w7, w6, #3
    movi            v4.2d, #0

.Loop_quant:

    ld1             {v18.8h}, [x0], #16
    ld1             {v20.4s, v21.4s}, [x1], #32

    sabdl           v6.4s, v18.4h, v31.4h
    sabdl2          v26.4s, v18.8h, v31.8h

    mul             v6.4s, v6.4s, v20.4s
    mul             v26.4s, v26.4s, v21.4s

    add             v7.4s, v6.4s, v3.4s
    add             v27.4s, v26.4s, v3.4s
    sshl            v7.4s, v7.4s, v1.4s
    sshl            v27.4s, v27.4s, v1.4s

    mls             v6.4s, v7.4s, v0.4s
    mls             v26.4s, v27.4s, v0.4s
    sshl            v16.4s, v6.4s, v2.4s
    sshl            v17.4s, v26.4s, v2.4s
    st1             {v16.4s, v17.4s}, [x2], #32

    // numsig
    uzp1            v7.8h, v7.8h, v27.8h
    cmeq            v16.8h, v7.8h, #0
    add             v4.8h, v4.8h, v16.8h

    // level *= sign
    cmlt            v5.8h, v18.8h, #0
    eor             v16.16b, v7.16b, v5.16b
    sub             v5.8h, v16.8h, v5.8h
    st1             {v5.8h}, [x3], #16

    subs            w7, w7, #1
    b.ne            .Loop_quant

    saddlv          s4, v4.8h
    fmov            w9, s4
    add             w0, w6, w9
    ret
endfunc

// uint32_t nquant_c(const int16_t* coef, const int32_t* quantCoeff, int16_t* qCoef, int qBits, int add, int numCoeff)
function PFX(nquant_neon)
    neg             x12, x3
    dup             v0.4s, w12             // q0= -qbits
    dup             v1.4s, w4              // add

    lsr             w6, w5, #3
    movi            v4.4s, #0              // v4= accumulate numsig
    movi            v5.2d, #0              // v5= zero-vector for SABDL(2)

.Loop_nquant:
    ld1             {v16.8h}, [x0], #16
    sub             w6, w6, #1

    sabdl           v17.4s, v16.4h, v5.4h  // v17 = level=abs(coef[blockpos])
    sabdl2          v18.4s, v16.8h, v5.8h  // v18 = level=abs(coef[blockpos])

    ld1             {v19.4s, v20.4s}, [x1], #32   // v19, v20 = quantCoeff[blockpos]
    mul             v17.4s, v17.4s, v19.4s // v17 = tmplevel = abs(level) * quantCoeff[blockpos];
    mul             v18.4s, v18.4s, v20.4s // v18 = tmplevel = abs(level) * quantCoeff[blockpos];

    add             v19.4s, v17.4s, v1.4s  // v20 = tmplevel+add
    add             v20.4s, v18.4s, v1.4s  // v21 = tmplevel+add
    sshl            v19.4s, v19.4s, v0.4s  // v20 = level =(tmplevel+add) >> qbits
    sshl            v20.4s, v20.4s, v0.4s  // v21 = level =(tmplevel+add) >> qbits

    // numsig
    uzp1            v19.8h, v19.8h, v20.8h
    cmeq            v20.8h, v19.8h, #0
    add             v4.8h, v4.8h, v20.8h

    // level *= sign
    cmlt            v16.8h, v16.8h, #0
    eor             v19.16b, v19.16b, v16.16b
    sub             v19.8h, v19.8h, v16.8h
    abs             v19.8h, v19.8h
    st1             {v19.8h}, [x2], #16

    cbnz            w6, .Loop_nquant

    saddlv          s4, v4.8h
    fmov            w9, s4
    add             w0, w5, w9
    ret
endfunc

// void ssimDist_c(const pixel* fenc, uint32_t fStride, const pixel* recon, intptr_t rstride, uint64_t *ssBlock, int shift, uint64_t *ac_k)
.macro ssimDist_1  v4 v5
    sub             v20.8h, \v4\().8h, \v5\().8h
    smull           v16.4s, \v4\().4h, \v4\().4h
    smull2          v17.4s, \v4\().8h, \v4\().8h
    smull           v18.4s, v20.4h, v20.4h
    smull2          v19.4s, v20.8h, v20.8h
    add             v0.4s, v0.4s, v16.4s
    add             v0.4s, v0.4s, v17.4s
    add             v1.4s, v1.4s, v18.4s
    add             v1.4s, v1.4s, v19.4s
.endm

function PFX(ssimDist4_neon)
    ssimDist_start
.rept 4
    ld1             {v4.s}[0], [x0], x1
    ld1             {v5.s}[0], [x2], x3
    uxtl            v4.8h, v4.8b
    uxtl            v5.8h, v5.8b
    sub             v2.4h, v4.4h, v5.4h
    smull           v3.4s, v4.4h, v4.4h
    smull           v2.4s, v2.4h, v2.4h
    add             v0.4s, v0.4s, v3.4s
    add             v1.4s, v1.4s, v2.4s
.endr
    ssimDist_end
    ret
endfunc

function PFX(ssimDist8_neon)
    ssimDist_start
.rept 8
    ld1             {v4.8b}, [x0], x1
    ld1             {v5.8b}, [x2], x3
    uxtl            v4.8h, v4.8b
    uxtl            v5.8h, v5.8b
    ssimDist_1      v4, v5
.endr
    ssimDist_end
    ret
endfunc

function PFX(ssimDist16_neon)
    mov w12, #16
    ssimDist_start
.Loop_ssimDist16:
    sub             w12, w12, #1
    ld1             {v4.16b}, [x0], x1
    ld1             {v5.16b}, [x2], x3
    uxtl            v6.8h, v4.8b
    uxtl            v7.8h, v5.8b
    uxtl2           v4.8h, v4.16b
    uxtl2           v5.8h, v5.16b
    ssimDist_1      v6, v7
    ssimDist_1      v4, v5
    cbnz            w12, .Loop_ssimDist16
    ssimDist_end
    ret
endfunc

function PFX(ssimDist32_neon)
    mov w12, #32
    ssimDist_start
.Loop_ssimDist32:
    sub             w12, w12, #1
    ld1             {v4.16b-v5.16b}, [x0], x1
    ld1             {v6.16b-v7.16b}, [x2], x3
    uxtl            v21.8h, v4.8b
    uxtl            v22.8h, v6.8b
    uxtl            v23.8h, v5.8b
    uxtl            v24.8h, v7.8b
    uxtl2           v25.8h, v4.16b
    uxtl2           v26.8h, v6.16b
    uxtl2           v27.8h, v5.16b
    uxtl2           v28.8h, v7.16b
    ssimDist_1      v21, v22
    ssimDist_1      v23, v24
    ssimDist_1      v25, v26
    ssimDist_1      v27, v28
    cbnz            w12, .Loop_ssimDist32
    ssimDist_end
    ret
endfunc

function PFX(ssimDist64_neon)
    mov w12, #64
    ssimDist_start
.Loop_ssimDist64:
    sub             w12, w12, #1
    ld1             {v4.16b-v7.16b}, [x0], x1
    ld1             {v16.16b-v19.16b}, [x2], x3
    uxtl            v21.8h, v4.8b
    uxtl            v22.8h, v16.8b
    uxtl            v23.8h, v5.8b
    uxtl            v24.8h, v17.8b
    uxtl2           v25.8h, v4.16b
    uxtl2           v26.8h, v16.16b
    uxtl2           v27.8h, v5.16b
    uxtl2           v28.8h, v17.16b
    ssimDist_1      v21, v22
    ssimDist_1      v23, v24
    ssimDist_1      v25, v26
    ssimDist_1      v27, v28
    uxtl            v21.8h, v6.8b
    uxtl            v22.8h, v18.8b
    uxtl            v23.8h, v7.8b
    uxtl            v24.8h, v19.8b
    uxtl2           v25.8h, v6.16b
    uxtl2           v26.8h, v18.16b
    uxtl2           v27.8h, v7.16b
    uxtl2           v28.8h, v19.16b
    ssimDist_1      v21, v22
    ssimDist_1      v23, v24
    ssimDist_1      v25, v26
    ssimDist_1      v27, v28
    cbnz            w12, .Loop_ssimDist64
    ssimDist_end
    ret
endfunc

// void normFact_c(const pixel* src, uint32_t blockSize, int shift, uint64_t *z_k)

.macro normFact_1  v4
    smull           v16.4s, \v4\().4h, \v4\().4h
    smull2          v17.4s, \v4\().8h, \v4\().8h
    add             v0.4s, v0.4s, v16.4s
    add             v0.4s, v0.4s, v17.4s
.endm

function PFX(normFact8_neon)
    normFact_start
.rept 8
    ld1             {v4.8b}, [x0], x1
    uxtl            v4.8h, v4.8b
    normFact_1      v4
.endr
    normFact_end
    ret
endfunc

function PFX(normFact16_neon)
    mov w12, #16
    normFact_start
.Loop_normFact16:
    sub             w12, w12, #1
    ld1             {v4.16b}, [x0], x1
    uxtl            v5.8h, v4.8b
    uxtl2           v4.8h, v4.16b
    normFact_1      v5
    normFact_1      v4
    cbnz            w12, .Loop_normFact16
    normFact_end
    ret
endfunc

function PFX(normFact32_neon)
    mov w12, #32
    normFact_start
.Loop_normFact32:
    sub             w12, w12, #1
    ld1             {v4.16b-v5.16b}, [x0], x1
    uxtl            v6.8h, v4.8b
    uxtl2           v4.8h, v4.16b
    uxtl            v7.8h, v5.8b
    uxtl2           v5.8h, v5.16b
    normFact_1      v4
    normFact_1      v5
    normFact_1      v6
    normFact_1      v7
    cbnz            w12, .Loop_normFact32
    normFact_end
    ret
endfunc

function PFX(normFact64_neon)
    mov w12, #64
    normFact_start
.Loop_normFact64:
    sub             w12, w12, #1
    ld1             {v4.16b-v7.16b}, [x0], x1
    uxtl            v26.8h, v4.8b
    uxtl2           v24.8h, v4.16b
    uxtl            v27.8h, v5.8b
    uxtl2           v25.8h, v5.16b
    normFact_1      v24
    normFact_1      v25
    normFact_1      v26
    normFact_1      v27
    uxtl            v26.8h, v6.8b
    uxtl2           v24.8h, v6.16b
    uxtl            v27.8h, v7.8b
    uxtl2           v25.8h, v7.16b
    normFact_1      v24
    normFact_1      v25
    normFact_1      v26
    normFact_1      v27
    cbnz            w12, .Loop_normFact64
    normFact_end
    ret
endfunc

// int scanPosLast(
//     const uint16_t *scan,      // x0
//     const coeff_t *coeff,      // x1
//     uint16_t *coeffSign,       // x2
//     uint16_t *coeffFlag,       // x3
//     uint8_t *coeffNum,         // x4
//     int numSig,                // x5
//     const uint16_t* scanCG4x4, // x6
//     const int trSize)          // x7
function PFX(scanPosLast_neon)
    // Convert unit of trSize stride from elements (int16) to bytes.
    add             x7, x7, x7

    // Load scan table and convert to bytes.
    ldp             q0, q1, [x6]
    uzp1            v0.16b, v0.16b, v1.16b  // v0 - Zigzag scan table

    movrel          x10, g_SPL_and_mask
    ldr             q28, [x10]              // v28 = mask for pmovmskb
    add             x10, x7, x7             // 2*x7
    add             x11, x7, x7, lsl #1     // 3*x7
    add             x9, x4, #1              // CG count

.Loop_spl:
    // Position of current CG.
    ldrh            w6, [x0], #32
    add             x6, x1, x6, lsl #1

    // Loading current CG and saturate to bytes.
    ldr             d2, [x6]
    ldr             d3, [x6, x7]
    ldr             d4, [x6, x10]
    ldr             d5, [x6, x11]
    mov             v2.d[1], v3.d[0]
    mov             v4.d[1], v5.d[0]
    sqxtn           v2.8b, v2.8h
    sqxtn2          v2.16b, v4.8h

    // Apply zigzag.
    tbl             v3.16b, {v2.16b}, v0.16b

    // Get zero/sign.
    cmeq            v5.16b, v3.16b, #0   // v5 = zero
    cmlt            v3.16b, v3.16b, #0   // v3 = negative

    //  val: w13 = pmovmskb(v3)
    // mask: w15 = pmovmskb(v4)
    and             v3.16b, v3.16b, v28.16b
    bic             v4.16b, v28.16b, v5.16b
    addp            v3.16b, v3.16b, v4.16b
    addp            v3.16b, v3.16b, v3.16b
    addp            v3.16b, v3.16b, v3.16b
    fmov            w13, s3
    lsr             w15, w13, #16

    // coeffNum = addv(v3 != 0) = 16 - addv(v5)
    addv            b5, v5.16b
    smov            w6, v5.b[0]
    add             w6, w6, #16
    sub             w5, w5, w6
    strb            w6, [x4], #1

    // coeffFlag = reverse_bit(w15) in 16-bit
    rbit            w12, w13
    strh            w12, [x3], #2

    // Pack bits from w13 into w14, based on w15 mask.
    and             w13, w13, w15
    mov             x14, xzr
    cbz             w15, .Lpext_end
.Loop_spl_1:
    clz             w6, w15
    lsl             w13, w13, w6
    lsl             w15, w15, w6
    extr            w14, w14, w13, #31
    bfm             w15, wzr, #1, #0
    cbnz            w15, .Loop_spl_1
.Lpext_end:
    strh            w14, [x2], #2

    cbnz            x5, .Loop_spl

    // Count trailing zeros.
    rbit            w13, w12
    clz             w13, w13
    lsr             w12, w12, w13
    strh            w12, [x3, #-2]

    // Get last pos.
    sub             x9, x4, x9
    eor             w13, w13, #15
    add             x0, x13, x9, lsl #4
    ret
endfunc

// uint32_t costCoeffNxN(
//    uint16_t *scan,        // x0
//    coeff_t *coeff,        // x1
//    intptr_t trSize,       // x2
//    uint16_t *absCoeff,    // x3
//    uint8_t *tabSigCtx,    // x4
//    uint16_t scanFlagMask, // x5
//    uint8_t *baseCtx,      // x6
//    int offset,            // x7
//    int scanPosSigOff,     // sp
//    int subPosBase)        // sp + 8, or sp + 4 on APPLE
function PFX(costCoeffNxN_neon)
    // abs(coeff)
    add             x2, x2, x2
    ld1             {v1.d}[0], [x1], x2
    ld1             {v1.d}[1], [x1], x2
    ld1             {v2.d}[0], [x1], x2
    ld1             {v2.d}[1], [x1], x2
    abs             v1.8h, v1.8h
    abs             v2.8h, v2.8h

    // WARNING: beyond-bound read here!
    // loading scan table
    ldr             w2, [sp]
    eor             w15, w2, #15
    add             x1, x0, x15, lsl #1
    ldp             q20, q21, [x1]
    uzp1            v20.16b, v20.16b, v21.16b
    movi            v21.16b, #15
    eor             v0.16b, v20.16b, v21.16b

    // reorder coeff
    uzp1           v22.16b, v1.16b, v2.16b
    uzp2           v23.16b, v1.16b, v2.16b
    tbl            v24.16b, {v22.16b}, v0.16b
    tbl            v25.16b, {v23.16b}, v0.16b
    zip1           v2.16b, v24.16b, v25.16b
    zip2           v3.16b, v24.16b, v25.16b

    // loading tabSigCtx (+offset)
    ldr             q1, [x4]
    tbl             v1.16b, {v1.16b}, v0.16b
    dup             v4.16b, w7
    movi            v5.16b, #0
    tbl             v4.16b, {v4.16b}, v5.16b
    add             v1.16b, v1.16b, v4.16b

    // register mapping
    // x0 - sum
    // x1 - entropyStateBits
    // v1 - sigCtx
    // {v3,v2} - abs(coeff)
    // x2 - scanPosSigOff
    // x3 - absCoeff
    // x4 - numNonZero
    // x5 - scanFlagMask
    // x6 - baseCtx
    mov             x0, #0
    movrel          x1, PFX_C(entropyStateBits)
    mov             x4, #0
    mov             x11, #0
    movi            v31.16b, #0
    cbz             x2, .idx_zero
.Loop_ccnn:
//   {
//        const uint32_t cnt = tabSigCtx[blkPos] + offset + posOffset;
//        ctxSig = cnt & posZeroMask;
//        const uint32_t mstate = baseCtx[ctxSig];
//        const uint32_t mps = mstate & 1;
//        const uint32_t stateBits = x265_entropyStateBits[mstate ^ sig];
//        uint32_t nextState = (stateBits >> 24) + mps;
//        if ((mstate ^ sig) == 1)
//            nextState = sig;
//        baseCtx[ctxSig] = (uint8_t)nextState;
//        sum += stateBits;
//    }
//    absCoeff[numNonZero] = tmpCoeff[blkPos];
//    numNonZero += sig;
//    scanPosSigOff--;

    add             x13, x3, x4, lsl #1
    sub             x2, x2, #1
    str             h2, [x13]             // absCoeff[numNonZero] = tmpCoeff[blkPos]
    fmov            w14, s1               // x14 = ctxSig
    uxtb            w14, w14
    ubfx            w11, w5, #0, #1       // x11 = sig
    lsr             x5, x5, #1
    add             x4, x4, x11           // numNonZero += sig
    ext             v1.16b, v1.16b, v31.16b, #1
    ext             v2.16b, v2.16b, v3.16b, #2
    ext             v3.16b, v3.16b, v31.16b, #2
    ldrb            w9, [x6, x14]         // mstate = baseCtx[ctxSig]
    and             w10, w9, #1           // mps = mstate & 1
    eor             w9, w9, w11           // x9 = mstate ^ sig
    add             x12, x1, x9, lsl #2
    ldr             w13, [x12]
    add             w0, w0, w13           // sum += x265_entropyStateBits[mstate ^ sig]
    ldrb            w13, [x12, #3]
    add             w10, w10, w13         // nextState = (stateBits >> 24) + mps
    cmp             w9, #1
    csel            w10, w11, w10, eq
    strb            w10, [x6, x14]
    cbnz            x2, .Loop_ccnn
.idx_zero:

    add             x13, x3, x4, lsl #1
    add             x4, x4, x15
    str             h2, [x13]              // absCoeff[numNonZero] = tmpCoeff[blkPos]

    ldr             x9, [sp, #STACK_ARG_OFFSET(1)]           // subPosBase
    uxth            w9, w9
    cmp             w9, #0
    cset            x2, eq
    add             x4, x4, x2
    cbz             x4, .exit_ccnn

    sub             w2, w2, #1
    uxtb            w2, w2
    fmov            w3, s1
    and             w2, w2, w3

    ldrb            w3, [x6, x2]         // mstate = baseCtx[ctxSig]
    eor             w4, w5, w3            // x5 = mstate ^ sig
    and             w3, w3, #1            // mps = mstate & 1
    add             x1, x1, x4, lsl #2
    ldr             w11, [x1]
    ldrb            w12, [x1, #3]
    add             w0, w0, w11           // sum += x265_entropyStateBits[mstate ^ sig]
    add             w3, w3, w12           // nextState = (stateBits >> 24) + mps
    cmp             w4, #1
    csel            w3, w5, w3, eq
    strb            w3, [x6, x2]
.exit_ccnn:
    ubfx            w0, w0, #0, #24
    ret
endfunc

const g_SPL_and_mask, align=8
.byte 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
endconst
