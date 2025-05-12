/*****************************************************************************
 * Copyright (C) 2022-2023 MulticoreWare, Inc
 *
 * Authors: David Chen <david.chen@myais.com.cn>
 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at license @ x265.com.
 *****************************************************************************/

#include "asm-sve.S"
#include "blockcopy8-common.S"

.arch armv8-a+sve

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4

.text

function PFX(blockfill_s_32x32_sve)
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_blockfill_s_32_32
    dup             v0.8h, w2
    mov             v1.16b, v0.16b
    mov             v2.16b, v0.16b
    mov             v3.16b, v0.16b
    lsl             x1, x1, #1
.rept 32
    st1             {v0.8h-v3.8h}, [x0], x1
.endr
    ret
.vl_gt_16_blockfill_s_32_32:
    cmp             x9, #48
    bgt             .vl_gt_48_blockfill_s_32_32
    dup             z0.h, w2
    ptrue           p0.h, vl16
.rept 32
    st1h            {z0.h}, p0, [x0]
    st1h            {z0.h}, p0, [x0, #1, mul vl]
    add             x0, x0, x1, lsl #1
.endr
    ret
.vl_gt_48_blockfill_s_32_32:
    dup             z0.h, w2
    ptrue           p0.h, vl32
.rept 32
    st1h            {z0.h}, p0, [x0]
    add             x0, x0, x1, lsl #1
.endr
    ret
endfunc

// void cpy2Dto1D_shl(int16_t* dst, const int16_t* src, intptr_t srcStride, int shift)
.macro cpy2Dto1D_shl_start_sve
    add             x2, x2, x2
    mov             z0.h, w3
.endm

function PFX(cpy2Dto1D_shl_16x16_sve)
    dup             z0.h, w3
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_cpy2Dto1D_shl_16x16
    cpy2Dto1D_shl_start_sve
    mov             w12, #4
.Loop_cpy2Dto1D_shl_16_sve:
    sub             w12, w12, #1
.rept 4
    ld1             {v2.16b-v3.16b}, [x1], x2
    sshl            v2.8h, v2.8h, v0.8h
    sshl            v3.8h, v3.8h, v0.8h
    st1             {v2.16b-v3.16b}, [x0], #32
.endr
    cbnz            w12, .Loop_cpy2Dto1D_shl_16_sve
    ret
.vl_gt_16_cpy2Dto1D_shl_16x16:
    ptrue           p0.h, vl16
.rept 16
    ld1h            {z1.h}, p0/z, [x1]
    lsl             z1.h, p0/m, z1.h, z0.h
    st1h            {z1.h}, p0, [x0]
    add             x1, x1, x2, lsl #1
    add             x0, x0, #32
.endr
    ret
endfunc

function PFX(cpy2Dto1D_shl_32x32_sve)
    dup             z0.h, w3
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_cpy2Dto1D_shl_32x32
    cpy2Dto1D_shl_start_sve
    mov             w12, #16
.Loop_cpy2Dto1D_shl_32_sve:
    sub             w12, w12, #1
.rept 2
    ld1             {v2.16b-v5.16b}, [x1], x2
    sshl            v2.8h, v2.8h, v0.8h
    sshl            v3.8h, v3.8h, v0.8h
    sshl            v4.8h, v4.8h, v0.8h
    sshl            v5.8h, v5.8h, v0.8h
    st1             {v2.16b-v5.16b}, [x0], #64
.endr
    cbnz            w12, .Loop_cpy2Dto1D_shl_32_sve
    ret
.vl_gt_16_cpy2Dto1D_shl_32x32:
    cmp             x9, #48
    bgt             .vl_gt_48_cpy2Dto1D_shl_32x32
    ptrue           p0.h, vl16
.rept 32
    ld1h            {z1.h}, p0/z, [x1]
    ld1h            {z2.h}, p0/z, [x1, #1, mul vl]
    lsl             z1.h, p0/m, z1.h, z0.h
    lsl             z2.h, p0/m, z2.h, z0.h
    st1h            {z1.h}, p0, [x0]
    st1h            {z2.h}, p0, [x0, #1, mul vl]
    add             x1, x1, x2, lsl #1
    add             x0, x0, #64
.endr
    ret
.vl_gt_48_cpy2Dto1D_shl_32x32:
    ptrue           p0.h, vl32
.rept 32
    ld1h            {z1.h}, p0/z, [x1]
    lsl             z1.h, p0/m, z1.h, z0.h
    st1h            {z1.h}, p0, [x0]
    add             x1, x1, x2, lsl #1
    add             x0, x0, #64
.endr
    ret
endfunc

function PFX(cpy2Dto1D_shl_64x64_sve)
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_cpy2Dto1D_shl_64x64
    cpy2Dto1D_shl_start_sve
    mov             w12, #32
    sub             x2, x2, #64
.Loop_cpy2Dto1D_shl_64_sve:
    sub             w12, w12, #1
.rept 2
    ld1             {v2.16b-v5.16b}, [x1], #64
    ld1             {v16.16b-v19.16b}, [x1], x2
    sshl            v2.8h, v2.8h, v0.8h
    sshl            v3.8h, v3.8h, v0.8h
    sshl            v4.8h, v4.8h, v0.8h
    sshl            v5.8h, v5.8h, v0.8h
    sshl            v16.8h, v16.8h, v0.8h
    sshl            v17.8h, v17.8h, v0.8h
    sshl            v18.8h, v18.8h, v0.8h
    sshl            v19.8h, v19.8h, v0.8h
    st1             {v2.16b-v5.16b}, [x0], #64
    st1             {v16.16b-v19.16b}, [x0], #64
.endr
    cbnz            w12, .Loop_cpy2Dto1D_shl_64_sve
    ret
.vl_gt_16_cpy2Dto1D_shl_64x64:
    dup             z0.h, w3
    mov             x8, #64
    mov             w12, #64
.L_init_cpy2Dto1D_shl_64x64:
    sub             w12, w12, 1
    mov             x9, #0
    whilelt         p0.h, x9, x8
.L_cpy2Dto1D_shl_64x64:
    ld1h            {z1.h}, p0/z, [x1, x9, lsl #1]
    lsl             z1.h, p0/m, z1.h, z0.h
    st1h            {z1.h}, p0, [x0, x9, lsl #1]
    inch            x9
    whilelt         p0.h, x9, x8
    b.first         .L_cpy2Dto1D_shl_64x64
    add             x1, x1, x2, lsl #1
    add             x0, x0, #128
    cbnz            w12, .L_init_cpy2Dto1D_shl_64x64
    ret
endfunc

// void cpy2Dto1D_shr(int16_t* dst, const int16_t* src, intptr_t srcStride, int shift)

function PFX(cpy2Dto1D_shr_4x4_sve)
    dup             z0.h, w3
    sub             w4, w3, #1
    dup             z1.h, w4
    ptrue           p0.h, vl8
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
    lsl             x2, x2, #1
    index           z3.d, #0, x2
    index           z4.d, #0, #8
.rept 2
    ld1d            {z5.d}, p0/z, [x1, z3.d]
    add             x1, x1, x2, lsl #1
    add             z5.h, p0/m, z5.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    st1d            {z5.d}, p0, [x0, z4.d]
    add             x0, x0, #16
.endr
    ret
endfunc

function PFX(cpy2Dto1D_shr_8x8_sve)
    dup             z0.h, w3
    sub             w4, w3, #1
    dup             z1.h, w4
    ptrue           p0.h, vl8
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
.rept 8
    ld1d            {z5.d}, p0/z, [x1]
    add             x1, x1, x2, lsl #1
    add             z5.h, p0/m, z5.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    st1d            {z5.d}, p0, [x0]
    add             x0, x0, #16
.endr
    ret
endfunc

function PFX(cpy2Dto1D_shr_16x16_sve)
    dup             z0.h, w3
    sub             w4, w3, #1
    dup             z1.h, w4
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_cpy2Dto1D_shr_16x16
    ptrue           p0.h, vl8
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
.rept 16
    ld1d            {z5.d}, p0/z, [x1]
    ld1d            {z6.d}, p0/z, [x1, #1, mul vl]
    add             x1, x1, x2, lsl #1
    add             z5.h, p0/m, z5.h, z2.h
    add             z6.h, p0/m, z6.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    asr             z6.h, p0/m, z6.h, z0.h
    st1d            {z5.d}, p0, [x0]
    st1d            {z6.d}, p0, [x0, #1, mul vl]
    add             x0, x0, #32
.endr
    ret
.vl_gt_16_cpy2Dto1D_shr_16x16:
    ptrue           p0.h, vl16
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
.rept 16
    ld1d            {z5.d}, p0/z, [x1]
    add             x1, x1, x2, lsl #1
    add             z5.h, p0/m, z5.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    st1d            {z5.d}, p0, [x0]
    add             x0, x0, #32
.endr
    ret
endfunc

function PFX(cpy2Dto1D_shr_32x32_sve)
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_cpy2Dto1D_shr_32x32
    cpy2Dto1D_shr_start
    mov             w12, #16
.Loop_cpy2Dto1D_shr_32_sve:
    sub             w12, w12, #1
.rept 2
    ld1             {v2.8h-v5.8h}, [x1], x2
    sub             v2.8h, v2.8h, v1.8h
    sub             v3.8h, v3.8h, v1.8h
    sub             v4.8h, v4.8h, v1.8h
    sub             v5.8h, v5.8h, v1.8h
    sshl            v2.8h, v2.8h, v0.8h
    sshl            v3.8h, v3.8h, v0.8h
    sshl            v4.8h, v4.8h, v0.8h
    sshl            v5.8h, v5.8h, v0.8h
    st1             {v2.8h-v5.8h}, [x0], #64
.endr
    cbnz            w12, .Loop_cpy2Dto1D_shr_32_sve
    ret
.vl_gt_16_cpy2Dto1D_shr_32x32:
    dup             z0.h, w3
    sub             w4, w3, #1
    dup             z1.h, w4
    cmp             x9, #48
    bgt             .vl_gt_48_cpy2Dto1D_shr_32x32
    ptrue           p0.h, vl16
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
.rept 32
    ld1d            {z5.d}, p0/z, [x1]
    ld1d            {z6.d}, p0/z, [x1, #1, mul vl]
    add             x1, x1, x2, lsl #1
    add             z5.h, p0/m, z5.h, z2.h
    add             z6.h, p0/m, z6.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    asr             z6.h, p0/m, z6.h, z0.h
    st1d            {z5.d}, p0, [x0]
    st1d            {z6.d}, p0, [x0, #1, mul vl]
    add             x0, x0, #64
.endr
    ret
.vl_gt_48_cpy2Dto1D_shr_32x32:
    ptrue           p0.h, vl32
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
.rept 32
    ld1d            {z5.d}, p0/z, [x1]
    add             x1, x1, x2, lsl #1
    add             z5.h, p0/m, z5.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    st1d            {z5.d}, p0, [x0]
    add             x0, x0, #64
.endr
    ret
endfunc

// void cpy1Dto2D_shr(int16_t* dst, const int16_t* src, intptr_t dstStride, int shift)

function PFX(cpy1Dto2D_shr_16x16_sve)
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_cpy1Dto2D_shr_16x16
    cpy1Dto2D_shr_start
    mov             w12, #4
.Loop_cpy1Dto2D_shr_16:
    sub             w12, w12, #1
.rept 4
    ld1             {v2.8h-v3.8h}, [x1], #32
    sub             v2.8h, v2.8h, v1.8h
    sub             v3.8h, v3.8h, v1.8h
    sshl            v2.8h, v2.8h, v0.8h
    sshl            v3.8h, v3.8h, v0.8h
    st1             {v2.8h-v3.8h}, [x0], x2
.endr
    cbnz            w12, .Loop_cpy1Dto2D_shr_16
    ret
.vl_gt_16_cpy1Dto2D_shr_16x16:
    dup             z0.h, w3
    sub             w4, w3, #1
    dup             z1.h, w4
    ptrue           p0.h, vl16
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
.rept 16
    ld1d            {z5.d}, p0/z, [x1]
    add             x1, x1, #32
    add             z5.h, p0/m, z5.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    st1d            {z5.d}, p0, [x0]
    add             x0, x0, x2, lsl #1
.endr
    ret
endfunc

function PFX(cpy1Dto2D_shr_32x32_sve)
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_cpy1Dto2D_shr_32x32
    cpy1Dto2D_shr_start
    mov             w12, #16
.Loop_cpy1Dto2D_shr_32_sve:
    sub             w12, w12, #1
.rept 2
    ld1             {v2.16b-v5.16b}, [x1], #64
    sub             v2.8h, v2.8h, v1.8h
    sub             v3.8h, v3.8h, v1.8h
    sub             v4.8h, v4.8h, v1.8h
    sub             v5.8h, v5.8h, v1.8h
    sshl            v2.8h, v2.8h, v0.8h
    sshl            v3.8h, v3.8h, v0.8h
    sshl            v4.8h, v4.8h, v0.8h
    sshl            v5.8h, v5.8h, v0.8h
    st1             {v2.16b-v5.16b}, [x0], x2
.endr
    cbnz            w12, .Loop_cpy1Dto2D_shr_32_sve
    ret
.vl_gt_16_cpy1Dto2D_shr_32x32:
    dup             z0.h, w3
    sub             w4, w3, #1
    dup             z1.h, w4
    cmp             x9, #48
    bgt             .vl_gt_48_cpy1Dto2D_shr_32x32
    ptrue           p0.h, vl16
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
.rept 32
    ld1d            {z5.d}, p0/z, [x1]
    ld1d            {z6.d}, p0/z, [x1, #1, mul vl]
    add             x1, x1, #64
    add             z5.h, p0/m, z5.h, z2.h
    add             z6.h, p0/m, z6.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    asr             z6.h, p0/m, z6.h, z0.h
    st1d            {z5.d}, p0, [x0]
    st1d            {z6.d}, p0, [x0, #1, mul vl]
    add             x0, x0, x2, lsl #1
.endr
    ret
.vl_gt_48_cpy1Dto2D_shr_32x32:
    ptrue           p0.h, vl32
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
.rept 32
    ld1d            {z5.d}, p0/z, [x1]
    add             x1, x1, #64
    add             z5.h, p0/m, z5.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    st1d            {z5.d}, p0, [x0]
    add             x0, x0, x2, lsl #1
.endr
    ret
endfunc

function PFX(cpy1Dto2D_shr_64x64_sve)
    dup             z0.h, w3
    sub             w4, w3, #1
    dup             z1.h, w4
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_cpy1Dto2D_shr_64x64
    ptrue           p0.h, vl8
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
.rept 64
    ld1d            {z5.d}, p0/z, [x1]
    ld1d            {z6.d}, p0/z, [x1, #1, mul vl]
    ld1d            {z7.d}, p0/z, [x1, #2, mul vl]
    ld1d            {z8.d}, p0/z, [x1, #3, mul vl]
    ld1d            {z9.d}, p0/z, [x1, #4, mul vl]
    ld1d            {z10.d}, p0/z, [x1, #5, mul vl]
    ld1d            {z11.d}, p0/z, [x1, #6, mul vl]
    ld1d            {z12.d}, p0/z, [x1, #7, mul vl]
    add             x1, x1, #128
    add             z5.h, p0/m, z5.h, z2.h
    add             z6.h, p0/m, z6.h, z2.h
    add             z7.h, p0/m, z7.h, z2.h
    add             z8.h, p0/m, z8.h, z2.h
    add             z9.h, p0/m, z9.h, z2.h
    add             z10.h, p0/m, z10.h, z2.h
    add             z11.h, p0/m, z11.h, z2.h
    add             z12.h, p0/m, z12.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    asr             z6.h, p0/m, z6.h, z0.h
    asr             z7.h, p0/m, z7.h, z0.h
    asr             z8.h, p0/m, z8.h, z0.h
    asr             z9.h, p0/m, z9.h, z0.h
    asr             z10.h, p0/m, z10.h, z0.h
    asr             z11.h, p0/m, z11.h, z0.h
    asr             z12.h, p0/m, z12.h, z0.h
    st1d            {z5.d}, p0, [x0]
    st1d            {z6.d}, p0, [x0, #1, mul vl]
    st1d            {z7.d}, p0, [x0, #2, mul vl]
    st1d            {z8.d}, p0, [x0, #3, mul vl]
    st1d            {z9.d}, p0, [x0, #4, mul vl]
    st1d            {z10.d}, p0, [x0, #5, mul vl]
    st1d            {z11.d}, p0, [x0, #6, mul vl]
    st1d            {z12.d}, p0, [x0, #7, mul vl]
    add             x0, x0, x2, lsl #1
.endr
    ret
.vl_gt_16_cpy1Dto2D_shr_64x64:
    cmp             x9, #48
    bgt             .vl_gt_48_cpy1Dto2D_shr_64x64
    ptrue           p0.h, vl16
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
.rept 64
    ld1d            {z5.d}, p0/z, [x1]
    ld1d            {z6.d}, p0/z, [x1, #1, mul vl]
    ld1d            {z7.d}, p0/z, [x1, #2, mul vl]
    ld1d            {z8.d}, p0/z, [x1, #3, mul vl]
    add             x1, x1, #128
    add             z5.h, p0/m, z5.h, z2.h
    add             z6.h, p0/m, z6.h, z2.h
    add             z7.h, p0/m, z7.h, z2.h
    add             z8.h, p0/m, z8.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    asr             z6.h, p0/m, z6.h, z0.h
    asr             z7.h, p0/m, z7.h, z0.h
    asr             z8.h, p0/m, z8.h, z0.h
    st1d            {z5.d}, p0, [x0]
    st1d            {z6.d}, p0, [x0, #1, mul vl]
    st1d            {z7.d}, p0, [x0, #2, mul vl]
    st1d            {z8.d}, p0, [x0, #3, mul vl]
    add             x0, x0, x2, lsl #1
.endr
    ret
.vl_gt_48_cpy1Dto2D_shr_64x64:
    cmp             x9, #112
    bgt             .vl_gt_112_cpy1Dto2D_shr_64x64
    ptrue           p0.h, vl32
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
.rept 64
    ld1d            {z5.d}, p0/z, [x1]
    ld1d            {z6.d}, p0/z, [x1, #1, mul vl]
    add             x1, x1, #128
    add             z5.h, p0/m, z5.h, z2.h
    add             z6.h, p0/m, z6.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    asr             z6.h, p0/m, z6.h, z0.h
    st1d            {z5.d}, p0, [x0]
    st1d            {z6.d}, p0, [x0, #1, mul vl]
    add             x0, x0, x2, lsl #1
.endr
    ret
.vl_gt_112_cpy1Dto2D_shr_64x64:
    ptrue           p0.h, vl64
    mov             z2.h, #1
    lsl             z2.h, p0/m, z2.h, z1.h
.rept 64
    ld1d            {z5.d}, p0/z, [x1]
    add             x1, x1, #128
    add             z5.h, p0/m, z5.h, z2.h
    asr             z5.h, p0/m, z5.h, z0.h
    st1d            {z5.d}, p0, [x0]
    add             x0, x0, x2, lsl #1
.endr
    ret
endfunc
