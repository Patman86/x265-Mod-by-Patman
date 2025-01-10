/*****************************************************************************
 * Copyright (C) 2024 MulticoreWare, Inc
 *
 * Authors: Gerda Zsejke More <gerdazsejke.more@arm.com>
 *
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

#include "asm.S"
#include "ssd-a-common.S"

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4

.text

.arch armv8-a+sve

#if HIGH_BIT_DEPTH
.macro SSE_PP_4x2
    ldr             d16, [x0]
    ldr             d17, [x2]
    ldr             d18, [x0, x1]
    ldr             d19, [x2, x3]
    uabd            v2.4h, v16.4h, v17.4h
    uabd            v3.4h, v18.4h, v19.4h
    udot            z0.d, z2.h, z2.h
    udot            z0.d, z3.h, z3.h
.endm

.macro SSE_PP_4xN h
function PFX(pixel_sse_pp_4x\h\()_sve)
    movi            v0.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

.rept (\h / 2) - 1
    SSE_PP_4x2
    add             x0, x0, x1, lsl #1
    add             x2, x2, x3, lsl #1
.endr
    SSE_PP_4x2

    fmov            w0, s0
    ret
endfunc
.endm

SSE_PP_4xN 4
SSE_PP_4xN 8

.macro SSE_PP_8xN h
function PFX(pixel_sse_pp_8x\h\()_sve)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

.rept \h / 2
    ld1             {v16.8h}, [x0], x1
    ld1             {v17.8h}, [x2], x3
    uabd            v2.8h, v16.8h, v17.8h
    udot            z0.d, z2.h, z2.h
    ld1             {v18.8h}, [x0], x1
    ld1             {v19.8h}, [x2], x3
    uabd            v3.8h, v18.8h, v19.8h
    udot            z1.d, z3.h, z3.h
.endr

    add             v0.2d, v0.2d, v1.2d
    addp            d0, v0.2d
    fmov            w0, s0
    ret
endfunc
.endm

SSE_PP_8xN 8
SSE_PP_8xN 16

.macro SSE_PP_16xN h
function PFX(pixel_sse_pp_16x\h\()_sve)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

    mov             w12, \h
.Loop_sse_pp_16x\h:
    sub             w12, w12, #1

    ld1             {v16.8h-v17.8h}, [x0], x1
    ld1             {v18.8h-v19.8h}, [x2], x3
    uabd            v2.8h, v16.8h, v18.8h
    udot            z0.d, z2.h, z2.h
    uabd            v3.8h, v17.8h, v19.8h
    udot            z1.d, z3.h, z3.h
    cbnz            w12, .Loop_sse_pp_16x\h

    add             v0.2d, v0.2d, v1.2d
    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc
.endm

SSE_PP_16xN 16
SSE_PP_16xN 32

.macro SSE_PP_32xN h
function  PFX(pixel_sse_pp_32x\h\()_sve)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

    mov             w12, \h
.Loop_sse_pp_32x\h:
    sub             w12, w12, #1

    ld1             {v16.8h-v17.8h}, [x0]
    ld1             {v20.8h-v21.8h}, [x2]
    uabd            v2.8h, v16.8h, v20.8h
    udot            z0.d, z2.h, z2.h
    uabd            v3.8h, v17.8h, v21.8h
    udot            z1.d, z3.h, z3.h

    ldp             q18, q19, [x0, #32]
    ldp             q22, q23, [x2, #32]
    uabd            v2.8h, v18.8h, v22.8h
    udot            z0.d, z2.h, z2.h
    uabd            v3.8h, v19.8h, v23.8h
    udot            z1.d, z3.h, z3.h

    add             x0, x0, x1
    add             x2, x2, x3
    cbnz            w12, .Loop_sse_pp_32x\h

    add             v0.2d, v0.2d, v1.2d
    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc
.endm

SSE_PP_32xN 32
SSE_PP_32xN 64

function PFX(pixel_sse_pp_64x64_sve)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

    mov             w12, #64
.Loop_sse_pp_64x1:
    sub             w12, w12, #1

    ld1             {v16.8h-v17.8h}, [x0]
    ld1             {v20.8h-v21.8h}, [x2]
    uabd            v2.8h, v16.8h, v20.8h
    udot            z0.d, z2.h, z2.h
    uabd            v3.8h, v17.8h, v21.8h
    udot            z1.d, z3.h, z3.h

    ldp             q18, q19, [x0, #32]
    ldp             q22, q23, [x2, #32]
    uabd            v2.8h, v18.8h, v22.8h
    udot            z0.d, z2.h, z2.h
    uabd            v3.8h, v19.8h, v23.8h
    udot            z1.d, z3.h, z3.h

    ldp             q16, q17, [x0, #64]
    ldp             q20, q21, [x2, #64]
    uabd            v2.8h, v16.8h, v20.8h
    udot            z0.d, z2.h, z2.h
    uabd            v3.8h, v17.8h, v21.8h
    udot            z1.d, z3.h, z3.h

    ldp             q18, q19,  [x0, #96]
    ldp             q22, q23, [x2, #96]
    uabd            v2.8h, v18.8h, v22.8h
    udot            z0.d, z2.h, z2.h
    uabd            v3.8h, v19.8h, v23.8h
    udot            z1.d, z3.h, z3.h

    add             x0, x0, x1
    add             x2, x2, x3
    cbnz            w12, .Loop_sse_pp_64x1

    add             v0.2d, v0.2d, v1.2d
    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc

#endif // HIGH_BIT_DEPTH

.macro SSE_SS_4x2
    ldr             d16, [x0]
    ldr             d17, [x2]
    ldr             d18, [x0, x1]
    ldr             d19, [x2, x3]
    sub             v2.4h, v16.4h, v17.4h
    sub             v3.4h, v18.4h, v19.4h
    sdot            z0.d, z2.h, z2.h
    sdot            z0.d, z3.h, z3.h
.endm

function PFX(pixel_sse_ss_4x4_sve)
    movi            v0.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

    SSE_SS_4x2
    add             x0, x0, x1, lsl #1
    add             x2, x2, x3, lsl #1
    SSE_SS_4x2

    fmov            w0, s0
    ret
endfunc

function PFX(pixel_sse_ss_8x8_sve)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

.rept 4
    ld1             {v16.8h}, [x0], x1
    ld1             {v17.8h}, [x2], x3
    sub             v2.8h, v16.8h, v17.8h
    sdot            z0.d, z2.h, z2.h
    ld1             {v18.8h}, [x0], x1
    ld1             {v19.8h}, [x2], x3
    sub             v3.8h, v18.8h, v19.8h
    sdot            z1.d, z3.h, z3.h
.endr

    add             v0.2d, v0.2d, v1.2d
    addp            d0, v0.2d
    fmov            w0, s0
    ret
endfunc

function PFX(pixel_sse_ss_16x16_sve)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

    mov             w12, #16
.Loop_sse_ss_16:
    sub             w12, w12, #1

    ld1             {v16.8h-v17.8h}, [x0], x1
    ld1             {v18.8h-v19.8h}, [x2], x3
    sub             v2.8h, v16.8h, v18.8h
    sdot            z0.d, z2.h, z2.h
    sub             v3.8h, v17.8h, v19.8h
    sdot            z1.d, z3.h, z3.h
    cbnz            w12, .Loop_sse_ss_16

    add             v0.2d, v0.2d, v1.2d
    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc

function PFX(pixel_sse_ss_32x32_sve)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

    mov             w12, #32
.Loop_sse_ss_32:
    sub             w12, w12, #1

    ld1             {v16.8h-v17.8h}, [x0]
    ld1             {v20.8h-v21.8h}, [x2]
    sub             v2.8h, v16.8h, v20.8h
    sdot            z0.d, z2.h, z2.h
    sub             v3.8h, v17.8h, v21.8h
    sdot            z1.d, z3.h, z3.h

    ldp             q18, q19, [x0, #32]
    ldp             q22, q23, [x2, #32]
    sub             v2.8h, v18.8h, v22.8h
    sdot            z0.d, z2.h, z2.h
    sub             v3.8h, v19.8h, v23.8h
    sdot            z1.d, z3.h, z3.h

    add             x0, x0, x1
    add             x2, x2, x3
    cbnz            w12, .Loop_sse_ss_32

    add             v0.2d, v0.2d, v1.2d
    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc


function PFX(pixel_sse_ss_64x64_sve)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

    mov             w12, #64
.Loop_sse_ss_64:
    sub             w12, w12, #1

    ld1             {v16.8h-v17.8h}, [x0]
    ld1             {v20.8h-v21.8h}, [x2]
    sub             v2.8h, v16.8h, v20.8h
    sdot            z0.d, z2.h, z2.h
    sub             v3.8h, v17.8h, v21.8h
    sdot            z1.d, z3.h, z3.h

    ldp             q18, q19, [x0, #32]
    ldp             q22, q23, [x2, #32]
    sub             v2.8h, v18.8h, v22.8h
    sdot            z0.d, z2.h, z2.h
    sub             v3.8h, v19.8h, v23.8h
    sdot            z1.d, z3.h, z3.h

    ldp             q16, q17, [x0, #64]
    ldp             q20, q21, [x2, #64]
    sub             v2.8h, v16.8h, v20.8h
    sdot            z0.d, z2.h, z2.h
    sub             v3.8h, v17.8h, v21.8h
    sdot            z1.d, z3.h, z3.h

    ldp             q18, q19,  [x0, #96]
    ldp             q22, q23, [x2, #96]
    sub             v2.8h, v18.8h, v22.8h
    sdot            z0.d, z2.h, z2.h
    sub             v3.8h, v19.8h, v23.8h
    sdot            z1.d, z3.h, z3.h

    add             x0, x0, x1
    add             x2, x2, x3
    cbnz            w12, .Loop_sse_ss_64

    add             v0.2d, v0.2d, v1.2d
    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc

function PFX(pixel_ssd_s_4x4_sve)
    movi            v0.4s, #0
    add             x1, x1, x1

    ldr             d16, [x0]
    ldr             d17, [x0, x1]
    sdot            z0.d, z16.h, z16.h
    sdot            z0.d, z17.h, z17.h
    add             x0, x0, x1, lsl #1
    ldr             d16, [x0]
    ldr             d17, [x0, x1]
    sdot            z0.d, z16.h, z16.h
    sdot            z0.d, z17.h, z17.h

    fmov            w0, s0
    ret
endfunc

function PFX(pixel_ssd_s_8x8_sve)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1

.rept 4
    ld1             {v16.8h}, [x0], x1
    sdot            z0.d, z16.h, z16.h
    ld1             {v17.8h}, [x0], x1
    sdot            z1.d, z17.h, z17.h
.endr

    add             v0.2d, v0.2d, v1.2d
    addp            d0, v0.2d
    fmov            w0, s0
    ret
endfunc

function PFX(pixel_ssd_s_16x16_sve)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1

    mov             w12, #16
.Loop_ssd_s_16:
    sub             w12, w12, #1

    ld1             {v16.8h-v17.8h}, [x0], x1
    sdot            z0.d, z16.h, z16.h
    sdot            z1.d, z17.h, z17.h
    cbnz            w12, .Loop_ssd_s_16

    add             v0.2d, v0.2d, v1.2d
    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc

function PFX(pixel_ssd_s_32x32_sve)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1

    mov             w12, #32
.Loop_ssd_s_32:
    sub             w12, w12, #1

    ldp             q16, q17, [x0]
    sdot            z0.d, z16.h, z16.h
    sdot            z1.d, z17.h, z17.h
    ldp             q16, q17, [x0, #32]
    sdot            z0.d, z16.h, z16.h
    sdot            z1.d, z17.h, z17.h

    add             x0, x0, x1
    cbnz            w12, .Loop_ssd_s_32

    add             v0.2d, v0.2d, v1.2d
    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc

function PFX(pixel_ssd_s_64x64_sve)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1

    mov             w12, #64
.Loop_ssd_s_64:
    sub             w12, w12, #1

    ldp             q16, q17, [x0]
    sdot            z0.d, z16.h, z16.h
    sdot            z1.d, z17.h, z17.h
    ldp             q16, q17, [x0, #32]
    sdot            z0.d, z16.h, z16.h
    sdot            z1.d, z17.h, z17.h
    ldp             q16, q17, [x0, #64]
    sdot            z0.d, z16.h, z16.h
    sdot            z1.d, z17.h, z17.h
    ldp             q16, q17,  [x0, #96]
    sdot            z0.d, z16.h, z16.h
    sdot            z1.d, z17.h, z17.h

    add             x0, x0, x1
    cbnz            w12, .Loop_ssd_s_64

    add             v0.2d, v0.2d, v1.2d
    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc
