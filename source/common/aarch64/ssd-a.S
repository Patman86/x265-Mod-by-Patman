/*****************************************************************************
 * Copyright (C) 2021 MulticoreWare, Inc
 *
 * Authors: Sebastian Pop <spop@amazon.com>
 *          Hari Limaye <hari.limaye@arm.com>
 *          Gerda Zsejke More <gerdazsejke.more@arm.com>
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

#if !HIGH_BIT_DEPTH
// Fully unrolled.
.macro SSE_PP_4xN h
function PFX(pixel_sse_pp_4x\h\()_neon)
    movi            v0.4s, #0
.rept \h / 2
    ldr             s16, [x0]
    ldr             s17, [x2]
    add             x0, x0, x1
    add             x2, x2, x3
    ld1             {v16.s}[1], [x0], x1
    ld1             {v17.s}[1], [x2], x3

    uabd            v1.8b, v16.8b, v17.8b
    umull           v20.8h, v1.8b, v1.8b
    uadalp          v0.4s, v20.8h
.endr
    ret_v0_w0
endfunc
.endm

SSE_PP_4xN 4
SSE_PP_4xN 8

// Fully unrolled.
.macro SSE_PP_8xN h
function PFX(pixel_sse_pp_8x\h\()_neon)
    movi            v0.4s, #0
.rept \h
    ld1             {v16.8b}, [x0], x1
    ld1             {v17.8b}, [x2], x3

    uabd            v1.8b, v16.8b, v17.8b
    umull           v20.8h, v1.8b, v1.8b
    uadalp          v0.4s, v20.8h
.endr
    ret_v0_w0
endfunc
.endm

SSE_PP_8xN 8
SSE_PP_8xN 16

// Fully unrolled.
.macro SSE_PP_16xN h
function PFX(pixel_sse_pp_16x\h\()_neon)
    movi            v0.4s, #0
    movi            v1.4s, #0
.rept \h
    ld1             {v16.16b}, [x0], x1
    ld1             {v17.16b}, [x2], x3

    uabd            v2.16b, v16.16b, v17.16b
    umull           v20.8h, v2.8b, v2.8b
    uadalp          v0.4s, v20.8h
    umull2          v21.8h, v2.16b, v2.16b
    uadalp          v1.4s, v21.8h
.endr
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc
.endm

SSE_PP_16xN 16
SSE_PP_16xN 32

// Loop unrolled to process 4 rows per iteration.
function PFX(pixel_sse_pp_32xh_neon), export=0
    movi            v0.4s, #0
    movi            v1.4s, #0
.Loop_sse_pp_32xh:
    sub             w4, w4, #1
.rept 4
    ld1             {v16.16b,v17.16b}, [x0], x1
    ld1             {v18.16b,v19.16b}, [x2], x3

    uabd            v2.16b, v16.16b, v18.16b
    uabd            v3.16b, v17.16b, v19.16b

    umull           v20.8h, v2.8b, v2.8b
    umull           v22.8h, v3.8b, v3.8b
    umull2          v21.8h, v2.16b, v2.16b
    umull2          v23.8h, v3.16b, v3.16b

    uadalp          v0.4s, v20.8h
    uadalp          v1.4s, v21.8h
    uadalp          v0.4s, v22.8h
    uadalp          v1.4s, v23.8h
.endr
    cbnz            w4, .Loop_sse_pp_32xh
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

.macro SSE_PP_32xN h
function PFX(pixel_sse_pp_32x\h\()_neon)
    mov             w4, \h / 4
    b               PFX(pixel_sse_pp_32xh_neon)
endfunc
.endm

SSE_PP_32xN 32
SSE_PP_32xN 64

// Loop unrolled to process 4 rows per iteration.
function PFX(pixel_sse_pp_64x64_neon)
    mov             w12, #16
    movi            v0.4s, #0
    movi            v1.4s, #0

.Loop_sse_pp_64:
    sub             w12, w12, #1
.rept 4
    ld1             {v16.16b-v19.16b}, [x0], x1
    ld1             {v20.16b-v23.16b}, [x2], x3

    uabd            v2.16b, v16.16b, v20.16b
    uabd            v3.16b, v17.16b, v21.16b
    uabd            v4.16b, v18.16b, v22.16b
    uabd            v5.16b, v19.16b, v23.16b

    umull           v24.8h, v2.8b, v2.8b
    umull           v28.8h, v4.8b, v4.8b
    umull           v26.8h, v3.8b, v3.8b
    umull           v30.8h, v5.8b, v5.8b
    umull2          v25.8h, v2.16b, v2.16b
    umull2          v27.8h, v3.16b, v3.16b
    umull2          v29.8h, v4.16b, v4.16b
    umull2          v31.8h, v5.16b, v5.16b

    uadalp          v0.4s, v24.8h
    uadalp          v1.4s, v25.8h
    uadalp          v0.4s, v26.8h
    uadalp          v1.4s, v27.8h
    uadalp          v0.4s, v28.8h
    uadalp          v1.4s, v29.8h
    uadalp          v0.4s, v30.8h
    uadalp          v1.4s, v31.8h
.endr
    cbnz            w12, .Loop_sse_pp_64
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

function PFX(pixel_sse_ss_4x4_neon)
    add             x1, x1, x1
    add             x3, x3, x3
    ld1             {v16.8b}, [x0], x1
    ld1             {v17.8b}, [x2], x3
    sub             v2.4h, v16.4h, v17.4h
    ld1             {v16.8b}, [x0], x1
    ld1             {v17.8b}, [x2], x3
    smull           v0.4s, v2.4h, v2.4h
    sub             v2.4h, v16.4h, v17.4h
    ld1             {v16.8b}, [x0], x1
    ld1             {v17.8b}, [x2], x3
    smlal           v0.4s, v2.4h, v2.4h
    sub             v2.4h, v16.4h, v17.4h
    ld1             {v16.8b}, [x0], x1
    smlal           v0.4s, v2.4h, v2.4h
    ld1             {v17.8b}, [x2], x3
    sub             v2.4h, v16.4h, v17.4h
    smlal           v0.4s, v2.4h, v2.4h
    ret_v0_w0
endfunc

function PFX(pixel_sse_ss_8x8_neon)
    add             x1, x1, x1
    add             x3, x3, x3
    ld1             {v16.16b}, [x0], x1
    ld1             {v17.16b}, [x2], x3
    sub             v2.8h, v16.8h, v17.8h
    ld1             {v16.16b}, [x0], x1
    ld1             {v17.16b}, [x2], x3
    smull           v0.4s, v2.4h, v2.4h
    smull2          v1.4s, v2.8h, v2.8h
    sub             v2.8h, v16.8h, v17.8h
.rept 6
    ld1             {v16.16b}, [x0], x1
    ld1             {v17.16b}, [x2], x3
    smlal           v0.4s, v2.4h, v2.4h
    smlal2          v1.4s, v2.8h, v2.8h
    sub             v2.8h, v16.8h, v17.8h
.endr
    smlal           v0.4s, v2.4h, v2.4h
    smlal2          v1.4s, v2.8h, v2.8h
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

function PFX(pixel_sse_ss_16x16_neon)
    add             x1, x1, x1
    add             x3, x3, x3
    mov             w12, #4
    movi            v0.16b, #0
    movi            v1.16b, #0
.Loop_sse_ss_16:
    sub             w12, w12, #1
.rept 4
    ld1             {v16.16b, v17.16b}, [x0], x1
    ld1             {v18.16b, v19.16b}, [x2], x3
    sub             v2.8h, v16.8h, v18.8h
    sub             v3.8h, v17.8h, v19.8h
    smlal           v0.4s, v2.4h, v2.4h
    smlal2          v1.4s, v2.8h, v2.8h
    smlal           v0.4s, v3.4h, v3.4h
    smlal2          v1.4s, v3.8h, v3.8h
.endr
    cbnz            w12, .Loop_sse_ss_16
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

function PFX(pixel_sse_ss_32x32_neon)
    add             x1, x1, x1
    add             x3, x3, x3

    mov             w12, #8
    movi            v0.16b, #0
    movi            v1.16b, #0
.Loop_sse_ss_32:
    sub             w12, w12, #1
.rept 4
    ld1             {v16.16b-v19.16b}, [x0], x1
    ld1             {v20.16b-v23.16b}, [x2], x3
    sub             v2.8h, v16.8h, v20.8h
    sub             v3.8h, v17.8h, v21.8h
    sub             v4.8h, v18.8h, v22.8h
    sub             v5.8h, v19.8h, v23.8h
    smlal           v0.4s, v2.4h, v2.4h
    smlal2          v1.4s, v2.8h, v2.8h
    smlal           v0.4s, v3.4h, v3.4h
    smlal2          v1.4s, v3.8h, v3.8h
    smlal           v0.4s, v4.4h, v4.4h
    smlal2          v1.4s, v4.8h, v4.8h
    smlal           v0.4s, v5.4h, v5.4h
    smlal2          v1.4s, v5.8h, v5.8h
.endr
    cbnz            w12, .Loop_sse_ss_32
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

function PFX(pixel_sse_ss_64x64_neon)
    add             x1, x1, x1
    add             x3, x3, x3
    sub             x1, x1, #64
    sub             x3, x3, #64

    mov             w12, #32
    movi            v0.16b, #0
    movi            v1.16b, #0
.Loop_sse_ss_64:
    sub             w12, w12, #1
.rept 2
    ld1             {v16.16b-v19.16b}, [x0], #64
    ld1             {v20.16b-v23.16b}, [x2], #64
    sub             v2.8h, v16.8h, v20.8h
    sub             v3.8h, v17.8h, v21.8h
    sub             v4.8h, v18.8h, v22.8h
    sub             v5.8h, v19.8h, v23.8h
    ld1             {v16.16b-v19.16b}, [x0], x1
    ld1             {v20.16b-v23.16b}, [x2], x3
    smlal           v0.4s, v2.4h, v2.4h
    smlal2          v1.4s, v2.8h, v2.8h
    smlal           v0.4s, v3.4h, v3.4h
    smlal2          v1.4s, v3.8h, v3.8h
    smlal           v0.4s, v4.4h, v4.4h
    smlal2          v1.4s, v4.8h, v4.8h
    smlal           v0.4s, v5.4h, v5.4h
    smlal2          v1.4s, v5.8h, v5.8h
    sub             v2.8h, v16.8h, v20.8h
    sub             v3.8h, v17.8h, v21.8h
    sub             v4.8h, v18.8h, v22.8h
    sub             v5.8h, v19.8h, v23.8h
    smlal           v0.4s, v2.4h, v2.4h
    smlal2          v1.4s, v2.8h, v2.8h
    smlal           v0.4s, v3.4h, v3.4h
    smlal2          v1.4s, v3.8h, v3.8h
    smlal           v0.4s, v4.4h, v4.4h
    smlal2          v1.4s, v4.8h, v4.8h
    smlal           v0.4s, v5.4h, v5.4h
    smlal2          v1.4s, v5.8h, v5.8h
.endr
    cbnz            w12, .Loop_sse_ss_64
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

function PFX(pixel_ssd_s_4x4_neon)
    add             x1, x1, x1
    ld1             {v4.8b}, [x0], x1
    ld1             {v5.8b}, [x0], x1
    ld1             {v6.8b}, [x0], x1
    ld1             {v7.8b}, [x0]
    smull           v0.4s, v4.4h, v4.4h
    smull           v1.4s, v5.4h, v5.4h
    smlal           v0.4s, v6.4h, v6.4h
    smlal           v1.4s, v7.4h, v7.4h
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

function PFX(pixel_ssd_s_8x8_neon)
    add             x1, x1, x1
    ld1             {v4.16b}, [x0], x1
    ld1             {v5.16b}, [x0], x1
    smull           v0.4s, v4.4h, v4.4h
    smull2          v1.4s, v4.8h, v4.8h
    smlal           v0.4s, v5.4h, v5.4h
    smlal2          v1.4s, v5.8h, v5.8h
.rept 3
    ld1             {v4.16b}, [x0], x1
    ld1             {v5.16b}, [x0], x1
    smlal           v0.4s, v4.4h, v4.4h
    smlal2          v1.4s, v4.8h, v4.8h
    smlal           v0.4s, v5.4h, v5.4h
    smlal2          v1.4s, v5.8h, v5.8h
.endr
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

function PFX(pixel_ssd_s_16x16_neon)
    add             x1, x1, x1
    mov             w12, #4
    movi            v0.16b, #0
    movi            v1.16b, #0
.Loop_ssd_s_16:
    sub             w12, w12, #1
.rept 2
    ld1             {v4.16b,v5.16b}, [x0], x1
    ld1             {v6.16b,v7.16b}, [x0], x1
    smlal           v0.4s, v4.4h, v4.4h
    smlal2          v1.4s, v4.8h, v4.8h
    smlal           v0.4s, v5.4h, v5.4h
    smlal2          v1.4s, v5.8h, v5.8h
    smlal           v0.4s, v6.4h, v6.4h
    smlal2          v1.4s, v6.8h, v6.8h
    smlal           v0.4s, v7.4h, v7.4h
    smlal2          v1.4s, v7.8h, v7.8h
.endr
    cbnz            w12, .Loop_ssd_s_16
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

function PFX(pixel_ssd_s_32x32_neon)
    add             x1, x1, x1
    mov             w12, #8
    movi            v0.16b, #0
    movi            v1.16b, #0
.Loop_ssd_s_32:
    sub             w12, w12, #1
.rept 4
    ld1             {v4.16b-v7.16b}, [x0], x1
    smlal           v0.4s, v4.4h, v4.4h
    smlal2          v1.4s, v4.8h, v4.8h
    smlal           v0.4s, v5.4h, v5.4h
    smlal2          v1.4s, v5.8h, v5.8h
    smlal           v0.4s, v6.4h, v6.4h
    smlal2          v1.4s, v6.8h, v6.8h
    smlal           v0.4s, v7.4h, v7.4h
    smlal2          v1.4s, v7.8h, v7.8h
.endr
    cbnz            w12, .Loop_ssd_s_32
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

function PFX(pixel_ssd_s_64x64_neon)
    add             x1, x1, x1
    sub             x1, x1, #64
    sub             x3, x3, #64

    mov             w12, #32
    movi            v0.16b, #0
    movi            v1.16b, #0
.Loop_ssd_ss_64:
    sub             w12, w12, #1
.rept 2
    ld1             {v16.16b-v19.16b}, [x0], #64
    ld1             {v20.16b-v23.16b}, [x0], x1
    smlal           v0.4s, v16.4h, v16.4h
    smlal2          v1.4s, v16.8h, v16.8h
    smlal           v0.4s, v17.4h, v17.4h
    smlal2          v1.4s, v17.8h, v17.8h
    smlal           v0.4s, v18.4h, v18.4h
    smlal2          v1.4s, v18.8h, v18.8h
    smlal           v0.4s, v19.4h, v19.4h
    smlal2          v1.4s, v19.8h, v19.8h
    smlal           v0.4s, v20.4h, v20.4h
    smlal2          v1.4s, v20.8h, v20.8h
    smlal           v0.4s, v21.4h, v21.4h
    smlal2          v1.4s, v21.8h, v21.8h
    smlal           v0.4s, v22.4h, v22.4h
    smlal2          v1.4s, v22.8h, v22.8h
    smlal           v0.4s, v23.4h, v23.4h
    smlal2          v1.4s, v23.8h, v23.8h
.endr
    cbnz            w12, .Loop_ssd_ss_64
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

#else // HIGH_BIT_DEPTH

.macro SSE_PP_4x2
    ldr             d16, [x0]
    ldr             d17, [x2]
    ldr             d18, [x0, x1]
    ldr             d19, [x2, x3]
    uabd            v2.4h, v16.4h, v17.4h
    uabd            v3.4h, v18.4h, v19.4h
    umlal           v0.4s, v2.4h, v2.4h
    umlal           v0.4s, v3.4h, v3.4h
.endm

.macro SSE_PP_4xN h
function PFX(pixel_sse_pp_4x\h\()_neon)
    movi            v0.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

.rept (\h / 2) - 1
    SSE_PP_4x2
    add             x0, x0, x1, lsl #1
    add             x2, x2, x3, lsl #1
.endr
    SSE_PP_4x2

    ret_v0_w0
endfunc
.endm

SSE_PP_4xN 4
SSE_PP_4xN 8

.macro SSE_PP_8xN h
function PFX(pixel_sse_pp_8x\h\()_neon)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3
.rept \h
    ld1             {v16.8h}, [x0], x1
    ld1             {v17.8h}, [x2], x3
    uabd            v2.8h, v16.8h, v17.8h
    umlal           v0.4s, v2.4h, v2.4h
    umlal2          v1.4s, v2.8h, v2.8h
.endr
    add             v0.4s, v0.4s, v1.4s

    ret_v0_w0
endfunc
.endm

SSE_PP_8xN 8
SSE_PP_8xN 16

.macro SSE_PP_16xN h
function PFX(pixel_sse_pp_16x\h\()_neon)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3
.rept \h
    ld1             {v16.8h-v17.8h}, [x0], x1
    ld1             {v18.8h-v19.8h}, [x2], x3
    uabd            v2.8h, v16.8h, v18.8h
    umlal           v0.4s, v2.4h, v2.4h
    umlal2          v1.4s, v2.8h, v2.8h
    uabd            v3.8h, v17.8h, v19.8h
    umlal           v0.4s, v3.4h, v3.4h
    umlal2          v1.4s, v3.8h, v3.8h
.endr

.if \h == 16
    add             v0.4s, v0.4s, v1.4s
    addv            s0, v0.4s
    fmov            w0, s0
.else
    uaddlv          d0, v0.4s
    uaddlv          d1, v1.4s
    add             d0, d0, d1
    fmov            x0, d0
.endif

    ret
endfunc
.endm

SSE_PP_16xN 16
SSE_PP_16xN 32

.macro SSE_PP_32xN h
function  PFX(pixel_sse_pp_32x\h\()_neon)
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
    umlal           v0.4s, v2.4h, v2.4h
    umlal2          v1.4s, v2.8h, v2.8h
    uabd            v3.8h, v17.8h, v21.8h
    umlal           v0.4s, v3.4h, v3.4h
    umlal2          v1.4s, v3.8h, v3.8h

    ldp             q18, q19, [x0, #32]
    ldp             q22, q23, [x2, #32]
    uabd            v2.8h, v18.8h, v22.8h
    umlal           v0.4s, v2.4h, v2.4h
    umlal2          v1.4s, v2.8h, v2.8h
    uabd            v3.8h, v19.8h, v23.8h
    umlal           v0.4s, v3.4h, v3.4h
    umlal2          v1.4s, v3.8h, v3.8h

    add             x0, x0, x1
    add             x2, x2, x3
    cbnz            w12, .Loop_sse_pp_32x\h

    uaddlv          d0, v0.4s
    uaddlv          d1, v1.4s
    add             d0, d0, d1
    fmov            x0, d0
    ret
endfunc
.endm

SSE_PP_32xN 32
SSE_PP_32xN 64

function PFX(pixel_sse_pp_64x64_neon)
    mov             w12, #64

    movi            v0.4s, #0
    movi            v1.4s, #0
    movi            v2.4s, #0
    movi            v3.4s, #0

    add             x1, x1, x1
    add             x3, x3, x3
.Loop_sse_pp_64x1:
    sub             w12, w12, #1

    ld1             {v16.8h-v17.8h}, [x0]
    ld1             {v20.8h-v21.8h}, [x2]
    uabd            v4.8h, v16.8h, v20.8h
    umlal           v0.4s, v4.4h, v4.4h
    umlal2          v1.4s, v4.8h, v4.8h
    uabd            v5.8h, v17.8h, v21.8h
    umlal           v0.4s, v5.4h, v5.4h
    umlal2          v1.4s, v5.8h, v5.8h

    ldp             q18, q19, [x0, #32]
    ldp             q22, q23, [x2, #32]
    uabd            v6.8h, v18.8h, v22.8h
    umlal           v2.4s, v6.4h, v6.4h
    umlal2          v3.4s, v6.8h, v6.8h
    uabd            v7.8h, v19.8h, v23.8h
    umlal           v2.4s, v7.4h, v7.4h
    umlal2          v3.4s, v7.8h, v7.8h

    ldp             q16, q17, [x0, #64]
    ldp             q20, q21, [x2, #64]
    uabd            v4.8h, v16.8h, v20.8h
    umlal           v0.4s, v4.4h, v4.4h
    umlal2          v1.4s, v4.8h, v4.8h
    uabd            v5.8h, v17.8h, v21.8h
    umlal           v0.4s, v5.4h, v5.4h
    umlal2          v1.4s, v5.8h, v5.8h

    ldp             q18, q19, [x0, #96]
    ldp             q22, q23, [x2, #96]
    uabd            v6.8h, v18.8h, v22.8h
    umlal           v2.4s, v6.4h, v6.4h
    umlal2          v3.4s, v6.8h, v6.8h
    uabd            v7.8h, v19.8h, v23.8h
    umlal           v2.4s, v7.4h, v7.4h
    umlal2          v3.4s, v7.8h, v7.8h

    add             x0, x0, x1
    add             x2, x2, x3
    cbnz            w12, .Loop_sse_pp_64x1

    uaddlv          d0, v0.4s
    uaddlv          d1, v1.4s
    add             d0, d0, d1
    uaddlv          d2, v2.4s
    uaddlv          d3, v3.4s
    add             d2, d2, d3

    add             d0, d0, d2
    fmov            x0, d0
    ret
endfunc

.macro SSE_SS_4x2
    ldr             d16, [x0]
    ldr             d17, [x2]
    ldr             d18, [x0, x1]
    ldr             d19, [x2, x3]
    sub             v2.4h, v16.4h, v17.4h
    sub             v3.4h, v18.4h, v19.4h
    smlal           v0.4s, v2.4h, v2.4h
    smlal           v0.4s, v3.4h, v3.4h
.endm

function PFX(pixel_sse_ss_4x4_neon)
    movi            v0.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

    SSE_SS_4x2
    add             x0, x0, x1, lsl 1
    add             x2, x2, x3, lsl 1
    SSE_SS_4x2

    ret_v0_w0
endfunc

function PFX(pixel_sse_ss_8x8_neon)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3
.rept 8
    ld1             {v16.8h}, [x0], x1
    ld1             {v17.8h}, [x2], x3
    sub             v2.8h, v16.8h, v17.8h
    smlal           v0.4s, v2.4h, v2.4h
    smlal2          v1.4s, v2.8h, v2.8h
.endr
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

function PFX(pixel_sse_ss_16x16_neon)
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
    smlal           v0.4s, v2.4h, v2.4h
    smlal2          v1.4s, v2.8h, v2.8h
    sub             v3.8h, v17.8h, v19.8h
    smlal           v0.4s, v3.4h, v3.4h
    smlal2          v1.4s, v3.8h, v3.8h
    cbnz            w12, .Loop_sse_ss_16

    add             v0.4s, v0.4s, v1.4s
    uaddlv          d0, v0.4s
    fmov            x0, d0
    ret
endfunc

function PFX(pixel_sse_ss_32x32_neon)
    movi            v0.4s, #0
    movi            v1.4s, #0
    movi            v2.4s, #0
    movi            v3.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3
    mov             w12, #32
.Loop_sse_ss_32:
    sub             w12, w12, #1
    ld1             {v16.8h-v17.8h}, [x0]
    ld1             {v18.8h-v19.8h}, [x2]
    sub             v4.8h, v16.8h, v18.8h
    smlal           v0.4s, v4.4h, v4.4h
    smlal2          v1.4s, v4.8h, v4.8h
    sub             v5.8h, v17.8h, v19.8h
    smlal           v0.4s, v5.4h, v5.4h
    smlal2          v1.4s, v5.8h, v5.8h

    ldp             q16, q17, [x0, #32]
    ldp             q18, q19, [x2, #32]
    sub             v6.8h, v16.8h, v18.8h
    smlal           v2.4s, v6.4h, v6.4h
    smlal2          v3.4s, v6.8h, v6.8h
    sub             v7.8h, v17.8h, v19.8h
    smlal           v2.4s, v7.4h, v7.4h
    smlal2          v3.4s, v7.8h, v7.8h

    add             x0, x0, x1
    add             x2, x2, x3
    cbnz            w12, .Loop_sse_ss_32

    uaddlp          v0.2d, v0.4s
    uadalp          v0.2d, v1.4s
    uadalp          v0.2d, v2.4s
    uadalp          v0.2d, v3.4s
    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc

function PFX(pixel_sse_ss_64x64_neon)
    movi            v0.4s, #0
    movi            v31.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

    mov             w12, #2
.Loop_sse_ss_64x32:
    sub             w12, w12, #1
    movi            v1.4s, #0
    movi            v2.4s, #0
    movi            v3.4s, #0
    movi            v4.4s, #0
    movi            v5.4s, #0
    movi            v6.4s, #0
    movi            v7.4s, #0
    movi            v8.4s, #0
    mov             w11, #32
.Loop_sse_ss_64x1:
    sub             w11, w11, #1

    ld1             {v16.8h-v17.8h}, [x0]
    ld1             {v18.8h-v19.8h}, [x2]
    sub             v20.8h, v16.8h, v18.8h
    smlal           v1.4s, v20.4h, v20.4h
    smlal2          v2.4s, v20.8h, v20.8h
    sub             v21.8h, v17.8h, v19.8h
    smlal           v1.4s, v21.4h, v21.4h
    smlal2          v2.4s, v21.8h, v21.8h

    ldp             q16, q17, [x0, #32]
    ldp             q18, q19, [x2, #32]
    sub             v22.8h, v16.8h, v18.8h
    smlal           v3.4s, v22.4h, v22.4h
    smlal2          v4.4s, v22.8h, v22.8h
    sub             v23.8h, v17.8h, v19.8h
    smlal           v3.4s, v23.4h, v23.4h
    smlal2          v4.4s, v23.8h, v23.8h

    ldp             q16, q17, [x0, #64]
    ldp             q18, q19, [x2, #64]
    sub             v20.8h, v16.8h, v18.8h
    smlal           v5.4s, v20.4h, v20.4h
    smlal2          v6.4s, v20.8h, v20.8h
    sub             v21.8h, v17.8h, v19.8h
    smlal           v5.4s, v21.4h, v21.4h
    smlal2          v6.4s, v21.8h, v21.8h

    ldp             q16, q17, [x0, #96]
    ldp             q18, q19, [x2, #96]
    sub             v22.8h, v16.8h, v18.8h
    smlal           v7.4s, v22.4h, v22.4h
    smlal2          v8.4s, v22.8h, v22.8h
    sub             v23.8h, v17.8h, v19.8h
    smlal           v7.4s, v23.4h, v23.4h
    smlal2          v8.4s, v23.8h, v23.8h

    add             x0, x0, x1
    add             x2, x2, x3
    cbnz            w11, .Loop_sse_ss_64x1

    uadalp          v0.2d, v1.4s
    uadalp          v0.2d, v2.4s
    uadalp          v0.2d, v3.4s
    uadalp          v0.2d, v4.4s
    uadalp          v0.2d, v5.4s
    uadalp          v0.2d, v6.4s
    uadalp          v0.2d, v7.4s
    uadalp          v0.2d, v8.4s
    cbnz            w12, .Loop_sse_ss_64x32

    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc

function PFX(pixel_ssd_s_4x4_neon)
    movi            v0.4s, #0
    add             x1, x1, x1

    ldr             d16, [x0]
    ldr             d17, [x0, x1]
    smlal           v0.4s, v16.4h, v16.4h
    smlal           v0.4s, v17.4h, v17.4h
    add             x0, x0, x1, lsl 1
    ldr             d16, [x0]
    ldr             d17, [x0, x1]
    smlal           v0.4s, v16.4h, v16.4h
    smlal           v0.4s, v17.4h, v17.4h

    ret_v0_w0
endfunc

function PFX(pixel_ssd_s_8x8_neon)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1
.rept 8
    ld1             {v16.8h}, [x0], x1
    smlal           v0.4s, v16.4h, v16.4h
    smlal2          v1.4s, v16.8h, v16.8h
.endr
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
endfunc

function PFX(pixel_ssd_s_16x16_neon)
    movi            v0.4s, #0
    movi            v1.4s, #0
    add             x1, x1, x1

    mov             w12, #16
.Loop_ssd_s_16:
    sub             w12, w12, #1
    ld1             {v16.8h-v17.8h}, [x0], x1
    smlal           v0.4s, v16.4h, v16.4h
    smlal2          v1.4s, v16.8h, v16.8h
    smlal           v0.4s, v17.4h, v17.4h
    smlal2          v1.4s, v17.8h, v17.8h
    cbnz            w12, .Loop_ssd_s_16

    add             v0.4s, v0.4s, v1.4s
    uaddlv          d0, v0.4s
    fmov            x0, d0
    ret
endfunc

function PFX(pixel_ssd_s_32x32_neon)
    movi            v0.4s, #0
    movi            v1.4s, #0
    movi            v2.4s, #0
    movi            v3.4s, #0
    add             x1, x1, x1

    mov             w12, #32
.Loop_ssd_s_32:
    sub             w12, w12, #1

    ldp             q16, q17, [x0]
    smlal           v0.4s, v16.4h, v16.4h
    smlal2          v1.4s, v16.8h, v16.8h
    smlal           v0.4s, v17.4h, v17.4h
    smlal2          v1.4s, v17.8h, v17.8h

    ldp             q16, q17, [x0, #32]
    smlal           v2.4s, v16.4h, v16.4h
    smlal2          v3.4s, v16.8h, v16.8h
    smlal           v2.4s, v17.4h, v17.4h
    smlal2          v3.4s, v17.8h, v17.8h

    add             x0, x0, x1
    cbnz            w12, .Loop_ssd_s_32

    uaddlp          v0.2d, v0.4s
    uadalp          v0.2d, v1.4s
    uadalp          v0.2d, v2.4s
    uadalp          v0.2d, v3.4s
    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc

function PFX(pixel_ssd_s_64x64_neon)
    movi            v0.4s, #0
    movi            v31.4s, #0
    add             x1, x1, x1
    add             x3, x3, x3

    mov             w12, #2
.Loop_ssd_s_64x32:
    sub             w12, w12, #1
    movi            v1.4s, #0
    movi            v2.4s, #0
    movi            v3.4s, #0
    movi            v4.4s, #0
    mov             w11, #32
.Loop_ssd_s_64x1:
    sub             w11, w11, #1

    ldp             q16, q17, [x0]
    smlal           v1.4s, v16.4h, v16.4h
    smlal2          v2.4s, v16.8h, v16.8h
    smlal           v1.4s, v17.4h, v17.4h
    smlal2          v2.4s, v17.8h, v17.8h

    ldp             q16, q17, [x0, #32]
    smlal           v3.4s, v16.4h, v16.4h
    smlal2          v4.4s, v16.8h, v16.8h
    smlal           v3.4s, v17.4h, v17.4h
    smlal2          v4.4s, v17.8h, v17.8h

    ldp             q16, q17, [x0, #64]
    smlal           v1.4s, v16.4h, v16.4h
    smlal2          v2.4s, v16.8h, v16.8h
    smlal           v1.4s, v17.4h, v17.4h
    smlal2          v2.4s, v17.8h, v17.8h

    ldp             q16, q17, [x0, #96]
    smlal           v3.4s, v16.4h, v16.4h
    smlal2          v4.4s, v16.8h, v16.8h
    smlal           v3.4s, v17.4h, v17.4h
    smlal2          v4.4s, v17.8h, v17.8h

    add             x0, x0, x1
    cbnz            w11, .Loop_ssd_s_64x1

    uadalp          v0.2d, v1.4s
    uadalp          v0.2d, v2.4s
    uadalp          v0.2d, v3.4s
    uadalp          v0.2d, v4.4s
    cbnz            w12, .Loop_ssd_s_64x32

    addp            d0, v0.2d
    fmov            x0, d0
    ret
endfunc

#endif // !HIGH_BIT_DEPTH
