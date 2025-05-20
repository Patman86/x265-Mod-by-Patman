/*****************************************************************************
 * Copyright (C) 2021 MulticoreWare, Inc
 *
 * Authors: Sebastian Pop <spop@amazon.com>
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

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4

.text

// uint32_t copy_count(int16_t* coeff, const int16_t* residual, intptr_t resiStride)
function PFX(copy_cnt_4_neon)
    lsl             x2, x2, #1
    movi            v4.8b, #0
.rept 2
    ld1             {v0.8b}, [x1], x2
    ld1             {v1.8b}, [x1], x2
    stp             d0, d1, [x0], #16
    cmeq            v0.4h, v0.4h, #0
    cmeq            v1.4h, v1.4h, #0
    add             v4.4h, v4.4h, v0.4h
    add             v4.4h, v4.4h, v1.4h
.endr
    saddlv          s4, v4.4h
    fmov            w12, s4
    add             w0, w12, #16
    ret
endfunc

function PFX(copy_cnt_8_neon)
    lsl             x2, x2, #1
    movi            v4.8b, #0
.rept 4
    ld1             {v0.16b}, [x1], x2
    ld1             {v1.16b}, [x1], x2
    stp             q0, q1, [x0], #32
    cmeq            v0.8h, v0.8h, #0
    cmeq            v1.8h, v1.8h, #0
    add             v4.8h, v4.8h, v0.8h
    add             v4.8h, v4.8h, v1.8h
.endr
    saddlv          s4, v4.8h
    fmov            w12, s4
    add             w0, w12, #64
    ret
endfunc

function PFX(copy_cnt_16_neon)
    lsl             x2, x2, #1
    movi            v4.8b, #0
.rept 16
    ld1             {v0.16b-v1.16b}, [x1], x2
    st1             {v0.16b-v1.16b}, [x0], #32
    cmeq            v0.8h, v0.8h, #0
    cmeq            v1.8h, v1.8h, #0
    add             v4.8h, v4.8h, v0.8h
    add             v4.8h, v4.8h, v1.8h
.endr
    saddlv          s4, v4.8h
    fmov            w12, s4
    add             w0, w12, #256
    ret
endfunc

function PFX(copy_cnt_32_neon)
    lsl             x2, x2, #1
    movi            v4.8b, #0
.rept 32
    ld1             {v0.16b-v3.16b}, [x1], x2
    st1             {v0.16b-v3.16b}, [x0], #64
    cmeq            v0.8h, v0.8h, #0
    cmeq            v1.8h, v1.8h, #0
    cmeq            v2.8h, v2.8h, #0
    cmeq            v3.8h, v3.8h, #0
    add             v0.8h, v0.8h, v1.8h
    add             v2.8h, v2.8h, v3.8h
    add             v4.8h, v4.8h, v0.8h
    add             v4.8h, v4.8h, v2.8h
.endr
    saddlv          s4, v4.8h
    fmov            w12, s4
    add             w0, w12, #1024
    ret
endfunc

// int  count_nonzero_c(const int16_t* quantCoeff)
function PFX(count_nonzero_4_neon)
    movi            v16.16b, #1
    movi            v17.16b, #0
    trn1            v16.16b, v16.16b, v17.16b
    ldp             q0, q1, [x0]
    cmhi            v0.8h, v0.8h, v17.8h
    cmhi            v1.8h, v1.8h, v17.8h
    and             v0.16b, v0.16b, v16.16b
    and             v1.16b, v1.16b, v16.16b
    add             v0.8h, v0.8h, v1.8h
    uaddlv          s0, v0.8h
    fmov            w0, s0
    ret
endfunc

.macro COUNT_NONZERO_8
    ld1             {v0.16b-v3.16b}, [x0], #64
    ld1             {v4.16b-v7.16b}, [x0], #64
    cmhi            v0.8h, v0.8h, v17.8h
    cmhi            v1.8h, v1.8h, v17.8h
    cmhi            v2.8h, v2.8h, v17.8h
    cmhi            v3.8h, v3.8h, v17.8h
    cmhi            v4.8h, v4.8h, v17.8h
    cmhi            v5.8h, v5.8h, v17.8h
    cmhi            v6.8h, v6.8h, v17.8h
    cmhi            v7.8h, v7.8h, v17.8h
    and             v0.16b, v0.16b, v16.16b
    and             v1.16b, v1.16b, v16.16b
    and             v2.16b, v2.16b, v16.16b
    and             v3.16b, v3.16b, v16.16b
    and             v4.16b, v4.16b, v16.16b
    and             v5.16b, v5.16b, v16.16b
    and             v6.16b, v6.16b, v16.16b
    and             v7.16b, v7.16b, v16.16b
    add             v0.8h, v0.8h, v1.8h
    add             v2.8h, v2.8h, v3.8h
    add             v4.8h, v4.8h, v5.8h
    add             v6.8h, v6.8h, v7.8h
    add             v0.8h, v0.8h, v2.8h
    add             v4.8h, v4.8h, v6.8h
    add             v0.8h, v0.8h, v4.8h
.endm

function PFX(count_nonzero_8_neon)
    movi            v16.16b, #1
    movi            v17.16b, #0
    trn1            v16.16b, v16.16b, v17.16b
    COUNT_NONZERO_8
    uaddlv          s0, v0.8h
    fmov            w0, s0
    ret
endfunc

function PFX(count_nonzero_16_neon)
    movi            v16.16b, #1
    movi            v17.16b, #0
    trn1            v16.16b, v16.16b, v17.16b
    movi            v18.16b, #0
.rept 4
    COUNT_NONZERO_8
    add             v18.16b, v18.16b, v0.16b
.endr
    uaddlv          s0, v18.8h
    fmov            w0, s0
    ret
endfunc

function PFX(count_nonzero_32_neon)
    movi            v16.16b, #1
    movi            v17.16b, #0
    trn1            v16.16b, v16.16b, v17.16b
    movi            v18.16b, #0
    mov             w12, #16
.Loop_count_nonzero_32:
    sub             w12, w12, #1
    COUNT_NONZERO_8
    add             v18.16b, v18.16b, v0.16b
    cbnz            w12, .Loop_count_nonzero_32

    uaddlv          s0, v18.8h
    fmov            w0, s0
    ret
endfunc
