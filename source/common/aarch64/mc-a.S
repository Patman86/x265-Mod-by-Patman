/*****************************************************************************
 * Copyright (C) 2020-2021 MulticoreWare, Inc
 *
 * Authors: Hongbin Liu <liuhongbin1@huawei.com>
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
#include "mc-a-common.S"

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4

.text

.macro pixel_avg_pp_4xN_neon h
function PFX(pixel_avg_pp_4x\h\()_neon)
.rept \h
    ldr             s0, [x2]
    ldr             s1, [x4]
    add             x2, x2, x3
    add             x4, x4, x5
    urhadd          v2.8b, v0.8b, v1.8b
    str             s2, [x0]
    add             x0, x0, x1
.endr
    ret
endfunc
.endm

pixel_avg_pp_4xN_neon 4
pixel_avg_pp_4xN_neon 8
pixel_avg_pp_4xN_neon 16

.macro pixel_avg_pp_8xN_neon h
function PFX(pixel_avg_pp_8x\h\()_neon)
.rept \h
    ld1             {v0.8b}, [x2], x3
    ld1             {v1.8b}, [x4], x5
    urhadd          v2.8b, v0.8b, v1.8b
    st1             {v2.8b}, [x0], x1
.endr
    ret
endfunc
.endm

pixel_avg_pp_8xN_neon 4
pixel_avg_pp_8xN_neon 8
pixel_avg_pp_8xN_neon 16
pixel_avg_pp_8xN_neon 32

function PFX(pixel_avg_pp_12x16_neon)
    sub             x1, x1, #4
    sub             x3, x3, #4
    sub             x5, x5, #4
.rept 16
    ldr             s0, [x2], #4
    ld1             {v1.8b}, [x2], x3
    ldr             s2, [x4], #4
    ld1             {v3.8b}, [x4], x5
    urhadd          v4.8b, v0.8b, v2.8b
    urhadd          v5.8b, v1.8b, v3.8b
    str             s4, [x0], #4
    st1             {v5.8b}, [x0], x1
.endr
    ret
endfunc

.macro pixel_avg_pp_16xN_neon h
function PFX(pixel_avg_pp_16x\h\()_neon)
.rept \h
    ld1             {v0.16b}, [x2], x3
    ld1             {v1.16b}, [x4], x5
    urhadd          v2.16b, v0.16b, v1.16b
    st1             {v2.16b}, [x0], x1
.endr
    ret
endfunc
.endm

pixel_avg_pp_16xN_neon 4
pixel_avg_pp_16xN_neon 8
pixel_avg_pp_16xN_neon 12
pixel_avg_pp_16xN_neon 16
pixel_avg_pp_16xN_neon 32

function PFX(pixel_avg_pp_16x64_neon)
    mov             w12, #8
.lpavg_16x64:
    sub             w12, w12, #1
.rept 8
    ld1             {v0.16b}, [x2], x3
    ld1             {v1.16b}, [x4], x5
    urhadd          v2.16b, v0.16b, v1.16b
    st1             {v2.16b}, [x0], x1
.endr
    cbnz            w12, .lpavg_16x64
    ret
endfunc

function PFX(pixel_avg_pp_24x32_neon)
    sub             x1, x1, #16
    sub             x3, x3, #16
    sub             x5, x5, #16
    mov             w12, #4
.lpavg_24x32:
    sub             w12, w12, #1
.rept 8
    ld1             {v0.16b}, [x2], #16
    ld1             {v1.8b}, [x2], x3
    ld1             {v2.16b}, [x4], #16
    ld1             {v3.8b}, [x4], x5
    urhadd          v0.16b, v0.16b, v2.16b
    urhadd          v1.8b, v1.8b, v3.8b
    st1             {v0.16b}, [x0], #16
    st1             {v1.8b}, [x0], x1
.endr
    cbnz            w12, .lpavg_24x32
    ret
endfunc

.macro pixel_avg_pp_32xN_neon h
function PFX(pixel_avg_pp_32x\h\()_neon)
.rept \h
    ld1             {v0.16b-v1.16b}, [x2], x3
    ld1             {v2.16b-v3.16b}, [x4], x5
    urhadd          v0.16b, v0.16b, v2.16b
    urhadd          v1.16b, v1.16b, v3.16b
    st1             {v0.16b-v1.16b}, [x0], x1
.endr
    ret
endfunc
.endm

pixel_avg_pp_32xN_neon 8
pixel_avg_pp_32xN_neon 16
pixel_avg_pp_32xN_neon 24

.macro pixel_avg_pp_32xN1_neon h
function PFX(pixel_avg_pp_32x\h\()_neon)
    mov             w12, #\h / 8
.lpavg_32x\h\():
    sub             w12, w12, #1
.rept 8
    ld1             {v0.16b-v1.16b}, [x2], x3
    ld1             {v2.16b-v3.16b}, [x4], x5
    urhadd          v0.16b, v0.16b, v2.16b
    urhadd          v1.16b, v1.16b, v3.16b
    st1             {v0.16b-v1.16b}, [x0], x1
.endr
    cbnz            w12, .lpavg_32x\h
    ret
endfunc
.endm

pixel_avg_pp_32xN1_neon 32
pixel_avg_pp_32xN1_neon 64

function PFX(pixel_avg_pp_48x64_neon)
    mov             w12, #8
.lpavg_48x64:
    sub             w12, w12, #1
.rept 8
    ld1             {v0.16b-v2.16b}, [x2], x3
    ld1             {v3.16b-v5.16b}, [x4], x5
    urhadd          v0.16b, v0.16b, v3.16b
    urhadd          v1.16b, v1.16b, v4.16b
    urhadd          v2.16b, v2.16b, v5.16b
    st1             {v0.16b-v2.16b}, [x0], x1
.endr
    cbnz            w12, .lpavg_48x64
    ret
endfunc

.macro pixel_avg_pp_64xN_neon h
function PFX(pixel_avg_pp_64x\h\()_neon)
    mov             w12, #\h / 4
.lpavg_64x\h\():
    sub             w12, w12, #1
.rept 4
    ld1             {v0.16b-v3.16b}, [x2], x3
    ld1             {v4.16b-v7.16b}, [x4], x5
    urhadd          v0.16b, v0.16b, v4.16b
    urhadd          v1.16b, v1.16b, v5.16b
    urhadd          v2.16b, v2.16b, v6.16b
    urhadd          v3.16b, v3.16b, v7.16b
    st1             {v0.16b-v3.16b}, [x0], x1
.endr
    cbnz            w12, .lpavg_64x\h
    ret
endfunc
.endm

pixel_avg_pp_64xN_neon 16
pixel_avg_pp_64xN_neon 32
pixel_avg_pp_64xN_neon 48
pixel_avg_pp_64xN_neon 64
