/*****************************************************************************
 * Copyright (C) 2022-2023 MulticoreWare, Inc
 *
 * Authors: David Chen <david.chen@myais.com.cn>
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

#include "asm-sve.S"
#include "mc-a-common.S"

.arch armv8-a+sve2

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4

.text

function PFX(pixel_avg_pp_12x16_sve2)
    sub             x1, x1, #4
    sub             x3, x3, #4
    sub             x5, x5, #4
    ptrue           p0.s, vl1
    ptrue           p1.b, vl8
    mov             x11, #4
.rept 16
    ld1w            {z0.s}, p0/z, [x2]
    ld1b            {z1.b}, p1/z, [x2, x11]
    ld1w            {z2.s}, p0/z, [x4]
    ld1b            {z3.b}, p1/z, [x4, x11]
    add             x2, x2, #4
    add             x2, x2, x3
    add             x4, x4, #4
    add             x4, x4, x5
    urhadd          z0.b, p1/m, z0.b, z2.b
    urhadd          z1.b, p1/m, z1.b, z3.b
    st1b            {z0.b}, p1, [x0]
    st1b            {z1.b}, p1, [x0, x11]
    add             x0, x0, #4
    add             x0, x0, x1
.endr
    ret
endfunc

function PFX(pixel_avg_pp_24x32_sve2)
    mov             w12, #4
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_pixel_avg_pp_24x32
    sub             x1, x1, #16
    sub             x3, x3, #16
    sub             x5, x5, #16
.lpavg_24x32_sve2:
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
    cbnz            w12, .lpavg_24x32_sve2
    ret
.vl_gt_16_pixel_avg_pp_24x32:
    mov             x10, #24
    mov             x11, #0
    whilelt         p0.b, x11, x10
.vl_gt_16_loop_pixel_avg_pp_24x32:
    sub             w12, w12, #1
.rept 8
    ld1b            {z0.b}, p0/z, [x2]
    ld1b            {z2.b}, p0/z, [x4]
    add             x2, x2, x3
    add             x4, x4, x5
    urhadd          z0.b, p0/m, z0.b, z2.b
    st1b            {z0.b}, p0, [x0]
    add             x0, x0, x1
.endr
    cbnz            w12, .vl_gt_16_loop_pixel_avg_pp_24x32
    ret
endfunc

.macro pixel_avg_pp_32xN_sve2 h
function PFX(pixel_avg_pp_32x\h\()_sve2)
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_pixel_avg_pp_32_\h
.rept \h
    ld1             {v0.16b-v1.16b}, [x2], x3
    ld1             {v2.16b-v3.16b}, [x4], x5
    urhadd          v0.16b, v0.16b, v2.16b
    urhadd          v1.16b, v1.16b, v3.16b
    st1             {v0.16b-v1.16b}, [x0], x1
.endr
    ret
.vl_gt_16_pixel_avg_pp_32_\h:
    ptrue           p0.b, vl32
.rept \h
    ld1b            {z0.b}, p0/z, [x2]
    ld1b            {z2.b}, p0/z, [x4]
    add             x2, x2, x3
    add             x4, x4, x5
    urhadd          z0.b, p0/m, z0.b, z2.b
    st1b            {z0.b}, p0, [x0]
    add             x0, x0, x1
.endr
    ret
endfunc
.endm

pixel_avg_pp_32xN_sve2 8
pixel_avg_pp_32xN_sve2 16
pixel_avg_pp_32xN_sve2 24

.macro pixel_avg_pp_32xN1_sve2 h
function PFX(pixel_avg_pp_32x\h\()_sve2)
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_pixel_avg_pp_32xN1_\h
    mov             w12, #\h / 8
.lpavg_sve2_32x\h\():
    sub             w12, w12, #1
.rept 8
    ld1             {v0.16b-v1.16b}, [x2], x3
    ld1             {v2.16b-v3.16b}, [x4], x5
    urhadd          v0.16b, v0.16b, v2.16b
    urhadd          v1.16b, v1.16b, v3.16b
    st1             {v0.16b-v1.16b}, [x0], x1
.endr
    cbnz            w12, .lpavg_sve2_32x\h
    ret
.vl_gt_16_pixel_avg_pp_32xN1_\h:
    ptrue           p0.b, vl32
    mov             w12, #\h / 8
.eq_32_loop_pixel_avg_pp_32xN1_\h\():
    sub             w12, w12, #1
.rept 8
    ld1b            {z0.b}, p0/z, [x2]
    ld1b            {z2.b}, p0/z, [x4]
    add             x2, x2, x3
    add             x4, x4, x5
    urhadd          z0.b, p0/m, z0.b, z2.b
    st1b            {z0.b}, p0, [x0]
    add             x0, x0, x1
.endr
    cbnz            w12, .eq_32_loop_pixel_avg_pp_32xN1_\h
    ret
endfunc
.endm

pixel_avg_pp_32xN1_sve2 32
pixel_avg_pp_32xN1_sve2 64

function PFX(pixel_avg_pp_48x64_sve2)
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_pixel_avg_pp_48x64
    mov             w12, #8
.lpavg_48x64_sve2:
    sub             w12, w12, #1
.rept 8
    ld1             {v0.16b-v2.16b}, [x2], x3
    ld1             {v3.16b-v5.16b}, [x4], x5
    urhadd          v0.16b, v0.16b, v3.16b
    urhadd          v1.16b, v1.16b, v4.16b
    urhadd          v2.16b, v2.16b, v5.16b
    st1             {v0.16b-v2.16b}, [x0], x1
.endr
    cbnz            w12, .lpavg_48x64_sve2
    ret
.vl_gt_16_pixel_avg_pp_48x64:
    cmp             x9, #32
    bgt             .vl_gt_32_pixel_avg_pp_48x64
    ptrue           p0.b, vl32
    ptrue           p1.b, vl16
    mov             w12, #8
.vl_eq_32_pixel_avg_pp_48x64:
    sub             w12, w12, #1
.rept 8
    ld1b            {z0.b}, p0/z, [x2]
    ld1b            {z1.b}, p1/z, [x2, #1, mul vl]
    ld1b            {z2.b}, p0/z, [x4]
    ld1b            {z3.b}, p1/z, [x4, #1, mul vl]
    add             x2, x2, x3
    add             x4, x4, x5
    urhadd          z0.b, p0/m, z0.b, z2.b
    urhadd          z1.b, p1/m, z1.b, z3.b
    st1b            {z0.b}, p0, [x0]
    st1b            {z1.b}, p1, [x0, #1, mul vl]
    add             x0, x0, x1
.endr
    cbnz            w12, .vl_eq_32_pixel_avg_pp_48x64
    ret
.vl_gt_32_pixel_avg_pp_48x64:
    mov             x10, #48
    mov             x11, #0
    whilelt         p0.b, x11, x10
    mov             w12, #8
.Loop_gt_32_pixel_avg_pp_48x64:
    sub             w12, w12, #1
.rept 8
    ld1b            {z0.b}, p0/z, [x2]
    ld1b            {z2.b}, p0/z, [x4]
    add             x2, x2, x3
    add             x4, x4, x5
    urhadd          z0.b, p0/m, z0.b, z2.b
    st1b            {z0.b}, p0, [x0]
    add             x0, x0, x1
.endr
    cbnz            w12, .Loop_gt_32_pixel_avg_pp_48x64
    ret
endfunc

.macro pixel_avg_pp_64xN_sve2 h
function PFX(pixel_avg_pp_64x\h\()_sve2)
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_pixel_avg_pp_64x\h
    mov             w12, #\h / 4
.lpavg_sve2_64x\h\():
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
    cbnz            w12, .lpavg_sve2_64x\h
    ret
.vl_gt_16_pixel_avg_pp_64x\h\():
    cmp             x9, #48
    bgt             .vl_gt_48_pixel_avg_pp_64x\h
    ptrue           p0.b, vl32
    mov             w12, #\h / 4
.vl_eq_32_pixel_avg_pp_64x\h\():
    sub             w12, w12, #1
.rept 4
    ld1b            {z0.b}, p0/z, [x2]
    ld1b            {z1.b}, p0/z, [x2, #1, mul vl]
    ld1b            {z2.b}, p0/z, [x4]
    ld1b            {z3.b}, p0/z, [x4, #1, mul vl]
    add             x2, x2, x3
    add             x4, x4, x5
    urhadd          z0.b, p0/m, z0.b, z2.b
    urhadd          z1.b, p0/m, z1.b, z3.b
    st1b            {z0.b}, p0, [x0]
    st1b            {z1.b}, p0, [x0, #1, mul vl]
    add             x0, x0, x1
.endr
    cbnz            w12, .vl_eq_32_pixel_avg_pp_64x\h
    ret
.vl_gt_48_pixel_avg_pp_64x\h\():
    ptrue           p0.b, vl64
    mov             w12, #\h / 4
.vl_eq_64_pixel_avg_pp_64x\h\():
    sub             w12, w12, #1
.rept 4
    ld1b            {z0.b}, p0/z, [x2]
    ld1b            {z2.b}, p0/z, [x4]
    add             x2, x2, x3
    add             x4, x4, x5
    urhadd          z0.b, p0/m, z0.b, z2.b
    st1b            {z0.b}, p0, [x0]
    add             x0, x0, x1
.endr
    cbnz            w12, .vl_eq_64_pixel_avg_pp_64x\h
    ret
endfunc
.endm

pixel_avg_pp_64xN_sve2 16
pixel_avg_pp_64xN_sve2 32
pixel_avg_pp_64xN_sve2 48
pixel_avg_pp_64xN_sve2 64
