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
#include "ssd-a-common.S"

.arch armv8-a+sve2

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4

.text

function PFX(pixel_ssd_s_4x4_sve2)
    ptrue           p0.b, vl8
    ld1b            {z16.b}, p0/z, [x0]
    add             x0, x0, x1, lsl #1
    smullb          z0.s, z16.h, z16.h
    smlalt          z0.s, z16.h, z16.h
.rept 3
    ld1b            {z16.b}, p0/z, [x0]
    add             x0, x0, x1, lsl #1
    smlalb          z0.s, z16.h, z16.h
    smlalt          z0.s, z16.h, z16.h
.endr
    uaddv           d3, p0, z0.s
    fmov            w0, s3
    ret
endfunc

function PFX(pixel_ssd_s_8x8_sve2)
    ptrue           p0.b, vl16
    ld1b            {z16.b}, p0/z, [x0]
    add             x0, x0, x1, lsl #1
    smullb          z0.s, z16.h, z16.h
    smlalt          z0.s, z16.h, z16.h
.rept 7
    ld1b            {z16.b}, p0/z, [x0]
    add             x0, x0, x1, lsl #1
    smlalb          z0.s, z16.h, z16.h
    smlalt          z0.s, z16.h, z16.h
.endr
    uaddv           d3, p0, z0.s
    fmov            w0, s3
    ret
endfunc

function PFX(pixel_ssd_s_16x16_sve2)
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_pixel_ssd_s_16x16
    add             x1, x1, x1
    mov             w12, #4
    movi            v0.16b, #0
    movi            v1.16b, #0
.Loop_ssd_s_16_sve2:
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
    cbnz            w12, .Loop_ssd_s_16_sve2
    add             v0.4s, v0.4s, v1.4s
    ret_v0_w0
.vl_gt_16_pixel_ssd_s_16x16:
    ptrue           p0.b, vl32
    ld1b            {z16.b}, p0/z, [x0]
    add             x0, x0, x1, lsl #1
    smullb          z0.s, z16.h, z16.h
    smlalt          z0.s, z16.h, z16.h
.rept 15
    ld1b            {z16.b}, p0/z, [x0]
    add             x0, x0, x1, lsl #1
    smlalb          z0.s, z16.h, z16.h
    smlalt          z0.s, z16.h, z16.h
.endr
    uaddv           d3, p0, z0.s
    fmov            w0, s3
    ret
endfunc

function PFX(pixel_ssd_s_32x32_sve2)
    rdvl            x9, #1
    cmp             x9, #16
    bgt             .vl_gt_16_pixel_ssd_s_32x32
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
.vl_gt_16_pixel_ssd_s_32x32:
    cmp             x9, #48
    bgt             .vl_gt_48_pixel_ssd_s_32x32
    ptrue           p0.b, vl32
    ld1b            {z16.b}, p0/z, [x0]
    ld1b            {z17.b}, p0/z, [x0, #1, mul vl]
    add             x0, x0, x1, lsl #1
    smullb          z0.s, z16.h, z16.h
    smlalt          z0.s, z16.h, z16.h
    smlalb          z0.s, z17.h, z17.h
    smlalt          z0.s, z17.h, z17.h
.rept 31
    ld1b            {z16.b}, p0/z, [x0]
    ld1b            {z17.b}, p0/z, [x0, #1, mul vl]
    add             x0, x0, x1, lsl #1
    smlalb          z0.s, z16.h, z16.h
    smlalt          z0.s, z16.h, z16.h
    smlalb          z0.s, z17.h, z17.h
    smlalt          z0.s, z17.h, z17.h
.endr
    uaddv           d3, p0, z0.s
    fmov            w0, s3
    ret
.vl_gt_48_pixel_ssd_s_32x32:
    ptrue           p0.b, vl64
    ld1b            {z16.b}, p0/z, [x0]
    add             x0, x0, x1, lsl #1
    smullb          z0.s, z16.h, z16.h
    smlalt          z0.s, z16.h, z16.h
.rept 31
    ld1b            {z16.b}, p0/z, [x0]
    add             x0, x0, x1, lsl #1
    smlalb          z0.s, z16.h, z16.h
    smlalt          z0.s, z16.h, z16.h
.endr
    uaddv           d3, p0, z0.s
    fmov            w0, s3
    ret
endfunc
