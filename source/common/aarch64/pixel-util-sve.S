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
#include "pixel-util-common.S"

.arch armv8-a+sve

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4

.text

function PFX(pixel_sub_ps_8x16_sve)
    lsl             x1, x1, #1
    ptrue           p0.h, vl8
.rept 8
    ld1b            {z0.h}, p0/z, [x2]
    ld1b            {z1.h}, p0/z, [x3]
    add             x2, x2, x4
    add             x3, x3, x5
    ld1b            {z2.h}, p0/z, [x2]
    ld1b            {z3.h}, p0/z, [x3]
    add             x2, x2, x4
    add             x3, x3, x5
    sub             z4.h, z0.h, z1.h
    sub             z5.h, z2.h, z3.h
    st1             {v4.8h}, [x0], x1
    st1             {v5.8h}, [x0], x1
.endr
    ret
endfunc

