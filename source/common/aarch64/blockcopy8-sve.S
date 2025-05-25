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
