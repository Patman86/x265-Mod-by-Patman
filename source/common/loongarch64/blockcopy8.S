/*****************************************************************************
 * Copyright (C) 2024 MulticoreWare, Inc
 *
 * Authors: jinbo <jinbo@loongson.cn>
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

#include "loongson_asm.S"

/*
 *void blockcopy_pp_c(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride)
 */
function x265_blockcopy_pp_2x2_lsx
    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3

    vstelm.h      vr0,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr1,    a0,     0,     0
endfunc

function x265_blockcopy_pp_2x4_lsx
    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.s         f2,     a2,     0
    fldx.s        f3,     a2,     a3

    vstelm.h      vr0,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr1,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr2,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr3,    a0,     0,     0
endfunc

.macro BLOCKCOPY_PP_W2_H4 w, h
function x265_blockcopy_pp_\w\()x\h\()_lsx
.rept \h/4 -1
    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.s         f2,     a2,     0
    fldx.s        f3,     a2,     a3
    alsl.d        a2,     a3,     a2,    1

    vstelm.h      vr0,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr1,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr2,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr3,    a0,     0,     0
    add.d         a0,     a0,     a1
.endr
    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.s         f2,     a2,     0
    fldx.s        f3,     a2,     a3

    vstelm.h      vr0,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr1,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr2,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr3,    a0,     0,     0
endfunc
.endm

BLOCKCOPY_PP_W2_H4 2, 8
BLOCKCOPY_PP_W2_H4 2, 16

function x265_blockcopy_pp_4x2_lsx
    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3

    fst.s         f0,     a0,     0
    fstx.s        f1,     a0,     a1
endfunc

function x265_blockcopy_pp_4x4_lsx
    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.s         f2,     a2,     0
    fldx.s        f3,     a2,     a3

    fst.s         f0,     a0,     0
    fstx.s        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.s         f2,     a0,     0
    fstx.s        f3,     a0,     a1
endfunc

.macro BLOCKCOPY_PP_W4_H4 w, h
function x265_blockcopy_pp_\w\()x\h\()_lsx
.rept \h/4 -1
    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.s         f2,     a2,     0
    fldx.s        f3,     a2,     a3
    alsl.d        a2,     a3,     a2,    1

    fst.s         f0,     a0,     0
    fstx.s        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.s         f2,     a0,     0
    fstx.s        f3,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
.endr
    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.s         f2,     a2,     0
    fldx.s        f3,     a2,     a3

    fst.s         f0,     a0,     0
    fstx.s        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.s         f2,     a0,     0
    fstx.s        f3,     a0,     a1
endfunc
.endm

BLOCKCOPY_PP_W4_H4 4, 8
BLOCKCOPY_PP_W4_H4 4, 16
BLOCKCOPY_PP_W4_H4 4, 32

function x265_blockcopy_pp_6x8_lsx
    fld.d         f0,     a2,     0
    fldx.d        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.d         f2,     a2,     0
    fldx.d        f3,     a2,     a3

    alsl.d        a2,     a3,     a2,    1
    fld.d         f4,     a2,     0
    fldx.d        f5,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.d         f6,     a2,     0
    fldx.d        f7,     a2,     a3

    fst.s         f0,     a0,     0
    vstelm.h      vr0,    a0,     4,     2
    add.d         a0,     a0,     a1
    fst.s         f1,     a0,     0
    vstelm.h      vr1,    a0,     4,     2
    add.d         a0,     a0,     a1
    fst.s         f2,     a0,     0
    vstelm.h      vr2,    a0,     4,     2
    add.d         a0,     a0,     a1
    fst.s         f3,     a0,     0
    vstelm.h      vr3,    a0,     4,     2
    add.d         a0,     a0,     a1

    fst.s         f4,     a0,     0
    vstelm.h      vr4,    a0,     4,     2
    add.d         a0,     a0,     a1
    fst.s         f5,     a0,     0
    vstelm.h      vr5,    a0,     4,     2
    add.d         a0,     a0,     a1
    fst.s         f6,     a0,     0
    vstelm.h      vr6,    a0,     4,     2
    add.d         a0,     a0,     a1
    fst.s         f7,     a0,     0
    vstelm.h      vr7,    a0,     4,     2
endfunc

function x265_blockcopy_pp_6x16_lsx
    li.w          t0,     16/4
1:
    fld.d         f0,     a2,     0
    fldx.d        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.d         f2,     a2,     0
    fldx.d        f3,     a2,     a3
    alsl.d        a2,     a3,     a2,    1

    fst.s         f0,     a0,     0
    vstelm.h      vr0,    a0,     4,     2
    add.d         a0,     a0,     a1
    fst.s         f1,     a0,     0
    vstelm.h      vr1,    a0,     4,     2
    add.d         a0,     a0,     a1
    fst.s         f2,     a0,     0
    vstelm.h      vr2,    a0,     4,     2
    add.d         a0,     a0,     a1
    fst.s         f3,     a0,     0
    vstelm.h      vr3,    a0,     4,     2
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc

function x265_blockcopy_pp_8x2_lsx
    fld.d         f0,     a2,     0
    fldx.d        f1,     a2,     a3

    fst.d         f0,     a0,     0
    fstx.d        f1,     a0,     a1
endfunc

function x265_blockcopy_pp_8x4_lsx
    fld.d         f0,     a2,     0
    fldx.d        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.d         f2,     a2,     0
    fldx.d        f3,     a2,     a3

    fst.d         f0,     a0,     0
    fstx.d        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.d         f2,     a0,     0
    fstx.d        f3,     a0,     a1
endfunc

function x265_blockcopy_pp_8x6_lsx
    fld.d         f0,     a2,     0
    fldx.d        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.d         f2,     a2,     0
    fldx.d        f3,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.d         f4,     a2,     0
    fldx.d        f5,     a2,     a3

    fst.d         f0,     a0,     0
    fstx.d        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.d         f2,     a0,     0
    fstx.d        f3,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.d         f4,     a0,     0
    fstx.d        f5,     a0,     a1
endfunc

.macro BLOCKCOPY_PP_W8_H4 w, h
function x265_blockcopy_pp_\w\()x\h\()_lsx
.rept \h/4 -1
    fld.d         f0,     a2,     0
    fldx.d        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.d         f2,     a2,     0
    fldx.d        f3,     a2,     a3
    alsl.d        a2,     a3,     a2,    1

    fst.d         f0,     a0,     0
    fstx.d        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.d         f2,     a0,     0
    fstx.d        f3,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
.endr

    fld.d         f0,     a2,     0
    fldx.d        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.d         f2,     a2,     0
    fldx.d        f3,     a2,     a3

    fst.d         f0,     a0,     0
    fstx.d        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.d         f2,     a0,     0
    fstx.d        f3,     a0,     a1
endfunc
.endm

BLOCKCOPY_PP_W8_H4 8, 8
BLOCKCOPY_PP_W8_H4 8, 12
BLOCKCOPY_PP_W8_H4 8, 16
BLOCKCOPY_PP_W8_H4 8, 32
BLOCKCOPY_PP_W8_H4 8, 64

.macro BLOCKCOPY_PP_W12_H4 w, h
function x265_blockcopy_pp_\w\()x\h\()_lsx
.rept \h/4 -1
    vld           vr0,    a2,     0
    vldx          vr1,    a2,     a3
    alsl.d        a2,     a3,     a2,    1
    vld           vr2,    a2,     0
    vldx          vr3,    a2,     a3
    alsl.d        a2,     a3,     a2,    1

    fst.d         f0,     a0,     0
    vstelm.w      vr0,    a0,     8,     2
    add.d         a0,     a0,     a1
    fst.d         f1,     a0,     0
    vstelm.w      vr1,    a0,     8,     2
    add.d         a0,     a0,     a1
    fst.d         f2,     a0,     0
    vstelm.w      vr2,    a0,     8,     2
    add.d         a0,     a0,     a1
    fst.d         f3,     a0,     0
    vstelm.w      vr3,    a0,     8,     2
    add.d         a0,     a0,     a1
.endr

    vld           vr0,    a2,     0
    vldx          vr1,    a2,     a3
    alsl.d        a2,     a3,     a2,    1
    vld           vr2,    a2,     0
    vldx          vr3,    a2,     a3

    fst.d         f0,     a0,     0
    vstelm.w      vr0,    a0,     8,     2
    add.d         a0,     a0,     a1
    fst.d         f1,     a0,     0
    vstelm.w      vr1,    a0,     8,     2
    add.d         a0,     a0,     a1
    fst.d         f2,     a0,     0
    vstelm.w      vr2,    a0,     8,     2
    add.d         a0,     a0,     a1
    fst.d         f3,     a0,     0
    vstelm.w      vr3,    a0,     8,     2
endfunc
.endm

BLOCKCOPY_PP_W12_H4 12, 16
BLOCKCOPY_PP_W12_H4 12, 32

.macro BLOCKCOPY_PP_W16_H4 w, h
function x265_blockcopy_pp_\w\()x\h\()_lsx
    li.w          t0,     \h/4
1:
    vld           vr0,    a2,     0
    vldx          vr1,    a2,     a3
    alsl.d        a2,     a3,     a2,    1
    vld           vr2,    a2,     0
    vldx          vr3,    a2,     a3
    alsl.d        a2,     a3,     a2,    1

    vst           vr0,    a0,     0
    vstx          vr1,    a0,     a1
    alsl.d        a0,     a1,     a0,    1
    vst           vr2,    a0,     0
    vstx          vr3,    a0,     a1
    alsl.d        a0,     a1,     a0,    1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

BLOCKCOPY_PP_W16_H4 16, 4
BLOCKCOPY_PP_W16_H4 16, 12

.macro BLOCKCOPY_PP_W16_H8 w, h
function x265_blockcopy_pp_\w\()x\h\()_lsx
    li.w          t0,     \h/8
1:
    vld           vr0,    a2,     0
    vldx          vr1,    a2,     a3
    alsl.d        a2,     a3,     a2,    1
    vld           vr2,    a2,     0
    vldx          vr3,    a2,     a3
    alsl.d        a2,     a3,     a2,    1

    vld           vr4,    a2,     0
    vldx          vr5,    a2,     a3
    alsl.d        a2,     a3,     a2,    1
    vld           vr6,    a2,     0
    vldx          vr7,    a2,     a3
    alsl.d        a2,     a3,     a2,    1

    vst           vr0,    a0,     0
    vstx          vr1,    a0,     a1
    alsl.d        a0,     a1,     a0,    1
    vst           vr2,    a0,     0
    vstx          vr3,    a0,     a1
    alsl.d        a0,     a1,     a0,    1

    vst           vr4,    a0,     0
    vstx          vr5,    a0,     a1
    alsl.d        a0,     a1,     a0,    1
    vst           vr6,    a0,     0
    vstx          vr7,    a0,     a1
    alsl.d        a0,     a1,     a0,    1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

BLOCKCOPY_PP_W16_H8 16, 8
BLOCKCOPY_PP_W16_H8 16, 16
BLOCKCOPY_PP_W16_H8 16, 24
BLOCKCOPY_PP_W16_H8 16, 32
BLOCKCOPY_PP_W16_H8 16, 64

.macro BLOCKCOPY_PP_W24_H4 w, h
function x265_blockcopy_pp_\w\()x\h\()_lsx
    li.w          t0,     \h/4
1:
    vld           vr0,    a2,     0
    vld           vr1,    a2,     16
    add.d         a2,     a2,     a3
    vld           vr2,    a2,     0
    vld           vr3,    a2,     16
    add.d         a2,     a2,     a3
    vld           vr4,    a2,     0
    vld           vr5,    a2,     16
    add.d         a2,     a2,     a3
    vld           vr6,    a2,     0
    vld           vr7,    a2,     16
    add.d         a2,     a2,     a3

    vst           vr0,    a0,     0
    vstelm.d      vr1,    a0,     16,    0
    add.d         a0,     a0,     a1
    vst           vr2,    a0,     0
    vstelm.d      vr3,    a0,     16,    0
    add.d         a0,     a0,     a1
    vst           vr4,    a0,     0
    vstelm.d      vr5,    a0,     16,    0
    add.d         a0,     a0,     a1
    vst           vr6,    a0,     0
    vstelm.d      vr7,    a0,     16,    0
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

BLOCKCOPY_PP_W24_H4 24, 32
BLOCKCOPY_PP_W24_H4 24, 64

.macro BLOCKCOPY_PP_W32_H4 w, h
function x265_blockcopy_pp_\w\()x\h\()_lsx
    li.w          t0,     \h/4
1:
    vld           vr0,    a2,     0
    vld           vr1,    a2,     16
    add.d         a2,     a2,     a3
    vld           vr2,    a2,     0
    vld           vr3,    a2,     16
    add.d         a2,     a2,     a3
    vld           vr4,    a2,     0
    vld           vr5,    a2,     16
    add.d         a2,     a2,     a3
    vld           vr6,    a2,     0
    vld           vr7,    a2,     16
    add.d         a2,     a2,     a3

    vst           vr0,    a0,     0
    vst           vr1,    a0,     16
    add.d         a0,     a0,     a1
    vst           vr2,    a0,     0
    vst           vr3,    a0,     16
    add.d         a0,     a0,     a1
    vst           vr4,    a0,     0
    vst           vr5,    a0,     16
    add.d         a0,     a0,     a1
    vst           vr6,    a0,     0
    vst           vr7,    a0,     16
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

BLOCKCOPY_PP_W32_H4 32, 8
BLOCKCOPY_PP_W32_H4 32, 16
BLOCKCOPY_PP_W32_H4 32, 24
BLOCKCOPY_PP_W32_H4 32, 32
BLOCKCOPY_PP_W32_H4 32, 48
BLOCKCOPY_PP_W32_H4 32, 64

.macro LASX_BLOCKCOPY_PP_W32_H4 w, h
function x265_blockcopy_pp_\w\()x\h\()_lasx
    li.w          t0,     \h/4
1:
    xvld          xr0,    a2,     0
    xvldx         xr1,    a2,     a3
    alsl.d        a2,     a3,     a2,   1
    xvld          xr2,    a2,     0
    xvldx         xr3,    a2,     a3
    alsl.d        a2,     a3,     a2,   1
.if \w == 32
    xvst          xr0,    a0,     0
    xvstx         xr1,    a0,     a1
    alsl.d        a0,     a1,     a0,   1
    xvst          xr2,    a0,     0
    xvstx         xr3,    a0,     a1
    alsl.d        a0,     a1,     a0,   1
.else
    vst           vr0,    a0,     0
    xvpermi.q     xr0,    xr0,    0x01
    fst.d         f0,     a0,     16
    add.d         a0,     a0,     a1
    vst           vr1,    a0,     0
    xvpermi.q     xr1,    xr1,    0x01
    fst.d         f1,     a0,     16
    add.d         a0,     a0,     a1
    vst           vr2,    a0,     0
    xvpermi.q     xr2,    xr2,    0x01
    fst.d         f2,     a0,     16
    add.d         a0,     a0,     a1
    vst           vr3,    a0,     0
    xvpermi.q     xr3,    xr3,    0x01
    fst.d         f3,     a0,     16
    add.d         a0,     a0,     a1
.endif

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LASX_BLOCKCOPY_PP_W32_H4 32, 8
LASX_BLOCKCOPY_PP_W32_H4 32, 16
LASX_BLOCKCOPY_PP_W32_H4 32, 24
LASX_BLOCKCOPY_PP_W32_H4 32, 32
LASX_BLOCKCOPY_PP_W32_H4 32, 48
LASX_BLOCKCOPY_PP_W32_H4 32, 64

LASX_BLOCKCOPY_PP_W32_H4 24, 32
LASX_BLOCKCOPY_PP_W32_H4 24, 64

.macro BLOCKCOPY_PP_W48_H4 w, h
function x265_blockcopy_pp_\w\()x\h\()_lsx
    li.w          t0,     \h/4
1:
    vld           vr0,    a2,     0
    vld           vr1,    a2,     16
    vld           vr2,    a2,     32
    add.d         a2,     a2,     a3
    vld           vr3,    a2,     0
    vld           vr4,    a2,     16
    vld           vr5,    a2,     32
    add.d         a2,     a2,     a3
    vld           vr6,    a2,     0
    vld           vr7,    a2,     16
    vld           vr8,    a2,     32
    add.d         a2,     a2,     a3
    vld           vr9,    a2,     0
    vld           vr10,   a2,     16
    vld           vr11,   a2,     32
    add.d         a2,     a2,     a3

    vst           vr0,    a0,     0
    vst           vr1,    a0,     16
    vst           vr2,    a0,     32
    add.d         a0,     a0,     a1
    vst           vr3,    a0,     0
    vst           vr4,    a0,     16
    vst           vr5,    a0,     32
    add.d         a0,     a0,     a1
    vst           vr6,    a0,     0
    vst           vr7,    a0,     16
    vst           vr8,    a0,     32
    add.d         a0,     a0,     a1
    vst           vr9,    a0,     0
    vst           vr10,   a0,     16
    vst           vr11,   a0,     32
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

BLOCKCOPY_PP_W48_H4 48, 64

.macro BLOCKCOPY_PP_W64_H4 w, h
function x265_blockcopy_pp_\w\()x\h\()_lsx
    li.w          t0,     \h/4
1:
    vld           vr0,    a2,     0
    vld           vr1,    a2,     16
    vld           vr2,    a2,     32
    vld           vr3,    a2,     48
    add.d         a2,     a2,     a3
    vld           vr4,    a2,     0
    vld           vr5,    a2,     16
    vld           vr6,    a2,     32
    vld           vr7,    a2,     48
    add.d         a2,     a2,     a3
    vld           vr8,    a2,     0
    vld           vr9,    a2,     16
    vld           vr10,   a2,     32
    vld           vr11,   a2,     48
    add.d         a2,     a2,     a3
    vld           vr12,   a2,     0
    vld           vr13,   a2,     16
    vld           vr14,   a2,     32
    vld           vr15,   a2,     48
    add.d         a2,     a2,     a3

    vst           vr0,    a0,     0
    vst           vr1,    a0,     16
    vst           vr2,    a0,     32
    vst           vr3,    a0,     48
    add.d         a0,     a0,     a1
    vst           vr4,    a0,     0
    vst           vr5,    a0,     16
    vst           vr6,    a0,     32
    vst           vr7,    a0,     48
    add.d         a0,     a0,     a1
    vst           vr8,    a0,     0
    vst           vr9,    a0,     16
    vst           vr10,   a0,     32
    vst           vr11,   a0,     48
    add.d         a0,     a0,     a1
    vst           vr12,   a0,     0
    vst           vr13,   a0,     16
    vst           vr14,   a0,     32
    vst           vr15,   a0,     48
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

BLOCKCOPY_PP_W64_H4 64, 16
BLOCKCOPY_PP_W64_H4 64, 32
BLOCKCOPY_PP_W64_H4 64, 48
BLOCKCOPY_PP_W64_H4 64, 64

.macro LASX_BLOCKCOPY_PP_W64_H4 w, h
function x265_blockcopy_pp_\w\()x\h\()_lasx
    li.w          t0,     \h/4
1:
    xvld          xr0,    a2,     0
    xvld          xr1,    a2,     32
    add.d         a2,     a2,     a3
    xvld          xr2,    a2,     0
    xvld          xr3,    a2,     32
    add.d         a2,     a2,     a3
    xvld          xr4,    a2,     0
    xvld          xr5,    a2,     32
    add.d         a2,     a2,     a3
    xvld          xr6,    a2,     0
    xvld          xr7,    a2,     32
    add.d         a2,     a2,     a3
.if \w == 64
    xvst          xr0,    a0,     0
    xvst          xr1,    a0,     32
    add.d         a0,     a0,     a1
    xvst          xr2,    a0,     0
    xvst          xr3,    a0,     32
    add.d         a0,     a0,     a1
    xvst          xr4,    a0,     0
    xvst          xr5,    a0,     32
    add.d         a0,     a0,     a1
    xvst          xr6,    a0,     0
    xvst          xr7,    a0,     32
    add.d         a0,     a0,     a1
.else
    xvst          xr0,    a0,     0
    vst           vr1,    a0,     32
    add.d         a0,     a0,     a1
    xvst          xr2,    a0,     0
    vst           vr3,    a0,     32
    add.d         a0,     a0,     a1
    xvst          xr4,    a0,     0
    vst           vr5,    a0,     32
    add.d         a0,     a0,     a1
    xvst          xr6,    a0,     0
    vst           vr7,    a0,     32
    add.d         a0,     a0,     a1
.endif
    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LASX_BLOCKCOPY_PP_W64_H4 64, 16
LASX_BLOCKCOPY_PP_W64_H4 64, 32
LASX_BLOCKCOPY_PP_W64_H4 64, 48
LASX_BLOCKCOPY_PP_W64_H4 64, 64

LASX_BLOCKCOPY_PP_W64_H4 48, 64

function x265_blockfill_s_4x4_lsx
    slli.d        a1,      a1,    1
    vreplgr2vr.h  vr10,    a2
    vstelm.d      vr10,    a0,    0,    0
.rept 3
    add.d         a0,      a0,    a1
    vstelm.d      vr10,    a0,    0,    0
.endr
endfunc

function x265_blockfill_s_8x8_lsx
    vreplgr2vr.h  vr10,    a2
    slli.d        a1,      a1,    1
    vst           vr10,    a0,    0
    vstx          vr10,    a0,    a1
.rept 3
    alsl.d        a0,      a1,    a0,   1
    vst           vr10,    a0,    0
    vstx          vr10,    a0,    a1
.endr
endfunc

function x265_blockfill_s_16x16_lsx
    vreplgr2vr.h  vr10,    a2
    addi.d        t4,      a0,    16
    slli.d        a1,      a1,    1
    vst           vr10,    a0,    0
    vst           vr10,    t4,    0
    vstx          vr10,    a0,    a1
    vstx          vr10,    t4,    a1
.rept 7
    alsl.d        a0,      a1,    a0,   1
    addi.d        t4,      a0,    16
    vst           vr10,    a0,    0
    vst           vr10,    t4,    0
    vstx          vr10,    a0,    a1
    vstx          vr10,    t4,    a1
.endr
endfunc

function x265_blockfill_s_32x32_lsx
    vreplgr2vr.h  vr10,    a2
    slli.d        a1,      a1,    1
.rept 32
    vst           vr10,    a0,    0
    vst           vr10,    a0,    16
    vst           vr10,    a0,    32
    vst           vr10,    a0,    48
    add.d         a0,      a0,    a1
.endr
endfunc

function x265_blockfill_s_64x64_lsx
    vreplgr2vr.h  vr10,    a2
    slli.d        a1,      a1,    1
.rept 64
.irp i, 0, 16, 32, 48, 64, 80, 96, 112
    vst           vr10,    a0,    \i
.endr
    add.d         a0,      a0,    a1
.endr
endfunc

function x265_blockfill_s_16x16_lasx
    xvreplgr2vr.h  xr10,    a2
    slli.d         a1,      a1,    1
    xvst           xr10,    a0,    0
    xvstx          xr10,    a0,    a1
.rept 7
    alsl.d         a0,      a1,    a0,   1
    xvst           xr10,    a0,    0
    xvstx          xr10,    a0,    a1
.endr
endfunc

function x265_blockfill_s_32x32_lasx
    xvreplgr2vr.h  xr10,    a2
    slli.d         a1,      a1,    1
.rept 32
    xvst           xr10,    a0,    0
    xvst           xr10,    a0,    32
    add.d          a0,      a0,    a1
.endr
endfunc

function x265_blockfill_s_64x64_lasx
    xvreplgr2vr.h  xr10,    a2
    slli.d         a1,      a1,    1
.rept 64
.irp i, 0, 32, 64, 96
    xvst           xr10,    a0,    \i
.endr
    add.d          a0,      a0,    a1
.endr
endfunc

function x265_cpy2Dto1D_shl_4x4_lsx
    slli.d         a2,      a2,    1
    fld.d          f0,      a1,    0
    fldx.d         f1,      a1,    a2
    vreplgr2vr.h   vr3,     a3
    vilvl.d        vr2,     vr1,   vr0
    vsll.h         vr4,     vr2,   vr3
    vst            vr4,     a0,    0

    alsl.d         a1,      a2,    a1,   1

    fld.d          f0,      a1,    0
    fldx.d         f1,      a1,    a2
    vreplgr2vr.h   vr3,     a3
    vilvl.d        vr2,     vr1,   vr0
    vsll.h         vr4,     vr2,   vr3
    vst            vr4,     a0,    16
endfunc

function x265_cpy2Dto1D_shl_8x8_lsx
    slli.d         a2,      a2,    1
.rept 4
    vld            vr0,     a1,    0
    vldx           vr1,     a1,    a2
    vreplgr2vr.h   vr3,     a3
    vsll.h         vr4,     vr0,   vr3
    vsll.h         vr5,     vr1,   vr3
    vst            vr4,     a0,    0
    vst            vr5,     a0,    16
    alsl.d         a1,      a2,    a1,   1
    addi.d         a0,      a0,    32
.endr
endfunc

function x265_cpy2Dto1D_shl_16x16_lsx
    slli.d         a2,      a2,    1
.rept 8
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    add.d          t1,      a1,    a2
    vld            vr2,     t1,    0
    vld            vr3,     t1,    16
    vreplgr2vr.h   vr4,     a3
    vsll.h         vr5,     vr0,   vr4
    vsll.h         vr6,     vr1,   vr4
    vsll.h         vr7,     vr2,   vr4
    vsll.h         vr8,     vr3,   vr4
    vst            vr5,     a0,    0
    vst            vr6,     a0,    16
    vst            vr7,     a0,    32
    vst            vr8,     a0,    48
    alsl.d         a1,      a2,    a1,   1
    addi.d         a0,      a0,    64
.endr
endfunc

function x265_cpy2Dto1D_shl_32x32_lsx
    slli.d         a2,      a2,    1
.rept 16
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vld            vr10,    a1,    32
    vld            vr11,    a1,    48
    add.d          t1,      a1,    a2
    vld            vr2,     t1,    0
    vld            vr3,     t1,    16
    vld            vr12,    t1,    32
    vld            vr13,    t1,    48
    vreplgr2vr.h   vr4,     a3
.irp i, vr0, vr1, vr10, vr11, vr2, vr3, vr12, vr13
    vsll.h         \i,      \i,   vr4
.endr
    vst            vr0,     a0,    0
    vst            vr1,     a0,    16
    vst            vr10,    a0,    32
    vst            vr11,    a0,    48
    vst            vr2,     a0,    64
    vst            vr3,     a0,    80
    vst            vr12,    a0,    96
    vst            vr13,    a0,    112
    alsl.d         a1,      a2,    a1,   1
    addi.d         a0,      a0,    128
.endr
endfunc

function x265_cpy2Dto1D_shl_64x64_lsx
    slli.d         a2,      a2,    1
.rept 64
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vld            vr2,     a1,    32
    vld            vr3,     a1,    48
    vld            vr4,     a1,    64
    vld            vr5,     a1,    80
    vld            vr6,     a1,    96
    vld            vr7,     a1,    112
    vreplgr2vr.h   vr8,     a3
.irp i, vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7
    vsll.h         \i,      \i,    vr8
.endr
    vst            vr0,     a0,    0
    vst            vr1,     a0,    16
    vst            vr2,     a0,    32
    vst            vr3,     a0,    48
    vst            vr4,     a0,    64
    vst            vr5,     a0,    80
    vst            vr6,     a0,    96
    vst            vr7,     a0,    112
    add.d          a1,      a2,    a1
    addi.d         a0,      a0,    128
.endr
endfunc

function x265_cpy2Dto1D_shl_8x8_lasx
    slli.d         a2,      a2,    1
.rept 4
    vld            vr0,     a1,    0
    vldx           vr1,     a1,    a2
    xvpermi.q      xr1,     xr0,   0x20
    xvreplgr2vr.h  xr3,     a3
    xvsll.h        xr4,     xr1,   xr3
    xvst           xr4,     a0,    0
    alsl.d         a1,      a2,    a1,   1
    addi.d         a0,      a0,    32
.endr
endfunc

function x265_cpy2Dto1D_shl_16x16_lasx
    slli.d         a2,      a2,    1
.rept 8
    xvld           xr0,     a1,    0
    xvldx          xr1,     a1,    a2
    xvreplgr2vr.h  xr4,     a3
    xvsll.h        xr6,     xr0,   xr4
    xvsll.h        xr8,     xr1,   xr4
    xvst           xr6,     a0,    0
    xvst           xr8,     a0,    32
    alsl.d         a1,      a2,    a1,   1
    addi.d         a0,      a0,    64
.endr
endfunc

function x265_cpy2Dto1D_shl_32x32_lasx
    slli.d         a2,      a2,    1
.rept 16
    xvld           xr0,     a1,    0
    xvldx          xr1,     a1,    a2
    addi.d         t1,      a1,    32
    xvld           xr2,     t1,    0
    xvldx          xr3,     t1,    a2
    xvreplgr2vr.h  xr4,     a3
    xvsll.h        xr6,     xr0,   xr4
    xvsll.h        xr7,     xr2,   xr4
    xvsll.h        xr8,     xr1,   xr4
    xvsll.h        xr9,     xr3,   xr4
    xvst           xr6,     a0,    0
    xvst           xr7,     a0,    32
    xvst           xr8,     a0,    64
    xvst           xr9,     a0,    96
    alsl.d         a1,      a2,    a1,   1
    addi.d         a0,      a0,    128
.endr
endfunc

function x265_cpy2Dto1D_shl_64x64_lasx
    slli.d         a2,      a2,    1
.rept 64
    xvld           xr0,     a1,    0
    xvld           xr1,     a1,    32
    xvld           xr2,     a1,    64
    xvld           xr3,     a1,    96
    xvreplgr2vr.h  xr4,     a3
    xvsll.h        xr6,     xr0,   xr4
    xvsll.h        xr7,     xr1,   xr4
    xvsll.h        xr8,     xr2,   xr4
    xvsll.h        xr9,     xr3,   xr4
    xvst           xr6,     a0,    0
    xvst           xr7,     a0,    32
    xvst           xr8,     a0,    64
    xvst           xr9,     a0,    96
    add.d          a1,      a2,    a1
    addi.d         a0,      a0,    128
.endr
endfunc

/*
 * void x265_blockcopy_ss_2x2_lsx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb)
 */
function x265_blockcopy_ss_2x2_lsx
    add.w         a3,     a3,     a3
    add.w         a1,     a1,     a1

    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3

    fst.s         f0,     a0,     0
    fstx.s        f1,     a0,     a1
endfunc

function x265_blockcopy_ss_2x4_lsx
    add.w         a3,     a3,     a3
    add.w         a1,     a1,     a1

    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.s         f2,     a2,     0
    fldx.s        f3,     a2,     a3

    fst.s         f0,     a0,     0
    fstx.s        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.s         f2,     a0,     0
    fstx.s        f3,     a0,     a1
endfunc

function x265_blockcopy_ss_4x4_lsx
    add.w         a3,     a3,     a3
    add.w         a1,     a1,     a1

    fld.d         f0,     a2,     0
    fldx.d        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.d         f2,     a2,     0
    fldx.d        f3,     a2,     a3

    fst.d         f0,     a0,     0
    fstx.d        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.d         f2,     a0,     0
    fstx.d        f3,     a0,     a1
endfunc

function x265_blockcopy_ss_4x8_lsx
    add.w         a3,     a3,     a3
    add.w         a1,     a1,     a1

    fld.d         f0,     a2,     0
    fldx.d        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.d         f2,     a2,     0
    fldx.d        f3,     a2,     a3
    alsl.d        a2,     a3,     a2,    1

    fst.d         f0,     a0,     0
    fstx.d        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.d         f2,     a0,     0
    fstx.d        f3,     a0,     a1
    alsl.d        a0,     a1,     a0,    1

    fld.d         f0,     a2,     0
    fldx.d        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.d         f2,     a2,     0
    fldx.d        f3,     a2,     a3

    fst.d         f0,     a0,     0
    fstx.d        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.d         f2,     a0,     0
    fstx.d        f3,     a0,     a1
endfunc

.macro LSX_BLOCKCOPY_SS_8xN h
function x265_blockcopy_ss_8x\h\()_lsx
    add.w         a3,     a3,     a3
    add.w         a1,     a1,     a1
    li.w          t0,     \h/2
1:
    vld           vr0,    a2,     0
    vldx          vr1,    a2,     a3
    alsl.d        a2,     a3,     a2,    1

    vst           vr0,    a0,     0
    vstx          vr1,    a0,     a1
    alsl.d        a0,     a1,     a0,    1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LSX_BLOCKCOPY_SS_8xN 8
LSX_BLOCKCOPY_SS_8xN 16

.macro LSX_BLOCKCOPY_SS_16xN h
function x265_blockcopy_ss_16x\h\()_lsx
    add.w         a3,     a3,     a3
    add.w         a1,     a1,     a1
    li.w          t0,     \h/2
1:
    vld           vr0,    a2,     0
    vld           vr1,    a2,     16
    add.d         a2,     a2,     a3
    vld           vr2,    a2,     0
    vld           vr3,    a2,     16
    add.d         a2,     a2,     a3

    vst           vr0,    a0,     0
    vst           vr1,    a0,     16
    add.d         a0,     a0,     a1
    vst           vr2,    a0,     0
    vst           vr3,    a0,     16
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LSX_BLOCKCOPY_SS_16xN 16
LSX_BLOCKCOPY_SS_16xN 32

.macro LASX_BLOCKCOPY_SS_16xN h
function x265_blockcopy_ss_16x\h\()_lasx
    add.w         a3,     a3,     a3
    add.w         a1,     a1,     a1
    li.w          t0,     \h/2
1:
    xvld          xr0,    a2,     0
    xvldx         xr1,    a2,     a3
    alsl.d        a2,     a3,     a2,    1

    xvst          xr0,    a0,     0
    xvstx         xr1,    a0,     a1
    alsl.d        a0,     a1,     a0,    1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LASX_BLOCKCOPY_SS_16xN 16
LASX_BLOCKCOPY_SS_16xN 32

.macro LSX_BLOCKCOPY_SS_32xN h
function x265_blockcopy_ss_32x\h\()_lsx
    add.w         a3,     a3,     a3
    add.w         a1,     a1,     a1
    li.w          t0,     \h/2
1:
    vld           vr0,    a2,     0
    vld           vr1,    a2,     16
    vld           vr2,    a2,     32
    vld           vr3,    a2,     48
    add.d         a2,     a2,     a3
    vld           vr4,    a2,     0
    vld           vr5,    a2,     16
    vld           vr6,    a2,     32
    vld           vr7,    a2,     48
    add.d         a2,     a2,     a3

    vst           vr0,    a0,     0
    vst           vr1,    a0,     16
    vst           vr2,    a0,     32
    vst           vr3,    a0,     48
    add.d         a0,     a0,     a1
    vst           vr4,    a0,     0
    vst           vr5,    a0,     16
    vst           vr6,    a0,     32
    vst           vr7,    a0,     48
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LSX_BLOCKCOPY_SS_32xN 32
LSX_BLOCKCOPY_SS_32xN 64

.macro LASX_BLOCKCOPY_SS_32xN h
function x265_blockcopy_ss_32x\h\()_lasx
    add.w         a3,     a3,     a3
    add.w         a1,     a1,     a1
    li.w          t0,     \h/2
1:
    xvld          xr0,    a2,     0
    xvld          xr1,    a2,     32
    add.d         a2,     a2,     a3
    xvld          xr2,    a2,     0
    xvld          xr3,    a2,     32
    add.d         a2,     a2,     a3

    xvst          xr0,    a0,     0
    xvst          xr1,    a0,     32
    add.d         a0,     a0,     a1
    xvst          xr2,    a0,     0
    xvst          xr3,    a0,     32
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LASX_BLOCKCOPY_SS_32xN 32
LASX_BLOCKCOPY_SS_32xN 64

function x265_blockcopy_ss_64x64_lsx
    add.w         a3,     a3,     a3
    add.w         a1,     a1,     a1
    li.w          t0,     64/2
1:
    vld           vr0,    a2,     0
    vld           vr1,    a2,     16
    vld           vr2,    a2,     32
    vld           vr3,    a2,     48
    vld           vr4,    a2,     64
    vld           vr5,    a2,     80
    vld           vr6,    a2,     96
    vld           vr7,    a2,     112
    add.d         a2,     a2,     a3
    vld           vr8,    a2,     0
    vld           vr9,    a2,     16
    vld           vr10,   a2,     32
    vld           vr11,   a2,     48
    vld           vr12,   a2,     64
    vld           vr13,   a2,     80
    vld           vr14,   a2,     96
    vld           vr15,   a2,     112
    add.d         a2,     a2,     a3

    vst           vr0,    a0,     0
    vst           vr1,    a0,     16
    vst           vr2,    a0,     32
    vst           vr3,    a0,     48
    vst           vr4,    a0,     64
    vst           vr5,    a0,     80
    vst           vr6,    a0,     96
    vst           vr7,    a0,     112
    add.d         a0,     a0,     a1
    vst           vr8,    a0,     0
    vst           vr9,    a0,     16
    vst           vr10,   a0,     32
    vst           vr11,   a0,     48
    vst           vr12,   a0,     64
    vst           vr13,   a0,     80
    vst           vr14,   a0,     96
    vst           vr15,   a0,     112
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc

function x265_blockcopy_ss_64x64_lasx
    add.w         a3,     a3,     a3
    add.w         a1,     a1,     a1
    li.w          t0,     64/2
1:
    xvld          xr0,    a2,     0
    xvld          xr1,    a2,     32
    xvld          xr2,    a2,     64
    xvld          xr3,    a2,     96
    add.d         a2,     a2,     a3
    xvld          xr4,    a2,     0
    xvld          xr5,    a2,     32
    xvld          xr6,    a2,     64
    xvld          xr7,    a2,     96
    add.d         a2,     a2,     a3

    xvst          xr0,    a0,     0
    xvst          xr1,    a0,     32
    xvst          xr2,    a0,     64
    xvst          xr3,    a0,     96
    add.d         a0,     a0,     a1
    xvst          xr4,    a0,     0
    xvst          xr5,    a0,     32
    xvst          xr6,    a0,     64
    xvst          xr7,    a0,     96
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc

/*
 * void x265_blockcopy_sp_2x2_lsx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb)
 */
function x265_blockcopy_sp_2x2_lsx
    add.w         a3,     a3,     a3

    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3
    vssrani.bu.h  vr1,    vr0,    0

    vstelm.h      vr1,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr1,    a0,     0,     4
endfunc

function x265_blockcopy_sp_2x4_lsx
    add.w         a3,     a3,     a3

    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.s         f2,     a2,     0
    fldx.s        f3,     a2,     a3
    vssrani.bu.h  vr1,    vr0,    0
    vssrani.bu.h  vr3,    vr2,    0

    vstelm.h      vr1,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr1,    a0,     0,     4
    add.d         a0,     a0,     a1
    vstelm.h      vr3,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.h      vr3,    a0,     0,     4
endfunc

.macro LSX_BLOCKCOPY_SP_4xN h
function x265_blockcopy_sp_4x\h\()_lsx
    add.w         a3,     a3,     a3
    li.w          t0,     \h/2
1:
    fld.d         f0,     a2,     0
    fldx.d        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    vssrani.bu.h  vr1,    vr0,    0

    vstelm.w      vr1,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.w      vr1,    a0,     0,     2
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LSX_BLOCKCOPY_SP_4xN 4
LSX_BLOCKCOPY_SP_4xN 8

.macro LSX_BLOCKCOPY_SP_8xN h
function x265_blockcopy_sp_8x\h\()_lsx
    add.w         a3,     a3,     a3
    li.w          t0,     \h/2
1:
    vld           vr0,    a2,     0
    vldx          vr1,    a2,     a3
    alsl.d        a2,     a3,     a2,    1
    vssrani.bu.h  vr1,    vr0,    0

    vstelm.d      vr1,    a0,     0,     0
    add.d         a0,     a0,     a1
    vstelm.d      vr1,    a0,     0,     1
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LSX_BLOCKCOPY_SP_8xN 8
LSX_BLOCKCOPY_SP_8xN 16

.macro LSX_BLOCKCOPY_SP_16xN h
function x265_blockcopy_sp_16x\h\()_lsx
    add.w         a3,     a3,     a3
    li.w          t0,     \h/2
1:
    vld           vr0,    a2,     0
    vld           vr1,    a2,     16
    add.d         a2,     a2,     a3
    vld           vr2,    a2,     0
    vld           vr3,    a2,     16
    add.d         a2,     a2,     a3
    vssrani.bu.h  vr1,    vr0,    0
    vssrani.bu.h  vr3,    vr2,    0

    vst           vr1,    a0,     0
    vstx          vr3,    a0,     a1
    alsl.d        a0,     a1,     a0,    1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LSX_BLOCKCOPY_SP_16xN 16
LSX_BLOCKCOPY_SP_16xN 32

.macro LASX_BLOCKCOPY_SP_16xN h
function x265_blockcopy_sp_16x\h\()_lasx
    add.w         a3,     a3,     a3
    li.w          t0,     \h/2
1:
    xvld          xr0,    a2,     0
    xvldx         xr1,    a2,     a3
    alsl.d        a2,     a3,     a2,    1
    xvssrani.bu.h xr1,    xr0,    0
    xvpermi.d     xr0,    xr1,    0xd8
    xvpermi.q     xr1,    xr0,    0x01

    vst           vr0,    a0,     0
    vstx          vr1,    a0,     a1
    alsl.d        a0,     a1,     a0,    1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LASX_BLOCKCOPY_SP_16xN 16
LASX_BLOCKCOPY_SP_16xN 32

.macro LSX_BLOCKCOPY_SP_32xN h
function x265_blockcopy_sp_32x\h\()_lsx
    add.w         a3,     a3,     a3
    li.w          t0,     \h/2
1:
    vld           vr0,    a2,     0
    vld           vr1,    a2,     16
    vld           vr2,    a2,     32
    vld           vr3,    a2,     48
    add.d         a2,     a2,     a3
    vld           vr4,    a2,     0
    vld           vr5,    a2,     16
    vld           vr6,    a2,     32
    vld           vr7,    a2,     48
    add.d         a2,     a2,     a3
    vssrani.bu.h  vr1,    vr0,    0
    vssrani.bu.h  vr3,    vr2,    0
    vssrani.bu.h  vr5,    vr4,    0
    vssrani.bu.h  vr7,    vr6,    0

    vst           vr1,    a0,     0
    vst           vr3,    a0,     16
    add.d         a0,     a0,     a1
    vst           vr5,    a0,     0
    vst           vr7,    a0,     16
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LSX_BLOCKCOPY_SP_32xN 32
LSX_BLOCKCOPY_SP_32xN 64

.macro LASX_BLOCKCOPY_SP_32xN h
function x265_blockcopy_sp_32x\h\()_lasx
    add.w         a3,     a3,     a3
    li.w          t0,     \h/2
1:
    xvld          xr0,    a2,     0
    xvld          xr1,    a2,     32
    add.d         a2,     a2,     a3
    xvld          xr2,    a2,     0
    xvld          xr3,    a2,     32
    add.d         a2,     a2,     a3
    xvssrani.bu.h xr1,    xr0,    0
    xvssrani.bu.h xr3,    xr2,    0
    xvpermi.d     xr0,    xr1,    0xd8
    xvpermi.d     xr1,    xr3,    0xd8

    xvst          xr0,    a0,     0
    xvstx         xr1,    a0,     a1
    alsl.d        a0,     a1,     a0,    1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LASX_BLOCKCOPY_SP_32xN 32
LASX_BLOCKCOPY_SP_32xN 64

function x265_blockcopy_sp_64x64_lsx
    add.w         a3,     a3,     a3
    li.w          t0,     64/2
1:
    vld           vr0,    a2,     0
    vld           vr1,    a2,     16
    vld           vr2,    a2,     32
    vld           vr3,    a2,     48
    vld           vr4,    a2,     64
    vld           vr5,    a2,     80
    vld           vr6,    a2,     96
    vld           vr7,    a2,     112
    add.d         a2,     a2,     a3
    vld           vr8,    a2,     0
    vld           vr9,    a2,     16
    vld           vr10,   a2,     32
    vld           vr11,   a2,     48
    vld           vr12,   a2,     64
    vld           vr13,   a2,     80
    vld           vr14,   a2,     96
    vld           vr15,   a2,     112
    add.d         a2,     a2,     a3
    vssrani.bu.h  vr1,    vr0,    0
    vssrani.bu.h  vr3,    vr2,    0
    vssrani.bu.h  vr5,    vr4,    0
    vssrani.bu.h  vr7,    vr6,    0
    vssrani.bu.h  vr9,    vr8,    0
    vssrani.bu.h  vr11,   vr10,   0
    vssrani.bu.h  vr13,   vr12,   0
    vssrani.bu.h  vr15,   vr14,   0

    vst           vr1,    a0,     0
    vst           vr3,    a0,     16
    vst           vr5,    a0,     32
    vst           vr7,    a0,     48
    add.d         a0,     a0,     a1
    vst           vr9,    a0,     0
    vst           vr11,   a0,     16
    vst           vr13,   a0,     32
    vst           vr15,   a0,     48
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc

function x265_blockcopy_sp_64x64_lasx
    add.w         a3,     a3,     a3
    li.w          t0,     64/2
1:
    xvld          xr0,    a2,     0
    xvld          xr1,    a2,     32
    xvld          xr2,    a2,     64
    xvld          xr3,    a2,     96
    add.d         a2,     a2,     a3
    xvld          xr4,    a2,     0
    xvld          xr5,    a2,     32
    xvld          xr6,    a2,     64
    xvld          xr7,    a2,     96
    add.d         a2,     a2,     a3
    xvssrani.bu.h xr1,    xr0,    0
    xvssrani.bu.h xr3,    xr2,    0
    xvssrani.bu.h xr5,    xr4,    0
    xvssrani.bu.h xr7,    xr6,    0
    xvpermi.d     xr0,    xr1,    0xd8
    xvpermi.d     xr2,    xr3,    0xd8
    xvpermi.d     xr4,    xr5,    0xd8
    xvpermi.d     xr6,    xr7,    0xd8

    xvst          xr0,    a0,     0
    xvst          xr2,    a0,     32
    add.d         a0,     a0,     a1
    xvst          xr4,    a0,     0
    xvst          xr6,    a0,     32
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc

/*
 * void x265_blockcopy_ps_2x4_lsx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb)
 */
function x265_blockcopy_ps_2x4_lsx
    add.w         a1,     a1,     a1

    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    fld.s         f2,     a2,     0
    fldx.s        f3,     a2,     a3
    vsllwil.hu.bu vr0,    vr0,    0
    vsllwil.hu.bu vr1,    vr1,    0
    vsllwil.hu.bu vr2,    vr2,    0
    vsllwil.hu.bu vr3,    vr3,    0

    fst.s         f0,     a0,     0
    fstx.s        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1
    fst.s         f2,     a0,     0
    fstx.s        f3,     a0,     a1
endfunc

.macro LSX_BLOCKCOPY_PS_4xN h
function x265_blockcopy_ps_4x\h\()_lsx
    add.w         a1,     a1,     a1
    li.w          t0,     \h/2
1:
    fld.s         f0,     a2,     0
    fldx.s        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    vsllwil.hu.bu vr0,    vr0,    0
    vsllwil.hu.bu vr1,    vr1,    0

    fst.d         f0,     a0,     0
    fstx.d        f1,     a0,     a1
    alsl.d        a0,     a1,     a0,    1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LSX_BLOCKCOPY_PS_4xN 4
LSX_BLOCKCOPY_PS_4xN 8

.macro LSX_BLOCKCOPY_PS_8xN h
function x265_blockcopy_ps_8x\h\()_lsx
    add.w         a1,     a1,     a1
    li.w          t0,     \h/2
1:
    fld.d         f0,     a2,     0
    fldx.d        f1,     a2,     a3
    alsl.d        a2,     a3,     a2,    1
    vsllwil.hu.bu vr0,    vr0,    0
    vsllwil.hu.bu vr1,    vr1,    0

    vst           vr0,    a0,     0
    vstx          vr1,    a0,     a1
    alsl.d        a0,     a1,     a0,    1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LSX_BLOCKCOPY_PS_8xN 8
LSX_BLOCKCOPY_PS_8xN 16

.macro LSX_BLOCKCOPY_PS_16xN h
function x265_blockcopy_ps_16x\h\()_lsx
    add.w         a1,     a1,     a1
    li.w          t0,     \h/2
1:
    vld           vr0,    a2,     0
    vldx          vr1,    a2,     a3
    alsl.d        a2,     a3,     a2,    1
    vexth.hu.bu   vr2,    vr0
    vexth.hu.bu   vr3,    vr1
    vsllwil.hu.bu vr0,    vr0,    0
    vsllwil.hu.bu vr1,    vr1,    0

    vst           vr0,    a0,     0
    vst           vr2,    a0,     16
    add.d         a0,     a0,     a1
    vst           vr1,    a0,     0
    vst           vr3,    a0,     16
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LSX_BLOCKCOPY_PS_16xN 16
LSX_BLOCKCOPY_PS_16xN 32

.macro LASX_BLOCKCOPY_PS_16xN h
function x265_blockcopy_ps_16x\h\()_lasx
    add.w         a1,     a1,     a1
    li.w          t0,     \h/2
1:
    vld           vr0,    a2,     0
    vldx          vr1,    a2,     a3
    alsl.d        a2,     a3,     a2,    1
    xvpermi.q     xr1,    xr0,    0x20
    xvpermi.d     xr1,    xr1,    0xd8
    xvexth.hu.bu  xr2,    xr1
    xvsllwil.hu.bu xr1,   xr1,    0

    xvst          xr1,    a0,     0
    xvstx         xr2,    a0,     a1
    alsl.d        a0,     a1,     a0,    1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LASX_BLOCKCOPY_PS_16xN 16
LASX_BLOCKCOPY_PS_16xN 32

.macro LSX_BLOCKCOPY_PS_32xN h
function x265_blockcopy_ps_32x\h\()_lsx
    add.w         a1,     a1,     a1
    li.w          t0,     \h/2
1:
    vld           vr0,    a2,     0
    vld           vr1,    a2,     16
    add.d         a2,     a2,     a3
    vld           vr4,    a2,     0
    vld           vr5,    a2,     16
    add.d         a2,     a2,     a3
    vexth.hu.bu   vr2,    vr0
    vexth.hu.bu   vr3,    vr1
    vsllwil.hu.bu vr0,    vr0,    0
    vsllwil.hu.bu vr1,    vr1,    0
    vexth.hu.bu   vr6,    vr4
    vexth.hu.bu   vr7,    vr5
    vsllwil.hu.bu vr4,    vr4,    0
    vsllwil.hu.bu vr5,    vr5,    0

    vst           vr0,    a0,     0
    vst           vr2,    a0,     16
    vst           vr1,    a0,     32
    vst           vr3,    a0,     48
    add.d         a0,     a0,     a1
    vst           vr4,    a0,     0
    vst           vr6,    a0,     16
    vst           vr5,    a0,     32
    vst           vr7,    a0,     48
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LSX_BLOCKCOPY_PS_32xN 32
LSX_BLOCKCOPY_PS_32xN 64

.macro LASX_BLOCKCOPY_PS_32xN h
function x265_blockcopy_ps_32x\h\()_lasx
    add.w         a1,     a1,     a1
    li.w          t0,     \h/2
1:
    xvld          xr0,    a2,     0
    xvldx         xr1,    a2,     a3
    alsl.d        a2,     a3,     a2,    1
    xvpermi.d     xr0,    xr0,    0xd8
    xvpermi.d     xr1,    xr1,    0xd8
    xvexth.hu.bu  xr2,    xr0
    xvexth.hu.bu  xr3,    xr1
    xvsllwil.hu.bu xr0,   xr0,    0
    xvsllwil.hu.bu xr1,   xr1,    0

    xvst          xr0,    a0,     0
    xvst          xr2,    a0,     32
    add.d         a0,     a0,     a1
    xvst          xr1,    a0,     0
    xvst          xr3,    a0,     32
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc
.endm

LASX_BLOCKCOPY_PS_32xN 32
LASX_BLOCKCOPY_PS_32xN 64

function x265_blockcopy_ps_64x64_lsx
    add.w         a1,     a1,     a1
    li.w          t0,     64/2
1:
    vld           vr0,    a2,     0
    vld           vr1,    a2,     16
    vld           vr2,    a2,     32
    vld           vr3,    a2,     48
    add.d         a2,     a2,     a3
    vld           vr8,    a2,     0
    vld           vr9,    a2,     16
    vld           vr10,   a2,     32
    vld           vr11,   a2,     48
    add.d         a2,     a2,     a3
    vexth.hu.bu   vr4,    vr0
    vexth.hu.bu   vr5,    vr1
    vexth.hu.bu   vr6,    vr2
    vexth.hu.bu   vr7,    vr3
    vsllwil.hu.bu vr0,    vr0,    0
    vsllwil.hu.bu vr1,    vr1,    0
    vsllwil.hu.bu vr2,    vr2,    0
    vsllwil.hu.bu vr3,    vr3,    0
    vexth.hu.bu   vr12,   vr8
    vexth.hu.bu   vr13,   vr9
    vexth.hu.bu   vr14,   vr10
    vexth.hu.bu   vr15,   vr11
    vsllwil.hu.bu vr8,    vr8,    0
    vsllwil.hu.bu vr9,    vr9,    0
    vsllwil.hu.bu vr10,   vr10,   0
    vsllwil.hu.bu vr11,   vr11,   0

    vst           vr0,    a0,     0
    vst           vr4,    a0,     16
    vst           vr1,    a0,     32
    vst           vr5,    a0,     48
    vst           vr2,    a0,     64
    vst           vr6,    a0,     80
    vst           vr3,    a0,     96
    vst           vr7,    a0,     112
    add.d         a0,     a0,     a1
    vst           vr8,    a0,     0
    vst           vr12,   a0,     16
    vst           vr9,    a0,     32
    vst           vr13,   a0,     48
    vst           vr10,   a0,     64
    vst           vr14,   a0,     80
    vst           vr11,   a0,     96
    vst           vr15,   a0,     112
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc

function x265_blockcopy_ps_64x64_lasx
    add.w         a1,     a1,     a1
    li.w          t0,     64/2
1:
    xvld          xr0,    a2,     0
    xvld          xr1,    a2,     32
    add.d         a2,     a2,     a3
    xvld          xr4,    a2,     0
    xvld          xr5,    a2,     32
    add.d         a2,     a2,     a3
    xvpermi.d     xr0,    xr0,    0xd8
    xvpermi.d     xr1,    xr1,    0xd8
    xvpermi.d     xr2,    xr4,    0xd8
    xvpermi.d     xr3,    xr5,    0xd8
    xvexth.hu.bu  xr4,    xr0
    xvexth.hu.bu  xr5,    xr1
    xvexth.hu.bu  xr6,    xr2
    xvexth.hu.bu  xr7,    xr3
    xvsllwil.hu.bu xr0,   xr0,    0
    xvsllwil.hu.bu xr1,   xr1,    0
    xvsllwil.hu.bu xr2,   xr2,    0
    xvsllwil.hu.bu xr3,   xr3,    0

    xvst          xr0,    a0,     0
    xvst          xr4,    a0,     32
    xvst          xr1,    a0,     64
    xvst          xr5,    a0,     96
    add.d         a0,     a0,     a1
    xvst          xr2,    a0,     0
    xvst          xr6,    a0,     32
    xvst          xr3,    a0,     64
    xvst          xr7,    a0,     96
    add.d         a0,     a0,     a1

    addi.w        t0,     t0,     -1
    bnez          t0,     1b
endfunc

/*
 *uint32_t x265_copy_count_4_lsx(int16_t* coeff, const int16_t* residual, intptr_t resiStride)
 */
function x265_copy_count_4_lsx
    add.w         a2,     a2,     a2

    fld.d         f0,     a1,     0
    fldx.d        f1,     a1,     a2
    alsl.d        a1,     a2,     a1,    1
    fld.d         f2,     a1,     0
    fldx.d        f3,     a1,     a2
    //store coeff
    vpackev.d     vr0,    vr1,    vr0
    vpackev.d     vr1,    vr3,    vr2
    vst           vr0,    a0,     0
    vst           vr1,    a0,     16
    //numSig
    vssrani.b.h   vr1,    vr0,    0
    vseqi.b       vr1,    vr1,    0
    vaddi.bu      vr1,    vr1,    1
    vhaddw.hu.bu  vr1,    vr1,    vr1
    vhaddw.wu.hu  vr1,    vr1,    vr1
    vhaddw.du.wu  vr1,    vr1,    vr1
    vhaddw.qu.du  vr1,    vr1,    vr1
    vpickve2gr.hu t0,     vr1,    0
    or            a0,     zero,   t0
endfunc

function x265_copy_count_8_lsx
    add.w         a2,     a2,     a2
    xor           t2,     t2,     t2
    li.w          t3,     8/4
1:
    vld           vr0,    a1,     0
    vldx          vr1,    a1,     a2
    alsl.d        a1,     a2,     a1,    1
    vld           vr2,    a1,     0
    vldx          vr3,    a1,     a2
    alsl.d        a1,     a2,     a1,    1
    //store coeff
    vst           vr0,    a0,     0
    vst           vr1,    a0,     16
    vst           vr2,    a0,     32
    vst           vr3,    a0,     48
    addi.d        a0,     a0,     64
    //numSig
    vssrani.b.h   vr1,    vr0,    0
    vssrani.b.h   vr3,    vr2,    0
    vseqi.b       vr1,    vr1,    0
    vseqi.b       vr3,    vr3,    0
    vadd.b        vr1,    vr1,    vr3
    vaddi.bu      vr1,    vr1,    2
    vhaddw.hu.bu  vr1,    vr1,    vr1
    vhaddw.wu.hu  vr1,    vr1,    vr1
    vhaddw.du.wu  vr1,    vr1,    vr1
    vhaddw.qu.du  vr1,    vr1,    vr1
    vpickve2gr.hu t0,     vr1,    0
    add.w         t2,     t2,     t0
    addi.w        t3,     t3,     -1
    bnez          t3,     1b
    or            a0,     zero,   t2
endfunc

function x265_copy_count_16_lsx
    add.w         a2,     a2,     a2
    xor           t2,     t2,     t2
    li.w          t3,     16/2
1:
    vld           vr0,    a1,     0
    vld           vr1,    a1,     16
    add.d         a1,     a1,     a2
    vld           vr2,    a1,     0
    vld           vr3,    a1,     16
    add.d         a1,     a1,     a2
    //store coeff
    vst           vr0,    a0,     0
    vst           vr1,    a0,     16
    vst           vr2,    a0,     32
    vst           vr3,    a0,     48
    addi.d        a0,     a0,     64
    //numSig
    vssrani.b.h   vr1,    vr0,    0
    vssrani.b.h   vr3,    vr2,    0
    vseqi.b       vr1,    vr1,    0
    vseqi.b       vr3,    vr3,    0
    vadd.b        vr1,    vr1,    vr3
    vaddi.bu      vr1,    vr1,    2
    vhaddw.hu.bu  vr1,    vr1,    vr1
    vhaddw.wu.hu  vr1,    vr1,    vr1
    vhaddw.du.wu  vr1,    vr1,    vr1
    vhaddw.qu.du  vr1,    vr1,    vr1
    vpickve2gr.hu t0,     vr1,    0
    add.w         t2,     t2,     t0
    addi.w        t3,     t3,     -1
    bnez          t3,     1b
    or            a0,     zero,   t2
endfunc

function x265_copy_count_16_lasx
    add.w         a2,     a2,     a2
    xor           t2,     t2,     t2
    li.w          t3,     16/4
1:
    xvld          xr0,    a1,     0
    xvldx         xr1,    a1,     a2
    alsl.d        a1,     a2,     a1,    1
    xvld          xr2,    a1,     0
    xvldx         xr3,    a1,     a2
    alsl.d        a1,     a2,     a1,    1
    //store coeff
    xvst          xr0,    a0,     0
    xvst          xr1,    a0,     32
    xvst          xr2,    a0,     64
    xvst          xr3,    a0,     96
    addi.d        a0,     a0,     128
    //numSig
    xvssrani.b.h  xr1,    xr0,    0
    xvssrani.b.h  xr3,    xr2,    0
    xvseqi.b      xr1,    xr1,    0
    xvseqi.b      xr3,    xr3,    0
    xvadd.b       xr1,    xr1,    xr3
    xvaddi.bu     xr1,    xr1,    2
    xvpermi.q     xr0,    xr1,    0x01
    vadd.b        vr1,    vr1,    vr0
    vhaddw.hu.bu  vr1,    vr1,    vr1
    vhaddw.wu.hu  vr1,    vr1,    vr1
    vhaddw.du.wu  vr1,    vr1,    vr1
    vhaddw.qu.du  vr1,    vr1,    vr1
    vpickve2gr.hu t0,     vr1,    0
    add.w         t2,     t2,     t0
    addi.w        t3,     t3,     -1
    bnez          t3,     1b
    or            a0,     zero,   t2
endfunc

function x265_copy_count_32_lsx
    add.w         a2,     a2,     a2
    xor           t2,     t2,     t2
    li.w          t3,     32/2
1:
    vld           vr0,    a1,     0
    vld           vr1,    a1,     16
    vld           vr2,    a1,     32
    vld           vr3,    a1,     48
    add.d         a1,     a1,     a2
    vld           vr4,    a1,     0
    vld           vr5,    a1,     16
    vld           vr6,    a1,     32
    vld           vr7,    a1,     48
    add.d         a1,     a1,     a2
    //store coeff
    vst           vr0,    a0,     0
    vst           vr1,    a0,     16
    vst           vr2,    a0,     32
    vst           vr3,    a0,     48
    vst           vr4,    a0,     64
    vst           vr5,    a0,     80
    vst           vr6,    a0,     96
    vst           vr7,    a0,     112
    addi.d        a0,     a0,     128
    //numSig
    vssrani.b.h   vr1,    vr0,    0
    vssrani.b.h   vr3,    vr2,    0
    vssrani.b.h   vr5,    vr4,    0
    vssrani.b.h   vr7,    vr6,    0
    vseqi.b       vr1,    vr1,    0
    vseqi.b       vr3,    vr3,    0
    vseqi.b       vr5,    vr5,    0
    vseqi.b       vr7,    vr7,    0
    vadd.b        vr1,    vr1,    vr3
    vadd.b        vr1,    vr1,    vr5
    vadd.b        vr1,    vr1,    vr7
    vaddi.bu      vr1,    vr1,    4
    vhaddw.hu.bu  vr1,    vr1,    vr1
    vhaddw.wu.hu  vr1,    vr1,    vr1
    vhaddw.du.wu  vr1,    vr1,    vr1
    vhaddw.qu.du  vr1,    vr1,    vr1
    vpickve2gr.hu t0,     vr1,    0
    add.w         t2,     t2,     t0
    addi.w        t3,     t3,     -1
    bnez          t3,     1b
    or            a0,     zero,   t2
endfunc

function x265_copy_count_32_lasx
    add.w         a2,     a2,     a2
    xor           t2,     t2,     t2
    li.w          t3,     32/2
1:
    xvld          xr0,    a1,     0
    xvld          xr1,    a1,     32
    add.d         a1,     a1,     a2
    xvld          xr2,    a1,     0
    xvld          xr3,    a1,     32
    add.d         a1,     a1,     a2
    //store coeff
    xvst          xr0,    a0,     0
    xvst          xr1,    a0,     32
    xvst          xr2,    a0,     64
    xvst          xr3,    a0,     96
    addi.d        a0,     a0,     128
    //numSig
    xvssrani.b.h  xr1,    xr0,    0
    xvssrani.b.h  xr3,    xr2,    0
    xvseqi.b      xr1,    xr1,    0
    xvseqi.b      xr3,    xr3,    0
    xvadd.b       xr1,    xr1,    xr3
    xvaddi.bu     xr1,    xr1,    2
    xvpermi.q     xr0,    xr1,    0x01
    vadd.b        vr1,    vr1,    vr0
    vhaddw.hu.bu  vr1,    vr1,    vr1
    vhaddw.wu.hu  vr1,    vr1,    vr1
    vhaddw.du.wu  vr1,    vr1,    vr1
    vhaddw.qu.du  vr1,    vr1,    vr1
    vpickve2gr.hu t0,     vr1,    0
    add.w         t2,     t2,     t0
    addi.w        t3,     t3,     -1
    bnez          t3,     1b
    or            a0,     zero,   t2
endfunc

function x265_cpy2Dto1D_shr_4x4_lsx
    slli.d         a2,      a2,    1
    fld.d          f0,      a1,    0
    fldx.d         f1,      a1,    a2
    vreplgr2vr.h   vr3,     a3
    vilvl.d        vr2,     vr1,   vr0
    vsrar.h        vr4,     vr2,   vr3
    vst            vr4,     a0,    0

    alsl.d         a1,      a2,    a1,   1

    fld.d          f0,      a1,    0
    fldx.d         f1,      a1,    a2
    vreplgr2vr.h   vr3,     a3
    vilvl.d        vr2,     vr1,   vr0
    vsrar.h        vr4,     vr2,   vr3
    vst            vr4,     a0,    16
endfunc

function x265_cpy2Dto1D_shr_8x8_lsx
    slli.d         a2,      a2,    1
.rept 4
    vld            vr0,     a1,    0
    vldx           vr1,     a1,    a2
    vreplgr2vr.h   vr3,     a3
    vsrar.h        vr4,     vr0,   vr3
    vsrar.h        vr5,     vr1,   vr3
    vst            vr4,     a0,    0
    vst            vr5,     a0,    16
    alsl.d         a1,      a2,    a1,   1
    addi.d         a0,      a0,    32
.endr
endfunc

function x265_cpy2Dto1D_shr_16x16_lsx
    slli.d         a2,      a2,    1
.rept 8
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    add.d          t1,      a1,    a2
    vld            vr2,     t1,    0
    vld            vr3,     t1,    16
    vreplgr2vr.h   vr4,     a3
    vsrar.h        vr5,     vr0,   vr4
    vsrar.h        vr6,     vr1,   vr4
    vsrar.h        vr7,     vr2,   vr4
    vsrar.h        vr8,     vr3,   vr4
    vst            vr5,     a0,    0
    vst            vr6,     a0,    16
    vst            vr7,     a0,    32
    vst            vr8,     a0,    48
    alsl.d         a1,      a2,    a1,   1
    addi.d         a0,      a0,    64
.endr
endfunc

function x265_cpy2Dto1D_shr_32x32_lsx
    slli.d         a2,      a2,    1
.rept 16
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vld            vr10,    a1,    32
    vld            vr11,    a1,    48
    add.d          t1,      a1,    a2
    vld            vr2,     t1,    0
    vld            vr3,     t1,    16
    vld            vr12,    t1,    32
    vld            vr13,    t1,    48
    vreplgr2vr.h   vr4,     a3

.irp i, vr0, vr1, vr10, vr11, vr2, vr3, vr12, vr13
    vsrar.h        \i,      \i,   vr4
.endr

    vst            vr0,     a0,    0
    vst            vr1,     a0,    16
    vst            vr10,    a0,    32
    vst            vr11,    a0,    48
    vst            vr2,     a0,    64
    vst            vr3,     a0,    80
    vst            vr12,    a0,    96
    vst            vr13,    a0,    112
    alsl.d         a1,      a2,    a1,   1
    addi.d         a0,      a0,    128
.endr
endfunc

function x265_cpy2Dto1D_shr_64x64_lsx
    slli.d         a2,      a2,    1
.rept 64
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vld            vr2,     a1,    32
    vld            vr3,     a1,    48
    vld            vr4,     a1,    64
    vld            vr5,     a1,    80
    vld            vr6,     a1,    96
    vld            vr7,     a1,    112
    vreplgr2vr.h   vr8,     a3

.irp i, vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7
    vsrar.h        \i,      \i,    vr8
.endr

    vst            vr0,     a0,    0
    vst            vr1,     a0,    16
    vst            vr2,     a0,    32
    vst            vr3,     a0,    48
    vst            vr4,     a0,    64
    vst            vr5,     a0,    80
    vst            vr6,     a0,    96
    vst            vr7,     a0,    112
    add.d          a1,      a2,    a1
    addi.d         a0,      a0,    128
.endr
endfunc

function x265_cpy2Dto1D_shr_8x8_lasx
    slli.d         a2,      a2,    1
.rept 4
    vld            vr0,     a1,    0
    vldx           vr1,     a1,    a2
    xvpermi.q      xr1,     xr0,   0x20
    xvreplgr2vr.h  xr3,     a3
    xvsrar.h       xr4,     xr1,   xr3
    xvst           xr4,     a0,    0
    alsl.d         a1,      a2,    a1,   1
    addi.d         a0,      a0,    32
.endr
endfunc

function x265_cpy2Dto1D_shr_16x16_lasx
    slli.d         a2,      a2,    1
.rept 8
    xvld           xr0,     a1,    0
    xvldx          xr1,     a1,    a2
    xvreplgr2vr.h  xr4,     a3
    xvsrar.h       xr6,     xr0,   xr4
    xvsrar.h       xr8,     xr1,   xr4
    xvst           xr6,     a0,    0
    xvst           xr8,     a0,    32
    alsl.d         a1,      a2,    a1,   1
    addi.d         a0,      a0,    64
.endr
endfunc

function x265_cpy2Dto1D_shr_32x32_lasx
    slli.d         a2,      a2,    1
.rept 16
    xvld           xr0,     a1,    0
    xvldx          xr1,     a1,    a2
    addi.d         t1,      a1,    32
    xvld           xr2,     t1,    0
    xvldx          xr3,     t1,    a2
    xvreplgr2vr.h  xr4,     a3
    xvsrar.h       xr6,     xr0,   xr4
    xvsrar.h       xr7,     xr2,   xr4
    xvsrar.h       xr8,     xr1,   xr4
    xvsrar.h       xr9,     xr3,   xr4
    xvst           xr6,     a0,    0
    xvst           xr7,     a0,    32
    xvst           xr8,     a0,    64
    xvst           xr9,     a0,    96
    alsl.d         a1,      a2,    a1,   1
    addi.d         a0,      a0,    128
.endr
endfunc

function x265_cpy2Dto1D_shr_64x64_lasx
    slli.d         a2,      a2,    1
.rept 64
    xvld           xr0,     a1,    0
    xvld           xr1,     a1,    32
    xvld           xr2,     a1,    64
    xvld           xr3,     a1,    96
    xvreplgr2vr.h  xr4,     a3
    xvsrar.h       xr6,     xr0,   xr4
    xvsrar.h       xr7,     xr1,   xr4
    xvsrar.h       xr8,     xr2,   xr4
    xvsrar.h       xr9,     xr3,   xr4
    xvst           xr6,     a0,    0
    xvst           xr7,     a0,    32
    xvst           xr8,     a0,    64
    xvst           xr9,     a0,    96
    add.d          a1,      a2,    a1
    addi.d         a0,      a0,    128
.endr
endfunc

function x265_cpy1Dto2D_shl_4x4_lsx
    slli.d         a2,      a2,    1
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vreplgr2vr.h   vr3,     a3
    vsll.h         vr4,     vr0,   vr3
    vsll.h         vr5,     vr1,   vr3
    vstelm.d       vr4,     a0,    0,    0
    add.d          a0,      a0,    a2
    vstelm.d       vr4,     a0,    0,    1
    add.d          a0,      a0,    a2
    vstelm.d       vr5,     a0,    0,    0
    add.d          a0,      a0,    a2
    vstelm.d       vr5,     a0,    0,    1
endfunc

function x265_cpy1Dto2D_shl_8x8_lsx
    slli.d         a2,      a2,    1
.rept 4
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vreplgr2vr.h   vr3,     a3
    vsll.h         vr4,     vr0,   vr3
    vsll.h         vr5,     vr1,   vr3
    vst            vr4,     a0,    0
    vstx           vr5,     a0,    a2
    alsl.d         a0,      a2,    a0,   1
    addi.d         a1,      a1,    32
.endr
endfunc

function x265_cpy1Dto2D_shl_16x16_lsx
    slli.d         a2,      a2,    1
.rept 8
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vld            vr2,     a1,    32
    vld            vr3,     a1,    48
    vreplgr2vr.h   vr4,     a3
    vsll.h         vr5,     vr0,   vr4
    vsll.h         vr6,     vr1,   vr4
    vsll.h         vr7,     vr2,   vr4
    vsll.h         vr8,     vr3,   vr4
    vst            vr5,     a0,    0
    vst            vr6,     a0,    16
    add.d          a0,      a2,    a0
    vst            vr7,     a0,    0
    vst            vr8,     a0,    16
    add.d          a0,      a2,    a0
    addi.d         a1,      a1,    64
.endr
endfunc

function x265_cpy1Dto2D_shl_32x32_lsx
    slli.d         a2,      a2,    1
.rept 16
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vld            vr10,    a1,    32
    vld            vr11,    a1,    48
    vld            vr2,     a1,    64
    vld            vr3,     a1,    80
    vld            vr12,    a1,    96
    vld            vr13,    a1,    112
    vreplgr2vr.h   vr4,     a3
.irp i, vr0, vr1, vr10, vr11, vr2, vr3, vr12, vr13
    vsll.h         \i,      \i,   vr4
.endr
    vst            vr0,     a0,    0
    vst            vr1,     a0,    16
    vst            vr10,    a0,    32
    vst            vr11,    a0,    48
    add.d          a0,      a0,    a2
    vst            vr2,     a0,    0
    vst            vr3,     a0,    16
    vst            vr12,    a0,    32
    vst            vr13,    a0,    48
    add.d          a0,      a2,    a0
    addi.d         a1,      a1,    128
.endr
endfunc

function x265_cpy1Dto2D_shl_64x64_lsx
    slli.d         a2,      a2,    1
.rept 64
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vld            vr2,     a1,    32
    vld            vr3,     a1,    48
    vld            vr4,     a1,    64
    vld            vr5,     a1,    80
    vld            vr6,     a1,    96
    vld            vr7,     a1,    112
    vreplgr2vr.h   vr8,     a3
.irp i, vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7
    vsll.h         \i,      \i,    vr8
.endr
    vst            vr0,     a0,    0
    vst            vr1,     a0,    16
    vst            vr2,     a0,    32
    vst            vr3,     a0,    48
    vst            vr4,     a0,    64
    vst            vr5,     a0,    80
    vst            vr6,     a0,    96
    vst            vr7,     a0,    112
    add.d          a0,      a2,    a0
    addi.d         a1,      a1,    128
.endr
endfunc

function x265_cpy1Dto2D_shl_8x8_lasx
    slli.d         a2,      a2,    1

.rept 2
    xvld           xr0,     a1,    0
    xvld           xr1,     a1,    32
    xvreplgr2vr.h  xr3,     a3
    xvsll.h        xr4,     xr0,   xr3
    xvsll.h        xr5,     xr1,   xr3
    vst            vr4,     a0,    0
    xvpermi.q      xr6,     xr4,   0x01
    vstx           vr6,     a0,    a2
    alsl.d         a0,      a2,    a0,   1
    vst            vr5,     a0,    0
    xvpermi.q      xr6,     xr5,   0x01
    vstx           vr6,     a0,    a2
    alsl.d         a0,      a2,    a0,   1
    addi.d         a1,      a1,    64
.endr
endfunc

function x265_cpy1Dto2D_shl_16x16_lasx
    slli.d         a2,      a2,    1
.rept 4
    xvld           xr0,     a1,    0
    xvld           xr1,     a1,    32
    xvld           xr2,     a1,    64
    xvld           xr3,     a1,    96
    xvreplgr2vr.h  xr4,     a3
    xvsll.h        xr5,     xr0,   xr4
    xvsll.h        xr6,     xr1,   xr4
    xvsll.h        xr7,     xr2,   xr4
    xvsll.h        xr8,     xr3,   xr4
    xvst           xr5,     a0,    0
    xvstx          xr6,     a0,    a2
    alsl.d         a0,      a2,    a0,   1
    xvst           xr7,     a0,    0
    xvstx          xr8,     a0,    a2
    alsl.d         a0,      a2,    a0,   1
    addi.d         a1,      a1,    128
.endr
endfunc

function x265_cpy1Dto2D_shl_32x32_lasx
    slli.d         a2,      a2,    1
.rept 16
    xvld           xr0,     a1,    0
    xvld           xr1,     a1,    32
    xvld           xr2,     a1,    64
    xvld           xr3,     a1,    96
    xvreplgr2vr.h  xr4,     a3
    xvsll.h        xr5,     xr0,   xr4
    xvsll.h        xr6,     xr1,   xr4
    xvsll.h        xr7,     xr2,   xr4
    xvsll.h        xr8,     xr3,   xr4
    xvst           xr5,     a0,    0
    xvst           xr6,     a0,    32
    add.d          a0,      a2,    a0
    xvst           xr7,     a0,    0
    xvst           xr8,     a0,    32
    add.d          a0,      a2,    a0
    addi.d         a1,      a1,    128
.endr
endfunc

function x265_cpy1Dto2D_shl_64x64_lasx
    slli.d         a2,      a2,    1
.rept 64
    xvld           xr0,     a1,    0
    xvld           xr1,     a1,    32
    xvld           xr2,     a1,    64
    xvld           xr3,     a1,    96
    xvreplgr2vr.h  xr4,     a3
    xvsll.h        xr5,     xr0,   xr4
    xvsll.h        xr6,     xr1,   xr4
    xvsll.h        xr7,     xr2,   xr4
    xvsll.h        xr8,     xr3,   xr4
    xvst           xr5,     a0,    0
    xvst           xr6,     a0,    32
    xvst           xr7,     a0,    64
    xvst           xr8,     a0,    96
    add.d          a0,      a2,    a0
    addi.d         a1,      a1,    128
.endr
endfunc

function x265_cpy1Dto2D_shr_4x4_lsx
    slli.d         a2,      a2,    1
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vreplgr2vr.h   vr3,     a3
    vsrar.h        vr4,     vr0,   vr3
    vsrar.h        vr5,     vr1,   vr3
    vstelm.d       vr4,     a0,    0,    0
    add.d          a0,      a0,    a2
    vstelm.d       vr4,     a0,    0,    1
    add.d          a0,      a0,    a2
    vstelm.d       vr5,     a0,    0,    0
    add.d          a0,      a0,    a2
    vstelm.d       vr5,     a0,    0,    1
endfunc

function x265_cpy1Dto2D_shr_8x8_lsx
    slli.d         a2,      a2,    1
.rept 4
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vreplgr2vr.h   vr3,     a3
    vsrar.h        vr4,     vr0,   vr3
    vsrar.h        vr5,     vr1,   vr3
    vst            vr4,     a0,    0
    vstx           vr5,     a0,    a2
    alsl.d         a0,      a2,    a0,   1
    addi.d         a1,      a1,    32
.endr
endfunc

function x265_cpy1Dto2D_shr_16x16_lsx
    slli.d         a2,      a2,    1
.rept 8
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vld            vr2,     a1,    32
    vld            vr3,     a1,    48
    vreplgr2vr.h   vr4,     a3
    vsrar.h        vr5,     vr0,   vr4
    vsrar.h        vr6,     vr1,   vr4
    vsrar.h        vr7,     vr2,   vr4
    vsrar.h        vr8,     vr3,   vr4
    vst            vr5,     a0,    0
    vst            vr6,     a0,    16
    add.d          a0,      a2,    a0
    vst            vr7,     a0,    0
    vst            vr8,     a0,    16
    add.d          a0,      a2,    a0
    addi.d         a1,      a1,    64
.endr
endfunc

function x265_cpy1Dto2D_shr_32x32_lsx
    slli.d         a2,      a2,    1
.rept 16
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vld            vr10,    a1,    32
    vld            vr11,    a1,    48
    vld            vr2,     a1,    64
    vld            vr3,     a1,    80
    vld            vr12,    a1,    96
    vld            vr13,    a1,    112
    vreplgr2vr.h   vr4,     a3
.irp i, vr0, vr1, vr10, vr11, vr2, vr3, vr12, vr13
    vsrar.h        \i,      \i,   vr4
.endr
    vst            vr0,     a0,    0
    vst            vr1,     a0,    16
    vst            vr10,    a0,    32
    vst            vr11,    a0,    48
    add.d          a0,      a0,    a2
    vst            vr2,     a0,    0
    vst            vr3,     a0,    16
    vst            vr12,    a0,    32
    vst            vr13,    a0,    48
    add.d          a0,      a2,    a0
    addi.d         a1,      a1,    128
.endr
endfunc

function x265_cpy1Dto2D_shr_64x64_lsx
    slli.d         a2,      a2,    1
.rept 64
    vld            vr0,     a1,    0
    vld            vr1,     a1,    16
    vld            vr2,     a1,    32
    vld            vr3,     a1,    48
    vld            vr4,     a1,    64
    vld            vr5,     a1,    80
    vld            vr6,     a1,    96
    vld            vr7,     a1,    112
    vreplgr2vr.h   vr8,     a3
.irp i, vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7
    vsrar.h        \i,      \i,    vr8
.endr
    vst            vr0,     a0,    0
    vst            vr1,     a0,    16
    vst            vr2,     a0,    32
    vst            vr3,     a0,    48
    vst            vr4,     a0,    64
    vst            vr5,     a0,    80
    vst            vr6,     a0,    96
    vst            vr7,     a0,    112
    add.d          a0,      a2,    a0
    addi.d         a1,      a1,    128
.endr
endfunc

function x265_cpy1Dto2D_shr_8x8_lasx
    slli.d         a2,      a2,    1
.rept 2
    xvld           xr0,     a1,    0
    xvld           xr1,     a1,    32
    xvreplgr2vr.h  xr3,     a3
    xvsrar.h       xr4,     xr0,   xr3
    xvsrar.h       xr5,     xr1,   xr3
    xvpermi.q      xr6,     xr4,   0x01
    xvpermi.q      xr7,     xr5,   0x01
    vst            vr4,     a0,    0
    vstx           vr6,     a0,    a2
    alsl.d         a0,      a2,    a0,   1
    vst            vr5,     a0,    0
    vstx           vr7,     a0,    a2
    addi.d         a1,      a1,    64
    alsl.d         a0,      a2,    a0,   1
.endr
endfunc

function x265_cpy1Dto2D_shr_16x16_lasx
    slli.d         a2,      a2,    1
.rept 4
    xvld           xr0,     a1,    0
    xvld           xr1,     a1,    32
    xvld           xr2,     a1,    64
    xvld           xr3,     a1,    96
    xvreplgr2vr.h  xr4,     a3
    xvsrar.h       xr5,     xr0,   xr4
    xvsrar.h       xr6,     xr1,   xr4
    xvsrar.h       xr7,     xr2,   xr4
    xvsrar.h       xr8,     xr3,   xr4
    xvst           xr5,     a0,    0
    xvstx          xr6,     a0,    a2
    alsl.d         a0,      a2,    a0,   1
    xvst           xr7,     a0,    0
    xvstx          xr8,     a0,    a2
    alsl.d         a0,      a2,    a0,   1
    addi.d         a1,      a1,    128
.endr
endfunc

function x265_cpy1Dto2D_shr_32x32_lasx
    slli.d         a2,      a2,    1
.rept 16
    xvld           xr0,     a1,    0
    xvld           xr1,     a1,    32
    xvld           xr2,     a1,    64
    xvld           xr3,     a1,    96
    xvreplgr2vr.h  xr4,     a3
    xvsrar.h       xr5,     xr0,   xr4
    xvsrar.h       xr6,     xr1,   xr4
    xvsrar.h       xr7,     xr2,   xr4
    xvsrar.h       xr8,     xr3,   xr4
    xvst           xr5,     a0,    0
    xvst           xr6,     a0,    32
    add.d          a0,      a2,    a0
    xvst           xr7,     a0,    0
    xvst           xr8,     a0,    32
    add.d          a0,      a2,    a0
    addi.d         a1,      a1,    128
.endr
endfunc

function x265_cpy1Dto2D_shr_64x64_lasx
    slli.d         a2,      a2,    1
.rept 64
    xvld           xr0,     a1,    0
    xvld           xr1,     a1,    32
    xvld           xr2,     a1,    64
    xvld           xr3,     a1,    96
    xvreplgr2vr.h  xr4,     a3
    xvsrar.h       xr5,     xr0,   xr4
    xvsrar.h       xr6,     xr1,   xr4
    xvsrar.h       xr7,     xr2,   xr4
    xvsrar.h       xr8,     xr3,   xr4
    xvst           xr5,     a0,    0
    xvst           xr6,     a0,    32
    xvst           xr7,     a0,    64
    xvst           xr8,     a0,    96
    add.d          a0,      a2,    a0
    addi.d         a1,      a1,    128
.endr
endfunc
