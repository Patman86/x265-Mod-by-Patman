/*****************************************************************************
 * mc-a.S: loongarch motion compensation
 *****************************************************************************
 * Copyright (C) 2023-2024 MulticoreWare, Inc
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

/*----------------------------------------------------------------------------
void pixelavg_pp(pixel dst, intptr_t dstride, const pixel src0,
                 intptr_t sstride0, const pixel* src1, intptr_t sstride1, int)
------------------------------------------------------------------------------*/

#include "loongson_asm.S"

.macro AVG_W4 w, h
function x265_pixel_avg_\w\()x\h\()_lsx
.rept \h/2
    fld.s       f0,     a2,     0
    fld.s       f1,     a4,     0
    fldx.s      f2,     a2,     a3
    fldx.s      f3,     a4,     a5
    vavgr.bu    vr0,    vr1,    vr0
    vavgr.bu    vr1,    vr3,    vr2
    fst.s       f0,     a0,     0
    fstx.s      f1,     a0,     a1
    alsl.d      a2,     a3,     a2,   1
    alsl.d      a4,     a5,     a4,   1
    alsl.d      a0,     a1,     a0,   1
.endr
endfunc
.endm

AVG_W4 4, 16
AVG_W4 4, 8
AVG_W4 4, 4

.macro AVG_W8 w, h
function x265_pixel_avg_\w\()x\h\()_lsx
.rept \h/2
    fld.d       f0,     a2,     0
    fld.d       f1,     a4,     0
    fldx.d      f2,     a2,     a3
    fldx.d      f3,     a4,     a5
    vavgr.bu    vr0,    vr1,    vr0
    vavgr.bu    vr1,    vr3,    vr2
    fst.d       f0,     a0,     0
    fstx.d      f1,     a0,     a1
    alsl.d      a2,     a3,     a2,   1
    alsl.d      a4,     a5,     a4,   1
    alsl.d      a0,     a1,     a0,   1
.endr
endfunc
.endm

AVG_W8 8, 32
AVG_W8 8, 16
AVG_W8 8, 8
AVG_W8 8, 4

.macro AVG_W16 w, h
function x265_pixel_avg_\w\()x\h\()_lsx
.rept \h
    vld         vr0,    a2,     0
    vld         vr1,    a4,     0
    vavgr.bu    vr0,    vr1,    vr0
.if \w == 12
    vstelm.d    vr0,    a0,     0,    0
    vstelm.w    vr0,    a0,     8,    2
.else
    vst         vr0,    a0,     0
.endif
    add.d       a2,     a2,     a3
    add.d       a4,     a4,     a5
    add.d       a0,     a0,     a1
.endr
endfunc
.endm

AVG_W16 12, 16
AVG_W16 16, 64
AVG_W16 16, 32
AVG_W16 16, 16
AVG_W16 16, 12
AVG_W16 16, 8
AVG_W16 16, 4

.macro AVG_W32 w, h
function x265_pixel_avg_\w\()x\h\()_lsx
.rept \h
    vld         vr0,    a2,     0
    vld         vr4,    a4,     0
    vld         vr1,    a2,     16
    vld         vr5,    a4,     16
    vavgr.bu    vr0,    vr4,    vr0
    vavgr.bu    vr1,    vr5,    vr1
    vst         vr0,    a0,     0
.if \w == 24
    vstelm.d    vr1,    a0,     16,   0
.else
    vst         vr1,    a0,     16
.endif
    add.d       a2,     a2,     a3
    add.d       a4,     a4,     a5
    add.d       a0,     a0,     a1
.endr
endfunc
.endm

AVG_W32 24, 32
AVG_W32 32, 64
AVG_W32 32, 32
AVG_W32 32, 24
AVG_W32 32, 16
AVG_W32 32, 8

.macro LASX_AVG_W32 w, h
function x265_pixel_avg_\w\()x\h\()_lasx
.rept \h/2 - 1
    xvld        xr0,    a2,     0
    xvld        xr1,    a4,     0
    xvldx       xr2,    a2,     a3
    xvldx       xr3,    a4,     a5
    xvavgr.bu   xr0,    xr1,    xr0
    xvavgr.bu   xr2,    xr3,    xr2
    xvst        xr0,    a0,     0
    xvstx       xr2,    a0,     a1
    alsl.d      a2,     a3,     a2,   1
    alsl.d      a4,     a5,     a4,   1
    alsl.d      a0,     a1,     a0,   1
.endr
    xvld        xr0,    a2,     0
    xvld        xr1,    a4,     0
    xvldx       xr2,    a2,     a3
    xvldx       xr3,    a4,     a5
    xvavgr.bu   xr0,    xr1,    xr0
    xvavgr.bu   xr2,    xr3,    xr2
    xvst        xr0,    a0,     0
    xvstx       xr2,    a0,     a1
endfunc
.endm

LASX_AVG_W32 32, 64
LASX_AVG_W32 32, 32
LASX_AVG_W32 32, 24
LASX_AVG_W32 32, 16
LASX_AVG_W32 32, 8

.macro AVG_W64 w, h
function x265_pixel_avg_\w\()x\h\()_lsx
    li.w        t0,     \h
1:  //h loop
    vld         vr0,    a2,     0
    vld         vr4,    a4,     0
    vld         vr1,    a2,     16
    vld         vr5,    a4,     16
    vld         vr2,    a2,     32
    vld         vr6,    a4,     32
.if \w > 48
    vld         vr3,    a2,     48
    vld         vr7,    a4,     48
.endif
    vavgr.bu    vr0,    vr4,    vr0
    vavgr.bu    vr1,    vr5,    vr1
    vavgr.bu    vr2,    vr6,    vr2
.if \w > 48
    vavgr.bu    vr3,    vr7,    vr3
.endif
    vst         vr0,    a0,     0
    vst         vr1,    a0,     16
    vst         vr2,    a0,     32
.if \w > 48
    vst         vr3,    a0,     48
.endif
    add.d       a2,     a2,     a3
    add.d       a4,     a4,     a5
    add.d       a0,     a0,     a1
    addi.w      t0,     t0,     -1
    bnez        t0,     1b
endfunc
.endm

AVG_W64 48, 64
AVG_W64 64, 64
AVG_W64 64, 48
AVG_W64 64, 32
AVG_W64 64, 16

.macro LASX_AVG_W64 w, h
function x265_pixel_avg_\w\()x\h\()_lasx
    li.w        t0,     \h
1:  //h loop
    xvld        xr0,    a2,     0
    xvld        xr1,    a2,     32
    xvld        xr2,    a4,     0
    xvld        xr3,    a4,     32
    xvavgr.bu   xr0,    xr2,    xr0
    xvavgr.bu   xr1,    xr3,    xr1
    xvst        xr0,    a0,     0
.if \w == 48
    vst         vr1,    a0,     32
.else
    xvst        xr1,    a0,     32
.endif
    add.d       a2,     a2,     a3
    add.d       a4,     a4,     a5
    add.d       a0,     a0,     a1
    addi.w      t0,     t0,     -1
    bnez        t0,     1b
endfunc
.endm

LASX_AVG_W64 48, 64
LASX_AVG_W64 64, 64
LASX_AVG_W64 64, 48
LASX_AVG_W64 64, 32
LASX_AVG_W64 64, 16

/*
 * void addAvg(const int16_t* src0, const int16_t* src1, pixel* dst,
 *             intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
 */
.macro ADDAVG_START
    li.w             t0,     16448
    slli.w           t1,     a3,     1
    slli.w           t2,     a4,     1
    vreplgr2vr.w     vr4,    t0
.endm

.macro ADDAVG_2xN  out0, idx
    fld.s            f0,     a0,     0
    fld.s            f1,     a1,     0
    fldx.s           f2,     a0,     t1
    fldx.s           f3,     a1,     t2
    vilvl.w          vr0,    vr2,    vr0
    vilvl.w          vr1,    vr3,    vr1
    vilvl.h          vr0,    vr1,    vr0
    vhaddw.w.h       \out0,  vr0,    vr0
    vadd.w           \out0,  \out0,  vr4
.if \idx == 0
    alsl.d           a0,     a3,     a0,    2
    alsl.d           a1,     a4,     a1,    2
.endif
.endm

function x265_addAvg_2x2_lsx
    ADDAVG_START
    ADDAVG_2xN       vr0,    1
    vssrani.h.w      vr0,    vr0,    7
    vssrani.bu.h     vr0,    vr0,    0
    vstelm.h         vr0,    a2,     0,     0
    add.d            a2,     a2,     a5
    vstelm.h         vr0,    a2,     0,     1
endfunc

function x265_addAvg_2x4_lsx
    ADDAVG_START
    ADDAVG_2xN       vr5,    0
    ADDAVG_2xN       vr6,    1
    vssrani.h.w      vr6,    vr5,    7
    vssrani.bu.h     vr0,    vr6,    0
    .irp i, 0, 1, 2
    vstelm.h         vr0,    a2,     0,     \i
    add.d            a2,     a2,     a5
    .endr
    vstelm.h         vr0,    a2,     0,     3
endfunc

function x265_addAvg_2x8_lsx
    ADDAVG_START
    ADDAVG_2xN       vr5,    0
    ADDAVG_2xN       vr6,    0
    ADDAVG_2xN       vr7,    0
    ADDAVG_2xN       vr8,    1
    vssrani.h.w      vr6,    vr5,    7
    vssrani.h.w      vr8,    vr7,    7
    vssrani.bu.h     vr8,    vr6,    0
    .irp i, 0, 1, 2, 3, 4, 5 ,6
    vstelm.h         vr8,    a2,     0,     \i
    add.d            a2,     a2,     a5
    .endr
    vstelm.h         vr8,    a2,     0,     7
endfunc

function x265_addAvg_2x16_lsx
    ADDAVG_START
    ADDAVG_2xN       vr5,    0
    ADDAVG_2xN       vr6,    0
    ADDAVG_2xN       vr7,    0
    ADDAVG_2xN       vr8,    0
    ADDAVG_2xN       vr9,    0
    ADDAVG_2xN       vr10,   0
    ADDAVG_2xN       vr11,   0
    ADDAVG_2xN       vr12,   1
    vssrani.h.w      vr6,    vr5,    7
    vssrani.h.w      vr8,    vr7,    7
    vssrani.bu.h     vr8,    vr6,    0
    vssrani.h.w      vr10,   vr9,    7
    vssrani.h.w      vr12,   vr11,   7
    vssrani.bu.h     vr12,   vr10,   0
    .irp i, 0, 1, 2, 3, 4, 5 ,6, 7
    vstelm.h         vr8,    a2,     0,     \i
    add.d            a2,     a2,     a5
    .endr
    .irp i, 0, 1, 2, 3, 4, 5 ,6
    vstelm.h         vr12,   a2,     0,     \i
    add.d            a2,     a2,     a5
    .endr
    vstelm.h         vr12,   a2,     0,     7
endfunc

function x265_addAvg_4x2_lsx
    ADDAVG_START
    fld.d            f0,     a0,     0
    fld.d            f1,     a1,     0
    fldx.d           f2,     a0,     t1
    fldx.d           f3,     a1,     t2
    vilvl.h          vr0,    vr1,    vr0
    vilvl.h          vr2,    vr3,    vr2
    vhaddw.w.h       vr0,    vr0,    vr0
    vhaddw.w.h       vr2,    vr2,    vr2
    vadd.w           vr0,    vr0,    vr4
    vadd.w           vr2,    vr2,    vr4
    vssrani.h.w      vr2,    vr0,    7
    vssrani.bu.h     vr2,    vr2,    0
    vstelm.w         vr2,    a2,     0,     0
    add.d            a2,     a2,     a5
    vstelm.w         vr2,    a2,     0,     1
endfunc

.macro ADDAVG_W4_H2 h
function x265_addAvg_4x\h\()_lsx
    ADDAVG_START
    li.w             t7,     \h/4
1:
    //h0h1
    fld.d            f0,     a0,     0
    fld.d            f1,     a1,     0
    fldx.d           f2,     a0,     t1
    fldx.d           f3,     a1,     t2
    alsl.d           a0,     a3,     a0,    2
    alsl.d           a1,     a4,     a1,    2
    vilvl.h          vr0,    vr1,    vr0
    vilvl.h          vr1,    vr3,    vr2
    vhaddw.w.h       vr0,    vr0,    vr0
    vhaddw.w.h       vr1,    vr1,    vr1
    vadd.w           vr0,    vr0,    vr4
    vadd.w           vr1,    vr1,    vr4
    vssrani.h.w      vr1,    vr0,    7
    //h2h3
    fld.d            f2,     a0,     0
    fld.d            f3,     a1,     0
    fldx.d           f5,     a0,     t1
    fldx.d           f6,     a1,     t2
    alsl.d           a0,     a3,     a0,    2
    alsl.d           a1,     a4,     a1,    2
    vilvl.h          vr2,    vr3,    vr2
    vilvl.h          vr3,    vr6,    vr5
    vhaddw.w.h       vr2,    vr2,    vr2
    vhaddw.w.h       vr3,    vr3,    vr3
    vadd.w           vr2,    vr2,    vr4
    vadd.w           vr3,    vr3,    vr4
    vssrani.h.w      vr3,    vr2,    7
    //pack h0h1h2h3 and store
    vssrani.bu.h     vr3,    vr1,    0
    .irp i, 0, 1, 2, 3
    vstelm.w         vr3,    a2,     0,     \i
    add.d            a2,     a2,     a5
    .endr
    addi.w           t7,     t7,     -1
    bnez             t7,     1b
endfunc
.endm

ADDAVG_W4_H2 4
ADDAVG_W4_H2 8
ADDAVG_W4_H2 16
ADDAVG_W4_H2 32

.macro ADDAVG_W6_H2 h
function x265_addAvg_6x\h\()_lsx
    ADDAVG_START
    li.w             t7,     \h/2
1:
    //h0
    vld              vr0,    a0,     0
    vld              vr1,    a1,     0
    vilvh.h          vr2,    vr1,    vr0
    vilvl.h          vr1,    vr1,    vr0
    vhaddw.w.h       vr0,    vr1,    vr1
    vhaddw.w.h       vr1,    vr2,    vr2
    vadd.w           vr0,    vr0,    vr4
    vadd.w           vr1,    vr1,    vr4
    //h1
    vldx             vr5,    a0,     t1
    vldx             vr6,    a1,     t2
    alsl.d           a0,     a3,     a0,    2
    alsl.d           a1,     a4,     a1,    2
    vilvh.h          vr7,    vr6,    vr5
    vilvl.h          vr6,    vr6,    vr5
    vhaddw.w.h       vr5,    vr6,    vr6
    vhaddw.w.h       vr6,    vr7,    vr7
    vadd.w           vr5,    vr5,    vr4
    vadd.w           vr6,    vr6,    vr4
    //pacck and store
    vssrani.h.w      vr5,    vr0,    7 //02
    vssrani.h.w      vr6,    vr1,    7 //13
    vssrani.bu.h     vr6,    vr5,    0 //0213
    vstelm.w         vr6,    a2,     0,     0
    vstelm.h         vr6,    a2,     4,     4
    add.d            a2,     a2,     a5
    vstelm.w         vr6,    a2,     0,     1
    vstelm.h         vr6,    a2,     4,     6
    add.d            a2,     a2,     a5
    addi.w           t7,     t7,     -1
    bnez             t7,     1b
endfunc
.endm

ADDAVG_W6_H2 8
ADDAVG_W6_H2 16

.macro ADDAVG_W8_H2 h
function x265_addAvg_8x\h\()_lsx
    ADDAVG_START
    li.w             t7,     \h/2
1:
    vld              vr0,    a0,     0
    vld              vr1,    a1,     0
    vldx             vr2,    a0,     t1
    vldx             vr3,    a1,     t2
    alsl.d           a0,     a3,     a0,    2
    alsl.d           a1,     a4,     a1,    2
    vilvh.h          vr5,    vr1,    vr0
    vilvl.h          vr0,    vr1,    vr0
    vilvh.h          vr6,    vr3,    vr2
    vilvl.h          vr3,    vr3,    vr2
    .irp i, vr5, vr0, vr6, vr3
    vhaddw.w.h       \i,     \i,     \i
    vadd.w           \i,     \i,     vr4
    .endr
    vssrani.h.w      vr5,    vr0,    7
    vssrani.h.w      vr6,    vr3,    7
    vssrani.bu.h     vr6,    vr5,    0
    vstelm.d         vr6,    a2,     0,     0
    add.d            a2,     a2,     a5
    vstelm.d         vr6,    a2,     0,     1
    add.d            a2,     a2,     a5
    addi.w           t7,     t7,     -1
    bnez             t7,     1b
endfunc
.endm

ADDAVG_W8_H2 2
ADDAVG_W8_H2 4
ADDAVG_W8_H2 6
ADDAVG_W8_H2 8
ADDAVG_W8_H2 12
ADDAVG_W8_H2 16
ADDAVG_W8_H2 32
ADDAVG_W8_H2 64

.macro ADDAVG_W12_H2 h
function x265_addAvg_12x\h\()_lsx
    ADDAVG_START
    li.w             t7,     \h/2
1:
    //h0
    vld              vr0,    a0,     0
    vld              vr1,    a0,     16
    vld              vr2,    a1,     0
    vld              vr3,    a1,     16
    add.d            a0,     a0,     t1
    add.d            a1,     a1,     t2
    //h1
    vld              vr5,    a0,     0
    vld              vr6,    a0,     16
    vld              vr7,    a1,     0
    vld              vr8,    a1,     16
    add.d            a0,     a0,     t1
    add.d            a1,     a1,     t2
    vilvh.h          vr9,    vr2,    vr0
    vilvl.h          vr0,    vr2,    vr0
    vilvl.h          vr2,    vr3,    vr1
    vilvh.h          vr10,   vr7,    vr5
    vilvl.h          vr5,    vr7,    vr5
    vilvl.h          vr3,    vr8,    vr6
    .irp i, vr9, vr0, vr2, vr10, vr5, vr3
    vhaddw.w.h       \i,     \i,     \i
    vadd.w           \i,     \i,     vr4
    .endr
    //pack and store
    vssrani.h.w      vr9,    vr0,    7
    vssrani.h.w      vr10,   vr5,    7
    vssrani.h.w      vr3,    vr2,    7
    vssrani.bu.h     vr10,   vr9,    0
    vssrani.bu.h     vr3,    vr3,    0
    vstelm.d         vr10,   a2,     0,     0
    vstelm.w         vr3,    a2,     8,     0
    add.d            a2,     a2,     a5
    vstelm.d         vr10,   a2,     0,     1
    vstelm.w         vr3,    a2,     8,     1
    add.d            a2,     a2,     a5
    addi.w           t7,     t7,     -1
    bnez             t7,     1b
endfunc
.endm

ADDAVG_W12_H2 16
ADDAVG_W12_H2 32

.macro ADDAVG_W16_H2 h
function x265_addAvg_16x\h\()_lsx
    ADDAVG_START
    li.w             t7,     \h/2
1:
    //h0
    vld              vr0,    a0,     0
    vld              vr1,    a0,     16
    vld              vr2,    a1,     0
    vld              vr3,    a1,     16
    add.d            a0,     a0,     t1
    add.d            a1,     a1,     t2
    //h1
    vld              vr6,    a0,     0
    vld              vr7,    a0,     16
    vld              vr8,    a1,     0
    vld              vr9,    a1,     16
    add.d            a0,     a0,     t1
    add.d            a1,     a1,     t2
    vilvh.h          vr11,   vr2,    vr0
    vilvl.h          vr10,   vr2,    vr0
    vilvh.h          vr13,   vr3,    vr1
    vilvl.h          vr12,   vr3,    vr1
    vilvh.h          vr15,   vr8,    vr6
    vilvl.h          vr14,   vr8,    vr6
    vilvh.h          vr17,   vr9,    vr7
    vilvl.h          vr16,   vr9,    vr7
    .irp i, vr10, vr11, vr12, vr13, vr14, vr15, vr16, vr17
    vhaddw.w.h       \i,     \i,     \i
    vadd.w           \i,     \i,     vr4
    .endr
    //pack and store
    vssrani.h.w      vr11,   vr10,   7
    vssrani.h.w      vr13,   vr12,   7
    vssrani.h.w      vr15,   vr14,   7
    vssrani.h.w      vr17,   vr16,   7
    vssrani.bu.h     vr13,   vr11,   0
    vssrani.bu.h     vr17,   vr15,   0
    vst              vr13,   a2,     0
    add.d            a2,     a2,     a5
    vst              vr17,   a2,     0
    add.d            a2,     a2,     a5
    addi.w           t7,     t7,     -1
    bnez             t7,     1b
endfunc
.endm

ADDAVG_W16_H2 4
ADDAVG_W16_H2 8
ADDAVG_W16_H2 12
ADDAVG_W16_H2 16
ADDAVG_W16_H2 24
ADDAVG_W16_H2 32
ADDAVG_W16_H2 64

.macro ADDAVG_W24_H2 h
function x265_addAvg_24x\h\()_lsx
    ADDAVG_START
    li.w             t7,     \h/2
    1:
    //h0
    vld              vr0,    a0,     0
    vld              vr1,    a0,     16
    vld              vr2,    a0,     32
    vld              vr3,    a1,     0
    vld              vr5,    a1,     16
    vld              vr6,    a1,     32
    add.d            a0,     a0,     t1
    add.d            a1,     a1,     t2
    //h1
    vld              vr7,    a0,     0
    vld              vr8,    a0,     16
    vld              vr9,    a0,     32
    vld              vr10,   a1,     0
    vld              vr11,   a1,     16
    vld              vr12,   a1,     32
    add.d            a0,     a0,     t1
    add.d            a1,     a1,     t2
    vilvh.h          vr14,   vr3,    vr0
    vilvl.h          vr13,   vr3,    vr0
    vilvh.h          vr16,   vr5,    vr1
    vilvl.h          vr15,   vr5,    vr1
    vilvh.h          vr18,   vr6,    vr2
    vilvl.h          vr17,   vr6,    vr2
    .irp i, vr13, vr14, vr15, vr16, vr17, vr18
    vhaddw.w.h       \i,     \i,     \i
    vadd.w           \i,     \i,     vr4
    .endr
    vssrani.h.w      vr14,   vr13,   7
    vssrani.h.w      vr16,   vr15,   7
    vssrani.h.w      vr18,   vr17,   7
    vssrani.bu.h     vr16,   vr14,   0
    vilvh.h          vr14,   vr10,   vr7
    vilvl.h          vr13,   vr10,   vr7
    vilvh.h          vr17,   vr11,   vr8
    vilvl.h          vr15,   vr11,   vr8
    vilvh.h          vr20,   vr12,   vr9
    vilvl.h          vr19,   vr12,   vr9
    .irp i, vr13, vr14, vr15, vr17, vr19, vr20
    vhaddw.w.h       \i,     \i,     \i
    vadd.w           \i,     \i,     vr4
    .endr
    vssrani.h.w      vr14,   vr13,   7
    vssrani.h.w      vr17,   vr15,   7
    vssrani.h.w      vr20,   vr19,   7
    vssrani.bu.h     vr17,   vr14,   0
    vssrani.bu.h     vr20,   vr18,   0
    vst              vr16,   a2,     0
    vstelm.d         vr20,   a2,     16,   0
    add.d            a2,     a2,     a5
    vst              vr17,   a2,     0
    vstelm.d         vr20,   a2,     16,   1
    add.d            a2,     a2,     a5
    addi.w           t7,     t7,     -1
    bnez             t7,     1b
endfunc
.endm

ADDAVG_W24_H2 32
ADDAVG_W24_H2 64

.macro ADDAVG_W32_H1
    vld              vr0,    a0,     0
    vld              vr1,    a0,     16
    vld              vr2,    a0,     32
    vld              vr3,    a0,     48
    vld              vr5,    a1,     0
    vld              vr6,    a1,     16
    vld              vr7,    a1,     32
    vld              vr8,    a1,     48
    add.d            a0,     a0,     t1
    add.d            a1,     a1,     t2
    vilvh.h          vr10,   vr5,    vr0
    vilvl.h          vr9,    vr5,    vr0
    vilvh.h          vr12,   vr6,    vr1
    vilvl.h          vr11,   vr6,    vr1
    vilvh.h          vr14,   vr7,    vr2
    vilvl.h          vr13,   vr7,    vr2
    vilvh.h          vr16,   vr8,    vr3
    vilvl.h          vr15,   vr8,    vr3
    .irp i, vr10, vr9, vr12, vr11, vr14, vr13, vr16, vr15
    vhaddw.w.h       \i,     \i,     \i
    vadd.w           \i,     \i,     vr4
    .endr
    vssrani.h.w      vr10,   vr9,    7
    vssrani.h.w      vr12,   vr11,   7
    vssrani.h.w      vr14,   vr13,   7
    vssrani.h.w      vr16,   vr15,   7
    vssrani.bu.h     vr12,   vr10,   0
    vssrani.bu.h     vr16,   vr14,   0
    vst              vr12,   a2,     0
    vst              vr16,   a2,     16
    add.d            a2,     a2,     a5
.endm

.macro ADDAVG_W32_H4 h
function x265_addAvg_32x\h\()_lsx
    ADDAVG_START
    li.w             t7,     \h/4
1:
    ADDAVG_W32_H1
    ADDAVG_W32_H1
    ADDAVG_W32_H1
    ADDAVG_W32_H1
    addi.w           t7,     t7,     -1
    bnez             t7,     1b
endfunc
.endm

ADDAVG_W32_H4 8
ADDAVG_W32_H4 16
ADDAVG_W32_H4 24
ADDAVG_W32_H4 32
ADDAVG_W32_H4 48
ADDAVG_W32_H4 64

.macro LASX_ADDAVG_W32_H1 w
    xvld             xr0,    a0,     0
    xvld             xr1,    a0,     32
    xvld             xr2,    a1,     0
    xvld             xr3,    a1,     32
    add.d            a0,     a0,     t1
    add.d            a1,     a1,     t2
    xvilvh.h         xr5,    xr2,    xr0
    xvilvl.h         xr4,    xr2,    xr0
    xvilvh.h         xr7,    xr3,    xr1
    xvilvl.h         xr6,    xr3,    xr1
    .irp i, xr4, xr5, xr6, xr7
    xvhaddw.w.h      \i,     \i,     \i
    xvadd.w          \i,     \i,     xr8
    .endr
    xvssrani.h.w     xr5,    xr4,    7
    xvssrani.h.w     xr7,    xr6,    7
    xvssrani.bu.h    xr7,    xr5,    0
    xvpermi.d        xr7,    xr7,    0xd8
.if \w == 24
    vst              vr7,    a2,     0
    xvpermi.q        xr7,    xr7,    0x01
    fst.d            f7,     a2,     16
.else
    xvst             xr7,    a2,     0
.endif
    add.d            a2,     a2,     a5
.endm

.macro LASX_ADDAVG_W32_H4 w, h
function x265_addAvg_\w\()x\h\()_lasx
    li.w             t0,     16448
    slli.w           t1,     a3,     1
    slli.w           t2,     a4,     1
    xvreplgr2vr.w    xr8,    t0
    li.w             t7,     \h/4
1:
    LASX_ADDAVG_W32_H1 \w
    LASX_ADDAVG_W32_H1 \w
    LASX_ADDAVG_W32_H1 \w
    LASX_ADDAVG_W32_H1 \w
    addi.w           t7,     t7,     -1
    bnez             t7,     1b
endfunc
.endm

LASX_ADDAVG_W32_H4 32 8
LASX_ADDAVG_W32_H4 32 16
LASX_ADDAVG_W32_H4 32 24
LASX_ADDAVG_W32_H4 32 32
LASX_ADDAVG_W32_H4 32 48
LASX_ADDAVG_W32_H4 32 64

LASX_ADDAVG_W32_H4 24 32
LASX_ADDAVG_W32_H4 24 64

function x265_addAvg_48x64_lsx
    ADDAVG_START
    li.w             t7,     64
1:
    vld              vr0,    a0,     0
    vld              vr1,    a0,     16
    vld              vr2,    a0,     32
    vld              vr3,    a0,     48
    vld              vr5,    a0,     64
    vld              vr6,    a0,     80
    vld              vr7,    a1,     0
    vld              vr8,    a1,     16
    vld              vr9,    a1,     32
    vld              vr10,   a1,     48
    vld              vr11,   a1,     64
    vld              vr12,   a1,     80
    add.d            a0,     a0,     t1
    add.d            a1,     a1,     t2
    vilvh.h          vr14,   vr7,    vr0
    vilvl.h          vr13,   vr7,    vr0
    vilvh.h          vr16,   vr8,    vr1
    vilvl.h          vr15,   vr8,    vr1
    vilvh.h          vr18,   vr9,    vr2
    vilvl.h          vr17,   vr9,    vr2
    vilvh.h          vr20,   vr10,   vr3
    vilvl.h          vr19,   vr10,   vr3
    vilvh.h          vr22,   vr11,   vr5
    vilvl.h          vr21,   vr11,   vr5
    vilvh.h          vr23,   vr12,   vr6
    vilvl.h          vr0,    vr12,   vr6
    .irp i, vr14, vr13, vr16, vr15, vr18, vr17, \
            vr20, vr19, vr22, vr21, vr23, vr0
    vhaddw.w.h       \i,     \i,     \i
    vadd.w           \i,     \i,     vr4
    .endr
    vssrani.h.w      vr14,   vr13,   7
    vssrani.h.w      vr16,   vr15,   7
    vssrani.h.w      vr18,   vr17,   7
    vssrani.h.w      vr20,   vr19,   7
    vssrani.h.w      vr22,   vr21,   7
    vssrani.h.w      vr23,   vr0,    7
    vssrani.bu.h     vr16,   vr14,   0
    vssrani.bu.h     vr20,   vr18,   0
    vssrani.bu.h     vr23,   vr22,   0
    vst              vr16,   a2,     0
    vst              vr20,   a2,     16
    vst              vr23,   a2,     32
    add.d            a2,     a2,     a5
    addi.w           t7,     t7,     -1
    bnez             t7,     1b
endfunc

.macro ADDAVG_W64_H0 in0, in1, in2, in3, out0
    vilvh.h          vr18,   \in0,   \in1
    vilvl.h          vr17,   \in0,   \in1
    vilvh.h          \out0,  \in2,   \in3
    vilvl.h          vr19,   \in2,   \in3
    .irp i, vr18, vr17, \out0, vr19
    vhaddw.w.h       \i,     \i,     \i
    vadd.w           \i,     \i,     vr4
    .endr
    vssrani.h.w      vr18,   vr17,   7
    vssrani.h.w      \out0,  vr19,   7
    vssrani.bu.h     \out0,  vr18,   0
.endm

.macro ADDAVG_W64_H1 h
function x265_addAvg_64x\h\()_lsx
    ADDAVG_START
    li.w             t7,     \h
1:
    vld              vr0,    a0,     0
    vld              vr1,    a0,     16
    vld              vr2,    a0,     32
    vld              vr3,    a0,     48
    vld              vr5,    a0,     64
    vld              vr6,    a0,     80
    vld              vr7,    a0,     96
    vld              vr8,    a0,     112
    vld              vr9,    a1,     0
    vld              vr10,   a1,     16
    vld              vr11,   a1,     32
    vld              vr12,   a1,     48
    vld              vr13,   a1,     64
    vld              vr14,   a1,     80
    vld              vr15,   a1,     96
    vld              vr16,   a1,     112
    add.d            a0,     a0,     t1
    add.d            a1,     a1,     t2
    ADDAVG_W64_H0 vr9, vr0, vr10, vr1, vr20
    ADDAVG_W64_H0 vr11, vr2, vr12, vr3, vr21
    ADDAVG_W64_H0 vr13, vr5, vr14, vr6, vr22
    ADDAVG_W64_H0 vr15, vr7, vr16, vr8, vr23
    vst              vr20,   a2,     0
    vst              vr21,   a2,     16
    vst              vr22,   a2,     32
    vst              vr23,   a2,     48
    add.d            a2,     a2,     a5
    addi.w           t7,     t7,     -1
    bnez             t7,     1b
endfunc
.endm

ADDAVG_W64_H1 16
ADDAVG_W64_H1 32
ADDAVG_W64_H1 48
ADDAVG_W64_H1 64

.macro LASX_ADDAVG_W64_H0 in0, in1, in2, in3, out0
    xvilvh.h         xr18,   \in0,   \in1
    xvilvl.h         xr17,   \in0,   \in1
    xvilvh.h         \out0,  \in2,   \in3
    xvilvl.h         xr19,   \in2,   \in3
    .irp i, xr18, xr17, \out0, xr19
    xvhaddw.w.h      \i,     \i,     \i
    xvadd.w          \i,     \i,     xr8
    .endr
    xvssrani.h.w     xr18,   xr17,   7
    xvssrani.h.w     \out0,  xr19,   7
    xvssrani.bu.h    \out0,  xr18,   0
    xvpermi.d        \out0,  \out0,  0xd8
.endm

.macro LASX_ADDAVG_W64_H1 w, h
function x265_addAvg_\w\()x\h\()_lasx
    li.w             t0,     16448
    slli.w           t1,     a3,     1
    slli.w           t2,     a4,     1
    xvreplgr2vr.w    xr8,    t0
    li.w             t7,     \h
1:
    xvld             xr0,    a0,     0
    xvld             xr1,    a0,     32
    xvld             xr2,    a0,     64
    xvld             xr3,    a0,     96
    xvld             xr4,    a1,     0
    xvld             xr5,    a1,     32
    xvld             xr6,    a1,     64
    xvld             xr7,    a1,     96
    add.d            a0,     a0,     t1
    add.d            a1,     a1,     t2
    LASX_ADDAVG_W64_H0 xr4, xr0, xr5, xr1, xr20
    LASX_ADDAVG_W64_H0 xr6, xr2, xr7, xr3, xr21
    xvst             xr20,   a2,     0
.if \w == 48
    vst              vr21,   a2,     32
.else
    xvst             xr21,   a2,     32
.endif
    add.d            a2,     a2,     a5
    addi.w           t7,     t7,     -1
    bnez             t7,     1b
endfunc
.endm

LASX_ADDAVG_W64_H1 64 16
LASX_ADDAVG_W64_H1 64 32
LASX_ADDAVG_W64_H1 64 48
LASX_ADDAVG_W64_H1 64 64

LASX_ADDAVG_W64_H1 48 64
