/*****************************************************************************
 * Copyright (C) 2024 MulticoreWare, Inc
 *
 * Authors: Hecai Yuan <yuanhecai@loongson.cn>
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

.macro pixel_sse_4x4_core_lsx out
    FLDS_LOADX_4  a0,     a1,     t0,   t1, f0, f1, f2, f3
    FLDS_LOADX_4  a2,     a3,     t2,   t3, f4, f5, f6, f7

    vilvl.w       vr0,    vr1,    vr0
    vilvl.w       vr1,    vr3,    vr2
    vilvl.w       vr4,    vr5,    vr4
    vilvl.w       vr5,    vr7,    vr6
    vilvl.d       vr0,    vr1,    vr0
    vilvl.d       vr4,    vr5,    vr4
    vsubwev.h.bu  vr1,    vr0,    vr4
    vsubwod.h.bu  vr2,    vr0,    vr4
    vmul.h        vr5,    vr1,    vr1
    vmul.h        vr6,    vr2,    vr2
    vhaddw.wu.hu  vr5,    vr5,    vr5
    vhaddw.wu.hu  vr6,    vr6,    vr6
    vadd.w        \out,   vr5,    vr6
.endm

function x265_pixel_sse_4x4_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     a1,     t0
    slli.d        t2,     a3,     1
    add.d         t3,     a3,     t2

    pixel_sse_4x4_core_lsx vr5

    vhaddw.d.w    vr5,    vr5,    vr5
    vhaddw.q.d    vr5,    vr5,    vr5
    vpickve2gr.w  a0,     vr5,    0
endfunc

function x265_pixel_sse_4x8_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     a1,     t0
    slli.d        t2,     a3,     1
    add.d         t3,     a3,     t2

    pixel_sse_4x4_core_lsx vr10

    alsl.d        a0,     a1,     a0,    2
    alsl.d        a2,     a3,     a2,    2
    pixel_sse_4x4_core_lsx vr5

    vadd.w        vr5,    vr5,    vr10
    vhaddw.d.w    vr5,    vr5,    vr5
    vhaddw.q.d    vr5,    vr5,    vr5
    vpickve2gr.w  a0,     vr5,    0
endfunc

.macro pixel_sse_8x4_core_lsx out
    FLDD_LOADX_4  a0,     a1,     t0,   t1, f0, f1, f2, f3
    FLDD_LOADX_4  a2,     a3,     t2,   t3, f4, f5, f6, f7
    vilvl.d       vr0,    vr1,    vr0
    vilvl.d       vr1,    vr3,    vr2
    vilvl.d       vr4,    vr5,    vr4
    vilvl.d       vr5,    vr7,    vr6
    vsubwev.h.bu  vr2,    vr0,    vr4
    vsubwod.h.bu  vr3,    vr0,    vr4
    vsubwev.h.bu  vr6,    vr1,    vr5
    vsubwod.h.bu  vr7,    vr1,    vr5
    vmul.h        vr2,    vr2,    vr2
    vmul.h        vr3,    vr3,    vr3
    vmul.h        vr6,    vr6,    vr6
    vmul.h        vr7,    vr7,    vr7
    vhaddw.wu.hu  vr2,    vr2,    vr2
    vhaddw.wu.hu  vr3,    vr3,    vr3
    vhaddw.wu.hu  vr6,    vr6,    vr6
    vhaddw.wu.hu  vr7,    vr7,    vr7
    vadd.w        vr2,    vr2,    vr3
    vadd.w        vr6,    vr6,    vr7
    vadd.w        \out,   vr2,    vr6
.endm

function x265_pixel_sse_8x8_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     a1,     t0
    slli.d        t2,     a3,     1
    add.d         t3,     a3,     t2

    pixel_sse_8x4_core_lsx vr10

    alsl.d        a0,     a1,     a0,   2
    alsl.d        a2,     a3,     a2,   2
    pixel_sse_8x4_core_lsx vr11

    vadd.w        vr10,   vr10,   vr11
    vhaddw.d.w    vr10,   vr10,   vr10
    vhaddw.q.d    vr10,   vr10,   vr10
    vpickve2gr.w  a0,     vr10,   0
endfunc

function x265_pixel_sse_8x16_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     a1,     t0
    slli.d        t2,     a3,     1
    add.d         t3,     a3,     t2

    pixel_sse_8x4_core_lsx vr10

.rept 3
    alsl.d        a0,     a1,     a0,   2
    alsl.d        a2,     a3,     a2,   2
    pixel_sse_8x4_core_lsx vr11
    vadd.w        vr10,   vr10,   vr11
.endr

    vhaddw.d.w    vr10,   vr10,   vr10
    vhaddw.q.d    vr10,   vr10,   vr10
    vpickve2gr.w  a0,     vr10,   0
endfunc

.macro pixel_sse_16x4_core_lsx out
    LSX_LOADX_4   a0,     a1,     t0,   t1, vr0, vr1, vr2, vr3
    LSX_LOADX_4   a2,     a3,     t2,   t3, vr4, vr5, vr6, vr7
    vsubwev.h.bu  vr8,    vr0,    vr4
    vsubwod.h.bu  vr9,    vr0,    vr4
    vsubwev.h.bu  vr10,   vr1,    vr5
    vsubwod.h.bu  vr11,   vr1,    vr5
    vsubwev.h.bu  vr12,   vr2,    vr6
    vsubwod.h.bu  vr13,   vr2,    vr6
    vsubwev.h.bu  vr14,   vr3,    vr7
    vsubwod.h.bu  vr15,   vr3,    vr7
    vmul.h        vr8,    vr8,    vr8
    vmul.h        vr9,    vr9,    vr9
    vmul.h        vr10,   vr10,   vr10
    vmul.h        vr11,   vr11,   vr11
    vmul.h        vr12,   vr12,   vr12
    vmul.h        vr13,   vr13,   vr13
    vmul.h        vr14,   vr14,   vr14
    vmul.h        vr15,   vr15,   vr15
    vhaddw.wu.hu  vr8,    vr8,    vr8
    vhaddw.wu.hu  vr9,    vr9,    vr9
    vhaddw.wu.hu  vr10,   vr10,   vr10
    vhaddw.wu.hu  vr11,   vr11,   vr11
    vhaddw.wu.hu  vr12,   vr12,   vr12
    vhaddw.wu.hu  vr13,   vr13,   vr13
    vhaddw.wu.hu  vr14,   vr14,   vr14
    vhaddw.wu.hu  vr15,   vr15,   vr15
    vadd.w        vr8,    vr8,    vr9
    vadd.w        vr9,    vr10,   vr11
    vadd.w        vr10,   vr12,   vr13
    vadd.w        vr11,   vr14,   vr15
    vadd.w        vr8,    vr8,    vr9
    vadd.w        vr9,    vr10,   vr11
    vadd.w        \out,   vr8,    vr9
.endm

function x265_pixel_sse_16x16_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     a1,     t0
    slli.d        t2,     a3,     1
    add.d         t3,     a3,     t2

    pixel_sse_16x4_core_lsx vr16

.rept 3
    alsl.d        a0,     a1,     a0,   2
    alsl.d        a2,     a3,     a2,   2
    pixel_sse_16x4_core_lsx vr17
    vadd.w        vr16,   vr16,   vr17
.endr

    vhaddw.d.w    vr16,   vr16,   vr16
    vhaddw.q.d    vr16,   vr16,   vr16
    vpickve2gr.w  a0,     vr16,   0
endfunc

function x265_pixel_sse_16x32_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     a1,     t0
    slli.d        t2,     a3,     1
    add.d         t3,     a3,     t2

    pixel_sse_16x4_core_lsx vr16

.rept 7
    alsl.d        a0,     a1,     a0,   2
    alsl.d        a2,     a3,     a2,   2
    pixel_sse_16x4_core_lsx vr17
    vadd.w        vr16,   vr16,   vr17
.endr

    vhaddw.d.w    vr16,   vr16,   vr16
    vhaddw.q.d    vr16,   vr16,   vr16
    vpickve2gr.w  a0,     vr16,   0
endfunc

.macro pixel_sse_32x_lsx h
function x265_pixel_sse_32x\h\()_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     a1,     t0
    slli.d        t2,     a3,     1
    add.d         t3,     a3,     t2

    pixel_sse_16x4_core_lsx vr16

    addi.d        a0,     a0,     16
    addi.d        a2,     a2,     16
    pixel_sse_16x4_core_lsx vr17
    vadd.w        vr16,   vr16,   vr17

.rept (\h>>2)-1
    addi.d        a0,     a0,     -16
    addi.d        a2,     a2,     -16
    alsl.d        a0,     a1,     a0,   2
    alsl.d        a2,     a3,     a2,   2
    pixel_sse_16x4_core_lsx vr17
    vadd.w        vr16,   vr16,   vr17

    addi.d        a0,     a0,     16
    addi.d        a2,     a2,     16
    pixel_sse_16x4_core_lsx vr17
    vadd.w        vr16,   vr16,   vr17
.endr

    vhaddw.d.w    vr16,   vr16,   vr16
    vhaddw.q.d    vr16,   vr16,   vr16
    vpickve2gr.w  a0,     vr16,   0
endfunc
.endm

pixel_sse_32x_lsx 32
pixel_sse_32x_lsx 64

function x265_pixel_sse_64x64_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     a1,     t0
    slli.d        t2,     a3,     1
    add.d         t3,     a3,     t2

    pixel_sse_16x4_core_lsx vr16

.rept 3
    addi.d        a0,     a0,     16
    addi.d        a2,     a2,     16
    pixel_sse_16x4_core_lsx vr17
    vadd.w        vr16,   vr16,   vr17
.endr

.rept 15
    addi.d        a0,     a0,     -48
    addi.d        a2,     a2,     -48
    alsl.d        a0,     a1,     a0,   2
    alsl.d        a2,     a3,     a2,   2
    pixel_sse_16x4_core_lsx vr17
    vadd.w        vr16,   vr16,   vr17

.rept 3
    addi.d        a0,     a0,     16
    addi.d        a2,     a2,     16

    pixel_sse_16x4_core_lsx vr17
    vadd.w        vr16,   vr16,   vr17
.endr
.endr

    vhaddw.d.w    vr16,   vr16,   vr16
    vhaddw.q.d    vr16,   vr16,   vr16
    vpickve2gr.w  a0,     vr16,   0
endfunc

.macro pixel_sse_16x4_core_lasx out
    LSX_LOADX_4    a0,     a1,     t0,   t1, vr0, vr1, vr2, vr3
    LSX_LOADX_4    a2,     a3,     t2,   t3, vr4, vr5, vr6, vr7

    xvpermi.q      xr1,    xr0,    0x20
    xvpermi.q      xr3,    xr2,    0x20
    xvpermi.q      xr5,    xr4,    0x20
    xvpermi.q      xr7,    xr6,    0x20

    xvsubwev.h.bu  xr10,   xr1,    xr5
    xvsubwod.h.bu  xr11,   xr1,    xr5
    xvsubwev.h.bu  xr14,   xr3,    xr7
    xvsubwod.h.bu  xr15,   xr3,    xr7
    xvmul.h        xr10,   xr10,   xr10
    xvmul.h        xr11,   xr11,   xr11
    xvmul.h        xr14,   xr14,   xr14
    xvmul.h        xr15,   xr15,   xr15
    xvhaddw.wu.hu  xr10,   xr10,   xr10
    xvhaddw.wu.hu  xr11,   xr11,   xr11
    xvhaddw.wu.hu  xr14,   xr14,   xr14
    xvhaddw.wu.hu  xr15,   xr15,   xr15
    xvadd.w        xr8,    xr10,   xr11
    xvadd.w        xr9,    xr14,   xr15
    xvadd.w        \out,   xr8,    xr9
.endm

.macro pixel_sse_16x_lasx h
function x265_pixel_sse_16x\h\()_lasx
    slli.d        t0,     a1,     1
    add.d         t1,     a1,     t0
    slli.d        t2,     a3,     1
    add.d         t3,     a3,     t2

    pixel_sse_16x4_core_lasx xr16

.rept (\h>>2)-1
    alsl.d        a0,     a1,     a0,   2
    alsl.d        a2,     a3,     a2,   2
    pixel_sse_16x4_core_lasx xr17
    xvadd.w       xr16,   xr16,   xr17
.endr

    xvpermi.q     xr17,   xr16,   0x01
    vadd.w        vr16,   vr16,   vr17
    vhaddw.d.w    vr16,   vr16,   vr16
    vhaddw.q.d    vr16,   vr16,   vr16
    vpickve2gr.w  a0,     vr16,   0
endfunc
.endm

pixel_sse_16x_lasx 16
pixel_sse_16x_lasx 32

.macro pixel_sse_32x2_core_lasx out
    xvld           xr0,    a0,     0
    xvldx          xr1,    a0,     a1
    xvld           xr2,    a2,     0
    xvldx          xr3,    a2,     a3

    xvsubwev.h.bu  xr10,   xr0,    xr2
    xvsubwod.h.bu  xr11,   xr0,    xr2
    xvsubwev.h.bu  xr14,   xr1,    xr3
    xvsubwod.h.bu  xr15,   xr1,    xr3
    xvmulwev.w.h   xr20,   xr10,   xr10
    xvmulwod.w.h   xr21,   xr10,   xr10
    xvmaddwev.w.h  xr20,   xr11,   xr11
    xvmaddwod.w.h  xr21,   xr11,   xr11
    xvmaddwev.w.h  xr20,   xr14,   xr14
    xvmaddwod.w.h  xr21,   xr14,   xr14
    xvmaddwev.w.h  xr20,   xr15,   xr15
    xvmaddwod.w.h  xr21,   xr15,   xr15
    xvadd.w        \out,   xr20,   xr21
.endm

.macro pixel_sse_32x_lasx h
function x265_pixel_sse_32x\h\()_lasx

    pixel_sse_32x2_core_lasx xr16

.rept (\h>>1)-1
    alsl.d        a0,     a1,     a0,   1
    alsl.d        a2,     a3,     a2,   1
    pixel_sse_32x2_core_lasx xr17
    xvadd.w       xr16,   xr16,   xr17
.endr

    xvpermi.q     xr17,   xr16,   0x01
    vadd.w        vr16,   vr16,   vr17
    vhaddw.d.w    vr16,   vr16,   vr16
    vhaddw.q.d    vr16,   vr16,   vr16
    vpickve2gr.w  a0,     vr16,   0
endfunc
.endm

pixel_sse_32x_lasx 32
pixel_sse_32x_lasx 64

function x265_pixel_sse_64x64_lasx

    pixel_sse_32x2_core_lasx xr16
    addi.d        a0,     a0,     32
    addi.d        a2,     a2,     32
    pixel_sse_32x2_core_lasx xr17
    xvadd.w xr16, xr16, xr17

.rept 31
    addi.d        a0,     a0,     -32
    addi.d        a2,     a2,     -32
    alsl.d        a0,     a1,     a0,   1
    alsl.d        a2,     a3,     a2,   1
    pixel_sse_32x2_core_lasx xr17
    xvadd.w       xr16,   xr16,   xr17

    addi.d        a0,     a0,     32
    addi.d        a2,     a2,     32
    pixel_sse_32x2_core_lasx xr17
    xvadd.w       xr16,   xr16,   xr17
.endr

    xvpermi.q     xr17,   xr16,   0x01
    vadd.w        vr16,   vr16,   vr17
    vhaddw.d.w    vr16,   vr16,   vr16
    vhaddw.q.d    vr16,   vr16,   vr16
    vpickve2gr.w  a0,     vr16,   0
endfunc

function x265_pixel_ssd_s_4x4_lsx
    slli.d        a1,     a1,     1
    fld.d         f0,     a0,     0
    fldx.d        f1,     a0,     a1
    alsl.d        a0,     a1,     a0, 1
    fld.d         f2,     a0,     0
    fldx.d        f3,     a0,     a1
    vilvl.d       vr1,    vr1,    vr0
    vilvl.d       vr3,    vr3,    vr2
    vmulwev.w.h   vr5,    vr1,    vr1
    vmulwev.w.h   vr6,    vr3,    vr3
    vmaddwod.w.h  vr5,    vr1,    vr1
    vmaddwod.w.h  vr6,    vr3,    vr3
    vadd.w        vr7,    vr5,    vr6
    vhaddw.d.w    vr7,    vr7,    vr7
    vhaddw.q.d    vr7,    vr7,    vr7
    vpickve2gr.w  a0,     vr7,    0
endfunc


function x265_pixel_ssd_s_8x8_lsx
    slli.d        a1,     a1,     1

    vld           vr0,    a0,     0
    vldx          vr1,    a0,     a1
    alsl.d        a0,     a1,     a0,   1
    vld           vr2,    a0,     0
    vldx          vr3,    a0,     a1
    vmulwev.w.h   vr4,    vr0,    vr0
    vmulwev.w.h   vr5,    vr1,    vr1
    vmulwev.w.h   vr6,    vr2,    vr2
    vmulwev.w.h   vr7,    vr3,    vr3
    vmaddwod.w.h  vr4,    vr0,    vr0
    vmaddwod.w.h  vr5,    vr1,    vr1
    vmaddwod.w.h  vr6,    vr2,    vr2
    vmaddwod.w.h  vr7,    vr3,    vr3

    alsl.d        a0,     a1,     a0,   1
    vld           vr0,    a0,     0
    vldx          vr1,    a0,     a1
    alsl.d        a0,     a1,     a0,   1
    vld           vr2,    a0,     0
    vldx          vr3,    a0,     a1
    vmaddwev.w.h  vr4,    vr0,    vr0
    vmaddwev.w.h  vr5,    vr1,    vr1
    vmaddwev.w.h  vr6,    vr2,    vr2
    vmaddwev.w.h  vr7,    vr3,    vr3
    vmaddwod.w.h  vr4,    vr0,    vr0
    vmaddwod.w.h  vr5,    vr1,    vr1
    vmaddwod.w.h  vr6,    vr2,    vr2
    vmaddwod.w.h  vr7,    vr3,    vr3

    vadd.w        vr8,    vr5,    vr4
    vadd.w        vr9,    vr7,    vr6
    vadd.w        vr7,    vr8,    vr9

    vhaddw.d.w    vr7,    vr7,    vr7
    vhaddw.q.d    vr7,    vr7,    vr7
    vpickve2gr.w  a0,     vr7,    0
endfunc

function x265_pixel_ssd_s_16x16_lsx
    slli.d        a1,     a1,     1

    vld           vr0,    a0,     0
    vld           vr1,    a0,     16
    add.d         a0,     a0,     a1
    vld           vr2,    a0,     0
    vld           vr3,    a0,     16

    vmulwev.w.h   vr4,    vr0,    vr0
    vmulwev.w.h   vr5,    vr1,    vr1
    vmulwev.w.h   vr6,    vr2,    vr2
    vmulwev.w.h   vr7,    vr3,    vr3
    vmaddwod.w.h  vr4,    vr0,    vr0
    vmaddwod.w.h  vr5,    vr1,    vr1
    vmaddwod.w.h  vr6,    vr2,    vr2
    vmaddwod.w.h  vr7,    vr3,    vr3

.rept 7
    add.d         a0,     a1,     a0
    vld           vr0,    a0,     0
    vld           vr1,    a0,     16
    add.d         a0,     a1,     a0
    vld           vr2,    a0,     0
    vld           vr3,    a0,     16
    vmaddwev.w.h  vr4,    vr0,    vr0
    vmaddwev.w.h  vr5,    vr1,    vr1
    vmaddwev.w.h  vr6,    vr2,    vr2
    vmaddwev.w.h  vr7,    vr3,    vr3
    vmaddwod.w.h  vr4,    vr0,    vr0
    vmaddwod.w.h  vr5,    vr1,    vr1
    vmaddwod.w.h  vr6,    vr2,    vr2
    vmaddwod.w.h  vr7,    vr3,    vr3
.endr
    vadd.w        vr8,    vr5,    vr4
    vadd.w        vr9,    vr7,    vr6
    vadd.w        vr7,    vr8,    vr9
    vhaddw.d.w    vr7,    vr7,    vr7
    vhaddw.q.d    vr7,    vr7,    vr7
    vpickve2gr.w  a0,     vr7,    0
endfunc

function x265_pixel_ssd_s_32x32_lsx
    slli.d        a1,     a1,     1

    vld           vr0,    a0,     0
    vld           vr1,    a0,     16
    vld           vr2,    a0,     32
    vld           vr3,    a0,     48

    vmulwev.w.h   vr4,    vr0,    vr0
    vmulwev.w.h   vr5,    vr1,    vr1
    vmulwev.w.h   vr6,    vr2,    vr2
    vmulwev.w.h   vr7,    vr3,    vr3
    vmaddwod.w.h  vr4,    vr0,    vr0
    vmaddwod.w.h  vr5,    vr1,    vr1
    vmaddwod.w.h  vr6,    vr2,    vr2
    vmaddwod.w.h  vr7,    vr3,    vr3

.rept 31
    add.d         a0,     a1,     a0
    vld           vr0,    a0,     0
    vld           vr1,    a0,     16
    vld           vr2,    a0,     32
    vld           vr3,    a0,     48
    vmaddwev.w.h  vr4,    vr0,    vr0
    vmaddwev.w.h  vr5,    vr1,    vr1
    vmaddwev.w.h  vr6,    vr2,    vr2
    vmaddwev.w.h  vr7,    vr3,    vr3
    vmaddwod.w.h  vr4,    vr0,    vr0
    vmaddwod.w.h  vr5,    vr1,    vr1
    vmaddwod.w.h  vr6,    vr2,    vr2
    vmaddwod.w.h  vr7,    vr3,    vr3
.endr
    vadd.w        vr8,    vr5,    vr4
    vadd.w        vr9,    vr7,    vr6
    vadd.w        vr7,    vr8,    vr9
    vhaddw.d.w    vr7,    vr7,    vr7
    vhaddw.q.d    vr7,    vr7,    vr7
    vpickve2gr.w  a0,     vr7,    0
endfunc

function x265_pixel_ssd_s_64x64_lsx
    slli.d        a1,     a1,     1

    vld           vr0,    a0,     0
    vld           vr1,    a0,     16
    vld           vr2,    a0,     32
    vld           vr3,    a0,     48
    vld           vr10,   a0,     64
    vld           vr11,   a0,     80
    vld           vr12,   a0,     96
    vld           vr13,   a0,     112
    vmulwev.w.h   vr4,    vr0,    vr0
    vmulwev.w.h   vr5,    vr1,    vr1
    vmulwev.w.h   vr6,    vr2,    vr2
    vmulwev.w.h   vr7,    vr3,    vr3
    vmulwev.w.h   vr14,   vr10,   vr10
    vmulwev.w.h   vr15,   vr11,   vr11
    vmulwev.w.h   vr16,   vr12,   vr12
    vmulwev.w.h   vr17,   vr13,   vr13
    vmaddwod.w.h  vr4,    vr0,    vr0
    vmaddwod.w.h  vr5,    vr1,    vr1
    vmaddwod.w.h  vr6,    vr2,    vr2
    vmaddwod.w.h  vr7,    vr3,    vr3
    vmaddwod.w.h  vr14,   vr10,   vr10
    vmaddwod.w.h  vr15,   vr11,   vr11
    vmaddwod.w.h  vr16,   vr12,   vr12
    vmaddwod.w.h  vr17,   vr13,   vr13

.rept 63
    add.d         a0,     a1,     a0

    vld           vr0,    a0,     0
    vld           vr1,    a0,     16
    vld           vr2,    a0,     32
    vld           vr3,    a0,     48
    vld           vr10,   a0,     64
    vld           vr11,   a0,     80
    vld           vr12,   a0,     96
    vld           vr13,   a0,     112
    vmaddwev.w.h  vr4,    vr0,    vr0
    vmaddwev.w.h  vr5,    vr1,    vr1
    vmaddwev.w.h  vr6,    vr2,    vr2
    vmaddwev.w.h  vr7,    vr3,    vr3
    vmaddwev.w.h  vr14,   vr10,   vr10
    vmaddwev.w.h  vr15,   vr11,   vr11
    vmaddwev.w.h  vr16,   vr12,   vr12
    vmaddwev.w.h  vr17,   vr13,   vr13
    vmaddwod.w.h  vr4,    vr0,    vr0
    vmaddwod.w.h  vr5,    vr1,    vr1
    vmaddwod.w.h  vr6,    vr2,    vr2
    vmaddwod.w.h  vr7,    vr3,    vr3
    vmaddwod.w.h  vr14,   vr10,   vr10
    vmaddwod.w.h  vr15,   vr11,   vr11
    vmaddwod.w.h  vr16,   vr12,   vr12
    vmaddwod.w.h  vr17,   vr13,   vr13
.endr
    vadd.w        vr8,    vr5,    vr4
    vadd.w        vr9,    vr7,    vr6
    vadd.w        vr18,   vr15,   vr14
    vadd.w        vr19,   vr17,   vr16
    vadd.w        vr7,    vr8,    vr9
    vadd.w        vr17,   vr18,   vr19
    vadd.w        vr7,    vr7,    vr17
    vhaddw.d.w    vr7,    vr7,    vr7
    vhaddw.q.d    vr7,    vr7,    vr7
    vpickve2gr.w  a0,     vr7,    0
endfunc

function x265_pixel_ssd_s_16x16_lasx
    slli.d        a1,     a1,     1

    xvld           xr0,   a0,     0
    xvldx          xr1,   a0,     a1
    alsl.d         a0,    a1,     a0,  1
    xvld           xr2,   a0,     0
    xvldx          xr3,   a0,     a1
    xvmulwev.w.h   xr4,   xr0,    xr0
    xvmulwev.w.h   xr5,   xr1,    xr1
    xvmulwev.w.h   xr6,   xr2,    xr2
    xvmulwev.w.h   xr7,   xr3,    xr3
    xvmaddwod.w.h  xr4,   xr0,    xr0
    xvmaddwod.w.h  xr5,   xr1,    xr1
    xvmaddwod.w.h  xr6,   xr2,    xr2
    xvmaddwod.w.h  xr7,   xr3,    xr3

.rept 3
    alsl.d        a0,     a1,     a0,  1
    xvld          xr0,    a0,     0
    xvldx         xr1,    a0,     a1
    alsl.d        a0,     a1,     a0,  1
    xvld          xr2,    a0,     0
    xvldx         xr3,    a0,     a1

    xvmaddwev.w.h xr4,    xr0,    xr0
    xvmaddwev.w.h xr5,    xr1,    xr1
    xvmaddwev.w.h xr6,    xr2,    xr2
    xvmaddwev.w.h xr7,    xr3,    xr3
    xvmaddwod.w.h xr4,    xr0,    xr0
    xvmaddwod.w.h xr5,    xr1,    xr1
    xvmaddwod.w.h xr6,    xr2,    xr2
    xvmaddwod.w.h xr7,    xr3,    xr3
.endr
    xvadd.w       xr8,    xr5,    xr4
    xvadd.w       xr9,    xr7,    xr6
    xvadd.w       xr7,    xr8,    xr9
    xvhaddw.d.w   xr7,    xr7,    xr7
    xvhaddw.q.d   xr7,    xr7,    xr7
    xvpickve2gr.w t0,     xr7,    0
    xvpickve2gr.w t1,     xr7,    4
    add.w         a0,     t0,     t1
endfunc

function x265_pixel_ssd_s_32x32_lasx
    slli.d        a1,     a1,     1

    xvld           xr0,   a0,     0
    xvld           xr1,   a0,     32
    add.d          a0,    a0,     a1
    xvld           xr2,   a0,     0
    xvld           xr3,   a0,     32
    xvmulwev.w.h   xr4,   xr0,    xr0
    xvmulwev.w.h   xr5,   xr1,    xr1
    xvmulwev.w.h   xr6,   xr2,    xr2
    xvmulwev.w.h   xr7,   xr3,    xr3
    xvmaddwod.w.h  xr4,   xr0,    xr0
    xvmaddwod.w.h  xr5,   xr1,    xr1
    xvmaddwod.w.h  xr6,   xr2,    xr2
    xvmaddwod.w.h  xr7,   xr3,    xr3

.rept 15
    add.d         a0,     a1,     a0
    xvld          xr0,    a0,     0
    xvld          xr1,    a0,     32
    add.d         a0,     a0,     a1
    xvld          xr2,    a0,     0
    xvld          xr3,    a0,     32

    xvmaddwev.w.h xr4,    xr0,    xr0
    xvmaddwev.w.h xr5,    xr1,    xr1
    xvmaddwev.w.h xr6,    xr2,    xr2
    xvmaddwev.w.h xr7,    xr3,    xr3
    xvmaddwod.w.h xr4,    xr0,    xr0
    xvmaddwod.w.h xr5,    xr1,    xr1
    xvmaddwod.w.h xr6,    xr2,    xr2
    xvmaddwod.w.h xr7,    xr3,    xr3
.endr
    xvadd.w       xr8,    xr5,    xr4
    xvadd.w       xr9,    xr7,    xr6
    xvadd.w       xr7,    xr8,    xr9
    xvhaddw.d.w   xr7,    xr7,    xr7
    xvhaddw.q.d   xr7,    xr7,    xr7
    xvpickve2gr.w t0,     xr7,    0
    xvpickve2gr.w t1,     xr7,    4
    add.w         a0,     t0,     t1
endfunc

function x265_pixel_ssd_s_64x64_lasx
    slli.d        a1,     a1,     1

    xvld           xr0,   a0,     0
    xvld           xr1,   a0,     32
    xvld           xr2,   a0,     64
    xvld           xr3,   a0,     96
    xvmulwev.w.h   xr4,   xr0,    xr0
    xvmulwev.w.h   xr5,   xr1,    xr1
    xvmulwev.w.h   xr6,   xr2,    xr2
    xvmulwev.w.h   xr7,   xr3,    xr3
    xvmaddwod.w.h  xr4,   xr0,    xr0
    xvmaddwod.w.h  xr5,   xr1,    xr1
    xvmaddwod.w.h  xr6,   xr2,    xr2
    xvmaddwod.w.h  xr7,   xr3,    xr3

.rept 63
    add.d         a0,     a1,     a0
    xvld          xr0,    a0,     0
    xvld          xr1,    a0,     32
    xvld          xr2,    a0,     64
    xvld          xr3,    a0,     96

    xvmaddwev.w.h xr4,    xr0,    xr0
    xvmaddwev.w.h xr5,    xr1,    xr1
    xvmaddwev.w.h xr6,    xr2,    xr2
    xvmaddwev.w.h xr7,    xr3,    xr3
    xvmaddwod.w.h xr4,    xr0,    xr0
    xvmaddwod.w.h xr5,    xr1,    xr1
    xvmaddwod.w.h xr6,    xr2,    xr2
    xvmaddwod.w.h xr7,    xr3,    xr3
.endr
    xvadd.w       xr8,    xr5,    xr4
    xvadd.w       xr9,    xr7,    xr6
    xvadd.w       xr7,    xr8,    xr9
    xvhaddw.d.w   xr7,    xr7,    xr7
    xvhaddw.q.d   xr7,    xr7,    xr7
    xvpickve2gr.w t0,     xr7,    0
    xvpickve2gr.w t1,     xr7,    4
    add.w         a0,     t0,     t1
endfunc

function x265_pixel_sse_ss_4x4_lsx
    slli.d        a1,     a1,     1
    slli.d        a3,     a3,     1
    vld           vr0,    a0,     0
    vldx          vr1,    a0,     a1
    vld           vr2,    a2,     0
    vldx          vr3,    a2,     a3
    alsl.d        a0,     a1,     a0,   1
    alsl.d        a2,     a3,     a2,   1
    vld           vr4,    a0,     0
    vldx          vr5,    a0,     a1
    vld           vr6,    a2,     0
    vldx          vr7,    a2,     a3
    vilvl.d       vr1,    vr1,    vr0
    vilvl.d       vr3,    vr3,    vr2
    vilvl.d       vr5,    vr5,    vr4
    vilvl.d       vr7,    vr7,    vr6
    vsubwev.w.h   vr8,    vr1,    vr3
    vsubwod.w.h   vr9,    vr1,    vr3
    vsubwev.w.h   vr10,   vr5,    vr7
    vsubwod.w.h   vr11,   vr5,    vr7
    vmul.w        vr4,    vr8,    vr8
    vmul.w        vr5,    vr9,    vr9
    vmadd.w       vr4,    vr10,   vr10
    vmadd.w       vr5,    vr11,   vr11
    vadd.w        vr5,    vr5,    vr4
    vhaddw.d.w    vr5,    vr5,    vr5
    vhaddw.q.d    vr5,    vr5,    vr5
    vpickve2gr.w  a0,     vr5,    0
endfunc

.macro pixel_sse_ss_8x4_core_lsx out
    vld           vr0,    a0,     0
    vldx          vr1,    a0,     a1
    vld           vr2,    a2,     0
    vldx          vr3,    a2,     a3
    alsl.d        a0,     a1,     a0,   1
    alsl.d        a2,     a3,     a2,   1
    vld           vr4,    a0,     0
    vldx          vr5,    a0,     a1
    vld           vr6,    a2,     0
    vldx          vr7,    a2,     a3
    vsubwev.w.h   vr8,    vr0,    vr2
    vsubwod.w.h   vr9,    vr0,    vr2
    vsubwev.w.h   vr10,   vr1,    vr3
    vsubwod.w.h   vr11,   vr1,    vr3
    vsubwev.w.h   vr12,   vr4,    vr6
    vsubwod.w.h   vr13,   vr4,    vr6
    vsubwev.w.h   vr14,   vr5,    vr7
    vsubwod.w.h   vr15,   vr5,    vr7
    vmul.w        vr4,    vr8,    vr8
    vmul.w        vr5,    vr9,    vr9
    vmadd.w       vr4,    vr10,   vr10
    vmadd.w       vr5,    vr11,   vr11
    vmadd.w       vr4,    vr12,   vr12
    vmadd.w       vr5,    vr13,   vr13
    vmadd.w       vr4,    vr14,   vr14
    vmadd.w       vr5,    vr15,   vr15
    vadd.w        \out,   vr4,    vr5
.endm

function x265_pixel_sse_ss_8x8_lsx
    slli.d        a1,     a1,     1
    slli.d        a3,     a3,     1

    pixel_sse_ss_8x4_core_lsx vr20

    alsl.d        a0,     a1,     a0,   1
    alsl.d        a2,     a3,     a2,   1

    pixel_sse_ss_8x4_core_lsx vr5

    vadd.w        vr5,    vr5,    vr20
    vhaddw.d.w    vr5,    vr5,    vr5
    vhaddw.q.d    vr5,    vr5,    vr5
    vpickve2gr.w  a0,     vr5,    0
endfunc

.macro pixel_sse_ss_16x2_core_lsx out
    vld           vr0,    a0,     0
    vld           vr1,    a0,     16
    vld           vr2,    a2,     0
    vld           vr3,    a2,     16
    add.d         a0,     a0,     a1
    add.d         a2,     a3,     a2
    vld           vr4,    a0,     0
    vld           vr5,    a0,     16
    vld           vr6,    a2,     0
    vld           vr7,    a2,     16
    vsubwev.w.h   vr8,    vr0,    vr2
    vsubwod.w.h   vr9,    vr0,    vr2
    vsubwev.w.h   vr10,   vr1,    vr3
    vsubwod.w.h   vr11,   vr1,    vr3
    vsubwev.w.h   vr12,   vr4,    vr6
    vsubwod.w.h   vr13,   vr4,    vr6
    vsubwev.w.h   vr14,   vr5,    vr7
    vsubwod.w.h   vr15,   vr5,    vr7
    vmul.w        vr4,    vr8,    vr8
    vmul.w        vr5,    vr9,    vr9
    vmadd.w       vr4,    vr10,   vr10
    vmadd.w       vr5,    vr11,   vr11
    vmadd.w       vr4,    vr12,   vr12
    vmadd.w       vr5,    vr13,   vr13
    vmadd.w       vr4,    vr14,   vr14
    vmadd.w       vr5,    vr15,   vr15
    vadd.w        \out,   vr4,    vr5
.endm

function x265_pixel_sse_ss_16x16_lsx
    slli.d        a1,     a1,     1
    slli.d        a3,     a3,     1

    pixel_sse_ss_16x2_core_lsx vr20

.rept 7
    add.d         a0,     a0,     a1
    add.d         a2,     a3,     a2
    pixel_sse_ss_16x2_core_lsx vr5
    vadd.w        vr20,   vr5,    vr20
.endr

    vhaddw.d.w    vr20,   vr20,   vr20
    vhaddw.q.d    vr20,   vr20,   vr20
    vpickve2gr.w  a0,     vr20,   0
endfunc

.macro pixel_sse_ss_32x1_core_lsx out
    vld           vr0,    a0,     0
    vld           vr1,    a0,     16
    vld           vr2,    a0,     32
    vld           vr3,    a0,     48
    vld           vr4,    a2,     0
    vld           vr5,    a2,     16
    vld           vr6,    a2,     32
    vld           vr7,    a2,     48
    vsubwev.w.h   vr8,    vr0,    vr4
    vsubwod.w.h   vr9,    vr0,    vr4
    vsubwev.w.h   vr10,   vr1,    vr5
    vsubwod.w.h   vr11,   vr1,    vr5
    vsubwev.w.h   vr12,   vr2,    vr6
    vsubwod.w.h   vr13,   vr2,    vr6
    vsubwev.w.h   vr14,   vr3,    vr7
    vsubwod.w.h   vr15,   vr3,    vr7
    vmul.w        vr4,    vr8,    vr8
    vmul.w        vr5,    vr9,    vr9
    vmadd.w       vr4,    vr10,   vr10
    vmadd.w       vr5,    vr11,   vr11
    vmadd.w       vr4,    vr12,   vr12
    vmadd.w       vr5,    vr13,   vr13
    vmadd.w       vr4,    vr14,   vr14
    vmadd.w       vr5,    vr15,   vr15
    vadd.w        \out,   vr4,    vr5
.endm

function x265_pixel_sse_ss_32x32_lsx
    slli.d        a1,     a1,     1
    slli.d        a3,     a3,     1

    pixel_sse_ss_32x1_core_lsx vr20

.rept 31
    add.d         a0,     a0,     a1
    add.d         a2,     a3,     a2
    pixel_sse_ss_32x1_core_lsx vr5
    vadd.w        vr20,   vr5,    vr20
.endr

    vhaddw.d.w    vr20,   vr20,   vr20
    vhaddw.q.d    vr20,   vr20,   vr20
    vpickve2gr.w  a0,     vr20,   0
endfunc

function x265_pixel_sse_ss_64x64_lsx
    slli.d        a1,     a1,     1
    slli.d        a3,     a3,     1

    pixel_sse_ss_32x1_core_lsx vr20

    addi.d        a0,     a0,     64
    addi.d        a2,     a2,     64
    pixel_sse_ss_32x1_core_lsx vr5
    vadd.w        vr20,   vr5, vr20

.rept 63
    addi.d        a0,     a0,     -64
    addi.d        a2,     a2,     -64
    add.d         a0,     a0,     a1
    add.d         a2,     a3,     a2
    pixel_sse_ss_32x1_core_lsx vr5
    vadd.w        vr20,   vr5,    vr20

    addi.d        a0,     a0,     64
    addi.d        a2,     a2,     64
    pixel_sse_ss_32x1_core_lsx vr5
    vadd.w        vr20,   vr5,    vr20
.endr

    vhaddw.d.w    vr20,   vr20,   vr20
    vhaddw.q.d    vr20,   vr20,   vr20
    vpickve2gr.w  a0,     vr20,   0
endfunc

.macro pixel_sse_ss_8x4_core_lasx out
    vld           vr0,    a0,     0
    vldx          vr1,    a0,     a1
    vld           vr2,    a2,     0
    vldx          vr3,    a2,     a3
    alsl.d        a0,     a1,     a0,   1
    alsl.d        a2,     a3,     a2,   1
    vld           vr4,    a0,     0
    vldx          vr5,    a0,     a1
    vld           vr6,    a2,     0
    vldx          vr7,    a2,     a3
    xvpermi.q     xr1,    xr0,    0x20
    xvpermi.q     xr3,    xr2,    0x20
    xvpermi.q     xr5,    xr4,    0x20
    xvpermi.q     xr7,    xr6,    0x20

    xvsubwev.w.h   xr10,  xr1,    xr3
    xvsubwod.w.h   xr11,  xr1,    xr3
    xvsubwev.w.h   xr14,  xr5,    xr7
    xvsubwod.w.h   xr15,  xr5,    xr7
    xvmul.w        xr4,   xr10,   xr10
    xvmul.w        xr5,   xr11,   xr11
    xvmadd.w       xr4,   xr14,   xr14
    xvmadd.w       xr5,   xr15,   xr15
    xvadd.w        \out,  xr4,    xr5
.endm

function x265_pixel_sse_ss_8x8_lasx
    slli.d        a1,     a1,     1
    slli.d        a3,     a3,     1

    pixel_sse_ss_8x4_core_lasx xr20

    alsl.d        a0,     a1,     a0,   1
    alsl.d        a2,     a3,     a2,   1

    pixel_sse_ss_8x4_core_lasx xr5

    xvadd.w        xr5,    xr5,    xr20
    xvpermi.q      xr6,    xr5,    0x01
    vadd.w         vr5,    vr5,    vr6
    vhaddw.d.w     vr5,    vr5,    vr5
    vhaddw.q.d     vr5,    vr5,    vr5
    vpickve2gr.w   a0,     vr5,    0
endfunc

.macro pixel_sse_ss_16x4_core_lasx out
    xvld          xr0,    a0,     0
    xvldx         xr1,    a0,     a1
    xvld          xr2,    a2,     0
    xvldx         xr3,    a2,     a3
    alsl.d        a0,     a1,     a0,   1
    alsl.d        a2,     a3,     a2,   1
    xvld          xr4,    a0,     0
    xvldx         xr5,    a0,     a1
    xvld          xr6,    a2,     0
    xvldx         xr7,    a2,     a3
    xvsubwev.w.h  xr8,    xr0,    xr2
    xvsubwod.w.h  xr9,    xr0,    xr2
    xvsubwev.w.h  xr10,   xr1,    xr3
    xvsubwod.w.h  xr11,   xr1,    xr3
    xvsubwev.w.h  xr12,   xr4,    xr6
    xvsubwod.w.h  xr13,   xr4,    xr6
    xvsubwev.w.h  xr14,   xr5,    xr7
    xvsubwod.w.h  xr15,   xr5,    xr7
    xvmul.w       xr4,    xr8,    xr8
    xvmul.w       xr5,    xr9,    xr9
    xvmadd.w      xr4,    xr10,   xr10
    xvmadd.w      xr5,    xr11,   xr11
    xvmadd.w      xr4,    xr12,   xr12
    xvmadd.w      xr5,    xr13,   xr13
    xvmadd.w      xr4,    xr14,   xr14
    xvmadd.w      xr5,    xr15,   xr15
    xvadd.w       \out,   xr4,    xr5
.endm

function x265_pixel_sse_ss_16x16_lasx
    slli.d        a1,     a1,     1
    slli.d        a3,     a3,     1

    pixel_sse_ss_16x4_core_lasx xr20

.rept 3
    alsl.d        a0,     a1,     a0,   1
    alsl.d        a2,     a3,     a2,   1

    pixel_sse_ss_16x4_core_lasx xr5
    xvadd.w       xr20,   xr5,    xr20
.endr

    xvpermi.q     xr6,    xr20,   0x01
    vadd.w        vr5,    vr20,   vr6
    vhaddw.d.w    vr5,    vr5,    vr5
    vhaddw.q.d    vr5,    vr5,    vr5
    vpickve2gr.w  a0,     vr5,    0
endfunc

.macro pixel_sse_ss_32x2_core_lasx out
    xvld          xr0,    a0,     0
    xvld          xr1,    a0,     32
    xvld          xr2,    a2,     0
    xvld          xr3,    a2,     32
    add.d         a0,     a1,     a0
    add.d         a2,     a3,     a2
    xvld          xr4,    a0,     0
    xvld          xr5,    a0,     32
    xvld          xr6,    a2,     0
    xvld          xr7,    a2,     32
    xvsubwev.w.h  xr8,    xr0,    xr2
    xvsubwod.w.h  xr9,    xr0,    xr2
    xvsubwev.w.h  xr10,   xr1,    xr3
    xvsubwod.w.h  xr11,   xr1,    xr3
    xvsubwev.w.h  xr12,   xr4,    xr6
    xvsubwod.w.h  xr13,   xr4,    xr6
    xvsubwev.w.h  xr14,   xr5,    xr7
    xvsubwod.w.h  xr15,   xr5,    xr7
    xvmul.w       xr4,    xr8,    xr8
    xvmul.w       xr5,    xr9,    xr9
    xvmadd.w      xr4,    xr10,   xr10
    xvmadd.w      xr5,    xr11,   xr11
    xvmadd.w      xr4,    xr12,   xr12
    xvmadd.w      xr5,    xr13,   xr13
    xvmadd.w      xr4,    xr14,   xr14
    xvmadd.w      xr5,    xr15,   xr15
    xvadd.w       \out,   xr4,    xr5
.endm

function x265_pixel_sse_ss_32x32_lasx
    slli.d        a1,     a1,     1
    slli.d        a3,     a3,     1

    pixel_sse_ss_32x2_core_lasx xr20

.rept 15
    add.d         a0,     a1,     a0
    add.d         a2,     a3,     a2
    pixel_sse_ss_32x2_core_lasx xr7
    xvadd.w       xr20,   xr7,    xr20
.endr

    xvpermi.q     xr6,    xr20,   0x01
    vadd.w        vr5,    vr20,   vr6
    vhaddw.d.w    vr5,    vr5,    vr5
    vhaddw.q.d    vr5,    vr5,    vr5
    vpickve2gr.w  a0,     vr5,    0
endfunc

.macro pixel_sse_ss_64x1_core_lasx out
    xvld          xr0,    a0,     0
    xvld          xr1,    a0,     32
    xvld          xr2,    a0,     64
    xvld          xr3,    a0,     96
    xvld          xr4,    a2,     0
    xvld          xr5,    a2,     32
    xvld          xr6,    a2,     64
    xvld          xr7,    a2,     96
    xvsubwev.w.h  xr8,    xr0,    xr4
    xvsubwod.w.h  xr9,    xr0,    xr4
    xvsubwev.w.h  xr10,   xr1,    xr5
    xvsubwod.w.h  xr11,   xr1,    xr5
    xvsubwev.w.h  xr12,   xr2,    xr6
    xvsubwod.w.h  xr13,   xr2,    xr6
    xvsubwev.w.h  xr14,   xr3,    xr7
    xvsubwod.w.h  xr15,   xr3,    xr7
    xvmul.w       xr4,    xr8,    xr8
    xvmul.w       xr5,    xr9,    xr9
    xvmadd.w      xr4,    xr10,   xr10
    xvmadd.w      xr5,    xr11,   xr11
    xvmadd.w      xr4,    xr12,   xr12
    xvmadd.w      xr5,    xr13,   xr13
    xvmadd.w      xr4,    xr14,   xr14
    xvmadd.w      xr5,    xr15,   xr15
    xvadd.w       \out,   xr4,    xr5
.endm

function x265_pixel_sse_ss_64x64_lasx
    slli.d        a1,     a1,     1
    slli.d        a3,     a3,     1

    pixel_sse_ss_64x1_core_lasx xr20

.rept 63
    add.d         a0,     a1,     a0
    add.d         a2,     a3,     a2
    pixel_sse_ss_64x1_core_lasx xr7
    xvadd.w       xr20,   xr7,    xr20
.endr

    xvpermi.q     xr6,    xr20,   0x01
    vadd.w        vr5,    vr20,   vr6
    vhaddw.d.w    vr5,    vr5,    vr5
    vhaddw.q.d    vr5,    vr5,    vr5
    vpickve2gr.w  a0,     vr5,    0
endfunc
