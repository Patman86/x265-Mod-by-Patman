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

#define FENC_STRIDE 64

.macro SUM_VEC_x4 in0, in1, in2, in3
    vhaddw.wu.hu   \in0,   \in0,  \in0
    vhaddw.wu.hu   \in1,   \in1,  \in1
    vhaddw.wu.hu   \in2,   \in2,  \in2
    vhaddw.wu.hu   \in3,   \in3,  \in3
    vhaddw.du.wu   \in0,   \in0,  \in0
    vhaddw.du.wu   \in1,   \in1,  \in1
    vhaddw.du.wu   \in2,   \in2,  \in2
    vhaddw.du.wu   \in3,   \in3,  \in3
    vhaddw.qu.du   \in0,   \in0,  \in0
    vhaddw.qu.du   \in1,   \in1,  \in1
    vhaddw.qu.du   \in2,   \in2,  \in2
    vhaddw.qu.du   \in3,   \in3,  \in3
.endm

.macro pixel_sad_x4_4x4_core_lsx out0, out1, out2, out3
    fld.s          f0,     a0,    0
    fld.s          f1,     a0,    FENC_STRIDE
    fld.s          f2,     a0,    FENC_STRIDE<<1
    fld.s          f3,     a0,    FENC_STRIDE*3
    FLDS_LOADX_4   a1,     a5,    t1,  t2,  f4, f8,  f12, f16
    FLDS_LOADX_4   a2,     a5,    t1,  t2,  f5, f9,  f13, f17
    FLDS_LOADX_4   a3,     a5,    t1,  t2,  f6, f10, f14, f18
    FLDS_LOADX_4   a4,     a5,    t1,  t2,  f7, f11, f15, f19

    vilvl.w        vr0,    vr1,   vr0
    vilvl.w        vr2,    vr3,   vr2
    vilvl.d        vr0,    vr2,   vr0
    vilvl.w        vr4,    vr8,   vr4
    vilvl.w        vr12,   vr16,  vr12
    vilvl.d        vr1,    vr12,  vr4
    vilvl.w        vr5,    vr9,   vr5
    vilvl.w        vr13,   vr17,  vr13
    vilvl.d        vr2,    vr13,  vr5
    vilvl.w        vr6,    vr10,  vr6
    vilvl.w        vr14,   vr18,  vr14
    vilvl.d        vr3,    vr14,  vr6
    vilvl.w        vr7,    vr11,  vr7
    vilvl.w        vr15,   vr19,  vr15
    vilvl.d        vr4,    vr15,  vr7

    vabsd.bu       vr1,    vr0,   vr1
    vabsd.bu       vr2,    vr0,   vr2
    vabsd.bu       vr3,    vr0,   vr3
    vabsd.bu       vr4,    vr0,   vr4
    vhaddw.hu.bu   \out0,  vr1,   vr1
    vhaddw.hu.bu   \out1,  vr2,   vr2
    vhaddw.hu.bu   \out2,  vr3,   vr3
    vhaddw.hu.bu   \out3,  vr4,   vr4
.endm

function x265_pixel_sad_x4_4x4_lsx
    slli.d         t1,     a5,    1
    add.d          t2,     a5,    t1

    pixel_sad_x4_4x4_core_lsx vr20, vr21, vr22, vr23

    SUM_VEC_x4 vr20, vr21, vr22, vr23

    // Store data to p_sad_array
    vstelm.w       vr20,   a6,    0,  0
    vstelm.w       vr21,   a6,    4,  0
    vstelm.w       vr22,   a6,    8,  0
    vstelm.w       vr23,   a6,   12,  0
endfunc

function x265_pixel_sad_x4_4x8_lsx
    slli.d         t1,     a5,    1
    add.d          t2,     a5,    t1

    pixel_sad_x4_4x4_core_lsx vr20, vr21, vr22, vr23

    alsl.d         a1,     a5,    a1,   2
    alsl.d         a2,     a5,    a2,   2
    alsl.d         a3,     a5,    a3,   2
    alsl.d         a4,     a5,    a4,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x4_4x4_core_lsx vr1, vr2, vr3, vr4

    vadd.h         vr16,   vr20,  vr1
    vadd.h         vr17,   vr21,  vr2
    vadd.h         vr18,   vr22,  vr3
    vadd.h         vr19,   vr23,  vr4

    SUM_VEC_x4 vr16, vr17, vr18, vr19

    // Store data to p_sad_array
    vstelm.w       vr16,   a6,    0,      0
    vstelm.w       vr17,   a6,    4,      0
    vstelm.w       vr18,   a6,    8,      0
    vstelm.w       vr19,   a6,    12,     0
endfunc

function x265_pixel_sad_x4_4x16_lsx
    slli.d         t1,     a5,    1
    add.d          t2,     a5,    t1

    pixel_sad_x4_4x4_core_lsx vr20, vr21, vr22, vr23

.rept 3
    alsl.d         a1,     a5,    a1,   2
    alsl.d         a2,     a5,    a2,   2
    alsl.d         a3,     a5,    a3,   2
    alsl.d         a4,     a5,    a4,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x4_4x4_core_lsx vr1, vr2, vr3, vr4

    vadd.h         vr20,   vr20,  vr1
    vadd.h         vr21,   vr21,  vr2
    vadd.h         vr22,   vr22,  vr3
    vadd.h         vr23,   vr23,  vr4
.endr

    SUM_VEC_x4 vr20, vr21, vr22, vr23

    // Store data to p_sad_array
    vstelm.w       vr20,   a6,    0,      0
    vstelm.w       vr21,   a6,    4,      0
    vstelm.w       vr22,   a6,    8,      0
    vstelm.w       vr23,   a6,    12,     0
endfunc

.macro pixel_sad_x4_8x4_core_lsx in0, in1, in2, in3, in4, out0, out1, out2, out3
    fld.d          f0,     \in0,  0
    fld.d          f1,     \in0,  FENC_STRIDE
    fld.d          f2,     \in0,  FENC_STRIDE<<1
    fld.d          f3,     \in0,  FENC_STRIDE*3
    FLDD_LOADX_4   \in1,   a5,    t1,  t2,  f4, f8,  f12, f16
    FLDD_LOADX_4   \in2,   a5,    t1,  t2,  f5, f9,  f13, f17
    FLDD_LOADX_4   \in3,   a5,    t1,  t2,  f6, f10, f14, f18
    FLDD_LOADX_4   \in4,   a5,    t1,  t2,  f7, f11, f15, f19
    vilvl.d        vr0,    vr1,   vr0
    vilvl.d        vr2,    vr3,   vr2
    vilvl.d        vr4,    vr8,   vr4
    vilvl.d        vr12,   vr16,  vr12
    vilvl.d        vr5,    vr9,   vr5
    vilvl.d        vr13,   vr17,  vr13
    vilvl.d        vr6,    vr10,  vr6
    vilvl.d        vr14,   vr18,  vr14
    vilvl.d        vr7,    vr11,  vr7
    vilvl.d        vr15,   vr19,  vr15
    vabsd.bu       vr4,    vr0,   vr4
    vabsd.bu       vr5,    vr0,   vr5
    vabsd.bu       vr6,    vr0,   vr6
    vabsd.bu       vr7,    vr0,   vr7
    vabsd.bu       vr12,   vr2,   vr12
    vabsd.bu       vr13,   vr2,   vr13
    vabsd.bu       vr14,   vr2,   vr14
    vabsd.bu       vr15,   vr2,   vr15
    vhaddw.hu.bu   vr4,    vr4,   vr4
    vhaddw.hu.bu   vr5,    vr5,   vr5
    vhaddw.hu.bu   vr6,    vr6,   vr6
    vhaddw.hu.bu   vr7,    vr7,   vr7
    vhaddw.hu.bu   vr12,   vr12,  vr12
    vhaddw.hu.bu   vr13,   vr13,  vr13
    vhaddw.hu.bu   vr14,   vr14,  vr14
    vhaddw.hu.bu   vr15,   vr15,  vr15
    vadd.h         \out0,  vr4,   vr12
    vadd.h         \out1,  vr5,   vr13
    vadd.h         \out2,  vr6,   vr14
    vadd.h         \out3,  vr7,   vr15
.endm

function x265_pixel_sad_x4_8x4_lsx
    slli.d         t1,     a5,    1
    add.d          t2,     a5,    t1

    pixel_sad_x4_8x4_core_lsx a0, a1, a2, a3, a4, vr16, vr17, vr18, vr19

    SUM_VEC_x4 vr16, vr17, vr18, vr19

    vstelm.w       vr16,   a6,    0,      0
    vstelm.w       vr17,   a6,    4,      0
    vstelm.w       vr18,   a6,    8,      0
    vstelm.w       vr19,   a6,    12,     0
endfunc

.macro pixel_sad_x4_8x_lsx h
function x265_pixel_sad_x4_8x\h\()_lsx
    slli.d         t1,     a5,    1
    add.d          t2,     a5,    t1

    pixel_sad_x4_8x4_core_lsx a0, a1, a2, a3, a4, vr20, vr21, vr22, vr23

.rept (\h>>2)-1
    alsl.d         a1,     a5,    a1,   2
    alsl.d         a2,     a5,    a2,   2
    alsl.d         a3,     a5,    a3,   2
    alsl.d         a4,     a5,    a4,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x4_8x4_core_lsx a0, a1, a2, a3, a4, vr16, vr17, vr18, vr19

    vadd.h         vr20,   vr16,  vr20
    vadd.h         vr21,   vr17,  vr21
    vadd.h         vr22,   vr18,  vr22
    vadd.h         vr23,   vr19,  vr23
.endr

    SUM_VEC_x4 vr20, vr21, vr22, vr23

    vstelm.w       vr20,   a6,    0,    0
    vstelm.w       vr21,   a6,    4,    0
    vstelm.w       vr22,   a6,    8,    0
    vstelm.w       vr23,   a6,    12,   0
endfunc
.endm

pixel_sad_x4_8x_lsx 8
pixel_sad_x4_8x_lsx 16
pixel_sad_x4_8x_lsx 32

.macro pixel_sad_x4_16x4_core_lsx in0, in1, in2, in3, in4, out0, out1, out2, out3
    vld            vr0,    \in0,  0
    vld            vr1,    \in0,  FENC_STRIDE
    vld            vr2,    \in0,  FENC_STRIDE<<1
    vld            vr3,    \in0,  FENC_STRIDE*3
    LSX_LOADX_4    \in1,   a5,    t1,  t2,  vr4, vr8, vr12, vr16
    LSX_LOADX_4    \in2,   a5,    t1,  t2,  vr5, vr9, vr13, vr17
    LSX_LOADX_4    \in3,   a5,    t1,  t2,  vr6, vr10, vr14, vr18
    LSX_LOADX_4    \in4,   a5,    t1,  t2,  vr7, vr11, vr15, vr19
    vabsd.bu       vr4,    vr0,   vr4
    vabsd.bu       vr5,    vr0,   vr5
    vabsd.bu       vr6,    vr0,   vr6
    vabsd.bu       vr7,    vr0,   vr7
    vabsd.bu       vr8,    vr1,   vr8
    vabsd.bu       vr9,    vr1,   vr9
    vabsd.bu       vr10,   vr1,   vr10
    vabsd.bu       vr11,   vr1,   vr11
    vabsd.bu       vr12,   vr2,   vr12
    vabsd.bu       vr13,   vr2,   vr13
    vabsd.bu       vr14,   vr2,   vr14
    vabsd.bu       vr15,   vr2,   vr15
    vabsd.bu       vr16,   vr3,   vr16
    vabsd.bu       vr17,   vr3,   vr17
    vabsd.bu       vr18,   vr3,   vr18
    vabsd.bu       vr19,   vr3,   vr19
    vhaddw.hu.bu   vr4,    vr4,   vr4
    vhaddw.hu.bu   vr5,    vr5,   vr5
    vhaddw.hu.bu   vr6,    vr6,   vr6
    vhaddw.hu.bu   vr7,    vr7,   vr7
    vhaddw.hu.bu   vr8,    vr8,   vr8
    vhaddw.hu.bu   vr9,    vr9,   vr9
    vhaddw.hu.bu   vr10,   vr10,  vr10
    vhaddw.hu.bu   vr11,   vr11,  vr11
    vhaddw.hu.bu   vr12,   vr12,  vr12
    vhaddw.hu.bu   vr13,   vr13,  vr13
    vhaddw.hu.bu   vr14,   vr14,  vr14
    vhaddw.hu.bu   vr15,   vr15,  vr15
    vhaddw.hu.bu   vr16,   vr16,  vr16
    vhaddw.hu.bu   vr17,   vr17,  vr17
    vhaddw.hu.bu   vr18,   vr18,  vr18
    vhaddw.hu.bu   vr19,   vr19,  vr19
    vadd.h         vr0,    vr4,   vr8
    vadd.h         vr1,    vr12,  vr16
    vadd.h         \out0,  vr0,   vr1
    vadd.h         vr0,    vr5,   vr9
    vadd.h         vr1,    vr13,  vr17
    vadd.h         \out1,  vr0,   vr1
    vadd.h         vr0,    vr6,   vr10
    vadd.h         vr1,    vr14,  vr18
    vadd.h         \out2,  vr0,   vr1
    vadd.h         vr0,    vr7,   vr11
    vadd.h         vr1,    vr15,  vr19
    vadd.h         \out3,  vr0,   vr1
.endm

function x265_pixel_sad_x4_12x16_lsx
    slli.d         t1,     a5,    1
    add.d          t2,     a5,    t1

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr20, vr21, vr22, vr23

.rept 3
    alsl.d         a1,     a5,    a1,   2
    alsl.d         a2,     a5,    a2,   2
    alsl.d         a3,     a5,    a3,   2
    alsl.d         a4,     a5,    a4,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr16,vr17, vr18, vr19

    vadd.h         vr20,   vr16,  vr20
    vadd.h         vr21,   vr17,  vr21
    vadd.h         vr22,   vr18,  vr22
    vadd.h         vr23,   vr19,  vr23
.endr

    vinsgr2vr.w    vr20,   zero,  3
    vinsgr2vr.w    vr21,   zero,  3
    vinsgr2vr.w    vr22,   zero,  3
    vinsgr2vr.w    vr23,   zero,  3

    SUM_VEC_x4 vr20, vr21, vr22, vr23

    vstelm.w       vr20,   a6,    0,    0
    vstelm.w       vr21,   a6,    4,    0
    vstelm.w       vr22,   a6,    8,    0
    vstelm.w       vr23,   a6,    12,   0
endfunc

function x265_pixel_sad_x4_16x4_lsx
    slli.d         t1,     a5,    1
    add.d          t2,     a5,    t1

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr20, vr21, vr22, vr23

    SUM_VEC_x4 vr20, vr21, vr22, vr23

    vstelm.w       vr20,   a6,    0,    0
    vstelm.w       vr21,   a6,    4,    0
    vstelm.w       vr22,   a6,    8,    0
    vstelm.w       vr23,   a6,    12,   0
endfunc

.macro pixel_sad_x4_16x_lsx h
function x265_pixel_sad_x4_16x\h\()_lsx
    slli.d         t1,     a5,    1
    add.d          t2,     a5,    t1

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr20, vr21, vr22, vr23

.rept (\h>>2)-1
    alsl.d         a1,     a5,    a1,   2
    alsl.d         a2,     a5,    a2,   2
    alsl.d         a3,     a5,    a3,   2
    alsl.d         a4,     a5,    a4,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr16,vr17, vr18, vr19

    vadd.h         vr20,   vr16,  vr20
    vadd.h         vr21,   vr17,  vr21
    vadd.h         vr22,   vr18,  vr22
    vadd.h         vr23,   vr19,  vr23
.endr

    SUM_VEC_x4 vr20, vr21, vr22, vr23

    vstelm.w       vr20,   a6,    0,    0
    vstelm.w       vr21,   a6,    4,    0
    vstelm.w       vr22,   a6,    8,    0
    vstelm.w       vr23,   a6,    12,   0
endfunc
.endm

pixel_sad_x4_16x_lsx 8
pixel_sad_x4_16x_lsx 12
pixel_sad_x4_16x_lsx 16
pixel_sad_x4_16x_lsx 32
pixel_sad_x4_16x_lsx 64

function x265_pixel_sad_x4_24x32_lsx
    slli.d         t1,     a5,    1
    add.d          t2,     a5,    t1

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr20, vr21, vr22, vr23

    addi.d         t3,     a1,    16
    addi.d         t4,     a2,    16
    addi.d         t5,     a3,    16
    addi.d         t6,     a4,    16
    addi.d         t7,     a0,    16

    pixel_sad_x4_8x4_core_lsx t7, t3, t4, t5, t6, vr16,vr17, vr18, vr19

    vadd.h         vr20,   vr16,  vr20
    vadd.h         vr21,   vr17,  vr21
    vadd.h         vr22,   vr18,  vr22
    vadd.h         vr23,   vr19,  vr23

.rept 7

    alsl.d         a1,     a5,    a1,   2
    alsl.d         a2,     a5,    a2,   2
    alsl.d         a3,     a5,    a3,   2
    alsl.d         a4,     a5,    a4,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr16,vr17, vr18, vr19

    vadd.h         vr20,   vr16,  vr20
    vadd.h         vr21,   vr17,  vr21
    vadd.h         vr22,   vr18,  vr22
    vadd.h         vr23,   vr19,  vr23

    alsl.d         t3,     a5,    t3,   2
    alsl.d         t4,     a5,    t4,   2
    alsl.d         t5,     a5,    t5,   2
    alsl.d         t6,     a5,    t6,   2
    addi.d         t7,     t7,    FENC_STRIDE<<2
    pixel_sad_x4_8x4_core_lsx t7, t3, t4, t5, t6, vr16,vr17, vr18, vr19

    vadd.h         vr20,   vr16,  vr20
    vadd.h         vr21,   vr17,  vr21
    vadd.h         vr22,   vr18,  vr22
    vadd.h         vr23,   vr19,  vr23

.endr

    SUM_VEC_x4 vr20, vr21, vr22, vr23

    vstelm.w       vr20,   a6,    0,    0
    vstelm.w       vr21,   a6,    4,    0
    vstelm.w       vr22,   a6,    8,    0
    vstelm.w       vr23,   a6,    12,   0
endfunc

.macro pixel_sad_x4_32x4_core_lsx
    alsl.d         a1,     a5,    a1,   2
    alsl.d         a2,     a5,    a2,   2
    alsl.d         a3,     a5,    a3,   2
    alsl.d         a4,     a5,    a4,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2
    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr16, vr17, vr18, vr19
    vadd.h         vr20,   vr16,  vr20
    vadd.h         vr21,   vr17,  vr21
    vadd.h         vr22,   vr18,  vr22
    vadd.h         vr23,   vr19,  vr23

    alsl.d         t3,     a5,    t3,   2
    alsl.d         t4,     a5,    t4,   2
    alsl.d         t5,     a5,    t5,   2
    alsl.d         t6,     a5,    t6,   2
    addi.d         t7,     t7,    FENC_STRIDE<<2
    pixel_sad_x4_16x4_core_lsx t7, t3, t4, t5, t6, vr16, vr17, vr18, vr19
    vadd.h         vr20,   vr16,  vr20
    vadd.h         vr21,   vr17,  vr21
    vadd.h         vr22,   vr18,  vr22
    vadd.h         vr23,   vr19,  vr23
.endm

.macro pixel_sad_x4_32x_lsx h
function x265_pixel_sad_x4_32x\h\()_lsx
    slli.d         t1,     a5,    1
    add.d          t2,     a5,    t1

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr20, vr21, vr22, vr23

    addi.d         t3,     a1,    16
    addi.d         t4,     a2,    16
    addi.d         t5,     a3,    16
    addi.d         t6,     a4,    16
    addi.d         t7,     a0,    16

    pixel_sad_x4_16x4_core_lsx t7, t3, t4, t5, t6, vr16,vr17, vr18, vr19

    vadd.h         vr20,   vr16,  vr20
    vadd.h         vr21,   vr17,  vr21
    vadd.h         vr22,   vr18,  vr22
    vadd.h         vr23,   vr19,  vr23

.rept (\h>>2)-1

    pixel_sad_x4_32x4_core_lsx

.endr

    SUM_VEC_x4 vr20, vr21, vr22, vr23

    vstelm.w       vr20,   a6,    0,    0
    vstelm.w       vr21,   a6,    4,    0
    vstelm.w       vr22,   a6,    8,    0
    vstelm.w       vr23,   a6,    12,   0
endfunc
.endm

pixel_sad_x4_32x_lsx 8
pixel_sad_x4_32x_lsx 16
pixel_sad_x4_32x_lsx 24
pixel_sad_x4_32x_lsx 32
pixel_sad_x4_32x_lsx 64

.macro pixel_sad_x4_64x4_core_lsx
    addi.d         a1,     a1,    -48
    addi.d         a2,     a2,    -48
    addi.d         a3,     a3,    -48
    addi.d         a4,     a4,    -48
    addi.d         a0,     a0,    -48

    alsl.d         a1,     a5,    a1,   2
    alsl.d         a2,     a5,    a2,   2
    alsl.d         a3,     a5,    a3,   2
    alsl.d         a4,     a5,    a4,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr16,vr17, vr18, vr19

    vhaddw.wu.hu   vr16,   vr16,  vr16
    vhaddw.wu.hu   vr17,   vr17,  vr17
    vhaddw.wu.hu   vr18,   vr18,  vr18
    vhaddw.wu.hu   vr19,   vr19,  vr19

    vadd.w         vr20,   vr16,  vr20
    vadd.w         vr21,   vr17,  vr21
    vadd.w         vr22,   vr18,  vr22
    vadd.w         vr23,   vr19,  vr23
.rept 3
    addi.d         a1,     a1,    16
    addi.d         a2,     a2,    16
    addi.d         a3,     a3,    16
    addi.d         a4,     a4,    16
    addi.d         a0,     a0,    16

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr16,vr17, vr18, vr19

    vhaddw.wu.hu   vr16,   vr16,  vr16
    vhaddw.wu.hu   vr17,   vr17,  vr17
    vhaddw.wu.hu   vr18,   vr18,  vr18
    vhaddw.wu.hu   vr19,   vr19,  vr19

    vadd.w         vr20,   vr16,  vr20
    vadd.w         vr21,   vr17,  vr21
    vadd.w         vr22,   vr18,  vr22
    vadd.w         vr23,   vr19,  vr23
.endr
.endm

.macro pixel_sad_x4_64x_lsx h
function x265_pixel_sad_x4_64x\h\()_lsx
    slli.d         t1,     a5,    1
    add.d          t2,     a5,    t1

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr20, vr21, vr22, vr23

    vhaddw.wu.hu   vr20,   vr20,  vr20
    vhaddw.wu.hu   vr21,   vr21,  vr21
    vhaddw.wu.hu   vr22,   vr22,  vr22
    vhaddw.wu.hu   vr23,   vr23,  vr23

.rept 3
    addi.d         a1,     a1,    16
    addi.d         a2,     a2,    16
    addi.d         a3,     a3,    16
    addi.d         a4,     a4,    16
    addi.d         a0,     a0,    16

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr16,vr17, vr18, vr19

    vhaddw.wu.hu   vr16,   vr16,  vr16
    vhaddw.wu.hu   vr17,   vr17,  vr17
    vhaddw.wu.hu   vr18,   vr18,  vr18
    vhaddw.wu.hu   vr19,   vr19,  vr19

    vadd.w         vr20,   vr16,  vr20
    vadd.w         vr21,   vr17,  vr21
    vadd.w         vr22,   vr18,  vr22
    vadd.w         vr23,   vr19,  vr23
.endr

.rept (\h>>2)-1
    pixel_sad_x4_64x4_core_lsx
.endr

    vhaddw.du.wu   vr20,   vr20,  vr20
    vhaddw.du.wu   vr21,   vr21,  vr21
    vhaddw.du.wu   vr22,   vr22,  vr22
    vhaddw.du.wu   vr23,   vr23,  vr23
    vhaddw.qu.du   vr20,   vr20,  vr20
    vhaddw.qu.du   vr21,   vr21,  vr21
    vhaddw.qu.du   vr22,   vr22,  vr22
    vhaddw.qu.du   vr23,   vr23,  vr23

    vstelm.w       vr20,   a6,    0,    0
    vstelm.w       vr21,   a6,    4,    0
    vstelm.w       vr22,   a6,    8,    0
    vstelm.w       vr23,   a6,    12,   0
endfunc
.endm

pixel_sad_x4_64x_lsx 16
pixel_sad_x4_64x_lsx 32
pixel_sad_x4_64x_lsx 48
pixel_sad_x4_64x_lsx 64

function x265_pixel_sad_x4_48x64_lsx
    slli.d         t1,     a5,    1
    add.d          t2,     a5,    t1

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr20, vr21, vr22, vr23

    vhaddw.wu.hu   vr20,   vr20,  vr20
    vhaddw.wu.hu   vr21,   vr21,  vr21
    vhaddw.wu.hu   vr22,   vr22,  vr22
    vhaddw.wu.hu   vr23,   vr23,  vr23

.rept 2
    addi.d         a1,     a1,    16
    addi.d         a2,     a2,    16
    addi.d         a3,     a3,    16
    addi.d         a4,     a4,    16
    addi.d         a0,     a0,    16

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr16,vr17, vr18, vr19

    vhaddw.wu.hu   vr16,   vr16,  vr16
    vhaddw.wu.hu   vr17,   vr17,  vr17
    vhaddw.wu.hu   vr18,   vr18,  vr18
    vhaddw.wu.hu   vr19,   vr19,  vr19

    vadd.w         vr20,   vr16,  vr20
    vadd.w         vr21,   vr17,  vr21
    vadd.w         vr22,   vr18,  vr22
    vadd.w         vr23,   vr19,  vr23
.endr

.rept 15
    addi.d         a1,     a1,    -32
    addi.d         a2,     a2,    -32
    addi.d         a3,     a3,    -32
    addi.d         a4,     a4,    -32
    addi.d         a0,     a0,    -32

    alsl.d         a1,     a5,    a1,   2
    alsl.d         a2,     a5,    a2,   2
    alsl.d         a3,     a5,    a3,   2
    alsl.d         a4,     a5,    a4,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr16,vr17, vr18, vr19

    vhaddw.wu.hu   vr16,   vr16,  vr16
    vhaddw.wu.hu   vr17,   vr17,  vr17
    vhaddw.wu.hu   vr18,   vr18,  vr18
    vhaddw.wu.hu   vr19,   vr19,  vr19

    vadd.w         vr20,   vr16,  vr20
    vadd.w         vr21,   vr17,  vr21
    vadd.w         vr22,   vr18,  vr22
    vadd.w         vr23,   vr19,  vr23
.rept 2
    addi.d         a1,     a1,    16
    addi.d         a2,     a2,    16
    addi.d         a3,     a3,    16
    addi.d         a4,     a4,    16
    addi.d         a0,     a0,    16

    pixel_sad_x4_16x4_core_lsx a0, a1, a2, a3, a4, vr16,vr17, vr18, vr19

    vhaddw.wu.hu   vr16,   vr16,  vr16
    vhaddw.wu.hu   vr17,   vr17,  vr17
    vhaddw.wu.hu   vr18,   vr18,  vr18
    vhaddw.wu.hu   vr19,   vr19,  vr19

    vadd.w         vr20,   vr16,  vr20
    vadd.w         vr21,   vr17,  vr21
    vadd.w         vr22,   vr18,  vr22
    vadd.w         vr23,   vr19,  vr23
.endr
.endr

    vhaddw.du.wu   vr20,   vr20,  vr20
    vhaddw.du.wu   vr21,   vr21,  vr21
    vhaddw.du.wu   vr22,   vr22,  vr22
    vhaddw.du.wu   vr23,   vr23,  vr23
    vhaddw.qu.du   vr20,   vr20,  vr20
    vhaddw.qu.du   vr21,   vr21,  vr21
    vhaddw.qu.du   vr22,   vr22,  vr22
    vhaddw.qu.du   vr23,   vr23,  vr23

    vstelm.w       vr20,   a6,    0,    0
    vstelm.w       vr21,   a6,    4,    0
    vstelm.w       vr22,   a6,    8,    0
    vstelm.w       vr23,   a6,    12,   0
endfunc

.macro pixel_sad_x4_8x4_core_lasx out0, out1, out2, out3
    // Load data from p_src, p_ref0, p_ref1, p_ref2 and p_ref3
    FLDD_LOADX_4   a1,     a5,    t1,  t2,  f4, f8,  f14, f18
    FLDD_LOADX_4   a2,     a5,    t1,  t2,  f5, f9,  f15, f19
    FLDD_LOADX_4   a3,     a5,    t1,  t2,  f6, f10, f16, f20
    FLDD_LOADX_4   a4,     a5,    t1,  t2,  f7, f11, f17, f21
    vilvl.d        vr4,    vr5,   vr4
    vilvl.d        vr6,    vr7,   vr6
    vilvl.d        vr8,    vr9,   vr8
    vilvl.d        vr10,   vr11,  vr10
    vilvl.d        vr14,   vr15,  vr14
    vilvl.d        vr16,   vr17,  vr16
    vilvl.d        vr18,   vr19,  vr18
    vilvl.d        vr20,   vr21,  vr20
    xvpermi.q      xr4,    xr6,   0x02
    xvpermi.q      xr8,    xr10,  0x02
    xvpermi.q      xr14,   xr16,  0x02
    xvpermi.q      xr18,   xr20,  0x02
    // Calculate the absolute value of the difference
    xvldrepl.d     xr7,    a0,    0
    xvldrepl.d     xr11,   a0,    FENC_STRIDE
    xvldrepl.d     xr17,   a0,    FENC_STRIDE<<1
    xvldrepl.d     xr21,   a0,    FENC_STRIDE*3
    xvabsd.bu      xr5,    xr7,   xr4
    xvabsd.bu      xr9,    xr11,  xr8
    xvabsd.bu      xr10,   xr17,  xr14
    xvabsd.bu      xr11,   xr21,  xr18
    xvaddwev.h.bu  \out0,  xr5,   xr9
    xvaddwod.h.bu  \out1,  xr5,   xr9
    xvaddwev.h.bu  \out2,  xr10,  xr11
    xvaddwod.h.bu  \out3,  xr10,  xr11
.endm

function x265_pixel_sad_x4_8x4_lasx
    slli.d         t1,     a5,    1
    add.d          t2,     t1,    a5
    slli.d         t3,     a5,    2

    pixel_sad_x4_8x4_core_lasx xr0, xr1, xr2, xr3

    xvadd.h        xr5,    xr0,   xr1
    xvadd.h        xr10,   xr2,   xr3
    xvadd.h        xr10,   xr10,  xr5
    xvhaddw.wu.hu  xr10,   xr10,  xr10
    xvhaddw.du.wu  xr10,   xr10,  xr10
    xvpermi.q      xr5,    xr10,  0x01
    xvpickev.w     xr10,   xr5,   xr10

    vst            vr10,   a6,    0
endfunc

.macro pixel_sad_x4_8x_lasx h
function x265_pixel_sad_x4_8x\h\()_lasx
    slli.d         t1,     a5,    1
    add.d          t2,     t1,    a5
    slli.d         t3,     a5,    2

    pixel_sad_x4_8x4_core_lasx xr0, xr1, xr2, xr3

.rept (\h>>2)-1
    add.d          a1,     a1,    t3
    add.d          a2,     a2,    t3
    add.d          a3,     a3,    t3
    add.d          a4,     a4,    t3
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x4_8x4_core_lasx xr12, xr13, xr14, xr15

    xvadd.h        xr0,    xr0,   xr12
    xvadd.h        xr1,    xr1,   xr13
    xvadd.h        xr2,    xr2,   xr14
    xvadd.h        xr3,    xr3,   xr15
.endr

    xvadd.h        xr5,    xr0,   xr1
    xvadd.h        xr10,   xr2,   xr3
    xvadd.h        xr10,   xr10,  xr5
    xvhaddw.wu.hu  xr10,   xr10,  xr10
    xvhaddw.du.wu  xr10,   xr10,  xr10
    xvpermi.q      xr5,    xr10,  0x01
    xvpickev.w     xr10,   xr5,   xr10

    vst            vr10,   a6,    0
endfunc
.endm

pixel_sad_x4_8x_lasx 8
pixel_sad_x4_8x_lasx 16
pixel_sad_x4_8x_lasx 32

.macro pixel_sad_x4_16x2_core_lasx out0, out1, out2, out3
    vld            vr2,    a0,    0
    vld            vr3,    a0,    FENC_STRIDE
    vld            vr4,    a1,    0
    vldx           vr8,    a1,    a5
    vld            vr5,    a2,    0
    vldx           vr9,    a2,    a5
    vld            vr6,    a3,    0
    vldx           vr10,   a3,    a5
    vld            vr7,    a4,    0
    vldx           vr11,   a4,    a5
    xvpermi.q      xr2,    xr3,   0x02
    xvpermi.q      xr4,    xr8,   0x02
    xvpermi.q      xr5,    xr9,   0x02
    xvpermi.q      xr6,    xr10,  0x02
    xvpermi.q      xr7,    xr11,  0x02
    // Calculate the absolute value of the difference
    xvabsd.bu      xr8,    xr2,   xr4
    xvabsd.bu      xr9,    xr2,   xr5
    xvabsd.bu      xr10,   xr2,   xr6
    xvabsd.bu      xr11,   xr2,   xr7
    xvhaddw.hu.bu  \out0,  xr8,   xr8
    xvhaddw.hu.bu  \out1,  xr9,   xr9
    xvhaddw.hu.bu  \out2,  xr10,  xr10
    xvhaddw.hu.bu  \out3,  xr11,  xr11
.endm

.macro pixel_sad_x4_16x2_core1_lasx
    xvori.b        xr17,   xr12,  0
    xvori.b        xr18,   xr13,  0
    xvpermi.q      xr12,   xr14,  0x02
    xvpermi.q      xr14,   xr17,  0x31
    xvpermi.q      xr13,   xr15,  0x02
    xvpermi.q      xr15,   xr18,  0x31
    xvadd.h        xr12,   xr12,  xr14
    xvadd.h        xr13,   xr13,  xr15
    xvhaddw.w.h    xr12,   xr12,  xr12
    xvhaddw.w.h    xr13,   xr13,  xr13
    xvhaddw.d.w    xr12,   xr12,  xr12
    xvhaddw.d.w    xr13,   xr13,  xr13
    xvhaddw.q.d    xr12,   xr12,  xr12
    xvhaddw.q.d    xr13,   xr13,  xr13
    xvpackev.w     xr13,   xr13,  xr12
.endm

function x265_pixel_sad_x4_16x4_lasx
    slli.d         t1,     a5,    1

    pixel_sad_x4_16x2_core_lasx xr12, xr13, xr14, xr15

    addi.d         a0,     a0,    FENC_STRIDE<<1
    add.d          a1,     a1,    t1
    add.d          a2,     a2,    t1
    add.d          a3,     a3,    t1
    add.d          a4,     a4,    t1

    pixel_sad_x4_16x2_core_lasx xr8, xr9, xr10, xr11

    xvadd.h        xr12,   xr12,  xr8
    xvadd.h        xr13,   xr13,  xr9
    xvadd.h        xr14,   xr14,  xr10
    xvadd.h        xr15,   xr15,  xr11

    pixel_sad_x4_16x2_core1_lasx

    // Store data to p_sad_array
    xvstelm.d      xr13,   a6,    0,    0
    xvstelm.d      xr13,   a6,    8,    2
endfunc

.macro pixel_sad_x4_16x_lasx h
function x265_pixel_sad_x4_16x\h\()_lasx
    slli.d         t1,     a5,    1

    pixel_sad_x4_16x2_core_lasx xr12, xr13, xr14, xr15

.rept (\h>>1)-1
    addi.d         a0,     a0,    FENC_STRIDE<<1
    add.d          a1,     a1,    t1
    add.d          a2,     a2,    t1
    add.d          a3,     a3,    t1
    add.d          a4,     a4,    t1

    pixel_sad_x4_16x2_core_lasx xr8, xr9, xr10, xr11

    xvadd.h        xr12,   xr12,  xr8
    xvadd.h        xr13,   xr13,  xr9
    xvadd.h        xr14,   xr14,  xr10
    xvadd.h        xr15,   xr15,  xr11
.endr

    pixel_sad_x4_16x2_core1_lasx

    xvstelm.d      xr13,   a6,    0,    0
    xvstelm.d      xr13,   a6,    8,    2
endfunc
.endm

pixel_sad_x4_16x_lasx 8
pixel_sad_x4_16x_lasx 12
pixel_sad_x4_16x_lasx 16
pixel_sad_x4_16x_lasx 32
pixel_sad_x4_16x_lasx 64

function x265_pixel_sad_x4_12x16_lasx
    slli.d         t1,     a5,    1

    pixel_sad_x4_16x2_core_lasx xr12, xr13, xr14, xr15

.rept 7
    addi.d         a0,     a0,    FENC_STRIDE<<1
    add.d          a1,     a1,    t1
    add.d          a2,     a2,    t1
    add.d          a3,     a3,    t1
    add.d          a4,     a4,    t1

    pixel_sad_x4_16x2_core_lasx xr8, xr9, xr10, xr11

    xvadd.h        xr12,   xr12,  xr8
    xvadd.h        xr13,   xr13,  xr9
    xvadd.h        xr14,   xr14,  xr10
    xvadd.h        xr15,   xr15,  xr11
.endr

.irp i, xr12, xr13, xr14, xr15
    xvinsgr2vr.w   \i,     zero,  3
    xvinsgr2vr.w   \i,     zero,  7
.endr

    pixel_sad_x4_16x2_core1_lasx

    xvstelm.d      xr13,   a6,    0,    0
    xvstelm.d      xr13,   a6,    8,    2
endfunc

.macro pixel_sad_x4_32x2_lasx out0, out1, out2, out3
    xvld           xr0,    a0,    0
    xvld           xr1,    a0,    FENC_STRIDE
    xvld           xr2,    a1,    0
    xvldx          xr3,    a1,    a5
    xvld           xr4,    a2,    0
    xvldx          xr5,    a2,    a5
    xvld           xr6,    a3,    0
    xvldx          xr7,    a3,    a5
    xvld           xr8,    a4,    0
    xvldx          xr9,    a4,    a5

    xvabsd.bu      xr10,   xr0,   xr2
    xvabsd.bu      xr11,   xr0,   xr4
    xvabsd.bu      xr12,   xr0,   xr6
    xvabsd.bu      xr13,   xr0,   xr8
    xvabsd.bu      xr2,    xr1,   xr3
    xvabsd.bu      xr4,    xr1,   xr5
    xvabsd.bu      xr6,    xr1,   xr7
    xvabsd.bu      xr8,    xr1,   xr9

.irp i, xr10, xr11, xr12, xr13, xr2, xr4, xr6, xr8
    xvhaddw.hu.bu  \i,     \i,    \i
.endr
   xvadd.h         \out0,  xr10,  xr2
   xvadd.h         \out1,  xr11,  xr4
   xvadd.h         \out2,  xr12,  xr6
   xvadd.h         \out3,  xr13,  xr8
.endm

.macro pixel_sad_x4_32x_lasx h
function x265_pixel_sad_x4_32x\h\()_lasx
    slli.d         t1,     a5,    1

    pixel_sad_x4_32x2_lasx xr14, xr15, xr16, xr17

.rept (\h>>1)-1
    addi.d         a0,     a0,    FENC_STRIDE<<1
    add.d          a1,     a1,    t1
    add.d          a2,     a2,    t1
    add.d          a3,     a3,    t1
    add.d          a4,     a4,    t1

    pixel_sad_x4_32x2_lasx xr10, xr11, xr12, xr13

    xvadd.h        xr16,   xr12,  xr16
    xvadd.h        xr17,   xr13,  xr17
    xvadd.h        xr14,   xr14,  xr10
    xvadd.h        xr15,   xr15,  xr11
.endr

    xvhaddw.w.h    xr14,   xr14,  xr14
    xvhaddw.w.h    xr15,   xr15,  xr15
    xvhaddw.w.h    xr16,   xr16,  xr16
    xvhaddw.w.h    xr17,   xr17,  xr17
    xvhaddw.d.w    xr14,   xr14,  xr14
    xvhaddw.d.w    xr15,   xr15,  xr15
    xvhaddw.d.w    xr16,   xr16,  xr16
    xvhaddw.d.w    xr17,   xr17,  xr17

    xvpickev.w     xr10,   xr15,  xr14
    xvpickev.w     xr11,   xr17,  xr16
    xvhaddw.d.w    xr10,   xr10,  xr10
    xvhaddw.d.w    xr11,   xr11,  xr11
    xvpickev.w     xr12,   xr11,  xr10
    xvpermi.q      xr13,   xr12,  0x01

    vadd.w         vr13,   vr13,  vr12
    vst            vr13,   a6,    0
endfunc
.endm

pixel_sad_x4_32x_lasx 8
pixel_sad_x4_32x_lasx 16
pixel_sad_x4_32x_lasx 24
pixel_sad_x4_32x_lasx 32
pixel_sad_x4_32x_lasx 64

function x265_pixel_sad_x4_24x32_lasx
    slli.d         t1,     a5,    1

    pixel_sad_x4_32x2_lasx xr14, xr15, xr16, xr17

.rept 15
    addi.d         a0,     a0,    FENC_STRIDE<<1
    add.d          a1,     a1,    t1
    add.d          a2,     a2,    t1
    add.d          a3,     a3,    t1
    add.d          a4,     a4,    t1

    pixel_sad_x4_32x2_lasx xr10, xr11, xr12, xr13

    xvadd.h        xr16,   xr12,  xr16
    xvadd.h        xr17,   xr13,  xr17
    xvadd.h        xr14,   xr14,  xr10
    xvadd.h        xr15,   xr15,  xr11
.endr

    xvinsgr2vr.d   xr14,   zero,  3
    xvinsgr2vr.d   xr15,   zero,  3
    xvinsgr2vr.d   xr16,   zero,  3
    xvinsgr2vr.d   xr17,   zero,  3

    xvhaddw.w.h    xr14,   xr14,  xr14
    xvhaddw.w.h    xr15,   xr15,  xr15
    xvhaddw.w.h    xr16,   xr16,  xr16
    xvhaddw.w.h    xr17,   xr17,  xr17
    xvhaddw.d.w    xr14,   xr14,  xr14
    xvhaddw.d.w    xr15,   xr15,  xr15
    xvhaddw.d.w    xr16,   xr16,  xr16
    xvhaddw.d.w    xr17,   xr17,  xr17
    xvpickev.w     xr10,   xr15,  xr14
    xvpickev.w     xr11,   xr17,  xr16
    xvhaddw.d.w    xr10,   xr10,  xr10
    xvhaddw.d.w    xr11,   xr11,  xr11
    xvpickev.w     xr12,   xr11,  xr10
    xvpermi.q      xr13,   xr12,  0x01

    vadd.w         vr13,   vr13,  vr12
    vst            vr13,   a6,    0
endfunc

.macro pixel_sad_x4_64x_lasx h
function x265_pixel_sad_x4_64x\h\()_lasx
    slli.d         t1,     a5,    1

    pixel_sad_x4_32x2_lasx xr14, xr15, xr16, xr17

    addi.d         a0,     a0,    32
    addi.d         a1,     a1,    32
    addi.d         a2,     a2,    32
    addi.d         a3,     a3,    32
    addi.d         a4,     a4,    32

    pixel_sad_x4_32x2_lasx xr10, xr11, xr12, xr13

    xvadd.h        xr16,   xr12,  xr16
    xvadd.h        xr17,   xr13,  xr17
    xvadd.h        xr14,   xr14,  xr10
    xvadd.h        xr15,   xr15,  xr11

.rept (\h>>1)-1
    addi.d         a0,     a0,    -32
    addi.d         a1,     a1,    -32
    addi.d         a2,     a2,    -32
    addi.d         a3,     a3,    -32
    addi.d         a4,     a4,    -32

    addi.d         a0,     a0,    FENC_STRIDE<<1
    add.d          a1,     a1,    t1
    add.d          a2,     a2,    t1
    add.d          a3,     a3,    t1
    add.d          a4,     a4,    t1

    pixel_sad_x4_32x2_lasx xr10, xr11, xr12, xr13

    xvadd.h        xr16,   xr12,  xr16
    xvadd.h        xr17,   xr13,  xr17
    xvadd.h        xr14,   xr14,  xr10
    xvadd.h        xr15,   xr15,  xr11

    addi.d         a0,     a0,    32
    addi.d         a1,     a1,    32
    addi.d         a2,     a2,    32
    addi.d         a3,     a3,    32
    addi.d         a4,     a4,    32

    pixel_sad_x4_32x2_lasx xr10, xr11, xr12, xr13

    xvadd.h        xr16,   xr12,  xr16
    xvadd.h        xr17,   xr13,  xr17
    xvadd.h        xr14,   xr14,  xr10
    xvadd.h        xr15,   xr15,  xr11
.endr

    xvhaddw.w.h    xr14,   xr14,  xr14
    xvhaddw.w.h    xr15,   xr15,  xr15
    xvhaddw.w.h    xr16,   xr16,  xr16
    xvhaddw.w.h    xr17,   xr17,  xr17
    xvhaddw.d.w    xr14,   xr14,  xr14
    xvhaddw.d.w    xr15,   xr15,  xr15
    xvhaddw.d.w    xr16,   xr16,  xr16
    xvhaddw.d.w    xr17,   xr17,  xr17

    xvpickev.w     xr10,   xr15,  xr14
    xvpickev.w     xr11,   xr17,  xr16
    xvhaddw.d.w    xr10,   xr10,  xr10
    xvhaddw.d.w    xr11,   xr11,  xr11
    xvpickev.w     xr12,   xr11,  xr10
    xvpermi.q      xr13,   xr12,  0x01

    vadd.w         vr13,   vr13,  vr12
    vst            vr13,   a6,    0
endfunc
.endm

pixel_sad_x4_64x_lasx 16
pixel_sad_x4_64x_lasx 32

.macro pixel_sad_x4_64x48_lasx h
function x265_pixel_sad_x4_64x\h\()_lasx
    slli.d         t1,     a5,    1

    pixel_sad_x4_32x2_lasx xr14, xr15, xr16, xr17

    xvhaddw.w.h    xr14,   xr14,  xr14
    xvhaddw.w.h    xr15,   xr15,  xr15
    xvhaddw.w.h    xr16,   xr16,  xr16
    xvhaddw.w.h    xr17,   xr17,  xr17

    addi.d         a0,     a0,    32
    addi.d         a1,     a1,    32
    addi.d         a2,     a2,    32
    addi.d         a3,     a3,    32
    addi.d         a4,     a4,    32

    pixel_sad_x4_32x2_lasx xr10, xr11, xr12, xr13

    xvhaddw.w.h    xr10,   xr10,  xr10
    xvhaddw.w.h    xr11,   xr11,  xr11
    xvhaddw.w.h    xr12,   xr12,  xr12
    xvhaddw.w.h    xr13,   xr13,  xr13

    xvadd.w        xr16,   xr12,  xr16
    xvadd.w        xr17,   xr13,  xr17
    xvadd.w        xr14,   xr14,  xr10
    xvadd.w        xr15,   xr15,  xr11

.rept (\h>>1)-1
    addi.d         a0,     a0,    -32
    addi.d         a1,     a1,    -32
    addi.d         a2,     a2,    -32
    addi.d         a3,     a3,    -32
    addi.d         a4,     a4,    -32

    addi.d         a0,     a0,    FENC_STRIDE<<1
    add.d          a1,     a1,    t1
    add.d          a2,     a2,    t1
    add.d          a3,     a3,    t1
    add.d          a4,     a4,    t1

    pixel_sad_x4_32x2_lasx xr10, xr11, xr12, xr13

    xvhaddw.w.h    xr10,   xr10,  xr10
    xvhaddw.w.h    xr11,   xr11,  xr11
    xvhaddw.w.h    xr12,   xr12,  xr12
    xvhaddw.w.h    xr13,   xr13,  xr13

    xvadd.w        xr16,   xr12,  xr16
    xvadd.w        xr17,   xr13,  xr17
    xvadd.w        xr14,   xr14,  xr10
    xvadd.w        xr15,   xr15,  xr11

    addi.d         a0,     a0,    32
    addi.d         a1,     a1,    32
    addi.d         a2,     a2,    32
    addi.d         a3,     a3,    32
    addi.d         a4,     a4,    32

    pixel_sad_x4_32x2_lasx xr10, xr11, xr12, xr13

    xvhaddw.w.h    xr10,   xr10,  xr10
    xvhaddw.w.h    xr11,   xr11,  xr11
    xvhaddw.w.h    xr12,   xr12,  xr12
    xvhaddw.w.h    xr13,   xr13,  xr13

    xvadd.w        xr16,   xr12,  xr16
    xvadd.w        xr17,   xr13,  xr17
    xvadd.w        xr14,   xr14,  xr10
    xvadd.w        xr15,   xr15,  xr11
.endr

    xvhaddw.d.w    xr14,   xr14,  xr14
    xvhaddw.d.w    xr15,   xr15,  xr15
    xvhaddw.d.w    xr16,   xr16,  xr16
    xvhaddw.d.w    xr17,   xr17,  xr17

    xvpickev.w     xr10,   xr15,  xr14
    xvpickev.w     xr11,   xr17,  xr16
    xvhaddw.d.w    xr10,   xr10,  xr10
    xvhaddw.d.w    xr11,   xr11,  xr11
    xvpickev.w     xr12,   xr11,  xr10
    xvpermi.q      xr13,   xr12,  0x01

    vadd.w         vr13,   vr13,  vr12
    vst            vr13,   a6,    0
endfunc
.endm

pixel_sad_x4_64x48_lasx 48
pixel_sad_x4_64x48_lasx 64

function x265_pixel_sad_x4_48x64_lasx
    slli.d         t1,     a5,    1

    pixel_sad_x4_32x2_lasx xr14, xr15, xr16, xr17

    addi.d         a0,     a0,    32
    addi.d         a1,     a1,    32
    addi.d         a2,     a2,    32
    addi.d         a3,     a3,    32
    addi.d         a4,     a4,    32

    pixel_sad_x4_32x2_lasx xr18, xr19, xr20, xr21

.rept 31
    addi.d         a0,     a0,    -32
    addi.d         a1,     a1,    -32
    addi.d         a2,     a2,    -32
    addi.d         a3,     a3,    -32
    addi.d         a4,     a4,    -32

    addi.d         a0,     a0,    FENC_STRIDE<<1
    add.d          a1,     a1,    t1
    add.d          a2,     a2,    t1
    add.d          a3,     a3,    t1
    add.d          a4,     a4,    t1

    pixel_sad_x4_32x2_lasx xr10, xr11, xr12, xr13

    xvadd.h        xr16,   xr12,  xr16
    xvadd.h        xr17,   xr13,  xr17
    xvadd.h        xr14,   xr14,  xr10
    xvadd.h        xr15,   xr15,  xr11

    addi.d         a0,     a0,    32
    addi.d         a1,     a1,    32
    addi.d         a2,     a2,    32
    addi.d         a3,     a3,    32
    addi.d         a4,     a4,    32

    pixel_sad_x4_32x2_lasx xr10, xr11, xr12, xr13

    xvadd.h        xr20,   xr12,  xr20
    xvadd.h        xr21,   xr13,  xr21
    xvadd.h        xr18,   xr18,  xr10
    xvadd.h        xr19,   xr19,  xr11
.endr

    xvreplgr2vr.w  xr23,   zero
    xvpermi.q      xr18,   xr23,  0x02
    xvpermi.q      xr19,   xr23,  0x02
    xvpermi.q      xr20,   xr23,  0x02
    xvpermi.q      xr21,   xr23,  0x02

    xvhaddw.w.h    xr14,   xr14,  xr14
    xvhaddw.w.h    xr15,   xr15,  xr15
    xvhaddw.w.h    xr16,   xr16,  xr16
    xvhaddw.w.h    xr17,   xr17,  xr17
    xvhaddw.w.h    xr18,   xr18,  xr18
    xvhaddw.w.h    xr19,   xr19,  xr19
    xvhaddw.w.h    xr20,   xr20,  xr20
    xvhaddw.w.h    xr21,   xr21,  xr21

    xvadd.w        xr14,   xr14,  xr18
    xvadd.w        xr15,   xr15,  xr19
    xvadd.w        xr16,   xr16,  xr20
    xvadd.w        xr17,   xr17,  xr21
    xvhaddw.d.w    xr14,   xr14,  xr14
    xvhaddw.d.w    xr15,   xr15,  xr15
    xvhaddw.d.w    xr16,   xr16,  xr16
    xvhaddw.d.w    xr17,   xr17,  xr17
    xvpickev.w     xr10,   xr15,  xr14
    xvpickev.w     xr11,   xr17,  xr16
    xvhaddw.d.w    xr10,   xr10,  xr10
    xvhaddw.d.w    xr11,   xr11,  xr11
    xvpickev.w     xr12,   xr11,  xr10
    xvpermi.q      xr13,   xr12,  0x01

    vadd.w         vr13,   vr13,  vr12
    vst            vr13,   a6,    0
endfunc

.macro pixel_sad_4x4_core_lsx out
    FLDS_LOADX_4    a0,    a1,   t1,  t3,  f3, f5, f7, f9
    FLDS_LOADX_4    a2,    a3,   t2,  t4,  f4, f6, f8, f10
    vilvl.w         vr3,   vr5,  vr3
    vilvl.w         vr4,   vr6,  vr4
    vilvl.w         vr7,   vr9,  vr7
    vilvl.w         vr8,   vr10, vr8
    vilvl.d         vr3,   vr7,  vr3
    vilvl.d         vr4,   vr8,  vr4
    vabsd.bu        vr5,   vr3,  vr4
    vhaddw.hu.bu    \out,  vr5,  vr5
.endm

function x265_pixel_sad_4x4_lsx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_4x4_core_lsx vr11

    vhaddw.wu.hu    vr11,  vr11, vr11
    vhaddw.du.wu    vr11,  vr11, vr11
    vhaddw.qu.du    vr11,  vr11, vr11
    vpickve2gr.wu   a0,    vr11,  0
endfunc

.macro pixel_sad_4x_lsx h
function x265_pixel_sad_4x\h\()_lsx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_4x4_core_lsx vr11

.rept (\h>>2)-1
    alsl.d          a0,    a1,   a0,  2
    alsl.d          a2,    a3,   a2,  2
    pixel_sad_4x4_core_lsx vr12
    vadd.h          vr11,  vr11, vr12
.endr

    vhaddw.wu.hu    vr11,  vr11, vr11
    vhaddw.du.wu    vr11,  vr11, vr11
    vhaddw.qu.du    vr11,  vr11, vr11
    vpickve2gr.wu   a0,    vr11, 0
endfunc
.endm

pixel_sad_4x_lsx 8
pixel_sad_4x_lsx 16

.macro pixel_sad_8x4_core_lsx out
    FLDD_LOADX_4    a0,    a1,   t1,  t3,  f3, f5, f7, f9
    FLDD_LOADX_4    a2,    a3,   t2,  t4,  f4, f6, f8, f10
    vilvl.d         vr3,   vr5,  vr3
    vilvl.d         vr7,   vr9,  vr7
    vilvl.d         vr4,   vr6,  vr4
    vilvl.d         vr8,   vr10, vr8
    vabsd.bu        vr11,  vr3,  vr4
    vabsd.bu        vr12,  vr7,  vr8
    vhaddw.hu.bu    vr11,  vr11, vr11
    vhaddw.hu.bu    vr12,  vr12, vr12
    vadd.h          \out,  vr11, vr12
.endm

function x265_pixel_sad_8x4_lsx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_8x4_core_lsx vr6

    vhaddw.wu.hu    vr6,   vr6,  vr6
    vhaddw.du.wu    vr6,   vr6,  vr6
    vhaddw.qu.du    vr6,   vr6,  vr6
    vpickve2gr.wu   a0,    vr6,  0
endfunc

.macro pixel_sad_8x_lsx h
function x265_pixel_sad_8x\h\()_lsx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_8x4_core_lsx vr13

.rept (\h>>2)-1
    alsl.d          a0,    a1,   a0,  2
    alsl.d          a2,    a3,   a2,  2
    pixel_sad_8x4_core_lsx vr14
    vadd.h          vr13,  vr13, vr14
.endr
    vhaddw.wu.hu    vr13,  vr13,  vr13
    vhaddw.du.wu    vr13,  vr13,  vr13
    vhaddw.qu.du    vr13,  vr13,  vr13
    vpickve2gr.wu   a0,    vr13,  0
endfunc
.endm

pixel_sad_8x_lsx 8
pixel_sad_8x_lsx 16
pixel_sad_8x_lsx 32

.macro pixel_sad_16x4_core_lsx out
    LSX_LOADX_4     a0,    a1,   t1, t3, vr0, vr1, vr2, vr3
    LSX_LOADX_4     a2,    a3,   t2, t4, vr4, vr5, vr6, vr7
    vabsd.bu        vr8,   vr0,  vr4
    vabsd.bu        vr9,   vr1,  vr5
    vabsd.bu        vr10,  vr2,  vr6
    vabsd.bu        vr11,  vr3,  vr7
    vhaddw.hu.bu    vr8,   vr8,  vr8
    vhaddw.hu.bu    vr9,   vr9,  vr9
    vhaddw.hu.bu    vr10,  vr10, vr10
    vhaddw.hu.bu    vr11,  vr11, vr11
    vadd.h          vr8,   vr8,  vr9
    vadd.h          vr9,   vr10, vr11
    vadd.h          \out,  vr8,  vr9
.endm

function x265_pixel_sad_12x16_lsx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_16x4_core_lsx vr13

.rept 3
    alsl.d          a0,    a1,   a0,   2
    alsl.d          a2,    a3,   a2,   2
    pixel_sad_16x4_core_lsx vr12
    vadd.h          vr13,  vr12, vr13
.endr

    vhaddw.wu.hu    vr13,  vr13, vr13
    vpickve2gr.wu   t1,    vr13, 2
    vhaddw.du.wu    vr13,  vr13, vr13
    vpickve2gr.wu   t2,    vr13, 0
    add.w           a0,    t1,   t2
endfunc

function x265_pixel_sad_16x4_lsx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_16x4_core_lsx vr13

    vhaddw.wu.hu    vr13,  vr13, vr13
    vhaddw.du.wu    vr13,  vr13, vr13
    vhaddw.qu.du    vr13,  vr13, vr13
    vpickve2gr.wu   a0,    vr13, 0
endfunc

.macro pixel_sad_16x_lsx h
function x265_pixel_sad_16x\h\()_lsx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_16x4_core_lsx vr13

.rept (\h>>2)-1
    alsl.d          a0,    a1,   a0,   2
    alsl.d          a2,    a3,   a2,   2
    pixel_sad_16x4_core_lsx vr12
    vadd.h          vr13,  vr12, vr13
.endr

    vhaddw.wu.hu    vr13,  vr13, vr13
    vhaddw.du.wu    vr13,  vr13, vr13
    vhaddw.qu.du    vr13,  vr13, vr13
    vpickve2gr.wu   a0,    vr13, 0
endfunc
.endm

pixel_sad_16x_lsx 8
pixel_sad_16x_lsx 12
pixel_sad_16x_lsx 16
pixel_sad_16x_lsx 32
pixel_sad_16x_lsx 64

function x265_pixel_sad_24x32_lsx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_16x4_core_lsx vr13

    addi.d          a0,    a0,   16
    addi.d          a2,    a2,   16
    pixel_sad_8x4_core_lsx vr15
    vadd.h          vr16,  vr15, vr13

.rept 7
    addi.d          a0,    a0,   -16
    addi.d          a2,    a2,   -16

    alsl.d          a0,    a1,   a0,   2
    alsl.d          a2,    a3,   a2,   2

    pixel_sad_16x4_core_lsx vr13

    addi.d          a0,    a0,   16
    addi.d          a2,    a2,   16
    pixel_sad_8x4_core_lsx vr15

    vadd.h          vr16,  vr16, vr13
    vadd.h          vr16,  vr16, vr15
.endr
    vhaddw.wu.hu    vr16,  vr16, vr16
    vhaddw.du.wu    vr16,  vr16, vr16
    vhaddw.qu.du    vr16,  vr16, vr16
    vpickve2gr.wu   a0,    vr16, 0
endfunc

.macro pixel_sad_32x_lsx h
function x265_pixel_sad_32x\h\()_lsx
    slli.d          t1,    a1,    1
    slli.d          t2,    a3,    1
    add.d           t3,    a1,    t1
    add.d           t4,    a3,    t2

    pixel_sad_16x4_core_lsx vr12

    addi.d          a0,    a0,    16
    addi.d          a2,    a2,    16

    pixel_sad_16x4_core_lsx vr13
    vadd.h          vr13,  vr12,  vr13

.rept (\h>>2)-1
    addi.d          a0,    a0,    -16
    addi.d          a2,    a2,    -16

    alsl.d          a0,    a1,    a0,   2
    alsl.d          a2,    a3,    a2,   2

    pixel_sad_16x4_core_lsx vr14

    addi.d          a0,    a0,    16
    addi.d          a2,    a2,    16

    pixel_sad_16x4_core_lsx vr15
    vadd.h          vr14,  vr14,  vr15
    vadd.h          vr13,  vr13,  vr14
.endr

    vhaddw.wu.hu    vr13,  vr13,  vr13
    vhaddw.du.wu    vr13,  vr13,  vr13
    vhaddw.qu.du    vr13,  vr13,  vr13
    vpickve2gr.wu   a0,    vr13,  0
endfunc
.endm

pixel_sad_32x_lsx 8
pixel_sad_32x_lsx 16
pixel_sad_32x_lsx 24
pixel_sad_32x_lsx 32
pixel_sad_32x_lsx 64

function x265_pixel_sad_64x16_lsx
    slli.d          t1,    a1,    1
    slli.d          t2,    a3,    1
    add.d           t3,    a1,    t1
    add.d           t4,    a3,    t2

    pixel_sad_16x4_core_lsx vr19

.rept 3
    addi.d          a0,    a0,    16
    addi.d          a2,    a2,    16

    pixel_sad_16x4_core_lsx vr13
    vadd.h          vr19,  vr19,  vr13
.endr

.rept 3
    addi.d          a0,    a0,    -48
    addi.d          a2,    a2,    -48

    alsl.d          a0,    a1,    a0,   2
    alsl.d          a2,    a3,    a2,   2

    pixel_sad_16x4_core_lsx vr14
    vadd.h          vr19,  vr19,  vr14

.rept 3
    addi.d          a0,    a0,    16
    addi.d          a2,    a2,    16

    pixel_sad_16x4_core_lsx vr15
    vadd.h          vr19,  vr19,  vr15
.endr
.endr

    vhaddw.wu.hu    vr19,  vr19,  vr19
    vhaddw.du.wu    vr19,  vr19,  vr19
    vhaddw.qu.du    vr19,  vr19,  vr19
    vpickve2gr.wu   a0,    vr19,  0
endfunc

function x265_pixel_sad_64x32_lsx
    slli.d          t1,    a1,    1
    slli.d          t2,    a3,    1
    add.d           t3,    a1,    t1
    add.d           t4,    a3,    t2

    pixel_sad_16x4_core_lsx vr19

.rept 3
    addi.d          a0,    a0,    16
    addi.d          a2,    a2,    16

    pixel_sad_16x4_core_lsx vr13
    vadd.h          vr19,  vr19,  vr13
.endr

.rept 7
    addi.d          a0,    a0,    -48
    addi.d          a2,    a2,    -48

    alsl.d          a0,    a1,    a0,   2
    alsl.d          a2,    a3,    a2,   2

    pixel_sad_16x4_core_lsx vr14
    vadd.h          vr19,  vr19,  vr14

.rept 3
    addi.d          a0,    a0,    16
    addi.d          a2,    a2,    16

    pixel_sad_16x4_core_lsx vr15
    vadd.h          vr19,  vr19,  vr15
.endr
.endr

    vhaddw.wu.hu    vr19,  vr19,  vr19
    vhaddw.du.wu    vr19,  vr19,  vr19
    vhaddw.qu.du    vr19,  vr19,  vr19
    vpickve2gr.wu   a0,    vr19,  0
endfunc

.macro pixel_sad_64x48_lsx h
function x265_pixel_sad_64x\h\()_lsx
    slli.d          t1,    a1,    1
    slli.d          t2,    a3,    1
    add.d           t3,    a1,    t1
    add.d           t4,    a3,    t2

    pixel_sad_16x4_core_lsx vr12
    vhaddw.wu.hu    vr19,  vr12,  vr12

.rept 3
    addi.d          a0,    a0,    16
    addi.d          a2,    a2,    16

    pixel_sad_16x4_core_lsx vr13
    vhaddw.wu.hu    vr13,  vr13,  vr13
    vadd.w          vr19,  vr19,  vr13
.endr

.rept (\h>>2)-1
    addi.d          a0,    a0,    -48
    addi.d          a2,    a2,    -48

    alsl.d          a0,    a1,    a0,   2
    alsl.d          a2,    a3,    a2,   2

    pixel_sad_16x4_core_lsx vr14
    vhaddw.wu.hu    vr14,  vr14,  vr14
    vadd.w          vr19,  vr19,  vr14

.rept 3
    addi.d          a0,    a0,    16
    addi.d          a2,    a2,    16

    pixel_sad_16x4_core_lsx vr15
    vhaddw.wu.hu    vr15,  vr15,  vr15
    vadd.w          vr19,  vr19,  vr15
.endr
.endr

    vhaddw.du.wu    vr19,  vr19,  vr19
    vhaddw.qu.du    vr19,  vr19,  vr19
    vpickve2gr.wu   a0,    vr19,  0
endfunc
.endm

pixel_sad_64x48_lsx 48
pixel_sad_64x48_lsx 64

function x265_pixel_sad_48x64_lsx
    slli.d          t1,    a1,    1
    slli.d          t2,    a3,    1
    add.d           t3,    a1,    t1
    add.d           t4,    a3,    t2

    pixel_sad_16x4_core_lsx vr12
    vhaddw.wu.hu    vr19,  vr12,  vr12

.rept 2
    addi.d          a0,    a0,    16
    addi.d          a2,    a2,    16

    pixel_sad_16x4_core_lsx vr13
    vhaddw.wu.hu    vr13,  vr13,  vr13
    vadd.w          vr19,  vr19,  vr13
.endr

.rept 15
    addi.d          a0,    a0,    -32
    addi.d          a2,    a2,    -32

    alsl.d          a0,    a1,    a0,   2
    alsl.d          a2,    a3,    a2,   2

    pixel_sad_16x4_core_lsx vr14
    vhaddw.wu.hu    vr14,  vr14,  vr14
    vadd.w          vr19,  vr19,  vr14

.rept 2
    addi.d          a0,    a0,    16
    addi.d          a2,    a2,    16

    pixel_sad_16x4_core_lsx vr15
    vhaddw.wu.hu    vr15,  vr15,  vr15
    vadd.w          vr19,  vr19,  vr15
.endr
.endr

    vhaddw.du.wu    vr19,  vr19,  vr19
    vhaddw.qu.du    vr19,  vr19,  vr19
    vpickve2gr.wu   a0,    vr19,  0
endfunc

.macro pixel_sad_32x4_core_lasx out
    LASX_LOADX_4    a0,    a1,   t1, t3, xr0, xr1, xr2, xr3
    LASX_LOADX_4    a2,    a3,   t2, t4, xr4, xr5, xr6, xr7

    xvabsd.bu       xr8,   xr0,  xr4
    xvabsd.bu       xr9,   xr1,  xr5
    xvabsd.bu       xr10,  xr2,  xr6
    xvabsd.bu       xr11,  xr3,  xr7
    xvhaddw.hu.bu   xr8,   xr8,  xr8
    xvhaddw.hu.bu   xr9,   xr9,  xr9
    xvhaddw.hu.bu   xr10,  xr10, xr10
    xvhaddw.hu.bu   xr11,  xr11, xr11
    xvadd.h         xr0,   xr8,  xr9
    xvadd.h         xr1,   xr11, xr10
    xvadd.h         \out,  xr0,  xr1
.endm

.macro pixel_sad_32x_lasx h
function x265_pixel_sad_32x\h\()_lasx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_32x4_core_lasx xr12

.rept (\h>>2)-1
    alsl.d          a0,    a1,   a0,   2
    alsl.d          a2,    a3,   a2,   2

    pixel_sad_32x4_core_lasx xr13

    xvadd.h         xr12,  xr13,  xr12
.endr

    xvhaddw.w.h     xr12,  xr12,  xr12
    xvhaddw.d.w     xr12,  xr12,  xr12
    xvhaddw.q.d     xr12,  xr12,  xr12
    xvpickve2gr.w   t2,    xr12,  0
    xvpickve2gr.w   t3,    xr12,  4
    add.d           a0,    t2,   t3
endfunc
.endm

pixel_sad_32x_lasx 8
pixel_sad_32x_lasx 16
pixel_sad_32x_lasx 24
pixel_sad_32x_lasx 32
pixel_sad_32x_lasx 64

.macro pixel_sad_64x_lasx h
function x265_pixel_sad_64x\h\()_lasx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_32x4_core_lasx xr12
    addi.d a0, a0, 32
    addi.d a2, a2, 32
    pixel_sad_32x4_core_lasx xr13
    xvadd.h xr12, xr13, xr12

.rept (\h>>2)-1
    addi.d a0, a0, -32
    addi.d a2, a2, -32

    alsl.d          a0,    a1,   a0,   2
    alsl.d          a2,    a3,   a2,   2

    pixel_sad_32x4_core_lasx xr13
    xvadd.h         xr12,  xr13,  xr12

    addi.d a0, a0, 32
    addi.d a2, a2, 32
    pixel_sad_32x4_core_lasx xr13
    xvadd.h xr12, xr13, xr12
.endr

    xvhaddw.w.h     xr12,  xr12,  xr12
    xvhaddw.d.w     xr12,  xr12,  xr12
    xvhaddw.q.d     xr12,  xr12,  xr12
    xvpickve2gr.w   t2,    xr12,  0
    xvpickve2gr.w   t3,    xr12,  4
    add.d           a0,    t2,   t3
endfunc
.endm

.macro pixel_sad_64x48_lasx h
function x265_pixel_sad_64x\h\()_lasx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_32x4_core_lasx xr12
    xvhaddw.wu.hu   xr12,  xr12, xr12
    addi.d          a0,    a0,   32
    addi.d          a2,    a2,   32
    pixel_sad_32x4_core_lasx xr13
    xvhaddw.wu.hu   xr13,  xr13, xr13
    xvadd.w         xr12,  xr13, xr12

.rept (\h>>2)-1
    addi.d          a0,    a0,   -32
    addi.d          a2,    a2,   -32

    alsl.d          a0,    a1,   a0,   2
    alsl.d          a2,    a3,   a2,   2

    pixel_sad_32x4_core_lasx xr13
    xvhaddw.wu.hu   xr13,  xr13, xr13
    xvadd.w         xr12,  xr13, xr12

    addi.d          a0,    a0,   32
    addi.d          a2,    a2,   32
    pixel_sad_32x4_core_lasx xr13
    xvhaddw.wu.hu   xr13,  xr13, xr13
    xvadd.w         xr12,  xr13, xr12
.endr

    xvhaddw.d.w     xr12,  xr12,  xr12
    xvhaddw.q.d     xr12,  xr12,  xr12
    xvpickve2gr.w   t2,    xr12,  0
    xvpickve2gr.w   t3,    xr12,  4
    add.d           a0,    t2,   t3
endfunc
.endm

pixel_sad_64x_lasx 16
pixel_sad_64x_lasx 32
pixel_sad_64x48_lasx 48
pixel_sad_64x48_lasx 64

function x265_pixel_sad_24x32_lasx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_32x4_core_lasx xr12
.rept 7
    alsl.d          a0,    a1,   a0,   2
    alsl.d          a2,    a3,   a2,   2

    pixel_sad_32x4_core_lasx xr13

    xvadd.h         xr12,  xr13,  xr12
.endr

    xvhaddw.w.h     xr12,  xr12,  xr12
    xvhaddw.d.w     xr12,  xr12,  xr12
    xvinsgr2vr.d    xr12,  zero,  3
    xvhaddw.q.d     xr12,  xr12,  xr12
    xvpickve2gr.w   t2,    xr12,  0
    xvpickve2gr.w   t3,    xr12,  4
    add.d           a0,    t2,    t3
endfunc

function x265_pixel_sad_48x64_lasx
    slli.d          t1,    a1,   1
    slli.d          t2,    a3,   1
    add.d           t3,    a1,   t1
    add.d           t4,    a3,   t2

    pixel_sad_32x4_core_lasx xr12
    xvhaddw.wu.hu   xr12,  xr12, xr12
    addi.d          a0,    a0,   32
    addi.d          a2,    a2,   32
    pixel_sad_16x4_core_lsx vr13
    vhaddw.wu.hu    vr13,  vr13, vr13

.rept 15
    addi.d          a0,    a0,   -32
    addi.d          a2,    a2,   -32

    alsl.d          a0,    a1,   a0,   2
    alsl.d          a2,    a3,   a2,   2

    pixel_sad_32x4_core_lasx xr14
    xvhaddw.wu.hu   xr14,  xr14, xr14
    xvadd.w         xr12,  xr14, xr12

    addi.d          a0,    a0,   32
    addi.d          a2,    a2,   32
    pixel_sad_16x4_core_lsx vr15
    vhaddw.wu.hu    vr15,  vr15, vr15
    vadd.w          vr13,  vr13, vr15
.endr

    xvinsgr2vr.d    xr13,  zero,  2
    xvinsgr2vr.d    xr13,  zero,  3
    xvadd.w         xr12,  xr12,  xr13
    xvhaddw.d.w     xr12,  xr12,  xr12
    xvhaddw.q.d     xr12,  xr12,  xr12
    xvpickve2gr.w   t2,    xr12,  0
    xvpickve2gr.w   t3,    xr12,  4
    add.d           a0,    t2,    t3
endfunc

.macro pixel_sad_x3_4x4_core_lsx out0, out1, out2
    fld.s          f3,     a0,    0
    fld.s          f7,     a0,    FENC_STRIDE
    fld.s          f11,    a0,    FENC_STRIDE<<1
    fld.s          f15,    a0,    FENC_STRIDE*3
    FLDS_LOADX_4   a1,     a4,    t1,  t2,  f4, f8,  f12, f16
    FLDS_LOADX_4   a2,     a4,    t1,  t2,  f5, f9,  f13, f17
    FLDS_LOADX_4   a3,     a4,    t1,  t2,  f6, f10, f14, f18

    vilvl.w        vr3,    vr7,   vr3
    vilvl.w        vr4,    vr8,   vr4
    vilvl.w        vr5,    vr9,   vr5
    vilvl.w        vr6,    vr10,  vr6
    vilvl.w        vr11,   vr15,  vr11
    vilvl.w        vr12,   vr16,  vr12
    vilvl.w        vr13,   vr17,  vr13
    vilvl.w        vr14,   vr18,  vr14
    vilvl.d        vr3,    vr11,  vr3
    vilvl.d        vr4,    vr12,  vr4
    vilvl.d        vr5,    vr13,  vr5
    vilvl.d        vr6,    vr14,  vr6

    vabsd.bu       \out0,  vr3,   vr4
    vabsd.bu       \out1,  vr3,   vr5
    vabsd.bu       \out2,  vr3,   vr6
.endm

.macro SUM_VEC_x3 in0, in1, in2
    vhaddw.wu.hu   \in0,   \in0,  \in0
    vhaddw.wu.hu   \in1,   \in1,  \in1
    vhaddw.wu.hu   \in2,   \in2,  \in2
    vhaddw.du.wu   \in0,   \in0,  \in0
    vhaddw.du.wu   \in1,   \in1,  \in1
    vhaddw.du.wu   \in2,   \in2,  \in2
    vhaddw.qu.du   \in0,   \in0,  \in0
    vhaddw.qu.du   \in1,   \in1,  \in1
    vhaddw.qu.du   \in2,   \in2,  \in2
.endm

function x265_pixel_sad_x3_4x4_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_4x4_core_lsx vr7, vr8, vr9

    vhaddw.hu.bu   vr7,    vr7,   vr7
    vhaddw.hu.bu   vr8,    vr8,   vr8
    vhaddw.hu.bu   vr9,    vr9,   vr9

    SUM_VEC_x3 vr7, vr8, vr9

    vstelm.w       vr7,    a5,    0,  0
    vstelm.w       vr8,    a5,    4,  0
    vstelm.w       vr9,    a5,    8,  0
endfunc

.macro pixel_sad_x3_4x8_core_lsx out0, out1, out2
    pixel_sad_x3_4x4_core_lsx vr0, vr1, vr2

    alsl.d         a1,     a4,    a1,   2
    alsl.d         a2,     a4,    a2,   2
    alsl.d         a3,     a4,    a3,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x3_4x4_core_lsx vr7, vr8, vr9

    vhaddw.hu.bu   vr0,    vr0,   vr0
    vhaddw.hu.bu   vr1,    vr1,   vr1
    vhaddw.hu.bu   vr2,    vr2,   vr2
    vhaddw.hu.bu   vr7,    vr7,   vr7
    vhaddw.hu.bu   vr8,    vr8,   vr8
    vhaddw.hu.bu   vr9,    vr9,   vr9
    vadd.h         \out0,  vr7,   vr0
    vadd.h         \out1,  vr8,   vr1
    vadd.h         \out2,  vr9,   vr2
.endm

function x265_pixel_sad_x3_4x8_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_4x8_core_lsx vr7, vr8, vr9

    SUM_VEC_x3 vr7, vr8, vr9

    vstelm.w       vr7,    a5,    0,  0
    vstelm.w       vr8,    a5,    4,  0
    vstelm.w       vr9,    a5,    8,  0
endfunc

function x265_pixel_sad_x3_4x16_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_4x8_core_lsx vr19, vr20, vr21

    addi.d         a0,     a0,    FENC_STRIDE<<2
    alsl.d         a1,     a4,    a1,   2
    alsl.d         a2,     a4,    a2,   2
    alsl.d         a3,     a4,    a3,   2
    pixel_sad_x3_4x8_core_lsx vr7, vr8, vr9

    vadd.h         vr7,    vr7,   vr19
    vadd.h         vr8,    vr8,   vr20
    vadd.h         vr9,    vr9,   vr21
    SUM_VEC_x3 vr7, vr8, vr9

    vstelm.w       vr7,    a5,    0,  0
    vstelm.w       vr8,    a5,    4,  0
    vstelm.w       vr9,    a5,    8,  0
endfunc

.macro pixel_sad_x3_8x4_core_lsx out0, out1, out2
    fld.d          f3,     a0,    0
    fld.d          f7,     a0,    FENC_STRIDE
    fld.d          f11,    a0,    FENC_STRIDE<<1
    fld.d          f15,    a0,    FENC_STRIDE*3
    FLDD_LOADX_4   a1,     a4,    t1,  t2,  f4, f8,  f12, f16
    FLDD_LOADX_4   a2,     a4,    t1,  t2,  f5, f9,  f13, f17
    FLDD_LOADX_4   a3,     a4,    t1,  t2,  f6, f10, f14, f18

    vilvl.d        vr3,    vr7,   vr3
    vilvl.d        vr4,    vr8,   vr4
    vilvl.d        vr5,    vr9,   vr5
    vilvl.d        vr6,    vr10,  vr6
    vilvl.d        vr11,   vr15,  vr11
    vilvl.d        vr12,   vr16,  vr12
    vilvl.d        vr13,   vr17,  vr13
    vilvl.d        vr14,   vr18,  vr14
    vabsd.bu       vr7,    vr3,   vr4
    vabsd.bu       vr8,    vr3,   vr5
    vabsd.bu       vr9,    vr3,   vr6
    vabsd.bu       vr3,    vr11,  vr12
    vabsd.bu       vr4,    vr11,  vr13
    vabsd.bu       vr5,    vr11,  vr14
    vhaddw.hu.bu   vr7,    vr7,   vr7
    vhaddw.hu.bu   vr8,    vr8,   vr8
    vhaddw.hu.bu   vr9,    vr9,   vr9
    vhaddw.hu.bu   vr3,    vr3,   vr3
    vhaddw.hu.bu   vr4,    vr4,   vr4
    vhaddw.hu.bu   vr5,    vr5,   vr5
    vadd.h         \out0,  vr7,   vr3
    vadd.h         \out1,  vr8,   vr4
    vadd.h         \out2,  vr9,   vr5
.endm

function x265_pixel_sad_x3_8x4_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_8x4_core_lsx vr7, vr8, vr9

    SUM_VEC_x3 vr7, vr8, vr9

    vstelm.w       vr7,    a5,    0,  0
    vstelm.w       vr8,    a5,    4,  0
    vstelm.w       vr9,    a5,    8,  0
endfunc

.macro sad_x3_8x_lsx h
function x265_pixel_sad_x3_8x\h\()_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_8x4_core_lsx vr20, vr21, vr22

.rept (\h>>2)-1
    alsl.d         a1,     a4,    a1,   2
    alsl.d         a2,     a4,    a2,   2
    alsl.d         a3,     a4,    a3,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x3_8x4_core_lsx vr7, vr8, vr9

    vadd.h         vr20,   vr7,   vr20
    vadd.h         vr21,   vr8,   vr21
    vadd.h         vr22,   vr9,   vr22
.endr

    SUM_VEC_x3 vr20, vr21, vr22

    vstelm.w       vr20,   a5,    0,  0
    vstelm.w       vr21,   a5,    4,  0
    vstelm.w       vr22,   a5,    8,  0
endfunc
.endm

sad_x3_8x_lsx 8
sad_x3_8x_lsx 16
sad_x3_8x_lsx 32

.macro pixel_sad_x3_16x4_core_lsx out0, out1, out2
    vld            vr0,    a0,    0
    vld            vr1,    a0,    FENC_STRIDE
    vld            vr2,    a0,    FENC_STRIDE<<1
    vld            vr3,    a0,    FENC_STRIDE*3
    LSX_LOADX_4    a1,     a4,    t1,  t2,  vr4, vr7, vr10, vr13
    LSX_LOADX_4    a2,     a4,    t1,  t2,  vr5, vr8, vr11, vr14
    LSX_LOADX_4    a3,     a4,    t1,  t2,  vr6, vr9, vr12, vr15
    vabsd.bu       vr4,    vr0,   vr4
    vabsd.bu       vr5,    vr0,   vr5
    vabsd.bu       vr6,    vr0,   vr6
    vabsd.bu       vr7,    vr1,   vr7
    vabsd.bu       vr8,    vr1,   vr8
    vabsd.bu       vr9,    vr1,   vr9
    vabsd.bu       vr10,   vr2,   vr10
    vabsd.bu       vr11,   vr2,   vr11
    vabsd.bu       vr12,   vr2,   vr12
    vabsd.bu       vr13,   vr3,   vr13
    vabsd.bu       vr14,   vr3,   vr14
    vabsd.bu       vr15,   vr3,   vr15
    vhaddw.hu.bu   vr4,    vr4,   vr4
    vhaddw.hu.bu   vr5,    vr5,   vr5
    vhaddw.hu.bu   vr6,    vr6,   vr6
    vhaddw.hu.bu   vr7,    vr7,   vr7
    vhaddw.hu.bu   vr8,    vr8,   vr8
    vhaddw.hu.bu   vr9,    vr9,   vr9
    vhaddw.hu.bu   vr10,   vr10,  vr10
    vhaddw.hu.bu   vr11,   vr11,  vr11
    vhaddw.hu.bu   vr12,   vr12,  vr12
    vhaddw.hu.bu   vr13,   vr13,  vr13
    vhaddw.hu.bu   vr14,   vr14,  vr14
    vhaddw.hu.bu   vr15,   vr15,  vr15
    vadd.h         vr0,    vr7,   vr4
    vadd.h         vr1,    vr13,  vr10
    vadd.h         vr2,    vr8,   vr5
    vadd.h         vr3,    vr14,  vr11
    vadd.h         vr4,    vr9,   vr6
    vadd.h         vr7,    vr15,  vr12
    vadd.h         \out0,  vr1,   vr0
    vadd.h         \out1,  vr3,   vr2
    vadd.h         \out2,  vr7,   vr4
.endm

function x265_pixel_sad_x3_16x4_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_16x4_core_lsx vr16, vr17, vr18

    SUM_VEC_x3 vr16, vr17, vr18

    vstelm.w       vr16,    a5,    0,  0
    vstelm.w       vr17,    a5,    4,  0
    vstelm.w       vr18,    a5,    8,  0
endfunc

.macro sad_x3_16x_lsx h
function x265_pixel_sad_x3_16x\h\()_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_16x4_core_lsx vr16, vr17, vr18

.rept (\h>>2)-1
    alsl.d         a1,     a4,    a1,   2
    alsl.d         a2,     a4,    a2,   2
    alsl.d         a3,     a4,    a3,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2
    pixel_sad_x3_16x4_core_lsx vr8, vr9, vr10

    vadd.h         vr16,    vr16,  vr8
    vadd.h         vr17,    vr17,  vr9
    vadd.h         vr18,    vr18,  vr10
.endr

    SUM_VEC_x3 vr16, vr17, vr18

    vstelm.w       vr16,    a5,    0,  0
    vstelm.w       vr17,    a5,    4,  0
    vstelm.w       vr18,    a5,    8,  0
endfunc
.endm

sad_x3_16x_lsx 8
sad_x3_16x_lsx 12
sad_x3_16x_lsx 16
sad_x3_16x_lsx 32
sad_x3_16x_lsx 64

function x265_pixel_sad_x3_12x16_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_16x4_core_lsx vr16, vr17, vr18

.rept 3
    alsl.d         a1,     a4,    a1,   2
    alsl.d         a2,     a4,    a2,   2
    alsl.d         a3,     a4,    a3,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x3_16x4_core_lsx vr8, vr9, vr10

    vadd.h         vr16,   vr16,  vr8
    vadd.h         vr17,   vr17,  vr9
    vadd.h         vr18,   vr18,  vr10
.endr

    vinsgr2vr.w    vr16,   zero,  3
    vinsgr2vr.w    vr17,   zero,  3
    vinsgr2vr.w    vr18,   zero,  3

    SUM_VEC_x3 vr16, vr17, vr18

    vstelm.w       vr16,    a5,    0,  0
    vstelm.w       vr17,    a5,    4,  0
    vstelm.w       vr18,    a5,    8,  0
endfunc

.macro pixel_sad_x3_24x4_lsx out0, out1, out2
    pixel_sad_x3_16x4_core_lsx \out0, \out1, \out2

    addi.d         a1,     a1,     16
    addi.d         a2,     a2,     16
    addi.d         a3,     a3,     16
    addi.d         a0,     a0,     16

    pixel_sad_x3_8x4_core_lsx vr16, vr17, vr18

    vadd.h         \out0,  \out0,  vr16
    vadd.h         \out1,  \out1,  vr17
    vadd.h         \out2,  \out2,  vr18
.endm

function x265_pixel_sad_x3_24x32_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_24x4_lsx vr19, vr20, vr21

.rept 7
    addi.d         a1,  a1,  -16
    addi.d         a2,  a2,  -16
    addi.d         a3,  a3,  -16
    addi.d         a0,  a0,  -16

    alsl.d         a1,     a4,    a1,   2
    alsl.d         a2,     a4,    a2,   2
    alsl.d         a3,     a4,    a3,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x3_24x4_lsx vr0, vr1, vr2

    vadd.h         vr19,   vr0,  vr19
    vadd.h         vr20,   vr1,  vr20
    vadd.h         vr21,   vr2,  vr21
.endr

    SUM_VEC_x3 vr19, vr20, vr21

    vstelm.w       vr19,    a5,    0,  0
    vstelm.w       vr20,    a5,    4,  0
    vstelm.w       vr21,    a5,    8,  0
endfunc

.macro pixel_sad_x3_32x4_core_lsx out0, out1, out2
    pixel_sad_x3_16x4_core_lsx \out0, \out1, \out2

    addi.d         a1,     a1,     16
    addi.d         a2,     a2,     16
    addi.d         a3,     a3,     16
    addi.d         a0,     a0,     16

    pixel_sad_x3_16x4_core_lsx vr8, vr9, vr10

    vadd.h         \out0,  \out0,  vr8
    vadd.h         \out1,  \out1,  vr9
    vadd.h         \out2,  \out2,  vr10
.endm

.macro sad_x3_32x_lsx h
function x265_pixel_sad_x3_32x\h\()_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_32x4_core_lsx vr16, vr17, vr18

.rept (\h>>2)-1
    addi.d         a1,     a1,    -16
    addi.d         a2,     a2,    -16
    addi.d         a3,     a3,    -16
    addi.d         a0,     a0,    -16

    alsl.d         a1,     a4,    a1,   2
    alsl.d         a2,     a4,    a2,   2
    alsl.d         a3,     a4,    a3,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x3_32x4_core_lsx vr19, vr20, vr21

    vadd.h         vr16,   vr16,  vr19
    vadd.h         vr17,   vr17,  vr20
    vadd.h         vr18,   vr18,  vr21
.endr

    SUM_VEC_x3 vr16, vr17, vr18

    vstelm.w       vr16,    a5,    0,  0
    vstelm.w       vr17,    a5,    4,  0
    vstelm.w       vr18,    a5,    8,  0
endfunc
.endm

sad_x3_32x_lsx 8
sad_x3_32x_lsx 16
sad_x3_32x_lsx 24
sad_x3_32x_lsx 32
sad_x3_32x_lsx 64

.macro pixel_sad_x3_48x4_core_lsx out0, out1, out2
    pixel_sad_x3_16x4_core_lsx \out0, \out1, \out2

.rept 2
    addi.d         a1,     a1,     16
    addi.d         a2,     a2,     16
    addi.d         a3,     a3,     16
    addi.d         a0,     a0,     16

    pixel_sad_x3_16x4_core_lsx vr8, vr9, vr10

    vadd.h         \out0,  \out0,  vr8
    vadd.h         \out1,  \out1,  vr9
    vadd.h         \out2,  \out2,  vr10
.endr
.endm

function x265_pixel_sad_x3_48x64_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_48x4_core_lsx vr16, vr17, vr18

    vhaddw.wu.hu   vr16,   vr16,  vr16
    vhaddw.wu.hu   vr17,   vr17,  vr17
    vhaddw.wu.hu   vr18,   vr18,  vr18

.rept 15
    addi.d         a1,     a1,    -32
    addi.d         a2,     a2,    -32
    addi.d         a3,     a3,    -32
    addi.d         a0,     a0,    -32

    alsl.d         a1,     a4,    a1,   2
    alsl.d         a2,     a4,    a2,   2
    alsl.d         a3,     a4,    a3,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x3_48x4_core_lsx vr19, vr20, vr21

    vhaddw.wu.hu   vr19,   vr19,  vr19
    vhaddw.wu.hu   vr20,   vr20,  vr20
    vhaddw.wu.hu   vr21,   vr21,  vr21

    vadd.w         vr16,   vr16,  vr19
    vadd.w         vr17,   vr17,  vr20
    vadd.w         vr18,   vr18,  vr21
.endr

    vhaddw.du.wu   vr16,   vr16,  vr16
    vhaddw.du.wu   vr17,   vr17,  vr17
    vhaddw.du.wu   vr18,   vr18,  vr18
    vhaddw.qu.du   vr16,   vr16,  vr16
    vhaddw.qu.du   vr17,   vr17,  vr17
    vhaddw.qu.du   vr18,   vr18,  vr18

    vstelm.w       vr16,    a5,    0,  0
    vstelm.w       vr17,    a5,    4,  0
    vstelm.w       vr18,    a5,    8,  0
endfunc

.macro pixel_sad_x3_64x4_core_lsx out0, out1, out2
    pixel_sad_x3_16x4_core_lsx \out0, \out1, \out2

.rept 3
    addi.d         a1,     a1,     16
    addi.d         a2,     a2,     16
    addi.d         a3,     a3,     16
    addi.d         a0,     a0,     16

    pixel_sad_x3_16x4_core_lsx vr8, vr9, vr10

    vadd.h         \out0,  \out0,  vr8
    vadd.h         \out1,  \out1,  vr9
    vadd.h         \out2,  \out2,  vr10
.endr
.endm

.macro sad_x3_64x_lsx h
function x265_pixel_sad_x3_64x\h\()_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_64x4_core_lsx vr16, vr17, vr18

.rept (\h>>2)-1
    addi.d         a1,     a1,    -48
    addi.d         a2,     a2,    -48
    addi.d         a3,     a3,    -48
    addi.d         a0,     a0,    -48

    alsl.d         a1,     a4,    a1,   2
    alsl.d         a2,     a4,    a2,   2
    alsl.d         a3,     a4,    a3,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x3_64x4_core_lsx vr19, vr20, vr21

    vadd.h         vr16,   vr16,  vr19
    vadd.h         vr17,   vr17,  vr20
    vadd.h         vr18,   vr18,  vr21
.endr

    SUM_VEC_x3 vr16, vr17, vr18

    vstelm.w       vr16,    a5,    0,  0
    vstelm.w       vr17,    a5,    4,  0
    vstelm.w       vr18,    a5,    8,  0
endfunc
.endm

sad_x3_64x_lsx 16
sad_x3_64x_lsx 32

.macro sad_x3_64x48_lsx h
function x265_pixel_sad_x3_64x\h\()_lsx
    slli.d         t1,     a4,    1
    add.d          t2,     a4,    t1

    pixel_sad_x3_64x4_core_lsx vr16, vr17, vr18

    vhaddw.wu.hu   vr16,   vr16,  vr16
    vhaddw.wu.hu   vr17,   vr17,  vr17
    vhaddw.wu.hu   vr18,   vr18,  vr18

.rept (\h>>2)-1
    addi.d         a1,     a1,    -48
    addi.d         a2,     a2,    -48
    addi.d         a3,     a3,    -48
    addi.d         a0,     a0,    -48

    alsl.d         a1,     a4,    a1,   2
    alsl.d         a2,     a4,    a2,   2
    alsl.d         a3,     a4,    a3,   2
    addi.d         a0,     a0,    FENC_STRIDE<<2

    pixel_sad_x3_64x4_core_lsx vr19, vr20, vr21

    vhaddw.wu.hu   vr19,   vr19,  vr19
    vhaddw.wu.hu   vr20,   vr20,  vr20
    vhaddw.wu.hu   vr21,   vr21,  vr21

    vadd.w         vr16,   vr16,  vr19
    vadd.w         vr17,   vr17,  vr20
    vadd.w         vr18,   vr18,  vr21
.endr

    vhaddw.du.wu   vr16,   vr16,  vr16
    vhaddw.du.wu   vr17,   vr17,  vr17
    vhaddw.du.wu   vr18,   vr18,  vr18
    vhaddw.qu.du   vr16,   vr16,  vr16
    vhaddw.qu.du   vr17,   vr17,  vr17
    vhaddw.qu.du   vr18,   vr18,  vr18
    vstelm.w       vr16,   a5,    0,  0
    vstelm.w       vr17,   a5,    4,  0
    vstelm.w       vr18,   a5,    8,  0
endfunc
.endm

sad_x3_64x48_lsx 48
sad_x3_64x48_lsx 64

.macro pixel_sad_x3_16x2_core_lasx out0, out1, out2
    vld           vr0,      a0,     0
    vld           vr1,      a0,     FENC_STRIDE
    vld           vr2,      a1,     0
    vldx          vr3,      a1,     a4
    vld           vr4,      a2,     0
    vldx          vr5,      a2,     a4
    vld           vr6,      a3,     0
    vldx          vr7,      a3,     a4

    xvpermi.q     xr1,      xr0,    0x20
    xvpermi.q     xr3,      xr2,    0x20
    xvpermi.q     xr5,      xr4,    0x20
    xvpermi.q     xr7,      xr6,    0x20
    xvabsd.bu     xr8,      xr1,    xr3
    xvabsd.bu     xr9,      xr1,    xr5
    xvabsd.bu     xr10,     xr1,    xr7
    xvhaddw.hu.bu \out0,    xr8,    xr8
    xvhaddw.hu.bu \out1,    xr9,    xr9
    xvhaddw.hu.bu \out2,    xr10,   xr10
.endm

.macro pixel_sad_x3_16x_lasx h
function x265_pixel_sad_x3_16x\h\()_lasx
    slli.d        t1,       a4,     1

    pixel_sad_x3_16x2_core_lasx xr20, xr21, xr22

.rept (\h>>1)-1
    addi.d        a0,       a0,     FENC_STRIDE<<1
    add.d         a1,       a1,     t1
    add.d         a2,       a2,     t1
    add.d         a3,       a3,     t1

    pixel_sad_x3_16x2_core_lasx xr0, xr1, xr2

    xvadd.h       xr20,     xr0,    xr20
    xvadd.h       xr21,     xr1,    xr21
    xvadd.h       xr22,     xr2,    xr22
.endr

    xvhaddw.w.h   xr20,     xr20,   xr20
    xvhaddw.w.h   xr21,     xr21,   xr21
    xvhaddw.w.h   xr22,     xr22,   xr22
    xvhaddw.d.w   xr20,     xr20,   xr20
    xvhaddw.d.w   xr21,     xr21,   xr21
    xvhaddw.d.w   xr22,     xr22,   xr22
    xvhaddw.q.d   xr20,     xr20,   xr20
    xvhaddw.q.d   xr21,     xr21,   xr21
    xvhaddw.q.d   xr22,     xr22,   xr22
    xvpermi.q     xr3,      xr20,   0x01
    xvpermi.q     xr4,      xr21,   0x01
    xvpermi.q     xr5,      xr22,   0x01
    vadd.w        vr20,     vr3,    vr20
    vadd.w        vr21,     vr4,    vr21
    vadd.w        vr22,     vr5,    vr22

    vstelm.w      vr20,     a5,     0,    0
    vstelm.w      vr21,     a5,     4,    0
    vstelm.w      vr22,     a5,     8,    0
endfunc
.endm

pixel_sad_x3_16x_lasx 4
pixel_sad_x3_16x_lasx 8
pixel_sad_x3_16x_lasx 12
pixel_sad_x3_16x_lasx 16
pixel_sad_x3_16x_lasx 32
pixel_sad_x3_16x_lasx 64

function x265_pixel_sad_x3_12x16_lasx
    slli.d        t1,       a4,     1

    pixel_sad_x3_16x2_core_lasx xr20, xr21, xr22

.rept 7

    addi.d        a0,       a0,     FENC_STRIDE<<1
    add.d         a1,       a1,     t1
    add.d         a2,       a2,     t1
    add.d         a3,       a3,     t1

    pixel_sad_x3_16x2_core_lasx xr0, xr1, xr2

    xvadd.h       xr20,     xr0,    xr20
    xvadd.h       xr21,     xr1,    xr21
    xvadd.h       xr22,     xr2,    xr22
.endr

.irp i, xr20, xr21, xr22
    xvinsgr2vr.w  \i,       zero,   3
    xvinsgr2vr.w  \i,       zero,   7
.endr

    xvhaddw.w.h   xr20,     xr20,   xr20
    xvhaddw.w.h   xr21,     xr21,   xr21
    xvhaddw.w.h   xr22,     xr22,   xr22
    xvhaddw.d.w   xr20,     xr20,   xr20
    xvhaddw.d.w   xr21,     xr21,   xr21
    xvhaddw.d.w   xr22,     xr22,   xr22
    xvhaddw.q.d   xr20,     xr20,   xr20
    xvhaddw.q.d   xr21,     xr21,   xr21
    xvhaddw.q.d   xr22,     xr22,   xr22
    xvpermi.q     xr3,      xr20,   0x01
    xvpermi.q     xr4,      xr21,   0x01
    xvpermi.q     xr5,      xr22,   0x01
    vadd.w        vr20,     vr3,    vr20
    vadd.w        vr21,     vr4,    vr21
    vadd.w        vr22,     vr5,    vr22

    vstelm.w      vr20,     a5,     0,    0
    vstelm.w      vr21,     a5,     4,    0
    vstelm.w      vr22,     a5,     8,    0
endfunc

.macro pixel_sad_x3_32x2_core_lasx out0, out1, out2
    xvld          xr0,      a0,    0
    xvld          xr1,      a0,    FENC_STRIDE
    xvld          xr2,      a1,    0
    xvldx         xr3,      a1,    a4
    xvld          xr4,      a2,    0
    xvldx         xr5,      a2,    a4
    xvld          xr6,      a3,    0
    xvldx         xr7,      a3,    a4

    xvabsd.bu     xr8,      xr0,   xr2
    xvabsd.bu     xr9,      xr0,   xr4
    xvabsd.bu     xr10,     xr0,   xr6
    xvabsd.bu     xr11,     xr1,   xr3
    xvabsd.bu     xr12,     xr1,   xr5
    xvabsd.bu     xr13,     xr1,   xr7
.irp i, xr8, xr9, xr10,  xr11, xr12, xr13
    xvhaddw.hu.bu \i,       \i,    \i
.endr
    xvadd.h       \out0,    xr8,   xr11
    xvadd.h       \out1,    xr9,   xr12
    xvadd.h       \out2,    xr10,  xr13
.endm

.macro pixel_sad_x3_32x_lasx h
function x265_pixel_sad_x3_32x\h\()_lasx
    slli.d        t1,       a4,     1

    pixel_sad_x3_32x2_core_lasx xr20, xr21, xr22

.rept (\h>>1)-1
    addi.d        a0,       a0,     FENC_STRIDE<<1
    add.d         a1,       a1,     t1
    add.d         a2,       a2,     t1
    add.d         a3,       a3,     t1

    pixel_sad_x3_32x2_core_lasx xr0, xr1, xr2

    xvadd.h       xr20,     xr0,    xr20
    xvadd.h       xr21,     xr1,    xr21
    xvadd.h       xr22,     xr2,    xr22
.endr

    xvhaddw.w.h   xr20,     xr20,   xr20
    xvhaddw.w.h   xr21,     xr21,   xr21
    xvhaddw.w.h   xr22,     xr22,   xr22
    xvhaddw.d.w   xr20,     xr20,   xr20
    xvhaddw.d.w   xr21,     xr21,   xr21
    xvhaddw.d.w   xr22,     xr22,   xr22
    xvhaddw.q.d   xr20,     xr20,   xr20
    xvhaddw.q.d   xr21,     xr21,   xr21
    xvhaddw.q.d   xr22,     xr22,   xr22
    xvpermi.q     xr3,      xr20,   0x01
    xvpermi.q     xr4,      xr21,   0x01
    xvpermi.q     xr5,      xr22,   0x01
    vadd.w        vr20,     vr3,    vr20
    vadd.w        vr21,     vr4,    vr21
    vadd.w        vr22,     vr5,    vr22

    vstelm.w      vr20,     a5,     0,    0
    vstelm.w      vr21,     a5,     4,    0
    vstelm.w      vr22,     a5,     8,    0
endfunc
.endm

pixel_sad_x3_32x_lasx 8
pixel_sad_x3_32x_lasx 16
pixel_sad_x3_32x_lasx 24
pixel_sad_x3_32x_lasx 32
pixel_sad_x3_32x_lasx 64

function x265_pixel_sad_x3_24x32_lasx
    slli.d        t1,       a4,     1

    pixel_sad_x3_32x2_core_lasx xr20, xr21, xr22

.rept 15
    addi.d        a0,       a0,     FENC_STRIDE<<1
    add.d         a1,       a1,     t1
    add.d         a2,       a2,     t1
    add.d         a3,       a3,     t1

    pixel_sad_x3_32x2_core_lasx xr0, xr1, xr2

    xvadd.h       xr20,     xr0,    xr20
    xvadd.h       xr21,     xr1,    xr21
    xvadd.h       xr22,     xr2,    xr22
.endr

    xvinsgr2vr.d  xr20,     zero,  3
    xvinsgr2vr.d  xr21,     zero,  3
    xvinsgr2vr.d  xr22,     zero,  3
    xvhaddw.w.h   xr20,     xr20,   xr20
    xvhaddw.w.h   xr21,     xr21,   xr21
    xvhaddw.w.h   xr22,     xr22,   xr22
    xvhaddw.d.w   xr20,     xr20,   xr20
    xvhaddw.d.w   xr21,     xr21,   xr21
    xvhaddw.d.w   xr22,     xr22,   xr22
    xvhaddw.q.d   xr20,     xr20,   xr20
    xvhaddw.q.d   xr21,     xr21,   xr21
    xvhaddw.q.d   xr22,     xr22,   xr22
    xvpermi.q     xr3,      xr20,   0x01
    xvpermi.q     xr4,      xr21,   0x01
    xvpermi.q     xr5,      xr22,   0x01
    vadd.w        vr20,     vr3,    vr20
    vadd.w        vr21,     vr4,    vr21
    vadd.w        vr22,     vr5,    vr22

    vstelm.w      vr20,     a5,     0,    0
    vstelm.w      vr21,     a5,     4,    0
    vstelm.w      vr22,     a5,     8,    0
endfunc

.macro pixel_sad_x3_64x_lasx h
function x265_pixel_sad_x3_64x\h\()_lasx
    slli.d        t1,       a4,     1

    pixel_sad_x3_32x2_core_lasx xr20, xr21, xr22

    addi.d        a0,       a0,     32
    addi.d        a1,       a1,     32
    addi.d        a2,       a2,     32
    addi.d        a3,       a3,     32

    pixel_sad_x3_32x2_core_lasx xr17, xr18, xr19

.rept (\h>>1)-1
    addi.d        a0,       a0,     -32
    addi.d        a1,       a1,     -32
    addi.d        a2,       a2,     -32
    addi.d        a3,       a3,     -32

    addi.d        a0,       a0,     FENC_STRIDE<<1
    add.d         a1,       a1,     t1
    add.d         a2,       a2,     t1
    add.d         a3,       a3,     t1

    pixel_sad_x3_32x2_core_lasx xr0, xr1, xr2

    xvadd.h       xr20,     xr0,    xr20
    xvadd.h       xr21,     xr1,    xr21
    xvadd.h       xr22,     xr2,    xr22

    addi.d        a0,       a0,     32
    addi.d        a1,       a1,     32
    addi.d        a2,       a2,     32
    addi.d        a3,       a3,     32

    pixel_sad_x3_32x2_core_lasx xr0, xr1, xr2

    xvadd.h       xr17,     xr0,    xr17
    xvadd.h       xr18,     xr1,    xr18
    xvadd.h       xr19,     xr2,    xr19
.endr

.irp i, xr17, xr18, xr19, xr20, xr21, xr22
    xvhaddw.w.h   \i,       \i,     \i
.endr

    xvadd.w       xr20,     xr20, xr17
    xvadd.w       xr21,     xr21, xr18
    xvadd.w       xr22,     xr22, xr19

    xvhaddw.d.w   xr20,     xr20,   xr20
    xvhaddw.d.w   xr21,     xr21,   xr21
    xvhaddw.d.w   xr22,     xr22,   xr22
    xvhaddw.q.d   xr20,     xr20,   xr20
    xvhaddw.q.d   xr21,     xr21,   xr21
    xvhaddw.q.d   xr22,     xr22,   xr22
    xvpermi.q     xr3,      xr20,   0x01
    xvpermi.q     xr4,      xr21,   0x01
    xvpermi.q     xr5,      xr22,   0x01
    vadd.w        vr20,     vr3,    vr20
    vadd.w        vr21,     vr4,    vr21
    vadd.w        vr22,     vr5,    vr22

    vstelm.w      vr20,     a5,     0,    0
    vstelm.w      vr21,     a5,     4,    0
    vstelm.w      vr22,     a5,     8,    0
endfunc
.endm

pixel_sad_x3_64x_lasx 16
pixel_sad_x3_64x_lasx 32
pixel_sad_x3_64x_lasx 48
pixel_sad_x3_64x_lasx 64

function x265_pixel_sad_x3_48x64_lasx
    slli.d        t1,       a4,     1

    pixel_sad_x3_32x2_core_lasx xr20, xr21, xr22

    addi.d        a0,       a0,     32
    addi.d        a1,       a1,     32
    addi.d        a2,       a2,     32
    addi.d        a3,       a3,     32

    pixel_sad_x3_32x2_core_lasx xr17, xr18, xr19

.rept 31
    addi.d        a0,       a0,     -32
    addi.d        a1,       a1,     -32
    addi.d        a2,       a2,     -32
    addi.d        a3,       a3,     -32

    addi.d        a0,       a0,     FENC_STRIDE<<1
    add.d         a1,       a1,     t1
    add.d         a2,       a2,     t1
    add.d         a3,       a3,     t1

    pixel_sad_x3_32x2_core_lasx xr0, xr1, xr2

    xvadd.h       xr20,     xr0,    xr20
    xvadd.h       xr21,     xr1,    xr21
    xvadd.h       xr22,     xr2,    xr22

    addi.d        a0,       a0,     32
    addi.d        a1,       a1,     32
    addi.d        a2,       a2,     32
    addi.d        a3,       a3,     32

    pixel_sad_x3_32x2_core_lasx xr0, xr1, xr2

    xvadd.h       xr17,     xr0,    xr17
    xvadd.h       xr18,     xr1,    xr18
    xvadd.h       xr19,     xr2,    xr19
.endr

    xvreplgr2vr.w xr23,     zero
    xvpermi.q     xr17,     xr23,   0x02
    xvpermi.q     xr18,     xr23,   0x02
    xvpermi.q     xr19,     xr23,   0x02

.irp i, xr17, xr18, xr19, xr20, xr21, xr22
    xvhaddw.w.h   \i,       \i,     \i
.endr

    xvadd.w       xr20,     xr20, xr17
    xvadd.w       xr21,     xr21, xr18
    xvadd.w       xr22,     xr22, xr19
    xvhaddw.d.w   xr20,     xr20,   xr20
    xvhaddw.d.w   xr21,     xr21,   xr21
    xvhaddw.d.w   xr22,     xr22,   xr22
    xvhaddw.q.d   xr20,     xr20,   xr20
    xvhaddw.q.d   xr21,     xr21,   xr21
    xvhaddw.q.d   xr22,     xr22,   xr22
    xvpermi.q     xr3,      xr20,   0x01
    xvpermi.q     xr4,      xr21,   0x01
    xvpermi.q     xr5,      xr22,   0x01
    vadd.w        vr20,     vr3,    vr20
    vadd.w        vr21,     vr4,    vr21
    vadd.w        vr22,     vr5,    vr22

    vstelm.w      vr20,     a5,     0,    0
    vstelm.w      vr21,     a5,     4,    0
    vstelm.w      vr22,     a5,     8,    0
endfunc
