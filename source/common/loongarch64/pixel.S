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

.macro pixel_satd_8x4_core_lsx in0, in1, out
    FLDD_LOADX_4  \in0,   a1,     t0,   t1, f0, f1, f2, f3
    FLDD_LOADX_4  \in1,   a3,     t2,   t3, f4, f5, f6, f7
    vilvl.d       vr0,    vr1,    vr0
    vilvl.d       vr1,    vr3,    vr2
    vilvl.d       vr2,    vr5,    vr4
    vilvl.d       vr3,    vr7,    vr6

    vsubwev.h.bu  vr4,    vr0,    vr2
    vsubwod.h.bu  vr5,    vr0,    vr2
    vsubwev.h.bu  vr6,    vr1,    vr3
    vsubwod.h.bu  vr7,    vr1,    vr3
    vadd.h        vr0,    vr4,    vr5
    vsub.h        vr1,    vr4,    vr5
    vadd.h        vr2,    vr6,    vr7
    vsub.h        vr3,    vr6,    vr7
    vpackev.h     vr4,    vr1,    vr0
    vpackod.h     vr5,    vr1,    vr0
    vpackev.h     vr6,    vr3,    vr2
    vpackod.h     vr7,    vr3,    vr2
    vadd.h        vr8,    vr4,    vr5
    vsub.h        vr9,    vr4,    vr5
    vadd.h        vr10,   vr6,    vr7
    vsub.h        vr11,   vr6,    vr7
    vilvl.d       vr4,    vr9,    vr8
    vilvh.d       vr5,    vr9,    vr8
    vilvl.d       vr6,    vr11,   vr10
    vilvh.d       vr7,    vr11,   vr10
    vadd.h        vr8,    vr4,    vr5
    vsub.h        vr9,    vr4,    vr5
    vadd.h        vr10,   vr6,    vr7
    vsub.h        vr11,   vr6,    vr7
    vadd.h        vr0,    vr8,    vr10
    vsub.h        vr1,    vr8,    vr10
    vadd.h        vr2,    vr9,    vr11
    vsub.h        vr3,    vr9,    vr11
    vadda.h       vr4,    vr1,    vr0
    vadda.h       vr5,    vr3,    vr2
    vadd.h        \out,   vr5,    vr4
.endm


/*
 * int satd_8x4(const pixel* pix1, intptr_t stride_pix1,
 *              const pixel* pix2, intptr_t stride_pix2)
 */
function x265_pixel_satd_8x4_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     t0,     a1
    slli.d        t2,     a3,     1
    add.d         t3,     t2,     a3

    pixel_satd_8x4_core_lsx a0, a2, vr12

    vhaddw.wu.hu  vr12,   vr12,   vr12
    vhaddw.du.wu  vr12,   vr12,   vr12
    vhaddw.qu.du  vr12,   vr12,   vr12
    vpickve2gr.wu t4,     vr12,   0
    srli.d        a0,     t4,     1
endfunc

.macro pixel_satd_4x4_core_lsx out
    vilvl.w         vr1,   vr2,  vr1
    vilvl.w         vr3,   vr4,  vr3
    vilvl.d         vr1,   vr3,  vr1
    vilvl.w         vr5,   vr6,  vr5
    vilvl.w         vr7,   vr8,  vr7
    vilvl.d         vr5,   vr7,  vr5

    vsubwev.h.bu    vr9,   vr1,  vr5
    vsubwod.h.bu    vr10,  vr1,  vr5
    vadd.h          vr11,  vr9,  vr10  /* a0 + a1 */
    vsub.h          vr12,  vr9,  vr10  /* a0 - a1 */
    vpackev.h       vr9,   vr12, vr11
    vpackod.h       vr10,  vr12, vr11
    vadd.h          vr11,  vr9,  vr10  /* b0 + b1 */
    vsub.h          vr12,  vr9,  vr10  /* b0 - b1 */
    vpackev.w       vr9,   vr12, vr11
    vpackod.w       vr10,  vr12, vr11
    vadd.h          vr11,  vr9,  vr10  /* HADAMARD4 */
    vsub.h          vr12,  vr9,  vr10
    vpackev.d       vr9,   vr12, vr11
    vpackod.d       vr10,  vr12, vr11
    vadd.h          vr11,  vr9,  vr10
    vsub.h          vr12,  vr9,  vr10
    vpackev.d       vr9,   vr12, vr11
    vpackod.d       vr10,  vr12, vr11
    vadda.h         \out,  vr9,  vr10
.endm

function x265_pixel_satd_4x4_lsx
    slli.d          t2,    a1,   1
    slli.d          t3,    a3,   1
    add.d           t4,    a1,   t2
    add.d           t5,    a3,   t3

    // Load data from pix1 and pix2
    FLDS_LOADX_4    a0,    a1,   t2,  t4,  f1, f2, f3, f4
    FLDS_LOADX_4    a2,    a3,   t3,  t5,  f5, f6, f7, f8
    pixel_satd_4x4_core_lsx vr13
    vhaddw.wu.hu    vr13,  vr13, vr13
    vhaddw.du.wu    vr13,  vr13, vr13
    vhaddw.qu.du    vr13,  vr13, vr13
    vpickve2gr.wu   t5,    vr13,  0
    srli.d          a0,    t5,   1
endfunc

.macro pixel_satd_4x h
function x265_pixel_satd_4x\h\()_lsx
    slli.d        t2,     a1,     1
    slli.d        t3,     a3,     1
    add.d         t4,     a1,     t2
    add.d         t5,     a3,     t3

    // Load data from pix1 and pix2
    FLDS_LOADX_4  a0,     a1,     t2,  t4,  f1, f2, f3, f4
    FLDS_LOADX_4  a2,     a3,     t3,  t5,  f5, f6, f7, f8
    pixel_satd_4x4_core_lsx vr13
    vhaddw.wu.hu  vr13,   vr13,   vr13
.rept (\h>>2)-1
    alsl.d        a0,     a1,     a0,  2
    alsl.d        a2,     a3,     a2,  2
    FLDS_LOADX_4  a0,     a1,     t2,  t4,  f1, f2, f3, f4
    FLDS_LOADX_4  a2,     a3,     t3,  t5,  f5, f6, f7, f8
    pixel_satd_4x4_core_lsx vr14
    vhaddw.wu.hu  vr14,   vr14,   vr14
    vadd.w        vr13,   vr14,   vr13
.endr
    vhaddw.du.wu  vr13,   vr13,   vr13
    vhaddw.qu.du  vr13,   vr13,   vr13
    vpickve2gr.wu t5,     vr13,   0
    srli.d        a0,     t5,     1
endfunc
.endm

pixel_satd_4x 8
pixel_satd_4x 16
pixel_satd_4x 32

function x265_pixel_satd_4x8_lasx
    slli.d          t2,    a1,   1
    slli.d          t3,    a3,   1
    add.d           t4,    a1,   t2
    add.d           t5,    a3,   t3
    // Load data from pix1 and pix2
    LSX_LOADX_4     a0,    a1,   t2,  t4,  vr1, vr2, vr3, vr4
    LSX_LOADX_4     a2,    a3,   t3,  t5,  vr5, vr6, vr7, vr8
    vilvl.w         vr1,   vr2,  vr1
    vilvl.w         vr3,   vr4,  vr3
    vilvl.d         vr9,   vr3,  vr1
    vilvl.w         vr5,   vr6,  vr5
    vilvl.w         vr7,   vr8,  vr7
    vilvl.d         vr10,  vr7,  vr5

    slli.d          t0,    a1,   2
    slli.d          t1,    a3,   2
    add.d           a0,    a0,   t0
    add.d           a2,    a2,   t1
    // Load data from pix1 and pix2
    LSX_LOADX_4     a0,    a1,   t2,  t4,  vr1, vr2, vr3, vr4
    LSX_LOADX_4     a2,    a3,   t3,  t5,  vr5, vr6, vr7, vr8
    vilvl.w         vr1,   vr2,  vr1
    vilvl.w         vr3,   vr4,  vr3
    vilvl.d         vr1,   vr3,  vr1
    vilvl.w         vr5,   vr6,  vr5
    vilvl.w         vr7,   vr8,  vr7
    vilvl.d         vr5,   vr7,  vr5
    xvpermi.q       xr1,   xr9,  0x20
    xvpermi.q       xr5,   xr10, 0x20

    xvsubwev.h.bu   xr9,   xr1,  xr5
    xvsubwod.h.bu   xr10,  xr1,  xr5
    xvadd.h         xr11,  xr9,  xr10  /* a0 + a1 */
    xvsub.h         xr12,  xr9,  xr10  /* a0 - a1 */
    xvpackev.h      xr9,   xr12, xr11
    xvpackod.h      xr10,  xr12, xr11
    xvadd.h         xr11,  xr9,  xr10  /* b0 + b1 */
    xvsub.h         xr12,  xr9,  xr10  /* b0 - b1 */
    xvpackev.w      xr9,   xr12, xr11
    xvpackod.w      xr10,  xr12, xr11
    xvadd.h         xr11,  xr9,  xr10  /* HADAMARD4 */
    xvsub.h         xr12,  xr9,  xr10
    xvpackev.d      xr9,   xr12, xr11
    xvpackod.d      xr10,  xr12, xr11
    xvadd.h         xr11,  xr9,  xr10
    xvsub.h         xr12,  xr9,  xr10
    xvpackev.d      xr9,   xr12, xr11
    xvpackod.d      xr10,  xr12, xr11
    xvadda.h        xr9,   xr9,  xr10
    xvhaddw.wu.hu   xr9,   xr9,  xr9
    xvhaddw.du.wu   xr9,   xr9,  xr9
    xvhaddw.qu.du   xr9,   xr9,  xr9
    xvpickve2gr.wu  t6,    xr9,  0
    xvpickve2gr.wu  t7,    xr9,  4
    add.d           t6,    t6,   t7
    srli.d          a0,    t6,   1
endfunc

.macro pixel_satd_4x8_core_lasx out
    LSX_LOADX_4 a0, a1, t2, t4, vr1, vr2, vr3, vr4
    LSX_LOADX_4 a2, a3, t3, t5, vr5, vr6, vr7, vr8
    vilvl.w         vr1,     vr2,     vr1
    vilvl.w         vr3,     vr4,     vr3
    vilvl.d         vr9,     vr3,     vr1
    vilvl.w         vr5,     vr6,     vr5
    vilvl.w         vr7,     vr8,     vr7
    vilvl.d         vr10,    vr7,     vr5

    slli.d          t0,      a1,      2
    slli.d          t1,      a3,      2
    add.d           a0,      a0,      t0
    LSX_LOADX_4 a0, a1, t2, t4, vr1, vr2, vr3, vr4
    add.d           a2,      a2,      t1
    LSX_LOADX_4 a2, a3, t3, t5, vr5, vr6, vr7, vr8
    vilvl.w         vr1,     vr2,     vr1
    vilvl.w         vr3,     vr4,     vr3
    vilvl.d         vr1,     vr3,     vr1
    vilvl.w         vr5,     vr6,     vr5
    vilvl.w         vr7,     vr8,     vr7
    vilvl.d         vr5,     vr7,     vr5
    xvpermi.q       xr1,     xr9,     0x20
    xvpermi.q       xr5,     xr10,    0x20

    xvsubwev.h.bu   xr9,     xr1,     xr5
    xvsubwod.h.bu   xr10,    xr1,     xr5
    xvadd.h         xr11,    xr9,     xr10  /* a0 + a1 */
    xvsub.h         xr12,    xr9,     xr10  /* a0 - a1 */
    xvpackev.h      xr9,     xr12,    xr11
    xvpackod.h      xr10,    xr12,    xr11
    xvadd.h         xr11,    xr9,     xr10  /* b0 + b1 */
    xvsub.h         xr12,    xr9,     xr10  /* b0 - b1 */
    xvpackev.w      xr9,     xr12,    xr11
    xvpackod.w      xr10,    xr12,    xr11
    xvadd.h         xr11,    xr9,     xr10  /* HADAMARD4 */
    xvsub.h         xr12,    xr9,     xr10
    xvpackev.d      xr9,     xr12,    xr11
    xvpackod.d      xr10,    xr12,    xr11
    xvadd.h         xr11,    xr9,     xr10
    xvsub.h         xr12,    xr9,     xr10
    xvpackev.d      xr9,     xr12,    xr11
    xvpackod.d      xr10,    xr12,    xr11
    xvadda.h        \out,    xr9,     xr10
.endm

.macro pixel_satd_4x_lasx h
function x265_pixel_satd_4x\h\()_lasx
    slli.d          t2,      a1,      1
    slli.d          t3,      a3,      1
    add.d           t4,      a1,      t2
    add.d           t5,      a3,      t3

    pixel_satd_4x8_core_lasx xr13
    xvhaddw.wu.hu   xr13,    xr13,    xr13

.rept (\h>>3)-1
    add.d           a0,      a0,      t0
    add.d           a2,      a2,      t1
    pixel_satd_4x8_core_lasx xr14
    xvhaddw.wu.hu   xr14,    xr14,    xr14
    xvadd.w         xr13,    xr13,    xr14
.endr

    xvhaddw.du.wu   xr13,    xr13,    xr13
    xvhaddw.qu.du   xr13,    xr13,    xr13
    xvpickve2gr.wu  t6,      xr13,    0
    xvpickve2gr.wu  t7,      xr13,    4
    add.d           t7,      t6,      t7
    srli.d          a0,      t7,      1
endfunc
.endm

pixel_satd_4x_lasx 16
pixel_satd_4x_lasx 32

function x265_pixel_satd_8x8_lasx
    slli.d          t2,      a1,      1
    slli.d          t5,      a3,      1
    add.d           t3,      a1,      t2
    add.d           t6,      a3,      t5
    slli.d          t4,      t2,      1
    slli.d          t7,      t5,      1
    LSX_LOADX_4 a0, a1, t2, t3, vr0, vr1, vr2, vr3
    add.d           a0,      a0,      t4
    LSX_LOADX_4 a0, a1, t2, t3, vr4, vr5, vr6, vr7
    LSX_LOADX_4 a2, a3, t5, t6, vr8, vr9, vr10, vr11
    add.d           a2,      a2,      t7
    LSX_LOADX_4 a2, a3, t5, t6, vr12, vr13, vr14, vr15

    vilvl.d         vr0,     vr1,     vr0
    vilvl.d         vr1,     vr3,     vr2
    vilvl.d         vr2,     vr5,     vr4
    vilvl.d         vr3,     vr7,     vr6
    xvpermi.q       xr0,     xr2,     0x02
    xvpermi.q       xr1,     xr3,     0x02
    vilvl.d         vr2,     vr9,     vr8
    vilvl.d         vr3,     vr11,    vr10
    vilvl.d         vr4,     vr13,    vr12
    vilvl.d         vr5,     vr15,    vr14
    xvpermi.q       xr2,     xr4,     0x02
    xvpermi.q       xr3,     xr5,     0x02

    // HADAMARD4
    xvsubwev.h.bu   xr4,     xr0,     xr2
    xvsubwod.h.bu   xr5,     xr0,     xr2
    xvsubwev.h.bu   xr6,     xr1,     xr3
    xvsubwod.h.bu   xr7,     xr1,     xr3
    xvadd.h         xr0,     xr4,     xr5
    xvsub.h         xr1,     xr4,     xr5
    xvadd.h         xr2,     xr6,     xr7
    xvsub.h         xr3,     xr6,     xr7
    xvpackev.h      xr4,     xr1,     xr0
    xvpackod.h      xr5,     xr1,     xr0
    xvpackev.h      xr6,     xr3,     xr2
    xvpackod.h      xr7,     xr3,     xr2
    xvadd.h         xr0,     xr4,     xr5
    xvsub.h         xr1,     xr4,     xr5
    xvadd.h         xr2,     xr6,     xr7
    xvsub.h         xr3,     xr6,     xr7
    xvilvl.h        xr4,     xr1,     xr0
    xvilvh.h        xr5,     xr1,     xr0
    xvilvl.h        xr6,     xr3,     xr2
    xvilvh.h        xr7,     xr3,     xr2
    xvadd.h         xr0,     xr4,     xr5
    xvadd.h         xr2,     xr6,     xr7
    xvsub.h         xr1,     xr4,     xr5
    xvsub.h         xr3,     xr6,     xr7
    xvadd.h         xr4,     xr0,     xr2
    xvadd.h         xr5,     xr1,     xr3
    xvsub.h         xr6,     xr0,     xr2
    xvsub.h         xr7,     xr1,     xr3
    xvadda.h        xr0,     xr4,     xr5
    xvadda.h        xr1,     xr6,     xr7
    xvadd.h         xr0,     xr0,     xr1
    xvhaddw.wu.hu   xr0,     xr0,     xr0
    xvhaddw.du.wu   xr0,     xr0,     xr0
    xvhaddw.qu.du   xr0,     xr0,     xr0
    xvpickve2gr.wu  t0,      xr0,     0
    xvpickve2gr.wu  t1,      xr0,     4
    add.w           t0,      t0,      t1
    srli.d          a0,      t0,      1
endfunc

.macro pixel_satd_8x h
function x265_pixel_satd_8x\h\()_lsx
    slli.d          t0,      a1,      1
    add.d           t1,      t0,      a1
    slli.d          t2,      a3,      1
    add.d           t3,      t2,      a3

    pixel_satd_8x4_core_lsx a0, a2, vr12
    vhaddw.wu.hu    vr12,    vr12,    vr12

.rept (\h>>2)-1
    alsl.d          a0,      a1,      a0,      2
    alsl.d          a2,      a3,      a2,      2
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr12,    vr13
.endr

    vhaddw.du.wu    vr12,    vr12,    vr12
    vhaddw.qu.du    vr12,    vr12,    vr12
    vpickve2gr.wu   t4,      vr12,    0
    srli.d          a0,      t4,      1
endfunc
.endm

pixel_satd_8x 8
pixel_satd_8x 12
pixel_satd_8x 16
pixel_satd_8x 32
pixel_satd_8x 64

.macro pixel_satd_8x8_core_lasx out
    LSX_LOADX_4 a0, a1, t2, t3, vr0, vr1, vr2, vr3
    add.d           a0,      a0,      t4
    LSX_LOADX_4 a0, a1, t2, t3, vr4, vr5, vr6, vr7
    LSX_LOADX_4 a2, a3, t5, t6, vr8, vr9, vr10, vr11
    add.d           a2,      a2,      t7
    LSX_LOADX_4 a2, a3, t5, t6, vr12, vr13, vr14, vr15
    vilvl.d         vr0,     vr1,     vr0
    vilvl.d         vr1,     vr3,     vr2
    vilvl.d         vr2,     vr5,     vr4
    vilvl.d         vr3,     vr7,     vr6
    xvpermi.q       xr0,     xr2,     0x02
    xvpermi.q       xr1,     xr3,     0x02
    vilvl.d         vr2,     vr9,     vr8
    vilvl.d         vr3,     vr11,    vr10
    vilvl.d         vr4,     vr13,    vr12
    vilvl.d         vr5,     vr15,    vr14
    xvpermi.q       xr2,     xr4,     0x02
    xvpermi.q       xr3,     xr5,     0x02

    // HADAMARD4
    xvsubwev.h.bu   xr4,     xr0,     xr2
    xvsubwod.h.bu   xr5,     xr0,     xr2
    xvsubwev.h.bu   xr6,     xr1,     xr3
    xvsubwod.h.bu   xr7,     xr1,     xr3
    xvadd.h         xr0,     xr4,     xr5
    xvsub.h         xr1,     xr4,     xr5
    xvadd.h         xr2,     xr6,     xr7
    xvsub.h         xr3,     xr6,     xr7
    xvpackev.h      xr4,     xr1,     xr0
    xvpackod.h      xr5,     xr1,     xr0
    xvpackev.h      xr6,     xr3,     xr2
    xvpackod.h      xr7,     xr3,     xr2
    xvadd.h         xr0,     xr4,     xr5
    xvsub.h         xr1,     xr4,     xr5
    xvadd.h         xr2,     xr6,     xr7
    xvsub.h         xr3,     xr6,     xr7
    xvilvl.h        xr4,     xr1,     xr0
    xvilvh.h        xr5,     xr1,     xr0
    xvilvl.h        xr6,     xr3,     xr2
    xvilvh.h        xr7,     xr3,     xr2
    xvadd.h         xr0,     xr4,     xr5
    xvadd.h         xr2,     xr6,     xr7
    xvsub.h         xr1,     xr4,     xr5
    xvsub.h         xr3,     xr6,     xr7
    xvadd.h         xr4,     xr0,     xr2
    xvadd.h         xr5,     xr1,     xr3
    xvsub.h         xr6,     xr0,     xr2
    xvsub.h         xr7,     xr1,     xr3
    xvadda.h        xr0,     xr4,     xr5
    xvadda.h        xr1,     xr6,     xr7
    xvadd.h         \out,    xr0,     xr1
.endm

.macro pixel_satd_8x_lasx h
function x265_pixel_satd_8x\h\()_lasx
    slli.d          t2,      a1,      1
    add.d           t3,      a1,      t2
    slli.d          t4,      a1,      2
    slli.d          t5,      a3,      1
    add.d           t6,      a3,      t5
    slli.d          t7,      a3,      2

    pixel_satd_8x8_core_lasx xr16
    xvhaddw.wu.hu   xr16,    xr16,    xr16

.rept (\h>>3)-1
    add.d           a0,      a0,      t4
    add.d           a2,      a2,      t7

    pixel_satd_8x8_core_lasx xr17
    xvhaddw.wu.hu   xr17,    xr17,    xr17
    xvadd.w         xr16,    xr17,    xr16
.endr

    xvhaddw.du.wu   xr16,    xr16,    xr16
    xvhaddw.qu.du   xr16,    xr16,    xr16
    xvpickve2gr.wu  t0,      xr16,    0
    xvpickve2gr.wu  t1,      xr16,    4
    add.w           t0,      t0,      t1
    srli.d          a0,      t0,      1
endfunc
.endm

pixel_satd_8x_lasx 16
pixel_satd_8x_lasx 32
pixel_satd_8x_lasx 64

function x265_pixel_satd_16x4_lsx
    slli.d          t0,      a1,      1
    add.d           t1,      t0,      a1
    slli.d          t2,      a3,      1
    add.d           t3,      t2,      a3

    pixel_satd_8x4_core_lsx a0, a2, vr12
    vhaddw.wu.hu    vr12,    vr12,    vr12
    addi.d          t5,      a0,      8
    addi.d          t6,      a2,      8
    pixel_satd_8x4_core_lsx t5, t6, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr13,    vr12

    vhaddw.du.wu    vr12,    vr12,    vr12
    vhaddw.qu.du    vr12,    vr12,    vr12
    vpickve2gr.wu   t4,      vr12,    0
    srli.d          a0,      t4,      1
endfunc

function x265_pixel_satd_16x8_lasx
    slli.d          t2,      a1,      1
    slli.d          t3,      a3,      1
    slli.d          t4,      t2,      1
    slli.d          t5,      t3,      1
    add.d           t6,      a1,      t2
    add.d           t7,      a3,      t3

    // Load data from pix1 and pix2
    LSX_LOADX_4 a0, a1, t2, t6, vr0, vr1, vr2, vr3
    add.d           a0,      a0,      t4
    LSX_LOADX_4 a0, a1, t2, t6, vr4, vr5, vr6, vr7
    LSX_LOADX_4 a2, a3, t3, t7, vr8, vr9, vr10, vr11
    add.d           a2,      a2,      t5
    LSX_LOADX_4 a2, a3, t3, t7,  vr12, vr13, vr14, vr15
    xvpermi.q       xr0,     xr4,     0x02
    xvpermi.q       xr1,     xr5,     0x02
    xvpermi.q       xr2,     xr6,     0x02
    xvpermi.q       xr3,     xr7,     0x02
    xvpermi.q       xr8,     xr12,    0x02
    xvpermi.q       xr9,     xr13,    0x02
    xvpermi.q       xr10,    xr14,    0x02
    xvpermi.q       xr11,    xr15,    0x02

    // HADAMARD4
    xvsubwev.h.bu   xr4,     xr0,     xr8
    xvsubwod.h.bu   xr5,     xr0,     xr8
    xvsubwev.h.bu   xr6,     xr1,     xr9
    xvsubwod.h.bu   xr7,     xr1,     xr9
    xvsubwev.h.bu   xr8,     xr2,     xr10
    xvsubwod.h.bu   xr9,     xr2,     xr10
    xvsubwev.h.bu   xr12,    xr3,     xr11
    xvsubwod.h.bu   xr13,    xr3,     xr11
    xvadd.h         xr0,     xr4,     xr5
    xvsub.h         xr1,     xr4,     xr5
    xvadd.h         xr2,     xr6,     xr7
    xvsub.h         xr3,     xr6,     xr7
    xvadd.h         xr4,     xr8,     xr9
    xvsub.h         xr5,     xr8,     xr9
    xvadd.h         xr6,     xr12,    xr13
    xvsub.h         xr7,     xr12,    xr13
    xvpackev.h      xr8,     xr5,     xr4
    xvpackod.h      xr9,     xr5,     xr4
    xvpackev.h      xr10,    xr7,     xr6
    xvpackod.h      xr11,    xr7,     xr6
    xvpackev.h      xr4,     xr1,     xr0
    xvpackod.h      xr5,     xr1,     xr0
    xvpackev.h      xr6,     xr3,     xr2
    xvpackod.h      xr7,     xr3,     xr2
    xvadd.h         xr0,     xr4,     xr5
    xvsub.h         xr1,     xr4,     xr5
    xvadd.h         xr2,     xr6,     xr7
    xvsub.h         xr3,     xr6,     xr7
    xvadd.h         xr4,     xr8,     xr9
    xvsub.h         xr5,     xr8,     xr9
    xvadd.h         xr6,     xr10,    xr11
    xvsub.h         xr7,     xr10,    xr11
    xvilvl.h        xr8,     xr1,     xr0
    xvilvl.h        xr9,     xr3,     xr2
    xvilvl.h        xr10,    xr5,     xr4
    xvilvl.h        xr11,    xr7,     xr6
    xvilvh.h        xr0,     xr1,     xr0
    xvilvh.h        xr1,     xr3,     xr2
    xvilvh.h        xr2,     xr5,     xr4
    xvilvh.h        xr3,     xr7,     xr6
    xvadd.h         xr4,     xr8,     xr9
    xvadd.h         xr6,     xr10,    xr11
    xvsub.h         xr5,     xr8,     xr9
    xvsub.h         xr7,     xr10,    xr11
    xvadd.h         xr8,     xr4,     xr6
    xvadd.h         xr9,     xr5,     xr7
    xvsub.h         xr10,    xr4,     xr6
    xvsub.h         xr11,    xr5,     xr7
    xvadd.h         xr4,     xr0,     xr1
    xvadd.h         xr6,     xr2,     xr3
    xvsub.h         xr5,     xr0,     xr1
    xvsub.h         xr7,     xr2,     xr3
    xvadd.h         xr0,     xr4,     xr6
    xvadd.h         xr1,     xr5,     xr7
    xvsub.h         xr2,     xr4,     xr6
    xvsub.h         xr3,     xr5,     xr7
    xvadda.h        xr8,     xr8,     xr9
    xvadda.h        xr9,     xr10,    xr11
    xvadda.h        xr0,     xr0,     xr1
    xvadda.h        xr1,     xr2,     xr3
    xvadd.h         xr8,     xr8,     xr9
    xvadd.h         xr0,     xr0,     xr1
    xvadd.h         xr0,     xr0,     xr8

    xvhaddw.wu.hu   xr0,     xr0,     xr0
    xvhaddw.du.wu   xr0,     xr0,     xr0
    xvhaddw.qu.du   xr0,     xr0,     xr0
    xvpickve2gr.wu  t0,      xr0,     0
    xvpickve2gr.wu  t1,      xr0,     4
    add.w           t0,      t0,      t1
    srli.d          a0,      t0,      1
endfunc

.macro pixel_satd_12x_lsx h
function x265_pixel_satd_12x\h\()_lsx
    slli.d          t0,      a1,      1
    add.d           t1,      t0,      a1
    slli.d          t2,      a3,      1
    add.d           t3,      t2,      a3

    pixel_satd_8x4_core_lsx a0, a2, vr21
    vhaddw.wu.hu    vr21,    vr21,    vr21

    addi.d          t5,      a0,      8
    addi.d          t6,      a2,      8
    FLDS_LOADX_4 t5, a1, t0, t1, f1, f2, f3, f4
    FLDS_LOADX_4 t6, a3, t2, t3, f5, f6, f7, f8
    pixel_satd_4x4_core_lsx vr22
    vhaddw.wu.hu    vr22,    vr22,    vr22

.rept (\h>>2)-1
    alsl.d          a0,      a1,      a0,      2
    alsl.d          a2,      a3,      a2,      2
    pixel_satd_8x4_core_lsx a0, a2, vr14
    vhaddw.wu.hu    vr14,    vr14,    vr14

    addi.d          t5,      a0,      8
    addi.d          t6,      a2,      8
    FLDS_LOADX_4 t5, a1, t0, t1, f1, f2, f3, f4
    FLDS_LOADX_4 t6, a3, t2, t3, f5, f6, f7, f8
    pixel_satd_4x4_core_lsx vr20
    vhaddw.wu.hu    vr20,    vr20,    vr20

    vadd.w          vr21,    vr21,    vr14
    vadd.w          vr22,    vr22,    vr20
.endr

    vadd.w          vr12,    vr21,    vr22
    vhaddw.du.wu    vr12,    vr12,    vr12
    vhaddw.qu.du    vr12,    vr12,    vr12
    vpickve2gr.wu   t4,      vr12,    0
    srli.d          a0,      t4,      1
endfunc
.endm

pixel_satd_12x_lsx 16
pixel_satd_12x_lsx 32

.macro pixel_satd_16x_lsx h
function x265_pixel_satd_16x\h\()_lsx
    slli.d          t0,      a1,      1
    add.d           t1,      t0,      a1
    slli.d          t2,      a3,      1
    add.d           t3,      t2,      a3

    pixel_satd_8x4_core_lsx a0, a2, vr12
    vhaddw.wu.hu    vr12,    vr12,    vr12

    addi.d          t5,      a0,      8
    addi.d          t6,      a2,      8
    pixel_satd_8x4_core_lsx t5, t6, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13

.rept (\h>>2)-1
    alsl.d          a0,      a1,      a0,      2
    alsl.d          a2,      a3,      a2,      2
    pixel_satd_8x4_core_lsx a0, a2, vr14
    vhaddw.wu.hu    vr14,    vr14,    vr14

    addi.d          t5,      a0,      8
    addi.d          t6,      a2,      8
    pixel_satd_8x4_core_lsx t5, t6, vr15
    vhaddw.wu.hu    vr15,    vr15,    vr15

    vadd.w          vr12,    vr14,    vr12
    vadd.w          vr13,    vr15,    vr13
.endr
    vadd.w          vr12,    vr12,    vr13

    vhaddw.du.wu    vr12,    vr12,    vr12
    vhaddw.qu.du    vr12,    vr12,    vr12
    vpickve2gr.wu   t4,      vr12,    0
    srli.d          a0,      t4,      1
endfunc
.endm

pixel_satd_16x_lsx 8
pixel_satd_16x_lsx 12
pixel_satd_16x_lsx 16
pixel_satd_16x_lsx 24
pixel_satd_16x_lsx 32

function x265_pixel_satd_16x64_lsx
    slli.d          t0,      a1,      1
    add.d           t1,      t0,      a1
    slli.d          t2,      a3,      1
    add.d           t3,      t2,      a3

    pixel_satd_8x4_core_lsx a0, a2, vr12
    vhaddw.wu.hu    vr12,    vr12,    vr12

    addi.d          t5,      a0,      8
    addi.d          t6,      a2,      8
    pixel_satd_8x4_core_lsx t5, t6, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr12,    vr13

.rept 15
    alsl.d          a0,      a1,      a0,      2
    alsl.d          a2,      a3,      a2,      2
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr12,    vr13

    addi.d          t5,      a0,      8
    addi.d          t6,      a2,      8
    pixel_satd_8x4_core_lsx t5, t6, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr12,    vr13
.endr

    vhaddw.du.wu    vr12,    vr12,    vr12
    vhaddw.qu.du    vr12,    vr12,    vr12
    vpickve2gr.wu   t4,      vr12,    0
    srli.d          a0,      t4,      1
endfunc

.macro pixel_satd_24x_lsx h
function x265_pixel_satd_24x\h\()_lsx
    slli.d          t0,      a1,      1
    add.d           t1,      t0,      a1
    slli.d          t2,      a3,      1
    add.d           t3,      t2,      a3

    pixel_satd_8x4_core_lsx a0, a2, vr12
    vhaddw.wu.hu    vr12,    vr12,    vr12

.rept 2
    addi.d          a0,      a0,      8
    addi.d          a2,      a2,      8
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr12,    vr13
.endr

.rept (\h>>2)-1
    addi.d          a0,      a0,      -16
    addi.d          a2,      a2,      -16
    alsl.d          a0,      a1,      a0,      2
    alsl.d          a2,      a3,      a2,      2
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr13,    vr12

.rept 2
    addi.d          a0,      a0,      8
    addi.d          a2,      a2,      8
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr13,    vr12
.endr
.endr

    vhaddw.du.wu    vr12,    vr12,    vr12
    vhaddw.qu.du    vr12,    vr12,    vr12
    vpickve2gr.wu   t4,      vr12,    0
    srli.d          a0,      t4,      1
endfunc
.endm

pixel_satd_24x_lsx 32
pixel_satd_24x_lsx 64

.macro pixel_satd_32x8_lsx h
function x265_pixel_satd_32x\h\()_lsx
    slli.d          t0,      a1,      1
    add.d           t1,      t0,      a1
    slli.d          t2,      a3,      1
    add.d           t3,      t2,      a3

    pixel_satd_8x4_core_lsx a0, a2, vr12
    vhaddw.wu.hu    vr12,    vr12,    vr12

.rept 3
    addi.d          a0,      a0,      8
    addi.d          a2,      a2,      8
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr12,    vr13
.endr

.rept (\h>>2)-1
    addi.d          a0,      a0,      -24
    addi.d          a2,      a2,      -24
    alsl.d          a0,      a1,      a0,      2
    alsl.d          a2,      a3,      a2,      2
    pixel_satd_8x4_core_lsx a0, a2, vr14
    vhaddw.wu.hu    vr14,    vr14,    vr14
    vadd.w          vr12,    vr12,    vr14

.rept 3
    addi.d          a0,      a0,      8
    addi.d          a2,      a2,      8
    pixel_satd_8x4_core_lsx a0, a2, vr15
    vhaddw.wu.hu    vr15,    vr15,    vr15
    vadd.w          vr12,    vr12,    vr15
.endr
.endr

    vhaddw.du.wu    vr12,    vr12,    vr12
    vhaddw.qu.du    vr12,    vr12,    vr12
    vpickve2gr.wu   t4,      vr12,    0
    srli.d          a0,      t4,      1
endfunc
.endm

pixel_satd_32x8_lsx 8
pixel_satd_32x8_lsx 16

.macro pixel_satd_32x24_lsx h
function x265_pixel_satd_32x\h\()_lsx
    slli.d          t0,      a1,      1
    add.d           t1,      t0,      a1
    slli.d          t2,      a3,      1
    add.d           t3,      t2,      a3

    pixel_satd_8x4_core_lsx a0, a2, vr12
    vhaddw.wu.hu    vr12,    vr12,    vr12

.rept 3
    addi.d          a0,      a0,      8
    addi.d          a2,      a2,      8
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr12,    vr13
.endr

.rept (\h>>2)-1
    addi.d          a0,      a0,      -24
    addi.d          a2,      a2,      -24
    alsl.d          a0,      a1,      a0,      2
    alsl.d          a2,      a3,      a2,      2
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr13,    vr12

.rept 3
    addi.d          a0,      a0,      8
    addi.d          a2,      a2,      8
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr13,    vr12
.endr
.endr

    vhaddw.du.wu    vr12,    vr12,    vr12
    vhaddw.qu.du    vr12,    vr12,    vr12
    vpickve2gr.wu   t4,      vr12,    0
    srli.d          a0,      t4,      1
endfunc
.endm

pixel_satd_32x24_lsx 24
pixel_satd_32x24_lsx 32
pixel_satd_32x24_lsx 48
pixel_satd_32x24_lsx 64

function x265_pixel_satd_48x64_lsx
    slli.d          t0,      a1,      1
    add.d           t1,      t0,      a1
    slli.d          t2,      a3,      1
    add.d           t3,      t2,      a3

    pixel_satd_8x4_core_lsx a0, a2, vr12
    vhaddw.wu.hu    vr12,    vr12,    vr12

.rept 5
    addi.d          a0,      a0,      8
    addi.d          a2,      a2,      8
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr13,    vr12
.endr

.rept 15
    addi.d          a0,      a0,      -40
    addi.d          a2,      a2,      -40
    alsl.d          a0,      a1,      a0,      2
    alsl.d          a2,      a3,      a2,      2
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr13,    vr12
.rept 5
    addi.d          a0,      a0,      8
    addi.d          a2,      a2,      8
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr13,    vr12
.endr
.endr

    vhaddw.du.wu    vr12,    vr12,    vr12
    vhaddw.qu.du    vr12,    vr12,    vr12
    vpickve2gr.wu   t4,      vr12,    0
    srli.d          a0,      t4,      1
endfunc

.macro pixel_satd_64x_lsx h
function x265_pixel_satd_64x\h\()_lsx
    slli.d          t0,      a1,      1
    add.d           t1,      t0,      a1
    slli.d          t2,      a3,      1
    add.d           t3,      t2,      a3

    pixel_satd_8x4_core_lsx a0, a2, vr12
    vhaddw.wu.hu    vr12,    vr12,    vr12

.rept 7
    addi.d          a0,      a0,      8
    addi.d          a2,      a2,      8
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr12,    vr13
.endr

.rept (\h>>2)-1
    addi.d          a0,      a0,      -56
    addi.d          a2,      a2,      -56
    alsl.d          a0,      a1,      a0,      2
    alsl.d          a2,      a3,      a2,      2
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr13,    vr12

.rept 7
    addi.d          a0,      a0,      8
    addi.d          a2,      a2,      8
    pixel_satd_8x4_core_lsx a0, a2, vr13
    vhaddw.wu.hu    vr13,    vr13,    vr13
    vadd.w          vr12,    vr13,    vr12
.endr
.endr

    vhaddw.du.wu    vr12,    vr12,    vr12
    vhaddw.qu.du    vr12,    vr12,    vr12
    vpickve2gr.wu   t4,      vr12,    0
    srli.d          a0,      t4,      1
endfunc
.endm

pixel_satd_64x_lsx 16
pixel_satd_64x_lsx 32
pixel_satd_64x_lsx 48
pixel_satd_64x_lsx 64

.macro pixel_satd_12x8_core_lasx out
    LSX_LOADX_4 a0, a1, t2, t6, vr0, vr1, vr2, vr3
    add.d           t0,      a0,      t4
    LSX_LOADX_4 t0, a1, t2, t6, vr4, vr5, vr6, vr7
    LSX_LOADX_4 a2, a3, t3, t7, vr8, vr9, vr10, vr11
    add.d           t1,      a2,      t5
    LSX_LOADX_4 t1, a3, t3, t7, vr12, vr13, vr14, vr15

.irp i, vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, \
    vr8, vr9, vr10, vr11, vr12, vr13, vr14, vr15
    vinsgr2vr.w     \i,    zero,    3
.endr

    xvpermi.q       xr0,     xr4,     0x02
    xvpermi.q       xr1,     xr5,     0x02
    xvpermi.q       xr2,     xr6,     0x02
    xvpermi.q       xr3,     xr7,     0x02
    xvpermi.q       xr8,     xr12,    0x02
    xvpermi.q       xr9,     xr13,    0x02
    xvpermi.q       xr10,    xr14,    0x02
    xvpermi.q       xr11,    xr15,    0x02

    // HADAMARD4
    xvsubwev.h.bu   xr4,     xr0,     xr8
    xvsubwod.h.bu   xr5,     xr0,     xr8
    xvsubwev.h.bu   xr6,     xr1,     xr9
    xvsubwod.h.bu   xr7,     xr1,     xr9
    xvsubwev.h.bu   xr8,     xr2,     xr10
    xvsubwod.h.bu   xr9,     xr2,     xr10
    xvsubwev.h.bu   xr12,    xr3,     xr11
    xvsubwod.h.bu   xr13,    xr3,     xr11
    xvadd.h         xr0,     xr4,     xr5
    xvsub.h         xr1,     xr4,     xr5
    xvadd.h         xr2,     xr6,     xr7
    xvsub.h         xr3,     xr6,     xr7
    xvadd.h         xr4,     xr8,     xr9
    xvsub.h         xr5,     xr8,     xr9
    xvadd.h         xr6,     xr12,    xr13
    xvsub.h         xr7,     xr12,    xr13
    xvpackev.h      xr8,     xr5,     xr4
    xvpackod.h      xr9,     xr5,     xr4
    xvpackev.h      xr10,    xr7,     xr6
    xvpackod.h      xr11,    xr7,     xr6
    xvpackev.h      xr4,     xr1,     xr0
    xvpackod.h      xr5,     xr1,     xr0
    xvpackev.h      xr6,     xr3,     xr2
    xvpackod.h      xr7,     xr3,     xr2
    xvadd.h         xr0,     xr4,     xr5
    xvsub.h         xr1,     xr4,     xr5
    xvadd.h         xr2,     xr6,     xr7
    xvsub.h         xr3,     xr6,     xr7
    xvadd.h         xr4,     xr8,     xr9
    xvsub.h         xr5,     xr8,     xr9
    xvadd.h         xr6,     xr10,    xr11
    xvsub.h         xr7,     xr10,    xr11
    xvilvl.h        xr8,     xr1,     xr0
    xvilvl.h        xr9,     xr3,     xr2
    xvilvl.h        xr10,    xr5,     xr4
    xvilvl.h        xr11,    xr7,     xr6
    xvilvh.h        xr0,     xr1,     xr0
    xvilvh.h        xr1,     xr3,     xr2
    xvilvh.h        xr2,     xr5,     xr4
    xvilvh.h        xr3,     xr7,     xr6
    xvadd.h         xr4,     xr8,     xr9
    xvadd.h         xr6,     xr10,    xr11
    xvsub.h         xr5,     xr8,     xr9
    xvsub.h         xr7,     xr10,    xr11
    xvadd.h         xr8,     xr4,     xr6
    xvadd.h         xr9,     xr5,     xr7
    xvsub.h         xr10,    xr4,     xr6
    xvsub.h         xr11,    xr5,     xr7
    xvadd.h         xr4,     xr0,     xr1
    xvadd.h         xr6,     xr2,     xr3
    xvsub.h         xr5,     xr0,     xr1
    xvsub.h         xr7,     xr2,     xr3
    xvadd.h         xr0,     xr4,     xr6
    xvadd.h         xr1,     xr5,     xr7
    xvsub.h         xr2,     xr4,     xr6
    xvsub.h         xr3,     xr5,     xr7
    xvadda.h        xr8,     xr8,     xr9
    xvadda.h        xr9,     xr10,    xr11
    xvadda.h        xr0,     xr0,     xr1
    xvadda.h        xr1,     xr2,     xr3
    xvadd.h         xr8,     xr8,     xr9
    xvadd.h         xr0,     xr0,     xr1
    xvadd.h         \out,    xr0,     xr8
.endm

.macro pixel_satd_12x_lasx h
function x265_pixel_satd_12x\h\()_lasx
    slli.d          t2,      a1,      1
    slli.d          t3,      a3,      1
    slli.d          t4,      a1,      2
    slli.d          t5,      a3,      2
    add.d           t6,      a1,      t2
    add.d           t7,      a3,      t3

    pixel_satd_12x8_core_lasx xr16
    xvhaddw.wu.hu   xr16,    xr16,    xr16

.rept (\h>>3)-1
    alsl.d          a0,      t4,      a0,    1
    alsl.d          a2,      t5,      a2,    1
    pixel_satd_12x8_core_lasx xr17
    xvhaddw.wu.hu   xr17,    xr17,    xr17
    xvadd.w         xr16,    xr17,    xr16
.endr

    xvhaddw.du.wu   xr16,    xr16,    xr16
    xvhaddw.qu.du   xr16,    xr16,    xr16
    xvpickve2gr.wu  t0,      xr16,    0
    xvpickve2gr.wu  t1,      xr16,    4
    add.w           t0,      t0,      t1
    srli.d          a0,      t0,      1
endfunc
.endm

pixel_satd_12x_lasx 16
pixel_satd_12x_lasx 32

.macro pixel_satd_16x8_core_lasx out
    LSX_LOADX_4 a0, a1, t2, t6, vr0, vr1, vr2, vr3
    add.d           t0,      a0,      t4
    LSX_LOADX_4 t0, a1, t2, t6, vr4, vr5, vr6, vr7
    LSX_LOADX_4 a2, a3, t3, t7, vr8, vr9, vr10, vr11
    add.d           t1,      a2,      t5
    LSX_LOADX_4 t1, a3, t3, t7, vr12, vr13, vr14, vr15
    xvpermi.q       xr0,     xr4,     0x02
    xvpermi.q       xr1,     xr5,     0x02
    xvpermi.q       xr2,     xr6,     0x02
    xvpermi.q       xr3,     xr7,     0x02
    xvpermi.q       xr8,     xr12,    0x02
    xvpermi.q       xr9,     xr13,    0x02
    xvpermi.q       xr10,    xr14,    0x02
    xvpermi.q       xr11,    xr15,    0x02

    // HADAMARD4
    xvsubwev.h.bu   xr4,     xr0,     xr8
    xvsubwod.h.bu   xr5,     xr0,     xr8
    xvsubwev.h.bu   xr6,     xr1,     xr9
    xvsubwod.h.bu   xr7,     xr1,     xr9
    xvsubwev.h.bu   xr8,     xr2,     xr10
    xvsubwod.h.bu   xr9,     xr2,     xr10
    xvsubwev.h.bu   xr12,    xr3,     xr11
    xvsubwod.h.bu   xr13,    xr3,     xr11
    xvadd.h         xr0,     xr4,     xr5
    xvsub.h         xr1,     xr4,     xr5
    xvadd.h         xr2,     xr6,     xr7
    xvsub.h         xr3,     xr6,     xr7
    xvadd.h         xr4,     xr8,     xr9
    xvsub.h         xr5,     xr8,     xr9
    xvadd.h         xr6,     xr12,    xr13
    xvsub.h         xr7,     xr12,    xr13
    xvpackev.h      xr8,     xr5,     xr4
    xvpackod.h      xr9,     xr5,     xr4
    xvpackev.h      xr10,    xr7,     xr6
    xvpackod.h      xr11,    xr7,     xr6
    xvpackev.h      xr4,     xr1,     xr0
    xvpackod.h      xr5,     xr1,     xr0
    xvpackev.h      xr6,     xr3,     xr2
    xvpackod.h      xr7,     xr3,     xr2
    xvadd.h         xr0,     xr4,     xr5
    xvsub.h         xr1,     xr4,     xr5
    xvadd.h         xr2,     xr6,     xr7
    xvsub.h         xr3,     xr6,     xr7
    xvadd.h         xr4,     xr8,     xr9
    xvsub.h         xr5,     xr8,     xr9
    xvadd.h         xr6,     xr10,    xr11
    xvsub.h         xr7,     xr10,    xr11
    xvilvl.h        xr8,     xr1,     xr0
    xvilvl.h        xr9,     xr3,     xr2
    xvilvl.h        xr10,    xr5,     xr4
    xvilvl.h        xr11,    xr7,     xr6
    xvilvh.h        xr0,     xr1,     xr0
    xvilvh.h        xr1,     xr3,     xr2
    xvilvh.h        xr2,     xr5,     xr4
    xvilvh.h        xr3,     xr7,     xr6
    xvadd.h         xr4,     xr8,     xr9
    xvadd.h         xr6,     xr10,    xr11
    xvsub.h         xr5,     xr8,     xr9
    xvsub.h         xr7,     xr10,    xr11
    xvadd.h         xr8,     xr4,     xr6
    xvadd.h         xr9,     xr5,     xr7
    xvsub.h         xr10,    xr4,     xr6
    xvsub.h         xr11,    xr5,     xr7
    xvadd.h         xr4,     xr0,     xr1
    xvadd.h         xr6,     xr2,     xr3
    xvsub.h         xr5,     xr0,     xr1
    xvsub.h         xr7,     xr2,     xr3
    xvadd.h         xr0,     xr4,     xr6
    xvadd.h         xr1,     xr5,     xr7
    xvsub.h         xr2,     xr4,     xr6
    xvsub.h         xr3,     xr5,     xr7
    xvadda.h        xr8,     xr8,     xr9
    xvadda.h        xr9,     xr10,    xr11
    xvadda.h        xr0,     xr0,     xr1
    xvadda.h        xr1,     xr2,     xr3
    xvadd.h         xr8,     xr8,     xr9
    xvadd.h         xr0,     xr0,     xr1
    xvadd.h         \out,    xr0,     xr8
.endm

.macro pixel_satd_16x_lasx h
function x265_pixel_satd_16x\h\()_lasx
    slli.d          t2,      a1,      1
    slli.d          t3,      a3,      1
    slli.d          t4,      a1,      2
    slli.d          t5,      a3,      2
    add.d           t6,      a1,      t2
    add.d           t7,      a3,      t3

    pixel_satd_16x8_core_lasx xr16
    xvhaddw.wu.hu   xr16,    xr16,    xr16

.rept (\h>>3)-1
    alsl.d          a0,      t4,      a0,    1
    alsl.d          a2,      t5,      a2,    1
    pixel_satd_16x8_core_lasx xr17
    xvhaddw.wu.hu   xr17,    xr17,    xr17
    xvadd.w         xr16,    xr17,    xr16
.endr

    xvhaddw.du.wu   xr16,    xr16,    xr16
    xvhaddw.qu.du   xr16,    xr16,    xr16
    xvpickve2gr.wu  t0,      xr16,    0
    xvpickve2gr.wu  t1,      xr16,    4
    add.w           t0,      t0,      t1
    srli.d          a0,      t0,      1
endfunc
.endm

pixel_satd_16x_lasx 16
pixel_satd_16x_lasx 24
pixel_satd_16x_lasx 32
pixel_satd_16x_lasx 64

.macro pixel_satd_24x_lasx h
function x265_pixel_satd_24x\h\()_lasx
    slli.d          t2,      a1,      1
    slli.d          t3,      a3,      1
    slli.d          t4,      a1,      2
    slli.d          t5,      a3,      2
    add.d           t6,      a1,      t2
    add.d           t7,      a3,      t3

    pixel_satd_12x8_core_lasx xr16
    addi.d          a0,      a0,      12
    addi.d          a2,      a2,      12
    pixel_satd_12x8_core_lasx xr17
    xvhaddw.wu.hu   xr16,    xr16,    xr16
    xvhaddw.wu.hu   xr17,    xr17,    xr17
    xvadd.w         xr16,    xr16,    xr17

.rept (\h>>3)-1
    addi.d  a0, a0, -12
    addi.d  a2, a2, -12
    alsl.d          a0,      t4,      a0,    1
    alsl.d          a2,      t5,      a2,    1
    pixel_satd_12x8_core_lasx xr17
    xvhaddw.wu.hu   xr17,    xr17,    xr17
    xvadd.w         xr16,    xr17,    xr16
    addi.d          a0,      a0,      12
    addi.d          a2,      a2,      12
    pixel_satd_12x8_core_lasx xr17
    xvhaddw.wu.hu   xr17,    xr17,    xr17
    xvadd.w         xr16,    xr16,    xr17
.endr

    xvhaddw.du.wu   xr16,    xr16,    xr16
    xvhaddw.qu.du   xr16,    xr16,    xr16
    xvpickve2gr.wu  t0,      xr16,    0
    xvpickve2gr.wu  t1,      xr16,    4
    add.w           t0,      t0,      t1
    srli.d          a0,      t0,      1
endfunc
.endm

pixel_satd_24x_lasx 32
pixel_satd_24x_lasx 64

function x265_pixel_satd_32x8_lasx
    slli.d          t2,      a1,      1
    slli.d          t3,      a3,      1
    slli.d          t4,      a1,      2
    slli.d          t5,      a3,      2
    add.d           t6,      a1,      t2
    add.d           t7,      a3,      t3

    pixel_satd_16x8_core_lasx xr16
    xvhaddw.wu.hu   xr16,    xr16,    xr16

    addi.d          a0,      a0,      16
    addi.d          a2,      a2,      16
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18

    xvhaddw.du.wu   xr16,    xr16,    xr16
    xvhaddw.qu.du   xr16,    xr16,    xr16
    xvpickve2gr.wu  t0,      xr16,    0
    xvpickve2gr.wu  t1,      xr16,    4
    add.w           t0,      t0,      t1
    srli.d          a0,      t0,      1
endfunc

.macro pixel_satd_32x_lasx h
function x265_pixel_satd_32x\h\()_lasx
    slli.d          t2,      a1,      1
    slli.d          t3,      a3,      1
    slli.d          t4,      a1,      2
    slli.d          t5,      a3,      2
    add.d           t6,      a1,      t2
    add.d           t7,      a3,      t3

    pixel_satd_16x8_core_lasx xr16
    xvhaddw.wu.hu   xr16,    xr16,    xr16

    addi.d          a0,      a0,      16
    addi.d          a2,      a2,      16
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18

.rept (\h>>3)-1
    addi.d          a0,      a0,      -16
    addi.d          a2,      a2,      -16

    alsl.d          a0,      a1,      a0,    3
    alsl.d          a2,      a3,      a2,    3
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18

    addi.d          a0,      a0,      16
    addi.d          a2,      a2,      16
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18
.endr

    xvhaddw.du.wu   xr16,    xr16,    xr16
    xvhaddw.qu.du   xr16,    xr16,    xr16
    xvpickve2gr.wu  t0,      xr16,    0
    xvpickve2gr.wu  t1,      xr16,    4
    add.w           t0,      t0,      t1
    srli.d          a0,      t0,      1
endfunc
.endm

.macro pixel_satd_32x24_lasx h
function x265_pixel_satd_32x\h\()_lasx
    slli.d          t2,      a1,      1
    slli.d          t3,      a3,      1
    slli.d          t4,      a1,      2
    slli.d          t5,      a3,      2
    add.d           t6,      a1,      t2
    add.d           t7,      a3,      t3

    pixel_satd_16x8_core_lasx xr16
    xvhaddw.wu.hu   xr16,    xr16,    xr16

    addi.d          a0,      a0,      16
    addi.d          a2,      a2,      16
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18

.rept (\h>>3)-1
    addi.d          a0,      a0,      -16
    addi.d          a2,      a2,      -16

    alsl.d          a0,      a1,      a0,    3
    alsl.d          a2,      a3,      a2,    3
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18

    addi.d          a0,      a0,      16
    addi.d          a2,      a2,      16
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18
.endr

    xvhaddw.du.wu   xr16,    xr16,    xr16
    xvhaddw.qu.du   xr16,    xr16,    xr16
    xvpickve2gr.wu  t0,      xr16,    0
    xvpickve2gr.wu  t1,      xr16,    4
    add.w           t0,      t0,      t1
    srli.d          a0,      t0,      1
endfunc
.endm

pixel_satd_32x_lasx 16
pixel_satd_32x24_lasx 24
pixel_satd_32x24_lasx 32
pixel_satd_32x24_lasx 48

function x265_pixel_satd_32x64_lasx
    slli.d          t2,      a1,      1
    slli.d          t3,      a3,      1
    slli.d          t4,      a1,      2
    slli.d          t5,      a3,      2
    add.d           t6,      a1,      t2
    add.d           t7,      a3,      t3

    pixel_satd_16x8_core_lasx xr16
    xvhaddw.wu.hu   xr16,    xr16,    xr16

    addi.d          a0,      a0,      16
    addi.d          a2,      a2,      16
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18

.rept 7
    addi.d          a0,      a0,      -16
    addi.d          a2,      a2,      -16

    alsl.d          a0,      a1,      a0,    3
    alsl.d          a2,      a3,      a2,    3
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18

    addi.d          a0,      a0,      16
    addi.d          a2,      a2,      16
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18
.endr

    xvhaddw.du.wu   xr16,    xr16,    xr16
    xvhaddw.qu.du   xr16,    xr16,    xr16
    xvpickve2gr.wu  t0,      xr16,    0
    xvpickve2gr.wu  t1,      xr16,    4
    add.w           t0,      t0,      t1
    srli.d          a0,      t0,      1
endfunc

function x265_pixel_satd_48x64_lasx
    slli.d          t2,      a1,      1
    slli.d          t3,      a3,      1
    slli.d          t4,      a1,      2
    slli.d          t5,      a3,      2
    add.d           t6,      a1,      t2
    add.d           t7,      a3,      t3

    pixel_satd_16x8_core_lasx xr16
    xvhaddw.wu.hu   xr16,    xr16,    xr16

.rept 2
    addi.d          a0,      a0,      16
    addi.d          a2,      a2,      16
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18
.endr

.rept 7
    addi.d          a0,      a0,      -32
    addi.d          a2,      a2,      -32

    alsl.d          a0,      a1,      a0,    3
    alsl.d          a2,      a3,      a2,    3
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18

.rept 2
    addi.d          a0,      a0,      16
    addi.d          a2,      a2,      16
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18
.endr
.endr

    xvhaddw.du.wu   xr16,    xr16,    xr16
    xvhaddw.qu.du   xr16,    xr16,    xr16
    xvpickve2gr.wu  t0,      xr16,    0
    xvpickve2gr.wu  t1,      xr16,    4
    add.w           t0,      t0,      t1
    srli.d          a0,      t0,      1
endfunc

.macro pixel_satd_64x_lasx h
function x265_pixel_satd_64x\h\()_lasx
    slli.d          t2,      a1,      1
    slli.d          t3,      a3,      1
    slli.d          t4,      a1,      2
    slli.d          t5,      a3,      2
    add.d           t6,      a1,      t2
    add.d           t7,      a3,      t3

    pixel_satd_16x8_core_lasx xr16
    xvhaddw.wu.hu   xr16,    xr16,    xr16

.rept 3
    addi.d          a0,      a0,      16
    addi.d          a2,      a2,      16
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18
.endr

.rept (\h>>3)-1
    addi.d          a0,      a0,      -48
    addi.d          a2,      a2,      -48

    alsl.d          a0,      a1,      a0,    3
    alsl.d          a2,      a3,      a2,    3
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18

.rept 3
    addi.d          a0,      a0,      16
    addi.d          a2,      a2,      16
    pixel_satd_16x8_core_lasx xr18
    xvhaddw.wu.hu   xr18,    xr18,    xr18
    xvadd.w         xr16,    xr16,    xr18
.endr
.endr

    xvhaddw.du.wu   xr16,    xr16,    xr16
    xvhaddw.qu.du   xr16,    xr16,    xr16
    xvpickve2gr.wu  t0,      xr16,    0
    xvpickve2gr.wu  t1,      xr16,    4
    add.w           t0,      t0,      t1
    srli.d          a0,      t0,      1
endfunc
.endm

pixel_satd_64x_lasx 16
pixel_satd_64x_lasx 32
pixel_satd_64x_lasx 48
pixel_satd_64x_lasx 64

.macro pixel_sa8d_8x8_core_lsx out0, out1, out2, out3
    FLDD_LOADX_4  a0,     a1,     t0, t1, f0, f1, f2, f3
    FLDD_LOADX_4  a2,     a3,     t2, t3, f4, f5, f6, f7
    vilvl.d       vr0,    vr1,    vr0
    vilvl.d       vr1,    vr3,    vr2
    vilvl.d       vr4,    vr5,    vr4
    vilvl.d       vr5,    vr7,    vr6
    vsubwev.h.bu  vr2,    vr0,    vr4
    vsubwod.h.bu  vr3,    vr0,    vr4
    vsubwev.h.bu  vr6,    vr1,    vr5
    vsubwod.h.bu  vr7,    vr1,    vr5
    vadd.h        vr8,    vr2,    vr3
    vsub.h        vr9,    vr2,    vr3
    vadd.h        vr10,   vr6,    vr7
    vsub.h        vr11,   vr6,    vr7
    vpackev.h     vr0,    vr9,    vr8
    vpackod.h     vr1,    vr9,    vr8
    vpackev.h     vr2,    vr11,   vr10
    vpackod.h     vr3,    vr11,   vr10
    vadd.h        vr4,    vr0,    vr1
    vsub.h        vr5,    vr0,    vr1
    vadd.h        vr6,    vr2,    vr3
    vsub.h        vr7,    vr2,    vr3
    vilvl.d       vr0,    vr5,    vr4
    vilvh.d       vr1,    vr5,    vr4
    vilvl.d       vr2,    vr7,    vr6
    vilvh.d       vr3,    vr7,    vr6
    vadd.h        vr4,    vr0,    vr1
    vsub.h        vr5,    vr0,    vr1
    vadd.h        vr6,    vr2,    vr3
    vsub.h        vr7,    vr2,    vr3

    // vr12 vr13 vr14 vr15
    vpickev.w     vr0,    vr5,    vr4
    vpickod.w     vr1,    vr5,    vr4
    vpickev.w     vr2,    vr7,    vr6
    vpickod.w     vr3,    vr7,    vr6
    vadd.h        vr8,    vr0,    vr1
    vsub.h        vr9,    vr0,    vr1
    vadd.h        vr10,   vr2,    vr3
    vsub.h        vr11,   vr2,    vr3
    vadd.h        vr12,   vr8,    vr10
    vadd.h        vr13,   vr9,    vr11
    vsub.h        vr14,   vr8,    vr10
    vsub.h        vr15,   vr9,    vr11

    alsl.d        t4,     a1,     a0,    2
    alsl.d        t5,     a3,     a2,    2
    FLDD_LOADX_4  t4,     a1,     t0, t1, f0, f1, f2, f3
    FLDD_LOADX_4  t5,     a3,     t2, t3, f4, f5, f6, f7
    vilvl.d       vr0,    vr1,    vr0
    vilvl.d       vr1,    vr3,    vr2
    vilvl.d       vr4,    vr5,    vr4
    vilvl.d       vr5,    vr7,    vr6
    vsubwev.h.bu  vr2,    vr0,    vr4
    vsubwod.h.bu  vr3,    vr0,    vr4
    vsubwev.h.bu  vr6,    vr1,    vr5
    vsubwod.h.bu  vr7,    vr1,    vr5
    vadd.h        vr8,    vr2,    vr3
    vsub.h        vr9,    vr2,    vr3
    vadd.h        vr10,   vr6,    vr7
    vsub.h        vr11,   vr6,    vr7
    vpackev.h     vr0,    vr9,    vr8
    vpackod.h     vr1,    vr9,    vr8
    vpackev.h     vr2,    vr11,   vr10
    vpackod.h     vr3,    vr11,   vr10
    vadd.h        vr4,    vr0,    vr1
    vsub.h        vr5,    vr0,    vr1
    vadd.h        vr6,    vr2,    vr3
    vsub.h        vr7,    vr2,    vr3
    vilvl.d       vr0,    vr5,    vr4
    vilvh.d       vr1,    vr5,    vr4
    vilvl.d       vr2,    vr7,    vr6
    vilvh.d       vr3,    vr7,    vr6
    vadd.h        vr4,    vr0,    vr1
    vsub.h        vr5,    vr0,    vr1
    vadd.h        vr6,    vr2,    vr3
    vsub.h        vr7,    vr2,    vr3

    // vr4 vr5 vr6 vr7
    vpickev.w     vr0,    vr5,    vr4
    vpickod.w     vr1,    vr5,    vr4
    vpickev.w     vr2,    vr7,    vr6
    vpickod.w     vr3,    vr7,    vr6
    vadd.h        vr8,    vr0,    vr1
    vsub.h        vr9,    vr0,    vr1
    vadd.h        vr10,   vr2,    vr3
    vsub.h        vr11,   vr2,    vr3
    vadd.h        vr4,    vr8,    vr10
    vadd.h        vr5,    vr9,    vr11
    vsub.h        vr6,    vr8,    vr10
    vsub.h        vr7,    vr9,    vr11

    vadd.h        vr0,    vr12,   vr4
    vadd.h        vr1,    vr13,   vr5
    vadd.h        vr2,    vr14,   vr6
    vadd.h        vr3,    vr15,   vr7
    vsub.h        vr8,    vr12,   vr4
    vsub.h        vr9,    vr13,   vr5
    vsub.h        vr10,   vr14,   vr6
    vsub.h        vr11,   vr15,   vr7
    vadda.h       \out0,  vr0,    vr8
    vadda.h       \out1,  vr1,    vr9
    vadda.h       \out2,  vr2,    vr10
    vadda.h       \out3,  vr3,    vr11
.endm

function x265_pixel_sa8d_8x8_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     t0,     a1
    slli.d        t2,     a3,     1
    add.d         t3,     t2,     a3
    pixel_sa8d_8x8_core_lsx vr0, vr1, vr2, vr3
    vadd.h        vr0,    vr0,    vr1
    vadd.h        vr1,    vr2,    vr3
    vadd.h        vr17,   vr0,    vr1
    vhaddw.wu.hu  vr17,   vr17,   vr17
    vhaddw.d.w    vr17,   vr17,   vr17
    vhaddw.q.d    vr17,   vr17,   vr17
    vpickve2gr.w  t5,     vr17,   0
    addi.d        t5,     t5,     2
    srli.d        a0,     t5,     2
endfunc

function x265_pixel_sa8d_8x16_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     t0,     a1
    slli.d        t2,     a3,     1
    add.d         t3,     t2,     a3

    pixel_sa8d_8x8_core_lsx vr0, vr1, vr2, vr3
    vadd.h        vr0,    vr0,    vr1
    vadd.h        vr1,    vr2,    vr3
    vadd.h        vr17,   vr0,    vr1
    vhaddw.wu.hu  vr17,   vr17,   vr17
    vhaddw.d.w    vr17,   vr17,   vr17
    vhaddw.q.d    vr17,   vr17,   vr17
    vpickve2gr.w  t6,     vr17,   0
    addi.d        t6,     t6,     2
    srli.d        t6,     t6,     2

    alsl.d        a0,     a1,     a0,    3
    alsl.d        a2,     a3,     a2,    3

    pixel_sa8d_8x8_core_lsx vr0, vr1, vr2, vr3
    vadd.h        vr0,    vr0,    vr1
    vadd.h        vr1,    vr2,    vr3
    vadd.h        vr17,   vr0,    vr1
    vhaddw.wu.hu  vr17,   vr17,   vr17
    vhaddw.d.w    vr17,   vr17,   vr17
    vhaddw.q.d    vr17,   vr17,   vr17
    vpickve2gr.w  t7,     vr17,   0
    addi.d        t7,     t7,     2
    srli.d        t7,     t7,     2

    add.d         a0,     t7,     t6
endfunc

.macro pixel_sa8d_16x16_core_lsx out
    pixel_sa8d_8x8_core_lsx vr0, vr1, vr2, vr3
    vadd.h        vr0,    vr0,    vr1
    vadd.h        vr1,    vr2,    vr3
    vadd.h        vr16,   vr0,    vr1

    addi.d        a0,     t6,     8
    addi.d        a2,     t7,     8
    pixel_sa8d_8x8_core_lsx vr0, vr1, vr2, vr3
    vadd.h        vr0,    vr0,    vr1
    vadd.h        vr1,    vr2,    vr3
    vadd.h        vr17,   vr0,    vr1

    alsl.d        a0,     a1,     t6,   3
    alsl.d        a2,     a3,     t7,   3
    pixel_sa8d_8x8_core_lsx vr0, vr1, vr2, vr3
    vadd.h        vr0,    vr0,    vr1
    vadd.h        vr1,    vr2,    vr3
    vadd.h        vr18,   vr0,    vr1

    addi.d        a0,     a0,     8
    addi.d        a2,     a2,     8
    pixel_sa8d_8x8_core_lsx vr0, vr1, vr2, vr3
    vadd.h        vr0,    vr0,    vr1
    vadd.h        vr1,    vr2,    vr3
    vadd.h        vr19,   vr0,    vr1

    vhaddw.wu.hu  vr16,   vr16,   vr16
    vhaddw.wu.hu  vr17,   vr17,   vr17
    vhaddw.wu.hu  vr18,   vr18,   vr18
    vhaddw.wu.hu  vr19,   vr19,   vr19
    vadd.w        vr16,   vr17,   vr16
    vadd.w        vr18,   vr19,   vr18
    vadd.w        vr17,   vr18,   vr16
    vhaddw.d.w    vr17,   vr17,   vr17
    vhaddw.q.d    vr17,   vr17,   vr17
    vpickve2gr.w  t4,     vr17,   0
    addi.d        t4,     t4,     2
    srli.d        \out,   t4,     2
.endm

function x265_pixel_sa8d_16x16_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     t0,     a1
    slli.d        t2,     a3,     1
    add.d         t3,     t2,     a3
    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx a0
endfunc

function x265_pixel_sa8d_16x32_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     t0,     a1
    slli.d        t2,     a3,     1
    add.d         t3,     t2,     a3
    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx t8

    alsl.d        t6,     a1,     t6,     3
    alsl.d        t7,     a3,     t7,     3
    alsl.d        t6,     a1,     t6,     3
    alsl.d        t7,     a3,     t7,     3
    add.d         a0,     t6,     zero
    add.d         a2,     t7,     zero
    pixel_sa8d_16x16_core_lsx t5
    add.d         a0,     t8,     t5
endfunc

function x265_pixel_sa8d_32x32_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     t0,     a1
    slli.d        t2,     a3,     1
    add.d         t3,     t2,     a3
    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx t8

    addi.d        a0,     t6,     16
    addi.d        a2,     t7,     16
    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx t5
    add.d         t8,     t5,     t8

    addi.d        t6,     t6,     -16
    addi.d        t7,     t7,     -16
    alsl.d        t6,     a1,     t6,     3
    alsl.d        t7,     a3,     t7,     3
    alsl.d        t6,     a1,     t6,     3
    alsl.d        t7,     a3,     t7,     3

    add.d         a0,     t6,     zero
    add.d         a2,     t7,     zero
    pixel_sa8d_16x16_core_lsx t5
    add.d         t8,     t8,     t5

    addi.d        a0,     t6,     16
    addi.d        a2,     t7,     16
    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx t5
    add.d         a0,     t8,     t5
endfunc

function x265_pixel_sa8d_32x64_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     t0,     a1
    slli.d        t2,     a3,     1
    add.d         t3,     t2,     a3

    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx t8

    addi.d        a0,     t6,     16
    addi.d        a2,     t7,     16
    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx t5
    add.d         t8,     t5,     t8

.rept 3
    addi.d        t6,     t6,     -16
    addi.d        t7,     t7,     -16

    alsl.d        t6,     a1,     t6,     3
    alsl.d        t7,     a3,     t7,     3
    alsl.d        a0,     a1,     t6,     3
    alsl.d        a2,     a3,     t7,     3

    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx t5
    add.d         t8,     t5,     t8

    addi.d        a0,     t6,     16
    addi.d        a2,     t7,     16
    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx t5
    add.d         t8,     t5,     t8
.endr
    add.d         a0,     t8,     zero

endfunc

function x265_pixel_sa8d_64x64_lsx
    slli.d        t0,     a1,     1
    add.d         t1,     t0,     a1
    slli.d        t2,     a3,     1
    add.d         t3,     t2,     a3

    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx t8

.rept 3
    addi.d        a0,     t6,     16
    addi.d        a2,     t7,     16
    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx t5
    add.d         t8,     t5,     t8
.endr

.rept 3
    addi.d        t6,     t6,     -48
    addi.d        t7,     t7,     -48

    alsl.d        t6,     a1,     t6,     3
    alsl.d        t7,     a3,     t7,     3
    alsl.d        a0,     a1,     t6,     3
    alsl.d        a2,     a3,     t7,     3

    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx t5
    add.d         t8,     t5,     t8

.rept 3
    addi.d        a0,     t6,     16
    addi.d        a2,     t7,     16
    add.d         t6,     a0,     zero
    add.d         t7,     a2,     zero
    pixel_sa8d_16x16_core_lsx t5
    add.d         t8,     t5,     t8
.endr
.endr

    add.d         a0,     t8,     zero
endfunc

.macro pixel_sa8d_8x4_lasx out0, out1
    // Load data from pix1 and pix2
    FLDD_LOADX_4    a0,    a1,   t2,  t4,  f1, f2, f3, f4
    FLDD_LOADX_4    a2,    a3,   t3,  t5,  f5, f6, f7, f8
    vilvl.d         vr1,   vr2,  vr1
    vilvl.d         vr3,   vr4,  vr3
    vilvl.d         vr5,   vr6,  vr5
    vilvl.d         vr7,   vr8,  vr7
    xvpermi.q       xr1,   xr3,  0x02
    xvpermi.q       xr5,   xr7,  0x02
    xvsubwev.h.bu   xr9,   xr1,  xr5
    xvsubwod.h.bu   xr10,  xr1,  xr5
    xvadd.h         xr11,  xr9,  xr10  /* a0 + a1 */
    xvsub.h         xr12,  xr9,  xr10  /* a0 - a1 */
    xvpackev.h      xr9,   xr12, xr11
    xvpackod.h      xr10,  xr12, xr11
    xvadd.h         xr11,  xr9,  xr10  /* HADAMARD4 */
    xvsub.h         xr12,  xr9,  xr10
    xvpackev.w      xr9,   xr12, xr11
    xvpackod.w      xr10,  xr12, xr11
    xvadd.h         xr11,  xr9,  xr10
    xvsub.h         xr12,  xr9,  xr10
    xvpackev.d      xr9,   xr12, xr11
    xvpackod.d      xr10,  xr12, xr11
    xvadd.h         xr11,  xr9,  xr10  /* HADAMARD4 */
    xvsub.h         xr12,  xr9,  xr10
    xvor.v          xr13,  xr11, xr11
    xvor.v          xr14,  xr12, xr12
    xvpermi.q       xr11,  xr12, 0x02
    xvpermi.q       xr13,  xr14, 0x13
    xvadd.h         \out0, xr11, xr13
    xvsub.h         \out1, xr11, xr13
.endm

.macro pixel_sa8d_8x8_core_lasx out
    pixel_sa8d_8x4_lasx xr15, xr16

    alsl.d          a0,    a1,   a0,   2
    alsl.d          a2,    a3,   a2,   2

    pixel_sa8d_8x4_lasx xr9, xr10

    xvadd.h         xr17,  xr15, xr9
    xvadd.h         xr18,  xr16, xr10
    xvsub.h         xr19,  xr15, xr9
    xvsub.h         xr20,  xr16, xr10
    xvadda.h        xr17,  xr17, xr18
    xvadda.h        xr19,  xr19, xr20
    xvadd.h         xr17,  xr17, xr19

    xvhaddw.wu.hu   xr17,  xr17, xr17
    xvhaddw.d.w     xr17,  xr17, xr17
    xvhaddw.q.d     xr17,  xr17, xr17
    xvpickve2gr.w   t6,    xr17, 0
    xvpickve2gr.w   t7,    xr17, 4
    add.d           t6,    t6,   t7
    addi.d          t6,    t6,   2
    srli.d          \out,  t6,   2
.endm

function x265_pixel_sa8d_8x8_lasx
    slli.d          t2,    a1,   1
    slli.d          t3,    a3,   1
    add.d           t4,    a1,   t2
    add.d           t5,    a3,   t3

    pixel_sa8d_8x8_core_lasx a0
endfunc

function x265_pixel_sa8d_8x16_lasx
    slli.d          t2,    a1,   1
    slli.d          t3,    a3,   1
    add.d           t4,    a1,   t2
    add.d           t5,    a3,   t3

    pixel_sa8d_8x8_core_lasx t8

    alsl.d          a0,    a1,   a0,   2
    alsl.d          a2,    a3,   a2,   2

    pixel_sa8d_8x8_core_lasx a0
    add.d           a0,    a0,   t8
endfunc

.macro pixel_sa8d_16x16_core_lasx out
    pixel_sa8d_8x4_lasx xr15, xr16

    add.d           a0,    a0,   t6
    add.d           a2,    a2,   t7

    pixel_sa8d_8x4_lasx xr9, xr10

    xvadd.h         xr17,  xr15, xr9
    xvadd.h         xr18,  xr16, xr10
    xvsub.h         xr19,  xr15, xr9
    xvsub.h         xr20,  xr16, xr10
    xvadda.h        xr17,  xr17, xr18
    xvadda.h        xr19,  xr19, xr20
    xvadd.h         xr21,  xr17, xr19

    add.d           a0,    a0,   t6
    add.d           a2,    a2,   t7

    pixel_sa8d_8x4_lasx xr15, xr16

    add.d           a0,    a0,   t6
    add.d           a2,    a2,   t7

    pixel_sa8d_8x4_lasx xr9, xr10

    xvadd.h         xr17,  xr15, xr9
    xvadd.h         xr18,  xr16, xr10
    xvsub.h         xr19,  xr15, xr9
    xvsub.h         xr20,  xr16, xr10
    xvadda.h        xr17,  xr17, xr18
    xvadda.h        xr19,  xr19, xr20
    xvadd.h         xr22,  xr17, xr19

    addi.d          a0,    a0,   8
    addi.d          a2,    a2,   8
    sub.d           t6,    zero, t6
    sub.d           t7,    zero, t7

    pixel_sa8d_8x4_lasx xr15, xr16

    add.d           a0,    a0,   t6
    add.d           a2,    a2,   t7

    pixel_sa8d_8x4_lasx xr9, xr10

    xvadd.h         xr17,  xr15, xr9
    xvadd.h         xr18,  xr16, xr10
    xvsub.h         xr19,  xr15, xr9
    xvsub.h         xr20,  xr16, xr10
    xvadda.h        xr17,  xr17, xr18
    xvadda.h        xr19,  xr19, xr20
    xvadd.h         xr23,  xr17, xr19

    add.d           a0,    a0,   t6
    add.d           a2,    a2,   t7

    pixel_sa8d_8x4_lasx xr15, xr16

    add.d           a0,    a0,   t6
    add.d           a2,    a2,   t7

    pixel_sa8d_8x4_lasx xr9, xr10

    xvadd.h         xr17,  xr15, xr9
    xvadd.h         xr18,  xr16, xr10
    xvsub.h         xr19,  xr15, xr9
    xvsub.h         xr20,  xr16, xr10
    xvadda.h        xr17,  xr17, xr18
    xvadda.h        xr19,  xr19, xr20
    xvadd.h         xr24,  xr17, xr19

    xvadd.h         xr21,  xr21, xr22
    xvadd.h         xr23,  xr23, xr24
    xvhaddw.wu.hu   xr21,  xr21, xr21
    xvhaddw.wu.hu   xr23,  xr23, xr23
    xvadd.w         xr21,  xr21, xr23
    xvhaddw.du.wu   xr21,  xr21, xr21
    xvhaddw.qu.du   xr21,  xr21, xr21
    xvpickve2gr.du  t0,    xr21, 0
    xvpickve2gr.du  t1,    xr21, 2
    add.d           t0,    t0,   t1
    addi.d          t0,    t0,   2
    srli.d          \out,  t0,   2
.endm

function x265_pixel_sa8d_16x16_lasx
    addi.d          sp,    sp,  -8
    fst.d           f24,   sp,   0
    slli.d          t2,    a1,   1
    slli.d          t3,    a3,   1
    add.d           t4,    a1,   t2
    add.d           t5,    a3,   t3
    slli.d          t6,    a1,   2
    slli.d          t7,    a3,   2

    pixel_sa8d_16x16_core_lasx a0

    fld.d           f24,   sp,   0
    addi.d          sp,    sp,   8
endfunc

function x265_pixel_sa8d_16x32_lasx
    addi.d          sp,    sp,  -8
    fst.d           f24,   sp,   0
    slli.d          t2,    a1,   1
    slli.d          t3,    a3,   1
    add.d           t4,    a1,   t2
    add.d           t5,    a3,   t3
    slli.d          t6,    a1,   2
    slli.d          t7,    a3,   2

    pixel_sa8d_16x16_core_lasx t8

    addi.d          a0,    a0,   -8
    addi.d          a2,    a2,   -8
    sub.d           t6,    zero, t6
    sub.d           t7,    zero, t7

    alsl.d          a0,    t6,   a0,   2
    alsl.d          a2,    t7,   a2,   2

    pixel_sa8d_16x16_core_lasx a0

    add.d           a0,    a0,   t8
    fld.d           f24,   sp,   0
    addi.d          sp,    sp,   8
endfunc

.macro pixel_sa8d_32x32_core_lasx out
    pixel_sa8d_16x16_core_lasx t8
    sub.d           t6,    zero, t6
    sub.d           t7,    zero, t7
    addi.d          a0,    a0,   8
    addi.d          a2,    a2,   8
    pixel_sa8d_16x16_core_lasx a4
    add.d           t8,    t8,   a4

    sub.d           t6,    zero, t6
    sub.d           t7,    zero, t7
    addi.d          a0,    a0,   -24
    addi.d          a2,    a2,   -24
    alsl.d          a0,    t6,   a0,  2
    alsl.d          a2,    t7,   a2,  2
    pixel_sa8d_16x16_core_lasx a4
    add.d           t8,    t8,   a4

    sub.d           t6,    zero, t6
    sub.d           t7,    zero, t7
    addi.d          a0,    a0,   8
    addi.d          a2,    a2,   8
    pixel_sa8d_16x16_core_lasx a4
    add.d           \out,  t8,   a4
.endm

function x265_pixel_sa8d_32x32_lasx
    addi.d          sp,    sp,  -8
    fst.d           f24,   sp,   0
    slli.d          t2,    a1,   1
    slli.d          t3,    a3,   1
    add.d           t4,    a1,   t2
    add.d           t5,    a3,   t3
    slli.d          t6,    a1,   2
    slli.d          t7,    a3,   2

    pixel_sa8d_32x32_core_lasx a0

    fld.d           f24,   sp,   0
    addi.d          sp,    sp,   8
endfunc

function x265_pixel_sa8d_32x64_lasx
    addi.d          sp,    sp,  -8
    fst.d           f24,   sp,   0
    slli.d          t2,    a1,   1
    slli.d          t3,    a3,   1
    add.d           t4,    a1,   t2
    add.d           t5,    a3,   t3
    slli.d          t6,    a1,   2
    slli.d          t7,    a3,   2
    addi.d          a5,    a0,   0
    addi.d          a6,    a2,   0

    pixel_sa8d_32x32_core_lasx a7

    sub.d           t6,    zero, t6
    sub.d           t7,    zero, t7
    alsl.d          a0,    t6,   a5,   3
    alsl.d          a2,    t7,   a6,   3

    pixel_sa8d_32x32_core_lasx a0

    add.d           a0,    a0,   a7

    fld.d           f24,   sp,   0
    addi.d          sp,    sp,   8
endfunc

function x265_pixel_sa8d_64x64_lasx
    addi.d          sp,    sp,  -8
    fst.d           f24,   sp,   0
    slli.d          t2,    a1,   1
    slli.d          t3,    a3,   1
    add.d           t4,    a1,   t2
    add.d           t5,    a3,   t3
    slli.d          t6,    a1,   2
    slli.d          t7,    a3,   2
    addi.d          a5,    a0,   0
    addi.d          a6,    a2,   0

    pixel_sa8d_32x32_core_lasx a7

    sub.d           t6,    zero, t6
    sub.d           t7,    zero, t7
    alsl.d          a0,    t6,   a5,   3
    alsl.d          a2,    t7,   a6,   3

    pixel_sa8d_32x32_core_lasx t0
    add.d           a7,    t0,   a7

    sub.d           t6,    zero, t6
    sub.d           t7,    zero, t7
    addi.d          a0,    a5,   32
    addi.d          a2,    a6,   32
    addi.d          a5,    a0,   0
    addi.d          a6,    a2,   0

    pixel_sa8d_32x32_core_lasx t0
    add.d           a7,    t0,   a7

    sub.d           t6,    zero, t6
    sub.d           t7,    zero, t7
    alsl.d          a0,    t6,   a5,   3
    alsl.d          a2,    t7,   a6,   3

    pixel_sa8d_32x32_core_lasx a0
    add.d           a0,    a0,   a7

    fld.d           f24,   sp,   0
    addi.d          sp,    sp,   8
endfunc

function x265_weight_pp_lsx
    ld.d            t0,    sp,   0   // offset
    slli.w          a5,    a5,   6   // w0 << correction

    nor             t5,    a5,   a5
    clz.w           t1,    t5
    blt             t1,    a7,   .unfoldedShift

    srl.w           t1,    a5,   a7
    vreplgr2vr.b    vr0,   t1
    srl.w           a6,    a6,   a7
    add.w           a6,    a6,   t0
    vreplgr2vr.h    vr1,   a6

    mul.w           t2,    t1,   a6
    addi.w          t2,    t2,   255
    add.w           t2,    t2,   t0
    srli.w          t2,    t2,   16
    bnez            t2,    .widenTo32Bit

    // 16-bit arithmetic is enough.
.loopHpp:
    addi.w          t1,    a3,   0
    addi.d          t3,    a0,   0
    addi.d          t4,    a1,   0
.loopWpp:
    vld             vr2,   t3,   0
    vor.v           vr3,   vr1,  vr1
    vor.v           vr4,   vr1,  vr1
    addi.d          t1,    t1,   -16
    vmaddwev.h.b    vr3,   vr2,  vr0
    vmaddwod.h.b    vr4,   vr2,  vr0
    vilvl.h         vr5,   vr4,  vr3
    vilvh.h         vr6,   vr4,  vr3
    vssrarni.bu.h   vr6,   vr5,  0
    vst             vr6,   t4,   0
    addi.d          t3,    t3,   16
    addi.d          t4,    t4,   16
    bnez            t1,    .loopWpp
    add.d           a1,    a1,   a2
    add.d           a0,    a0,   a2
    addi.w          a4,    a4,   -1
    bnez            a4,    .loopHpp
    b               .weight_pp_lsx_end

    // 32-bit arithmetic is needed.
.widenTo32Bit:
.loopHpp32:
    addi.w          t1,    a3,   0
    addi.d          t3,    a0,   0
    addi.d          t4,    a1,   0
.loopWpp32:
    vor.v           vr3,   vr1,  vr1
    vor.v           vr4,   vr1,  vr1
    fld.d           f2,    t3,   0
    addi.d          t1,    t1,   -8
    vsllwil.hu.bu   vr2,   vr2,  0
    vmaddwev.w.h    vr3,   vr2,  vr0
    vmaddwod.w.h    vr4,   vr2,  vr0
    vilvl.w         vr5,   vr4,  vr3
    vilvh.w         vr6,   vr4,  vr3
    vssrarni.hu.w   vr6,   vr5,  0
    vssrarni.bu.h   vr6,   vr6,  0
    vstelm.d        vr6,   t4,   0,   0
    addi.d          t3,    t3,   8
    addi.d          t4,    t4,   8
    bnez            t1,    .loopWpp32
    add.d           a1,    a1,   a2
    add.d           a0,    a0,   a2
    addi.w          a4,    a4,   -1
    bnez            a4,    .loopHpp32
    b               .weight_pp_lsx_end

.unfoldedShift:
    vreplgr2vr.h    vr0,   a5
    vreplgr2vr.w    vr1,   a6    // round
    vreplgr2vr.w    vr10,  a7
    vreplgr2vr.w    vr11,  t0    // offset
.loopHppUS:
    addi.d          t3,    a0,   0
    addi.d          t4,    a1,   0
    addi.d          t1,    a3,   0
.loopWppUS:
    vor.v           vr3,   vr1,  vr1
    vor.v           vr4,   vr1,  vr1
    fld.d           f2,    t3,   0
    addi.d          t1,    t1,   -8
    vsllwil.hu.bu   vr2,   vr2,  0
    vmaddwev.w.h    vr3,   vr2,  vr0
    vmaddwod.w.h    vr4,   vr2,  vr0
    vilvl.w         vr5,   vr4,  vr3
    vilvh.w         vr6,   vr4,  vr3
    vsrl.w          vr5,   vr5,  vr10
    vsrl.w          vr6,   vr6,  vr10
    vadd.w          vr5,   vr5,  vr11
    vadd.w          vr6,   vr6,  vr11
    vssrarni.hu.w   vr6,   vr5,  0
    vssrarni.bu.h   vr6,   vr6,  0
    vstelm.d        vr6,   t4,   0,    0
    addi.d          t3,    t3,   8
    addi.d          t4,    t4,   8
    bnez            t1,    .loopWppUS
    add.d           a1,    a1,   a2
    add.d           a0,    a0,   a2
    addi.w          a4,    a4,   -1
    bnez            a4,    .loopHppUS

.weight_pp_lsx_end:

endfunc

function x265_pixel_sub_ps_4x4_lsx
    slli.d          a1,    a1,   1
    slli.d          t0,    a4,   1
    add.d           t1,    a4,   t0
    slli.d          t2,    a5,   1
    add.d           t3,    a5,   t2
    FLDS_LOADX_4 a2, a4, t0, t1, f0, f1, f2, f3
    FLDS_LOADX_4 a3, a5, t2, t3, f4, f5, f6, f7
    vilvl.w         vr1,   vr1,  vr0
    vilvl.w         vr3,   vr3,  vr2
    vilvl.w         vr5,   vr5,  vr4
    vilvl.w         vr7,   vr7,  vr6
    vilvl.d         vr0,   vr3,  vr1
    vilvl.d         vr2,   vr7,  vr5
    vsubwev.h.bu    vr1,   vr0,  vr2
    vsubwod.h.bu    vr3,   vr0,  vr2
    vilvl.h         vr4,   vr3,  vr1
    vilvh.h         vr6,   vr3,  vr1
    vstelm.d        vr4,   a0,   0,    0
    add.d           a0,    a0,   a1
    vstelm.d        vr4,   a0,   0,    1
    add.d           a0,    a0,   a1
    vstelm.d        vr6,   a0,   0,    0
    add.d           a0,    a0,   a1
    vstelm.d        vr6,   a0,   0,    1
endfunc

function x265_pixel_sub_ps_4x8_lsx
    slli.d          a1,    a1,   1
    slli.d          t0,    a4,   1
    add.d           t1,    a4,   t0
    slli.d          t2,    a5,   1
    add.d           t3,    a5,   t2
    FLDS_LOADX_4 a2, a4, t0, t1, f0, f1, f2, f3
    FLDS_LOADX_4 a3, a5, t2, t3, f4, f5, f6, f7
    vilvl.w         vr1,   vr1,  vr0
    vilvl.w         vr3,   vr3,  vr2
    vilvl.w         vr5,   vr5,  vr4
    vilvl.w         vr7,   vr7,  vr6
    vilvl.d         vr0,   vr3,  vr1
    vilvl.d         vr2,   vr7,  vr5
    vsubwev.h.bu    vr1,   vr0,  vr2
    vsubwod.h.bu    vr3,   vr0,  vr2
    vilvl.h         vr4,   vr3,  vr1
    vilvh.h         vr6,   vr3,  vr1
    vstelm.d        vr4,   a0,   0,    0
    add.d           a0,    a0,   a1
    vstelm.d        vr4,   a0,   0,    1
    add.d           a0,    a0,   a1
    vstelm.d        vr6,   a0,   0,    0
    add.d           a0,    a0,   a1
    vstelm.d        vr6,   a0,   0,    1
    add.d           a0,    a0,   a1

    alsl.d          a2,    a4,   a2,   2
    alsl.d          a3,    a5,   a3,   2

    FLDS_LOADX_4 a2, a4, t0, t1, f0, f1, f2, f3
    FLDS_LOADX_4 a3, a5, t2, t3, f4, f5, f6, f7
    vilvl.w         vr1,   vr1,  vr0
    vilvl.w         vr3,   vr3,  vr2
    vilvl.w         vr5,   vr5,  vr4
    vilvl.w         vr7,   vr7,  vr6
    vilvl.d         vr0,   vr3,  vr1
    vilvl.d         vr2,   vr7,  vr5
    vsubwev.h.bu    vr1,   vr0,  vr2
    vsubwod.h.bu    vr3,   vr0,  vr2
    vilvl.h         vr4,   vr3,  vr1
    vilvh.h         vr6,   vr3,  vr1
    vstelm.d        vr4,   a0,   0,    0
    add.d           a0,    a0,   a1
    vstelm.d        vr4,   a0,   0,    1
    add.d           a0,    a0,   a1
    vstelm.d        vr6,   a0,   0,    0
    add.d           a0,    a0,   a1
    vstelm.d        vr6,   a0,   0,    1
endfunc

.macro pixel_sub_ps_8x h
function x265_pixel_sub_ps_8x\h\()_lsx
    slli.d          a1,    a1,   1
    slli.d          t0,    a4,   1
    add.d           t1,    a4,   t0
    slli.d          t2,    a5,   1
    add.d           t3,    a5,   t2

.rept (\h>>2)
    FLDD_LOADX_4 a2, a4, t0, t1, f0, f1, f2, f3
    FLDD_LOADX_4 a3, a5, t2, t3, f4, f5, f6, f7
    vilvl.d         vr1,   vr1,  vr0
    vilvl.d         vr3,   vr3,  vr2
    vilvl.d         vr5,   vr5,  vr4
    vilvl.d         vr7,   vr7,  vr6
    vsubwev.h.bu    vr0,   vr1,  vr5
    vsubwod.h.bu    vr2,   vr1,  vr5
    vsubwev.h.bu    vr4,   vr3,  vr7
    vsubwod.h.bu    vr6,   vr3,  vr7
    vilvl.h         vr8,   vr2,  vr0
    vilvh.h         vr9,   vr2,  vr0
    vilvl.h         vr1,   vr6,  vr4
    vilvh.h         vr5,   vr6,  vr4
    vst             vr8,   a0,   0
    vstx            vr9,   a0,   a1
    alsl.d          a0,    a1,   a0,  1
    vst             vr1,   a0,   0
    vstx            vr5,   a0,   a1

    alsl.d          a2,    a4,   a2,  2
    alsl.d          a3,    a5,   a3,  2
    alsl.d          a0,    a1,   a0,  1
.endr
endfunc
.endm

pixel_sub_ps_8x 8
pixel_sub_ps_8x 16

.macro sub_ps_16x4_core_lsx
    LSX_LOADX_4 a2, a4, t0, t1, vr0, vr1, vr2, vr3
    LSX_LOADX_4 a3, a5, t2, t3, vr4, vr5, vr6, vr7
    vsubwev.h.bu    vr8,   vr0,  vr4
    vsubwod.h.bu    vr9,   vr0,  vr4
    vsubwev.h.bu    vr10,  vr1,  vr5
    vsubwod.h.bu    vr11,  vr1,  vr5
    vsubwev.h.bu    vr12,  vr2,  vr6
    vsubwod.h.bu    vr13,  vr2,  vr6
    vsubwev.h.bu    vr14,  vr3,  vr7
    vsubwod.h.bu    vr15,  vr3,  vr7

    vilvl.h         vr0,   vr9,  vr8
    vilvh.h         vr1,   vr9,  vr8
    vilvl.h         vr2,   vr11, vr10
    vilvh.h         vr3,   vr11, vr10
    vilvl.h         vr4,   vr13, vr12
    vilvh.h         vr5,   vr13, vr12
    vilvl.h         vr6,   vr15, vr14
    vilvh.h         vr7,   vr15, vr14

    addi.d          t4,    a0,   16
    vst             vr0,   a0,   0
    vst             vr1,   t4,   0
    vstx            vr2,   a0,   a1
    vstx            vr3,   t4,   a1
    alsl.d          a0,    a1,   a0,  1
    addi.d          t4,    a0,   16
    vst             vr4,   a0,   0
    vst             vr5,   t4,   0
    vstx            vr6,   a0,   a1
    vstx            vr7,   t4,   a1
.endm

.macro pixel_sub_ps_16x h
function x265_pixel_sub_ps_16x\h\()_lsx
    slli.d          a1,    a1,   1
    slli.d          t0,    a4,   1
    add.d           t1,    a4,   t0
    slli.d          t2,    a5,   1
    add.d           t3,    a5,   t2

.rept (\h>>2)

    sub_ps_16x4_core_lsx

    alsl.d          a2,    a4,   a2,  2
    alsl.d          a3,    a5,   a3,  2
    alsl.d          a0,    a1,   a0,  1
.endr
endfunc
.endm

pixel_sub_ps_16x 16
pixel_sub_ps_16x 32

.macro sub_ps_32x4_core_lsx
    LSX_LOADX_4 a2, a4, t0, t1, vr0, vr1, vr2, vr3
    LSX_LOADX_4 a3, a5, t2, t3, vr4, vr5, vr6, vr7
    vsubwev.h.bu    vr8,   vr0,  vr4
    vsubwod.h.bu    vr9,   vr0,  vr4
    vsubwev.h.bu    vr10,  vr1,  vr5
    vsubwod.h.bu    vr11,  vr1,  vr5
    vsubwev.h.bu    vr12,  vr2,  vr6
    vsubwod.h.bu    vr13,  vr2,  vr6
    vsubwev.h.bu    vr14,  vr3,  vr7
    vsubwod.h.bu    vr15,  vr3,  vr7

    vilvl.h         vr16,  vr9,  vr8
    vilvh.h         vr17,  vr9,  vr8
    vilvl.h         vr18,  vr11, vr10
    vilvh.h         vr19,  vr11, vr10
    vilvl.h         vr20,  vr13, vr12
    vilvh.h         vr21,  vr13, vr12
    vilvl.h         vr22,  vr15, vr14
    vilvh.h         vr23,  vr15, vr14

    addi.d          t5,    a2,   16
    addi.d          t6,    a3,   16

    LSX_LOADX_4 t5, a4, t0, t1, vr0, vr1, vr2, vr3
    LSX_LOADX_4 t6, a5, t2, t3, vr4, vr5, vr6, vr7

    vsubwev.h.bu    vr8,   vr0,  vr4
    vsubwod.h.bu    vr9,   vr0,  vr4
    vsubwev.h.bu    vr10,  vr1,  vr5
    vsubwod.h.bu    vr11,  vr1,  vr5
    vsubwev.h.bu    vr12,  vr2,  vr6
    vsubwod.h.bu    vr13,  vr2,  vr6
    vsubwev.h.bu    vr14,  vr3,  vr7
    vsubwod.h.bu    vr15,  vr3,  vr7

    vilvl.h         vr0,   vr9,  vr8
    vilvh.h         vr1,   vr9,  vr8
    vilvl.h         vr2,   vr11, vr10
    vilvh.h         vr3,   vr11, vr10
    vilvl.h         vr4,   vr13, vr12
    vilvh.h         vr5,   vr13, vr12
    vilvl.h         vr6,   vr15, vr14
    vilvh.h         vr7,   vr15, vr14

    vst             vr16,  a0,   0
    vst             vr17,  a0,   16
    vst             vr0,   a0,   32
    vst             vr1,   a0,   48
    add.d           a0,    a0,   a1
    vst             vr18,  a0,   0
    vst             vr19,  a0,   16
    vst             vr2,   a0,   32
    vst             vr3,   a0,   48
    add.d           a0,    a0,   a1
    vst             vr20,  a0,   0
    vst             vr21,  a0,   16
    vst             vr4,   a0,   32
    vst             vr5,   a0,   48
    add.d           a0,    a0,   a1
    vst             vr22,  a0,   0
    vst             vr23,  a0,   16
    vst             vr6,   a0,   32
    vst             vr7,   a0,   48
    add.d           a0,    a0,   a1
.endm

.macro pixel_sub_ps_32x h
function x265_pixel_sub_ps_32x\h\()_lsx
    slli.d          a1,    a1,   1
    slli.d          t0,    a4,   1
    add.d           t1,    a4,   t0
    slli.d          t2,    a5,   1
    add.d           t3,    a5,   t2

.rept (\h>>2)
    sub_ps_32x4_core_lsx

    alsl.d          a2,    a4,   a2,  2
    alsl.d          a3,    a5,   a3,  2
.endr
endfunc
.endm

pixel_sub_ps_32x 32
pixel_sub_ps_32x 64

.macro sub_ps_64x4_core_lsx
    LSX_LOADX_4 a2, a4, t0, t1, vr0, vr1, vr2, vr3
    LSX_LOADX_4 a3, a5, t2, t3, vr4, vr5, vr6, vr7
    vsubwev.h.bu    vr8,   vr0,  vr4
    vsubwod.h.bu    vr9,   vr0,  vr4
    vsubwev.h.bu    vr10,  vr1,  vr5
    vsubwod.h.bu    vr11,  vr1,  vr5
    vsubwev.h.bu    vr12,  vr2,  vr6
    vsubwod.h.bu    vr13,  vr2,  vr6
    vsubwev.h.bu    vr14,  vr3,  vr7
    vsubwod.h.bu    vr15,  vr3,  vr7

    vilvl.h         vr16,  vr9,  vr8
    vilvh.h         vr17,  vr9,  vr8
    vilvl.h         vr18,  vr11, vr10
    vilvh.h         vr19,  vr11, vr10
    vilvl.h         vr20,  vr13, vr12
    vilvh.h         vr21,  vr13, vr12
    vilvl.h         vr22,  vr15, vr14
    vilvh.h         vr23,  vr15, vr14

    addi.d          t5,    a2,   16
    addi.d          t6,    a3,   16

    LSX_LOADX_4 t5, a4, t0, t1, vr0, vr1, vr2, vr3
    LSX_LOADX_4 t6, a5, t2, t3, vr4, vr5, vr6, vr7

    vsubwev.h.bu    vr8,   vr0,  vr4
    vsubwod.h.bu    vr9,   vr0,  vr4
    vsubwev.h.bu    vr10,  vr1,  vr5
    vsubwod.h.bu    vr11,  vr1,  vr5
    vsubwev.h.bu    vr12,  vr2,  vr6
    vsubwod.h.bu    vr13,  vr2,  vr6
    vsubwev.h.bu    vr14,  vr3,  vr7
    vsubwod.h.bu    vr15,  vr3,  vr7

    vilvl.h         vr0,   vr9,  vr8
    vilvh.h         vr1,   vr9,  vr8
    vilvl.h         vr2,   vr11, vr10
    vilvh.h         vr3,   vr11, vr10
    vilvl.h         vr4,   vr13, vr12
    vilvh.h         vr5,   vr13, vr12
    vilvl.h         vr6,   vr15, vr14
    vilvh.h         vr7,   vr15, vr14

    addi.d          t4,    a0,   0
    vst             vr16,  t4,   0
    vst             vr17,  t4,   16
    vst             vr0,   t4,   32
    vst             vr1,   t4,   48
    add.d           t4,    t4,   a1
    vst             vr18,  t4,   0
    vst             vr19,  t4,   16
    vst             vr2,   t4,   32
    vst             vr3,   t4,   48
    add.d           t4,    t4,   a1
    vst             vr20,  t4,   0
    vst             vr21,  t4,   16
    vst             vr4,   t4,   32
    vst             vr5,   t4,   48
    add.d           t4,    t4,   a1
    vst             vr22,  t4,   0
    vst             vr23,  t4,   16
    vst             vr6,   t4,   32
    vst             vr7,   t4,   48

    addi.d          t5,    t5,   16
    addi.d          t6,    t6,   16
    addi.d          t4,    a0,   64

    LSX_LOADX_4 t5, a4, t0, t1, vr0, vr1, vr2, vr3
    LSX_LOADX_4 t6, a5, t2, t3, vr4, vr5, vr6, vr7
    vsubwev.h.bu    vr8,   vr0,  vr4
    vsubwod.h.bu    vr9,   vr0,  vr4
    vsubwev.h.bu    vr10,  vr1,  vr5
    vsubwod.h.bu    vr11,  vr1,  vr5
    vsubwev.h.bu    vr12,  vr2,  vr6
    vsubwod.h.bu    vr13,  vr2,  vr6
    vsubwev.h.bu    vr14,  vr3,  vr7
    vsubwod.h.bu    vr15,  vr3,  vr7

    vilvl.h         vr16,  vr9,  vr8
    vilvh.h         vr17,  vr9,  vr8
    vilvl.h         vr18,  vr11, vr10
    vilvh.h         vr19,  vr11, vr10
    vilvl.h         vr20,  vr13, vr12
    vilvh.h         vr21,  vr13, vr12
    vilvl.h         vr22,  vr15, vr14
    vilvh.h         vr23,  vr15, vr14

    addi.d          t5,    t5,   16
    addi.d          t6,    t6,   16

    LSX_LOADX_4 t5, a4, t0, t1, vr0, vr1, vr2, vr3
    LSX_LOADX_4 t6, a5, t2, t3, vr4, vr5, vr6, vr7

    vsubwev.h.bu    vr8,   vr0,  vr4
    vsubwod.h.bu    vr9,   vr0,  vr4
    vsubwev.h.bu    vr10,  vr1,  vr5
    vsubwod.h.bu    vr11,  vr1,  vr5
    vsubwev.h.bu    vr12,  vr2,  vr6
    vsubwod.h.bu    vr13,  vr2,  vr6
    vsubwev.h.bu    vr14,  vr3,  vr7
    vsubwod.h.bu    vr15,  vr3,  vr7
    vilvl.h         vr0,   vr9,  vr8
    vilvh.h         vr1,   vr9,  vr8
    vilvl.h         vr2,   vr11, vr10
    vilvh.h         vr3,   vr11, vr10
    vilvl.h         vr4,   vr13, vr12
    vilvh.h         vr5,   vr13, vr12
    vilvl.h         vr6,   vr15, vr14
    vilvh.h         vr7,   vr15, vr14

    vst             vr16,  t4,   0
    vst             vr17,  t4,   16
    vst             vr0,   t4,   32
    vst             vr1,   t4,   48
    add.d           t4,    t4,   a1
    vst             vr18,  t4,   0
    vst             vr19,  t4,   16
    vst             vr2,   t4,   32
    vst             vr3,   t4,   48
    add.d           t4,    t4,   a1
    vst             vr20,  t4,   0
    vst             vr21,  t4,   16
    vst             vr4,   t4,   32
    vst             vr5,   t4,   48
    add.d           t4,    t4,   a1
    vst             vr22,  t4,   0
    vst             vr23,  t4,   16
    vst             vr6,   t4,   32
    vst             vr7,   t4,   48
.endm

function x265_pixel_sub_ps_64x64_lsx
    slli.d          a1,    a1,   1
    slli.d          t0,    a4,   1
    add.d           t1,    a4,   t0
    slli.d          t2,    a5,   1
    add.d           t3,    a5,   t2

.rept 16
    sub_ps_64x4_core_lsx

    alsl.d          a2,    a4,   a2,  2
    alsl.d          a3,    a5,   a3,  2
    alsl.d          a0,    a1,   a0,  2
.endr
endfunc

.macro sub_ps_32x2_core_lasx
    xvld             xr0,   a2,   0
    xvldx            xr1,   a2,   a4
    xvld             xr2,   a3,   0
    xvldx            xr3,   a3,   a5

    xvsubwev.h.bu    xr10,  xr0,  xr2
    xvsubwod.h.bu    xr11,  xr0,  xr2
    xvsubwev.h.bu    xr12,  xr1,  xr3
    xvsubwod.h.bu    xr13,  xr1,  xr3

    xvilvl.h         xr2,   xr11, xr10
    xvilvh.h         xr3,   xr11, xr10
    xvilvl.h         xr4,   xr13, xr12
    xvilvh.h         xr5,   xr13, xr12

    xvor.v           xr6,   xr3,  xr3
    xvor.v           xr7,   xr5,  xr5
    xvpermi.q        xr3,   xr2,  0x20
    xvpermi.q        xr6,   xr2,  0x31
    xvpermi.q        xr5,   xr4,  0x20
    xvpermi.q        xr7,   xr4,  0x31

    xvst             xr3,   a0,   0
    xvst             xr6,   a0,   32
    add.d            t4,    a0,   a1
    xvst             xr5,   t4,   0
    xvst             xr7,   t4,   32
.endm

.macro pixel_sub_ps_32x_lasx h
function x265_pixel_sub_ps_32x\h\()_lasx
    slli.d          a1,    a1,   1

.rept (\h>>1)

    sub_ps_32x2_core_lasx

    alsl.d          a2,    a4,   a2,  1
    alsl.d          a3,    a5,   a3,  1
    alsl.d          a0,    a1,   a0,  1
.endr
endfunc
.endm

pixel_sub_ps_32x_lasx 32
pixel_sub_ps_32x_lasx 64

function x265_pixel_sub_ps_64x64_lasx
    slli.d           a1,    a1,   1

.rept 32

    xvld             xr0,   a2,   0
    xvldx            xr1,   a2,   a4
    xvld             xr2,   a3,   0
    xvldx            xr3,   a3,   a5

    xvsubwev.h.bu    xr10,  xr0,  xr2
    xvsubwod.h.bu    xr11,  xr0,  xr2
    xvsubwev.h.bu    xr12,  xr1,  xr3
    xvsubwod.h.bu    xr13,  xr1,  xr3

    xvilvl.h         xr2,   xr11, xr10
    xvilvh.h         xr3,   xr11, xr10
    xvilvl.h         xr4,   xr13, xr12
    xvilvh.h         xr5,   xr13, xr12

    xvor.v           xr6,   xr3,  xr3
    xvor.v           xr7,   xr5,  xr5
    xvpermi.q        xr3,   xr2,  0x20
    xvpermi.q        xr6,   xr2,  0x31
    xvpermi.q        xr5,   xr4,  0x20
    xvpermi.q        xr7,   xr4,  0x31

    add.d            t4,    a0,   a1
    xvst             xr3,   a0,   0
    xvst             xr6,   a0,   32
    xvst             xr5,   t4,   0
    xvst             xr7,   t4,   32

    addi.d           t6,    a2,   32
    addi.d           t7,    a3,   32

    xvld             xr0,   t6,   0
    xvldx            xr1,   t6,   a4
    xvld             xr2,   t7,   0
    xvldx            xr3,   t7,   a5

    xvsubwev.h.bu    xr10,  xr0,  xr2
    xvsubwod.h.bu    xr11,  xr0,  xr2
    xvsubwev.h.bu    xr12,  xr1,  xr3
    xvsubwod.h.bu    xr13,  xr1,  xr3

    xvilvl.h         xr2,   xr11, xr10
    xvilvh.h         xr3,   xr11, xr10
    xvilvl.h         xr4,   xr13, xr12
    xvilvh.h         xr5,   xr13, xr12

    xvor.v           xr6,   xr3,  xr3
    xvor.v           xr7,   xr5,  xr5
    xvpermi.q        xr3,   xr2,  0x20
    xvpermi.q        xr6,   xr2,  0x31
    xvpermi.q        xr5,   xr4,  0x20
    xvpermi.q        xr7,   xr4,  0x31

    xvst             xr3,   a0,   64
    xvst             xr6,   a0,   96
    xvst             xr5,   t4,   64
    xvst             xr7,   t4,   96

    alsl.d          a2,    a4,   a2,  1
    alsl.d          a3,    a5,   a3,  1
    alsl.d          a0,    a1,   a0,  1

.endr
endfunc

function x265_pixel_add_ps_4x4_lsx
    slli.w          a5,    a5,   1
    slli.d          t0,    a4,   1
    add.d           t1,    a4,   t0
    slli.d          t2,    a5,   1
    add.d           t3,    a5,   t2
    FLDS_LOADX_4 a2, a4, t0, t1, f0, f1, f2, f3
    FLDD_LOADX_4 a3, a5, t2, t3, f4, f5, f6, f7
    vilvl.w         vr1,   vr1,  vr0
    vilvl.w         vr3,   vr3,  vr2
    vilvl.d         vr5,   vr5,  vr4
    vilvl.d         vr7,   vr7,  vr6
    vsllwil.hu.bu   vr1,   vr1,  0
    vsllwil.hu.bu   vr3,   vr3,  0
    vadd.h          vr2,   vr5,  vr1
    vadd.h          vr4,   vr7,  vr3
    vssrarni.bu.h   vr4,   vr2,  0
    vstelm.w        vr4,   a0,   0,    0
    add.d           a0,    a0,   a1
    vstelm.w        vr4,   a0,   0,    1
    add.d           a0,    a0,   a1
    vstelm.w        vr4,   a0,   0,    2
    add.d           a0,    a0,   a1
    vstelm.w        vr4,   a0,   0,    3
endfunc

function x265_pixel_add_ps_4x8_lsx
    slli.w          a5,    a5,   1
    slli.d          t0,    a4,   1
    add.d           t1,    a4,   t0
    slli.d          t2,    a5,   1
    add.d           t3,    a5,   t2
    FLDS_LOADX_4 a2, a4, t0, t1, f0, f1, f2, f3
    FLDD_LOADX_4 a3, a5, t2, t3, f4, f5, f6, f7
    vilvl.w         vr1,   vr1,  vr0
    vilvl.w         vr3,   vr3,  vr2
    vilvl.d         vr5,   vr5,  vr4
    vilvl.d         vr7,   vr7,  vr6
    vsllwil.hu.bu   vr1,   vr1,  0
    vsllwil.hu.bu   vr3,   vr3,  0
    vadd.h          vr2,   vr5,  vr1
    vadd.h          vr4,   vr7,  vr3
    vssrarni.bu.h   vr4,   vr2,  0
    vstelm.w        vr4,   a0,   0,    0
    add.d           a0,    a0,   a1
    vstelm.w        vr4,   a0,   0,    1
    add.d           a0,    a0,   a1
    vstelm.w        vr4,   a0,   0,    2
    add.d           a0,    a0,   a1
    vstelm.w        vr4,   a0,   0,    3
    add.d           a0,    a0,   a1

    alsl.d          a2,    a4,   a2,   2
    alsl.d          a3,    a5,   a3,   2

    FLDS_LOADX_4 a2, a4, t0, t1, f0, f1, f2, f3
    FLDD_LOADX_4 a3, a5, t2, t3, f4, f5, f6, f7
    vilvl.w         vr1,   vr1,  vr0
    vilvl.w         vr3,   vr3,  vr2
    vilvl.d         vr5,   vr5,  vr4
    vilvl.d         vr7,   vr7,  vr6
    vsllwil.hu.bu   vr1,   vr1,  0
    vsllwil.hu.bu   vr3,   vr3,  0
    vadd.h          vr2,   vr5,  vr1
    vadd.h          vr4,   vr7,  vr3
    vssrarni.bu.h   vr4,   vr2,  0
    vstelm.w        vr4,   a0,   0,    0
    add.d           a0,    a0,   a1
    vstelm.w        vr4,   a0,   0,    1
    add.d           a0,    a0,   a1
    vstelm.w        vr4,   a0,   0,    2
    add.d           a0,    a0,   a1
    vstelm.w        vr4,   a0,   0,    3
    add.d           a0,    a0,   a1
endfunc

.macro pixel_add_ps_8x h
function x265_pixel_add_ps_8x\h\()_lsx
    slli.w          a5,    a5,   1
    slli.d          t0,    a4,   1
    add.d           t1,    a4,   t0
    slli.d          t2,    a5,   1
    add.d           t3,    a5,   t2
.rept (\h>>2)
    FLDD_LOADX_4 a2, a4, t0, t1, f0, f1, f2, f3
    LSX_LOADX_4 a3, a5, t2, t3, vr4, vr5, vr6, vr7
    vsllwil.hu.bu   vr0,   vr0,  0
    vsllwil.hu.bu   vr1,   vr1,  0
    vsllwil.hu.bu   vr2,   vr2,  0
    vsllwil.hu.bu   vr3,   vr3,  0
    vadd.h          vr8,   vr4,  vr0
    vadd.h          vr9,   vr5,  vr1
    vadd.h          vr10,  vr6,  vr2
    vadd.h          vr11,  vr7,  vr3
    vssrarni.bu.h   vr9,   vr8,  0
    vssrarni.bu.h   vr11,  vr10, 0
    vstelm.d        vr9,   a0,   0,    0
    add.d           a0,    a0,   a1
    vstelm.d        vr9,   a0,   0,    1
    add.d           a0,    a0,   a1
    vstelm.d        vr11,  a0,   0,    0
    add.d           a0,    a0,   a1
    vstelm.d        vr11,  a0,   0,    1
    add.d           a0,    a0,   a1

    alsl.d          a2,    a4,   a2,   2
    alsl.d          a3,    a5,   a3,   2
.endr
endfunc
.endm

pixel_add_ps_8x 8
pixel_add_ps_8x 16

.macro pixel_add_ps_16x h
function x265_pixel_add_ps_16x\h\()_lsx
    slli.w          a5,    a5,    1
    slli.d          t0,    a4,    1
    add.d           t1,    a4,    t0
    slli.d          t2,    a5,    1
    add.d           t3,    a5,    t2
    addi.d          t5,    a3,    0
.rept (\h>>2)
    LSX_LOADX_4 a2, a4, t0, t1, vr0, vr1, vr2, vr3
    LSX_LOADX_4 t5, a5, t2, t3, vr4, vr5, vr6, vr7
    addi.d          a3,    t5,    16
    LSX_LOADX_4 a3, a5, t2, t3, vr8, vr9, vr10, vr11

    vsllwil.hu.bu   vr12,  vr0,   0
    vexth.hu.bu     vr13,  vr0
    vsllwil.hu.bu   vr14,  vr1,   0
    vexth.hu.bu     vr15,  vr1
    vadd.h          vr16,  vr4,   vr12
    vadd.h          vr17,  vr8,   vr13
    vadd.h          vr18,  vr5,   vr14
    vadd.h          vr19,  vr9,   vr15
    vssrarni.bu.h   vr17,  vr16,  0
    vssrarni.bu.h   vr19,  vr18,  0
    vst             vr17,  a0,    0
    vstx            vr19,  a0,    a1

    alsl.d          a0,    a1,    a0,   1
    vsllwil.hu.bu   vr12,  vr2,   0
    vexth.hu.bu     vr13,  vr2
    vsllwil.hu.bu   vr14,  vr3,   0
    vexth.hu.bu     vr15,  vr3
    vadd.h          vr16,  vr6,   vr12
    vadd.h          vr17,  vr10,  vr13
    vadd.h          vr18,  vr7,   vr14
    vadd.h          vr19,  vr11,  vr15
    vssrarni.bu.h   vr17,  vr16,  0
    vssrarni.bu.h   vr19,  vr18,  0
    vst             vr17,  a0,    0
    vstx            vr19,  a0,    a1

    alsl.d          a0,    a1,    a0,  1
    alsl.d          a2,    a4,    a2,  2
    alsl.d          t5,    a5,    t5,  2
.endr
endfunc
.endm

pixel_add_ps_16x 16
pixel_add_ps_16x 32

.macro pixel_add_ps_32x h
function x265_pixel_add_ps_32x\h\()_lsx
    slli.w          a5,    a5,    1
.rept \h
    vld             vr0,   a2,    0
    vld             vr1,   a2,    16
    vld             vr2,   a3,    0
    vld             vr3,   a3,    16
    vld             vr4,   a3,    32
    vld             vr5,   a3,    48
    vsllwil.hu.bu   vr6,   vr0,   0
    vexth.hu.bu     vr7,   vr0
    vsllwil.hu.bu   vr8,   vr1,   0
    vexth.hu.bu     vr9,   vr1
    vadd.h          vr10,  vr2,   vr6
    vadd.h          vr11,  vr3,   vr7
    vadd.h          vr12,  vr4,   vr8
    vadd.h          vr13,  vr5,   vr9
    vssrarni.bu.h   vr11,  vr10,  0
    vssrarni.bu.h   vr13,  vr12,  0
    vst             vr11,  a0,    0
    vst             vr13,  a0,    16
    add.d           a2,    a4,    a2
    add.d           a3,    a5,    a3
    add.d           a0,    a1,    a0
.endr
endfunc
.endm

pixel_add_ps_32x 32
pixel_add_ps_32x 64

function x265_pixel_add_ps_64x64_lsx
    slli.w          a5,    a5,    1
.rept 64
    vld             vr0,   a2,    0
    vld             vr1,   a2,    16
    vld             vr2,   a2,    32
    vld             vr3,   a2,    48
    vld             vr4,   a3,    0
    vld             vr5,   a3,    16
    vld             vr6,   a3,    32
    vld             vr7,   a3,    48
    vld             vr8,   a3,    64
    vld             vr9,   a3,    80
    vld             vr10,  a3,    96
    vld             vr11,  a3,    112
    vsllwil.hu.bu   vr12,  vr0,   0
    vexth.hu.bu     vr13,  vr0
    vsllwil.hu.bu   vr14,  vr1,   0
    vexth.hu.bu     vr15,  vr1
    vsllwil.hu.bu   vr16,  vr2,   0
    vexth.hu.bu     vr17,  vr2
    vsllwil.hu.bu   vr18,  vr3,   0
    vexth.hu.bu     vr19,  vr3
    vadd.h          vr4,   vr4,   vr12
    vadd.h          vr5,   vr5,   vr13
    vadd.h          vr6,   vr6,   vr14
    vadd.h          vr7,   vr7,   vr15
    vadd.h          vr8,   vr8,   vr16
    vadd.h          vr9,   vr9,   vr17
    vadd.h          vr10,  vr10,  vr18
    vadd.h          vr11,  vr11,  vr19
    vssrarni.bu.h   vr5,   vr4,   0
    vssrarni.bu.h   vr7,   vr6,   0
    vssrarni.bu.h   vr9,   vr8,   0
    vssrarni.bu.h   vr11,  vr10,  0
    vst             vr5,   a0,    0
    vst             vr7,   a0,    16
    vst             vr9,   a0,    32
    vst             vr11,  a0,    48

    add.d           a2,    a4,    a2
    add.d           a3,    a5,    a3
    add.d           a0,    a1,    a0
.endr
endfunc

.macro pixel_add_ps_16x_lasx h
function x265_pixel_add_ps_16x\h\()_lasx
    slli.w          a5,    a5,     1
    slli.d          t0,    a4,     1
    add.d           t1,    a4,     t0
    slli.d          t2,    a5,     1
    add.d           t3,    a5,     t2
    addi.d          t5,    a3,     0
.rept (\h>>2)
    LSX_LOADX_4 a2, a4, t0, t1, vr0, vr1, vr2, vr3
    LSX_LOADX_4 t5, a5, t2, t3, vr4, vr5, vr6, vr7
    addi.d          a3,    t5,    16
    LSX_LOADX_4 a3, a5, t2, t3, vr8, vr9, vr10, vr11

    vext2xv.hu.bu   xr12,  xr0
    vext2xv.hu.bu   xr13,  xr1
    vext2xv.hu.bu   xr14,  xr2
    vext2xv.hu.bu   xr15,  xr3
    xvpermi.q       xr8,   xr4,    0x20
    xvpermi.q       xr9,   xr5,    0x20
    xvpermi.q       xr10,  xr6,    0x20
    xvpermi.q       xr11,  xr7,    0x20
    xvadd.h         xr16,  xr8,    xr12
    xvadd.h         xr17,  xr9,    xr13
    xvadd.h         xr18,  xr10,   xr14
    xvadd.h         xr19,  xr11,   xr15
    xvssrarni.bu.h  xr17,  xr16,   0
    xvssrarni.bu.h  xr19,  xr18,   0
    xvpermi.d       xr12,  xr17,   0xd8
    xvpermi.d       xr13,  xr19,   0xd8
    xvpermi.q       xr14,  xr12,   0x01
    xvpermi.q       xr15,  xr13,   0x01
    vst             vr12,  a0,     0
    vstx            vr14,  a0,     a1
    alsl.d          a0,    a1,     a0,   1
    vst             vr13,  a0,     0
    vstx            vr15,  a0,     a1

    alsl.d          a0,    a1,     a0,   1
    alsl.d          a2,    a4,     a2,   2
    alsl.d          t5,    a5,     t5,   2
.endr
endfunc
.endm

pixel_add_ps_16x_lasx 16
pixel_add_ps_16x_lasx 32

.macro pixel_add_ps_32x_lasx h
function x265_pixel_add_ps_32x\h\()_lasx
    slli.w          a5,    a5,    1
    addi.d          t5,    a3,    0
.rept (\h>>1)

    xvld            xr0,   a2,    0
    xvldx           xr1,   a2,    a4
    xvld            xr2,   t5,    0
    xvldx           xr3,   t5,    a5
    addi.d          a3,    t5,    32
    xvld            xr4,   a3,    0
    xvldx           xr5,   a3,    a5
    xvpermi.q       xr20,  xr0,   0x01
    xvpermi.q       xr21,  xr1,   0x01
    vext2xv.hu.bu   xr6,   xr0
    vext2xv.hu.bu   xr7,   xr20
    vext2xv.hu.bu   xr8,   xr1
    vext2xv.hu.bu   xr9,   xr21
    xvadd.h         xr10,  xr6,   xr2
    xvadd.h         xr11,  xr7,   xr4
    xvadd.h         xr12,  xr8,   xr3
    xvadd.h         xr13,  xr9,   xr5
    xvssrarni.bu.h  xr11,  xr10,  0
    xvssrarni.bu.h  xr13,  xr12,  0
    xvpermi.d       xr11,  xr11,  0xd8
    xvpermi.d       xr13,  xr13,  0xd8
    xvst            xr11,  a0,    0
    xvstx           xr13,  a0,    a1

    alsl.d          a0,    a1,    a0,   1
    alsl.d          a2,    a4,    a2,   1
    alsl.d          t5,    a5,    t5,   1
.endr
endfunc
.endm

pixel_add_ps_32x_lasx 32
pixel_add_ps_32x_lasx 64

function x265_pixel_add_ps_64x64_lasx
    slli.w          a5,    a5,    1

.rept 32

    add.d           t3,    a2,    a4
    xvld            xr0,   a2,    0
    xvld            xr1,   a2,    32
    xvld            xr2,   t3,    0
    xvld            xr3,   t3,    32

    add.d           t5,    a3,    a5
    xvld            xr4,   a3,    0
    xvld            xr5,   a3,    32
    xvld            xr6,   a3,    64
    xvld            xr7,   a3,    96
    xvld            xr8,   t5,    0
    xvld            xr9,   t5,    32
    xvld            xr10,  t5,    64
    xvld            xr11,  t5,    96

    xvpermi.q       xr20,  xr0,   0x01
    xvpermi.q       xr21,  xr1,   0x01
    xvpermi.q       xr22,  xr2,   0x01
    xvpermi.q       xr23,  xr3,   0x01
    vext2xv.hu.bu   xr12,  xr0
    vext2xv.hu.bu   xr13,  xr20
    vext2xv.hu.bu   xr14,  xr1
    vext2xv.hu.bu   xr15,  xr21
    vext2xv.hu.bu   xr16,  xr2
    vext2xv.hu.bu   xr17,  xr22
    vext2xv.hu.bu   xr18,  xr3
    vext2xv.hu.bu   xr19,  xr23
    xvadd.h         xr4,   xr4,   xr12
    xvadd.h         xr5,   xr5,   xr13
    xvadd.h         xr6,   xr6,   xr14
    xvadd.h         xr7,   xr7,   xr15
    xvadd.h         xr8,   xr8,   xr16
    xvadd.h         xr9,   xr9,   xr17
    xvadd.h         xr10,  xr10,  xr18
    xvadd.h         xr11,  xr11,  xr19
    xvssrarni.bu.h  xr5,   xr4,   0
    xvssrarni.bu.h  xr7,   xr6,   0
    xvssrarni.bu.h  xr9,   xr8,   0
    xvssrarni.bu.h  xr11,  xr10,  0
    xvpermi.d       xr5,   xr5,   0xd8
    xvpermi.d       xr7,   xr7,   0xd8
    xvpermi.d       xr9,   xr9,   0xd8
    xvpermi.d       xr11,  xr11,  0xd8
    xvst            xr5,   a0,    0
    xvst            xr7,   a0,    32
    add.d           t2,    a0,    a1
    xvst            xr9,   t2,    0
    xvst            xr11,  t2,    32

    alsl.d          a0,    a1,    a0,    1
    alsl.d          a2,    a4,    a2,    1
    alsl.d          a3,    a5,    a3,    1
.endr
endfunc

function x265_pixel_var_4x4_lsx
    slli.d         t0,     a1,     1
    add.d          t1,     t0,     a1
    FLDS_LOADX_4   a0,     a1,     t0, t1, f0, f1, f2, f3
    vilvl.w        vr4,    vr1,    vr0
    vilvl.w        vr5,    vr3,    vr2
    vilvl.d        vr6,    vr5,    vr4
    vhaddw.hu.bu   vr7,    vr6,    vr6

    vmulwev.h.bu   vr0,    vr6,    vr6
    vmulwod.h.bu   vr1,    vr6,    vr6
    vhaddw.wu.hu   vr0,    vr0,    vr0
    vhaddw.wu.hu   vr1,    vr1,    vr1
    vadd.w         vr2,    vr0,    vr1

    vhaddw.wu.hu   vr7,    vr7,    vr7
    vhaddw.du.wu   vr7,    vr7,    vr7
    vhaddw.qu.du   vr7,    vr7,    vr7
    vpickve2gr.wu  t0,     vr7,    0

    vhaddw.du.wu   vr2,    vr2,    vr2
    vhaddw.qu.du   vr2,    vr2,    vr2
    vpickve2gr.wu  t1,     vr2,    0
    slli.d         t1,     t1,     32
    add.d          a0,     t0,     t1
endfunc

function x265_pixel_var_8x8_lsx
    slli.d         t0,     a1,     1
    add.d          t1,     t0,     a1
    FLDD_LOADX_4   a0,     a1,     t0, t1, f0, f1, f2, f3
    vilvl.d        vr4,    vr1,    vr0
    vilvl.d        vr5,    vr3,    vr2
    vhaddw.hu.bu   vr6,    vr4,    vr4
    vhaddw.hu.bu   vr7,    vr5,    vr5
    vadd.h         vr7,    vr6,    vr7

    vmulwev.h.bu   vr0,    vr4,    vr4
    vmulwod.h.bu   vr1,    vr4,    vr4
    vmulwev.h.bu   vr2,    vr5,    vr5
    vmulwod.h.bu   vr3,    vr5,    vr5
    vhaddw.wu.hu   vr0,    vr0,    vr0
    vhaddw.wu.hu   vr1,    vr1,    vr1
    vhaddw.wu.hu   vr2,    vr2,    vr2
    vhaddw.wu.hu   vr3,    vr3,    vr3
    vadd.w         vr4,    vr0,    vr1
    vadd.w         vr5,    vr2,    vr3
    vadd.w         vr8,    vr4,    vr5

    alsl.d         a0,     a1,     a0,   2

    FLDD_LOADX_4   a0,     a1,     t0, t1, f0, f1, f2, f3
    vilvl.d        vr4,    vr1,    vr0
    vilvl.d        vr5,    vr3,    vr2
    vhaddw.hu.bu   vr6,    vr4,    vr4
    vhaddw.hu.bu   vr9,    vr5,    vr5
    vadd.h         vr9,    vr6,    vr9

    vmulwev.h.bu   vr0,    vr4,    vr4
    vmulwod.h.bu   vr1,    vr4,    vr4
    vmulwev.h.bu   vr2,    vr5,    vr5
    vmulwod.h.bu   vr3,    vr5,    vr5
    vhaddw.wu.hu   vr0,    vr0,    vr0
    vhaddw.wu.hu   vr1,    vr1,    vr1
    vhaddw.wu.hu   vr2,    vr2,    vr2
    vhaddw.wu.hu   vr3,    vr3,    vr3
    vadd.w         vr4,    vr0,    vr1
    vadd.w         vr5,    vr2,    vr3
    vadd.w         vr10,   vr4,    vr5

    vadd.h         vr7,    vr7,    vr9
    vadd.w         vr2,    vr8,    vr10
    vhaddw.wu.hu   vr7,    vr7,    vr7
    vhaddw.du.wu   vr7,    vr7,    vr7
    vhaddw.qu.du   vr7,    vr7,    vr7
    vpickve2gr.wu  t0,     vr7,    0

    vhaddw.du.wu   vr2,    vr2,    vr2
    vhaddw.qu.du   vr2,    vr2,    vr2
    vpickve2gr.wu  t1,     vr2,    0
    slli.d         t1,     t1,     32
    add.d          a0,     t0,     t1
endfunc

.macro pixel_var_16x4_core_lsx out0, out1
    LSX_LOADX_4    a0,     a1,     t0, t1, vr0, vr1, vr2, vr3

    vhaddw.hu.bu   vr4,    vr0,    vr0
    vhaddw.hu.bu   vr5,    vr1,    vr1
    vhaddw.hu.bu   vr6,    vr2,    vr2
    vhaddw.hu.bu   vr7,    vr3,    vr3
    vadd.h         vr8,    vr4,    vr5
    vadd.h         vr15,   vr6,    vr7
    vadd.h         \out0,  vr15,   vr8

    vmulwev.h.bu   vr4,    vr0,    vr0
    vmulwod.h.bu   vr5,    vr0,    vr0
    vmulwev.h.bu   vr6,    vr1,    vr1
    vmulwod.h.bu   vr7,    vr1,    vr1
    vmulwev.h.bu   vr15,   vr2,    vr2
    vmulwod.h.bu   vr16,   vr2,    vr2
    vmulwev.h.bu   vr17,   vr3,    vr3
    vmulwod.h.bu   vr18,   vr3,    vr3

    vhaddw.wu.hu   vr4,    vr4,    vr4
    vhaddw.wu.hu   vr5,    vr5,    vr5
    vhaddw.wu.hu   vr6,    vr6,    vr6
    vhaddw.wu.hu   vr7,    vr7,    vr7
    vhaddw.wu.hu   vr15,   vr15,   vr15
    vhaddw.wu.hu   vr16,   vr16,   vr16
    vhaddw.wu.hu   vr17,   vr17,   vr17
    vhaddw.wu.hu   vr18,   vr18,   vr18

    vadd.w         vr0,    vr4,    vr15
    vadd.w         vr1,    vr5,    vr16
    vadd.w         vr2,    vr6,    vr17
    vadd.w         vr3,    vr7,    vr18
    vadd.w         vr4,    vr0,    vr1
    vadd.w         vr15,   vr2,    vr3
    vadd.w         \out1,  vr4,    vr15
.endm

function x265_pixel_var_16x16_lsx
    slli.d         t0,     a1,     1
    add.d          t1,     t0,     a1

    pixel_var_16x4_core_lsx vr9, vr10

.rept 3
    alsl.d a0, a1, a0, 2
    pixel_var_16x4_core_lsx vr11, vr12
    vadd.h         vr9,    vr9,     vr11
    vadd.w         vr10,   vr10,    vr12
.endr

    vhaddw.wu.hu   vr9,    vr9,    vr9
    vhaddw.du.wu   vr9,    vr9,    vr9
    vhaddw.qu.du   vr9,    vr9,    vr9
    vpickve2gr.wu  t0,     vr9,    0

    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vpickve2gr.wu  t1,     vr10,   0
    slli.d         t1,     t1,     32
    add.d          a0,     t0,     t1
endfunc

function x265_pixel_var_32x32_lsx
    slli.d         t0,     a1,     1
    add.d          t1,     t0,     a1

    pixel_var_16x4_core_lsx vr9, vr10
    addi.d         a0,     a0,     16
    pixel_var_16x4_core_lsx vr11, vr12
    vadd.h         vr9,    vr9,    vr11
    vadd.w         vr10,   vr10,   vr12

.rept 7
    addi.d         a0,     a0,     -16
    alsl.d         a0,     a1,     a0,   2
    pixel_var_16x4_core_lsx vr11, vr12
    vadd.h         vr9,    vr9,    vr11
    vadd.w         vr10,   vr10,   vr12

    addi.d         a0,     a0,     16
    pixel_var_16x4_core_lsx vr11, vr12
    vadd.h         vr9,    vr9,    vr11
    vadd.w         vr10,   vr10,   vr12
.endr

    vhaddw.wu.hu   vr9,    vr9,    vr9
    vhaddw.du.wu   vr9,    vr9,    vr9
    vhaddw.qu.du   vr9,    vr9,    vr9
    vpickve2gr.wu  t0,     vr9,    0

    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vpickve2gr.wu  t1,     vr10,   0
    slli.d         t1,     t1,     32
    add.d          a0,     t0,     t1
endfunc

function x265_pixel_var_64x64_lsx
    slli.d         t0,     a1,     1
    add.d          t1,     t0,     a1

    pixel_var_16x4_core_lsx vr9, vr10
    vhaddw.wu.hu vr9, vr9, vr9
.rept 3
    addi.d         a0,     a0,     16
    pixel_var_16x4_core_lsx vr11, vr12
    vhaddw.wu.hu vr11, vr11, vr11
    vadd.w         vr9,    vr9,    vr11
    vadd.w         vr10,   vr10,   vr12
.endr

.rept 15
    addi.d         a0,     a0,     -48
    alsl.d         a0,     a1,     a0,   2
    pixel_var_16x4_core_lsx vr11, vr12
    vhaddw.wu.hu vr11, vr11, vr11
    vadd.w         vr9,    vr9,    vr11
    vadd.w         vr10,   vr10,   vr12
.rept 3
    addi.d         a0,     a0,     16
    pixel_var_16x4_core_lsx vr11, vr12
    vhaddw.wu.hu vr11, vr11, vr11
    vadd.w         vr9,    vr9,    vr11
    vadd.w         vr10,   vr10,   vr12
.endr
.endr

    //vhaddw.wu.hu   vr9,    vr9,    vr9
    vhaddw.du.wu   vr9,    vr9,    vr9
    vhaddw.qu.du   vr9,    vr9,    vr9
    vpickve2gr.wu  t0,     vr9,    0

    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vpickve2gr.wu  t1,     vr10,   0
    slli.d         t1,     t1,     32
    add.d          a0,     t0,     t1
endfunc

function x265_pixel_var_8x8_lasx
    slli.d         t0,     a1,     1
    add.d          t1,     t0,     a1
    FLDD_LOADX_4   a0,     a1,     t0, t1, f0, f1, f2, f3
    alsl.d         a0,     a1,     a0,   2
    FLDD_LOADX_4   a0,     a1,     t0, t1, f4, f5, f6, f7
    vilvl.d        vr0,    vr1,    vr0
    vilvl.d        vr1,    vr3,    vr2
    vilvl.d        vr4,    vr5,    vr4
    vilvl.d        vr5,    vr7,    vr6
    xvpermi.q      xr4,    xr0,    0x20
    xvpermi.q      xr5,    xr1,    0x20
    xvhaddw.hu.bu  xr6,    xr4,    xr4
    xvhaddw.hu.bu  xr7,    xr5,    xr5
    xvadd.h        xr16,   xr6,    xr7

    xvmulwev.h.bu  xr8,    xr4,    xr4
    xvmulwod.h.bu  xr9,    xr4,    xr4
    xvmulwev.h.bu  xr10,   xr5,    xr5
    xvmulwod.h.bu  xr11,   xr5,    xr5
    xvhaddw.wu.hu  xr8,    xr8,    xr8
    xvhaddw.wu.hu  xr9,    xr9,    xr9
    xvhaddw.wu.hu  xr10,   xr10,   xr10
    xvhaddw.wu.hu  xr11,   xr11,   xr11
    xvadd.w        xr14,   xr8,    xr9
    xvadd.w        xr15,   xr10,   xr11
    xvadd.w        xr17,   xr14,   xr15

    xvhaddw.wu.hu  xr16,   xr16,   xr16
    xvhaddw.du.wu  xr16,   xr16,   xr16
    xvhaddw.qu.du  xr16,   xr16,   xr16
    xvpickve2gr.wu t4,     xr16,   0
    xvpickve2gr.wu t7,     xr16,   4
    add.w          t4,     t4,     t7    // sum

    xvhaddw.du.wu  xr17,   xr17,   xr17
    xvhaddw.qu.du  xr17,   xr17,   xr17
    xvpickve2gr.du t6,     xr17,   0     // sqr
    xvpickve2gr.du t7,     xr17,   2
    add.d          t6,     t6,     t7

    slli.d         t5,     t6,     32
    add.d          a0,     t4,     t5
endfunc

function x265_pixel_var_16x16_lasx
    slli.d         t0,     a1,     1
    add.d          t1,     t0,     a1
    LSX_LOADX_4    a0,     a1,     t0, t1, vr0, vr1, vr2, vr3
    xvpermi.q      xr2,    xr0,    0x20
    xvpermi.q      xr3,    xr1,    0x20
    xvhaddw.hu.bu  xr4,    xr2,    xr2
    xvhaddw.hu.bu  xr5,    xr3,    xr3
    xvadd.h        xr13,   xr4,    xr5

    xvmulwev.h.bu  xr9,    xr2,    xr2
    xvmulwod.h.bu  xr10,   xr2,    xr2
    xvmulwev.h.bu  xr11,   xr3,    xr3
    xvmulwod.h.bu  xr12,   xr3,    xr3
    xvhaddw.wu.hu  xr9,    xr9,    xr9
    xvhaddw.wu.hu  xr10,   xr10,   xr10
    xvhaddw.wu.hu  xr11,   xr11,   xr11
    xvhaddw.wu.hu  xr12,   xr12,   xr12
    xvadd.w        xr0,    xr10,   xr9
    xvadd.w        xr1,    xr12,   xr11
    xvadd.w        xr14,   xr0,    xr1

.rept 3
    alsl.d         a0,     a1,     a0,   2
    LSX_LOADX_4    a0,     a1,     t0, t1, vr0, vr1, vr2, vr3

    xvpermi.q      xr2,    xr0,    0x20
    xvpermi.q      xr3,    xr1,    0x20
    xvhaddw.hu.bu  xr4,    xr2,    xr2
    xvhaddw.hu.bu  xr5,    xr3,    xr3
    xvadd.h        xr6,    xr4,    xr5
    xvadd.h        xr13,   xr6,    xr13

    xvmulwev.h.bu  xr9,    xr2,    xr2
    xvmulwod.h.bu  xr10,   xr2,    xr2
    xvmulwev.h.bu  xr11,   xr3,    xr3
    xvmulwod.h.bu  xr12,   xr3,    xr3
    xvhaddw.wu.hu  xr9,    xr9,    xr9
    xvhaddw.wu.hu  xr10,   xr10,   xr10
    xvhaddw.wu.hu  xr11,   xr11,   xr11
    xvhaddw.wu.hu  xr12,   xr12,   xr12
    xvadd.w        xr0,    xr10,   xr9
    xvadd.w        xr1,    xr12,   xr11
    xvadd.w        xr7,    xr0,    xr1
    xvadd.w        xr14,   xr7,    xr14
.endr

    xvhaddw.wu.hu  xr13,   xr13,   xr13
    xvhaddw.du.wu  xr13,   xr13,   xr13
    xvhaddw.qu.du  xr13,   xr13,   xr13
    xvpickve2gr.wu t4,     xr13,   0
    xvpickve2gr.wu t7,     xr13,   4
    add.w          t4,     t4,     t7

    xvhaddw.du.wu  xr14,   xr14,   xr14
    xvhaddw.qu.du  xr14,   xr14,   xr14
    xvpickve2gr.du t6,     xr14,   0     // sqr
    xvpickve2gr.du t7,     xr14,   2
    add.d          t6,     t6,     t7

    slli.d         t5,     t6,     32
    add.d          a0,     t4,     t5
endfunc

.macro pixel_var_32x4_core_lasx out0, out1
    LASX_LOADX_4   a0,     a1,     t0, t1, xr0, xr1, xr2, xr3
    xvhaddw.hu.bu  xr4,    xr0,    xr0
    xvhaddw.hu.bu  xr5,    xr1,    xr1
    xvhaddw.hu.bu  xr6,    xr2,    xr2
    xvhaddw.hu.bu  xr7,    xr3,    xr3
    xvadd.h        xr10,   xr4,    xr5
    xvadd.h        xr11,   xr6,    xr7
    xvadd.h        \out0,  xr11,   xr10

    xvmulwev.h.bu  xr4,    xr0,    xr0
    xvmulwod.h.bu  xr5,    xr0,    xr0
    xvmulwev.h.bu  xr6,    xr1,    xr1
    xvmulwod.h.bu  xr7,    xr1,    xr1
    xvmulwev.h.bu  xr8,    xr2,    xr2
    xvmulwod.h.bu  xr9,    xr2,    xr2
    xvmulwev.h.bu  xr10,   xr3,    xr3
    xvmulwod.h.bu  xr11,   xr3,    xr3
    xvhaddw.wu.hu  xr4,    xr4,    xr4
    xvhaddw.wu.hu  xr5,    xr5,    xr5
    xvhaddw.wu.hu  xr6,    xr6,    xr6
    xvhaddw.wu.hu  xr7,    xr7,    xr7
    xvhaddw.wu.hu  xr8,    xr8,    xr8
    xvhaddw.wu.hu  xr9,    xr9,    xr9
    xvhaddw.wu.hu  xr10,   xr10,   xr10
    xvhaddw.wu.hu  xr11,   xr11,   xr11
    xvadd.w        xr0,    xr4,    xr5
    xvadd.w        xr1,    xr6,    xr7
    xvadd.w        xr2,    xr8,    xr9
    xvadd.w        xr3,    xr10,   xr11
    xvadd.w        xr4,    xr0,    xr1
    xvadd.w        xr5,    xr2,    xr3
    xvadd.w        \out1,  xr4,    xr5
.endm

function x265_pixel_var_32x32_lasx
    slli.d         t0,     a1,     1
    add.d          t1,     t0,     a1

    pixel_var_32x4_core_lasx xr13, xr14

.rept 7
    alsl.d a0, a1, a0, 2
    pixel_var_32x4_core_lasx xr15, xr16
    xvadd.h        xr13,   xr13,   xr15
    xvadd.w        xr14,   xr14,   xr16
.endr

    xvhaddw.wu.hu  xr13,   xr13,   xr13
    xvhaddw.du.wu  xr13,   xr13,   xr13
    xvhaddw.qu.du  xr13,   xr13,   xr13
    xvpickve2gr.wu t4,     xr13,   0
    xvpickve2gr.wu t7,     xr13,   4
    add.w          t4,     t4,     t7

    xvhaddw.du.wu  xr14,   xr14,   xr14
    xvhaddw.qu.du  xr14,   xr14,   xr14
    xvpickve2gr.du t6,     xr14,   0     // sqr
    xvpickve2gr.du t7,     xr14,   2
    add.d          t6,     t6,     t7

    slli.d         t5,     t6,     32
    add.d          a0,     t4,     t5
endfunc

function x265_pixel_var_64x64_lasx
    slli.d         t0,     a1,     1
    add.d          t1,     t0,     a1

    pixel_var_32x4_core_lasx xr13, xr14
    addi.d         a0,     a0,     32
    pixel_var_32x4_core_lasx xr15, xr16
    xvadd.h        xr13,   xr15,   xr13
    xvadd.w        xr14,   xr14,   xr16

.rept 15
    addi.d         a0,     a0,     -32
    alsl.d         a0,     a1,     a0,    2
    pixel_var_32x4_core_lasx xr15, xr16
    xvadd.h        xr13,   xr13,   xr15
    xvadd.w        xr14,   xr14,   xr16

    addi.d         a0,     a0,     32
    pixel_var_32x4_core_lasx xr15, xr16
    xvadd.h        xr13,   xr15,   xr13
    xvadd.w        xr14,   xr14,   xr16
.endr

    xvhaddw.wu.hu  xr13,   xr13,   xr13
    xvhaddw.du.wu  xr13,   xr13,   xr13
    xvhaddw.qu.du  xr13,   xr13,   xr13
    xvpickve2gr.wu t4,     xr13,   0
    xvpickve2gr.wu t7,     xr13,   4
    add.w          t4,     t4,     t7

    xvhaddw.du.wu  xr14,   xr14,   xr14
    xvhaddw.qu.du  xr14,   xr14,   xr14
    xvpickve2gr.du t6,     xr14,   0     // sqr
    xvpickve2gr.du t7,     xr14,   2
    add.d          t6,     t6,     t7

    slli.d         t5,     t6,     32
    add.d          a0,     t4,     t5
endfunc

function x265_pixel_getResidual_4x4_lsx
    slli.d         t3,     a3,     1

    fld.s          f0,     a0,     0
    fld.s          f1,     a1,     0
    fldx.s         f2,     a0,     a3
    fldx.s         f3,     a1,     a3
    vilvl.w        vr2,    vr2,    vr0
    vilvl.w        vr3,    vr3,    vr1
    vsllwil.hu.bu  vr2,    vr2,    0
    vsllwil.hu.bu  vr3,    vr3,    0
    vsub.h         vr4,    vr2,    vr3
    vstelm.d       vr4,    a2,     0,   0
    add.d          a2,     a2,     t3
    vstelm.d       vr4,    a2,     0,   1

    alsl.d         a0,     a3,     a0,  1
    alsl.d         a1,     a3,     a1,  1
    add.d          a2,     a2,     t3

    fld.s          f0,     a0,     0
    fld.s          f1,     a1,     0
    fldx.s         f2,     a0,     a3
    fldx.s         f3,     a1,     a3
    vilvl.w        vr2,    vr2,    vr0
    vilvl.w        vr3,    vr3,    vr1
    vsllwil.hu.bu  vr2,    vr2,    0
    vsllwil.hu.bu  vr3,    vr3,    0
    vsub.h         vr4,    vr2,    vr3
    vstelm.d       vr4,    a2,     0,   0
    add.d          a2,     a2,     t3
    vstelm.d       vr4,    a2,     0,   1
endfunc

function x265_pixel_getResidual_8x8_lsx
    slli.d         t3,     a3,     1

.rept 4
    fld.d          f0,     a0,     0
    fld.d          f1,     a1,     0
    fldx.d         f2,     a0,     a3
    fldx.d         f3,     a1,     a3
    vsllwil.hu.bu  vr0,    vr0,    0
    vsllwil.hu.bu  vr1,    vr1,    0
    vsllwil.hu.bu  vr2,    vr2,    0
    vsllwil.hu.bu  vr3,    vr3,    0
    vsub.h         vr4,    vr0,    vr1
    vsub.h         vr5,    vr2,    vr3
    vst            vr4,    a2,     0
    vstx           vr5,    a2,     t3

    alsl.d         a0,     a3,     a0,  1
    alsl.d         a1,     a3,     a1,  1
    alsl.d         a2,     t3,     a2,  1
.endr
endfunc

function x265_pixel_getResidual_16x16_lsx
    slli.d         t3,     a3,     1

.rept 8
    vld            vr0,    a0,     0
    vld            vr1,    a1,     0
    vldx           vr2,    a0,     a3
    vldx           vr3,    a1,     a3
    vsubwev.h.bu   vr4,    vr0,    vr1
    vsubwod.h.bu   vr5,    vr0,    vr1
    vsubwev.h.bu   vr6,    vr2,    vr3
    vsubwod.h.bu   vr7,    vr2,    vr3
    vilvl.h        vr8,    vr5,    vr4
    vilvh.h        vr9,    vr5,    vr4
    vilvl.h        vr10,   vr7,    vr6
    vilvh.h        vr11,   vr7,    vr6

    add.d          t2,     a2,     t3
    vst            vr8,    a2,     0
    vst            vr9,    a2,     16
    vst            vr10,   t2,     0
    vst            vr11,   t2,     16

    alsl.d         a0,     a3,     a0,  1
    alsl.d         a1,     a3,     a1,  1
    alsl.d         a2,     t3,     a2,  1
.endr
endfunc

function x265_pixel_getResidual_32x32_lsx
    slli.d         t3,     a3,     1

.rept 32
    vld            vr0,    a0,     0
    vld            vr1,    a1,     0
    vld            vr2,    a0,     16
    vld            vr3,    a1,     16
    vsubwev.h.bu   vr4,    vr0,    vr1
    vsubwod.h.bu   vr5,    vr0,    vr1
    vsubwev.h.bu   vr6,    vr2,    vr3
    vsubwod.h.bu   vr7,    vr2,    vr3
    vilvl.h        vr8,    vr5,    vr4
    vilvh.h        vr9,    vr5,    vr4
    vilvl.h        vr10,   vr7,    vr6
    vilvh.h        vr11,   vr7,    vr6
    vst            vr8,    a2,     0
    vst            vr9,    a2,     16
    vst            vr10,   a2,     32
    vst            vr11,   a2,     48

    add.d          a0,     a3,     a0
    add.d          a1,     a3,     a1
    add.d          a2,     t3,     a2
.endr
endfunc

function x265_pixel_getResidual_64x64_lsx
    slli.d         t3,     a3,     1

.rept 64
    vld            vr0,    a0,     0
    vld            vr1,    a1,     0
    vld            vr2,    a0,     16
    vld            vr3,    a1,     16
    vld            vr10,   a0,     32
    vld            vr11,   a1,     32
    vld            vr12,   a0,     48
    vld            vr13,   a1,     48
    vsubwev.h.bu   vr4,    vr0,    vr1
    vsubwod.h.bu   vr5,    vr0,    vr1
    vsubwev.h.bu   vr6,    vr2,    vr3
    vsubwod.h.bu   vr7,    vr2,    vr3
    vsubwev.h.bu   vr14,   vr10,   vr11
    vsubwod.h.bu   vr15,   vr10,   vr11
    vsubwev.h.bu   vr16,   vr12,   vr13
    vsubwod.h.bu   vr17,   vr12,   vr13
    vilvl.h        vr0,    vr5,    vr4
    vilvh.h        vr1,    vr5,    vr4
    vilvl.h        vr2,    vr7,    vr6
    vilvh.h        vr3,    vr7,    vr6
    vilvl.h        vr10,   vr15,   vr14
    vilvh.h        vr11,   vr15,   vr14
    vilvl.h        vr12,   vr17,   vr16
    vilvh.h        vr13,   vr17,   vr16
    vst            vr0,    a2,     0
    vst            vr1,    a2,     16
    vst            vr2,    a2,     32
    vst            vr3,    a2,     48
    vst            vr10,   a2,     64
    vst            vr11,   a2,     80
    vst            vr12,   a2,     96
    vst            vr13,   a2,     112

    add.d          a0,     a3,     a0
    add.d          a1,     a3,     a1
    add.d          a2,     t3,     a2
.endr
endfunc

function x265_pixel_getResidual_32x32_lasx
    slli.d         t3,     a3,     1

.rept 16
    xvld           xr0,    a0,     0
    xvld           xr1,    a1,     0
    xvldx          xr2,    a0,     a3
    xvldx          xr3,    a1,     a3
    xvsubwev.h.bu  xr4,    xr0,    xr1
    xvsubwod.h.bu  xr5,    xr0,    xr1
    xvsubwev.h.bu  xr6,    xr2,    xr3
    xvsubwod.h.bu  xr7,    xr2,    xr3
    xvilvl.h       xr8,    xr5,    xr4
    xvilvh.h       xr9,    xr5,    xr4
    xvilvl.h       xr10,   xr7,    xr6
    xvilvh.h       xr11,   xr7,    xr6
    xvpermi.q      xr12,   xr8,    0x10
    xvpermi.q      xr13,   xr10,   0x10
    xvpermi.q      xr8,    xr9,    0x02
    xvpermi.q      xr10,   xr11,   0x02
    xvpermi.q      xr9,    xr12,   0x31
    xvpermi.q      xr11,   xr13,   0x31
    xvst           xr8,    a2,     0
    xvst           xr9,    a2,     32
    add.d          t2,     a2,     t3
    xvst           xr10,   t2,     0
    xvst           xr11,   t2,     32

    alsl.d         a0,     a3,     a0,  1
    alsl.d         a1,     a3,     a1,  1
    alsl.d         a2,     t3,     a2,  1
.endr
endfunc

function x265_pixel_getResidual_64x64_lasx
    slli.d         t3,     a3,     1

.rept 64
    xvld           xr0,    a0,     0
    xvld           xr1,    a1,     0
    xvld           xr2,    a0,     32
    xvld           xr3,    a1,     32
    xvsubwev.h.bu  xr4,    xr0,    xr1
    xvsubwod.h.bu  xr5,    xr0,    xr1
    xvsubwev.h.bu  xr6,    xr2,    xr3
    xvsubwod.h.bu  xr7,    xr2,    xr3
    xvilvl.h       xr8,    xr5,    xr4
    xvilvh.h       xr9,    xr5,    xr4
    xvilvl.h       xr10,   xr7,    xr6
    xvilvh.h       xr11,   xr7,    xr6
    xvpermi.q      xr12,   xr8,    0x10
    xvpermi.q      xr13,   xr10,   0x10
    xvpermi.q      xr8,    xr9,    0x02
    xvpermi.q      xr10,   xr11,   0x02
    xvpermi.q      xr9,    xr12,   0x31
    xvpermi.q      xr11,   xr13,   0x31
    xvst           xr8,    a2,     0
    xvst           xr9,    a2,     32
    xvst           xr10,   a2,     64
    xvst           xr11,   a2,     96

    add.d          a0,     a3,     a0
    add.d          a1,     a3,     a1
    add.d          a2,     t3,     a2
.endr
endfunc

.macro pixel_normFact_8x8_core_lsx out0, out1, out2, out3
    vld            vr0,    a0,     0
    vld            vr1,    a0,     16
    vld            vr2,    a0,     32
    vld            vr3,    a0,     48
    vsrl.b         vr0,    vr0,    vr4
    vsrl.b         vr1,    vr1,    vr4
    vsrl.b         vr2,    vr2,    vr4
    vsrl.b         vr3,    vr3,    vr4
    vsllwil.hu.bu  vr5,    vr0,    0
    vexth.hu.bu    vr6,    vr0
    vsllwil.hu.bu  vr7,    vr1,    0
    vexth.hu.bu    vr8,    vr1
    vsllwil.hu.bu  vr9,    vr2,    0
    vexth.hu.bu    vr10,   vr2
    vsllwil.hu.bu  vr11,   vr3,    0
    vexth.hu.bu    vr12,   vr3

    vmaddwev.w.h   \out0,  vr5,    vr5
    vmaddwev.w.h   \out1,  vr6,    vr6
    vmaddwev.w.h   \out2,  vr7,    vr7
    vmaddwev.w.h   \out3,  vr8,    vr8
    vmaddwod.w.h   \out0,  vr5,    vr5
    vmaddwod.w.h   \out1,  vr6,    vr6
    vmaddwod.w.h   \out2,  vr7,    vr7
    vmaddwod.w.h   \out3,  vr8,    vr8
    vmaddwev.w.h   \out0,  vr9,    vr9
    vmaddwev.w.h   \out1,  vr10,   vr10
    vmaddwev.w.h   \out2,  vr11,   vr11
    vmaddwev.w.h   \out3,  vr12,   vr12
    vmaddwod.w.h   \out0,  vr9,    vr9
    vmaddwod.w.h   \out1,  vr10,   vr10
    vmaddwod.w.h   \out2,  vr11,   vr11
    vmaddwod.w.h   \out3,  vr12,   vr12
.endm

function x265_pixel_normFact_8x8_lsx
    vreplgr2vr.b   vr4,    a2

    vxor.v         vr13,   vr13,   vr13
    vxor.v         vr14,   vr14,   vr14
    vxor.v         vr15,   vr15,   vr15
    vxor.v         vr16,   vr16,   vr16

    pixel_normFact_8x8_core_lsx vr13, vr14, vr15, vr16

    vadd.w         vr13,   vr13,   vr14
    vadd.w         vr15,   vr15,   vr16
    vadd.w         vr12,   vr13,   vr15
    vhaddw.du.wu   vr12,   vr12,   vr12
    vhaddw.qu.du   vr12,   vr12,   vr12
    vstelm.d       vr12,   a3,     0,      0
endfunc

.macro pixel_normFact_lsx w, h
function x265_pixel_normFact_\w\()x\h\()_lsx
    vreplgr2vr.b   vr4,    a2

    vxor.v         vr13,   vr13,   vr13
    vxor.v         vr14,   vr14,   vr14
    vxor.v         vr15,   vr15,   vr15
    vxor.v         vr16,   vr16,   vr16

    pixel_normFact_8x8_core_lsx vr13, vr14, vr15, vr16

.rept (\w)*(\h)/64-1
    addi.d         a0,     a0,     64

    pixel_normFact_8x8_core_lsx vr13, vr14, vr15, vr16
.endr

    vadd.w         vr13,   vr13,   vr14
    vadd.w         vr15,   vr15,   vr16
    vadd.w         vr12,   vr13,   vr15
    vhaddw.du.wu   vr12,   vr12,   vr12
    vhaddw.qu.du   vr12,   vr12,   vr12
    vstelm.d       vr12,   a3,     0,      0
endfunc
.endm

pixel_normFact_lsx 16, 16
pixel_normFact_lsx 32, 32
pixel_normFact_lsx 64, 64

.macro pixel_normFact_8x8_core_lasx out0, out1, out2, out3
    xvld           xr0,    a0,     0
    xvld           xr1,    a0,     32
    xvsrl.b        xr0,    xr0,    xr4
    xvsrl.b        xr1,    xr1,    xr4
    xvsllwil.hu.bu xr5,    xr0,    0
    xvexth.hu.bu   xr6,    xr0
    xvsllwil.hu.bu xr7,    xr1,    0
    xvexth.hu.bu   xr8,    xr1

    xvmaddwev.w.h  \out0,  xr5,    xr5
    xvmaddwev.w.h  \out1,  xr6,    xr6
    xvmaddwev.w.h  \out2,  xr7,    xr7
    xvmaddwev.w.h  \out3,  xr8,    xr8
    xvmaddwod.w.h  \out0,  xr5,    xr5
    xvmaddwod.w.h  \out1,  xr6,    xr6
    xvmaddwod.w.h  \out2,  xr7,    xr7
    xvmaddwod.w.h  \out3,  xr8,    xr8
.endm

function x265_pixel_normFact_8x8_lasx
    xvreplgr2vr.b  xr4,    a2

    xvxor.v        xr13,   xr13,   xr13
    xvxor.v        xr14,   xr14,   xr14
    xvxor.v        xr15,   xr15,   xr15
    xvxor.v        xr16,   xr16,   xr16

    pixel_normFact_8x8_core_lasx xr13, xr14, xr15, xr16

    xvadd.w        xr13,   xr13,   xr14
    xvadd.w        xr15,   xr15,   xr16
    xvadd.w        xr12,   xr13,   xr15
    xvpermi.q      xr17,   xr12,   0x01
    vadd.w         vr12,   vr12,   vr17
    vhaddw.du.wu   vr12,   vr12,   vr12
    vhaddw.qu.du   vr12,   vr12,   vr12
    vstelm.d       vr12,   a3,     0,      0
endfunc

.macro pixel_normFact_lasx w, h
function x265_pixel_normFact_\w\()x\h\()_lasx
    xvreplgr2vr.b  xr4,    a2

    xvxor.v        xr13,   xr13,   xr13
    xvxor.v        xr14,   xr14,   xr14
    xvxor.v        xr15,   xr15,   xr15
    xvxor.v        xr16,   xr16,   xr16

    pixel_normFact_8x8_core_lasx xr13, xr14, xr15, xr16

.rept (\w)*(\h)/64-1
    addi.d         a0,     a0,     64

    pixel_normFact_8x8_core_lasx xr13, xr14, xr15, xr16
.endr

    xvadd.w        xr13,   xr13,   xr14
    xvadd.w        xr15,   xr15,   xr16
    xvadd.w        xr12,   xr13,   xr15
    xvpermi.q      xr17,   xr12,   0x01
    vadd.w         vr12,   vr12,   vr17
    vhaddw.du.wu   vr12,   vr12,   vr12
    vhaddw.qu.du   vr12,   vr12,   vr12
    vstelm.d       vr12,   a3,     0,      0
endfunc
.endm

pixel_normFact_lasx 16, 16
pixel_normFact_lasx 32, 32
pixel_normFact_lasx 64, 64

function x265_pixel_planecopy_cp_lsx
    vreplgr2vr.b   vr2,    a6
    addi.w         a5,     a5,     -1

.loop_h:
    addi.d         t0,     a0,     0
    addi.d         t1,     a2,     0
    add.d          t2,     zero,   zero
.loop_w:
    vld            vr0,    t0,     0
    vsll.b         vr0,    vr0,    vr2
    vst            vr0,    t1,     0
    addi.d         t2,     t2,     16
    addi.d         t0,     t0,     16
    addi.d         t1,     t1,     16
    blt            t2,     a4,     .loop_w

    add.d          a0,     a0,     a1
    add.d          a2,     a2,     a3
    addi.w         a5,     a5,     -1
    bnez           a5,     .loop_h

    add.d          t3,     a4,     zero
    srli.d         t3,     t3,     3
.loopw8:
    fld.d          f0,     a0,     0
    vsll.b         vr0,    vr0,    vr2
    vstelm.d       vr0,    a2,     0,     0
    addi.d         a4,     a4,     -8
    addi.d         t3,     t3,     -1
    addi.d         a0,     a0,     8
    addi.d         a2,     a2,     8
    bnez           t3,     .loopw8

    addi.d         a5,     zero,   8
    sub.d          a5,     a5,     a4
    sub.d          a0,     a0,     a5
    sub.d          a2,     a2,     a5
    fld.d          f0,     a0,     0
    vsll.b         vr0,    vr0,    vr2
    vstelm.d       vr0,    a2,     0,     0
endfunc

function x265_pixel_planecopy_sp_lsx
    vreplgr2vr.h   vr10,   a6      // shift
    vreplgr2vr.h   vr11,   a7      // mask
    addi.w         a5,     a5,     -1
    slli.d         a1,     a1,     1
.sp_loop_h:
    addi.d         t0,     a0,     0
    addi.d         t1,     a2,     0
    add.d          t2,     zero,   zero
.sp_loop_w:
    vld            vr0,    t0,     0
    vld            vr1,    t0,     16
    vsrl.h         vr0,    vr0,    vr10
    vsrl.h         vr1,    vr1,    vr10
    vand.v         vr0,    vr0,    vr11
    vand.v         vr1,    vr1,    vr11
    vssrarni.bu.h  vr1,    vr0,    0
    vst            vr1,    t1,     0
    addi.d         t2,     t2,     16
    addi.d         t0,     t0,     32
    addi.d         t1,     t1,     16
    blt            t2,     a4,     .sp_loop_w

    add.d          a0,     a0,     a1
    add.d          a2,     a2,     a3
    addi.w         a5,     a5,     -1
    bnez           a5,     .sp_loop_h

    add.d          t3,     a4,     zero
    srli.d         t3,     t3,     4
.sp_loopw32:
    vld            vr0,    a0,     0
    vld            vr1,    a0,     16
    vsrl.h         vr0,    vr0,    vr10
    vsrl.h         vr1,    vr1,    vr10
    vand.v         vr0,    vr0,    vr11
    vand.v         vr1,    vr1,    vr11
    vssrarni.bu.h  vr1,    vr0,    0
    vst            vr1,    a2,     0
    addi.d         a4,     a4,     -16
    addi.d         t3,     t3,     -1
    addi.d         a0,     a0,     32
    addi.d         a2,     a2,     16
    bnez           t3,     .sp_loopw32

    addi.d         a5,     zero,   16
    sub.d          a5,     a5,     a4
    sub.d          a0,     a0,     a5
    sub.d          a0,     a0,     a5
    sub.d          a2,     a2,     a5
    vld            vr0,    a0,     0
    vld            vr1,    a0,     16
    vsrl.h         vr0,    vr0,    vr10
    vsrl.h         vr1,    vr1,    vr10
    vand.v         vr0,    vr0,    vr11
    vand.v         vr1,    vr1,    vr11
    vssrarni.bu.h  vr1,    vr0,    0
    vst            vr1,    a2,     0
endfunc

function x265_pixel_planecopy_sp_shl_lsx
    vreplgr2vr.h   vr10,   a6      // shift
    vreplgr2vr.h   vr11,   a7      // mask
    addi.w         a5,     a5,     -1
    slli.d         a1,     a1,     1
.spshl_loop_h:
    addi.d         t0,     a0,     0
    addi.d         t1,     a2,     0
    add.d          t2,     zero,   zero
.spshl_loop_w:
    vld            vr0,    t0,     0
    vld            vr1,    t0,     16
    vsll.h         vr0,    vr0,    vr10
    vsll.h         vr1,    vr1,    vr10
    vand.v         vr0,    vr0,    vr11
    vand.v         vr1,    vr1,    vr11
    vssrarni.bu.h  vr1,    vr0,    0
    vst            vr1,    t1,     0
    addi.d         t2,     t2,     16
    addi.d         t0,     t0,     32
    addi.d         t1,     t1,     16
    blt            t2,     a4,     .spshl_loop_w

    add.d          a0,     a0,     a1
    add.d          a2,     a2,     a3
    addi.w         a5,     a5,     -1
    bnez           a5,     .spshl_loop_h

    add.d          t3,     a4,     zero
    srli.d         t3,     t3,     4
.spshl_loopw32:
    vld            vr0,    a0,     0
    vld            vr1,    a0,     16
    vsll.h         vr0,    vr0,    vr10
    vsll.h         vr1,    vr1,    vr10
    vand.v         vr0,    vr0,    vr11
    vand.v         vr1,    vr1,    vr11
    vssrarni.bu.h  vr1,    vr0,    0
    vst            vr1,    a2,     0
    addi.d         a4,     a4,     -16
    addi.d         t3,     t3,     -1
    addi.d         a0,     a0,     32
    addi.d         a2,     a2,     16
    bnez           t3,     .spshl_loopw32

    addi.d         a5,     zero,   16
    sub.d          a5,     a5,     a4
    sub.d          a0,     a0,     a5
    sub.d          a0,     a0,     a5
    sub.d          a2,     a2,     a5
    vld            vr0,    a0,     0
    vld            vr1,    a0,     16
    vsll.h         vr0,    vr0,    vr10
    vsll.h         vr1,    vr1,    vr10
    vand.v         vr0,    vr0,    vr11
    vand.v         vr1,    vr1,    vr11
    vssrarni.bu.h  vr1,    vr0,    0
    vst            vr1,    a2,     0
endfunc

function x265_pixel_planecopy_pp_shr_lsx
    vreplgr2vr.b   vr2,    a6
    addi.w         a5,     a5,     -1

.ppshr_loop_h:
    addi.d         t0,     a0,     0
    addi.d         t1,     a2,     0
    add.d          t2,     zero,   zero
.ppshr_loop_w:
    vld            vr0,    t0,     0
    vsrl.b         vr0,    vr0,    vr2
    vst            vr0,    t1,     0
    addi.d         t2,     t2,     16
    addi.d         t0,     t0,     16
    addi.d         t1,     t1,     16
    blt            t2,     a4,     .ppshr_loop_w

    add.d          a0,     a0,     a1
    add.d          a2,     a2,     a3
    addi.w         a5,     a5,     -1
    bnez           a5,     .ppshr_loop_h

    add.d          t3,     a4,     zero
    srli.d         t3,     t3,     3
.ppshr_loopw8:
    fld.d          f0,     a0,     0
    vsrl.b         vr0,    vr0,    vr2
    vstelm.d       vr0,    a2,     0,     0
    addi.d         a4,     a4,     -8
    addi.d         t3,     t3,     -1
    addi.d         a0,     a0,     8
    addi.d         a2,     a2,     8
    bnez           t3,     .ppshr_loopw8

    addi.d         a5,     zero,   8
    sub.d          a5,     a5,     a4
    sub.d          a0,     a0,     a5
    sub.d          a2,     a2,     a5
    fld.d          f0,     a0,     0
    vsrl.b         vr0,    vr0,    vr2
    vstelm.d       vr0,    a2,     0,     0
endfunc

function x265_scale2D_64to32_lsx
    addi.d         t0,     zero,   32

.loop_scale2D_lsx:
    vld            vr0,    a1,     0
    vld            vr1,    a1,     16
    vld            vr2,    a1,     32
    vld            vr3,    a1,     48
    add.d          a1,     a1,     a2
    addi.d         t0,     t0,     -1
    vld            vr4,    a1,     0
    vld            vr5,    a1,     16
    vld            vr6,    a1,     32
    vld            vr7,    a1,     48
    add.d          a1,     a1,     a2

.irp i, vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7
    vhaddw.hu.bu   \i,     \i,     \i
.endr
    vadd.h         vr0,    vr0,    vr4
    vadd.h         vr1,    vr1,    vr5
    vadd.h         vr2,    vr2,    vr6
    vadd.h         vr3,    vr3,    vr7
    vssrarni.bu.h  vr1,    vr0,    2
    vssrarni.bu.h  vr3,    vr2,    2
    vst            vr1,    a0,     0
    vst            vr3,    a0,     16
    addi.d         a0,     a0,     32
    bnez           t0,     .loop_scale2D_lsx
endfunc

function x265_scale1D_128to64_lsx
.rept 2
    vld            vr0,    a1,     0
    vld            vr1,    a1,     16
    vld            vr2,    a1,     32
    vld            vr3,    a1,     48
    vld            vr4,    a1,     64
    vld            vr5,    a1,     80
    vld            vr6,    a1,     96
    vld            vr7,    a1,     112

    vpickev.b      vr8,    vr1,    vr0
    vpickod.b      vr9,    vr1,    vr0
    vpickev.b      vr10,   vr3,    vr2
    vpickod.b      vr11,   vr3,    vr2
    vpickev.b      vr12,   vr5,    vr4
    vpickod.b      vr13,   vr5,    vr4
    vpickev.b      vr14,   vr7,    vr6
    vpickod.b      vr15,   vr7,    vr6
    vavgr.bu       vr0,    vr8,    vr9
    vavgr.bu       vr1,    vr10,   vr11
    vavgr.bu       vr2,    vr12,   vr13
    vavgr.bu       vr3,    vr14,   vr15
    vst            vr0,    a0,     0
    vst            vr1,    a0,     16
    vst            vr2,    a0,     32
    vst            vr3,    a0,     48
    addi.d         a0,     a0,     64
    addi.d         a1,     a1,     128
.endr
endfunc

function x265_scale2D_64to32_lasx
    addi.d         t0,     zero,   32

.loop_scale2D_lasx:
    xvld           xr0,    a1,     0
    xvld           xr1,    a1,     32
    add.d          a1,     a1,     a2
    addi.d         t0,     t0,     -1
    xvld           xr4,    a1,     0
    xvld           xr5,    a1,     32
    add.d          a1,     a1,     a2

.irp i, xr0, xr1, xr4, xr5
    xvhaddw.hu.bu  \i,     \i,     \i
.endr
    xvadd.h        xr0,    xr0,    xr4
    xvadd.h        xr1,    xr1,    xr5
    xvssrarni.bu.h xr1,    xr0,    2
    xvpermi.d      xr2,    xr1,    0xd8
    xvst           xr2,    a0,     0
    addi.d         a0,     a0,     32
    bnez           t0,     .loop_scale2D_lasx
endfunc

function x265_scale1D_128to64_lasx
.rept 2
    xvld           xr0,    a1,     0
    xvld           xr1,    a1,     32
    xvld           xr2,    a1,     64
    xvld           xr3,    a1,     96

    xvpickev.b     xr8,    xr1,    xr0
    xvpickod.b     xr9,    xr1,    xr0
    xvpickev.b     xr10,   xr3,    xr2
    xvpickod.b     xr11,   xr3,    xr2
    xvavgr.bu      xr0,    xr8,    xr9
    xvavgr.bu      xr1,    xr10,   xr11
    xvpermi.d      xr2,    xr0,    0xd8
    xvpermi.d      xr3,    xr1,    0xd8
    xvst           xr2,    a0,     0
    xvst           xr3,    a0,     32
    addi.d         a0,     a0,     64
    addi.d         a1,     a1,     128
.endr
endfunc

function x265_transpose_4x4_lsx
    fld.s          f0,     a1,     0
    fldx.s         f1,     a1,     a2
    alsl.d         a1,     a2,     a1,   1
    fld.s          f2,     a1,     0
    fldx.s         f3,     a1,     a2

    vilvl.b        vr1,    vr1,    vr0
    vilvl.b        vr3,    vr3,    vr2
    vilvl.h        vr4,    vr3,    vr1
    vst            vr4,    a0,     0
endfunc

function x265_transpose_8x8_lsx
    fld.d          f0,     a1,     0
    fldx.d         f1,     a1,     a2
    alsl.d         a1,     a2,     a1,   1
    fld.d          f2,     a1,     0
    fldx.d         f3,     a1,     a2
    alsl.d         a1,     a2,     a1,   1
    fld.d          f4,     a1,     0
    fldx.d         f5,     a1,     a2
    alsl.d         a1,     a2,     a1,   1
    fld.d          f6,     a1,     0
    fldx.d         f7,     a1,     a2

    vilvl.b        vr1,    vr1,    vr0
    vilvl.b        vr3,    vr3,    vr2
    vilvl.b        vr5,    vr5,    vr4
    vilvl.b        vr7,    vr7,    vr6
    vilvl.h        vr2,    vr3,    vr1
    vilvh.h        vr4,    vr3,    vr1
    vilvl.h        vr6,    vr7,    vr5
    vilvh.h        vr8,    vr7,    vr5
    vilvl.w        vr0,    vr6,    vr2
    vilvh.w        vr1,    vr6,    vr2
    vilvl.w        vr3,    vr8,    vr4
    vilvh.w        vr5,    vr8,    vr4

    vst            vr0,    a0,     0
    vst            vr1,    a0,     16
    vst            vr3,    a0,     32
    vst            vr5,    a0,     48
endfunc

.macro transpose_16x16_core_lsx in0, in1, in2, in3, in4, in5, in6, in7, \
                                in8, in9, in10, in11, in12, in13, in14, in15, in16
    vilvl.b        vr16,   vr2,    vr0
    vilvl.b        vr17,   vr3,    vr1
    vilvl.b        vr18,   vr6,    vr4
    vilvl.b        vr19,   vr7,    vr5
    vilvl.b        vr20,   vr17,   vr16
    vilvl.b        vr21,   vr19,   vr18
    vilvh.b        vr22,   vr17,   vr16
    vilvh.b        vr23,   vr19,   vr18
    vilvl.w        vr24,   vr21,   vr20
    vilvl.w        vr25,   vr23,   vr22
    vilvh.w        vr26,   vr21,   vr20
    vilvh.w        vr27,   vr23,   vr22

    vilvl.b        vr16,   vr10,   vr8
    vilvl.b        vr17,   vr11,   vr9
    vilvl.b        vr18,   vr14,   vr12
    vilvl.b        vr19,   vr15,   vr13
    vilvl.b        vr20,   vr17,   vr16
    vilvl.b        vr21,   vr19,   vr18
    vilvh.b        vr22,   vr17,   vr16
    vilvh.b        vr23,   vr19,   vr18
    vilvl.w        vr16,   vr21,   vr20
    vilvl.w        vr17,   vr23,   vr22
    vilvh.w        vr18,   vr21,   vr20
    vilvh.w        vr19,   vr23,   vr22

    vilvl.d        vr20,   vr16,   vr24
    vilvl.d        vr21,   vr18,   vr26
    vilvh.d        vr22,   vr16,   vr24
    vilvh.d        vr23,   vr18,   vr26
    vst            vr20,   \in16,  \in0
    vst            vr22,   \in16,  \in1
    vst            vr21,   \in16,  \in2
    vst            vr23,   \in16,  \in3

    vilvl.d        vr20,   vr17,   vr25
    vilvl.d        vr21,   vr19,   vr27
    vilvh.d        vr22,   vr17,   vr25
    vilvh.d        vr23,   vr19,   vr27
    vst            vr20,   \in16,  \in4
    vst            vr22,   \in16,  \in5
    vst            vr21,   \in16,  \in6
    vst            vr23,   \in16,  \in7

    vilvh.b        vr16,   vr2,    vr0
    vilvh.b        vr17,   vr3,    vr1
    vilvh.b        vr18,   vr6,    vr4
    vilvh.b        vr19,   vr7,    vr5
    vilvl.b        vr20,   vr17,   vr16
    vilvl.b        vr21,   vr19,   vr18
    vilvh.b        vr22,   vr17,   vr16
    vilvh.b        vr23,   vr19,   vr18
    vilvl.w        vr24,   vr21,   vr20
    vilvl.w        vr25,   vr23,   vr22
    vilvh.w        vr26,   vr21,   vr20
    vilvh.w        vr27,   vr23,   vr22

    vilvh.b        vr16,   vr10,   vr8
    vilvh.b        vr17,   vr11,   vr9
    vilvh.b        vr18,   vr14,   vr12
    vilvh.b        vr19,   vr15,   vr13
    vilvl.b        vr20,   vr17,   vr16
    vilvl.b        vr21,   vr19,   vr18
    vilvh.b        vr22,   vr17,   vr16
    vilvh.b        vr23,   vr19,   vr18
    vilvl.w        vr16,   vr21,   vr20
    vilvl.w        vr17,   vr23,   vr22
    vilvh.w        vr18,   vr21,   vr20
    vilvh.w        vr19,   vr23,   vr22

    vilvl.d        vr20,   vr16,   vr24
    vilvl.d        vr21,   vr18,   vr26
    vilvh.d        vr22,   vr16,   vr24
    vilvh.d        vr23,   vr18,   vr26
    vst            vr20,   \in16,  \in8
    vst            vr22,   \in16,  \in9
    vst            vr21,   \in16,  \in10
    vst            vr23,   \in16,  \in11

    vilvl.d        vr20,   vr17,   vr25
    vilvl.d        vr21,   vr19,   vr27
    vilvh.d        vr22,   vr17,   vr25
    vilvh.d        vr23,   vr19,   vr27
    vst            vr20,   \in16,  \in12
    vst            vr22,   \in16,  \in13
    vst            vr21,   \in16,  \in14
    vst            vr23,   \in16,  \in15
.endm

.macro vld_16x16_b_lsx
    vld            vr0,    a1,     0
    vldx           vr1,    a1,     a2
    alsl.d         a1,     a2,     a1,   1
    vld            vr2,    a1,     0
    vldx           vr3,    a1,     a2
    alsl.d         a1,     a2,     a1,   1
    vld            vr4,    a1,     0
    vldx           vr5,    a1,     a2
    alsl.d         a1,     a2,     a1,   1
    vld            vr6,    a1,     0
    vldx           vr7,    a1,     a2
    alsl.d         a1,     a2,     a1,   1
    vld            vr8,    a1,     0
    vldx           vr9,    a1,     a2
    alsl.d         a1,     a2,     a1,   1
    vld            vr10,   a1,     0
    vldx           vr11,   a1,     a2
    alsl.d         a1,     a2,     a1,   1
    vld            vr12,   a1,     0
    vldx           vr13,   a1,     a2
    alsl.d         a1,     a2,     a1,   1
    vld            vr14,   a1,     0
    vldx           vr15,   a1,     a2
.endm

function x265_transpose_16x16_lsx
    addi.d         sp,     sp,     -32
    fst.d          f24,    sp,     0
    fst.d          f25,    sp,     8
    fst.d          f26,    sp,     16
    fst.d          f27,    sp,     24

    vld_16x16_b_lsx

    transpose_16x16_core_lsx 0, 16, 32, 48, 64, 80, 96, 112, \
                             128, 144, 160, 176, 192, 208, 224, 240, a0

    fld.d          f24,    sp,     0
    fld.d          f25,    sp,     8
    fld.d          f26,    sp,     16
    fld.d          f27,    sp,     24
    addi.d         sp,     sp,     32
endfunc

function x265_transpose_32x32_lsx
    addi.d         sp,     sp,     -32
    fst.d          f24,    sp,     0
    fst.d          f25,    sp,     8
    fst.d          f26,    sp,     16
    fst.d          f27,    sp,     24

    bne            a0,     a1,     .COMPARE_A0A1
    addi.d         sp,     sp,     -256

    addi.d         t0,     a1,     0

    vld_16x16_b_lsx

    transpose_16x16_core_lsx 0, 32, 64, 96, 128, 160, 192, 224, \
                             256, 288, 320, 352, 384, 416, 448, 480, a0

    alsl.d         a1,     a2,     a1,   1

    vld_16x16_b_lsx

    transpose_16x16_core_lsx 0, 16, 32, 48, 64, 80, 96, 112, \
                             128, 144, 160, 176, 192, 208, 224, 240, sp

    addi.d         a1,     t0,     16

    vld_16x16_b_lsx

    vld            vr16,   sp,     0
    vld            vr17,   sp,     16
    vld            vr18,   sp,     32
    vld            vr19,   sp,     48
    vld            vr20,   sp,     64
    vld            vr21,   sp,     80
    vld            vr22,   sp,     96
    vld            vr23,   sp,     112

    vst            vr16,   a0,     16
    vst            vr17,   a0,     48
    vst            vr18,   a0,     80
    vst            vr19,   a0,     112
    vst            vr20,   a0,     144
    vst            vr21,   a0,     176
    vst            vr22,   a0,     208
    vst            vr23,   a0,     240

    addi.d         t5,     sp,     128
    addi.d         t6,     a0,     272-16

    vld            vr16,   t5,     0
    vld            vr17,   t5,     16
    vld            vr18,   t5,     32
    vld            vr19,   t5,     48
    vld            vr20,   t5,     64
    vld            vr21,   t5,     80
    vld            vr22,   t5,     96
    vld            vr23,   t5,     112

    vst            vr16,   t6,     16
    vst            vr17,   t6,     48
    vst            vr18,   t6,     80
    vst            vr19,   t6,     112
    vst            vr20,   t6,     144
    vst            vr21,   t6,     176
    vst            vr22,   t6,     208
    vst            vr23,   t6,     240

    addi.d         a0,     a0,     512

    transpose_16x16_core_lsx 0, 32, 64, 96, 128, 160, 192, 224, \
                             256, 288, 320, 352, 384, 416, 448, 480, a0

    alsl.d         a1,     a2,     a1,   1

    vld_16x16_b_lsx

    transpose_16x16_core_lsx 16, 48, 80, 112, 144, 176, 208, 240, \
                             272, 304, 336, 368, 400, 432, 464, 496, a0

    addi.d         sp,     sp,     256
    b              .TRANSPOSE_32X32_END
.COMPARE_A0A1:

    addi.d         t0,     a1,     0

    vld_16x16_b_lsx

    transpose_16x16_core_lsx 0, 32, 64, 96, 128, 160, 192, 224, \
                             256, 288, 320, 352, 384, 416, 448, 480, a0

    alsl.d         a1,     a2,     a1,   1

    vld_16x16_b_lsx

    transpose_16x16_core_lsx 16, 48, 80, 112, 144, 176, 208, 240, \
                             272, 304, 336, 368, 400, 432, 464, 496, a0

    addi.d         a1,     t0,     16
    addi.d         a0,     a0,     512

    vld_16x16_b_lsx

    transpose_16x16_core_lsx 0, 32, 64, 96, 128, 160, 192, 224, \
                             256, 288, 320, 352, 384, 416, 448, 480, a0

    alsl.d         a1,     a2,     a1,   1

    vld_16x16_b_lsx

    transpose_16x16_core_lsx 16, 48, 80, 112, 144, 176, 208, 240, \
                             272, 304, 336, 368, 400, 432, 464, 496, a0

.TRANSPOSE_32X32_END:
    fld.d          f24,    sp,     0
    fld.d          f25,    sp,     8
    fld.d          f26,    sp,     16
    fld.d          f27,    sp,     24
    addi.d         sp,     sp,     32
endfunc

function x265_transpose_64x64_lsx
    addi.d         sp,     sp,     -32
    fst.d          f24,    sp,     0
    fst.d          f25,    sp,     8
    fst.d          f26,    sp,     16
    fst.d          f27,    sp,     24

    addi.d         t0,     a1,     0
    addi.d         t1,     a0,     0

    vld_16x16_b_lsx

    transpose_16x16_core_lsx 0, 64, 128, 192, 256, 320, 384, 448, \
                             512, 576, 640, 704, 768, 832, 896, 960, a0

.rept 3
    alsl.d         a1,     a2,     a1,   1
    addi.d         a0,     a0,     16

    vld_16x16_b_lsx

    transpose_16x16_core_lsx 0, 64, 128, 192, 256, 320, 384, 448, \
                             512, 576, 640, 704, 768, 832, 896, 960, a0
.endr

.rept 3
    addi.d         a0,     a0,     -48
    addi.d         t0,     t0,     16
    addi.d         a1,     t0,     0
    addi.d         a0,     a0,     1024

    vld_16x16_b_lsx

    transpose_16x16_core_lsx 0, 64, 128, 192, 256, 320, 384, 448, \
                             512, 576, 640, 704, 768, 832, 896, 960, a0

.rept 3
    alsl.d         a1,     a2,     a1,   1
    addi.d         a0,     a0,     16

    vld_16x16_b_lsx

    transpose_16x16_core_lsx 0, 64, 128, 192, 256, 320, 384, 448, \
                             512, 576, 640, 704, 768, 832, 896, 960, a0
.endr
.endr

    fld.d          f24,    sp,     0
    fld.d          f25,    sp,     8
    fld.d          f26,    sp,     16
    fld.d          f27,    sp,     24
    addi.d         sp,     sp,     32
endfunc

function x265_ssimDist4_lsx
    fld.s          f0,     a0,     0
    fld.s          f1,     a2,     0
    fldx.s         f2,     a0,     a1
    fldx.s         f3,     a2,     a3
    alsl.d         a0,     a1,     a0,   1
    alsl.d         a2,     a3,     a2,   1
    fld.s          f4,     a0,     0
    fld.s          f5,     a2,     0
    fldx.s         f6,     a0,     a1
    fldx.s         f7,     a2,     a3

    // ssBlock
    vilvl.w        vr2,    vr2,    vr0
    vilvl.w        vr6,    vr6,    vr4
    vilvl.w        vr3,    vr3,    vr1
    vilvl.w        vr7,    vr7,    vr5
    vilvl.d        vr6,    vr6,    vr2
    vilvl.d        vr7,    vr7,    vr3
    vsubwev.h.bu   vr8,    vr6,    vr7
    vsubwod.h.bu   vr9,    vr6,    vr7
    vmulwev.w.h    vr2,    vr8,    vr8
    vmulwod.w.h    vr3,    vr8,    vr8
    vmulwev.w.h    vr4,    vr9,    vr9
    vmulwod.w.h    vr5,    vr9,    vr9
    vadd.w         vr2,    vr2,    vr3
    vadd.w         vr4,    vr4,    vr5
    vadd.w         vr10,   vr2,    vr4
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.d       vr10,   a4,     0,     0

    // ac_k
    vsllwil.hu.bu  vr22,   vr6,    0
    vexth.hu.bu    vr23,   vr6
    vmulwev.w.hu   vr2,    vr22,   vr22
    vmulwod.w.hu   vr3,    vr22,   vr22
    vmaddwev.w.h   vr2,    vr23,   vr23
    vmaddwod.w.h   vr3,    vr23,   vr23
    vadd.w         vr10,   vr2,    vr3
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.w       vr10,   a6,     0,     0
endfunc

function x265_ssimDist8_lsx

.irp i, vr12, vr13, vr14, vr15, vr16, vr17, vr18, vr19
    vxor.v         \i,     \i,     \i
.endr

.rept 2
    fld.d          f0,     a0,     0
    fld.d          f1,     a2,     0
    fldx.d         f2,     a0,     a1
    fldx.d         f3,     a2,     a3
    alsl.d         a0,     a1,     a0,   1
    alsl.d         a2,     a3,     a2,   1
    fld.d          f4,     a0,     0
    fld.d          f5,     a2,     0
    fldx.d         f6,     a0,     a1
    fldx.d         f7,     a2,     a3

    // ssBlock
    vilvl.d        vr2,    vr2,    vr0
    vilvl.d        vr6,    vr6,    vr4
    vilvl.d        vr3,    vr3,    vr1
    vilvl.d        vr7,    vr7,    vr5
    vsubwev.h.bu   vr8,    vr2,    vr3
    vsubwod.h.bu   vr9,    vr2,    vr3
    vsubwev.h.bu   vr10,   vr6,    vr7
    vsubwod.h.bu   vr11,   vr6,    vr7
    vmaddwev.w.h   vr12,   vr8,    vr8
    vmaddwod.w.h   vr13,   vr8,    vr8
    vmaddwev.w.h   vr14,   vr9,    vr9
    vmaddwod.w.h   vr15,   vr9,    vr9
    vmaddwev.w.h   vr12,   vr10,   vr10
    vmaddwod.w.h   vr13,   vr10,   vr10
    vmaddwev.w.h   vr14,   vr11,   vr11
    vmaddwod.w.h   vr15,   vr11,   vr11

    // ac_k
    vsllwil.hu.bu  vr20,   vr2,    0
    vexth.hu.bu    vr21,   vr2
    vsllwil.hu.bu  vr22,   vr6,    0
    vexth.hu.bu    vr23,   vr6
    vmaddwev.w.hu  vr16,   vr20,   vr20
    vmaddwod.w.hu  vr17,   vr20,   vr20
    vmaddwev.w.hu  vr18,   vr21,   vr21
    vmaddwod.w.hu  vr19,   vr21,   vr21
    vmaddwev.w.hu  vr16,   vr22,   vr22
    vmaddwod.w.hu  vr17,   vr22,   vr22
    vmaddwev.w.hu  vr18,   vr23,   vr23
    vmaddwod.w.hu  vr19,   vr23,   vr23

    alsl.d         a0,     a1,     a0,   1
    alsl.d         a2,     a3,     a2,   1
.endr

    vadd.w         vr12,   vr12,   vr13
    vadd.w         vr14,   vr14,   vr15
    vadd.w         vr10,   vr12,   vr14
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.d       vr10,   a4,     0,     0

    vadd.w         vr4,    vr16,   vr17
    vadd.w         vr8,    vr18,   vr19
    vadd.w         vr10,   vr8,    vr4
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.w       vr10,   a6,     0,     0
endfunc

function x265_ssimDist16_lsx

.irp i, vr12, vr13, vr14, vr15, vr16, vr17, vr18, vr19
    vxor.v         \i,     \i,     \i
.endr

.rept 8
    vld            vr0,    a0,     0
    vld            vr1,    a2,     0
    vldx           vr2,    a0,     a1
    vldx           vr3,    a2,     a3
    alsl.d         a0,     a1,     a0,   1
    alsl.d         a2,     a3,     a2,   1

    // ssBlock
    vsubwev.h.bu   vr8,    vr0,    vr1
    vsubwod.h.bu   vr9,    vr0,    vr1
    vsubwev.h.bu   vr10,   vr2,    vr3
    vsubwod.h.bu   vr11,   vr2,    vr3
    vmaddwev.w.h   vr12,   vr8,    vr8
    vmaddwod.w.h   vr13,   vr8,    vr8
    vmaddwev.w.h   vr14,   vr9,    vr9
    vmaddwod.w.h   vr15,   vr9,    vr9
    vmaddwev.w.h   vr12,   vr10,   vr10
    vmaddwod.w.h   vr13,   vr10,   vr10
    vmaddwev.w.h   vr14,   vr11,   vr11
    vmaddwod.w.h   vr15,   vr11,   vr11

    // ac_k
    vsllwil.hu.bu  vr20,   vr0,    0
    vexth.hu.bu    vr21,   vr0
    vsllwil.hu.bu  vr22,   vr2,    0
    vexth.hu.bu    vr23,   vr2
    vmaddwev.w.hu  vr16,   vr20,   vr20
    vmaddwod.w.hu  vr17,   vr20,   vr20
    vmaddwev.w.hu  vr18,   vr21,   vr21
    vmaddwod.w.hu  vr19,   vr21,   vr21
    vmaddwev.w.hu  vr16,   vr22,   vr22
    vmaddwod.w.hu  vr17,   vr22,   vr22
    vmaddwev.w.hu  vr18,   vr23,   vr23
    vmaddwod.w.hu  vr19,   vr23,   vr23
.endr
    vadd.w         vr12,   vr12,   vr13
    vadd.w         vr14,   vr14,   vr15
    vadd.w         vr10,   vr12,   vr14
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.d       vr10,   a4,     0,     0

    vadd.w         vr16,   vr16,   vr17
    vadd.w         vr18,   vr18,   vr19
    vadd.w         vr10,   vr16,   vr18
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.w       vr10,   a6,     0,     0
endfunc

function x265_ssimDist32_lsx

.irp i, vr12, vr13, vr14, vr15, vr16, vr17, vr18, vr19
    vxor.v         \i,     \i,     \i
.endr

.rept 32
    vld            vr0,    a0,     0
    vld            vr1,    a2,     0
    vld            vr2,    a0,     16
    vld            vr3,    a2,     16
    add.d          a0,     a1,     a0
    add.d          a2,     a3,     a2

    // ssBlock
    vsubwev.h.bu   vr8,    vr0,    vr1
    vsubwod.h.bu   vr9,    vr0,    vr1
    vsubwev.h.bu   vr10,   vr2,    vr3
    vsubwod.h.bu   vr11,   vr2,    vr3
    vmaddwev.w.h   vr12,   vr8,    vr8
    vmaddwod.w.h   vr13,   vr8,    vr8
    vmaddwev.w.h   vr14,   vr9,    vr9
    vmaddwod.w.h   vr15,   vr9,    vr9
    vmaddwev.w.h   vr12,   vr10,   vr10
    vmaddwod.w.h   vr13,   vr10,   vr10
    vmaddwev.w.h   vr14,   vr11,   vr11
    vmaddwod.w.h   vr15,   vr11,   vr11

    // ac_k
    vsllwil.hu.bu  vr20,   vr0,    0
    vexth.hu.bu    vr21,   vr0
    vsllwil.hu.bu  vr22,   vr2,    0
    vexth.hu.bu    vr23,   vr2
    vmaddwev.w.hu  vr16,   vr20,   vr20
    vmaddwod.w.hu  vr17,   vr20,   vr20
    vmaddwev.w.hu  vr18,   vr21,   vr21
    vmaddwod.w.hu  vr19,   vr21,   vr21
    vmaddwev.w.hu  vr16,   vr22,   vr22
    vmaddwod.w.hu  vr17,   vr22,   vr22
    vmaddwev.w.hu  vr18,   vr23,   vr23
    vmaddwod.w.hu  vr19,   vr23,   vr23
.endr
    vadd.w         vr12,   vr12,   vr13
    vadd.w         vr14,   vr14,   vr15
    vadd.w         vr10,   vr12,   vr14
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.d       vr10,   a4,     0,     0

    vadd.w         vr16,   vr16,   vr17
    vadd.w         vr18,   vr18,   vr19
    vadd.w         vr10,   vr16,   vr18
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.w       vr10,   a6,     0,     0
endfunc

function x265_ssimDist64_lsx

.irp i, vr12, vr13, vr14, vr15, vr16, vr17, vr18, vr19
    vxor.v         \i,     \i,     \i
.endr

.rept 64
    vld            vr0,    a0,     0
    vld            vr1,    a2,     0
    vld            vr2,    a0,     16
    vld            vr3,    a2,     16
    vld            vr4,    a0,     32
    vld            vr5,    a2,     32
    vld            vr6,    a0,     48
    vld            vr7,    a2,     48
    add.d          a0,     a1,     a0
    add.d          a2,     a3,     a2

    // ssBlock
    vsubwev.h.bu   vr8,    vr0,    vr1
    vsubwod.h.bu   vr9,    vr0,    vr1
    vsubwev.h.bu   vr10,   vr2,    vr3
    vsubwod.h.bu   vr11,   vr2,    vr3
    vsubwev.h.bu   vr20,   vr4,    vr5
    vsubwod.h.bu   vr21,   vr4,    vr5
    vsubwev.h.bu   vr22,   vr6,    vr7
    vsubwod.h.bu   vr23,   vr6,    vr7

    vmaddwev.w.h   vr12,   vr8,    vr8
    vmaddwod.w.h   vr13,   vr8,    vr8
    vmaddwev.w.h   vr14,   vr9,    vr9
    vmaddwod.w.h   vr15,   vr9,    vr9
    vmaddwev.w.h   vr12,   vr10,   vr10
    vmaddwod.w.h   vr13,   vr10,   vr10
    vmaddwev.w.h   vr14,   vr11,   vr11
    vmaddwod.w.h   vr15,   vr11,   vr11
    vmaddwev.w.h   vr12,   vr20,   vr20
    vmaddwod.w.h   vr13,   vr20,   vr20
    vmaddwev.w.h   vr14,   vr21,   vr21
    vmaddwod.w.h   vr15,   vr21,   vr21
    vmaddwev.w.h   vr12,   vr22,   vr22
    vmaddwod.w.h   vr13,   vr22,   vr22
    vmaddwev.w.h   vr14,   vr23,   vr23
    vmaddwod.w.h   vr15,   vr23,   vr23

    // ac_k
    vsllwil.hu.bu  vr20,   vr0,    0
    vexth.hu.bu    vr21,   vr0
    vsllwil.hu.bu  vr22,   vr2,    0
    vexth.hu.bu    vr23,   vr2
    vsllwil.hu.bu  vr8,    vr4,    0
    vexth.hu.bu    vr9,    vr4
    vsllwil.hu.bu  vr10,   vr6,    0
    vexth.hu.bu    vr11,   vr6

    vmaddwev.w.hu  vr16,   vr20,   vr20
    vmaddwod.w.hu  vr17,   vr20,   vr20
    vmaddwev.w.hu  vr18,   vr21,   vr21
    vmaddwod.w.hu  vr19,   vr21,   vr21
    vmaddwev.w.hu  vr16,   vr22,   vr22
    vmaddwod.w.hu  vr17,   vr22,   vr22
    vmaddwev.w.hu  vr18,   vr23,   vr23
    vmaddwod.w.hu  vr19,   vr23,   vr23
    vmaddwev.w.hu  vr16,   vr8,    vr8
    vmaddwod.w.hu  vr17,   vr8,    vr8
    vmaddwev.w.hu  vr18,   vr9,    vr9
    vmaddwod.w.hu  vr19,   vr9,    vr9
    vmaddwev.w.hu  vr16,   vr10,   vr10
    vmaddwod.w.hu  vr17,   vr10,   vr10
    vmaddwev.w.hu  vr18,   vr11,   vr11
    vmaddwod.w.hu  vr19,   vr11,   vr11
.endr
    vadd.w         vr12,   vr12,   vr13
    vadd.w         vr14,   vr14,   vr15
    vadd.w         vr10,   vr12,   vr14
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.d       vr10,   a4,     0,     0

    vadd.w         vr4,    vr16,   vr17
    vadd.w         vr8,    vr18,   vr19
    vadd.w         vr10,   vr8,    vr4
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.w       vr10,   a6,     0,     0
endfunc

function x265_ssimDist8_lasx

.irp i, xr12, xr13, xr14, xr15, xr16, xr17, xr18, xr19
    xvxor.v        \i,     \i,     \i
.endr

.rept 2
    fld.d          f0,     a0,     0
    fld.d          f1,     a2,     0
    fldx.d         f2,     a0,     a1
    fldx.d         f3,     a2,     a3
    alsl.d         a0,     a1,     a0,   1
    alsl.d         a2,     a3,     a2,   1
    fld.d          f4,     a0,     0
    fld.d          f5,     a2,     0
    fldx.d         f6,     a0,     a1
    fldx.d         f7,     a2,     a3

    // ssBlock
    vilvl.d        vr2,    vr2,    vr0
    vilvl.d        vr6,    vr6,    vr4
    vilvl.d        vr3,    vr3,    vr1
    vilvl.d        vr7,    vr7,    vr5
    xvpermi.q      xr6,    xr2,    0x20
    xvpermi.q      xr7,    xr3,    0x20
    xvsubwev.h.bu  xr10,   xr6,    xr7
    xvsubwod.h.bu  xr11,   xr6,    xr7
    xvmaddwev.w.h  xr12,   xr10,   xr10
    xvmaddwod.w.h  xr13,   xr10,   xr10
    xvmaddwev.w.h  xr14,   xr11,   xr11
    xvmaddwod.w.h  xr15,   xr11,   xr11

    // ac_k
    xvsllwil.hu.bu xr0,    xr6,    0
    xvexth.hu.bu   xr1,    xr6
    xvmaddwev.w.hu xr16,   xr0,    xr0
    xvmaddwod.w.hu xr17,   xr0,    xr0
    xvmaddwev.w.hu xr18,   xr1,    xr1
    xvmaddwod.w.hu xr19,   xr1,    xr1

    alsl.d         a0,     a1,     a0,   1
    alsl.d         a2,     a3,     a2,   1
.endr

    xvadd.w        xr12,   xr12,   xr13
    xvadd.w        xr14,   xr14,   xr15
    xvadd.w        xr10,   xr12,   xr14
    xvpermi.q      xr12,   xr10,   0x01
    vadd.w         vr10,   vr10,   vr12
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.d       vr10,   a4,     0,     0

    xvadd.w        xr8,    xr18,   xr19
    xvadd.w        xr8,    xr8,    xr16
    xvadd.w        xr8,    xr8,    xr17
    xvpermi.q      xr4,    xr8,    0x01
    vadd.w         vr10,   vr8,    vr4
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.w       vr10,   a6,     0,     0
endfunc

function x265_ssimDist16_lasx

.irp i, xr12, xr13, xr14, xr15, xr16, xr17, xr18, xr19
    xvxor.v        \i,     \i,     \i
.endr

.rept 8
    vld            vr0,    a0,     0
    vld            vr1,    a2,     0
    vldx           vr2,    a0,     a1
    vldx           vr3,    a2,     a3
    alsl.d         a0,     a1,     a0,   1
    alsl.d         a2,     a3,     a2,   1

    xvpermi.q      xr2,    xr0,    0x20
    xvpermi.q      xr3,    xr1,    0x20

    // ssBlock
    xvsubwev.h.bu  xr10,   xr2,    xr3
    xvsubwod.h.bu  xr11,   xr2,    xr3
    xvmaddwev.w.h  xr12,   xr10,   xr10
    xvmaddwod.w.h  xr13,   xr10,   xr10
    xvmaddwev.w.h  xr14,   xr11,   xr11
    xvmaddwod.w.h  xr15,   xr11,   xr11

    // ac_k
    xvsllwil.hu.bu xr0,    xr2,    0
    xvexth.hu.bu   xr1,    xr2
    xvmaddwev.w.hu xr16,   xr0,    xr0
    xvmaddwod.w.hu xr17,   xr0,    xr0
    xvmaddwev.w.hu xr18,   xr1,    xr1
    xvmaddwod.w.hu xr19,   xr1,    xr1
.endr

    xvadd.w        xr12,   xr12,   xr13
    xvadd.w        xr14,   xr14,   xr15
    xvadd.w        xr10,   xr12,   xr14
    xvpermi.q      xr12,   xr10,   0x01
    vadd.w         vr10,   vr10,   vr12
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.d       vr10,   a4,     0,     0

    xvadd.w        xr8,    xr18,   xr19
    xvadd.w        xr8,    xr8,    xr16
    xvadd.w        xr8,    xr8,    xr17
    xvpermi.q      xr4,    xr8,    0x01
    vadd.w         vr10,   vr8,    vr4
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.w       vr10,   a6,     0,     0
endfunc

function x265_ssimDist32_lasx

.irp i, xr12, xr13, xr14, xr15, xr16, xr17, xr18, xr19
    xvxor.v        \i,     \i,     \i
.endr

.rept 16
    xvld           xr0,    a0,     0
    xvld           xr1,    a2,     0
    xvldx          xr2,    a0,     a1
    xvldx          xr3,    a2,     a3
    alsl.d         a0,     a1,     a0,   1
    alsl.d         a2,     a3,     a2,   1

    // ssBlock
    xvsubwev.h.bu  xr8,    xr0,    xr1
    xvsubwod.h.bu  xr9,    xr0,    xr1
    xvsubwev.h.bu  xr10,   xr2,    xr3
    xvsubwod.h.bu  xr11,   xr2,    xr3
    xvmaddwev.w.h  xr12,   xr8,    xr8
    xvmaddwod.w.h  xr13,   xr8,    xr8
    xvmaddwev.w.h  xr14,   xr9,    xr9
    xvmaddwod.w.h  xr15,   xr9,    xr9
    xvmaddwev.w.h  xr12,   xr10,   xr10
    xvmaddwod.w.h  xr13,   xr10,   xr10
    xvmaddwev.w.h  xr14,   xr11,   xr11
    xvmaddwod.w.h  xr15,   xr11,   xr11

    // ac_k
    xvsllwil.hu.bu xr8,    xr0,    0
    xvexth.hu.bu   xr9,    xr0
    xvsllwil.hu.bu xr10,   xr2,    0
    xvexth.hu.bu   xr11,   xr2

    xvmaddwev.w.hu xr16,   xr8,    xr8
    xvmaddwod.w.hu xr17,   xr8,    xr8
    xvmaddwev.w.hu xr18,   xr9,    xr9
    xvmaddwod.w.hu xr19,   xr9,    xr9
    xvmaddwev.w.hu xr16,   xr10,   xr10
    xvmaddwod.w.hu xr17,   xr10,   xr10
    xvmaddwev.w.hu xr18,   xr11,   xr11
    xvmaddwod.w.hu xr19,   xr11,   xr11
.endr

    xvadd.w        xr12,   xr12,   xr13
    xvadd.w        xr14,   xr14,   xr15
    xvadd.w        xr10,   xr12,   xr14
    xvpermi.q      xr12,   xr10,   0x01
    vadd.w         vr10,   vr10,   vr12
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.d       vr10,   a4,     0,     0

    xvadd.w        xr9,    xr16,   xr17
    xvadd.w        xr8,    xr18,   xr19
    xvadd.w        xr8,    xr8,    xr9
    xvpermi.q      xr4,    xr8,    0x01
    vadd.w         vr10,   vr8,    vr4
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.w       vr10,   a6,     0,     0
endfunc

function x265_ssimDist64_lasx

.irp i, xr12, xr13, xr14, xr15, xr16, xr17, xr18, xr19
    xvxor.v        \i,     \i,     \i
.endr

.rept 64
    xvld           xr0,    a0,     0
    xvld           xr1,    a2,     0
    xvld           xr2,    a0,     32
    xvld           xr3,    a2,     32
    add.d          a0,     a1,     a0
    add.d          a2,     a3,     a2

    // ssBlock
    xvsubwev.h.bu  xr8,    xr0,    xr1
    xvsubwod.h.bu  xr9,    xr0,    xr1
    xvsubwev.h.bu  xr10,   xr2,    xr3
    xvsubwod.h.bu  xr11,   xr2,    xr3
    xvmaddwev.w.h  xr12,   xr8,    xr8
    xvmaddwod.w.h  xr13,   xr8,    xr8
    xvmaddwev.w.h  xr14,   xr9,    xr9
    xvmaddwod.w.h  xr15,   xr9,    xr9
    xvmaddwev.w.h  xr12,   xr10,   xr10
    xvmaddwod.w.h  xr13,   xr10,   xr10
    xvmaddwev.w.h  xr14,   xr11,   xr11
    xvmaddwod.w.h  xr15,   xr11,   xr11

    // ac_k
    xvsllwil.hu.bu xr8,    xr0,    0
    xvexth.hu.bu   xr9,    xr0
    xvsllwil.hu.bu xr10,   xr2,    0
    xvexth.hu.bu   xr11,   xr2

    xvmaddwev.w.hu xr16,   xr8,    xr8
    xvmaddwod.w.hu xr17,   xr8,    xr8
    xvmaddwev.w.hu xr18,   xr9,    xr9
    xvmaddwod.w.hu xr19,   xr9,    xr9
    xvmaddwev.w.hu xr16,   xr10,   xr10
    xvmaddwod.w.hu xr17,   xr10,   xr10
    xvmaddwev.w.hu xr18,   xr11,   xr11
    xvmaddwod.w.hu xr19,   xr11,   xr11
.endr

    xvadd.w        xr12,   xr12,   xr13
    xvadd.w        xr14,   xr14,   xr15
    xvadd.w        xr10,   xr12,   xr14
    xvpermi.q      xr12,   xr10,   0x01
    vadd.w         vr10,   vr10,   vr12
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.d       vr10,   a4,     0,     0

    xvadd.w        xr9,    xr16,   xr17
    xvadd.w        xr8,    xr18,   xr19
    xvadd.w        xr8,    xr8,    xr9
    xvpermi.q      xr4,    xr8,    0x01
    vadd.w         vr10,   vr8,    vr4
    vhaddw.du.wu   vr10,   vr10,   vr10
    vhaddw.qu.du   vr10,   vr10,   vr10
    vstelm.w       vr10,   a6,     0,     0
endfunc

const IF_INTERNAL_PREC_VALUE
.rept 8
.short 8192
.endr
endconst

function x265_weight_sp_lsx
    slli.d         a2,     a2,     1

    ld.d           t7,     sp,     0  // shift
    ld.d           t8,     sp,     8  // offset
    andi           t0,     a4,     7
    srli.w         t1,     a4,     3
    la.local       t4,     IF_INTERNAL_PREC_VALUE
    vreplgr2vr.h   vr4,    a6         // w0
    vreplgr2vr.w   vr10,   t8         // offset
    vreplgr2vr.w   vr11,   a7         // round
    vreplgr2vr.w   vr12,   t7         // shift
    vld            vr13,   t4,     0  // IF_INTERNAL_PREC
    beqz           t1,     .splooph1

.splooph:
    addi.d         t2,     a0,     0
    addi.d         t3,     a1,     0
    addi.d         t5,     t1,     0
.sploopw:
    vld            vr1,    t2,     0
    vmulwev.w.h    vr2,    vr1,    vr4
    vmulwod.w.h    vr3,    vr1,    vr4
    vmaddwev.w.h   vr2,    vr4,    vr13
    vmaddwod.w.h   vr3,    vr4,    vr13
    vadd.w         vr2,    vr2,    vr11
    vadd.w         vr3,    vr3,    vr11
    vsra.w         vr2,    vr2,    vr12
    vsra.w         vr3,    vr3,    vr12
    vadd.w         vr2,    vr2,    vr10
    vadd.w         vr3,    vr3,    vr10
    vilvl.w        vr6,    vr3,    vr2
    vilvh.w        vr7,    vr3,    vr2
    vssrarni.h.w   vr7,    vr6,    0
    vssrarni.bu.h  vr7,    vr7,    0
    vstelm.d       vr7,    t3,     0,    0
    addi.d         t5,     t5,     -1
    addi.d         t2,     t2,     16
    addi.d         t3,     t3,     8
    beqz           t5,     .spwidthless8
    b              .sploopw
.spwidthless8:

    andi           t0,     a4,     7

    beqz           t0,     .sploophend

    vld            vr1,    t2,     0
    vmulwev.w.h    vr2,    vr1,    vr4
    vmulwod.w.h    vr3,    vr1,    vr4
    vmaddwev.w.h   vr2,    vr4,    vr13
    vmaddwod.w.h   vr3,    vr4,    vr13
    vadd.w         vr2,    vr2,    vr11
    vadd.w         vr3,    vr3,    vr11
    vsra.w         vr2,    vr2,    vr12
    vsra.w         vr3,    vr3,    vr12
    vadd.w         vr2,    vr2,    vr10
    vadd.w         vr3,    vr3,    vr10
    vilvl.w        vr6,    vr3,    vr2
    vilvh.w        vr7,    vr3,    vr2
    vssrarni.h.w   vr7,    vr6,    0
    vssrarni.bu.h  vr7,    vr7,    0
.vstless8:
    vstelm.b       vr7,    t3,     0,    0
    vbsrl.v        vr7,    vr7,    1
    addi.d         t3,     t3,     1
    addi.d         t0,     t0,     -1
    bnez           t0,     .vstless8

.sploophend:
    add.d          a0,     a0,     a2
    add.d          a1,     a1,     a3
    addi.d         a5,     a5,     -1
    bnez           a5,     .splooph
    b              .spend

.splooph1:
    addi.d         t2,     a0,     0
    addi.d         t3,     a1,     0
    andi           t0,     a4,     7

    vld            vr1,    t2,     0
    vmulwev.w.h    vr2,    vr1,    vr4
    vmulwod.w.h    vr3,    vr1,    vr4
    vmaddwev.w.h   vr2,    vr4,    vr13
    vmaddwod.w.h   vr3,    vr4,    vr13
    vadd.w         vr2,    vr2,    vr11
    vadd.w         vr3,    vr3,    vr11
    vsra.w         vr2,    vr2,    vr12
    vsra.w         vr3,    vr3,    vr12
    vadd.w         vr2,    vr2,    vr10
    vadd.w         vr3,    vr3,    vr10
    vilvl.w        vr6,    vr3,    vr2
    vilvh.w        vr7,    vr3,    vr2
    vssrarni.h.w   vr7,    vr6,    0
    vssrarni.bu.h  vr7,    vr7,    0
.vstless81:
    vstelm.b       vr7,    t3,     0,  0
    vbsrl.v        vr7,    vr7,    1
    addi.d         t3,     t3,     1
    addi.d         t0,     t0,     -1
    bnez           t0,     .vstless81

    add.d          a0,     a0,     a2
    add.d          a1,     a1,     a3
    addi.d         a5,     a5,     -1
    bnez           a5,     .splooph1
.spend:
endfunc

const ssim_c1c2
.word 416, 416, 416, 416
.word 235963, 235963, 235963, 235963
endconst

function x265_ssim_end4_lsx
    vld            vr0,    a0,     0
    vld            vr1,    a0,     16
    vld            vr2,    a0,     32
    vld            vr3,    a0,     48
    vld            vr4,    a0,     64
    vld            vr5,    a1,     0
    vld            vr6,    a1,     16
    vld            vr7,    a1,     32
    vld            vr8,    a1,     48
    vld            vr9,    a1,     64

    vadd.w         vr0,    vr0,    vr5
    vadd.w         vr1,    vr1,    vr6
    vadd.w         vr2,    vr2,    vr7
    vadd.w         vr3,    vr3,    vr8
    vadd.w         vr4,    vr4,    vr9

    vadd.w         vr5,    vr0,    vr1
    vadd.w         vr6,    vr1,    vr2
    vadd.w         vr7,    vr2,    vr3
    vadd.w         vr8,    vr3,    vr4
    LSX_TRANSPOSE4x4_W vr5, vr6, vr7, vr8, vr0, vr1, vr2, vr3, \
                       vr9, vr10

    vmul.w         vr4,    vr1,    vr0  // s2*s1
    vmul.w         vr5,    vr1,    vr1  // s2*s2
    vmadd.w        vr5,    vr0,    vr0  // s1*s1 + s2*s2
    vslli.w        vr4,    vr4,    1    // 2 * fs1 * fs2
    vslli.w        vr3,    vr3,    7
    vslli.w        vr2,    vr2,    6
    la.local       t5,     ssim_c1c2
    vsub.w         vr3,    vr3,    vr4  // vocar*2
    vsub.w         vr2,    vr2,    vr5  // vars
    vld            vr9,    t5,     0    // ssim_c1
    vld            vr10,   t5,     16   // ssim_c2
    vadd.w         vr4,    vr4,    vr9
    vadd.w         vr3,    vr3,    vr10
    vadd.w         vr5,    vr5,    vr9
    vadd.w         vr2,    vr2,    vr10

    vffint.s.w     vr4,    vr4
    vffint.s.w     vr3,    vr3
    vffint.s.w     vr5,    vr5
    vffint.s.w     vr2,    vr2

    vfmul.s        vr9,    vr4,    vr3
    vfmul.s        vr10,   vr5,    vr2
    vfdiv.s        vr11,   vr9,    vr10  // ssim
    vreplgr2vr.w   vr0,    zero
.widthless4:
    fadd.s         f0,     f11,    f0
    vbsrl.v        vr11,   vr11,   4
    addi.w         a2,     a2,     -1
    bnez           a2,     .widthless4
endfunc

function x265_ssim_4x4x2_lsx
    addi.d         t0,     a0,     0
    addi.d         t1,     a2,     0

.rept 2
    fld.s          f0,     a0,     0
    fldx.s         f1,     a0,     a1
    alsl.d         a0,     a1,     a0, 1
    fld.s          f2,     a0,     0
    fldx.s         f3,     a0,     a1

    fld.s          f4,     a2,     0
    fldx.s         f5,     a2,     a3
    alsl.d         a2,     a3,     a2, 1
    fld.s          f6,     a2,     0
    fldx.s         f7,     a2,     a3

    vilvl.w        vr1,    vr1,    vr0
    vilvl.w        vr3,    vr3,    vr2
    vilvl.w        vr5,    vr5,    vr4
    vilvl.w        vr7,    vr7,    vr6
    vilvl.d        vr0,    vr3,    vr1
    vilvl.d        vr2,    vr7,    vr5
    vhaddw.hu.bu   vr8,    vr0,    vr0
    vhaddw.hu.bu   vr9,    vr2,    vr2
    vhaddw.wu.hu   vr8,    vr8,    vr8
    vhaddw.wu.hu   vr9,    vr9,    vr9
    vhaddw.du.wu   vr8,    vr8,    vr8
    vhaddw.du.wu   vr9,    vr9,    vr9
    vhaddw.qu.du   vr8,    vr8,    vr8  // s1
    vhaddw.qu.du   vr9,    vr9,    vr9  // s2

    vsllwil.hu.bu  vr1,    vr1,    0
    vsllwil.hu.bu  vr3,    vr3,    0
    vsllwil.hu.bu  vr5,    vr5,    0
    vsllwil.hu.bu  vr7,    vr7,    0

    vmulwev.w.h    vr0,    vr3,    vr3
    vmulwod.w.h    vr4,    vr3,    vr3
    vmulwev.w.h    vr6,    vr1,    vr5
    vmaddwev.w.h   vr0,    vr7,    vr7
    vmaddwod.w.h   vr4,    vr7,    vr7
    vmaddwod.w.h   vr6,    vr1,    vr5
    vmaddwev.w.h   vr0,    vr1,    vr1
    vmaddwod.w.h   vr4,    vr1,    vr1
    vmaddwev.w.h   vr6,    vr3,    vr7
    vmaddwev.w.h   vr0,    vr5,    vr5
    vmaddwod.w.h   vr4,    vr5,    vr5
    vmaddwod.w.h   vr6,    vr3,    vr7
    vadd.w         vr0,    vr0,    vr4
    vhaddw.d.w     vr6,    vr6,    vr6
    vhaddw.d.w     vr0,    vr0,    vr0
    vhaddw.q.d     vr6,    vr6,    vr6  // s12
    vhaddw.q.d     vr0,    vr0,    vr0  // ss

    vilvl.w        vr9,    vr9,    vr8
    vilvl.w        vr6,    vr6,    vr0
    vilvl.d        vr7,    vr6,    vr9
    vst            vr7,    a4,     0

    addi.d         a0,     t0,     4
    addi.d         a2,     t1,     4
    addi.d         a4,     a4,     16
.endr
endfunc
