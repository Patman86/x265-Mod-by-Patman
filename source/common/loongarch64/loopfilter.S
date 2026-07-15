/*
*****************************************************************************
* Copyright (C) 2013-2024 MulticoreWare, Inc
*
* Authors: Hao Chen <chenhao@loongson.cn>
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
*****************************************************************************
*/

#include "loongson_asm.S"

const shuf
.byte 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
endconst

/*x265_saoCuOrgE0_lsx(pixel * rec, int8_t * offsetEo, int width,
                      int8_t* signLeft, intptr_t stride)
*/
function x265_saoCuOrgE0_lsx
    la.local     t2,    shuf
    vldrepl.b    vr0,   a3,    0
    vldi         vr10,  1
    vld          vr11,  a1,    0
    vld          vr12,  t2,    0
    move         t0,    a0
    move         t1,    a2
.loop_e0:
    vld          vr1,   t0,    0
    vld          vr2,   t0,    1
    vslt.bu      vr3,   vr2,   vr1
    vslt.bu      vr4,   vr1,   vr2
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vshuf.b      vr7,   vr6,   vr0,   vr12
    vadd.b       vr8,   vr7,   vr5
    vreplvei.b   vr0,   vr6,   15
    vaddi.bu     vr9,   vr8,   2
    vshuf.b      vr7,   vr11,  vr11,  vr9
    vaddwev.h.bu.b  vr8,   vr1,   vr7
    vaddwod.h.bu.b  vr9,   vr1,   vr7
    vssrani.bu.h vr9,   vr8,   0
    vshuf4i.w    vr3,   vr9,   0x4E
    vilvl.b      vr4,   vr3,   vr9
    vst          vr4,   t0,    0
    addi.d       t1,    t1,    -16
    addi.d       t0,    t0,    16
    bnez         t1,    .loop_e0
    vldrepl.b    vr0,   a3,    1
    add.d        t0,    a0,    a4
.loopH_e0:
    vld          vr1,   t0,    0
    vld          vr2,   t0,    1
    vslt.bu      vr3,   vr2,   vr1
    vslt.bu      vr4,   vr1,   vr2
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vshuf.b      vr7,   vr6,   vr0,   vr12
    vadd.b       vr8,   vr7,   vr5
    vreplvei.b   vr0,   vr6,   15
    vaddi.bu     vr9,   vr8,   2
    vshuf.b      vr7,   vr11,  vr11,  vr9
    vaddwev.h.bu.b  vr8,   vr1,   vr7
    vaddwod.h.bu.b  vr9,   vr1,   vr7
    vssrani.bu.h vr9,   vr8,   0
    vshuf4i.w    vr3,   vr9,   0x4E
    vilvl.b      vr4,   vr3,   vr9
    vst          vr4,   t0,    0
    addi.d       a2,    a2,    -16
    addi.d       t0,    t0,    16
    bnez         a2,    .loopH_e0
endfunc

/*x265_saoCuOrgE1_lsx(pixel* rec, int8_t* upBuff1,
                      int8_t* offsetEo, intptr_t stride, int width)
*/
function x265_saoCuOrgE1_lsx
    vldi         vr10,  1
    vld          vr11,  a2,    0
    srli.d       t0,    a4,    4
.loop_e1:
    vld          vr0,   a0,    0
    vldx         vr1,   a0,    a3
    vld          vr2,   a1,    0
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vshuf.b      vr9,   vr11,  vr11,   vr8
    vaddwev.h.bu.b  vr3,   vr0,   vr9
    vaddwod.h.bu.b  vr4,   vr0,   vr9
    vssrani.bu.h vr4,   vr3,   0
    vshuf4i.w    vr5,   vr4,   0x4E
    addi.d       t0,    t0,    -1
    vilvl.b      vr7,   vr5,   vr4
    vst          vr6,   a1,    0
    vst          vr7,   a0,    0
    addi.d       a1,    a1,    16
    addi.d       a0,    a0,    16
    bnez         t0,    .loop_e1
endfunc

/*x265_saoCuOrgE1_2Rows_lsx(pixel* rec, int8_t* upBuff1, int8_t* offsetEo,
                            intptr_t stride, int width)
*/
function x265_saoCuOrgE1_2Rows_lsx
    vldi         vr10,  1
    vld          vr11,  a2,    0
    srli.d       t0,    a4,    4
    move         t1,    a0
    move         t2,    a1
    move         t3,    t0
.loop_2rows_e1:
    vld          vr0,   t1,    0
    vldx         vr1,   t1,    a3
    vld          vr2,   t2,    0
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vshuf.b      vr9,   vr11,  vr11,   vr8
    vaddwev.h.bu.b  vr3,   vr0,   vr9
    vaddwod.h.bu.b  vr4,   vr0,   vr9
    vssrani.bu.h vr4,   vr3,   0
    vshuf4i.w    vr5,   vr4,   0x4E
    addi.d       t0,    t0,    -1
    vilvl.b      vr7,   vr5,   vr4
    vst          vr6,   t2,    0
    vst          vr7,   t1,    0
    addi.d       t2,    t2,    16
    addi.d       t1,    t1,    16
    bnez         t0,    .loop_2rows_e1
    add.d        a0,    a0,    a3
.loopH_2rows_e1:
    vld          vr0,   a0,    0
    vldx         vr1,   a0,    a3
    vld          vr2,   a1,    0
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vshuf.b      vr9,   vr11,  vr11,   vr8
    vaddwev.h.bu.b  vr3,   vr0,   vr9
    vaddwod.h.bu.b  vr4,   vr0,   vr9
    vssrani.bu.h vr4,   vr3,   0
    vshuf4i.w    vr5,   vr4,   0x0E
    addi.d       t3,    t3,    -1
    vilvl.b      vr7,   vr5,   vr4
    vst          vr6,   a1,    0
    vst          vr7,   a0,    0
    addi.d       a1,    a1,    16
    addi.d       a0,    a0,    16
    bnez         t3,    .loopH_2rows_e1
endfunc

/*x265_saoCuOrgE2_lsx(pixel * rec, int8_t * bufft, int8_t * buff1,
                      int8_t * offsetEo, int width, intptr_t stride)
*/
function x265_saoCuOrgE2_lsx
    vldi         vr10,  1
    vld          vr11,  a3,    0
    srli.d       t0,    a4,    4
    andi         t1,    a4,    0xF
    addi.d       a5,    a5,    1
    addi.d       a1,    a1,    1
    beqz         t0,    .loop_end_e2
.loop_e2:
    vld          vr0,   a0,    0
    vldx         vr1,   a0,    a5
    vld          vr2,   a2,    0
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vshuf.b      vr9,   vr11,  vr11,   vr8
    vaddwev.h.bu.b  vr3,   vr0,   vr9
    vaddwod.h.bu.b  vr4,   vr0,   vr9
    vssrani.bu.h vr4,   vr3,   0
    vshuf4i.w    vr5,   vr4,   0x04E
    addi.d       t0,    t0,    -1
    vilvl.b      vr7,   vr5,   vr4
    vst          vr6,   a1,    0
    vst          vr7,   a0,    0
    addi.d       a1,    a1,    16
    addi.d       a0,    a0,    16
    addi.d       a2,    a2,    16
    blt          zero,  t0,    .loop_e2
.loop_end_e2:
    beqz         t1,    .end_e2
    vld          vr0,   a0,    0
    vldx         vr1,   a0,    a5
    vld          vr2,   a2,    0
    andi         t3,    t1,    0x8
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vshuf.b      vr9,   vr11,  vr11,   vr8
    vaddwev.h.bu.b  vr3,   vr0,   vr9
    vaddwod.h.bu.b  vr4,   vr0,   vr9
    vssrani.bu.h vr4,   vr3,   0
    vshuf4i.w    vr5,   vr4,   0x4E
    vilvl.b      vr7,   vr5,   vr4
    beqz         t3,    .e2_res
    fst.d        f6,    a1,    0
    fst.d        f7,    a0,    0
    addi.d       a1,    a1,    8
    addi.d       a0,    a0,    8
    addi.d       t1,    t1,    -8
    vshuf4i.w    vr6,   vr6,   0x4E
    vshuf4i.w    vr7,   vr7,   0x4E
    beqz         t1,    .end_e2
.e2_res:
    vilvl.d      vr7,   vr7,   vr6
.loop_e2_res:
    vstelm.b     vr7,   a1,    0,    0
    vstelm.b     vr7,   a0,    0,    8
    addi.d       a1,    a1,    1
    addi.d       a0,    a0,    1
    addi.d       t1,    t1,    -1
    vbsrl.v      vr7,   vr7,   1
    blt          zero,  t1,    .loop_e2_res
.end_e2:
endfunc

/*x265_saoCuOrgE3_lsx(pixel *rec, int8_t *upBuff1, int8_t *offsetEo,
                      intptr_t stride, int startX, int endX)
*/
function x265_saoCuOrgE3_lsx
    add.d        a1,    a1,    a4
    addi.d       a4,    a4,    1
    vldi         vr10,  1
    vld          vr11,  a2,    0
    sub.d        t0,    a5,    a4
    add.d        a0,    a0,    a4
    srli.d       t1,    t0,    4
    andi         t2,    t0,    0xF
    beqz         t1,    .loop_end_e3
.loop_e3:
    vld          vr0,   a0,    0
    vldx         vr1,   a0,    a3
    vld          vr2,   a1,    1
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vshuf.b      vr9,   vr11,  vr11,   vr8
    vaddwev.h.bu.b  vr3,   vr0,   vr9
    vaddwod.h.bu.b  vr4,   vr0,   vr9
    vssrani.bu.h vr4,   vr3,   0
    vshuf4i.w    vr5,   vr4,   0x04E
    addi.d       t1,    t1,    -1
    vilvl.b      vr7,   vr5,   vr4
    vst          vr6,   a1,    0
    vst          vr7,   a0,    0
    addi.d       a1,    a1,    16
    addi.d       a0,    a0,    16
    blt          zero,  t1,    .loop_e3
.loop_end_e3:
    beqz         t2,    .end_e3
    vld          vr0,   a0,    0
    vldx         vr1,   a0,    a3
    vld          vr2,   a1,    1
    andi         t3,    t2,    0x8
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vshuf.b      vr9,   vr11,  vr11,   vr8
    vaddwev.h.bu.b  vr3,   vr0,   vr9
    vaddwod.h.bu.b  vr4,   vr0,   vr9
    vssrani.bu.h vr4,   vr3,   0
    vshuf4i.w    vr5,   vr4,   0x4E
    vilvl.b      vr7,   vr5,   vr4
    beqz         t3,    .e3_res
    fst.d        f6,    a1,    0
    fst.d        f7,    a0,    0
    addi.d       a1,    a1,    8
    addi.d       a0,    a0,    8
    addi.d       t2,    t2,    -8
    vshuf4i.w    vr6,   vr6,   0x4E
    vshuf4i.w    vr7,   vr7,   0x4E
    beqz         t2,    .end_e3
.e3_res:
    vilvl.d      vr7,   vr7,   vr6
.loop_e3_res:
    vstelm.b     vr7,   a1,    0,    0
    vstelm.b     vr7,   a0,    0,    8
    addi.d       a1,    a1,    1
    addi.d       a0,    a0,    1
    addi.d       t2,    t2,    -1
    vbsrl.v      vr7,   vr7,   1
    blt          zero,  t2,    .loop_e3_res
.end_e3:
endfunc

/*x265_saoCuOrgB0_lsx(pixel* rec, const int8_t* offset, int ctuWidth,
                      int ctuHeight, intptr_t stride)
*/
function x265_saoCuOrgB0_lsx
    srli.d       t0,    a3,    1
    srli.d       t1,    a2,    4
    slli.d       t2,    a4,    1
    andi         t3,    a3,    1
    vld          vr0,   a1,    0
    vld          vr1,   a1,    16
    beqz         t0,    .loopH_b0_end
.loopH_b0:
    move         t4,    a0
    move         t5,    t1
.loopW_b0:
    vld          vr2,   t4,    0
    vldx         vr3,   t4,    a4
    vsrli.b      vr4,   vr2,   3
    vsrli.b      vr5,   vr3,   3
    vshuf.b      vr6,   vr1,   vr0,  vr4
    vshuf.b      vr7,   vr1,   vr0,  vr5
    vaddwev.h.bu.b  vr8,  vr2,  vr6
    vaddwev.h.bu.b  vr10, vr3,  vr7
    vaddwod.h.bu.b  vr9,  vr2,  vr6
    vaddwod.h.bu.b  vr11, vr3,  vr7
    vssrani.bu.h vr10,  vr8,   0
    vssrani.bu.h vr11,  vr9,   0
    vilvl.b      vr12,  vr11,  vr10
    vilvh.b      vr13,  vr11,  vr10
    vst          vr12,  t4,    0
    vstx         vr13,  t4,    a4
    addi.d       t5,    t5,    -1
    addi.d       t4,    t4,    16
    blt          zero,  t5,    .loopW_b0
    addi.d       t0,    t0,    -1
    add.d        a0,    a0,    t2
    blt          zero,  t0,    .loopH_b0
.loopH_b0_end:
    beqz         t3,    .end_b0
.loopW1_b0:
    vld          vr2,   a0,    0
    vsrli.b      vr4,   vr2,   3
    vshuf.b      vr6,   vr1,   vr0,  vr4
    vaddwev.h.bu.b  vr8,  vr2,  vr6
    vaddwod.h.bu.b  vr10, vr2,  vr6
    vssrani.bu.h vr10,  vr8,   0
    vshuf4i.w    vr11,  vr10,  0x4E
    vilvl.b      vr12,  vr11,  vr10
    vst          vr12,  a0,    0
    addi.d       t1,    t1,    -1
    addi.d       a0,    a0,    16
    blt          zero,  t1,    .loopW1_b0
.end_b0:
endfunc

/* x265_calSign_lsx(int8_t *dst, const pixel *src1, const pixel *src2, const int endX) */
function x265_calSign_lsx
    srli.d       t0,    a3,    4
    andi         t1,    a3,    15
    vldi         vr10,  1
    beqz         t0,    .loop_sign_lsx_end
.loop_sign_lsx:
    vld          vr0,   a1,    0
    vld          vr1,   a2,    0
    addi.d       t0,    t0,    -1
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vst          vr5,   a0,    0
    addi.d       a1,    a1,    16
    addi.d       a2,    a2,    16
    addi.d       a0,    a0,    16
    blt          zero,  t0,    .loop_sign_lsx
.loop_sign_lsx_end:
    beqz         t1,    .end_sign_lsx
    vld          vr0,   a1,    0
    vld          vr1,   a2,    0
    andi         t2,    t1,    8
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    beqz         t2,    .loop_sign_res_lsx
    fst.d        f5,    a0,    0
    addi.d       t1,    t1,    -8
    addi.d       a0,    a0,    8
    vshuf4i.w    vr5,   vr5,   0x4E
    beqz         t1,    .end_sign_lsx
.loop_sign_res_lsx:
    vstelm.b     vr5,   a0,    0,    0
    addi.d       t1,    t1,    -1
    addi.d       a0,    a0,    1
    vbsrl.v      vr5,   vr5,   1
    blt          zero,  t1,    .loop_sign_res_lsx
.end_sign_lsx:
endfunc

/* x265_calSign_lasx(int8_t *dst, const pixel *src1, const pixel *src2, const int endX) */
function x265_calSign_lasx
    srli.d       t0,    a3,    5
    andi         t1,    a3,    31
    xvldi        xr10,  1
    beqz         t0,    .loop_sign_lasx_end
.loop_sign_lasx:
    xvld         xr0,   a1,    0
    xvld         xr1,   a2,    0
    addi.d       t0,    t0,    -1
    xvslt.bu     xr3,   xr1,   xr0
    xvslt.bu     xr4,   xr0,   xr1
    xvand.v      xr3,   xr3,   xr10
    xvor.v       xr5,   xr3,   xr4
    xvst         xr5,   a0,    0
    addi.d       a1,    a1,    32
    addi.d       a2,    a2,    32
    addi.d       a0,    a0,    32
    blt          zero,  t0,    .loop_sign_lasx
.loop_sign_lasx_end:
    beqz         t1,    .end_sign_lasx
    xvld         xr0,   a1,    0
    xvld         xr1,   a2,    0
    andi         t2,    t1,    16
    xvslt.bu     xr3,   xr1,   xr0
    xvslt.bu     xr4,   xr0,   xr1
    xvand.v      xr3,   xr3,   xr10
    xvor.v       xr5,   xr3,   xr4
    beqz         t2,    .loop_sign_res_lasx
    vst          vr5,   a0,    0
    addi.d       t1,    t1,    -16
    addi.d       a0,    a0,    16
    xvpermi.d    xr5,   xr5,   0x4E
    beqz         t1,    .end_sign_lasx
.loop_sign_res_lasx:
    vstelm.b     vr5,   a0,    0,    0
    addi.d       t1,    t1,    -1
    addi.d       a0,    a0,    1
    vbsrl.v      vr5,   vr5,   1
    blt          zero,  t1,    .loop_sign_res_lasx
.end_sign_lasx:
endfunc

/* x265_pelFilterLumaStrong_V_lsx(pixel* src, intptr_t srcStep, intptr_t offset,
                                  int32_t tcP, int32_t tcQ)
*/
function x265_pelFilterLumaStrong_V_lsx
    addi.d       a0,    a0,    -3
    vreplgr2vr.h vr0,   a3   //tcP
    vreplgr2vr.h vr1,   a4   //tcQ
    add.d        t0,    a0,    a1
    add.d        t1,    t0,    a1
    add.d        t2,    t1,    a1
    vilvl.d      vr0,   vr1,   vr0  //[tcP, tcQ]
    vldi         vr10,  0
    vneg.h       vr1,   vr0         //[-tcP, -tcQ]

    fld.d        f2,    a0,    -1   //src[m0,m1, m2, m3, m4, m5,m6, m7]
    fld.d        f3,    t0,    -1
    fld.d        f4,    t1,    -1
    fld.d        f5,    t2,    -1

    vilvl.b      vr6,   vr3,   vr2 //[m0, m0, ... m3, m3, m4, m4 ... m7, m7]
    vilvl.b      vr7,   vr5,   vr4 //[m0, m0]

    vpackev.h    vr2,   vr7,   vr6 //[m0, m2, m4, m6]
    vpackod.h    vr3,   vr7,   vr6 //[m1, m3, m5, m7]

    vilvl.b      vr4,   vr10,  vr2  //[m0, m2]
    vilvh.b      vr5,   vr10,  vr2  //[m4, m6]
    vilvl.b      vr6,   vr10,  vr3  //[m1, m3]
    vilvh.b      vr7,   vr10,  vr3  //[m5, m7]

    vadd.h       vr8,   vr4,   vr6  //[m0+m1, m2+m3]
    vadd.h       vr9,   vr5,   vr7  //[m4+m5, m6+m7]

    vshuf4i.w    vr11,  vr8,   0x4E //[m2+m3, m0+m1]
    vslli.h      vr14,  vr8,   1    //2*[m0+m1, m2+m3]
    vslli.h      vr15,  vr9,   1    //2*[m4+m5, m6+m7]
    vadd.h       vr2,   vr5,   vr6  //[m1+m4, m6+m3]
    vadd.h       vr12,  vr9,   vr11 //[m2+m3+m4+m5, m0+m1+m6+m7]
    vpermi.w     vr15,  vr14,  0xE4 //2*[m0+m1, m6+m7]

    vpermi.w     vr9,   vr8,   0x4E //[m2+m3, m4+m5]
    vpermi.w     vr7,   vr4,   0x4E //[m2, m5]
    vshuf4i.w    vr5,   vr5,   0x4E //[m6, m4]
    vshuf4i.w    vr12,  vr12,  0x44 //[m2+m3+m4+m5, m2+m3+m4+m5]
    vadd.h       vr3,   vr2,   vr9  //[m1+m2+m3+m4, m3+m4+m5+m6]
    vilvl.d      vr11,  vr5,   vr6  //[m1, m6]
    vilvh.d      vr13,  vr5,   vr6  //[m3, m4]
    vsrlri.h     vr4,   vr3,   2    //([m1+m2+m3+m4, m3+m4+m5+m6]+2)>>2
    vadd.h       vr5,   vr15,  vr3  //[2m0+2m1+m1+m2+m3+m4, m3+m4+m5+m6+2m6+2m7]
    vadd.h       vr6,   vr12,  vr3  //[m1+2m2+2m3+2m4+m5, m2+2m3+2m4+2m5+m6]
    vsrlri.h     vr5,   vr5,   3
    vsrlri.h     vr6,   vr6,   3
    vsub.h       vr4,   vr4,   vr7  //([m1+m2+m3+m4, m3+m4+m5+m6]+2)>>2 - [m2, m5]
    vsub.h       vr5,   vr5,   vr11 //([2m0+2m1+m1+m2+m3+m4, m3+m4+m5+m6+2m6+2m7]+4)>>3 - [m1, m6]
    vsub.h       vr6,   vr6,   vr13 //([m1+2m2+2m3+2m4+m5, m2+2m3+2m4+2m5+m6]+4)>>3 - [m3, m4]
    vclip.h      vr3,   vr4,   vr1,   vr0  //[-2, 1]
    vclip.h      vr2,   vr5,   vr1,   vr0  //[-3, 2]
    vclip.h      vr4,   vr6,   vr1,   vr0  //[-1, 0]

    vadd.h       vr2,   vr2,   vr11   //[-3, 2]
    vadd.h       vr3,   vr3,   vr7    //[-2, 1]
    vadd.h       vr4,   vr4,   vr13   //[-1, 0]
    vpackev.b    vr5,   vr3,   vr2    //[-3,-2, 2, 1]
    vshuf4i.w    vr14,  vr4,   0x4E   //[0, -1]
    vpackev.b    vr6,   vr2,   vr3    //[-2,-3, 1, 2]
    vpackev.b    vr7,   vr14,  vr4    //[-1, 0, 0, -1]
    vilvl.h      vr8,   vr7,   vr5    //[-3,-2,-1,0]
    vstelm.w     vr8,   a0,    0,     0
    vstelm.w     vr8,   t0,    0,     1
    vstelm.w     vr8,   t1,    0,     2
    vstelm.w     vr8,   t2,    0,     3
    vstelm.h     vr6,   a0,    4,     4
    vstelm.h     vr6,   t0,    4,     5
    vstelm.h     vr6,   t1,    4,     6
    vstelm.h     vr6,   t2,    4,     7
endfunc

function x265_pelFilterLumaStrong_H_lsx
    slli.d       t0,    a2,    1    //offset * 2
    slli.d       t2,    a2,    2    //offset * 4
    add.d        t1,    a2,    t0   //offset * 3
    vreplgr2vr.h vr0,   a3   //tcP
    vreplgr2vr.h vr1,   a4   //tcQ
    sub.d        t3,    a0,    t2
    vilvl.d      vr0,   vr1,   vr0  //[tcP, tcQ]
    vldi         vr10,  0
    vneg.h       vr1,   vr0         //[-tcP, -tcQ]

    fld.s        f2,    t3,    0    //[m0]
    fldx.s       f3,    t3,    a2   //[m1]
    fldx.s       f4,    t3,    t0   //[m2]
    fldx.s       f5,    t3,    t1   //[m3]
    fld.s        f6,    a0,    0    //[m4]
    fldx.s       f7,    a0,    a2   //[m5]
    fldx.s       f8,    a0,    t0   //[m6]
    fldx.s       f9,    a0,    t1   //[m7]

    vilvl.w      vr11,  vr4,   vr2  //[m0, m2]
    vilvl.w      vr12,  vr8,   vr6  //[m4, m6]
    vilvl.w      vr13,  vr5,   vr3  //[m1, m3]
    vilvl.w      vr14,  vr9,   vr7  //[m5, m7]

    vilvl.b      vr4,   vr10,  vr11  //[m0, m2]
    vilvl.b      vr5,   vr10,  vr12  //[m4, m6]
    vilvl.b      vr6,   vr10,  vr13  //[m1, m3]
    vilvl.b      vr7,   vr10,  vr14  //[m5, m7]

    vadd.h       vr8,   vr4,   vr6  //[m0+m1, m2+m3]
    vadd.h       vr9,   vr5,   vr7  //[m4+m5, m6+m7]

    vshuf4i.w    vr11,  vr8,   0x4E //[m2+m3, m0+m1]
    vslli.h      vr14,  vr8,   1    //2*[m0+m1, m2+m3]
    vslli.h      vr15,  vr9,   1    //2*[m4+m5, m6+m7]
    vadd.h       vr2,   vr5,   vr6  //[m1+m4, m6+m3]
    vadd.h       vr12,  vr9,   vr11 //[m2+m3+m4+m5, m0+m1+m6+m7]
    vpermi.w     vr15,  vr14,  0xE4 //2*[m0+m1, m6+m7]

    vpermi.w     vr9,   vr8,   0x4E //[m2+m3, m4+m5]
    vpermi.w     vr7,   vr4,   0x4E //[m2, m5]
    vshuf4i.w    vr5,   vr5,   0x4E //[m6, m4]
    vshuf4i.w    vr12,  vr12,  0x44 //[m2+m3+m4+m5, m2+m3+m4+m5]
    vadd.h       vr3,   vr2,   vr9  //[m1+m2+m3+m4, m3+m4+m5+m6]
    vilvl.d      vr11,  vr5,   vr6  //[m1, m6]
    vilvh.d      vr13,  vr5,   vr6  //[m3, m4]
    vsrlri.h     vr4,   vr3,   2    //([m1+m2+m3+m4, m3+m4+m5+m6]+2)>>2
    vadd.h       vr5,   vr15,  vr3  //[2m0+2m1+m1+m2+m3+m4, m3+m4+m5+m6+2m6+2m7]
    vadd.h       vr6,   vr12,  vr3  //[m1+2m2+2m3+2m4+m5, m2+2m3+2m4+2m5+m6]
    vsrlri.h     vr5,   vr5,   3
    vsrlri.h     vr6,   vr6,   3
    vsub.h       vr4,   vr4,   vr7  //([m1+m2+m3+m4, m3+m4+m5+m6]+2)>>2 - [m2, m5]
    vsub.h       vr5,   vr5,   vr11 //([2m0+2m1+m1+m2+m3+m4, m3+m4+m5+m6+2m6+2m7]+4)>>3 - [m1, m6]
    vsub.h       vr6,   vr6,   vr13 //([m1+2m2+2m3+2m4+m5, m2+2m3+2m4+2m5+m6]+4)>>3 - [m3, m4]
    vclip.h      vr3,   vr4,   vr1,   vr0  //[-2, 1]
    vclip.h      vr2,   vr5,   vr1,   vr0  //[-3, 2]
    vclip.h      vr4,   vr6,   vr1,   vr0  //[-1, 0]

    vadd.h       vr2,   vr2,   vr11   //[-3, 2]
    vadd.h       vr3,   vr3,   vr7    //[-2, 1]
    vadd.h       vr4,   vr4,   vr13   //[-1, 0]

    sub.d        t3,    a0,    t1   //-offset*3
    sub.d        t4,    a0,    t0   //-offset*2
    sub.d        t5,    a0,    a2   //-offset
    add.d        t6,    a0,    a2   //offset
    add.d        t7,    a0,    t0   //offset*2
    vpickev.b    vr5,   vr3,   vr2   //[-3, 2, -2, 1]
    vpickev.b    vr6,   vr10,  vr4   //[-1, 0]

    vstelm.w     vr5,   t3,    0,    0
    vstelm.w     vr5,   t4,    0,    2
    vstelm.w     vr5,   t6,    0,    3
    vstelm.w     vr5,   t7,    0,    1
    vstelm.w     vr6,   t5,    0,    0
    vstelm.w     vr6,   a0,    0,    1
endfunc

/*x265_pelFilterChroma_V_lsx(pixel* src, intptr_t srcStep, intptr_t offset,
                             int32_t tc, int32_t maskP, int32_t maskQ)
*/
function x265_pelFilterChroma_V_lsx
    addi.d       a0,    a0,    -1
    vreplgr2vr.h vr0,   a3   //tcP
    vldi         vr10,  0
    add.d        t0,    a0,    a1
    add.d        t1,    t0,    a1
    add.d        t2,    t1,    a1
    vneg.h       vr1,   vr0         //[-tcP, -tcQ]
    vreplgr2vr.h vr2,   a4   //maskP
    vreplgr2vr.h vr3,   a5   //maskQ

    fld.s        f4,    a0,    -1   //src[m2, m3, m4, m5]
    fld.s        f5,    t0,    -1
    fld.s        f6,    t1,    -1
    fld.s        f7,    t2,    -1

    vilvl.b      vr8,   vr5,   vr4   //[m2, m2, m3, m3, m4, m4, m5, m5]
    vilvl.b      vr9,   vr7,   vr6
    vilvl.h      vr11,  vr9,   vr8   //[m2, m3, m4, m5]

    vilvl.b      vr4,   vr10,  vr11  //[m2, m3]
    vilvh.b      vr6,   vr10,  vr11  //[m4, m5]
    vbsrl.v      vr5,   vr4,   8     //[m3]
    vbsrl.v      vr7,   vr6,   8     //[m5]

    vsub.h       vr8,   vr6,   vr5   //[m4-m3]
    vsub.h       vr9,   vr4,   vr7   //[m2-m5]
    vslli.h      vr8,   vr8,   2     //[m4-m3] * 4
    vadd.h       vr11,  vr8,   vr9
    vsrari.h     vr12,  vr11,  3
    vclip.h      vr13,  vr12,  vr1,  vr0  //[delta]
    vand.v       vr14,  vr13,  vr2   //-offset
    vand.v       vr15,  vr13,  vr3   //0
    vadd.h       vr8,   vr5,   vr14
    vsub.h       vr9,   vr6,   vr15

    vilvl.h      vr11,  vr9,   vr8
    vmaxi.h      vr13,  vr11,  0
    vsat.hu      vr14,  vr13,  7
    vpickev.b    vr12,  vr10,  vr14
    vstelm.h     vr12,  a0,    0,     0
    vstelm.h     vr12,  t0,    0,     1
    vstelm.h     vr12,  t1,    0,     2
    vstelm.h     vr12,  t2,    0,     3
endfunc

function x265_pelFilterChroma_H_lsx
    slli.d       t0,    a2,    1    //offset * 2
    vreplgr2vr.h vr0,   a3   //tcP
    vldi         vr10,  0
    sub.d        t1,    a0,    t0
    vneg.h       vr1,   vr0         //[-tcP, -tcQ]
    vreplgr2vr.h vr2,   a4   //maskP
    vreplgr2vr.h vr3,   a5   //maskQ

    fld.s        f4,    t1,    0   //src[m2]
    fldx.s       f5,    t1,    a2  //src[m3]
    fld.s        f6,    a0,    0   //src[m4]
    fldx.s       f7,    a0,    a2  //src[m5]

    vilvl.b      vr4,   vr10,  vr4
    vilvl.b      vr5,   vr10,  vr5
    vilvl.b      vr6,   vr10,  vr6
    vilvl.b      vr7,   vr10,  vr7

    vsub.h       vr8,   vr6,   vr5   //[m4-m3]
    vsub.h       vr9,   vr4,   vr7   //[m2-m5]
    vslli.h      vr8,   vr8,   2     //[m4-m3] * 4
    vadd.h       vr11,  vr8,   vr9
    vsrari.h     vr12,  vr11,  3
    vclip.h      vr13,  vr12,  vr1,  vr0  //[delta]
    vand.v       vr14,  vr13,  vr2   //-offset
    vand.v       vr15,  vr13,  vr3   //0
    vadd.h       vr8,   vr5,   vr14
    vsub.h       vr9,   vr6,   vr15

    vssrani.bu.h vr9,   vr8,   0
    fstx.s       f9,    t1,    a2
    vstelm.w     vr9,   a0,    0,    2
endfunc

.macro STATS_COUNT in0, in1, i0, i1
    vpickve2gr.b t4,    \in0,  \i0
    vpickve2gr.h t8,    \in1,  \i1
    ldx.w        t6,    sp,    t4
    ldx.w        t7,    t5,    t4
    add.w        t6,    t6,    t8
    addi.w       t7,    t7,    1
    stx.w        t6,    sp,    t4
    stx.w        t7,    t5,    t4
.endm

/*x265_saoCuStatsE0_lsx(const int16_t *diff, const pixel *rec, intptr_t stride, int endX,
                        int endY, int32_t *stats, int32_t *count)
*/
function x265_saoCuStatsE0_lsx
    la.local     t2,    shuf
    addi.d       sp,    sp,    -48
    vldi         vr10,  1
    vldi         vr11,  0
    vld          vr12,  t2,    0
    addi.d       t5,    sp,    20
    vst          vr11,  sp,    0
    vst          vr11,  sp,    16
    vst          vr11,  sp,    32
.loopS_Y_e0:
    ld.bu        t3,    a1,    0
    ld.bu        t4,    a1,    -1
    addi.d       a7,    zero,  -1
    sltu         t7,    t3,    t4   //rec[0] < rec[-1] ? 1:0
    sltu         t6,    t4,    t3   //rec[-1] < rec[0] ? 1:0
    maskeqz      t8,    a7,    t7
    or           t6,    t6,    t8
    vreplgr2vr.b vr0,   t6
    srli.d       t0,    a3,    4
    move         t3,    a1
    move         t2,    a0
    beqz         t0,    .ResS_X_e0
.loopS_X_e0:
    vld          vr1,   t3,    0
    vld          vr2,   t3,    1
    vld          vr13,  t2,    0
    vld          vr14,  t2,    16
    vslt.bu      vr3,   vr2,   vr1
    vslt.bu      vr4,   vr1,   vr2
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4   //signRight
    vneg.b       vr6,   vr5          //signLeft
    vshuf.b      vr7,   vr6,   vr0,   vr12
    vadd.b       vr8,   vr7,   vr5
    vreplvei.b   vr0,   vr6,   15
    vaddi.bu     vr9,   vr8,   2    //edgeType
    vslli.b      vr9,   vr9,   2
.irp i, 0, 1, 2, 3, 4, 5, 6, 7
    STATS_COUNT vr9, vr13, \i, \i
.endr
    STATS_COUNT vr9, vr14, 8,  0
    STATS_COUNT vr9, vr14, 9,  1
    STATS_COUNT vr9, vr14, 10, 2
    STATS_COUNT vr9, vr14, 11, 3
    STATS_COUNT vr9, vr14, 12, 4
    STATS_COUNT vr9, vr14, 13, 5
    STATS_COUNT vr9, vr14, 14, 6
    STATS_COUNT vr9, vr14, 15, 7
    addi.d       t0,    t0,    -1
    addi.d       t3,    t3,    16
    addi.d       t2,    t2,    32
    bnez         t0,    .loopS_X_e0
.ResS_X_e0:
    andi         t1,    a3,    15
    beqz         t1,    .endS_X_e0
    addi.d       t4,    zero,  8
    vld          vr1,   t3,    0
    vld          vr2,   t3,    1
    vld          vr13,  t2,    0
    vld          vr14,  t2,    16
    vslt.bu      vr3,   vr2,   vr1
    vslt.bu      vr4,   vr1,   vr2
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4   //signRight
    vneg.b       vr6,   vr5          //signLeft
    vshuf.b      vr7,   vr6,   vr0,   vr12
    vadd.b       vr8,   vr7,   vr5
    vreplvei.b   vr0,   vr6,   15
    vaddi.bu     vr9,   vr8,   2    //edgeType
    vslli.b      vr9,   vr9,   2
    blt          t1,    t4,    .resS_X_e0
.irp i, 0, 1, 2, 3, 4, 5, 6, 7
    STATS_COUNT vr9, vr13, \i, \i
.endr
    addi.d       t1,    t1,    -8
    beqz         t1,    .endS_X_e0
    vmov         vr13,  vr14
    vbsrl.v      vr9,   vr9,   8
.resS_X_e0:
    vpickve2gr.b t4,    vr9,   0
    vpickve2gr.h t8,    vr13,  0
    ldx.w        t6,    sp,    t4
    ldx.w        t7,    t5,    t4
    add.w        t6,    t6,    t8
    addi.w       t7,    t7,    1
    stx.w        t6,    sp,    t4
    stx.w        t7,    t5,    t4
    vbsrl.v      vr9,   vr9,   1
    vbsrl.v      vr13,  vr13,  2
    addi.d       t1,    t1,    -1
    bnez         t1,    .resS_X_e0
.endS_X_e0:
    addi.d       a4,    a4,    -1
    add.d        a1,    a1,    a2
    addi.d       a0,    a0,    128
    bnez         a4,    .loopS_Y_e0
    vld          vr1,   sp,    0
    vld          vr2,   t5,    0
    vld          vr3,   a5,    0
    vld          vr4,   a6,    0
    vshuf4i.w    vr5,   vr1,   0xD2
    vshuf4i.w    vr6,   vr2,   0xD2
    vadd.w       vr5,   vr5,   vr3
    vadd.w       vr6,   vr6,   vr4
    ld.w         t2,    a5,    16
    ld.w         t3,    a6,    16
    ld.w         t4,    sp,    16
    ld.w         t6,    t5,    16
    add.w        t2,    t2,    t4
    add.w        t3,    t3,    t6
    vst          vr5,   a5,    0
    vst          vr6,   a6,    0
    st.w         t2,    a5,    16
    st.w         t3,    a6,    16
    addi.d       sp,    sp,    48
endfunc

/*x265_saoCuStatsE1_lsx(const int16_t *diff, const pixel *rec, intptr_t stride, int8_t *upBuff1,
                        int endx, int endY, int32_t *stats, int32_t *count)
*/
function x265_saoCuStatsE1_lsx
    addi.d       sp,    sp,    -64
    srli.d       t2,    a4,    4
    andi         t3,    a4,    15
    vldi         vr10,  1
    vldi         vr11,  0
    vst          vr11,  sp,    0
    vst          vr11,  sp,    16
    vst          vr11,  sp,    32
    st.d         a0,    sp,    40
    st.d         a1,    sp,    48
    st.d         a3,    sp,    56
    addi.d       t5,    sp,    20
.loopS_Y_e1:
    ld.d         a0,    sp,    40
    ld.d         a1,    sp,    48
    ld.d         a3,    sp,    56
    move         t0,    t2
    move         t1,    t3
    beqz         t0,    .ResS_X_e1
.loopS_X_e1:
    vld          vr0,   a1,    0
    vldx         vr1,   a1,    a2
    vld          vr2,   a3,    0
    vld          vr13,  a0,    0
    vld          vr14,  a0,    16
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vst          vr6,   a3,    0
    vslli.b      vr9,   vr8,   2
.irp i, 0, 1, 2, 3, 4, 5, 6, 7
    STATS_COUNT vr9, vr13, \i, \i
.endr
    STATS_COUNT vr9, vr14, 8,  0
    STATS_COUNT vr9, vr14, 9,  1
    STATS_COUNT vr9, vr14, 10, 2
    STATS_COUNT vr9, vr14, 11, 3
    STATS_COUNT vr9, vr14, 12, 4
    STATS_COUNT vr9, vr14, 13, 5
    STATS_COUNT vr9, vr14, 14, 6
    STATS_COUNT vr9, vr14, 15, 7
    addi.d       t0,    t0,    -1
    addi.d       a1,    a1,    16
    addi.d       a0,    a0,    32
    addi.d       a3,    a3,    16
    bnez         t0,    .loopS_X_e1
.ResS_X_e1:
    beqz         t1,    .endS_X_e1
    addi.d       t4,    zero,  8
    vld          vr0,   a1,    0
    vldx         vr1,   a1,    a2
    vld          vr2,   a3,    0
    vld          vr13,  a0,    0
    vld          vr14,  a0,    16
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vslli.b      vr9,   vr8,   2
    blt          t1,    t4,    .resS_X_e1
    fst.d        f6,    a3,    0
.irp i, 0, 1, 2, 3, 4, 5, 6, 7
    STATS_COUNT vr9, vr13, \i, \i
.endr
    addi.d       t1,    t1,    -8
    addi.d       a3,    a3,    8
    beqz         t1,    .endS_X_e1
    vmov         vr13,  vr14
    vbsrl.v      vr9,   vr9,   8
    vbsrl.v      vr6,   vr6,   8
.resS_X_e1:
    vpickve2gr.b t4,    vr9,   0
    vpickve2gr.h t8,    vr13,  0
    ldx.w        t6,    sp,    t4
    ldx.w        t7,    t5,    t4
    add.w        t6,    t6,    t8
    addi.w       t7,    t7,    1
    stx.w        t6,    sp,    t4
    stx.w        t7,    t5,    t4
    vstelm.b     vr6,   a3,    0,   0
    vbsrl.v      vr9,   vr9,   1
    vbsrl.v      vr13,  vr13,  2
    vbsrl.v      vr6,   vr6,   1
    addi.d       t1,    t1,    -1
    addi.d       a3,    a3,    1
    bnez         t1,    .resS_X_e1
.endS_X_e1:
    ld.d         a0,    sp,    40
    ld.d         a1,    sp,    48
    addi.d       a5,    a5,    -1
    addi.d       a0,    a0,    128
    add.d        a1,    a1,    a2
    st.d         a0,    sp,    40
    st.d         a1,    sp,    48
    bnez         a5,    .loopS_Y_e1
    vld          vr1,   sp,    0
    vld          vr2,   t5,    0
    vld          vr3,   a6,    0
    vld          vr4,   a7,    0
    vshuf4i.w    vr5,   vr1,   0xD2
    vshuf4i.w    vr6,   vr2,   0xD2
    vadd.w       vr5,   vr5,   vr3
    vadd.w       vr6,   vr6,   vr4
    ld.w         t2,    a6,    16
    ld.w         t3,    a7,    16
    ld.w         t4,    sp,    16
    ld.w         t6,    t5,    16
    add.w        t2,    t2,    t4
    add.w        t3,    t3,    t6
    vst          vr5,   a6,    0
    vst          vr6,   a7,    0
    st.w         t2,    a6,    16
    st.w         t3,    a7,    16
    addi.d       sp,    sp,    64
endfunc

/*x265_saoCuStatsE2_lsx(const int16_t *diff, const pixel *rec, intptr_t stride, int8_t *upBuff1,
                        int8_t *upBufft, int endX, int endY, int32_t *stats, int32_t *count)
*/
function x265_saoCuStatsE2_lsx
    addi.d       sp,    sp,    -72
    srli.d       t2,    a5,    4
    andi         t3,    a5,    15
    addi.d       a5,    a2,    1
    vldi         vr10,  1
    vldi         vr11,  0
    vst          vr11,  sp,    0
    vst          vr11,  sp,    16
    vst          vr11,  sp,    32
    st.d         a0,    sp,    40
    st.d         a1,    sp,    48
    st.d         a3,    sp,    56  //upBuff1
    st.d         a4,    sp,    64  //upBufft
    addi.d       t5,    sp,    20
.loopS_Y_e2:
    ld.d         a0,    sp,    40
    ld.d         a1,    sp,    48
    ld.d         a3,    sp,    56
    ld.d         a4,    sp,    64
    ldx.bu       t0,    a1,    a2
    ld.bu        t1,    a1,    -1
    addi.d       t4,    zero,  -1
    sltu         t7,    t0,    t1
    sltu         t6,    t1,    t0
    maskeqz      t8,    t4,    t7
    or           t6,    t6,    t8
    st.b         t6,    a4,    0
    move         t0,    t2
    move         t1,    t3
    beqz         t0,    .ResS_X_e2
.loopS_X_e2:
    vld          vr0,   a1,    0
    vldx         vr1,   a1,    a5
    vld          vr13,  a0,    0
    vld          vr14,  a0,    16
    vld          vr2,   a3,    0
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vst          vr6,   a4,    1
    vslli.b      vr9,   vr8,   2
.irp i, 0, 1, 2, 3, 4, 5, 6, 7
    STATS_COUNT vr9, vr13, \i, \i
.endr
    STATS_COUNT vr9, vr14, 8,  0
    STATS_COUNT vr9, vr14, 9,  1
    STATS_COUNT vr9, vr14, 10, 2
    STATS_COUNT vr9, vr14, 11, 3
    STATS_COUNT vr9, vr14, 12, 4
    STATS_COUNT vr9, vr14, 13, 5
    STATS_COUNT vr9, vr14, 14, 6
    STATS_COUNT vr9, vr14, 15, 7
    addi.d       t0,    t0,    -1
    addi.d       a1,    a1,    16
    addi.d       a0,    a0,    32
    addi.d       a3,    a3,    16
    addi.d       a4,    a4,    16
    bnez         t0,    .loopS_X_e2
.ResS_X_e2:
    beqz         t1,    .endS_X_e2
    addi.d       t4,    zero,  8
    vld          vr0,   a1,    0
    vldx         vr1,   a1,    a5
    vld          vr2,   a3,    0
    vld          vr13,  a0,    0
    vld          vr14,  a0,    16
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vslli.b      vr9,   vr8,   2
    blt          t1,    t4,    .resS_X_e2
    fst.d        f6,    a4,    1
.irp i, 0, 1, 2, 3, 4, 5, 6, 7
    STATS_COUNT vr9, vr13, \i, \i
.endr
    addi.d       t1,    t1,    -8
    beqz         t1,    .endS_X_e2
    vmov         vr13,  vr14
    addi.d       a4,    a4,    8
    vbsrl.v      vr9,   vr9,   8
    vbsrl.v      vr6,   vr6,   8
.resS_X_e2:
    vpickve2gr.b t4,    vr9,   0
    vpickve2gr.h t8,    vr13,  0
    ldx.w        t6,    sp,    t4
    ldx.w        t7,    t5,    t4
    add.w        t6,    t6,    t8
    addi.w       t7,    t7,    1
    stx.w        t6,    sp,    t4
    stx.w        t7,    t5,    t4
    vstelm.b     vr6,   a4,    1,   0
    vbsrl.v      vr9,   vr9,   1
    vbsrl.v      vr13,  vr13,  2
    vbsrl.v      vr6,   vr6,   1
    addi.d       t1,    t1,    -1
    addi.d       a4,    a4,    1
    bnez         t1,    .resS_X_e2
.endS_X_e2:
    ld.d         a0,    sp,    40
    ld.d         a1,    sp,    48
    ld.d         a3,    sp,    56
    ld.d         a4,    sp,    64
    addi.d       a6,    a6,    -1
    addi.d       a0,    a0,    128
    add.d        a1,    a1,    a2
    st.d         a0,    sp,    40
    st.d         a1,    sp,    48
    st.d         a4,    sp,    56
    st.d         a3,    sp,    64
    bnez         a6,    .loopS_Y_e2
    ld.d         t0,    sp,    72
    vld          vr1,   sp,    0
    vld          vr2,   t5,    0
    vld          vr3,   a7,    0
    vld          vr4,   t0,    0
    vshuf4i.w    vr5,   vr1,   0xD2
    vshuf4i.w    vr6,   vr2,   0xD2
    vadd.w       vr5,   vr5,   vr3
    vadd.w       vr6,   vr6,   vr4
    ld.w         t2,    a7,    16
    ld.w         t3,    t0,    16
    ld.w         t4,    sp,    16
    ld.w         t6,    t5,    16
    add.w        t2,    t2,    t4
    add.w        t3,    t3,    t6
    vst          vr5,   a7,    0
    vst          vr6,   t0,    0
    st.w         t2,    a7,    16
    st.w         t3,    t0,    16
    addi.d       sp,    sp,    72
endfunc

/*x265_saoCuStatsE3_lsx(const int16_t *diff, const pixel *rec, intptr_t stride, int8_t *upBuff1,
                        int endX, int endY, int32_t *stats, int32_t *count)
*/
function x265_saoCuStatsE3_lsx
    addi.d       sp,    sp,    -64
    srli.d       t2,    a4,    4
    addi.d       t3,    a2,    -1
    vldi         vr10,  1
    vldi         vr11,  0
    vst          vr11,  sp,    0
    vst          vr11,  sp,    16
    vst          vr11,  sp,    32
    st.d         a0,    sp,    40
    st.d         a1,    sp,    48
    st.d         a3,    sp,    56
    addi.d       t5,    sp,    20
.loopS_Y_e3:
    ld.d         a0,    sp,    40
    ld.d         a1,    sp,    48
    ld.d         a3,    sp,    56
    move         t0,    t2
    andi         t1,    a4,    15
    beqz         t0,    .ResS_X_e3
.loopS_X_e3:
    vld          vr0,   a1,    0
    vldx         vr1,   a1,    t3
    vld          vr2,   a3,    0
    vld          vr13,  a0,    0
    vld          vr14,  a0,    16
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vst          vr6,   a3,    -1
    vslli.b      vr9,   vr8,   2
.irp i, 0, 1, 2, 3, 4, 5, 6, 7
    STATS_COUNT vr9, vr13, \i, \i
.endr
    STATS_COUNT vr9, vr14, 8,  0
    STATS_COUNT vr9, vr14, 9,  1
    STATS_COUNT vr9, vr14, 10, 2
    STATS_COUNT vr9, vr14, 11, 3
    STATS_COUNT vr9, vr14, 12, 4
    STATS_COUNT vr9, vr14, 13, 5
    STATS_COUNT vr9, vr14, 14, 6
    STATS_COUNT vr9, vr14, 15, 7
    addi.d       t0,    t0,    -1
    addi.d       a1,    a1,    16
    addi.d       a0,    a0,    32
    addi.d       a3,    a3,    16
    bnez         t0,    .loopS_X_e3
.ResS_X_e3:
    beqz         t1,    .endS_X_e3
    addi.d       t4,    zero,  8
    vld          vr0,   a1,    0
    vldx         vr1,   a1,    t3
    vld          vr2,   a3,    0
    vld          vr13,  a0,    0
    vld          vr14,  a0,    16
    add.d        a1,    a1,    t1
    vslt.bu      vr3,   vr1,   vr0
    vslt.bu      vr4,   vr0,   vr1
    vand.v       vr3,   vr3,   vr10
    vor.v        vr5,   vr3,   vr4
    vneg.b       vr6,   vr5
    vadd.b       vr7,   vr5,   vr2
    vaddi.bu     vr8,   vr7,   2
    vslli.b      vr9,   vr8,   2
    blt          t1,    t4,    .resS_X_e3
    fst.d        f6,    a3,    -1
.irp i, 0, 1, 2, 3, 4, 5, 6, 7
    STATS_COUNT vr9, vr13, \i, \i
.endr
    addi.d       t1,    t1,    -8
    addi.d       a3,    a3,    8
    beqz         t1,    .endS_X_e3
    vmov         vr13,  vr14
    vbsrl.v      vr9,   vr9,   8
    vbsrl.v      vr6,   vr6,   8
.resS_X_e3:
    vpickve2gr.b t4,    vr9,   0
    vpickve2gr.h t8,    vr13,  0
    ldx.w        t6,    sp,    t4
    ldx.w        t7,    t5,    t4
    add.w        t6,    t6,    t8
    addi.w       t7,    t7,    1
    stx.w        t6,    sp,    t4
    stx.w        t7,    t5,    t4
    vstelm.b     vr6,   a3,    -1,   0
    vbsrl.v      vr9,   vr9,   1
    vbsrl.v      vr13,  vr13,  2
    vbsrl.v      vr6,   vr6,   1
    addi.d       t1,    t1,    -1
    addi.d       a3,    a3,    1
    bnez         t1,    .resS_X_e3
.endS_X_e3:
    ldx.bu       t0,    a1,    t3
    ld.bu        t1,    a1,    0
    addi.d       t4,    zero,  -1
    sltu         t7,    t0,    t1
    sltu         t6,    t1,    t0
    maskeqz      t8,    t4,    t7
    addi.d       a5,    a5,    -1
    or           t6,    t6,    t8
    st.b         t6,    a3,    -1
    ld.d         a0,    sp,    40
    ld.d         a1,    sp,    48
    addi.d       a0,    a0,    128
    add.d        a1,    a1,    a2
    st.d         a0,    sp,    40
    st.d         a1,    sp,    48
    bnez         a5,    .loopS_Y_e3
    vld          vr1,   sp,    0
    vld          vr2,   t5,    0
    vld          vr3,   a6,    0
    vld          vr4,   a7,    0
    vshuf4i.w    vr5,   vr1,   0xD2
    vshuf4i.w    vr6,   vr2,   0xD2
    vadd.w       vr5,   vr5,   vr3
    vadd.w       vr6,   vr6,   vr4
    ld.w         t2,    a6,    16
    ld.w         t3,    a7,    16
    ld.w         t4,    sp,    16
    ld.w         t6,    t5,    16
    add.w        t2,    t2,    t4
    add.w        t3,    t3,    t6
    vst          vr5,   a6,    0
    vst          vr6,   a7,    0
    st.w         t2,    a6,    16
    st.w         t3,    a7,    16
    addi.d       sp,    sp,    64
endfunc

.macro STATS_COUNT_B in0, in1, i0, i1
    vpickve2gr.b  t3,  \in0,   \i0
    vpickve2gr.h  t6,  \in1,   \i1
    ldx.w         t7,  a5,     t3
    ldx.w         t8,  a6,     t3
    add.w         t7,  t7,     t6
    addi.w        t8,  t8,     1
    stx.w         t7,  a5,     t3
    stx.w         t8,  a6,     t3
.endm
/*x265_saoCuStatsB0_lsx(const int16_t *diff, const pixel *rec, intptr_t stride,
                        int endX, int endY, int32_t *stats, int32_t *count)
*/
function x265_saoCuStatsB0_lsx
    srli.d       t2,    a3,    4
    addi.d       a7,    zero,  8
.loopS_Y_b0:
    move         t0,    t2
    andi         t1,    a3,    15
    move         t5,    a1
    move         t4,    a0
    beqz         t0,    .ResS_X_b0
.loopS_X_b0:
    vld          vr0,   t5,    0
    vld          vr2,   t4,    0
    vld          vr3,   t4,    16
    vsrli.b      vr4,   vr0,   3
    vslli.b      vr5,   vr4,   2
.irp i, 0, 1, 2, 3, 4, 5, 6, 7
    STATS_COUNT_B vr5, vr2, \i, \i
.endr
    STATS_COUNT_B vr5, vr3, 8,  0
    STATS_COUNT_B vr5, vr3, 9,  1
    STATS_COUNT_B vr5, vr3, 10, 2
    STATS_COUNT_B vr5, vr3, 11, 3
    STATS_COUNT_B vr5, vr3, 12, 4
    STATS_COUNT_B vr5, vr3, 13, 5
    STATS_COUNT_B vr5, vr3, 14, 6
    STATS_COUNT_B vr5, vr3, 15, 7
    addi.d       t0,    t0,    -1
    addi.d       t5,    t5,    16
    addi.d       t4,    t4,    32
    bnez         t0,    .loopS_X_b0
.ResS_X_b0:
    beqz         t1,    .endS_X_b0
    vld          vr0,   t5,    0
    vld          vr2,   t4,    0
    vld          vr3,   t4,    16
    vsrli.b      vr4,   vr0,   3
    vslli.b      vr5,   vr4,   2
    blt          t1,    a7,    .resS_X_b0
.irp i, 0, 1, 2, 3, 4, 5, 6, 7
    STATS_COUNT_B vr5, vr2, \i, \i
.endr
    addi.d       t1,    t1,    -8
    beqz         t1,    .endS_X_b0
    vmov         vr2,   vr3
    vbsrl.v      vr5,   vr5,   8
.resS_X_b0:
    STATS_COUNT_B vr5, vr2,  0,  0
    vbsrl.v      vr5,   vr5,   1
    vbsrl.v      vr2,   vr2,   2
    addi.d       t1,    t1,    -1
    bnez         t1,    .resS_X_b0
.endS_X_b0:
    addi.d       a4,    a4,    -1
    add.d        a1,    a1,    a2
    addi.d       a0,    a0,    128
    bnez         a4,    .loopS_Y_b0
endfunc
