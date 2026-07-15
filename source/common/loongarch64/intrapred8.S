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
******************************************************************************/

#include "loongson_asm.S"

.macro INTRAFILTER_LSX_16 offset, src0, src1, src2, src3, src4, src5, dst0, dst1
    // load samples[offset ~ offset+15], samples[offset+1 ~ offset+16], samples[offset+2 ~ offset+17]
    vld             \src0,  a0,     \offset
    vld             \src2,  a0,     \offset + 1
    vld             \src3,  a0,     \offset + 9
    vld             \src4,  a0,     \offset + 2
    vexth.hu.bu     \src1,  \src0
    vsllwil.hu.bu   \src0,  \src0,  0
    vsllwil.hu.bu   \src2,  \src2,  1
    vsllwil.hu.bu   \src3,  \src3,  1
    vexth.hu.bu     \src5,  \src4
    vsllwil.hu.bu   \src4,  \src4,  0

    // calculate filtered[offset+1 ~ offset+16]
    vadd.h          \dst0,  \src0,  \src2
    vadd.h          \dst1,  \src1,  \src3
    vadd.h          \dst0,  \dst0,  \src4
    vadd.h          \dst1,  \dst1,  \src5
    vaddi.hu        \dst0,  \dst0,  2
    vaddi.hu        \dst1,  \dst1,  2
.endm

// void x265_intra_filter_4x4_lsx(const pixel* samples, pixel* filtered)
function x265_intra_filter_4x4_lsx
    ld.bu           t0,     a0,     0
    ld.bu           t1,     a0,     8
    ld.bu           t2,     a0,     16

    // calculate filtered[1~16]
    INTRAFILTER_LSX_16      0,  vr0,  vr1,  vr2,  vr3,  vr4,  vr5,  vr6,  vr7

    // calculate filtered[0]
    vpickve2gr.hu   t5,     vr0,    1
    alsl.d          t5,     t0,     t5,     1
    vpickve2gr.hu   t6,     vr1,    1
    add.d           t5,     t6,     t5
    addi.d          t5,     t5,     2
    srli.d          t5,     t5,     2
    st.b            t5,     a1,     0

    // calculate filtered[tuSize2 + 1]
    alsl.d          t6,     t6,     t0,     1
    vpickve2gr.hu   t4,     vr1,    2
    add.d           t6,     t6,     t4
    addi.d          t6,     t6,     2
    vinsgr2vr.h     vr7,    t6,     0
    vsrlni.b.h      vr7,    vr6,    2

    // calculate filtered[tuSize2] and filtered[tuSize2 + tuSize2]
    vinsgr2vr.b     vr7,    t1,     7
    vinsgr2vr.b     vr7,    t2,     15
    vst             vr7,    a1,     1       // store filtered[1~16]
endfunc

// void x265_intra_filter_8x8_lsx(const pixel* samples, pixel* filtered)
function x265_intra_filter_8x8_lsx
    ld.bu           t0,     a0,     0
    ld.bu           t1,     a0,     16
    ld.bu           t2,     a0,     32

    // calculate filtered[1~32]
    INTRAFILTER_LSX_16      0,  vr0,  vr1,  vr4,  vr5,  vr8,  vr9,  vr12, vr13
    INTRAFILTER_LSX_16      16, vr2,  vr3,  vr6,  vr7,  vr10, vr11, vr14, vr15

    // calculate filtered[0]
    vpickve2gr.hu   t5,     vr0,    1
    alsl.d          t5,     t0,     t5,     1
    vpickve2gr.hu   t6,     vr2,    1
    add.d           t5,     t6,     t5
    addi.d          t5,     t5,     2
    srli.d          t5,     t5,     2
    st.b            t5,     a1,     0

    // calculate filtered[tuSize2 + 1]
    alsl.d          t6,     t6,     t0,     1
    vpickve2gr.hu   t4,     vr2,    2
    add.d           t6,     t6,     t4
    addi.d          t6,     t6,     2
    vinsgr2vr.h     vr14,   t6,     0
    vsrlni.b.h      vr13,   vr12,   2
    vsrlni.b.h      vr15,   vr14,   2

    // calculate filtered[tuSize2] and filtered[tuSize2 + tuSize2]
    vinsgr2vr.b     vr13,   t1,     15
    vinsgr2vr.b     vr15,   t2,     15
    vst             vr13,   a1,     1
    vst             vr15,   a1,     17
endfunc

// void x265_intra_filter_16x16_lsx(const pixel* samples, pixel* filtered)
function x265_intra_filter_16x16_lsx
    ld.bu           t0,     a0,     0
    ld.bu           t1,     a0,     32
    ld.bu           t2,     a0,     64

    // calculate filtered[1~32]
    INTRAFILTER_LSX_16      0,  vr0,  vr1,  vr4,  vr5,  vr8,  vr9,  vr12, vr13
    INTRAFILTER_LSX_16      16, vr2,  vr3,  vr6,  vr7,  vr10, vr11, vr14, vr15

    // calculate filtered[0]
    vpickve2gr.hu   t5,     vr0,    1
    alsl.d          t5,     t0,     t5,     1
    vpickve2gr.hu   t6,     vr11,   7
    add.d           t5,     t6,     t5
    addi.d          t5,     t5,     2
    srli.d          t5,     t5,     2
    st.b            t5,     a1,     0
    vsrlni.b.h      vr13,   vr12,   2
    vsrlni.b.h      vr15,   vr14,   2
    vinsgr2vr.b     vr15,   t1,     15
    vst             vr13,   a1,     1
    vst             vr15,   a1,     17

    // calculate filtered[33~64]
    INTRAFILTER_LSX_16      32, vr0,  vr1,  vr4,  vr5,  vr8,  vr9,  vr12, vr13
    INTRAFILTER_LSX_16      48, vr2,  vr3,  vr6,  vr7,  vr10, vr11, vr14, vr15

    // calculate filtered[tuSize2 + 1]
    alsl.d          t6,     t6,     t0,     1
    vpickve2gr.hu   t4,     vr0,    2
    add.d           t6,     t6,     t4
    addi.d          t6,     t6,     2
    vinsgr2vr.h     vr12,   t6,     0
    vsrlni.b.h      vr13,   vr12,   2
    vsrlni.b.h      vr15,   vr14,   2
    vinsgr2vr.b     vr15,   t2,     15
    vst             vr13,   a1,     33
    vst             vr15,   a1,     49
endfunc

// void x265_intra_filter_32x32_lsx(const pixel* samples, pixel* filtered)
function x265_intra_filter_32x32_lsx
    ld.bu           t0,     a0,     0
    ld.bu           t1,     a0,     64
    ld.bu           t2,     a0,     128

    // calculate filtered[1~32]
    INTRAFILTER_LSX_16      0,  vr0,  vr1,  vr4,  vr5,  vr8,  vr9,  vr12, vr13
    INTRAFILTER_LSX_16      16, vr2,  vr3,  vr6,  vr7,  vr10, vr11, vr14, vr15

    vsrlni.b.h      vr13,   vr12,   2
    vsrlni.b.h      vr15,   vr14,   2
    vst             vr13,   a1,     1
    vst             vr15,   a1,     17
    vpickve2gr.hu   t5,     vr0,    1       // save samples[1] for calculating samples[65]

    // calculate filtered[33~64]
    INTRAFILTER_LSX_16      32, vr0,  vr1,  vr4,  vr5,  vr8,  vr9,  vr12, vr13
    INTRAFILTER_LSX_16      48, vr2,  vr3,  vr6,  vr7,  vr10, vr11, vr14, vr15

    // calculate filtered[0]
    alsl.d          t5,     t0,     t5,     1
    vpickve2gr.hu   t6,     vr11,   7
    add.d           t5,     t6,     t5
    addi.d          t5,     t5,     2
    srli.d          t5,     t5,     2
    st.b            t5,     a1,     0
    vsrlni.b.h      vr13,   vr12,   2
    vsrlni.b.h      vr15,   vr14,   2
    vinsgr2vr.b     vr15,   t1,     15
    vst             vr13,   a1,     33
    vst             vr15,   a1,     49

    // calculate filtered[65~96]
    INTRAFILTER_LSX_16      64, vr0,  vr1,  vr4,  vr5,  vr8,  vr9,  vr12, vr13
    INTRAFILTER_LSX_16      80, vr2,  vr3,  vr6,  vr7,  vr10, vr11, vr14, vr15

    // calculate filtered[tuSize2 + 1]
    alsl.d          t6,     t6,     t0,     1
    vpickve2gr.hu   t4,     vr0,    2
    add.d           t6,     t6,     t4
    addi.d          t6,     t6,     2
    vinsgr2vr.h     vr12,   t6,     0
    vsrlni.b.h      vr13,   vr12,   2
    vsrlni.b.h      vr15,   vr14,   2
    vst             vr13,   a1,     65
    vst             vr15,   a1,     81

    // calculate filtered[97~128]
    INTRAFILTER_LSX_16      96, vr0,  vr1,  vr4,  vr5,  vr8,  vr9,  vr12, vr13
    INTRAFILTER_LSX_16      112,vr2,  vr3,  vr6,  vr7,  vr10, vr11, vr14, vr15
    vsrlni.b.h      vr13,   vr12,   2
    vsrlni.b.h      vr15,   vr14,   2
    vinsgr2vr.b     vr15,   t2,     15
    vst             vr13,   a1,     97
    vst             vr15,   a1,     113
endfunc

.macro INTRAFILTER_LASX_16 src0, src1, src2, dst0, offset
    // load samples[offset ~ offset+15], samples[offset+1 ~ offset+16], samples[offset+2 ~ offset+17]
    xvld            \src0,  a0,     \offset
    xvld            \src1,  a0,     \offset + 1
    xvld            \src2,  a0,     \offset + 2
    vext2xv.hu.bu   \src0,  \src0
    vext2xv.hu.bu   \src1,  \src1
    xvslli.h        \src1,  \src1,  1
    vext2xv.hu.bu   \src2,  \src2

    // calculate filtered[[offset+1 ~ offset+16]
    xvadd.h         \dst0,  \src0,  \src1
    xvadd.h         \dst0,  \dst0,  \src2
    xvssrarni.bu.h  \dst0,  \dst0,  2
.endm

// void x265_intra_filter_4x4_lasx(const pixel* samples, pixel* filtered)
function x265_intra_filter_4x4_lasx
    ld.bu           t0,     a0,     0
    ld.bu           t1,     a0,     8
    ld.bu           t2,     a0,     16

    // calculate filtered[1~16]
    INTRAFILTER_LASX_16     xr0,  xr1,  xr2,  xr3,  0

    // calculate filtered[0]
    ld.bu           t5,     a0,     1
    alsl.d          t5,     t0,     t5,     1
    ld.bu           t6,     a0,     9
    add.d           t5,     t6,     t5
    addi.d          t5,     t5,     2
    srli.d          t5,     t5,     2
    st.b            t5,     a1,     0

    // store filtered[1~16]
    xvpermi.d       xr3,    xr3,    0b00001000
    vst             vr3,    a1,     1
    st.b            t1,     a1,     8               // store filtered[8]

    // calculate filtered[tuSize2 + 1]
    slli.d          t6,     t6,     1
    add.d           t6,     t6,     t0
    ld.bu           t4,     a0,     10
    add.d           t6,     t6,     t4
    addi.d          t6,     t6,     2
    srli.d          t6,     t6,     2
    st.b            t6,     a1,     9
    st.b            t2,     a1,     16              // store filtered[16]
endfunc

// void x265_intra_filter_8x8_lasx(const pixel* samples, pixel* filtered)
function x265_intra_filter_8x8_lasx
    ld.bu           t0,     a0,     0
    ld.bu           t1,     a0,     16
    ld.bu           t2,     a0,     32

    // calculate filtered[1 ~ 32]
    INTRAFILTER_LASX_16     xr0,  xr2,  xr4,  xr6,  0
    INTRAFILTER_LASX_16     xr1,  xr3,  xr5,  xr7,  16

    // calculate filtered[0]
    ld.bu           t5,     a0,     1
    alsl.d          t5,     t0,     t5,     1
    ld.bu           t6,     a0,     17
    add.d           t5,     t6,     t5
    addi.d          t5,     t5,     2
    srli.d          t5,     t5,     2
    st.b            t5,     a1,     0

    // store filtered[1~32]
    xvshuf4i.d      xr6,    xr7,    0b00001000
    xvpermi.d       xr6,    xr6,    0b11011000
    xvst            xr6,    a1,     1
    st.b            t1,     a1,     16              // store filtered[16]

    slli.d          t6,     t6,     1
    add.d           t6,     t6,     t0
    ld.bu           t4,     a0,     18
    add.d           t6,     t6,     t4
    addi.d          t6,     t6,     2
    srli.d          t6,     t6,     2
    st.b            t6,     a1,     17              // store filtered[17]
    st.b            t2,     a1,     32              // store filtered[32]
endfunc

// void x265_intra_filter_16x16_lasx(const pixel* samples, pixel* filtered)
function x265_intra_filter_16x16_lasx
    ld.bu           t0,     a0,     0
    ld.bu           t1,     a0,     32
    ld.bu           t2,     a0,     64

    // calculate filtered[1 ~ 64]
    INTRAFILTER_LASX_16     xr0,  xr4,  xr8,  xr12, 0
    INTRAFILTER_LASX_16     xr1,  xr5,  xr9,  xr13, 16
    INTRAFILTER_LASX_16     xr2,  xr6,  xr10, xr14, 32
    INTRAFILTER_LASX_16     xr3,  xr7,  xr11, xr15, 48

    // calculate filtered[0]
    ld.bu           t5,     a0,     1
    alsl.d          t5,     t0,     t5,     1
    ld.bu           t6,     a0,     33
    add.d           t5,     t6,     t5
    addi.d          t5,     t5,     2
    srli.d          t5,     t5,     2
    st.b            t5,     a1,     0

    // store filtered[1~32]
    xvshuf4i.d      xr12,   xr13,   0b00001000
    xvpermi.d       xr12,   xr12,   0b11011000
    xvst            xr12,   a1,     1
    st.b            t1,     a1,     32              // store filtered[32]

    // store filtered[33~64]
    xvshuf4i.d      xr14,   xr15,   0b00001000
    xvpermi.d       xr14,   xr14,   0b11011000
    xvst            xr14,   a1,     33
    alsl.d          t6,     t6,     t0,     1
    ld.bu           t4,     a0,     34
    add.d           t6,     t6,     t4
    addi.d          t6,     t6,     2
    srli.d          t6,     t6,     2
    st.b            t6,     a1,     33              // store filtered[33]
    st.b            t2,     a1,     64              // store filtered[64]
endfunc

// void x265_intra_filter_32x32_lasx(const pixel* samples, pixel* filtered)
function x265_intra_filter_32x32_lasx
    ld.bu           t0,     a0,     0
    ld.bu           t1,     a0,     64
    ld.bu           t2,     a0,     128

    //calculate filtered[1~64]
    INTRAFILTER_LASX_16     xr0,  xr4,  xr8,  xr12, 0
    INTRAFILTER_LASX_16     xr1,  xr5,  xr9,  xr13, 16
    INTRAFILTER_LASX_16     xr2,  xr6,  xr10, xr14, 32
    INTRAFILTER_LASX_16     xr3,  xr7,  xr11, xr15, 48

    // store filtered[1~32]
    xvshuf4i.d      xr12,   xr13,   0b00001000
    xvpermi.d       xr12,   xr12,   0b11011000
    xvst            xr12,   a1,     1
    // store filtered[33~64]
    xvshuf4i.d      xr14,   xr15,   0b00001000
    xvpermi.d       xr14,   xr14,   0b11011000
    xvst            xr14,   a1,     33

    // calculate filtered[65~128]
    INTRAFILTER_LASX_16     xr0,  xr4,  xr8,  xr12, 64
    INTRAFILTER_LASX_16     xr1,  xr5,  xr9,  xr13, 80
    INTRAFILTER_LASX_16     xr2,  xr6,  xr10, xr14, 96
    INTRAFILTER_LASX_16     xr3,  xr7,  xr11, xr15, 112

    // calculate filtered[0]
    ld.bu           t5,     a0,     1
    alsl.d          t5,     t0,     t5,     1
    ld.bu           t6,     a0,     65
    add.d           t5,     t6,     t5
    addi.d          t5,     t5,     2
    srli.d          t5,     t5,     2
    st.b            t5,     a1,     0

    // store filtered[65~96]
    xvshuf4i.d      xr12,   xr13,   0b00001000
    xvpermi.d       xr12,   xr12,   0b11011000
    xvst            xr12,   a1,     65
    // store filtered[97~128]
    xvshuf4i.d      xr14,   xr15,   0b00001000
    xvpermi.d       xr14,   xr14,   0b11011000
    xvst            xr14,   a1,     97
    st.b            t1,     a1,     64              // store filtered[64]
    alsl.d          t6,     t6,     t0,     1
    ld.bu           t4,     a0,     66
    add.d           t6,     t6,     t4
    addi.d          t6,     t6,     2
    srli.d          t6,     t6,     2
    st.b            t6,     a1,     65              // store filtered[65]
    st.b            t2,     a1,     128             // store filtered[128]
endfunc

.macro CALCULATE_INTRA_PRED_PLANAR  type, src0, src1, src2, src3, \
                                    src4, src5, src6, dst0, dst1
    \type\()vreplgr2vr.h    \dst0,    zero
    \type\()vreplgr2vr.h    \dst1,    zero
    \type\()vmadd.h         \dst0,    \src0,    \src1
    \type\()vmadd.h         \dst1,    \src2,    \src3
    \type\()vadd.h          \dst0,    \dst0,    \src4
    \type\()vadd.h          \dst1,    \dst1,    \src5
    \type\()vadd.h          \dst0,    \dst0,    \src6
    \type\()vadd.h          \dst0,    \dst1,    \dst0
.endm

// void x265_intra_pred_planar_4x4_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int /*dirMode*/, int /*bFilter*/)
const planar_4x4_lsx
.short 3, 2, 1, 0, 1, 2, 3, 4
endconst
function x265_intra_pred_planar_4x4_lsx     // blkSize = 1 << 2 = 4
    // prepare data
    ld.bu           t2,     a2,     5
    ld.bu           t3,     a2,     13
    fld.d           f0,     a2,     1
    fld.d           f1,     a2,     9
    vsllwil.hu.bu   vr0,    vr0,    0
    vsllwil.hu.bu   vr1,    vr1,    0

    la.local        t4,     planar_4x4_lsx
    vld             vr2,    t4,     0
    vld             vr3,    t4,     8

    vreplgr2vr.h    vr4,    t2
    vreplgr2vr.h    vr5,    t3
    vreplgr2vr.h    vr6,    t3
    vmul.h          vr4,    vr3,    vr4
    addi.d          t7,     zero,   4
    vreplgr2vr.h    vr7,    t7

    addi.d          t4,     zero,   0
    addi.d          t5,     zero,   3
    addi.d          t6,     a0,     0

.LOOP_INTRA_PRED_PLANAR_4x4_LSX:
    vreplve.h       vr9,    vr1,    t4
    vreplgr2vr.h    vr10,    t5
    CALCULATE_INTRA_PRED_PLANAR  , vr2, vr9, vr10, vr0, vr4, vr6, vr7, vr8, vr11
    vsrlni.b.h      vr8,    vr8,    3
    vstelm.w        vr8,    t6,     0,      0       // store dst[y*dstStride+x]

    // prepare for next loop
    addi.d          t4,     t4,     1
    addi.d          t5,     t5,     -1
    vadd.h          vr6,    vr6,    vr5
    add.d           t6,     t6,     a1
    blt             t4,     t7,     .LOOP_INTRA_PRED_PLANAR_4x4_LSX
endfunc

// void x265_intra_pred_planar_8x8_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int /*dirMode*/, int /*bFilter*/)
const planar_8x8_lsx
.short 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8
endconst
function x265_intra_pred_planar_8x8_lsx     // blkSize = 1 << 3 = 8
    // prepare data
    ld.bu           t2,     a2,     9
    ld.bu           t3,     a2,     25
    fld.d           f0,     a2,     1
    fld.d           f1,     a2,     17
    vsllwil.hu.bu   vr0,    vr0,    0
    vsllwil.hu.bu   vr1,    vr1,    0

    la.local        t4,     planar_8x8_lsx
    vld             vr2,    t4,     0
    vld             vr3,    t4,     16

    vreplgr2vr.h    vr4,    t2
    vreplgr2vr.h    vr5,    t3
    vreplgr2vr.h    vr6,    t3
    vmul.h          vr4,    vr3,    vr4
    addi.d          t7,     zero,   8
    vreplgr2vr.h    vr7,    t7

    addi.d          t4,     zero,   0
    addi.d          t5,     zero,   7
    addi.d          t6,     a0,     0

.LOOP_INTRA_PRED_PLANAR_8x8_LSX:
    vreplve.h       vr9,    vr1,    t4
    vreplgr2vr.h    vr10,    t5
    CALCULATE_INTRA_PRED_PLANAR  , vr2, vr9, vr10, vr0, vr4, vr6, vr7, vr8, vr11
    vsrlni.b.h      vr8,    vr8,    4
    vstelm.d        vr8,    t6,     0,      0       // store dst[y*dstStride+x]

    // prepare for next loop
    addi.d          t4,     t4,     1
    addi.d          t5,     t5,     -1
    vadd.h          vr6,    vr6,    vr5
    add.d           t6,     t6,     a1
    blt             t4,     t7,     .LOOP_INTRA_PRED_PLANAR_8x8_LSX
endfunc

// void x265_intra_pred_planar_16x16_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int /*dirMode*/, int /*bFilter*/)
const planar_16x16_lsx
.short 15, 14, 13, 12, 11, 10, 9,  8
.short 7,  6,  5,  4,  3,  2,  1,  0
.short 1,  2,  3,  4,  5,  6,  7,  8
.short 9,  10, 11, 12, 13, 14, 15, 16
endconst
function x265_intra_pred_planar_16x16_lsx   // blkSize = 1 << 4 = 16
    // prepare data
    ld.bu           t2,     a2,     17
    ld.bu           t3,     a2,     49
    fld.d           f0,     a2,     1
    fld.d           f10,    a2,     9
    vld             vr1,    a2,     33
    vsllwil.hu.bu   vr0,    vr0,    0
    vsllwil.hu.bu   vr10,   vr10,   0

    la.local        t4,     planar_16x16_lsx
    vld             vr2,    t4,     0
    vld             vr12,   t4,     16
    vld             vr3,    t4,     32
    vld             vr13,   t4,     48

    vreplgr2vr.h    vr4,    t2
    vreplgr2vr.h    vr5,    t3
    vreplgr2vr.h    vr6,    t3
    vmul.h          vr14,   vr13,   vr4
    vmul.h          vr4,    vr3,    vr4
    addi.d          t7,     zero,   16
    vreplgr2vr.h    vr7,    t7

    addi.d          t4,     zero,   0
    addi.d          t5,     zero,   15
    addi.d          t6,     a0,     0

.LOOP_INTRA_PRED_PLANAR_16x16_LSX:
    vreplve.b       vr9,    vr1,    t4
    vsllwil.hu.bu   vr9,    vr9,    0
    vreplgr2vr.h    vr11,   t5
    CALCULATE_INTRA_PRED_PLANAR  , vr2,  vr9, vr11, vr0,  vr4,  vr6, vr7, vr8,  vr15
    CALCULATE_INTRA_PRED_PLANAR  , vr12, vr9, vr11, vr10, vr14, vr6, vr7, vr18, vr16
    vsrlni.b.h      vr18,   vr8,    5
    vst             vr18,   t6,     0       // store dst[y*dstStride+x]

    // prepare for next loop
    addi.d          t4,     t4,     1
    addi.d          t5,     t5,     -1
    vadd.h          vr6,    vr6,    vr5
    add.d           t6,     t6,     a1
    blt             t4,     t7,     .LOOP_INTRA_PRED_PLANAR_16x16_LSX
endfunc

// void x265_intra_pred_planar_32x32_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int /*dirMode*/, int /*bFilter*/)
const planar_32x32_lsx
.short 31, 30, 29, 28, 27, 26, 25, 24
.short 23, 22, 21, 20, 19, 18, 17, 16
.short 15, 14, 13, 12, 11, 10,  9,  8
.short  7,  6,  5,  4,  3,  2,  1,  0
.short  1,  2,  3,  4,  5,  6,  7,  8
.short  9, 10, 11, 12, 13, 14, 15, 16
.short 17, 18, 19, 20, 21, 22, 23, 24
.short 25, 26, 27, 28, 29, 30, 31, 32
endconst
function x265_intra_pred_planar_32x32_lsx   // blkSize = 1 << 5 = 32
    // prepare data
    ld.bu           t2,     a2,     33
    ld.bu           t3,     a2,     97
    fld.d           f0,     a2,     1
    fld.d           f1,     a2,     9
    fld.d           f2,     a2,     17
    fld.d           f3,     a2,     25
    vld             vr4,    a2,     65
    vld             vr21,   a2,     81
    vsllwil.hu.bu   vr0,    vr0,    0
    vsllwil.hu.bu   vr1,    vr1,    0
    vsllwil.hu.bu   vr2,    vr2,    0
    vsllwil.hu.bu   vr3,    vr3,    0

    la.local        t4,     planar_32x32_lsx
    vld             vr5,    t4,     0
    vld             vr6,    t4,     16
    vld             vr7,    t4,     32
    vld             vr8,    t4,     48
    vld             vr9,    t4,     64
    vld             vr10,   t4,     80
    vld             vr11,   t4,     96
    vld             vr12,   t4,     112

    vreplgr2vr.h    vr13,   t2
    vmul.h          vr9,    vr13,   vr9
    vmul.h          vr10,   vr13,   vr10
    vmul.h          vr11,   vr13,   vr11
    vmul.h          vr12,   vr13,   vr12
    vreplgr2vr.h    vr13,   t3
    vreplgr2vr.h    vr14,   t3
    addi.d          t7,     zero,   32
    vreplgr2vr.h    vr15,   t7

    addi.d          t4,     zero,   0
    addi.d          t5,     zero,   31
    addi.d          t6,     a0,     0
    addi.d          t8,     zero,   16
    vreplve.b       vr20,   vr4,    t4
.LOOP_INTRA_PRED_PLANAR_32x32_LSX:
    vsllwil.hu.bu   vr20,   vr20,   0
    vreplgr2vr.h    vr22,   t5
    CALCULATE_INTRA_PRED_PLANAR  , vr5,  vr20, vr22, vr0,  vr9,  vr14, vr15, vr16, vr23
    CALCULATE_INTRA_PRED_PLANAR  , vr6,  vr20, vr22, vr1,  vr10, vr14, vr15, vr17, vr23
    vsrlni.b.h      vr17,   vr16,   6
    vst             vr17,   t6,     0
    CALCULATE_INTRA_PRED_PLANAR  , vr7,  vr20, vr22, vr2,  vr11, vr14, vr15, vr18, vr23
    CALCULATE_INTRA_PRED_PLANAR  , vr8,  vr20, vr22, vr3,  vr12, vr14, vr15, vr19, vr23
    vsrlni.b.h      vr19,   vr18,   6
    vst             vr19,   t6,     16      // store dst[y*dstStride+(16~31)]

    // prepare for next loop
    addi.d          t4,     t4,     1
    addi.d          t5,     t5,     -1
    vadd.h          vr14,   vr14,   vr13
    add.d           t6,     t6,     a1
    // if y >= 16, switch to high-short part loop
    bge             t4,     t8,     .INTRA_PRED_PLANAR_32x32_LSX_SWITCH_TO_HIHG_HALF_PART_LOOP
    vreplve.b       vr20,   vr4,    t4

.INTRA_PRED_PLANAR_32x32_LSX_WHILE_JUDGE:
    blt             t4,     t7,     .LOOP_INTRA_PRED_PLANAR_32x32_LSX
    b               .INTRA_PRED_PLANAR_32x32_LSX_TERMINATE

.INTRA_PRED_PLANAR_32x32_LSX_SWITCH_TO_HIHG_HALF_PART_LOOP:
    // new_index = y - 16, for get the high-short part data of left[]
    sub.d           a3,     t4,     t8
    vreplve.b       vr20,   vr21,   a3      // rep left[y] to vr20, y = {16 ~ 31}
    b               .INTRA_PRED_PLANAR_32x32_LSX_WHILE_JUDGE
.INTRA_PRED_PLANAR_32x32_LSX_TERMINATE:
endfunc

// void x265_intra_pred_planar_16x16_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int /*dirMode*/, int /*bFilter*/)
const planar_16x16_lasx
.short 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0
.short  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16
endconst
function x265_intra_pred_planar_16x16_lasx  // blkSize = 1 << 4 = 16
    // prepare data
    ld.bu           t2,     a2,     17
    ld.bu           t3,     a2,     49
    vld             vr0,    a2,     1
    vld             vr1,    a2,     33
    vext2xv.hu.bu   xr0,    xr0

    la.local        t4,     planar_16x16_lasx
    xvld            xr2,    t4,     0
    xvld            xr3,    t4,     32

    xvreplgr2vr.h   xr4,    t2
    xvreplgr2vr.h   xr5,    t3
    xvreplgr2vr.h   xr6,    t3
    xvmul.h         xr4,    xr3,    xr4
    addi.d          t7,     zero,   16
    xvreplgr2vr.h   xr7,    t7

    addi.d          t4,     zero,   0
    addi.d          t5,     zero,   15
    addi.d          t6,     a0,     0

.LOOP_INTRA_PRED_PLANAR_16x16_LASX:
    xvreplve.b      xr9,    xr1,    t4
    vext2xv.hu.bu   xr9,    xr9
    xvreplgr2vr.h   xr10,   t5
    CALCULATE_INTRA_PRED_PLANAR       x, xr2, xr9, xr10, xr0, xr4, xr6, xr7, xr8, xr11
    xvsrlni.b.h     xr8,    xr8,    5
    xvpermi.d       xr8,    xr8,    0b00001000
    vst             vr8,    t6,     0

    // prepare for next loop
    addi.d          t4,     t4,     1
    addi.d          t5,     t5,     -1
    xvadd.h         xr6,    xr6,    xr5
    add.d           t6,     t6,     a1
    blt             t4,     t7,     .LOOP_INTRA_PRED_PLANAR_16x16_LASX
endfunc

// void x265_intra_pred_planar_32x32_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int /*dirMode*/, int /*bFilter*/)
const planar_32x32_lasx
.short 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16
.short 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0
.short  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16
.short 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
endconst
function x265_intra_pred_planar_32x32_lasx   // blkSize = 1 << 5 = 32
    // prepare data
    ld.bu           t2,     a2,     33
    ld.bu           t3,     a2,     97
    vld             vr0,    a2,     1
    vld             vr10,   a2,     17
    vld             vr1,    a2,     65
    vld             vr11,   a2,     81
    vext2xv.hu.bu   xr0,    xr0
    vext2xv.hu.bu   xr10,   xr10

    la.local        t4,     planar_32x32_lasx
    xvld            xr2,    t4,     0
    xvld            xr12,   t4,     32
    xvld            xr3,    t4,     64
    xvld            xr13,   t4,     96

    xvreplgr2vr.h   xr4,    t2
    xvreplgr2vr.h   xr5,    t3
    xvreplgr2vr.h   xr6,    t3
    xvmul.h         xr14,   xr13,   xr4
    xvmul.h         xr4,    xr3,    xr4
    addi.d          t7,     zero,   32
    xvreplgr2vr.h   xr7,    t7

    addi.d          t4,     zero,   0
    addi.d          t5,     zero,   31
    addi.d          t6,     a0,     0
    addi.d          a4,     zero,   16
    xvreplve.b      xr9,    xr1,    t4
    vext2xv.hu.bu   xr9,    xr9
.LOOP_INTRA_PRED_PLANAR_32x32_LASX:
    xvreplgr2vr.h   xr19,   t5
    CALCULATE_INTRA_PRED_PLANAR     x, xr2,  xr9, xr19, xr0,  xr4,  xr6,  xr7, xr8,  xr13
    CALCULATE_INTRA_PRED_PLANAR     x, xr12, xr9, xr19, xr10, xr14, xr6,  xr7, xr18, xr23
    xvsrlni.b.h     xr8,    xr8,    6
    xvsrlni.b.h     xr18,   xr18,   6
    xvpermi.d       xr8,    xr8,    0b00001000
    xvpermi.d       xr18,   xr18,   0b00001000
    xvpermi.q       xr18,   xr8,    0b00100000
    xvst            xr18,   t6,     0

    // prepare for next loop
    addi.d          t4,     t4,     1
    addi.d          t5,     t5,     -1
    xvadd.h         xr6,    xr6,    xr5
    add.d           t6,     t6,     a1
    // if y >= 16, switch to high-short part loop
    bge             t4,     a4,     .INTRA_PRED_PLANAR_32x32_LASX_SWITCH_TO_HIHG_HALF_PART_LOOP
    xvreplve.b      xr9,    xr1,    t4
    vext2xv.hu.bu   xr9,    xr9

.INTRA_PRED_PLANAR_32x32_LASX_WHILE_JUDGE:
    blt             t4,     t7,     .LOOP_INTRA_PRED_PLANAR_32x32_LASX
    b               .INTRA_PRED_PLANAR_32x32_LASX_TERMINATE

.INTRA_PRED_PLANAR_32x32_LASX_SWITCH_TO_HIHG_HALF_PART_LOOP:
    // new_index = y - 16, for get the high-short part data of left[]
    sub.d           a3,     t4,     a4
    xvreplve.b      xr9,    xr11,   a3
    vext2xv.hu.bu   xr9,    xr9
    b               .INTRA_PRED_PLANAR_32x32_LASX_WHILE_JUDGE

.INTRA_PRED_PLANAR_32x32_LASX_TERMINATE:
endfunc

/*
void intra_pred_ang8_c(pixel* dst, intptr_t dstStride, const pixel *srcPix0,
                      int dirMode, int bFilter)
*/
function x265_intra_pred_ang8_2_lsx
    addi.d    t0,    a2,    18
    vld       vr0,   t0,    0 //load srcPix0

    vbsrl.v   vr1,   vr0,   1
    fst.d     f0,    a0,    0
    fstx.d    f1,    a0,    a1
    alsl.d    a0,    a1,    a0,   1
    vbsrl.v   vr1,   vr0,   2
    vbsrl.v   vr2,   vr0,   3
    fst.d     f1,    a0,    0
    fstx.d    f2,    a0,    a1
    alsl.d    a0,    a1,    a0,   1
    vbsrl.v   vr1,   vr0,   4
    vbsrl.v   vr2,   vr0,   5
    fst.d     f1,    a0,    0
    fstx.d    f2,    a0,    a1
    alsl.d    a0,    a1,    a0,   1
    vbsrl.v   vr1,   vr0,   6
    vbsrl.v   vr2,   vr0,   7
    fst.d     f1,    a0,    0
    fstx.d    f2,    a0,    a1
endfunc

function x265_intra_pred_ang8_34_lsx
    addi.d    t0,    a2,    2
    vld       vr0,   t0,    0 //load srcPix0

    vbsrl.v   vr1,   vr0,   1
    fst.d     f0,    a0,    0
    fstx.d    f1,    a0,    a1
    alsl.d    a0,    a1,    a0,   1
    vbsrl.v   vr1,   vr0,   2
    vbsrl.v   vr2,   vr0,   3
    fst.d     f1,    a0,    0
    fstx.d    f2,    a0,    a1
    alsl.d    a0,    a1,    a0,   1
    vbsrl.v   vr1,   vr0,   4
    vbsrl.v   vr2,   vr0,   5
    fst.d     f1,    a0,    0
    fstx.d    f2,    a0,    a1
    alsl.d    a0,    a1,    a0,   1
    vbsrl.v   vr1,   vr0,   6
    vbsrl.v   vr2,   vr0,   7
    fst.d     f1,    a0,    0
    fstx.d    f2,    a0,    a1
endfunc

.macro TRANSPOSE4x16_B src0, src1, src2, src3, \
                       dst0, dst1, dst2, dst3
    vilvl.b    \dst0,   \src1,   \src0 // 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 (0 1)
    vilvh.b    \dst1,   \src1,   \src0 // 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 (4 5)
    vilvl.b    \dst2,   \src3,   \src2 // 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 (2 3)
    vilvh.b    \dst3,   \src3,   \src2 // 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 (6 7)

    vilvl.h    \src0,   \dst2,   \dst0 // 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 (0 1 2 3)
    vilvh.h    \src1,   \dst2,   \dst0 // 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7 (0 1 2 3)
    vilvl.h    \src2,   \dst3,   \dst1 // 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 (4 5 6 7)
    vilvh.h    \src3,   \dst3,   \dst1 // 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7 (4 5 6 7)

    vilvl.w    \dst0,   \src2,   \src0 // 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
    vilvh.w    \dst1,   \src2,   \src0 // 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3
    vilvl.w    \dst2,   \src3,   \src1 // 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5
    vilvh.w    \dst3,   \src3,   \src1 // 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7
.endm

const ang8_mode3, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-26, 26, 32-20, 20, 32-14, 14, 32-8, 8, 32-2, 2, 32-28, 28, 32-22, 22, 32-16, 16
endconst
function x265_intra_pred_ang8_3_lsx
    addi.d              t0,     a2,     1
    li.w                t1,     33
    beq                 t1,     a3,     1f
    addi.d              t0,     a2,     17
1:
    vld                 vr0,    t0,     0   //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 dirMode=3
    la.local            t0,     ang8_mode3
    vld                 vr1,    t0,     16  //fraction
    vld                 vr20,   t0,     0   //shuf

    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr9,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beq                 t1,     a3,     2f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   3f
2:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
3:
endfunc

.macro ANG8_STORE8x8_B
    xvstelm.d    xr7,    a0,    0,    0
    add.d        a0,     a0,    a1
    xvstelm.d    xr7,    a0,    0,    2
    add.d        a0,     a0,    a1
    xvstelm.d    xr7,    a0,    0,    1
    add.d        a0,     a0,    a1
    xvstelm.d    xr7,    a0,    0,    3
    add.d        a0,     a0,    a1
    xvstelm.d    xr13,   a0,    0,    0
    add.d        a0,     a0,    a1
    xvstelm.d    xr13,   a0,    0,    2
    add.d        a0,     a0,    a1
    xvstelm.d    xr13,   a0,    0,    1
    add.d        a0,     a0,    a1
    xvstelm.d    xr13,   a0,    0,    3
.endm

const lasx_ang8_mode3, align=5
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 4, 5, 5, 6, 6, 7, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 5, 6, 6, 7, 7, 8
.byte 32-26, 26, 32-20, 20, 32-14, 14, 32-8, 8, 32-2, 2, 32-28, 28, 32-22, 22, 32-16, 16
endconst
function x265_intra_pred_ang8_3_lasx
    vld                 vr0,    a2,     17 //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    la.local            t0,     lasx_ang8_mode3
    xvld                xr1,    t0,     32 //fraction
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvld                xr2,    t0,     0  //shuf
    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode33, align=5
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9
.byte 32-26, 26, 32-14, 14, 32-2, 2, 32-22, 22, 0,0,0,0,0,0,0,0, 32-20, 20, 32-8, 8, 32-28, 28, 32-16, 16, 0,0,0,0,0,0,0,0
endconst
function x265_intra_pred_ang8_33_lasx
    vld                 vr0,    a2,     1
    la.local            t0,     lasx_ang8_mode33
    xvld                xr1,    t0,     32  //fraction
    xvld                xr20,   t0,     0   //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr3,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr4,    xr3,    xr2 //offset = 0 1

    xvaddi.bu           xr20,   xr20,   2 //offset = 2 3
    xvrepl128vei.h      xr5,    xr1,    1
    xvshuf.b            xr6,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr7,    xr6,    xr5

    xvaddi.bu           xr20,   xr20,   2 //offset = 4 4
    xvrepl128vei.h      xr8,    xr1,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr20
    xvpermi.q           xr9,    xr9,    0x00
    xvmulwev.h.bu.b     xr10,   xr9,    xr8

    xvaddi.bu           xr20,   xr20,   1 //offset = 5 6
    xvrepl128vei.h      xr11,   xr1,    3
    xvshuf.b            xr12,   xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr13,   xr12,   xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr6,    xr5
    xvmaddwod.h.bu.b    xr10,   xr9,    xr8
    xvmaddwod.h.bu.b    xr13,   xr12,   xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const ang8_mode4, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-21, 21, 32-10, 10, 32-31, 31, 32-20, 20, 32-9, 9, 32-30, 30, 32-19, 19, 32-8, 8
endconst
function x265_intra_pred_ang8_4_lsx
    addi.d              t0,     a2,     1
    li.w                t1,     32
    beq                 t1,     a3,     1f
    addi.d              t0,     a2,     17
1:
    vld                 vr0,    t0,     0   //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 dirMode=4
    la.local            t0,     ang8_mode4
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beq                 t1,     a3,     2f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   3f
2:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
3:
endfunc

const lasx_ang8_mode4, align=5
.byte 0, 1, 1, 2, 1, 2, 2, 3, 3, 4, 3, 4, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 4, 5, 4, 5, 5, 6, 6, 7
.byte 32-21, 21, 32-10, 10, 32-31, 31, 32-20, 20, 32-9, 9, 32-30, 30, 32-19, 19, 32-8, 8
endconst
function x265_intra_pred_ang8_4_lasx
    xvld                xr0,    a2,     17 //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    la.local            t0,     lasx_ang8_mode4
    xvld                xr1,    t0,     32 //fraction
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvld                xr2,    t0,     0  //shuf
    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode32, align=5
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9
.byte 32-21, 21, 32-31, 31, 32-9, 9, 32-19, 19, 0,0,0,0,0,0,0,0, 32-10, 10, 32-20, 20, 32-30, 30, 32-8, 8, 0,0,0,0,0,0,0,0
endconst
function x265_intra_pred_ang8_32_lasx
    vld                 vr0,    a2,     1
    la.local            t0,     lasx_ang8_mode32
    xvld                xr1,    t0,     32  //fraction
    xvld                xr20,   t0,     0   //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr3,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr4,    xr3,    xr2 //offset = 0 1

    xvaddi.bu           xr20,   xr20,   1 //offset = 1 2
    xvrepl128vei.h      xr5,    xr1,    1
    xvshuf.b            xr6,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr7,    xr6,    xr5

    xvaddi.bu           xr20,   xr20,   2 //offset = 3 3
    xvrepl128vei.h      xr8,    xr1,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr20
    xvpermi.q           xr9,    xr9,    0x00
    xvmulwev.h.bu.b     xr10,   xr9,    xr8

    xvaddi.bu           xr20,   xr20,   1 //offset = 4 5
    xvrepl128vei.h      xr11,   xr1,    3
    xvshuf.b            xr12,   xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr13,   xr12,   xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr6,    xr5
    xvmaddwod.h.bu.b    xr10,   xr9,    xr8
    xvmaddwod.h.bu.b    xr13,   xr12,   xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const ang8_mode5, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-17, 17, 32-2, 2, 32-19, 19, 32-4, 4, 32-21, 21, 32-6, 6, 32-23, 23, 32-8, 8
endconst
function x265_intra_pred_ang8_5_lsx
    addi.d              t0,     a2,     1
    li.w                t1,     31
    beq                 t1,     a3,     1f
    addi.d              t0,     a2,     17
1:
    vld                 vr0,    t0,     0   //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 dirMode=5
    la.local            t0,     ang8_mode5
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr6,    vr1,    5
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beq                 t1,     a3,     2f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   3f
2:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
3:
endfunc

const lasx_ang8_mode5, align=5
.byte 0, 1, 1, 2, 1, 2, 2, 3, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 3, 4, 4, 5, 4, 5, 5, 6
.byte 32-17, 17, 32-2, 2, 32-19, 19, 32-4, 4, 32-21, 21, 32-6, 6, 32-23, 23, 32-8, 8
endconst
function x265_intra_pred_ang8_5_lasx
    vld                 vr0,    a2,     17 //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    la.local            t0,     lasx_ang8_mode5
    xvld                xr1,    t0,     32 //fraction
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvld                xr2,    t0,     0  //shuf
    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode31, align=5
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9
.byte 32-17, 17, 32-19, 19, 32-21, 21, 32-23, 23, 0,0,0,0,0,0,0,0, 32-2, 2, 32-4, 4, 32-6, 6, 32-8, 8, 0,0,0,0,0,0,0,0
endconst
function x265_intra_pred_ang8_31_lasx
    vld                 vr0,    a2,     1
    la.local            t0,     lasx_ang8_mode31
    xvld                xr1,    t0,     32  //fraction
    xvld                xr20,   t0,     0   //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr3,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr4,    xr3,    xr2 //offset = 0 1

    xvaddi.bu           xr20,   xr20,   1 //offset = 1 2
    xvrepl128vei.h      xr5,    xr1,    1
    xvshuf.b            xr6,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr7,    xr6,    xr5

    xvaddi.bu           xr20,   xr20,   1 //offset = 2 3
    xvrepl128vei.h      xr8,    xr1,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr10,   xr9,    xr8

    xvaddi.bu           xr20,   xr20,   1 //offset = 3 4
    xvrepl128vei.h      xr11,   xr1,    3
    xvshuf.b            xr12,   xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr13,   xr12,   xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr6,    xr5
    xvmaddwod.h.bu.b    xr10,   xr9,    xr8
    xvmaddwod.h.bu.b    xr13,   xr12,   xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const ang8_mode6, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-13, 13, 32-26, 26, 32-7, 7, 32-20, 20, 32-1, 1, 32-14, 14, 32-27, 27, 32-8, 8
endconst
function x265_intra_pred_ang8_6_lsx
    addi.d              t0,     a2,     1
    li.w                t1,     30
    beq                 t1,     a3,     1f
    addi.d              t0,     a2,     17
1:
    vld                 vr0,    t0,     0   //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 dirMode=6
    la.local            t0,     ang8_mode6
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr9,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr9,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr9,    vr11

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beq                 t1,     a3,     2f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   3f
2:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
3:
endfunc

const lasx_ang8_mode6, align=5
.byte 0, 1, 0, 1, 1, 2, 1, 2, 2, 3, 2, 3, 2, 3, 3, 4, 1, 2, 1, 2, 2, 3, 2, 3, 3, 4, 3, 4, 3, 4, 4, 5
.byte 32-13, 13, 32-26, 26, 32-7, 7, 32-20, 20, 32-1, 1, 32-14, 14, 32-27, 27, 32-8, 8
endconst
function x265_intra_pred_ang8_6_lasx
    vld                 vr0,    a2,     17 //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    la.local            t0,     lasx_ang8_mode6
    xvld                xr1,    t0,     32 //fraction
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvld                xr2,    t0,     0  //shuf
    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode30, align=5
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9
.byte 32-13, 13, 32-7, 7, 32-1, 1, 32-27, 27, 0,0,0,0,0,0,0,0, 32-26, 26, 32-20, 20, 32-14, 14, 32-8, 8, 0,0,0,0,0,0,0,0
endconst
function x265_intra_pred_ang8_30_lasx
    vld                 vr0,    a2,     1
    la.local            t0,     lasx_ang8_mode30
    xvld                xr1,    t0,     32  //fraction
    xvld                xr20,   t0,     0   //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr3,    xr0,    xr0,    xr20
    xvpermi.q           xr6,    xr3,    0x11 //offset = 1 1
    xvpermi.q           xr3,    xr3,    0x00
    xvmulwev.h.bu.b     xr4,    xr3,    xr2  //offset = 0 0

    xvrepl128vei.h      xr5,    xr1,    1
    xvmulwev.h.bu.b     xr7,    xr6,    xr5

    xvaddi.bu           xr20,   xr20,   2 //offset = 2 3
    xvrepl128vei.h      xr8,    xr1,    2
    xvshuf.b            xr12,   xr0,    xr0,    xr20
    xvpermi.q           xr9,    xr12,   0x00 //offset = 2 2
    xvmulwev.h.bu.b     xr10,   xr9,    xr8

    xvrepl128vei.h      xr11,   xr1,    3
    xvmulwev.h.bu.b     xr13,   xr12,   xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr6,    xr5
    xvmaddwod.h.bu.b    xr10,   xr9,    xr8
    xvmaddwod.h.bu.b    xr13,   xr12,   xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const ang8_mode7, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-9, 9, 32-18, 18, 32-27, 27, 32-4, 4, 32-13, 13, 32-22, 22, 32-31, 31, 32-8, 8
endconst
function x265_intra_pred_ang8_7_lsx
    addi.d              t0,     a2,     1
    li.w                t1,     29
    beq                 t1,     a3,     1f
    addi.d              t0,     a2,     17
1:
    vld                 vr0,    t0,     0   //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 dirMode=7
    la.local            t0,     ang8_mode7
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beq                 t1,     a3,     2f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   3f
2:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
3:
endfunc

const lasx_ang8_mode7, align=5
.byte 0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 3, 1, 2, 1, 2, 1, 2, 2, 3, 2, 3, 2, 3, 2, 3, 3, 4
.byte 32-9, 9, 32-18, 18, 32-27, 27, 32-4, 4, 32-13, 13, 32-22, 22, 32-31, 31, 32-8, 8
endconst
function x265_intra_pred_ang8_7_lasx
    vld                 vr0,    a2,     17 //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    la.local            t0,     lasx_ang8_mode7
    xvld                xr1,    t0,     32 //fraction
    xvld                xr2,    t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode29, align=5
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9
.byte 32-9, 9, 32-27, 27, 32-13, 13, 32-31, 31, 0,0,0,0,0,0,0,0, 32-18, 18, 32-4, 4, 32-22, 22, 32-8, 8, 0,0,0,0,0,0,0,0
endconst
function x265_intra_pred_ang8_29_lasx
    vld                 vr0,    a2,     1
    la.local            t0,     lasx_ang8_mode29
    xvld                xr1,    t0,     32 //fraction
    xvld                xr20,   t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr6,    xr0,    xr0,    xr20
    xvpermi.q           xr3,    xr6,    0x00
    xvmulwev.h.bu.b     xr4,    xr3,    xr2 //offset = 0 0

    xvrepl128vei.h      xr5,    xr1,    1
    xvmulwev.h.bu.b     xr7,    xr6,    xr5 //offset = 0 1

    xvaddi.bu           xr20,   xr20,   1 //offset = 1 2
    xvrepl128vei.h      xr8,    xr1,    2
    xvshuf.b            xr12,   xr0,    xr0,    xr20
    xvpermi.q           xr9,    xr12,   0x00 //offset = 1 1
    xvmulwev.h.bu.b     xr10,   xr9,    xr8

    xvrepl128vei.h      xr11,   xr1,    3
    xvmulwev.h.bu.b     xr13,   xr12,   xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr6,    xr5
    xvmaddwod.h.bu.b    xr10,   xr9,    xr8
    xvmaddwod.h.bu.b    xr13,   xr12,   xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const ang8_mode8, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-5, 5, 32-10, 10, 32-15, 15, 32-20, 20, 32-25, 25, 32-30, 30, 32-3, 3, 32-8, 8
endconst
function x265_intra_pred_ang8_8_lsx
    addi.d              t0,     a2,     1
    li.w                t1,     28
    beq                 t1,     a3,     1f
    addi.d              t0,     a2,     17
1:
    vld                 vr0,    t0,     0   //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 dirMode=8
    la.local            t0,     ang8_mode8
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr12,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr12,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beq                 t1,     a3,     2f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   3f
2:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
3:
endfunc

const lasx_ang8_mode8, align=5
.byte 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 3, 2, 3
.byte 32-5, 5, 32-10, 10, 32-15, 15, 32-20, 20, 32-25, 25, 32-30, 30, 32-3, 3, 32-8, 8
endconst
function x265_intra_pred_ang8_8_lasx
    vld                 vr0,    a2,     17 //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    la.local            t0,     lasx_ang8_mode8
    xvld                xr1,    t0,     32 //fraction
    xvld                xr2,    t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode28, align=5
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9
.byte 32-5, 5, 32-15, 15, 32-25, 25, 32-3, 3, 0,0,0,0,0,0,0,0, 32-10, 10, 32-20, 20, 32-30, 30, 32-8, 8, 0,0,0,0,0,0,0,0
endconst
function x265_intra_pred_ang8_28_lasx
    vld                 vr0,    a2,     1
    la.local            t0,     lasx_ang8_mode28
    xvld                xr1,    t0,     32 //fraction
    xvld                xr20,   t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr6,    xr0,    xr0,    xr20
    xvpermi.q           xr12,   xr6,    0x11
    xvpermi.q           xr3,    xr6,    0x00
    xvmulwev.h.bu.b     xr4,    xr3,    xr2 //offset = 0 0

    xvrepl128vei.h      xr5,    xr1,    1
    xvmulwev.h.bu.b     xr7,    xr3,    xr5

    xvaddi.bu           xr20,   xr20,   1
    xvrepl128vei.h      xr8,    xr1,    2
    xvmulwev.h.bu.b     xr10,   xr3,    xr8

    xvrepl128vei.h      xr11,   xr1,    3
    xvmulwev.h.bu.b     xr13,   xr12,   xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr3,    xr5
    xvmaddwod.h.bu.b    xr10,   xr3,    xr8
    xvmaddwod.h.bu.b    xr13,   xr12,   xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const ang8_mode9, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-2, 2, 32-4, 4, 32-6, 6, 32-8, 8, 32-10, 10, 32-12, 12, 32-14, 14, 32-16, 16
endconst
function x265_intra_pred_ang8_9_lsx
    addi.d              t0,     a2,     1
    li.w                t1,     27
    beq                 t1,     a3,     1f
    addi.d              t0,     a2,     17
1:
    vld                 vr0,    t0,     0   //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 dirMode=9
    la.local            t0,     ang8_mode9
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beq                 t1,     a3,     2f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   3f
2:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
3:
endfunc

const lasx_ang8_mode9, align=5
.byte 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2
.byte 32-2, 2, 32-4, 4, 32-6, 6, 32-8, 8, 32-10, 10, 32-12, 12, 32-14, 14, 32-16, 16
endconst
function x265_intra_pred_ang8_9_lasx
    vld                 vr0,    a2,     17 //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    la.local            t0,     lasx_ang8_mode9
    xvld                xr1,    t0,     32 //fraction
    xvld                xr2,    t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode27, align=5
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9
.byte 32-2, 2, 32-6, 6, 32-10, 10, 32-14, 14, 0,0,0,0,0,0,0,0, 32-4, 4, 32-8, 8, 32-12, 12, 32-16, 16, 0,0,0,0,0,0,0,0
endconst
function x265_intra_pred_ang8_27_lasx
    vld                 vr0,    a2,     1
    la.local            t0,     lasx_ang8_mode27
    xvld                xr1,    t0,     32 //fraction
    xvld                xr20,   t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr6,    xr0,    xr0,    xr20
    xvpermi.q           xr3,    xr6,    0x00
    xvmulwev.h.bu.b     xr4,    xr3,    xr2 //offset = 0 0

    xvrepl128vei.h      xr5,    xr1,    1
    xvmulwev.h.bu.b     xr7,    xr3,    xr5

    xvrepl128vei.h      xr8,    xr1,    2
    xvmulwev.h.bu.b     xr10,   xr3,    xr8

    xvrepl128vei.h      xr11,   xr1,    3
    xvmulwev.h.bu.b     xr13,   xr3,    xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr3,    xr5
    xvmaddwod.h.bu.b    xr10,   xr3,    xr8
    xvmaddwod.h.bu.b    xr13,   xr3,    xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

function x265_intra_pred_ang8_10_lsx
    addi.d              t0,     a2,     17
    vld                 vr0,    t0,     0 //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 dirMode=10
    vreplvei.b          vr2,    vr0,    0
    vreplvei.b          vr3,    vr0,    1
    vreplvei.b          vr4,    vr0,    2
    vreplvei.b          vr5,    vr0,    3
    vreplvei.b          vr6,    vr0,    4
    vreplvei.b          vr7,    vr0,    5
    vreplvei.b          vr8,    vr0,    6
    vreplvei.b          vr9,    vr0,    7

    beqz                a4,     1f

    //bFilter
    vld                 vr10,   a2,     0
    vbsrl.v             vr11,   vr10,   1
    vsllwil.hu.bu       vr11,   vr11,   0 //srcPix[width2 + 1 + y]
    vsllwil.hu.bu       vr10,   vr10,   0
    vreplvei.h          vr13,   vr10,   0 //topLeft
    vsllwil.hu.bu       vr14,   vr2,    0
    vreplvei.h          vr14,   vr14,   0 //top
    vsub.h              vr11,   vr11,   vr13
    vsrai.h             vr11,   vr11,   1
    vadd.h              vr11,   vr11,   vr14
    vssrani.bu.h        vr2,    vr11,   0
1:
    fst.d               f2,     a0,     0
    fstx.d              f3,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    fst.d               f4,     a0,     0
    fstx.d              f5,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    fst.d               f6,     a0,     0
    fstx.d              f7,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    fst.d               f8,     a0,     0
    fstx.d              f9,     a0,     a1
endfunc

function x265_intra_pred_ang8_26_lsx
    addi.d              t0,     a2,     1
    vld                 vr0,    t0,     0 //1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 dirMode=26

    beqz                a4,     1f

    //bFilter
    addi.d              t0,     a2,     17
    vld                 vr1,    t0,     0
    vsllwil.hu.bu       vr2,    vr1,    0 //srcPix[width2 + 1 + y]
    vldrepl.b           vr3,    a2,     0
    vsllwil.hu.bu       vr3,    vr3,    0 //topLeft
    vsllwil.hu.bu       vr4,    vr0,    0
    vreplvei.h          vr4,    vr4,    0
    vsub.h              vr2,    vr2,    vr3
    vsrai.h             vr2,    vr2,    1
    vadd.h              vr2,    vr2,    vr4
    vssrani.bu.h        vr2,    vr2,    0

    vextrins.b          vr0,    vr2,    0x00
    fst.d               f0,     a0,     0
    vextrins.b          vr0,    vr2,    0x01
    fstx.d              f0,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    vextrins.b          vr0,    vr2,    0x02
    fst.d               f0,     a0,     0
    vextrins.b          vr0,    vr2,    0x03
    fstx.d              f0,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    vextrins.b          vr0,    vr2,    0x04
    fst.d               f0,     a0,     0
    vextrins.b          vr0,    vr2,    0x05
    fstx.d              f0,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    vextrins.b          vr0,    vr2,    0x06
    fst.d               f0,     a0,     0
    vextrins.b          vr0,    vr2,    0x07
    fstx.d              f0,     a0,     a1
    b                   2f
1:
    fst.d               f0,     a0,     0
    fstx.d              f0,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    fst.d               f0,     a0,     0
    fstx.d              f0,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    fst.d               f0,     a0,     0
    fstx.d              f0,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    fst.d               f0,     a0,     0
    fstx.d              f0,     a0,     a1
2:
endfunc

const ang8_mode11, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-30, 30, 32-28, 28, 32-26, 26, 32-24, 24, 32-22, 22, 32-20, 20, 32-18, 18, 32-16, 16
endconst
function x265_intra_pred_ang8_11_lsx
    li.w                t2,     16
    addi.w              t1,     a3,     -25
    maskeqz             t0,     t2,     t1 // dirMode=25
    add.d               t0,     t0,     a2

    vld                 vr0,    t0,     0   //16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    vldrepl.b           vr20,   a2,     0
    vextrins.b          vr0,    vr20,   0x00 //insert to ref_pix[-1]

    la.local            t0,     ang8_mode11
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beqz                t1,     1f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   2f
1:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
2:
endfunc

const lasx_ang8_mode11, align=5
.byte 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2
.byte 32-30, 30, 32-28, 28, 32-26, 26, 32-24, 24, 32-22, 22, 32-20, 20, 32-18, 18, 32-16, 16
endconst
function x265_intra_pred_ang8_11_lasx
    vld                 vr0,    a2,     16 //16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    vldrepl.b           vr1,    a2,     0
    vextrins.b          vr0,    vr1,    0x00
    la.local            t0,     lasx_ang8_mode11
    xvld                xr1,    t0,     32 //fraction
    xvld                xr2,    t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode25, align=5
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-30, 30, 32-26, 26, 32-22, 22, 32-18, 18, 0,0,0,0,0,0,0,0, 32-28, 28, 32-24, 24, 32-20, 20, 32-16, 16, 0,0,0,0,0,0,0,0
endconst
function x265_intra_pred_ang8_25_lasx
    vld                 vr0,    a2,     0
    la.local            t0,     lasx_ang8_mode25
    xvld                xr1,    t0,     32 //fraction
    xvld                xr20,   t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr3,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr4,    xr3,    xr2 //offset = 0 0

    xvrepl128vei.h      xr5,    xr1,    1
    xvmulwev.h.bu.b     xr7,    xr3,    xr5

    xvrepl128vei.h      xr8,    xr1,    2
    xvmulwev.h.bu.b     xr10,   xr3,    xr8

    xvrepl128vei.h      xr11,   xr1,    3
    xvmulwev.h.bu.b     xr13,   xr3,    xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr3,    xr5
    xvmaddwod.h.bu.b    xr10,   xr3,    xr8
    xvmaddwod.h.bu.b    xr13,   xr3,    xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const ang8_mode12, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-27, 27, 32-22, 22, 32-17, 17, 32-12, 12, 32-7, 7, 32-2, 2, 32-29, 29, 32-24, 24
endconst
function x265_intra_pred_ang8_12_lsx
    li.w                t2,     16
    addi.w              t1,     a3,     -24
    maskeqz             t0,     t2,     t1 //dirMode=24
    masknez             t3,     t2,     t1 //dirMode=12
    add.d               t0,     t0,     a2
    add.d               t2,     t3,     a2

    vld                 vr0,    t0,     0   //16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    vbsll.v             vr0,    vr0,    1   //x 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
    vld                 vr20,   a2,     0
    vld                 vr21,   t2,     0
    vextrins.b          vr0,    vr21,   0x06 //insert to ref_pix[-2]
    vextrins.b          vr0,    vr20,   0x10 //insert to ref_pix[-1]

    la.local            t0,     ang8_mode12
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vbsrl.v             vr21,   vr0,    1   //ref_pix[-1]
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr12,   vr0,    vr0,    vr20 //ref_pix[-2]
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr12,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr12,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beqz                t1,     1f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   2f
1:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
2:
endfunc

const lasx_ang8_mode12, align=5
.byte 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 1, 2, 1, 2
.byte 32-27, 27, 32-22, 22, 32-17, 17, 32-12, 12, 32-7, 7, 32-2, 2, 32-29, 29, 32-24, 24
endconst
function x265_intra_pred_ang8_12_lasx
    vld                 vr0,    a2,     15 //15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    vld                 vr1,    a2,     0
    vextrins.b          vr0,    vr1,    0x06
    vextrins.b          vr0,    vr1,    0x10
    la.local            t0,     lasx_ang8_mode12
    xvld                xr1,    t0,     32 //fraction
    xvld                xr2,    t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode24, align=5
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-27, 27, 32-17, 17, 32-7, 7, 32-29, 29, 0,0,0,0,0,0,0,0, 32-22, 22, 32-12, 12, 32-2, 2, 32-24, 24, 0,0,0,0,0,0,0,0
endconst
function x265_intra_pred_ang8_24_lasx
    vld                 vr0,    a2,     0
    vbsll.v             vr0,    vr0,    1
    vldrepl.b           vr1,    a2,     16+6
    vextrins.b          vr0,    vr1,    0x00
    la.local            t0,     lasx_ang8_mode24
    xvld                xr1,    t0,     32 //fraction
    xvld                xr20,   t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvaddi.bu           xr14,   xr20,   1
    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr3,    xr0,    xr0,    xr14 //offset = -1 -1
    xvmulwev.h.bu.b     xr4,    xr3,    xr2

    xvrepl128vei.h      xr5,    xr1,    1
    xvmulwev.h.bu.b     xr7,    xr3,    xr5

    xvrepl128vei.h      xr8,    xr1,    2
    xvmulwev.h.bu.b     xr10,   xr3,    xr8

    xvshuf.b            xr6,    xr0,    xr0,    xr20
    xvrepl128vei.h      xr11,   xr1,    3
    xvmulwev.h.bu.b     xr13,   xr6,    xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr3,    xr5
    xvmaddwod.h.bu.b    xr10,   xr3,    xr8
    xvmaddwod.h.bu.b    xr13,   xr6,    xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const ang8_mode13, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-23, 23, 32-14, 14, 32-5, 5, 32-28, 28, 32-19, 19, 32-10, 10, 32-1, 1, 32-24, 24
endconst
function x265_intra_pred_ang8_13_lsx
    li.w                t2,     16
    addi.w              t1,     a3,     -23
    maskeqz             t0,     t2,     t1 //dirMode=23
    masknez             t3,     t2,     t1 //dirMode=13
    add.d               t0,     t0,     a2
    add.d               t2,     t3,     a2

    vld                 vr0,    t0,     0   //16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    vbsll.v             vr0,    vr0,    2   //x 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
    vld                 vr20,   a2,     0
    vld                 vr21,   t2,     0
    vextrins.b          vr0,    vr21,   0x07 //insert to ref_pix[-3]
    vextrins.b          vr0,    vr21,   0x14 //insert to ref_pix[-2]
    vextrins.b          vr0,    vr20,   0x20 //insert to ref_pix[-1]

    la.local            t0,     ang8_mode13
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vbsrl.v             vr21,   vr0,    2   //ref_pix[-1]
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vbsrl.v             vr21,   vr0,    1  //ref_pix[-2]
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr21,   vr21,     vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr12,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20 //ref_pix[-3]
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beqz                t1,     1f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   2f
1:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
2:
endfunc

const lasx_ang8_mode13, align=5
.byte 2, 3, 2, 3, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 3, 4, 3, 4, 3, 4, 2, 3, 2, 3, 2, 3, 2, 3, 1, 2
.byte 32-23, 23, 32-14, 14, 32-5, 5, 32-28, 28, 32-19, 19, 32-10, 10, 32-1, 1, 32-24, 24
endconst
function x265_intra_pred_ang8_13_lasx
    vld                 vr0,    a2,     14 //14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    vld                 vr1,    a2,     0
    vextrins.b          vr0,    vr1,    0x07
    vextrins.b          vr0,    vr1,    0x14
    vextrins.b          vr0,    vr1,    0x20
    la.local            t0,     lasx_ang8_mode13
    xvld                xr1,    t0,     32 //fraction
    xvld                xr2,    t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode23, align=5
.byte 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-23, 23, 32-5, 5, 32-19, 19, 32-1, 1, 0,0,0,0,0,0,0,0, 32-14, 14, 32-28, 28, 32-10, 10, 32-24, 24, 0,0,0,0,0,0,0,0
endconst
function x265_intra_pred_ang8_23_lasx
    vld                 vr0,    a2,     0
    vbsll.v             vr0,    vr0,    2
    vld                 vr1,    a2,     16
    vextrins.b          vr0,    vr1,    0x07
    vextrins.b          vr0,    vr1,    0x14
    la.local            t0,     lasx_ang8_mode23
    xvld                xr1,    t0,     32 //fraction
    xvld                xr20,   t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvaddi.bu           xr14,   xr20,   1 //offset = -1 -2
    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr6,    xr0,    xr0,    xr14
    xvpermi.q           xr3,    xr6,    0x00 //-1-1
    xvpermi.q           xr9,    xr6,    0x11 //-2-2
    xvmulwev.h.bu.b     xr4,    xr3,    xr2

    xvrepl128vei.h      xr5,    xr1,    1
    xvmulwev.h.bu.b     xr7,    xr6,    xr5

    xvrepl128vei.h      xr8,    xr1,    2
    xvmulwev.h.bu.b     xr10,   xr9,    xr8

    xvshuf.b            xr12,   xr0,    xr0,    xr20
    xvrepl128vei.h      xr11,   xr1,    3
    xvmulwev.h.bu.b     xr13,   xr12,   xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr6,    xr5
    xvmaddwod.h.bu.b    xr10,   xr9,    xr8
    xvmaddwod.h.bu.b    xr13,   xr12,   xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const ang8_mode14, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-19, 19, 32-6, 6, 32-25, 25, 32-12, 12, 32-31, 31, 32-18, 18, 32-5, 5, 32-24, 24
endconst
function x265_intra_pred_ang8_14_lsx
    li.w                t2,     16
    addi.w              t1,     a3,     -22
    maskeqz             t0,     t2,     t1 //dirMode=22
    masknez             t3,     t2,     t1 //dirMode=14
    add.d               t0,     t0,     a2
    add.d               t2,     t3,     a2

    vld                 vr0,    t0,     0   //16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    vldrepl.b           vr20,   a2,     0
    vld                 vr21,   t2,     0
    vbsll.v             vr0,    vr0,    3   //x x x 16 17 18 19 20 21 22 23 24 25 26 27 28 29
    vextrins.b          vr0,    vr21,   0x07 //insert to ref_pix[-4]
    vextrins.b          vr0,    vr21,   0x15 //insert to ref_pix[-3]
    vextrins.b          vr0,    vr21,   0x22 //insert to ref_pix[-2]
    vextrins.b          vr0,    vr20,   0x30 //insert to ref_pix[-1]

    la.local            t0,     ang8_mode14
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vbsrl.v             vr21,   vr0,    3   //ref_pix[-1]
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vbsrl.v             vr6,    vr0,    2  //ref_pix[-2]
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr9,    vr6,    vr6,     vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr9,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr9,    vr11

    vbsrl.v             vr3,    vr0,    1  //ref_pix[-3]
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr12,   vr3,    vr3,     vr20
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,     vr20 //ref_pix[-4]
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beqz                t1,     1f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   2f
1:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
2:
endfunc

const lasx_ang8_mode14, align=5
.byte 3, 4, 3, 4, 2, 3, 2, 3, 1, 2, 1, 2, 1, 2, 0, 1, 4, 5, 4, 5, 3, 4, 3, 4, 2, 3, 2, 3, 2, 3, 1, 2
.byte 32-19, 19, 32-6, 6, 32-25, 25, 32-12, 12, 32-31, 31, 32-18, 18, 32-5, 5, 32-24, 24
.byte 7, 5, 2, 0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
endconst
function x265_intra_pred_ang8_14_lasx
    vld                 vr0,    a2,     17 //17 18 19 20 21 22 23 24 25 26 27 28 29
    vld                 vr1,    a2,     0
    la.local            t0,     lasx_ang8_mode14
    vld                 vr2,    t0,     48
    vshuf.b             vr0,    vr0,    vr1,    vr2
    xvld                xr1,    t0,     32 //fraction
    xvld                xr2,    t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode22, align=5
.byte 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-19, 19, 32-25, 25, 32-31, 31, 32-5, 5, 0,0,0,0,0,0,0,0, 32-6, 6, 32-12, 12, 32-18, 18, 32-24, 24, 0,0,0,0,0,0,0,0
.byte 7, 5, 2, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
endconst
function x265_intra_pred_ang8_22_lasx
    vld                 vr0,    a2,     0
    vld                 vr1,    a2,     16
    la.local            t0,     lasx_ang8_mode22
    vld                 vr2,    t0,     64
    vshuf.b             vr0,    vr0,    vr1,    vr2
    xvld                xr1,    t0,     32 //fraction
    xvld                xr20,   t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvaddi.bu           xr14,   xr20,   2 //offset = -1-2
    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr6,    xr0,    xr0,    xr14
    xvpermi.q           xr3,    xr6,    0x00 //-1-1
    xvpermi.q           xr6,    xr6,    0x11 //-2-2
    xvmulwev.h.bu.b     xr4,    xr3,    xr2

    xvrepl128vei.h      xr5,    xr1,    1
    xvmulwev.h.bu.b     xr7,    xr6,    xr5

    xvshuf.b            xr12,   xr0,    xr0,    xr20 //offset = -3-4
    xvpermi.q           xr9,    xr12,   0x00
    xvrepl128vei.h      xr8,    xr1,    2
    xvmulwev.h.bu.b     xr10,   xr9,    xr8

    xvrepl128vei.h      xr11,   xr1,    3
    xvmulwev.h.bu.b     xr13,   xr12,   xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr6,    xr5
    xvmaddwod.h.bu.b    xr10,   xr9,    xr8
    xvmaddwod.h.bu.b    xr13,   xr12,   xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const ang8_mode15, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-15, 15, 32-30, 30, 32-13, 13, 32-28, 28, 32-11, 11, 32-26, 26, 32-9, 9, 32-24, 24
endconst
function x265_intra_pred_ang8_15_lsx
    li.w                t2,     16
    addi.w              t1,     a3,     -21
    maskeqz             t0,     t2,     t1 //dirMode=21
    masknez             t3,     t2,     t1 //dirMode=15
    add.d               t0,     t0,     a2
    add.d               t2,     t3,     a2

    vld                 vr0,    t0,     0   //16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    vldrepl.b           vr20,   a2,     0
    vld                 vr21,   t2,     0
    vbsll.v             vr0,    vr0,    4   //x x x x 16 17 18 19 20 21 22 23 24 25 26 27 28 29
    vextrins.b          vr0,    vr21,   0x08
    vextrins.b          vr0,    vr21,   0x16
    vextrins.b          vr0,    vr21,   0x24
    vextrins.b          vr0,    vr21,   0x32
    vextrins.b          vr0,    vr20,   0x40 //insert to ref_pix[-1]

    la.local            t0,     ang8_mode15
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vbsrl.v             vr21,   vr0,    4   //ref_pix[-1]
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vbsrl.v             vr21,   vr0,    3   //ref_pix[-2]
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr21,   vr0,    2   //ref_pix[-3]
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vbsrl.v             vr21,   vr0,    1   //ref_pix[-4]
    vreplvei.h          vr6,    vr1,    5
    vshuf.b             vr18,   vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr9,    vr18,   vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr18,   vr11

    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,     vr20 //ref_pix[-5]
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr18,   vr6
    vmaddwod.h.bu.b     vr14,   vr18,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beqz                t1,     1f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   2f
1:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
2:
endfunc

const lasx_ang8_mode15, align=5
.byte 4, 5, 3, 4, 3, 4, 2, 3, 2, 3, 1, 2, 1, 2, 0, 1, 5, 6, 4, 5, 4, 5, 3, 4, 3, 4, 2, 3, 2, 3, 1, 2
.byte 32-15, 15, 32-30, 30, 32-13, 13, 32-28, 28, 32-11, 11, 32-26, 26, 32-9, 9, 32-24, 24
.byte 8, 6, 4, 2, 0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
endconst
function x265_intra_pred_ang8_15_lasx
    vld                 vr0,    a2,     17 //17 18 19 20 21 22 23 24 25 26 27 28 29
    vld                 vr1,    a2,     0
    la.local            t0,     lasx_ang8_mode15
    vld                 vr2,    t0,     48
    vshuf.b             vr0,    vr0,    vr1,    vr2
    xvld                xr1,    t0,     32 //fraction
    xvld                xr2,    t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode21, align=5
.byte 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-15, 15, 32-13, 13, 32-11, 11, 32-9, 9, 0,0,0,0,0,0,0,0, 32-30, 30, 32-28, 28, 32-26, 26, 32-24, 24, 0,0,0,0,0,0,0,0
.byte 8, 6, 4, 2, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
endconst
function x265_intra_pred_ang8_21_lasx
    vld                 vr0,    a2,     0
    vld                 vr1,    a2,     16
    la.local            t0,     lasx_ang8_mode21
    vld                 vr2,    t0,     64
    vshuf.b             vr0,    vr0,    vr1,    vr2
    xvld                xr1,    t0,     32 //fraction
    xvld                xr20,   t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvaddi.bu           xr14,   xr20,   3 //offset = -1-2
    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr3,    xr0,    xr0,    xr14
    xvmulwev.h.bu.b     xr4,    xr3,    xr2

    xvaddi.bu           xr14,   xr20,   2 //offset = -2-3
    xvrepl128vei.h      xr5,    xr1,    1
    xvshuf.b            xr6,    xr0,    xr0,    xr14
    xvmulwev.h.bu.b     xr7,    xr6,    xr5

    xvaddi.bu           xr14,   xr20,   1 //offset = -3-4
    xvshuf.b            xr9,    xr0,    xr0,    xr14
    xvrepl128vei.h      xr8,    xr1,    2
    xvmulwev.h.bu.b     xr10,   xr9,    xr8

    xvshuf.b            xr12,   xr0,    xr0,    xr20
    xvrepl128vei.h      xr11,   xr1,    3
    xvmulwev.h.bu.b     xr13,   xr12,   xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr6,    xr5
    xvmaddwod.h.bu.b    xr10,   xr9,    xr8
    xvmaddwod.h.bu.b    xr13,   xr12,   xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const ang8_mode16, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-11, 11, 32-22, 22, 32-1, 1, 32-12, 12, 32-23, 23, 32-2, 2, 32-13, 13, 32-24, 24
endconst
function x265_intra_pred_ang8_16_lsx
    li.w                t2,     16
    addi.w              t1,     a3,     -20
    maskeqz             t0,     t2,     t1 //dirMode=20
    masknez             t3,     t2,     t1 //dirMode=16
    add.d               t0,     t0,     a2
    add.d               t2,     t3,     a2

    vld                 vr0,    t0,     0   //16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    vldrepl.b           vr20,   a2,     0
    vld                 vr21,   t2,     0
    vbsll.v             vr0,    vr0,    5   //x x x x x 16 17 18 19 20 21 22 23 24 25 26 27 28 29
    vextrins.b          vr0,    vr21,   0x08
    vextrins.b          vr0,    vr21,   0x16
    vextrins.b          vr0,    vr21,   0x25
    vextrins.b          vr0,    vr21,   0x33
    vextrins.b          vr0,    vr21,   0x42
    vextrins.b          vr0,    vr20,   0x50 //insert to ref_pix[-1]

    la.local            t0,     ang8_mode16
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vbsrl.v             vr21,   vr0,    5   //ref_pix[-1]
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vbsrl.v             vr21,   vr0,    4   //ref_pix[-2]
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr21,   vr0,    3   //ref_pix[-3]
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr21,   vr0,    2   //ref_pix[-4]
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr12,   vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vbsrl.v             vr21,   vr0,    1   //ref_pix[-5]
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr18,   vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr14,   vr18,   vr11

    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,     vr20 //ref_pix[-6]
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr18,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beqz                t1,     1f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   2f
1:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
2:
endfunc

const lasx_ang8_mode16, align=5
.byte 5, 6, 4, 5, 4, 5, 3, 4, 2, 3, 2, 3, 1, 2, 0, 1, 6, 7, 5, 6, 5, 6, 4, 5, 3, 4, 3, 4, 2, 3, 1, 2
.byte 32-11, 11, 32-22, 22, 32-1, 1, 32-12, 12, 32-23, 23, 32-2, 2, 32-13, 13, 32-24, 24
.byte 8, 6, 5, 3, 2, 0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
endconst
function x265_intra_pred_ang8_16_lasx
    vld                 vr0,    a2,     17 //17 18 19 20 21 22 23 24 25 26 27 28 29
    vld                 vr1,    a2,     0
    la.local            t0,     lasx_ang8_mode16
    vld                 vr2,    t0,     48
    vshuf.b             vr0,    vr0,    vr1,    vr2
    xvld                xr1,    t0,     32 //fraction
    xvld                xr2,    t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode20, align=5
.byte 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-11, 11, 32-1, 1, 32-23, 23, 32-13, 13, 0,0,0,0,0,0,0,0, 32-22, 22, 32-12, 12, 32-2, 2, 32-24, 24, 0,0,0,0,0,0,0,0
.byte 8, 6, 5, 3, 2, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
endconst
function x265_intra_pred_ang8_20_lasx
    vld                 vr0,    a2,     0
    vld                 vr1,    a2,     16
    la.local            t0,     lasx_ang8_mode20
    vld                 vr2,    t0,     64
    vshuf.b             vr0,    vr0,    vr1,    vr2
    xvld                xr1,    t0,     32 //fraction
    xvld                xr20,   t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvaddi.bu           xr14,   xr20,   4 //offset = -1-2
    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr3,    xr0,    xr0,    xr14
    xvmulwev.h.bu.b     xr4,    xr3,    xr2

    xvaddi.bu           xr14,   xr20,   3 //offset = -2-3
    xvrepl128vei.h      xr5,    xr1,    1
    xvshuf.b            xr6,    xr0,    xr0,    xr14
    xvmulwev.h.bu.b     xr7,    xr6,    xr5

    xvaddi.bu           xr14,   xr20,   2 //offset = -3-4
    xvrepl128vei.h      xr8,    xr1,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr14
    xvpermi.q           xr9,    xr9,    0x11
    xvmulwev.h.bu.b     xr10,   xr9,    xr8

    xvrepl128vei.h      xr11,   xr1,    3
    xvshuf.b            xr12,   xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr13,   xr12,   xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr6,    xr5
    xvmaddwod.h.bu.b    xr10,   xr9,    xr8
    xvmaddwod.h.bu.b    xr13,   xr12,   xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const ang8_mode17, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-6, 6, 32-12, 12, 32-18, 18, 32-24, 24, 32-30, 30, 32-4, 4, 32-10, 10, 32-16, 16
endconst
function x265_intra_pred_ang8_17_lsx
    li.w                t2,     16
    addi.w              t1,     a3,     -19
    maskeqz             t0,     t2,     t1 //dirMode=19
    masknez             t3,     t2,     t1 //dirMode=17
    add.d               t0,     t0,     a2
    add.d               t2,     t3,     a2

    vld                 vr0,    t0,     0   //16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    vldrepl.b           vr20,   a2,     0
    vld                 vr21,   t2,     0
    vbsll.v             vr0,    vr0,    6   //x x x x x x x 16 17 18 19 20 21 22 23 24 25 26 27 28 29
    vextrins.b          vr0,    vr21,   0x42
    vextrins.b          vr0,    vr21,   0x51
    vextrins.b          vr0,    vr20,   0x60 //insert to ref_pix[-1]
    vshuf4i.b           vr21,   vr21,   0x1b //0001 1011 reverse 4567 to 7654
    vextrins.w          vr0,    vr21,   0x01 //insert 7 6 5 4

    la.local            t0,     ang8_mode17
    vld                 vr20,   t0,     0   //shuf
    vld                 vr1,    t0,     16  //fraction

    vbsrl.v             vr21,   vr0,    6   //ref_pix[-1]
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2 //(32 - fraction) * ref[offset + x]

    vbsrl.v             vr21,   vr0,    5   //ref_pix[-2]
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vbsrl.v             vr21,   vr0,    4   //ref_pix[-3]
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr12,   vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vbsrl.v             vr21,   vr0,    3   //ref_pix[-4]
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr14,   vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr13,   vr14,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr14,   vr11

    vbsrl.v             vr21,   vr0,    2   //ref_pix[-5]
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr12,   vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vbsrl.v             vr21,   vr0,    1   //ref_pix[-6]
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr18,   vr21,   vr21,    vr20
    vmulwev.h.bu.b      vr14,   vr18,   vr11

    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,     vr20 //ref_pix[-6]
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr18,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

    beqz                t1,     1f

    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr4, vr7, vr10, vr13

    vstelm.d            vr4,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr4,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr7,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr10,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr13,   a0,     0,    1
    b                   2f
1:
    vstelm.d            vr5,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    0
    add.d               a0,     a0,     a1
    vstelm.d            vr5,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr9,    a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr14,   a0,     0,    1
    add.d               a0,     a0,     a1
    vstelm.d            vr17,   a0,     0,    1
2:
endfunc

const lasx_ang8_mode17, align=5
.byte 6, 7, 5, 6, 4, 5, 3, 4, 2, 3, 2, 3, 1, 2, 0, 1, 7, 8, 6, 7, 5, 6, 4, 5, 3, 4, 3, 4, 2, 3, 1, 2
.byte 32-6, 6, 32-12, 12, 32-18, 18, 32-24, 24, 32-30, 30, 32-4, 4, 32-10, 10, 32-16, 16
.byte 7, 6, 5, 4, 2, 1, 0, 16, 17, 18, 19, 20, 21, 22, 23, 24
endconst
function x265_intra_pred_ang8_17_lasx
    vld                 vr0,    a2,     17 //17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    vld                 vr1,    a2,     0
    la.local            t0,     lasx_ang8_mode17
    vld                 vr2,    t0,     48
    vshuf.b             vr0,    vr0,    vr1,   vr2
    xvld                xr1,    t0,     32 //fraction
    xvld                xr2,    t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00
    xvpermi.q           xr1,    xr1,    0x00

    xvshuf.b            xr3,    xr0,    xr0,    xr2 //row 0-1
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr5,    xr0,    xr0,    xr2 //row 2-3
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr6,    xr0,    xr0,    xr2 //row 4-5
    xvaddi.bu           xr2,    xr2,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr2 //row 6-7

    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvmulwev.h.bu.b     xr7,    xr5,    xr1
    xvmulwev.h.bu.b     xr10,   xr6,    xr1
    xvmulwev.h.bu.b     xr13,   xr9,    xr1

    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr7,    xr5,    xr1
    xvmaddwod.h.bu.b    xr10,   xr6,    xr1
    xvmaddwod.h.bu.b    xr13,   xr9,    xr1

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

const lasx_ang8_mode19, align=5
.byte 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-6, 6, 32-18, 18, 32-30, 30, 32-10, 10, 0,0,0,0,0,0,0,0, 32-12, 12, 32-24, 24, 32-4, 4, 32-16, 16, 0,0,0,0,0,0,0,0
.byte 7, 6, 5, 4, 2, 1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
endconst
function x265_intra_pred_ang8_19_lasx
    vld                 vr0,    a2,     0
    vld                 vr1,    a2,     16
    la.local            t0,     lasx_ang8_mode19
    vld                 vr2,    t0,     64
    vshuf.b             vr0,    vr0,    vr1,    vr2
    xvld                xr1,    t0,     32 //fraction
    xvld                xr20,   t0,     0  //shuf
    xvpermi.q           xr0,    xr0,    0x00

    xvaddi.bu           xr14,   xr20,   5 //offset = -1-2
    xvrepl128vei.h      xr2,    xr1,    0
    xvshuf.b            xr3,    xr0,    xr0,    xr14
    xvmulwev.h.bu.b     xr4,    xr3,    xr2

    xvaddi.bu           xr14,   xr20,   3 //offset = -3-4
    xvrepl128vei.h      xr5,    xr1,    1
    xvshuf.b            xr6,    xr0,    xr0,    xr14
    xvmulwev.h.bu.b     xr7,    xr6,    xr5

    xvaddi.bu           xr14,   xr20,   2 //offset = -4-5
    xvrepl128vei.h      xr8,    xr1,    2
    xvshuf.b            xr9,    xr0,    xr0,    xr14
    xvpermi.q           xr9,    xr9,    0x11
    xvmulwev.h.bu.b     xr10,   xr9,    xr8

    xvrepl128vei.h      xr11,   xr1,    3
    xvshuf.b            xr12,   xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr13,   xr12,   xr11

    xvmaddwod.h.bu.b    xr4,    xr3,    xr2 //fraction * ref[offset + x + 1]
    xvmaddwod.h.bu.b    xr7,    xr6,    xr5
    xvmaddwod.h.bu.b    xr10,   xr9,    xr8
    xvmaddwod.h.bu.b    xr13,   xr12,   xr11

    xvssrarni.bu.h      xr7,    xr4,    5 //+ 16) >> 5)
    xvssrarni.bu.h      xr13,   xr10,   5
    ANG8_STORE8x8_B
endfunc

function x265_intra_pred_ang8_18_lsx
    vld                 vr0,    a2,     0   //0-15
    vld                 vr21,   a2,     16
    vextrins.b          vr21,   vr0,    0x00 //insert ref_pix[-1]
    vshuf4i.b           vr21,   vr21,   0x1b //0001 1011 reverse 4567 to 7654
    vbsll.v             vr0,    vr0,    7
    vextrins.w          vr0,    vr21,   0x01 //insert 7 6 5 4
    vextrins.w          vr0,    vr21,   0x10 //insert 0 1 2 3

    vbsrl.v             vr1,    vr0,    7   //ref_pix[-1]
    vbsrl.v             vr2,    vr0,    6   //ref_pix[-2]
    vbsrl.v             vr3,    vr0,    5   //ref_pix[-3]
    vbsrl.v             vr4,    vr0,    4   //ref_pix[-4]
    vbsrl.v             vr5,    vr0,    3   //ref_pix[-5]
    vbsrl.v             vr6,    vr0,    2   //ref_pix[-6]
    vbsrl.v             vr7,    vr0,    1   //ref_pix[-7]

    fst.d               f1,     a0,     0
    fstx.d              f2,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    fst.d               f3,     a0,     0
    fstx.d              f4,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    fst.d               f5,     a0,     0
    fstx.d              f6,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    fst.d               f7,     a0,     0
    fstx.d              f0,     a0,     a1
endfunc

function x265_intra_pred_ang16_2_lsx
    li.w                t0,     32
    addi.w              t1,     a3,     -34
    maskeqz             t2,     t0,     t1
    add.d               t0,     t2,     a2
    addi.d              t0,     t0,     2

    vld                 vr0,    t0,     0
    vld                 vr1,    t0,     16

    vst                 vr0,    a0,     0
    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xf0
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xf1
    vst                 vr0,    a0,     0
    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xf2
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xf3
    vst                 vr0,    a0,     0
    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xf4
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xf5
    vst                 vr0,    a0,     0
    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xf6
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xf7
    vst                 vr0,    a0,     0
    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xf8
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xf9
    vst                 vr0,    a0,     0
    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xfa
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xfb
    vst                 vr0,    a0,     0
    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xfc
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xfd
    vst                 vr0,    a0,     0
    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xfe
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1
endfunc

.macro STORE8x8_B
    vstelm.d            vr5,    t3,     0,     0
    vstelmx.d           vr9,    t3,     a1,    0
    add.d               t3,     t3,     a1
    vstelm.d            vr14,   t3,     0,     0
    vstelmx.d           vr17,   t3,     a1,    0
    add.d               t3,     t3,     a1
    vstelm.d            vr5,    t3,     0,     1
    vstelmx.d           vr9,    t3,     a1,    1
    add.d               t3,     t3,     a1
    vstelm.d            vr14,   t3,     0,     1
    vstelmx.d           vr17,   t3,     a1,    1
.endm

.macro STORE8x16_B_TRANSPOSE
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr10, vr11, vr12, vr13

    vpackev.d           vr3,    vr10,   vr18
    vpackev.d           vr4,    vr11,   vr19
    vpackev.d           vr5,    vr12,   vr21
    vpackev.d           vr6,    vr13,   vr22
    vpackod.d           vr7,    vr10,   vr18
    vpackod.d           vr8,    vr11,   vr19
    vpackod.d           vr9,    vr12,   vr21
    vpackod.d           vr10,   vr13,   vr22

    vst                 vr3,    a0,     0
    vstx                vr7,    a0,     a1
    alsl.d              a0,     a1,     a0,    1
    vst                 vr4,    a0,     0
    vstx                vr8,    a0,     a1
    alsl.d              a0,     a1,     a0,    1
    vst                 vr5,    a0,     0
    vstx                vr9,    a0,     a1
    alsl.d              a0,     a1,     a0,    1
    vst                 vr6,    a0,     0
    vstx                vr10,   a0,     a1
    alsl.d              a0,     a1,     a0,    1
.endm

const ang16_mode3, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-26, 26, 32-20, 20, 32-14, 14, 32-8, 8, 32-2, 2, 32-28, 28, 32-22, 22, 32-16, 16
.byte 32-10, 10, 32-4, 4, 32-30, 30, 32-24, 24, 32-18, 18, 32-12, 12, 32-6, 6, 32-0, 0
endconst
.macro INTRA_PRED_ANG16_3 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 3
    addi.d              t0,     a2,     33
.else
    addi.d              t0,     a2,     1
.endif
    la.local            t1,     ang16_mode3
    vld                 vr20,   t1,     0  //shuf
    li.w                t2,     2
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction
1:
    vld                 vr0,    t0,     0  //offset = 0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 1
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vbsrl.v             vr0,    vr0,    1  //offset = 2
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr9,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 3
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 4
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 5
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 6
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5
.if \mode == 3
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vld                 vr0,    t0,     7  //offset = 7

    vreplvei.h          vr2,    vr23,   0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 8
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr0,    vr0,    1 //offset = 9
    vreplvei.h          vr11,   vr23,   3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 10
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 11
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 12
    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 13

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13
.if \mode == 3
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8
.endif
    addi.d              t2,     t2,     -1
    addi.d              t0,     t0,     8
    bnez                t2,     1b
endfunc
.endm

INTRA_PRED_ANG16_3 3
INTRA_PRED_ANG16_3 33

.macro ANG16_MODE3_PROCESS_4x16_B idx0, idx1, idx2
    vld                 vr0,    a2,     \idx0
    xvpermi.q           xr0,    xr0,    0x00
    xvshuf.b            xr3,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvshuf.b            xr5,    xr0,    xr0,    xr21
    xvmulwev.h.bu.b     xr6,    xr5,    xr1
    vld                 vr0,    a2,     \idx1
    xvpermi.q           xr0,    xr0,    0x00
    xvshuf.b            xr7,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr8,    xr7,    xr1
    xvshuf.b            xr9,    xr0,    xr0,    xr21
    xvmulwev.h.bu.b     xr10,   xr9,    xr1
    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr6,    xr5,    xr1
    xvmaddwod.h.bu.b    xr8,    xr7,    xr1
    xvmaddwod.h.bu.b    xr10,   xr9,    xr1
    xvssrarni.bu.h      xr6,    xr4,    5
    xvssrarni.bu.h      xr10,   xr8,    5
    xvpermi.d           xr4,    xr6,    0xd8
    xvpermi.d           xr8,    xr10,   0xd8
    xvpermi.q           xr6,    xr4,    0x11
    xvpermi.q           xr10,   xr8,    0x11
    vst                 vr4,    a0,     0
    vstx                vr6,    a0,     a1
    alsl.d              a0,     a1,     a0,    1
    vst                 vr8,    a0,     0
    vstx                vr10,   a0,     a1
.if \idx2 == 1
    alsl.d              a0,     a1,     a0,    1
.endif
.endm
const lasx_ang16_mode3, align=5
.byte 0,1,1,2,2,3,3,4,4,5,4,5,5,6,6,7,7,8,8,9,8,9,9,10,10,11,11,12,12,13,13,14
.byte 32-26, 26, 32-20, 20, 32-14, 14, 32-8, 8, 32-2, 2, 32-28, 28, 32-22, 22, 32-16, 16
.byte 32-10, 10, 32-4, 4, 32-30, 30, 32-24, 24, 32-18, 18, 32-12, 12, 32-6, 6, 32-0, 0
endconst
function x265_intra_pred_ang16_3_lasx
    la.local            t1,     lasx_ang16_mode3
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1
    ANG16_MODE3_PROCESS_4x16_B  33+0,   33+2,  1
    ANG16_MODE3_PROCESS_4x16_B  33+4,   33+6,  1
    ANG16_MODE3_PROCESS_4x16_B  33+8,   33+10, 1
    ANG16_MODE3_PROCESS_4x16_B  33+12,  33+14, 0
endfunc

const ang16_mode4, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-21, 21, 32-10, 10, 32-31, 31, 32-20, 20, 32-9, 9, 32-30, 30, 32-19, 19, 32-8, 8
.byte 32-29, 29, 32-18, 18, 32-7, 7, 32-28, 28, 32-17, 17, 32-6, 6, 32-27, 27, 32-16, 16
endconst
.macro INTRA_PRED_ANG16_4 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 4
    addi.d              t0,     a2,     33
.else
    addi.d              t0,     a2,     1
.endif
    la.local            t1,     ang16_mode4
    vld                 vr20,   t1,     0  //shuf
    li.w                t2,     2
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction
1:
    vld                 vr0,    t0,     0  //offset = 0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 1
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 2
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 3
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 4
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 5
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 4
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 6
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vbsrl.v             vr0,    vr0,    1  //offset = 7
    vreplvei.h          vr8,    vr23,   2
    vshuf.b             vr9,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr9,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr9,    vr11

    vld                 vr0,    t0,     8 //offset = 8
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 9
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 10
    vreplvei.h          vr15,   vr23,   7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 4
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8
.endif
    addi.d              t2,     t2,     -1
    addi.d              t0,     t0,     8
    bnez                t2,     1b
endfunc
.endm

INTRA_PRED_ANG16_4 4
INTRA_PRED_ANG16_4 32

.macro ANG16_MODE4_PROCESS_4x16_B idx0, idx1
    vld                 vr0,    a2,     \idx0
    xvpermi.q           xr0,    xr0,    0x00
    xvshuf.b            xr3,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvshuf.b            xr5,    xr0,    xr0,    xr21
    xvmulwev.h.bu.b     xr6,    xr5,    xr1
    xvshuf.b            xr7,    xr0,    xr0,    xr22
    xvmulwev.h.bu.b     xr8,    xr7,    xr1
    xvshuf.b            xr9,    xr0,    xr0,    xr23
    xvmulwev.h.bu.b     xr10,   xr9,    xr1
    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr6,    xr5,    xr1
    xvmaddwod.h.bu.b    xr8,    xr7,    xr1
    xvmaddwod.h.bu.b    xr10,   xr9,    xr1
    xvssrarni.bu.h      xr6,    xr4,    5
    xvssrarni.bu.h      xr10,   xr8,    5
    xvpermi.d           xr4,    xr6,    0xd8
    xvpermi.d           xr8,    xr10,   0xd8
    xvpermi.q           xr6,    xr4,    0x11
    xvpermi.q           xr10,   xr8,    0x11
    vst                 vr4,    a0,     0
    vstx                vr6,    a0,     a1
    alsl.d              a0,     a1,     a0,    1
    vst                 vr8,    a0,     0
    vstx                vr10,   a0,     a1
.if \idx1 == 1
    alsl.d              a0,     a1,     a0,    1
.endif
.endm
const lasx_ang16_mode4, align=5
.byte 0,1,1,2,1,2,2,3,3,4,3,4,4,5,5,6,5,6,6,7,7,8,7,8,8,9,9,10,9,10,10,11
.byte 32-21, 21, 32-10, 10, 32-31, 31, 32-20, 20, 32-9, 9, 32-30, 30, 32-19, 19, 32-8, 8
.byte 32-29, 29, 32-18, 18, 32-7, 7, 32-28, 28, 32-17, 17, 32-6, 6, 32-27, 27, 32-16, 16
endconst
function x265_intra_pred_ang16_4_lasx
    la.local            t1,     lasx_ang16_mode4
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    xvaddi.bu           xr22,   xr20,   2  //shuf
    xvaddi.bu           xr23,   xr20,   3  //shuf
    ANG16_MODE4_PROCESS_4x16_B  33+0,   1
    ANG16_MODE4_PROCESS_4x16_B  33+4,   1
    ANG16_MODE4_PROCESS_4x16_B  33+8,   1
    ANG16_MODE4_PROCESS_4x16_B  33+12,  0
endfunc

const ang16_mode5, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-17, 17, 32-2, 2, 32-19, 19, 32-4, 4, 32-21, 21, 32-6, 6, 32-23, 23, 32-8, 8
.byte 32-25, 25, 32-10, 10, 32-27, 27, 32-12, 12, 32-29, 29, 32-14, 14, 32-31, 31, 32-16, 16
endconst
.macro INTRA_PRED_ANG16_5 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 5
    addi.d              t0,     a2,     33
.else
    addi.d              t0,     a2,     1
.endif
    la.local            t1,     ang16_mode5
    vld                 vr20,   t1,     0  //shuf
    li.w                t2,     2
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction
1:
    vld                 vr0,    t0,     0  //offset = 0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 1
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 2
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 3
    vreplvei.h          vr6,    vr1,    5
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 4
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 5
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 5
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 6
    vreplvei.h          vr11,   vr23,   3
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 7
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vld                 vr0,    t0,     8  //offset = 8
    vreplvei.h          vr15,   vr23,   7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 5
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8
.endif
    addi.d              t2,     t2,     -1
    addi.d              t0,     t0,     8
    bnez                t2,     1b
endfunc
.endm

INTRA_PRED_ANG16_5 5
INTRA_PRED_ANG16_5 31

const lasx_ang16_mode5, align=5
.byte 0,1,1,2,1,2,2,3,2,3,3,4,3,4,4,5,4,5,5,6,5,6,6,7,6,7,7,8,7,8,8,9
.byte 32-17, 17, 32-2, 2, 32-19, 19, 32-4, 4, 32-21, 21, 32-6, 6, 32-23, 23, 32-8, 8
.byte 32-25, 25, 32-10, 10, 32-27, 27, 32-12, 12, 32-29, 29, 32-14, 14, 32-31, 31, 32-16, 16
endconst
function x265_intra_pred_ang16_5_lasx
    la.local            t1,     lasx_ang16_mode5
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    xvaddi.bu           xr22,   xr20,   2  //shuf
    xvaddi.bu           xr23,   xr20,   3  //shuf
    ANG16_MODE4_PROCESS_4x16_B  33+0,   1
    ANG16_MODE4_PROCESS_4x16_B  33+4,   1
    ANG16_MODE4_PROCESS_4x16_B  33+8,   1
    ANG16_MODE4_PROCESS_4x16_B  33+12,  0
endfunc

const ang16_mode6, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-13, 13, 32-26, 26, 32-7, 7, 32-20, 20, 32-1, 1, 32-14, 14, 32-27, 27, 32-8, 8
.byte 32-21, 21, 32-2, 2, 32-15, 15, 32-28, 28, 32-9, 9, 32-22, 22, 32-3, 3, 32-16, 16
endconst
.macro INTRA_PRED_ANG16_6 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 6
    addi.d              t0,     a2,     33
.else
    addi.d              t0,     a2,     1
.endif
    la.local            t1,     ang16_mode6
    vld                 vr20,   t1,     0  //shuf
    li.w                t2,     2
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction
1:
    vld                 vr0,    t0,     0  //offset = 0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vbsrl.v             vr0,    vr0,    1  //offset = 1
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr6,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr6,    vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 2
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 3
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 6
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 4
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr6,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr6,    vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 5
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 6
    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr15,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 6
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8
.endif
    addi.d              t2,     t2,     -1
    addi.d              t0,     t0,     8
    bnez                t2,     1b
endfunc
.endm

INTRA_PRED_ANG16_6 6
INTRA_PRED_ANG16_6 30

.macro ANG16_MODE6_PROCESS_4x16_B idx0, idx1
.if \idx0 > 0
    vld                 vr0,    a2,     \idx0
.else
    vbsrl.v             vr0,    vr0,    4
.endif
    xvpermi.q           xr0,    xr0,    0x00
    xvshuf.b            xr3,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvshuf.b            xr5,    xr0,    xr0,    xr21
    xvmulwev.h.bu.b     xr6,    xr5,    xr1
    xvshuf.b            xr7,    xr0,    xr0,    xr22
    xvmulwev.h.bu.b     xr8,    xr7,    xr1
    xvshuf.b            xr9,    xr0,    xr0,    xr23
    xvmulwev.h.bu.b     xr10,   xr9,    xr1
    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr6,    xr5,    xr1
    xvmaddwod.h.bu.b    xr8,    xr7,    xr1
    xvmaddwod.h.bu.b    xr10,   xr9,    xr1
    xvssrarni.bu.h      xr6,    xr4,    5
    xvssrarni.bu.h      xr10,   xr8,    5
    xvpermi.d           xr4,    xr6,    0xd8
    xvpermi.d           xr8,    xr10,   0xd8
    xvpermi.q           xr6,    xr4,    0x11
    xvpermi.q           xr10,   xr8,    0x11
    vst                 vr4,    a0,     0
    vstx                vr6,    a0,     a1
    alsl.d              a0,     a1,     a0,    1
    vst                 vr8,    a0,     0
    vstx                vr10,   a0,     a1
.if \idx1 == 1
    alsl.d              a0,     a1,     a0,    1
.endif
.endm
const lasx_ang16_mode6, align=5
.byte 0,1,0,1,1,2,1,2,2,3,2,3,2,3,3,4,3,4,4,5,4,5,4,5,5,6,5,6,6,7,6,7
.byte 32-13, 13, 32-26, 26, 32-7, 7, 32-20, 20, 32-1, 1, 32-14, 14, 32-27, 27, 32-8, 8
.byte 32-21, 21, 32-2, 2, 32-15, 15, 32-28, 28, 32-9, 9, 32-22, 22, 32-3, 3, 32-16, 16
endconst
function x265_intra_pred_ang16_6_lasx
    la.local            t1,     lasx_ang16_mode6
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    xvaddi.bu           xr22,   xr20,   2  //shuf
    xvaddi.bu           xr23,   xr20,   3  //shuf
    ANG16_MODE6_PROCESS_4x16_B  33+0,   1
    ANG16_MODE6_PROCESS_4x16_B  0   ,   1
    ANG16_MODE6_PROCESS_4x16_B  33+8,   1
    ANG16_MODE6_PROCESS_4x16_B  0   ,   0
endfunc

const ang16_mode7, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-9, 9, 32-18, 18, 32-27, 27, 32-4, 4, 32-13, 13, 32-22, 22, 32-31, 31, 32-8, 8
.byte 32-17, 17, 32-26, 26, 32-3, 3, 32-12, 12, 32-21, 21, 32-30, 30, 32-7, 7, 32-16, 16
endconst
.macro INTRA_PRED_ANG16_7 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 7
    addi.d              t0,     a2,     33
.else
    addi.d              t0,     a2,     1
.endif
    la.local            t1,     ang16_mode7
    vld                 vr20,   t1,     0  //shuf
    li.w                t2,     2
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction
1:
    vld                 vr0,    t0,     0  //offset = 0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 1
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 2
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 7
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vbsrl.v             vr0,    vr0,    1  //offset = 3
    vreplvei.h          vr8,    vr23,   2
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 4
    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr15,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 7
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8
.endif
    addi.d              t2,     t2,     -1
    addi.d              t0,     t0,     8
    bnez                t2,     1b
endfunc
.endm

INTRA_PRED_ANG16_7 7
INTRA_PRED_ANG16_7 29

const lasx_ang16_mode7, align=5
.byte 0,1,0,1,0,1,1,2,1,2,1,2,1,2,2,3,2,3,2,3,3,4,3,4,3,4,3,4,4,5,4,5
.byte 32-9, 9, 32-18, 18, 32-27, 27, 32-4, 4, 32-13, 13, 32-22, 22, 32-31, 31, 32-8, 8
.byte 32-17, 17, 32-26, 26, 32-3, 3, 32-12, 12, 32-21, 21, 32-30, 30, 32-7, 7, 32-16, 16
endconst
function x265_intra_pred_ang16_7_lasx
    la.local            t1,     lasx_ang16_mode7
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    xvaddi.bu           xr22,   xr20,   2  //shuf
    xvaddi.bu           xr23,   xr20,   3  //shuf
    ANG16_MODE6_PROCESS_4x16_B  33+0,   1
    ANG16_MODE6_PROCESS_4x16_B  0   ,   1
    ANG16_MODE6_PROCESS_4x16_B  33+8,   1
    ANG16_MODE6_PROCESS_4x16_B  0   ,   0
endfunc

const ang16_mode8, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-5, 5, 32-10, 10, 32-15, 15, 32-20, 20, 32-25, 25, 32-30, 30, 32-3, 3, 32-8, 8
.byte 32-13, 13, 32-18, 18, 32-23, 23, 32-28, 28, 32-1, 1, 32-6, 6, 32-11, 11, 32-16, 16
endconst
.macro INTRA_PRED_ANG16_8 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 8
    addi.d              t0,     a2,     33
.else
    addi.d              t0,     a2,     1
.endif
    la.local            t1,     ang16_mode8
    vld                 vr20,   t1,     0  //shuf
    li.w                t2,     2
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction
1:
    vld                 vr0,    t0,     0  //offset = 0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1  //offset = 1
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr8,    vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr8,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 8
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr8,    vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr8,    vr5

    vreplvei.h          vr3,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr8,    vr3

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr8,    vr11

    vmaddwod.h.bu.b     vr4,    vr8,    vr2
    vmaddwod.h.bu.b     vr7,    vr8,    vr5
    vmaddwod.h.bu.b     vr10,   vr8,    vr3
    vmaddwod.h.bu.b     vr13,   vr8,    vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 2
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr15,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 8
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8
.endif
    addi.d              t2,     t2,     -1
    addi.d              t0,     t0,     8
    bnez                t2,     1b
endfunc
.endm

INTRA_PRED_ANG16_8 8
INTRA_PRED_ANG16_8 28

const lasx_ang16_mode8, align=5
.byte 0,1,0,1,0,1,0,1,0,1,0,1,1,2,1,2,1,2,1,2,1,2,1,2,2,3,2,3,2,3,2,3
.byte 32-5, 5, 32-10, 10, 32-15, 15, 32-20, 20, 32-25, 25, 32-30, 30, 32-3, 3, 32-8, 8
.byte 32-13, 13, 32-18, 18, 32-23, 23, 32-28, 28, 32-1, 1, 32-6, 6, 32-11, 11, 32-16, 16
endconst
function x265_intra_pred_ang16_8_lasx
    la.local            t1,     lasx_ang16_mode8
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    xvaddi.bu           xr22,   xr20,   2  //shuf
    xvaddi.bu           xr23,   xr20,   3  //shuf
    ANG16_MODE6_PROCESS_4x16_B  33+0,   1
    ANG16_MODE6_PROCESS_4x16_B  0   ,   1
    ANG16_MODE6_PROCESS_4x16_B  33+8,   1
    ANG16_MODE6_PROCESS_4x16_B  0   ,   0
endfunc

const ang16_mode9, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-2, 2, 32-4, 4, 32-6, 6, 32-8, 8, 32-10, 10, 32-12, 12, 32-14, 14, 32-16, 16
.byte 32-18, 18, 32-20, 20, 32-22, 22, 32-24, 24, 32-26, 26, 32-28, 28, 32-30, 30, 32-0, 0
endconst
.macro INTRA_PRED_ANG16_9 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 9
    addi.d              t0,     a2,     33
.else
    addi.d              t0,     a2,     1
.endif
    la.local            t1,     ang16_mode9
    vld                 vr20,   t1,     0  //shuf
    li.w                t2,     2
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction
1:
    vld                 vr0,    t0,     0  //offset = 0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 9
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr6,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr3,    vr6

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr6
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 1

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 9
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8
.endif
    addi.d              t2,     t2,     -1
    addi.d              t0,     t0,     8
    bnez                t2,     1b
endfunc
.endm

INTRA_PRED_ANG16_9 9
INTRA_PRED_ANG16_9 27

const lasx_ang16_mode9, align=5
.byte 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,2
.byte 32-2, 2, 32-4, 4, 32-6, 6, 32-8, 8, 32-10, 10, 32-12, 12, 32-14, 14, 32-16, 16
.byte 32-18, 18, 32-20, 20, 32-22, 22, 32-24, 24, 32-26, 26, 32-28, 28, 32-30, 30, 32-0, 0
endconst
function x265_intra_pred_ang16_9_lasx
    la.local            t1,     lasx_ang16_mode9
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    xvaddi.bu           xr22,   xr20,   2  //shuf
    xvaddi.bu           xr23,   xr20,   3  //shuf
    ANG16_MODE6_PROCESS_4x16_B  33+0,   1
    ANG16_MODE6_PROCESS_4x16_B  0   ,   1
    ANG16_MODE6_PROCESS_4x16_B  33+8,   1
    ANG16_MODE6_PROCESS_4x16_B  0   ,   0
endfunc

function x265_intra_pred_ang16_10_lsx
    vld                 vr0,    a2,     33
    //transpose
    vreplvei.b          vr1,    vr0,    0
    vreplvei.b          vr2,    vr0,    1
    vreplvei.b          vr3,    vr0,    2
    vreplvei.b          vr4,    vr0,    3
    vreplvei.b          vr5,    vr0,    4
    vreplvei.b          vr6,    vr0,    5
    vreplvei.b          vr7,    vr0,    6
    vreplvei.b          vr8,    vr0,    7
    vreplvei.b          vr9,    vr0,    8
    vreplvei.b          vr10,   vr0,    9
    vreplvei.b          vr11,   vr0,    10
    vreplvei.b          vr12,   vr0,    11
    vreplvei.b          vr13,   vr0,    12
    vreplvei.b          vr14,   vr0,    13
    vreplvei.b          vr15,   vr0,    14
    vreplvei.b          vr16,   vr0,    15

    beqz                a4,     1f

    //bFilter != 0
    vldrepl.b           vr18,   a2,     0
    vsllwil.hu.bu       vr17,   vr1,    0 //top
    vsllwil.hu.bu       vr18,   vr18,   0 //topLeft
    vld                 vr19,   a2,     1 //srcPixel[width2 + 1 + y]
    vexth.hu.bu         vr20,   vr19
    vsllwil.hu.bu       vr19,   vr19,   0
    vsub.h              vr19,   vr19,   vr18
    vsub.h              vr20,   vr20,   vr18
    vsrai.h             vr19,   vr19,   1
    vsrai.h             vr20,   vr20,   1
    vadd.h              vr19,   vr19,   vr17
    vadd.h              vr20,   vr20,   vr17
    vssrani.bu.h        vr20,   vr19,   0
    vmov                vr1,    vr20
1:
    vst                 vr1,    a0,     0
    vstx                vr2,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr3,    a0,     0
    vstx                vr4,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr5,    a0,     0
    vstx                vr6,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr7,    a0,     0
    vstx                vr8,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr9,    a0,     0
    vstx                vr10,   a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr11,   a0,     0
    vstx                vr12,   a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr13,   a0,     0
    vstx                vr14,   a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr15,   a0,     0
    vstx                vr16,   a0,     a1
endfunc

.macro STORE_2x16 idx0, idx1
    vextrins.b          vr0,    vr20,   \idx0
    vst                 vr0,    a0,     0
    vextrins.b          vr0,    vr20,   \idx1
    vstx                vr0,    a0,     a1
.if \idx1 < 0x0f
    alsl.d              a0,     a1,     a0,   1
.endif
.endm

function x265_intra_pred_ang16_26_lsx
    vld                 vr0,    a2,     1

    beqz                a4,     1f

    //bFilter != 0
    vldrepl.h           vr18,   a2,     0
    vsllwil.hu.bu       vr18,   vr18,   0
    vreplvei.h          vr17,   vr18,   1 //top
    vreplvei.h          vr18,   vr18,   0 //topLeft
    vld                 vr19,   a2,     33//srcPixel[width2 + 1 + y]
    vexth.hu.bu         vr20,   vr19
    vsllwil.hu.bu       vr19,   vr19,   0
    vsub.h              vr19,   vr19,   vr18
    vsub.h              vr20,   vr20,   vr18
    vsrai.h             vr19,   vr19,   1
    vsrai.h             vr20,   vr20,   1
    vadd.h              vr19,   vr19,   vr17
    vadd.h              vr20,   vr20,   vr17
    vssrani.bu.h        vr20,   vr19,   0

    STORE_2x16          0x00,   0x01
    STORE_2x16          0x02,   0x03
    STORE_2x16          0x04,   0x05
    STORE_2x16          0x06,   0x07
    STORE_2x16          0x08,   0x09
    STORE_2x16          0x0a,   0x0b
    STORE_2x16          0x0c,   0x0d
    STORE_2x16          0x0e,   0x0f
    b                   2f
1:
    vst                 vr0,    a0,     0
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr0,    a0,     0
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr0,    a0,     0
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr0,    a0,     0
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr0,    a0,     0
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr0,    a0,     0
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr0,    a0,     0
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr0,    a0,     0
    vstx                vr0,    a0,     a1
2:
endfunc

const ang16_mode11, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-30, 30, 32-28, 28, 32-26, 26, 32-24, 24, 32-22, 22, 32-20, 20, 32-18, 18, 32-16, 16
.byte 32-14, 14, 32-12, 12, 32-10, 10, 32-8, 8, 32-6, 6, 32-4, 4, 32-2, 2, 32-0, 0
endconst
.macro INTRA_PRED_ANG16_11 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 11
    addi.d              t0,     a2,     32
.else
    addi.d              t0,     a2,     0
.endif
    la.local            t1,     ang16_mode11
    vld                 vr20,   t1,     0  //shuf
    li.w                t2,     2
    li.w                t4,     2
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction
1:
    vld                 vr0,    t0,     0  //offset = -1

    blt                 t2,     t4,     2f //second loop dont need insert
    vldrepl.b           vr2,    a2,     0
    vextrins.b          vr0,    vr2,    0x00 //insert ref_pix[-1]
2:
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 11
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr6,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr3,    vr6

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr6
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13 //fraction=0

.if \mode == 11
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8
.endif
    addi.d              t2,     t2,     -1
    addi.d              t0,     t0,     8
    bnez                t2,     1b
endfunc
.endm

INTRA_PRED_ANG16_11 11
INTRA_PRED_ANG16_11 25

.macro ANG16_MODE11_PROCESS_4x16_B idx0
    xvpermi.q           xr0,    xr0,    0x00
    xvshuf.b            xr3,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvshuf.b            xr5,    xr0,    xr0,    xr21
    xvmulwev.h.bu.b     xr6,    xr5,    xr1
    xvshuf.b            xr7,    xr0,    xr0,    xr22
    xvmulwev.h.bu.b     xr8,    xr7,    xr1
    xvshuf.b            xr9,    xr0,    xr0,    xr23
    xvmulwev.h.bu.b     xr10,   xr9,    xr1
    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr6,    xr5,    xr1
    xvmaddwod.h.bu.b    xr8,    xr7,    xr1
    xvmaddwod.h.bu.b    xr10,   xr9,    xr1
    xvssrarni.bu.h      xr6,    xr4,    5
    xvssrarni.bu.h      xr10,   xr8,    5
    xvpermi.d           xr4,    xr6,    0xd8
    xvpermi.d           xr8,    xr10,   0xd8
    xvpermi.q           xr6,    xr4,    0x11
    xvpermi.q           xr10,   xr8,    0x11
    vst                 vr4,    a0,     0
    vstx                vr6,    a0,     a1
    alsl.d              a0,     a1,     a0,    1
    vst                 vr8,    a0,     0
    vstx                vr10,   a0,     a1
.if \idx0 == 1
    alsl.d              a0,     a1,     a0,    1
.endif
.endm
const lasx_ang16_mode11, align=5
.byte 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1
.byte 32-30, 30, 32-28, 28, 32-26, 26, 32-24, 24, 32-22, 22, 32-20, 20, 32-18, 18, 32-16, 16
.byte 32-14, 14, 32-12, 12, 32-10, 10, 32-8, 8, 32-6, 6, 32-4, 4, 32-2, 2, 32-0, 0
endconst
function x265_intra_pred_ang16_11_lasx
    la.local            t1,     lasx_ang16_mode11
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    xvaddi.bu           xr22,   xr20,   2  //shuf
    xvaddi.bu           xr23,   xr20,   3  //shuf
    vld                 vr0,    a2,     32
    vldrepl.b           vr11,   a2,     0
    vextrins.b          vr0,    vr11,   0x00
    ANG16_MODE11_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr0,    4
    ANG16_MODE11_PROCESS_4x16_B 1
    vld                 vr0,    a2,     32+8
    ANG16_MODE11_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr0,    4
    ANG16_MODE11_PROCESS_4x16_B 0
endfunc

const ang16_mode12, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-27, 27, 32-22, 22, 32-17, 17, 32-12, 12, 32-7, 7, 32-2, 2, 32-29, 29, 32-24, 24
.byte 32-19, 19, 32-14, 14, 32-9, 9, 32-4, 4, 32-31, 31, 32-26, 26, 32-21, 21, 32-16, 16
endconst
.macro INTRA_PRED_ANG16_12 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 12
    addi.d              t0,     a2,     32
.else
    addi.d              t0,     a2,     0
.endif
    la.local            t1,     ang16_mode12
    vld                 vr20,   t1,     0  //shuf
    li.w                t2,     2
    li.w                t4,     2
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction
1:
    vld                 vr0,    t0,     0

    blt                 t2,     t4,     2f //second loop dont need insert
.if \mode == 12
    vld                 vr2,    a2,     0
.else
    vld                 vr2,    a2,     32
.endif
    vldrepl.b           vr3,    a2,     0
    vbsll.v             vr0,    vr0,    2
    vextrins.b          vr0,    vr2,    0x0d //insert ref_pix[-3]
    vextrins.b          vr0,    vr2,    0x16 //insert ref_pix[-2]
    vextrins.b          vr0,    vr3,    0x20 //insert ref_pix[-1]
2:
    vbsrl.v             vr12,   vr0,    2  //offset = -1
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr12,   vr12,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr12,   vr0,    1  //offset = -2
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr8,    vr12,   vr12,    vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr8,    vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr8,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 12
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr8,    vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr8,    vr5

    vreplvei.h          vr6,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr8,    vr6

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr8,    vr11

    vmaddwod.h.bu.b     vr4,    vr8,    vr2
    vmaddwod.h.bu.b     vr7,    vr8,    vr5
    vmaddwod.h.bu.b     vr10,   vr8,    vr6
    vmaddwod.h.bu.b     vr13,   vr8,    vr11

    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr12,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr3,    vr12

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr12

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 12
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8
.endif
    addi.d              t2,     t2,     -1
    addi.d              t0,     t0,     6  //src offset for next loop
    bnez                t2,     1b
endfunc
.endm

INTRA_PRED_ANG16_12 12
INTRA_PRED_ANG16_12 24

const lasx_ang16_mode12, align=5
.byte 2,3,2,3,2,3,2,3,2,3,2,3,1,2,1,2,1,2,1,2,1,2,1,2,0,1,0,1,0,1,0,1
.byte 32-27, 27, 32-22, 22, 32-17, 17, 32-12, 12, 32-7, 7, 32-2, 2, 32-29, 29, 32-24, 24
.byte 32-19, 19, 32-14, 14, 32-9, 9, 32-4, 4, 32-31, 31, 32-26, 26, 32-21, 21, 32-16, 16
.byte 13, 6, 0, 17, 18, 19, 20, 21, 22, 23, 24
endconst
function x265_intra_pred_ang16_12_lasx
    la.local            t1,     lasx_ang16_mode12
    vld                 vr12,   t1,     64
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    xvaddi.bu           xr22,   xr20,   2  //shuf
    xvaddi.bu           xr23,   xr20,   3  //shuf
    vld                 vr0,    a2,     32
    vld                 vr11,   a2,     0
    vshuf.b             vr0,    vr0,    vr11,   vr12 //Get the reference pixels
    ANG16_MODE11_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr0,    4
    ANG16_MODE11_PROCESS_4x16_B 1
    vld                 vr0,    a2,     32+6
    ANG16_MODE11_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr0,    4
    ANG16_MODE11_PROCESS_4x16_B 0
endfunc

const ang16_mode13, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-23, 23, 32-14, 14, 32-5, 5, 32-28, 28, 32-19, 19, 32-10, 10, 32-1, 1, 32-24, 24
.byte 32-15, 15, 32-6, 6, 32-29, 29, 32-20, 20, 32-11, 11, 32-2, 2, 32-25, 25, 32-16, 16
endconst
.macro INTRA_PRED_ANG16_13 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 13
    addi.d              t0,     a2,     32
.else
    addi.d              t0,     a2,     0
.endif
    la.local            t1,     ang16_mode13
    vld                 vr20,   t1,     0  //shuf
    li.w                t2,     2
    li.w                t4,     2
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction
1:
    vld                 vr0,    t0,     0

    blt                 t2,     t4,     2f //second loop dont need insert
.if \mode == 13
    vld                 vr2,    a2,     0
.else
    vld                 vr2,    a2,     32
.endif
    vldrepl.b           vr3,    a2,     0
    vbsll.v             vr0,    vr0,    4
    vextrins.b          vr0,    vr2,    0x0e //insert ref_pix[-5]
    vextrins.b          vr0,    vr2,    0x1b //insert ref_pix[-4]
    vextrins.b          vr0,    vr2,    0x27 //insert ref_pix[-3]
    vextrins.b          vr0,    vr2,    0x34 //insert ref_pix[-2]
    vextrins.b          vr0,    vr3,    0x40 //insert ref_pix[-1]
2:
    vbsrl.v             vr12,   vr0,    4  //offset = -1
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr12,   vr12,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vbsrl.v             vr12,   vr0,    3  //offset = -2
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr16,   vr12,   vr12,    vr20
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vbsrl.v             vr12,   vr0,    2  //offset = -3
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 13
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vbsrl.v             vr12,   vr0,    1  //offset = -4
    vreplvei.h          vr6,    vr23,   2
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr10,   vr8,    vr6

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr8,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr8,    vr6
    vmaddwod.h.bu.b     vr13,   vr8,    vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr8,    vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -5
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr12,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr3,    vr12

    vmaddwod.h.bu.b     vr5,    vr8,    vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr12

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 13
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8
.endif
    addi.d              t2,     t2,     -1
    addi.d              t0,     t0,     4  //src offset for next loop
    bnez                t2,     1b
endfunc
.endm

INTRA_PRED_ANG16_13 13
INTRA_PRED_ANG16_13 23

const lasx_ang16_mode13, align=5
.byte 4,5,4,5,4,5,3,4,3,4,3,4,3,4,2,3,2,3,2,3,1,2,1,2,1,2,1,2,0,1,0,1
.byte 32-23, 23, 32-14, 14, 32-5, 5, 32-28, 28, 32-19, 19, 32-10, 10, 32-1, 1, 32-24, 24
.byte 32-15, 15, 32-6, 6, 32-29, 29, 32-20, 20, 32-11, 11, 32-2, 2, 32-25, 25, 32-16, 16
.byte 14, 11, 7, 4, 0, 17, 18, 19, 20, 21, 22, 23, 24
endconst
function x265_intra_pred_ang16_13_lasx
    la.local            t1,     lasx_ang16_mode13
    vld                 vr12,   t1,     64
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    xvaddi.bu           xr22,   xr20,   2  //shuf
    xvaddi.bu           xr23,   xr20,   3  //shuf
    vld                 vr0,    a2,     32
    vld                 vr11,   a2,     0
    vshuf.b             vr0,    vr0,    vr11,   vr12 //Get the reference pixels
    ANG16_MODE11_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr0,    4
    ANG16_MODE11_PROCESS_4x16_B 1
    vld                 vr0,    a2,     32+4
    ANG16_MODE11_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr0,    4
    ANG16_MODE11_PROCESS_4x16_B 0
endfunc

const ang16_mode14, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-19, 19, 32-6, 6, 32-25, 25, 32-12, 12, 32-31, 31, 32-18, 18, 32-5, 5, 32-24, 24
.byte 32-11, 11, 32-30, 30, 32-17, 17, 32-4, 4, 32-23, 23, 32-10, 10, 32-29, 29, 32-16, 16
endconst
.macro INTRA_PRED_ANG16_14 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 14
    addi.d              t0,     a2,     32
.else
    addi.d              t0,     a2,     0
.endif
    la.local            t1,     ang16_mode14
    vld                 vr20,   t1,     0  //shuf
    li.w                t2,     2
    li.w                t4,     2
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction
1:
    vld                 vr0,    t0,     0

    blt                 t2,     t4,     2f //second loop dont need insert
.if \mode == 14
    vld                 vr2,    a2,     0
.else
    vld                 vr2,    a2,     32
.endif
    vldrepl.b           vr3,    a2,     0
    vbsll.v             vr0,    vr0,    6
    vextrins.b          vr0,    vr2,    0x0f //insert ref_pix[-7]
    vextrins.b          vr0,    vr2,    0x1c //insert ref_pix[-6]
    vextrins.b          vr0,    vr2,    0x2a //insert ref_pix[-5]
    vextrins.b          vr0,    vr2,    0x37 //insert ref_pix[-4]
    vextrins.b          vr0,    vr2,    0x45 //insert ref_pix[-3]
    vextrins.b          vr0,    vr2,    0x52 //insert ref_pix[-2]
    vextrins.b          vr0,    vr3,    0x60 //insert ref_pix[-1]
2:
    vbsrl.v             vr12,   vr0,    6  //offset = -1
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vbsrl.v             vr12,   vr0,    5  //offset = -2
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr6,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr6,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr6,    vr11

    vbsrl.v             vr3,    vr0,    4  //offset = -3
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr16,   vr3,    vr3,    vr20
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vbsrl.v             vr12,   vr0,    3  //offset = -4
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 14
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr12,   vr0,    2  //offset = -5
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr8,    vr5

    vreplvei.h          vr6,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr8,    vr6

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr8,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr8,    vr5
    vmaddwod.h.bu.b     vr10,   vr8,    vr6
    vmaddwod.h.bu.b     vr13,   vr8,    vr11

    vbsrl.v             vr16,   vr0,    1  //offset = -6
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr8,    vr16,   vr16,   vr20
    vmulwev.h.bu.b      vr5,    vr8,    vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -5
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr12,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr3,    vr12

    vmaddwod.h.bu.b     vr5,    vr8,    vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr12

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 14
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8
.endif
    addi.d              t2,     t2,     -1
    addi.d              t0,     t0,     2  //src offset for next loop
    bnez                t2,     1b
endfunc
.endm

INTRA_PRED_ANG16_14 14
INTRA_PRED_ANG16_14 22

const lasx_ang16_mode14, align=5
.byte 6,7,6,7,5,6,5,6,4,5,4,5,4,5,3,4,3,4,2,3,2,3,2,3,1,2,1,2,0,1,0,1
.byte 32-19, 19, 32-6, 6, 32-25, 25, 32-12, 12, 32-31, 31, 32-18, 18, 32-5, 5, 32-24, 24
.byte 32-11, 11, 32-30, 30, 32-17, 17, 32-4, 4, 32-23, 23, 32-10, 10, 32-29, 29, 32-16, 16
.byte 15, 12, 10, 7, 5, 2, 0, 17, 18, 19, 20, 21, 22, 23, 24
endconst
function x265_intra_pred_ang16_14_lasx
    la.local            t1,     lasx_ang16_mode14
    vld                 vr12,   t1,     64
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    xvaddi.bu           xr22,   xr20,   2  //shuf
    xvaddi.bu           xr23,   xr20,   3  //shuf
    vld                 vr0,    a2,     32
    vld                 vr11,   a2,     0
    vshuf.b             vr0,    vr0,    vr11,   vr12 //Get the reference pixels
    ANG16_MODE11_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr0,    4
    ANG16_MODE11_PROCESS_4x16_B 1
    vld                 vr0,    a2,     32+2
    ANG16_MODE11_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr0,    4
    ANG16_MODE11_PROCESS_4x16_B 0
endfunc

const ang16_mode15, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-15, 15, 32-30, 30, 32-13, 13, 32-28, 28, 32-11, 11, 32-26, 26, 32-9, 9, 32-24, 24
.byte 32-7, 7, 32-22, 22, 32-5, 5, 32-20, 20, 32-3, 3, 32-18, 18, 32-1, 1, 32-16, 16
.byte 15, 13, 11, 9, 8, 6, 4, 2, 16, 17, 18, 19, 20, 21, 22, 23
endconst
.macro INTRA_PRED_ANG16_15 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 15
    vld                 vr0,    a2,     32
    vld                 vr2,    a2,     0
.else
    vld                 vr0,    a2,     0
    vld                 vr2,    a2,     32
.endif

    la.local            t1,     ang16_mode15
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction

    //col 0-7
    vld                 vr4,    t1,     48
    vshuf.b             vr5,    vr0,    vr2,    vr4 //15 13 11 9 8 6 4 2 * 33 34 35 36 37 38 39
.if \mode == 15
    vextrins.b          vr5,    vr2,    0x80 // *==>0
.endif
    vbsrl.v             vr12,   vr5,    8    //offset = -1
    vextrins.b          vr12,   vr0,    0x88 //0 33 34 35 36 37 38 39 40 0 ...
    vmov                vr0,    vr5

    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr12,   vr0,    7  //offset = -2
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr12,   vr0,    6  //offset = -3
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vbsrl.v             vr12,   vr0,    5  //offset = -4
    vreplvei.h          vr6,    vr1,    5
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr12,   vr0,    4  //offset = -5
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5
.if \mode == 15
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr12,   vr0,    3  //offset = -6
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr8,    vr5

    vreplvei.h          vr6,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr8,    vr6

    vbsrl.v             vr12,   vr0,    2  //offset = -7
    vreplvei.h          vr11,   vr23,   3
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr8,    vr5
    vmaddwod.h.bu.b     vr10,   vr8,    vr6
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vbsrl.v             vr12,   vr0,    1  //offset = -8
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vreplvei.h          vr12,   vr23,   7
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -9
    vmulwev.h.bu.b      vr17,   vr3,    vr12

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr12

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5
.if \mode == 15
    STORE8x16_B_TRANSPOSE

    //col 8-15
    vld                 vr2,    a2,     33 //32
    vldrepl.b           vr3,    a2,     0
    vbsll.v             vr0,    vr2,    1
    vextrins.b          vr0,    vr3,    0x00 //ref_pix[-1]
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8

    //col 8-15
    vld                 vr0,    a2,     0
    vldrepl.b           vr2,    a2,     16
.endif
    vbsrl.v             vr12,   vr0,    8  //offset = -1
    vextrins.b          vr12,   vr2,    0x8f

    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr12,   vr0,    7  //offset = -2
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr12,   vr0,    6  //offset = -3
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vbsrl.v             vr12,   vr0,    5  //offset = -4
    vreplvei.h          vr6,    vr1,    5
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr12,   vr0,    4  //offset = -5
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5
.if \mode == 15
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr12,   vr0,    3  //offset = -6
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr8,    vr5

    vreplvei.h          vr6,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr8,    vr6

    vbsrl.v             vr12,   vr0,    2  //offset = -7
    vreplvei.h          vr11,   vr23,   3
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr8,    vr5
    vmaddwod.h.bu.b     vr10,   vr8,    vr6
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vbsrl.v             vr12,   vr0,    1  //offset = -8
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vreplvei.h          vr12,   vr23,   7
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -9
    vmulwev.h.bu.b      vr17,   vr3,    vr12

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr12

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5
.if \mode == 15
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
.endif
endfunc
.endm

INTRA_PRED_ANG16_15 15
INTRA_PRED_ANG16_15 21

const lasx_ang16_mode15, align=5
.byte 8,9,7,8,7,8,6,7,6,7,5,6,5,6,4,5,4,5,3,4,3,4,2,3,2,3,1,2,1,2,0,1
.byte 32-15, 15, 32-30, 30, 32-13, 13, 32-28, 28, 32-11, 11, 32-26, 26, 32-9, 9, 32-24, 24
.byte 32-7, 7, 32-22, 22, 32-5, 5, 32-20, 20, 32-3, 3, 32-18, 18, 32-1, 1, 32-16, 16
.byte 15, 13, 11, 9, 8, 6, 4, 2, 0, 17, 18, 19, 20, 21, 22, 23
endconst
function x265_intra_pred_ang16_15_lasx
    la.local            t1,     lasx_ang16_mode15
    vld                 vr12,   t1,     64
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    xvaddi.bu           xr22,   xr20,   2  //shuf
    xvaddi.bu           xr23,   xr20,   3  //shuf
    vld                 vr0,    a2,     32
    vld                 vr11,   a2,     0
    vmov                vr13,   vr0
    vshuf.b             vr0,    vr0,    vr11,   vr12 //Get the reference pixels
    ANG16_MODE11_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr0,    4
    vextrins.b          vr0,    vr13,   0xc8
    ANG16_MODE11_PROCESS_4x16_B 1
    vmov                vr0,    vr13
    vextrins.b          vr0,    vr11,   0x00
    ANG16_MODE11_PROCESS_4x16_B 1
    vld                 vr0,    a2,     32+4
    ANG16_MODE11_PROCESS_4x16_B 0
endfunc

const ang16_mode16, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-11, 11, 32-22, 22, 32-1, 1, 32-12, 12, 32-23, 23, 32-2, 2, 32-13, 13, 32-24, 24
.byte 32-3, 3, 32-14, 14, 32-25, 25, 32-4, 4, 32-15, 15, 32-26, 26, 32-5, 5, 32-16, 16
.byte 15, 14, 12, 11, 9, 8, 6, 5, 3, 2, 16, 17, 18, 19, 20, 21
endconst
.macro INTRA_PRED_ANG16_16 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 16
    vld                 vr0,    a2,     32
    vld                 vr2,    a2,     0
.else
    vld                 vr0,    a2,     0
    vld                 vr2,    a2,     32
.endif

    la.local            t1,     ang16_mode16
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction

    //col 0-7
    vmov                vr12,   vr0
.if \mode == 16
    vextrins.b          vr12,   vr2,    0x00 //offset = -1
.endif

    vreplvei.h          vr3,    vr1,    0
    vshuf.b             vr6,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr4,    vr6,    vr3

    vbsll.v             vr12,   vr12,   1  //offset = -2
    vextrins.b          vr12,   vr2,    0x02
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr9,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr9,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vbsll.v             vr12,   vr12,   1  //offset = -3
    vextrins.b          vr12,   vr2,    0x03
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr6,    vr3
    vmaddwod.h.bu.b     vr7,    vr9,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vld                 vr5,    t1,     48
    vshuf.b             vr0,    vr0,    vr2,    vr5
.if \mode == 16
    vextrins.b          vr0,    vr2,    0xa0
.endif
    vbsrl.v             vr12,   vr0,    7 //offset = -4

    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr12,   vr0,    6 //offset = -5
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr12,   vr0,    5  //offset = -6
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5
.if \mode == 16
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr12,   vr0,    4  //offset = -7
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr8,    vr5

    vbsrl.v             vr12,   vr0,    3  //offset = -8
    vreplvei.h          vr6,    vr23,   2
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr10,   vr3,    vr6

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr8,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr6
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vbsrl.v             vr12,   vr0,    2  //offset = -9
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vbsrl.v             vr12,   vr0,    1  //offset = -10
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vreplvei.h          vr12,   vr23,   7
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -11
    vmulwev.h.bu.b      vr17,   vr3,    vr12

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr12

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5
.if \mode == 16
    STORE8x16_B_TRANSPOSE

    //col 8-15
    vld                 vr0,    a2,     33 //32
    vbsrl.v             vr12,   vr0,    7  //offset = -1
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8

    //col 8-15
    vld                 vr0,    a2,     1
    vbsrl.v             vr12,   vr0,    7 //offset = -1
.endif

    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr12,   vr0,    6  //offset = -2
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr9,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr9,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vbsrl.v             vr12,   vr0,    5  //offset = -3
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr9,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vbsrl.v             vr12,   vr0,    4 //offset = -4
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr12,   vr0,    3 //offset = -5
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr12,   vr0,    2  //offset = -6
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5
.if \mode == 16
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr12,   vr0,    1  //offset = -7
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr8,    vr5

    vreplvei.h          vr6,    vr23,   2
    vshuf.b             vr3,    vr0,    vr0,   vr20 //offset = -8
    vmulwev.h.bu.b      vr10,   vr3,    vr6

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr8,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr6
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vld                 vr1,    a2,     0
    vbsll.v             vr12,   vr0,    1  //offset = -9
    vextrins.b          vr12,   vr1,    0x00
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr5,    vr16,   vr2

.if \mode == 20
    vld                 vr1,    a2,     32
.endif
    vbsll.v             vr12,   vr12,   1  //offset = -10
    vextrins.b          vr12,   vr1,    0x02
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsll.v             vr0,    vr12,   1
    vextrins.b          vr0,    vr1,    0x03
    vreplvei.h          vr12,   vr23,   7
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -11
    vmulwev.h.bu.b      vr17,   vr3,    vr12

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr12

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5
.if \mode == 16
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
.endif
endfunc
.endm

INTRA_PRED_ANG16_16 16
INTRA_PRED_ANG16_16 20

const lasx_ang16_mode16, align=5
.byte 10,11,9,10,9,10,8,9,7,8,7,8,6,7,5,6,5,6,4,5,3,4,3,4,2,3,1,2,1,2,0,1
.byte 32-11, 11, 32-22, 22, 32-1, 1, 32-12, 12, 32-23, 23, 32-2, 2, 32-13, 13, 32-24, 24
.byte 32-3, 3, 32-14, 14, 32-25, 25, 32-4, 4, 32-15, 15, 32-26, 26, 32-5, 5, 32-16, 16
.byte 15, 14, 12, 11, 9, 8, 6, 5, 3, 2, 0, 17, 18, 19, 20, 21
.byte 0, 1, 2, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
endconst
function x265_intra_pred_ang16_16_lasx
    la.local            t1,     lasx_ang16_mode16
    vld                 vr12,   t1,     64
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    xvaddi.bu           xr22,   xr20,   2  //shuf
    xvaddi.bu           xr23,   xr20,   3  //shuf
    vld                 vr0,    a2,     32
    vld                 vr11,   a2,     0
    vmov                vr13,   vr0
    vshuf.b             vr0,    vr0,    vr11,   vr12 //Get the reference pixels
    ANG16_MODE11_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr0,    4
    vextrins.h          vr0,    vr13,   0x63
    vextrins.b          vr0,    vr13,   0xe8
    ANG16_MODE11_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr0,    4
    vld                 vr12,   t1,     80
    vshuf.b             vr0,    vr13,   vr0,    vr12
    ANG16_MODE11_PROCESS_4x16_B 1
    vld                 vr0,    a2,     32+2
    ANG16_MODE11_PROCESS_4x16_B 0
endfunc

const ang16_mode17, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-6, 6, 32-12, 12, 32-18, 18, 32-24, 24, 32-30, 30, 32-4, 4, 32-10, 10, 32-16, 16
.byte 32-22, 22, 32-28, 28, 32-2, 2, 32-8, 8, 32-14, 14, 32-20, 20, 32-26, 26, 32-0, 0
.byte 15, 14, 12, 11, 10, 9, 7, 6, 5, 4, 2, 1, 16, 17, 18, 19
endconst
.macro INTRA_PRED_ANG16_17 mode
function x265_intra_pred_ang16_\mode\()_lsx
.if \mode == 17
    vld                 vr0,    a2,     32
    vld                 vr2,    a2,     0
.else
    vld                 vr0,    a2,     0
    vld                 vr2,    a2,     32
.endif

    la.local            t1,     ang16_mode17
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction

    //col 0-7
    vmov                vr12,   vr0
.if \mode == 17
    vextrins.b          vr12,   vr2,    0x00 //offset = -1
.endif

    vreplvei.h          vr3,    vr1,    0
    vshuf.b             vr6,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr4,    vr6,    vr3

    vbsll.v             vr12,   vr12,   1  //offset = -2
    vextrins.b          vr12,   vr2,    0x01
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr9,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr9,    vr5

    vbsll.v             vr12,   vr12,   1  //offset = -3
    vextrins.b          vr12,   vr2,    0x02
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vbsll.v             vr12,   vr12,   1  //offset = -4
    vextrins.b          vr12,   vr2,    0x04
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr15,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr13,   vr15,   vr11

    vmaddwod.h.bu.b     vr4,    vr6,    vr3
    vmaddwod.h.bu.b     vr7,    vr9,    vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr15,   vr11

    vbsll.v             vr12,   vr12,   1 //offset = -5
    vextrins.b          vr12,   vr2,    0x05
    vreplvei.h          vr18,   vr1,    4
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr18

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vld                 vr8,    t1,     48
    vshuf.b             vr0,    vr0,    vr2,    vr8
.if \mode == 17
    vextrins.b          vr0,    vr2,    0xc0
.endif

    vbsrl.v             vr12,   vr0,    7 //offset = -6
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr12,   vr0,    6  //offset = -7
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr18
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5
.if \mode == 17
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vbsrl.v             vr12,   vr0,    5  //offset = -8
    vreplvei.h          vr2,    vr23,   0
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr12,   vr0,    4  //offset = -9
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr8,    vr5

    vreplvei.h          vr6,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr8,    vr6

    vbsrl.v             vr12,   vr0,    3  //offset = -10
    vreplvei.h          vr11,   vr23,   3
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr8,    vr5
    vmaddwod.h.bu.b     vr10,   vr8,    vr6
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vbsrl.v             vr12,   vr0,    2  //offset = -11
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vbsrl.v             vr12,   vr0,    1  //offset = -12
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -13
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13 //fraction is 0
.if \mode == 17
    STORE8x16_B_TRANSPOSE

    //col 8-15
    vld                 vr0,    a2,     33 //32
    vbsrl.v             vr12,   vr0,    7  //offset = -1
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
    addi.d              a0,     a0,     8

    //col 8-15
    vld                 vr0,    a2,     1
    vbsrl.v             vr12,   vr0,    7 //offset = -1
.endif

    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr12,   vr0,    6  //offset = -2
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr9,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr9,    vr5

    vbsrl.v             vr12,   vr0,    5  //offset = -3
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr15,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr10,   vr15,   vr8

    vbsrl.v             vr12,   vr0,    4  //offset = -4
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr9,    vr5
    vmaddwod.h.bu.b     vr10,   vr15,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vbsrl.v             vr12,   vr0,    3 //offset = -5
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr12,   vr0,    2 //offset = -6
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr12,   vr0,    1  //offset = -7
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5
.if \mode == 17
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    addi.d              t3,     a0,     0
    STORE8x8_B
.endif
    vreplvei.h          vr2,    vr23,   0
    vshuf.b             vr16,   vr0,    vr0,    vr20 //offset = -8
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vld                 vr1,    a2,     0
    vbsll.v             vr12,   vr0,    1  //offset = -9
    vextrins.b          vr12,   vr1,    0x00
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr7,    vr8,    vr5

.if \mode == 19
     vld                vr1,    a2,     32
.endif
    vreplvei.h          vr6,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr8,    vr6

    vbsll.v             vr12,   vr12,   1  //offset = -10
    vextrins.b          vr12,   vr1,    0x01
    vreplvei.h          vr11,   vr23,   3
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr8,    vr5
    vmaddwod.h.bu.b     vr10,   vr8,    vr6
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vbsll.v             vr12,   vr12,   1  //offset = -11
    vextrins.b          vr12,   vr1,    0x02
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr16,   vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vbsll.v             vr12,   vr12,   1  //offset = -12
    vextrins.b          vr12,   vr1,    0x04
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr8,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vbsll.v             vr0,    vr12,   1
    vextrins.b          vr0,    vr1,    0x05
    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -13
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13 //fraction is 0
.if \mode == 17
    STORE8x16_B_TRANSPOSE
.else
    add.d               t3,     t3,     a1
    STORE8x8_B
.endif
endfunc
.endm

INTRA_PRED_ANG16_17 17
INTRA_PRED_ANG16_17 19

.macro ANG16_MODE17_PROCESS_4x16_B idx0
    xvpermi.q           xr0,    xr0,    0x00
    xvshuf.b            xr3,    xr0,    xr0,    xr20
    xvmulwev.h.bu.b     xr4,    xr3,    xr1
    xvshuf.b            xr5,    xr0,    xr0,    xr21
    xvmulwev.h.bu.b     xr6,    xr5,    xr1
    xvpermi.q           xr2,    xr2,    0x00
    xvshuf.b            xr7,    xr2,    xr2,    xr20
    xvmulwev.h.bu.b     xr8,    xr7,    xr1
    xvshuf.b            xr9,    xr2,    xr2,    xr21
    xvmulwev.h.bu.b     xr10,   xr9,    xr1
    xvmaddwod.h.bu.b    xr4,    xr3,    xr1
    xvmaddwod.h.bu.b    xr6,    xr5,    xr1
    xvmaddwod.h.bu.b    xr8,    xr7,    xr1
    xvmaddwod.h.bu.b    xr10,   xr9,    xr1
    xvssrarni.bu.h      xr6,    xr4,    5
    xvssrarni.bu.h      xr10,   xr8,    5
    xvpermi.d           xr4,    xr6,    0xd8
    xvpermi.d           xr8,    xr10,   0xd8
    xvpermi.q           xr6,    xr4,    0x11
    xvpermi.q           xr10,   xr8,    0x11
    vst                 vr4,    a0,     0
    vstx                vr6,    a0,     a1
    alsl.d              a0,     a1,     a0,    1
    vst                 vr8,    a0,     0
    vstx                vr10,   a0,     a1
.if \idx0 == 1
    alsl.d              a0,     a1,     a0,    1
.endif
.endm

const lasx_ang16_mode17, align=5
.byte 12,13,11,12,10,11,9,10,8,9,8,9,7,8,6,7,5,6,4,5,4,5,3,4,2,3,1,2,0,1,0,1
.byte 32-6, 6, 32-12, 12, 32-18, 18, 32-24, 24, 32-30, 30, 32-4, 4, 32-10, 10, 32-16, 16
.byte 32-22, 22, 32-28, 28, 32-2, 2, 32-8, 8, 32-14, 14, 32-20, 20, 32-26, 26, 32-0, 0
.byte 15, 14, 12, 11, 10, 9, 7, 6, 5, 4, 2, 1, 0, 17, 18, 19
endconst
function x265_intra_pred_ang16_17_lasx
    la.local            t1,     lasx_ang16_mode17
    vld                 vr12,   t1,     64
    xvld                xr20,   t1,     0  //shuf
    xvld                xr1,    t1,     32 //fraction
    xvaddi.bu           xr21,   xr20,   1  //shuf
    vld                 vr0,    a2,     32
    vld                 vr11,   a2,     0
    vmov                vr13,   vr0
    vshuf.b             vr0,    vr0,    vr11,   vr12 //Get the reference pixels
    vbsrl.v             vr2,    vr0,    2
    vextrins.h          vr2,    vr13,   0x72
    ANG16_MODE17_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr2,    2
    vextrins.h          vr0,    vr13,   0x73
    vbsrl.v             vr2,    vr0,    2
    vextrins.h          vr2,    vr13,   0x74
    ANG16_MODE17_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr2,    2
    vextrins.h          vr0,    vr13,   0x75
    vbsrl.v             vr2,    vr0,    2
    vextrins.h          vr2,    vr13,   0x76
    ANG16_MODE17_PROCESS_4x16_B 1
    vbsrl.v             vr0,    vr2,    2
    vextrins.h          vr0,    vr13,   0x77
    vld                 vr2,    a2,     32+2
    ANG16_MODE17_PROCESS_4x16_B 1
endfunc

function x265_intra_pred_ang16_18_lsx
    vld                 vr0,    a2,     0  //0  1  2  ...
    vld                 vr1,    a2,     33 //33 34 35 ...

    vst                 vr0,    a0,     0

    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x00
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x01
    vst                 vr0,    a0,     0
    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x02
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x03
    vst                 vr0,    a0,     0
    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x04
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x05
    vst                 vr0,    a0,     0
    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x06
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x07
    vst                 vr0,    a0,     0
    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x08
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x09
    vst                 vr0,    a0,     0
    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x0a
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x0b
    vst                 vr0,    a0,     0
    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x0c
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1

    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x0d
    vst                 vr0,    a0,     0
    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0x0e
    vstx                vr0,    a0,     a1
endfunc

.macro INTRA_PRED_ANG32_2_CORE in0, in1
    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    \in0
    vst                 vr0,    a0,     0
    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    \in1
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1
.endm

.macro INTRA_PRED_ANG32_2 mode
function x265_intra_pred_ang32_\mode\()_lsx
    addi.d              t0,     a0,     0
    addi.d              t1,     a2,     0
.rept 2
.rept 2
.if \mode == 2
    vld                 vr0,    a2,     66 //66 67 ...
    vld                 vr1,    a2,     66+16
.else
    vld                 vr0,    a2,     2 //2 3 ...
    vld                 vr1,    a2,     2+16
.endif
    vst                 vr0,    a0,     0
    vbsrl.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr1,    0xf0
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,    1
    INTRA_PRED_ANG32_2_CORE     0xf1,   0xf2
    INTRA_PRED_ANG32_2_CORE     0xf3,   0xf4
    INTRA_PRED_ANG32_2_CORE     0xf5,   0xf6
    INTRA_PRED_ANG32_2_CORE     0xf7,   0xf8
    INTRA_PRED_ANG32_2_CORE     0xf9,   0xfa
    INTRA_PRED_ANG32_2_CORE     0xfb,   0xfc
    INTRA_PRED_ANG32_2_CORE     0xfd,   0xfe
    addi.d              a2,     a2,     16
.endr
    addi.d              a0,     t0,     16
    addi.d              a2,     t1,     16
.endr
endfunc
.endm

INTRA_PRED_ANG32_2 2
INTRA_PRED_ANG32_2 34

const ang32_mode3, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-26, 26, 32-20, 20, 32-14, 14, 32-8, 8, 32-2, 2, 32-28, 28, 32-22, 22, 32-16, 16
.byte 32-10, 10, 32-4, 4, 32-30, 30, 32-24, 24, 32-18, 18, 32-12, 12, 32-6, 6, 32-0, 0
endconst
.macro INTRA_PRED_ANG32_3 mode
function x265_intra_pred_ang32_\mode\()_lsx
.if \mode == 3
    addi.d              t0,     a2,     65
    addi.d              t2,     a0,     0
.else
    addi.d              t0,     a2,     1
    addi.d              t3,     a0,     0
.endif
    la.local            t1,     ang32_mode3
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16 //fraction
    vld                 vr23,   t1,     32 //fraction
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = 0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 1
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vbsrl.v             vr0,    vr0,    1  //offset = 2
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr9,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 3
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 4
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 5
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 6
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 3
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vld                 vr0,    t0,     7  //offset = 7

    vreplvei.h          vr2,    vr23,   0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 8
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr0,    vr0,    1 //offset = 9
    vreplvei.h          vr11,   vr23,   3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 10
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 11
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 12
    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 13

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 3
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t2,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vld                 vr0,    t0,     14 //offset = 14
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vbsrl.v             vr0,    vr0,    1  //offset = 15
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr9,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 16
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 17
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 18
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 19
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 3
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vld                 vr0,    t0,     20
    vreplvei.h          vr2,    vr23,   0
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = 20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 21
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 22
    vreplvei.h          vr11,   vr23,   3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 23
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 24
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 25
    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 26
    //vreplvei.h          vr15,   vr23,   7
    //vshuf.b             vr16,   vr0,    vr0,    vr20
    //vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    //vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13
.if \mode == 3
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t2,   3
    addi.d              t2,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif
    addi.w              t7,     t7,     -1
    addi.d              t0,     t0,     8
    bnez                t7,     1b
endfunc
.endm

INTRA_PRED_ANG32_3 3
INTRA_PRED_ANG32_3 33

const ang32_mode4, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-21, 21, 32-10, 10, 32-31, 31, 32-20, 20, 32-9, 9, 32-30, 30, 32-19, 19, 32-8, 8
.byte 32-29, 29, 32-18, 18, 32-7, 7, 32-28, 28, 32-17, 17, 32-6, 6, 32-27, 27, 32-16, 16
.byte 32-5, 5, 32-26, 26, 32-15, 15, 32-4, 4, 32-25, 25, 32-14, 14, 32-3, 3, 32-24, 24
.byte 32-13, 13, 32-2, 2, 32-23, 23, 32-12, 12, 32-1, 1, 32-22, 22, 32-11, 11, 32-0, 0
endconst
.macro INTRA_PRED_ANG32_4 mode
function x265_intra_pred_ang32_\mode\()_lsx
    addi.d              sp,     sp,     -16
    fst.d               f24,    sp,     0
    fst.d               f25,    sp,     8
.if \mode == 4
    addi.d              t0,     a2,     65
    addi.d              t2,     a0,     0
.else
    addi.d              t0,     a2,     1
    addi.d              t3,     a0,     0
.endif
    la.local            t1,     ang32_mode4
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    vld                 vr24,   t1,     16*3  //fraction
    vld                 vr25,   t1,     16*4  //fraction
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = 0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 1
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 2
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 3
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 4
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 5
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 4
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 6
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vld                 vr0,    t0,     7 //offset = 7
    vreplvei.h          vr8,    vr23,   2
    vshuf.b             vr9,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr9,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr9,    vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 8
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 9
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 10
    vreplvei.h          vr15,   vr23,   7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 4
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t2,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vbsrl.v             vr0,    vr0,    1 //offset = 11
    vreplvei.h          vr2,    vr24,   0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr24,   1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vbsrl.v             vr0,    vr0,    1  //offset = 12
    vreplvei.h          vr8,    vr24,   2
    vshuf.b             vr9,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 13
    vreplvei.h          vr11,   vr24,   3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr24,   4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 14
    vreplvei.h          vr6,    vr24,   5
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vld                 vr0,    t0,     15 //offset = 15
    vreplvei.h          vr11,   vr24,   6
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vreplvei.h          vr15,   vr24,   7
    vmulwev.h.bu.b      vr17,   vr8,    vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr8,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 4
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vbsrl.v             vr0,    vr0,    1 //offset = 16
    vreplvei.h          vr2,    vr25,   0
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = 20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 17
    vreplvei.h          vr5,    vr25,   1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr25,   2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 18
    vreplvei.h          vr11,   vr25,   3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 19
    vreplvei.h          vr2,    vr25,   4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr25,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 20
    vreplvei.h          vr11,   vr25,   6
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 21

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 4
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t2,   3
    addi.d              t2,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif

    addi.w              t7,     t7,     -1
    addi.d              t0,     t0,     8
    bnez                t7,     1b
    fld.d               f24,    sp,     0
    fld.d               f25,    sp,     8
    addi.d              sp,     sp,     16
endfunc
.endm

INTRA_PRED_ANG32_4 4
INTRA_PRED_ANG32_4 32

const ang32_mode5, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-17, 17, 32-2, 2, 32-19, 19, 32-4, 4, 32-21, 21, 32-6, 6, 32-23, 23, 32-8, 8
.byte 32-25, 25, 32-10, 10, 32-27, 27, 32-12, 12, 32-29, 29, 32-14, 14, 32-31, 31, 32-16, 16
.byte 32-1, 1, 32-18, 18, 32-3, 3, 32-20, 20, 32-5, 5, 32-22, 22, 32-7, 7, 32-24, 24
.byte 32-9, 9, 32-26, 26, 32-11, 11, 32-28, 28, 32-13, 13, 32-30, 30, 32-15, 15, 32-0, 0
endconst
.macro INTRA_PRED_ANG32_5 mode
function x265_intra_pred_ang32_\mode\()_lsx
    addi.d              sp,     sp,     -16
    fst.d               f24,    sp,     0
    fst.d               f25,    sp,     8
.if \mode == 5
    addi.d              t0,     a2,     65
    addi.d              t2,     a0,     0
.else
    addi.d              t0,     a2,     1
    addi.d              t3,     a0,     0
.endif
    la.local            t1,     ang32_mode5
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    vld                 vr24,   t1,     16*3  //fraction
    vld                 vr25,   t1,     16*4  //fraction
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = 0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 1
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 2
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 3
    vreplvei.h          vr6,    vr1,    5
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 4
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 5
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 5
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr0,    vr0,    1  //offset = 6
    vreplvei.h          vr11,   vr23,   3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vbsrl.v             vr0,    vr0,    1  //offset = 7
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vld                 vr0,    t0,     8 //offset = 8
    vreplvei.h          vr15,   vr23,   7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 5
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t2,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vbsrl.v             vr0,    vr0,    1 //offset = 9
    vreplvei.h          vr2,    vr24,   0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr24,   1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vbsrl.v             vr0,    vr0,    1  //offset = 10
    vreplvei.h          vr8,    vr24,   2
    vshuf.b             vr9,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vreplvei.h          vr11,   vr24,   3
    vmulwev.h.bu.b      vr13,   vr9,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr9,    vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 11
    vreplvei.h          vr2,    vr24,   4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr24,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 12
    vreplvei.h          vr11,   vr24,   6
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vreplvei.h          vr15,   vr24,   7
    vmulwev.h.bu.b      vr17,   vr8,    vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr8,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 5
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vbsrl.v             vr0,    vr0,    1 //offset = 13
    vreplvei.h          vr2,    vr25,   0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr25,   1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vbsrl.v             vr0,    vr0,    1  //offset = 14
    vreplvei.h          vr8,    vr25,   2
    vshuf.b             vr9,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vreplvei.h          vr11,   vr25,   3
    vmulwev.h.bu.b      vr13,   vr9,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr9,    vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 15
    vreplvei.h          vr2,    vr25,   4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr25,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vld                 vr0,    t0,     16  //offset = 16
    vreplvei.h          vr11,   vr25,   6
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr0,    vr0,    1  //offset = 17

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 5
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t2,   3
    addi.d              t2,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif

    addi.w              t7,     t7,     -1
    addi.d              t0,     t0,     8
    bnez                t7,     1b
    fld.d               f24,    sp,     0
    fld.d               f25,    sp,     8
    addi.d              sp,     sp,     16
endfunc
.endm

INTRA_PRED_ANG32_5 5
INTRA_PRED_ANG32_5 31

const ang32_mode6, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-13, 13, 32-26, 26, 32-7, 7, 32-20, 20, 32-1, 1, 32-14, 14, 32-27, 27, 32-8, 8
.byte 32-21, 21, 32-2, 2, 32-15, 15, 32-28, 28, 32-9, 9, 32-22, 22, 32-3, 3, 32-16, 16
.byte 32-29, 29, 32-10, 10, 32-23, 23, 32-4, 4, 32-17, 17, 32-30, 30, 32-11, 11, 32-24, 24
.byte 32-5, 5, 32-18, 18, 32-31, 31, 32-12, 12, 32-25, 25, 32-6, 6, 32-19, 19, 32-0, 0
endconst
.macro INTRA_PRED_ANG32_6 mode
function x265_intra_pred_ang32_\mode\()_lsx
    addi.d              sp,     sp,     -16
    fst.d               f24,    sp,     0
    fst.d               f25,    sp,     8
.if \mode == 6
    addi.d              t0,     a2,     65
.else
    addi.d              t0,     a2,     1
.endif
    la.local            t1,     ang32_mode6
    addi.d              t3,     a0,     0
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    vld                 vr24,   t1,     16*3  //fraction
    vld                 vr25,   t1,     16*4  //fraction
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     0 //offset = 0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vbsrl.v             vr0,    vr0,    1 //offset = 1
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr9,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr9,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr9,    vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 2
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 3
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 6
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 4
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr6,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr6,    vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 5
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 6
    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vreplvei.h          vr15,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 6
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t3,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vreplvei.h          vr2,    vr24,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 7
    vreplvei.h          vr5,    vr24,   1
    vshuf.b             vr6,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr24,   2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vld                 vr0,    t0,     8 //offset = 8
    vreplvei.h          vr11,   vr24,   3
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr24,   4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr24,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 9
    vreplvei.h          vr11,   vr24,   6
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vreplvei.h          vr15,   vr24,   7
    vmulwev.h.bu.b      vr17,   vr8,    vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr8,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 6
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vbsrl.v             vr0,    vr0,    1 //offset = 10
    vreplvei.h          vr2,    vr25,   0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr25,   1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr25,   2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vbsrl.v             vr0,    vr0,    1 //offset = 11
    vreplvei.h          vr11,   vr25,   3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr25,   4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 12
    vreplvei.h          vr6,    vr25,   5
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr25,   6
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 13

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 6
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t3,   3
    addi.d              t3,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif

    addi.w              t7,     t7,     -1
    addi.d              t0,     t0,     8
    bnez                t7,     1b
    fld.d               f24,    sp,     0
    fld.d               f25,    sp,     8
    addi.d              sp,     sp,     16
endfunc
.endm

INTRA_PRED_ANG32_6 6
INTRA_PRED_ANG32_6 30

const ang32_mode7, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-9, 9, 32-18, 18, 32-27, 27, 32-4, 4, 32-13, 13, 32-22, 22, 32-31, 31, 32-8, 8
.byte 32-17, 17, 32-26, 26, 32-3, 3, 32-12, 12, 32-21, 21, 32-30, 30, 32-7, 7, 32-16, 16
.byte 32-25, 25, 32-2, 2, 32-11, 11, 32-20, 20, 32-29, 29, 32-6, 6, 32-15, 15, 32-24, 24
.byte 32-1, 1, 32-10, 10, 32-19, 19, 32-28, 28, 32-5, 5, 32-14, 14, 32-23, 23, 32-0, 0
endconst
.macro INTRA_PRED_ANG32_7 mode
function x265_intra_pred_ang32_\mode\()_lsx
    addi.d              sp,     sp,     -16
    fst.d               f24,    sp,     0
    fst.d               f25,    sp,     8
.if \mode == 7
    addi.d              t0,     a2,     65
.else
    addi.d              t0,     a2,     1
.endif
    la.local            t1,     ang32_mode7
    addi.d              t3,     a0,     0
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    vld                 vr24,   t1,     16*3  //fraction
    vld                 vr25,   t1,     16*4  //fraction
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     0 //offset = 0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vbsrl.v             vr0,    vr0,    1 //offset = 1
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 2
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 7
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vbsrl.v             vr0,    vr0,    1 //offset = 3
    vreplvei.h          vr8,    vr23,   2
    vshuf.b             vr12,   vr0,    vr0,   vr20
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 4
    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vreplvei.h          vr15,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 7
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t3,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vreplvei.h          vr2,    vr24,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 5
    vreplvei.h          vr5,    vr24,   1
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr24,   2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr24,   3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr24,   4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 6
    vreplvei.h          vr6,    vr24,   5
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr24,   6
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vreplvei.h          vr15,   vr24,   7
    vmulwev.h.bu.b      vr17,   vr8,    vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11
    vmaddwod.h.bu.b     vr17,   vr8,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 7
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vbsrl.v             vr0,    vr0,    1 //offset = 7
    vreplvei.h          vr2,    vr25,   0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr25,   1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr25,   2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr25,   3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vld                 vr0,    t0,     8 //offset = 8
    vreplvei.h          vr2,    vr25,   4
    vshuf.b             vr8,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr8,    vr2

    vreplvei.h          vr6,    vr25,   5
    vmulwev.h.bu.b      vr9,    vr8,    vr6

    vreplvei.h          vr11,   vr25,   6
    vmulwev.h.bu.b      vr14,   vr8,    vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 13

    vmaddwod.h.bu.b     vr5,    vr8,    vr2
    vmaddwod.h.bu.b     vr9,    vr8,    vr6
    vmaddwod.h.bu.b     vr14,   vr8,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 7
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t3,   3
    addi.d              t3,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif

    addi.w              t7,     t7,     -1
    addi.d              t0,     t0,     8
    bnez                t7,     1b
    fld.d               f24,    sp,     0
    fld.d               f25,    sp,     8
    addi.d              sp,     sp,     16
endfunc
.endm

INTRA_PRED_ANG32_7 7
INTRA_PRED_ANG32_7 29

const ang32_mode8, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-5, 5, 32-10, 10, 32-15, 15, 32-20, 20, 32-25, 25, 32-30, 30, 32-3, 3, 32-8, 8
.byte 32-13, 13, 32-18, 18, 32-23, 23, 32-28, 28, 32-1, 1, 32-6, 6, 32-11, 11, 32-16, 16
.byte 32-21, 21, 32-26, 26, 32-31, 31, 32-4, 4, 32-9, 9, 32-14, 14, 32-19, 19, 32-24, 24
.byte 32-29, 29, 32-2, 2, 32-7, 7, 32-12, 12, 32-17, 17, 32-22, 22, 32-27, 27, 32-0, 0
endconst
.macro INTRA_PRED_ANG32_8 mode
function x265_intra_pred_ang32_\mode\()_lsx
    addi.d              sp,     sp,     -16
    fst.d               f24,    sp,     0
    fst.d               f25,    sp,     8
.if \mode == 8
    addi.d              t0,     a2,     65
.else
    addi.d              t0,     a2,     1
.endif
    la.local            t1,     ang32_mode8
    addi.d              t3,     a0,     0
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    vld                 vr24,   t1,     16*3  //fraction
    vld                 vr25,   t1,     16*4  //fraction
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     0 //offset = 0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr0,    vr0,    1 //offset = 1
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 8
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 2
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vreplvei.h          vr15,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 8
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t3,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vreplvei.h          vr2,    vr24,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vreplvei.h          vr5,    vr24,   1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vreplvei.h          vr8,    vr24,   2
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vbsrl.v             vr0,    vr0,    1 //offset = 3
    vreplvei.h          vr11,   vr24,   3
    vshuf.b             vr12,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr24,   4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr24,   5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr24,   6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vreplvei.h          vr15,   vr24,   7
    vmulwev.h.bu.b      vr17,   vr12,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr12,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 8
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vreplvei.h          vr2,    vr25,   0
    vmulwev.h.bu.b      vr4,    vr12,   vr2

    vbsrl.v             vr0,    vr0,    1 //offset = 4
    vreplvei.h          vr5,    vr25,   1
    vshuf.b             vr3,    vr0,    vr0,   vr20
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr25,   2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr25,   3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr12,   vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr25,   4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr25,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr25,   6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 5

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 8
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t3,   3
    addi.d              t3,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif

    addi.w              t7,     t7,     -1
    addi.d              t0,     t0,     8
    bnez                t7,     1b
    fld.d               f24,    sp,     0
    fld.d               f25,    sp,     8
    addi.d              sp,     sp,     16
endfunc
.endm

INTRA_PRED_ANG32_8 8
INTRA_PRED_ANG32_8 28

const ang32_mode9, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-2, 2, 32-4, 4, 32-6, 6, 32-8, 8, 32-10, 10, 32-12, 12, 32-14, 14, 32-16, 16
.byte 32-18, 18, 32-20, 20, 32-22, 22, 32-24, 24, 32-26, 26, 32-28, 28, 32-30, 30, 32-0, 0
endconst
.macro INTRA_PRED_ANG32_9 mode
function x265_intra_pred_ang32_\mode\()_lsx
.if \mode == 9
    addi.d              t0,     a2,     65
.else
    addi.d              t0,     a2,     1
.endif
    la.local            t1,     ang32_mode9
    addi.d              t3,     a0,     0
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     0 //offset = 0
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 9
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 1
    vreplvei.h          vr15,   vr23,   7
    vshuf.b             vr16,   vr0,    vr0,   vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 9
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t3,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vreplvei.h          vr2,    vr1,    0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 9
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vbsrl.v             vr0,    vr0,    1 //offset = 2

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 9
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t3,   3
    addi.d              t3,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif

    addi.w              t7,     t7,     -1
    addi.d              t0,     t0,     8
    bnez                t7,     1b
endfunc
.endm

INTRA_PRED_ANG32_9 9
INTRA_PRED_ANG32_9 27

function x265_intra_pred_ang32_10_lsx
    addi.d              t0,     a0,     0
    addi.d              t1,     a2,     0
    addi.d              t3,     a0,     0
    vldrepl.b           vr17,   a2,     65
    vldrepl.b           vr18,   a2,     0
    vsllwil.hu.bu       vr17,   vr17,   0 //top
    vsllwil.hu.bu       vr18,   vr18,   0 //topLeft
.rept 2
    vld                 vr0,    a2,     65
    //transpose
    vreplvei.b          vr1,    vr0,    0
    vreplvei.b          vr2,    vr0,    1
    vreplvei.b          vr3,    vr0,    2
    vreplvei.b          vr4,    vr0,    3
    vreplvei.b          vr5,    vr0,    4
    vreplvei.b          vr6,    vr0,    5
    vreplvei.b          vr7,    vr0,    6
    vreplvei.b          vr8,    vr0,    7
    vreplvei.b          vr9,    vr0,    8
    vreplvei.b          vr10,   vr0,    9
    vreplvei.b          vr11,   vr0,    10
    vreplvei.b          vr12,   vr0,    11
    vreplvei.b          vr13,   vr0,    12
    vreplvei.b          vr14,   vr0,    13
    vreplvei.b          vr15,   vr0,    14
    vreplvei.b          vr16,   vr0,    15
.rept 2
    vst                 vr1,    a0,     0
    vstx                vr2,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr3,    a0,     0
    vstx                vr4,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr5,    a0,     0
    vstx                vr6,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr7,    a0,     0
    vstx                vr8,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr9,    a0,     0
    vstx                vr10,   a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr11,   a0,     0
    vstx                vr12,   a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr13,   a0,     0
    vstx                vr14,   a0,     a1
    alsl.d              a0,     a1,     a0,   1
    vst                 vr15,   a0,     0
    vstx                vr16,   a0,     a1
    addi.d              a0,     t0,     16
.endr
    beqz                a4,     1f
    //bFilter != 0
    vld                 vr19,   a2,     1 //srcPixel[width2 + 1 + y]
    vexth.hu.bu         vr20,   vr19
    vsllwil.hu.bu       vr19,   vr19,   0
    vsub.h              vr19,   vr19,   vr18
    vsub.h              vr20,   vr20,   vr18
    vsrai.h             vr19,   vr19,   1
    vsrai.h             vr20,   vr20,   1
    vadd.h              vr19,   vr19,   vr17
    vadd.h              vr20,   vr20,   vr17
    vssrani.bu.h        vr20,   vr19,   0
    vst                 vr20,   t3,     0 //update Row 0
    addi.d              t3,     t3,     16
1:
    alsl.d              a0,     a1,     t0,   4
    addi.d              t0,     a0,     0
    addi.d              a2,     t1,     16
.endr
endfunc

function x265_intra_pred_ang32_26_lsx
    addi.d              t0,     a0,     0
    addi.d              t1,     a2,     0
    addi.d              t2,     a0,     0
    vldrepl.h           vr18,   a2,     0
    vsllwil.hu.bu       vr18,   vr18,   0
    vreplvei.h          vr17,   vr18,   1 //top
    vreplvei.h          vr18,   vr18,   0 //topLeft
.rept 2
    vld                 vr0,    a2,     1
    //store no transpose
.rept 15
    vst                 vr0,    a0,     0
    vstx                vr0,    a0,     a1
    alsl.d              a0,     a1,     a0,   1
.endr
    vst                 vr0,    a0,     0
    vstx                vr0,    a0,     a1
    beqz                a4,     1f
    //bFilter != 0
    vld                 vr19,   a2,     65//srcPixel[width2 + 1 + y]
    vexth.hu.bu         vr20,   vr19
    vsllwil.hu.bu       vr19,   vr19,   0
    vsub.h              vr19,   vr19,   vr18
    vsub.h              vr20,   vr20,   vr18
    vsrai.h             vr19,   vr19,   1
    vsrai.h             vr20,   vr20,   1
    vadd.h              vr19,   vr19,   vr17
    vadd.h              vr20,   vr20,   vr17
    vssrani.bu.h        vr20,   vr19,   0
    //update Col 0
    .irp i, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    vstelm.b            vr20,   t2,     0,   \i
    add.d               t2,     t2,     a1
    .endr
1:
    addi.d              a0,     t0,     16
    addi.d              a2,     t1,     16
.endr
endfunc

const ang32_mode11, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-30, 30, 32-28, 28, 32-26, 26, 32-24, 24, 32-22, 22, 32-20, 20, 32-18, 18, 32-16, 16
.byte 32-14, 14, 32-12, 12, 32-10, 10, 32-8, 8, 32-6, 6, 32-4, 4, 32-2, 2, 32-0, 0
endconst
.macro INTRA_PRED_ANG32_11 mode
function x265_intra_pred_ang32_\mode\()_lsx
.if \mode == 11
    addi.d              t0,     a2,     64-1
    vldrepl.b           vr2,    a2,     0
    vldrepl.b           vr3,    a2,     16
.else
    addi.d              t0,     a2,     0-1
    vldrepl.b           vr3,    a2,     64+16
.endif
    la.local            t1,     ang32_mode11
    addi.d              t3,     a0,     0
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    li.w                t6,     4
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     0
    blt                 t7,     t6,     2f //only insert in the beginning
    vextrins.b          vr0,    vr3,    0x00 //ref_pix[-2]
.if \mode == 11
    vextrins.b          vr0,    vr2,    0x10 //ref_pix[-1]
.endif
2:
    vbsrl.v             vr12,   vr0,    1 //offset = -1
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr12,   vr12,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 11
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr12,   vr13

.if \mode == 11
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t3,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vshuf.b             vr16,   vr0,    vr0,   vr20 //offset = -2
    vreplvei.h          vr2,    vr1,    0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 11
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 11
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t3,   3
    addi.d              t3,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif

    addi.d              t0,     t0,     8
    addi.w              t7,     t7,     -1
    bnez                t7,     1b
endfunc
.endm

INTRA_PRED_ANG32_11 11
INTRA_PRED_ANG32_11 25

const ang32_mode12, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-27, 27, 32-22, 22, 32-17, 17, 32-12, 12, 32-7, 7, 32-2, 2, 32-29, 29, 32-24, 24
.byte 32-19, 19, 32-14, 14, 32-9, 9, 32-4, 4, 32-31, 31, 32-26, 26, 32-21, 21, 32-16, 16
.byte 32-11, 11, 32-6, 6, 32-1, 1, 32-28, 28, 32-23, 23, 32-18, 18, 32-13, 13, 32-8, 8
.byte 32-3, 3, 32-30, 30, 32-25, 25, 32-20, 20, 32-15, 15, 32-10, 10, 32-5, 5, 32-0, 0
endconst
.macro INTRA_PRED_ANG32_12 mode
function x265_intra_pred_ang32_\mode\()_lsx
    addi.d              sp,     sp,     -16
    fst.d               f24,    sp,     0
    fst.d               f25,    sp,     8
.if \mode == 12
    addi.d              t0,     a2,     64-4
    vld                 vr2,    a2,     0  // 6 13
    vld                 vr3,    a2,     16 // 19 26
.else
    addi.d              t0,     a2,     0-4
    vld                 vr2,    a2,     64
    vld                 vr3,    a2,     64+16
.endif
    la.local            t1,     ang32_mode12
    addi.d              t3,     a0,     0
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    vld                 vr24,   t1,     16*3  //fraction
    vld                 vr25,   t1,     16*4  //fraction
    li.w                t6,     4
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     0

    blt                 t7,     t6,     2f //only insert in the beginning
    vextrins.b          vr0,    vr3,    0x0a //ref_pix[-5]
    vextrins.b          vr0,    vr3,    0x13 //ref_pix[-4]
    vextrins.b          vr0,    vr2,    0x2d //ref_pix[-3]
    vextrins.b          vr0,    vr2,    0x36 //ref_pix[-2]
.if \mode == 12
    vextrins.b          vr0,    vr2,    0x40 //ref_pix[-1]
.endif
2:
    vbsrl.v             vr22,   vr0,    4 //offset = -1
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr22,   vr0,    3 //offst = -2
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr12,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr12,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 12
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr12,   vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr12,   vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr17,   vr0,    2 //offset = -3
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr16,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vreplvei.h          vr15,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 12
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t3,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vreplvei.h          vr2,    vr24,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vreplvei.h          vr5,    vr24,   1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vreplvei.h          vr8,    vr24,   2
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vbsrl.v             vr22,   vr0,    1 // offset = -4
    vreplvei.h          vr11,   vr24,   3
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr24,   4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr24,   5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr24,   6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vreplvei.h          vr15,   vr24,   7
    vmulwev.h.bu.b      vr17,   vr12,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr12,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 12
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vreplvei.h          vr2,    vr25,   0
    vmulwev.h.bu.b      vr4,    vr12,   vr2

    vshuf.b             vr16,   vr0,    vr0,   vr20 //offset = -5
    vreplvei.h          vr5,    vr25,   1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vreplvei.h          vr8,    vr25,   2
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vreplvei.h          vr11,   vr25,   3
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr12,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vreplvei.h          vr2,    vr25,   4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr25,   5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vreplvei.h          vr11,   vr25,   6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 12
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t3,   3
    addi.d              t3,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif

    addi.d              t0,     t0,     8
    addi.w              t7,     t7,     -1
    bnez                t7,     1b
    fld.d               f24,    sp,     0
    fld.d               f25,    sp,     8
    addi.d              sp,     sp,     16
endfunc
.endm

INTRA_PRED_ANG32_12 12
INTRA_PRED_ANG32_12 24

const ang32_mode13, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-23, 23, 32-14, 14, 32-5, 5, 32-28, 28, 32-19, 19, 32-10, 10, 32-1, 1, 32-24, 24
.byte 32-15, 15, 32-6, 6, 32-29, 29, 32-20, 20, 32-11, 11, 32-2, 2, 32-25, 25, 32-16, 16
.byte 32-7, 7, 32-30, 30, 32-21, 21, 32-12, 12, 32-3, 3, 32-26, 26, 32-17, 17, 32-8, 8
.byte 32-31, 31, 32-22, 22, 32-13, 13, 32-4, 4, 32-27, 27, 32-18, 18, 32-9, 9, 32-0, 0
endconst
.macro INTRA_PRED_ANG32_13 mode
function x265_intra_pred_ang32_\mode\()_lsx
    addi.d              sp,     sp,     -16
    fst.d               f24,    sp,     0
    fst.d               f25,    sp,     8
.if \mode == 13
    addi.d              t0,     a2,     64-8
    vld                 vr2,    a2,     0  // 4 7 11 14
    vld                 vr3,    a2,     16 // 18 21 25 28
.else
    addi.d              t0,     a2,     0-8
    vld                 vr2,    a2,     64
    vld                 vr3,    a2,     64+16
.endif
    la.local            t1,     ang32_mode13
    addi.d              t3,     a0,     0
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    vld                 vr24,   t1,     16*3  //fraction
    vld                 vr25,   t1,     16*4  //fraction
    li.w                t5,     3
    li.w                t6,     4
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     0

    blt                 t7,     t6,     2f //Only insert in the 1st loop
    vextrins.b          vr0,    vr3,    0x0c //ref_pix[-9]
    vextrins.b          vr0,    vr3,    0x19 //ref_pix[-8]
    vextrins.b          vr0,    vr3,    0x25 //ref_pix[-7]
    vextrins.b          vr0,    vr3,    0x32 //ref_pix[-6]
    vextrins.b          vr0,    vr2,    0x4e //ref_pix[-5]
    vextrins.b          vr0,    vr2,    0x5b //ref_pix[-4]
    vextrins.b          vr0,    vr2,    0x67 //ref_pix[-3]
    vextrins.b          vr0,    vr2,    0x74 //ref_pix[-2]
.if \mode == 13
    vextrins.b          vr0,    vr2,    0x80 //ref_pix[-1]
.endif
2:
.if \mode == 13
    bne                 t7,     t5,     3f //Need to adjust ref_pix[-1]'s value
    vldrepl.b           vr2,    a2,     0  //in the 2nd loop
    vextrins.b          vr0,    vr2,    0x00
.endif
3:
    vldrepl.b           vr2,    t0,     16
    vbsrl.v             vr22,   vr0,    8 //offset = -1
    vextrins.b          vr22,   vr2,    0x80
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr3,    vr8

    vbsrl.v             vr22,   vr0,    7 //offset = -2
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr3,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vbsrl.v             vr22,   vr0,    6 //offset = -3
    vreplvei.h          vr15,   vr1,    7
    vshuf.b             vr16,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 13
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vreplvei.h          vr5,    vr23,   1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vbsrl.v             vr17,   vr0,    5 //offset = -4
    vreplvei.h          vr8,    vr23,   2
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vbsrl.v             vr17,   vr0,    4 //offset = -5
    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr16,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vreplvei.h          vr15,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 13
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t3,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vreplvei.h          vr2,    vr24,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr22,   vr0,    3 //offset = -6
    vreplvei.h          vr5,    vr24,   1
    vshuf.b             vr12,   vr22,   vr22,    vr20
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vreplvei.h          vr8,    vr24,   2
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vreplvei.h          vr11,   vr24,   3
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr24,   4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vbsrl.v             vr22,   vr0,    2 //offset = -7
    vreplvei.h          vr6,    vr24,   5
    vshuf.b             vr3,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vreplvei.h          vr11,   vr24,   6
    vmulwev.h.bu.b      vr14,   vr3,   vr11

    vreplvei.h          vr15,   vr24,   7
    vmulwev.h.bu.b      vr17,   vr3,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr3,   vr6
    vmaddwod.h.bu.b     vr14,   vr3,   vr11
    vmaddwod.h.bu.b     vr17,   vr3,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 13
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vbsrl.v             vr17,   vr0,    1 //offset = -8
    vreplvei.h          vr2,    vr25,   0
    vshuf.b             vr16,   vr17,   vr17,  vr20
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vreplvei.h          vr5,    vr25,   1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vreplvei.h          vr8,    vr25,   2
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vreplvei.h          vr11,   vr25,   3
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vshuf.b             vr16,   vr0,    vr0,   vr20
    vreplvei.h          vr2,    vr25,   4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr25,   5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vreplvei.h          vr11,   vr25,   6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 13
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t3,    3
    addi.d              t3,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif

    addi.d              t0,     t0,     8
    addi.w              t7,     t7,     -1
    bnez                t7,     1b
    fld.d               f24,    sp,     0
    fld.d               f25,    sp,     8
    addi.d              sp,     sp,     16
endfunc
.endm

INTRA_PRED_ANG32_13 13
INTRA_PRED_ANG32_13 23

const ang32_mode14, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-19, 19, 32-6, 6, 32-25, 25, 32-12, 12, 32-31, 31, 32-18, 18, 32-5, 5, 32-24, 24
.byte 32-11, 11, 32-30, 30, 32-17, 17, 32-4, 4, 32-23, 23, 32-10, 10, 32-29, 29, 32-16, 16
.byte 32-3, 3, 32-22, 22, 32-9, 9, 32-28, 28, 32-15, 15, 32-2, 2, 32-21, 21, 32-8, 8
.byte 32-27, 27, 32-14, 14, 32-1, 1, 32-20, 20, 32-7, 7, 32-26, 26, 32-13, 13, 32-0, 0
.byte 30, 27, 25, 22, 20, 17, 15, 12, 10, 7, 5, 2, 0, 0, 0, 0
endconst
.macro INTRA_PRED_ANG32_14 mode
function x265_intra_pred_ang32_\mode\()_lsx
    addi.d              sp,     sp,     -16-48
    fst.d               f24,    sp,     0
    fst.d               f25,    sp,     8
    addi.d              t4,     sp,     16
.if \mode == 14
    addi.d              t0,     a2,     64
    vld                 vr4,    a2,     0
    vld                 vr5,    a2,     16
.else
    addi.d              t0,     a2,     0
    vld                 vr4,    a2,     64
    vld                 vr5,    a2,     64+16
.endif
    la.local            t1,     ang32_mode14
    addi.d              t3,     a0,     0
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    vld                 vr24,   t1,     16*3  //fraction
    vld                 vr25,   t1,     16*4  //fraction
    //sort src in stack in ascending order
    vld                 vr2,    t1,     16*5
    vshuf.b             vr3,    vr5,    vr4,    vr2
    vst                 vr3,    t4,     0
.if \mode == 22
    vldrepl.b           vr2,    a2,     0
    vstelm.b            vr2,    t4,     12,     0
.endif
    vld                 vr3,    t0,     1
    vst                 vr3,    t4,     13
    vld                 vr2,    t0,     1+16
    vst                 vr2,    t4,     13+16
    addi.d              t0,     t4,     13
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     -4
    vbsrl.v             vr22,   vr0,    3 //offset = -1
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr3,    vr5

    vbsrl.v             vr22,   vr0,    2 //offset = -2
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr3,    vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr22,   vr0,    1 //offset = -3
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vreplvei.h          vr15,   vr1,    7 //offset = -4
    vshuf.b             vr16,   vr0,    vr0,    vr20
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 14
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vld                 vr0,    t0,     -7
    vbsrl.v             vr17,   vr0,    2 //offset = -5
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr17,   vr0,    1 //offset = -6
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr23,   5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr23,   6
    vshuf.b             vr16,   vr0,    vr0,    vr20 //offset = -7
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vreplvei.h          vr15,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 14
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t3,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vreplvei.h          vr2,    vr24,   0
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vld                 vr0,    t0,     -10
    vbsrl.v             vr22,   vr0,    2 //offset = -8
    vreplvei.h          vr5,    vr24,   1
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vreplvei.h          vr8,    vr24,   2
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vbsrl.v             vr22,   vr0,    1 //offset = -9
    vreplvei.h          vr11,   vr24,   3
    vshuf.b             vr3,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr24,   4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr24,   5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vshuf.b             vr12,   vr0,    vr0,    vr20 //offset = -10
    vreplvei.h          vr11,   vr24,   6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vreplvei.h          vr15,   vr24,   7
    vmulwev.h.bu.b      vr17,   vr12,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr12,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 14
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vld                 vr0,    t0,     -13
    vbsrl.v             vr17,   vr0,    2 //offset = -11
    vreplvei.h          vr2,    vr25,   0
    vshuf.b             vr16,   vr17,   vr17,  vr20
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vreplvei.h          vr5,    vr25,   1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vreplvei.h          vr8,    vr25,   2
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vbsrl.v             vr17,   vr0,    1 //offset = -12
    vreplvei.h          vr11,   vr25,   3
    vshuf.b             vr12,   vr17,   vr17,  vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vreplvei.h          vr2,    vr25,   4
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vshuf.b             vr16,   vr0,    vr0,   vr20 //offset = -13
    vreplvei.h          vr6,    vr25,   5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vreplvei.h          vr11,   vr25,   6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 14
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t3,   3
    addi.d              t3,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif

    addi.d              t0,     t0,     8
    addi.w              t7,     t7,     -1
    bnez                t7,     1b
    fld.d               f24,    sp,     0
    fld.d               f25,    sp,     8
    addi.d              sp,     sp,     16+48
endfunc
.endm

INTRA_PRED_ANG32_14 14
INTRA_PRED_ANG32_14 22

const ang32_mode15, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-15, 15, 32-30, 30, 32-13, 13, 32-28, 28, 32-11, 11, 32-26, 26, 32-9, 9, 32-24, 24
.byte 32-7, 7, 32-22, 22, 32-5, 5, 32-20, 20, 32-3, 3, 32-18, 18, 32-1, 1, 32-16, 16
.byte 32-31, 31, 32-14, 14, 32-29, 29, 32-12, 12, 32-27, 27, 32-10, 10, 32-25, 25, 32-8, 8
.byte 32-23, 23, 32-6, 6, 32-21, 21, 32-4, 4, 32-19, 19, 32-2, 2, 32-17, 17, 32-0, 0
.byte 30, 28, 26, 24, 23, 21, 19, 17, 15, 13, 11, 9, 8, 6, 4, 2
endconst
.macro INTRA_PRED_ANG32_15 mode
function x265_intra_pred_ang32_\mode\()_lsx
    addi.d              sp,     sp,     -16-56
    fst.d               f24,    sp,     0
    fst.d               f25,    sp,     8
    addi.d              t4,     sp,     16
.if \mode == 15
    addi.d              t0,     a2,     64
    vld                 vr4,    a2,     0
    vld                 vr5,    a2,     16
.else
    addi.d              t0,     a2,     0
    vld                 vr4,    a2,     64
    vld                 vr5,    a2,     64+16
.endif
    la.local            t1,     ang32_mode15
    addi.d              t3,     a0,     0
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    vld                 vr24,   t1,     16*3  //fraction
    vld                 vr25,   t1,     16*4  //fraction
    //sort src in stack in ascending order
    vld                 vr2,    t1,     16*5
    vshuf.b             vr3,    vr5,    vr4,    vr2
    vst                 vr3,    t4,     0
    vldrepl.b           vr2,    a2,     0
    vstelm.b            vr2,    t4,     16,     0 //[-1]
    vld                 vr3,    t0,     1
    vst                 vr3,    t4,     17
    vld                 vr2,    t0,     1+16
    vst                 vr2,    t4,     17+16
    addi.d              t0,     t4,     17
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     -5
    vbsrl.v             vr22,   vr0,    4 //offset = -1
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr22,   vr0,    3 //offset = -2
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vbsrl.v             vr22,   vr0,    2 //offset = -3
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr16,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vreplvei.h          vr2,    vr1,    4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vbsrl.v             vr22,   vr0,    1 //offset = -4
    vreplvei.h          vr6,    vr1,    5
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -5
    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 15
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vld                 vr0,    t0,     -9
    vbsrl.v             vr17,   vr0,    3 //offset = -6
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vbsrl.v             vr17,   vr0,    2 //0ffset = -7
    vreplvei.h          vr11,   vr23,   3
    vshuf.b             vr16,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vreplvei.h          vr2,    vr23,   4
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vbsrl.v             vr17,   vr0,    1 //offst = -8
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -9
    vreplvei.h          vr15,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 15
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t3,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vld                 vr0,    t0,     -13
    vbsrl.v             vr22,   vr0,    3 //offset = -10
    vreplvei.h          vr2,    vr24,   0
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr4,    vr12,   vr2

    vreplvei.h          vr5,    vr24,   1
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vbsrl.v             vr22,   vr0,    2 //offset = -11
    vreplvei.h          vr8,    vr24,   2
    vshuf.b             vr16,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vreplvei.h          vr11,   vr24,   3
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr12,   vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vbsrl.v             vr22,   vr0,    1 //offset = -12
    vreplvei.h          vr2,    vr24,   4
    vshuf.b             vr16,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr24,   5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -13
    vreplvei.h          vr11,   vr24,   6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vreplvei.h          vr15,   vr24,   7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 15
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vld                 vr0,    t0,     -17
    vbsrl.v             vr17,   vr0,    3 //offset = -14
    vreplvei.h          vr2,    vr25,   0
    vshuf.b             vr16,   vr17,   vr17,  vr20
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vreplvei.h          vr5,    vr25,   1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vbsrl.v             vr17,   vr0,    2 //offset = -15
    vreplvei.h          vr8,    vr25,   2
    vshuf.b             vr12,   vr17,   vr17,  vr20
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vreplvei.h          vr11,   vr25,   3
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr17,   vr0,    1 //offset = -16
    vreplvei.h          vr2,    vr25,   4
    vshuf.b             vr12,   vr17,   vr17,  vr20
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr25,   5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vshuf.b             vr16,   vr0,    vr0,   vr20
    vreplvei.h          vr11,   vr25,   6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 15
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t3,   3
    addi.d              t3,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif

    addi.d              t0,     t0,     8
    addi.w              t7,     t7,     -1
    bnez                t7,     1b
    fld.d               f24,    sp,     0
    fld.d               f25,    sp,     8
    addi.d              sp,     sp,     16+56
endfunc
.endm

INTRA_PRED_ANG32_15 15
INTRA_PRED_ANG32_15 21

const ang32_mode16, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-11, 11, 32-22, 22, 32-1, 1, 32-12, 12, 32-23, 23, 32-2, 2, 32-13, 13, 32-24, 24
.byte 32-3, 3, 32-14, 14, 32-25, 25, 32-4, 4, 32-15, 15, 32-26, 26, 32-5, 5, 32-16, 16
.byte 32-27, 27, 32-6, 6, 32-17, 17, 32-28, 28, 32-7, 7, 32-18, 18, 32-29, 29, 32-8, 8
.byte 32-19, 19, 32-30, 30, 32-9, 9, 32-20, 20, 32-31, 31, 32-10, 10, 32-21, 21, 32-0, 0
.byte 30, 29, 27, 26, 24, 23, 21, 20, 18, 17, 15, 14, 12, 11, 9, 8
endconst
.macro INTRA_PRED_ANG32_16 mode
function x265_intra_pred_ang32_\mode\()_lsx
    addi.d              sp,     sp,     -16-56
    fst.d               f24,    sp,     0
    fst.d               f25,    sp,     8
    addi.d              t4,     sp,     16
.if \mode == 16
    addi.d              t0,     a2,     64
    vld                 vr4,    a2,     0
    vld                 vr5,    a2,     16
.else
    addi.d              t0,     a2,     0
    vld                 vr4,    a2,     64
    vld                 vr5,    a2,     64+16
.endif
    la.local            t1,     ang32_mode16
    addi.d              t3,     a0,     0
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    vld                 vr24,   t1,     16*3  //fraction
    vld                 vr25,   t1,     16*4  //fraction
    //sort src in stack in ascending order
    vld                 vr2,    t1,     16*5
    vshuf.b             vr3,    vr5,    vr4,    vr2
    vst                 vr3,    t4,     0
    vstelm.b            vr4,    t4,     16,     6
    vstelm.b            vr4,    t4,     17,     5
    vstelm.b            vr4,    t4,     18,     3
    vstelm.b            vr4,    t4,     19,     2
    vldrepl.b           vr3,    a2,     0
    vstelm.b            vr3,    t4,     20,     0 //[-1]
    vld                 vr2,    t0,     1
    vst                 vr2,    t4,     21
    vld                 vr3,    t0,     1+16
    vst                 vr3,    t4,     21+16
    addi.d              t0,     t4,     21
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     -6
    vbsrl.v             vr22,   vr0,    5 //offset = -1
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr22,   vr0,    4 //offset = -2
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vbsrl.v             vr22,   vr0,    3 //offset = -3
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr16,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vbsrl.v             vr22,   vr0,    2 //offset = -4
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr16,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vbsrl.v             vr22,   vr0,    1 //offset = -5
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -6
    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 16
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vreplvei.h          vr2,    vr23,   0
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vld                 vr0,    t0,     -11
    vbsrl.v             vr17,   vr0,    4 //offset = -7
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vbsrl.v             vr17,   vr0,    3 //0ffset = -8
    vreplvei.h          vr8,    vr23,   2
    vshuf.b             vr16,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vreplvei.h          vr11,   vr23,   3
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vbsrl.v             vr17,   vr0,    2 //offst = -9
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr16,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vbsrl.v             vr17,   vr0,    1 //offst = -10
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -11
    vreplvei.h          vr15,   vr23,   7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 16
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t3,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vld                 vr0,    t0,     -16
    vbsrl.v             vr22,   vr0,    4 //offset = -12
    vreplvei.h          vr2,    vr24,   0
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr4,    vr12,   vr2

    vreplvei.h          vr5,    vr24,   1
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vbsrl.v             vr22,   vr0,    3 //offset = -13
    vreplvei.h          vr8,    vr24,   2
    vshuf.b             vr16,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vbsrl.v             vr22,   vr0,    2 //offset = -14
    vreplvei.h          vr11,   vr24,   3
    vshuf.b             vr3,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr12,   vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vreplvei.h          vr2,    vr24,   4
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vbsrl.v             vr22,   vr0,    1 //offset = -15
    vreplvei.h          vr6,    vr24,   5
    vshuf.b             vr16,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vshuf.b             vr12,   vr0,    vr0,    vr20 //offset = -16
    vreplvei.h          vr11,   vr24,   6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vreplvei.h          vr15,   vr24,   7
    vmulwev.h.bu.b      vr17,   vr12,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr12,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 16
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vld                 vr0,    t0,     -21
    vbsrl.v             vr17,   vr0,    4 //offset = -17
    vreplvei.h          vr2,    vr25,   0
    vshuf.b             vr16,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr4,    vr16,   vr2

    vbsrl.v             vr17,   vr0,    3 //offset = -18
    vreplvei.h          vr5,    vr25,   1
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vreplvei.h          vr8,    vr25,   2
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vbsrl.v             vr17,   vr0,    2 //offset = -19
    vreplvei.h          vr11,   vr25,   3
    vshuf.b             vr3,    vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr16,   vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vbsrl.v             vr17,   vr0,    1 //offset = -20
    vreplvei.h          vr2,    vr25,   4
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr5,    vr12,   vr2

    vreplvei.h          vr6,    vr25,   5
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vshuf.b             vr16,   vr0,    vr0,    vr20 //offset = -21
    vreplvei.h          vr11,   vr25,   6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vmaddwod.h.bu.b     vr5,    vr12,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 16
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t3,   3
    addi.d              t3,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif

    addi.d              t0,     t0,     8
    addi.w              t7,     t7,     -1
    bnez                t7,     1b
    fld.d               f24,    sp,     0
    fld.d               f25,    sp,     8
    addi.d              sp,     sp,     16+56
endfunc
.endm

INTRA_PRED_ANG32_16 16
INTRA_PRED_ANG32_16 20

const ang32_mode17, align=4
.byte 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
.byte 32-6, 6, 32-12, 12, 32-18, 18, 32-24, 24, 32-30, 30, 32-4, 4, 32-10, 10, 32-16, 16
.byte 32-22, 22, 32-28, 28, 32-2, 2, 32-8, 8, 32-14, 14, 32-20, 20, 32-26, 26, 32-0, 0
.byte 31, 30, 28, 27, 26, 25, 23, 22, 21, 20, 18, 17, 16, 15, 14, 12, 11
.byte 10, 9, 7, 6, 5, 4, 2, 1, 0
endconst
.macro INTRA_PRED_ANG32_17 mode
function x265_intra_pred_ang32_\mode\()_lsx
    addi.d              sp,     sp,     -64
.if \mode == 17
    addi.d              t0,     a2,     64
    vld                 vr4,    a2,     0
    vld                 vr5,    a2,     16
.else
    addi.d              t0,     a2,     0
    vld                 vr4,    a2,     64
    vld                 vr5,    a2,     64+16
.endif
    la.local            t1,     ang32_mode17
    addi.d              t3,     a0,     0
    vld                 vr20,   t1,     0  //shuf
    vld                 vr1,    t1,     16*1  //fraction
    vld                 vr23,   t1,     16*2  //fraction
    //sort src in stack in ascending order
    vld                 vr2,    t1,     16*3
    vshuf.b             vr3,    vr5,    vr4,    vr2
    vst                 vr3,    sp,     0
    vld                 vr2,    t1,     16*4
    vshuf.b             vr3,    vr4,    vr4,    vr2
    vst                 vr3,    sp,     16
    vldrepl.b           vr2,    a2,     0
    vstelm.b            vr2,    sp,     25,     0 //ref_pix[-1]
    vld                 vr3,    t0,     1
    vst                 vr3,    sp,     26
    vld                 vr2,    t0,     1+16
    vst                 vr2,    sp,     26+16
    addi.d              t0,     sp,     26
    li.w                t7,     4
1:
    //Row 0 - 7
    vld                 vr0,    t0,     -7
    vbsrl.v             vr22,   vr0,    6 //offset = -1
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr3,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr22,   vr0,    5 //offset = -2
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vbsrl.v             vr22,   vr0,    4 //offset = -3
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr16,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr10,   vr16,   vr8

    vbsrl.v             vr22,   vr0,    3 //offset = -4
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr9,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr13,   vr9,    vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr9,    vr11

    vbsrl.v             vr22,   vr0,    2 //offset = -5
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr16,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr16,   vr6

    vbsrl.v             vr22,   vr0,    1 //offset = -6
    vreplvei.h          vr11,   vr1,    6
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -7
    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr3,    vr15

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr16,   vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr3,    vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 17
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 8 - 15
    vld                 vr0,    t0,     -13
    vbsrl.v             vr17,   vr0,    5 //offset = -8
    vreplvei.h          vr2,    vr23,   0
    vshuf.b             vr3,    vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr17,   vr0,    4 //offset = -9
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr7,    vr12,   vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vbsrl.v             vr17,   vr0,    3 //offset = -10
    vreplvei.h          vr11,   vr23,   3
    vshuf.b             vr16,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr13,   vr16,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr12,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr16,   vr11

    vbsrl.v             vr17,   vr0,    2 //offst = -11
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr16,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr5,    vr16,   vr2

    vbsrl.v             vr17,   vr0,    1 //offst = -12
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vshuf.b             vr3,    vr0,    vr0,    vr20 //offset = -13
    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr3,    vr11

    vmaddwod.h.bu.b     vr5,    vr16,   vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr3,    vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 17
    STORE8x16_B_TRANSPOSE
    addi.d              a0,     t3,     16
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 16 - 23
    vld                 vr0,    t0,     -20
    vbsrl.v             vr22,   vr0,    6 //offset = -14
    vreplvei.h          vr2,    vr1,    0
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr4,    vr12,   vr2

    vbsrl.v             vr22,   vr0,    5 //offset = -15
    vreplvei.h          vr5,    vr1,    1
    vshuf.b             vr16,   vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vbsrl.v             vr22,   vr0,    4 //offset = -16
    vreplvei.h          vr8,    vr1,    2
    vshuf.b             vr9,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vbsrl.v             vr22,   vr0,    3 //offset = -17
    vreplvei.h          vr11,   vr1,    3
    vshuf.b             vr3,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr13,   vr3,    vr11

    vmaddwod.h.bu.b     vr4,    vr12,   vr2
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr3,    vr11

    vbsrl.v             vr22,   vr0,    2 //offset = -18
    vreplvei.h          vr2,    vr1,    4
    vshuf.b             vr3,    vr22,   vr22,   vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vreplvei.h          vr6,    vr1,    5
    vmulwev.h.bu.b      vr9,    vr3,    vr6

    vbsrl.v             vr22,   vr0,    1 //offst = -19
    vshuf.b             vr12,   vr22,   vr22,   vr20
    vreplvei.h          vr11,   vr1,    6
    vmulwev.h.bu.b      vr14,   vr12,   vr11

    vshuf.b             vr16,   vr0,    vr0,    vr20 //offset = -20
    vreplvei.h          vr15,   vr1,    7
    vmulwev.h.bu.b      vr17,   vr16,   vr15

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr3,    vr6
    vmaddwod.h.bu.b     vr14,   vr12,   vr11
    vmaddwod.h.bu.b     vr17,   vr16,   vr15

    vssrarni.bu.h       vr5,    vr4,    5
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr17,   vr13,   5

.if \mode == 17
    TRANSPOSE4x16_B     vr5, vr9, vr14, vr17, \
                        vr18, vr19, vr21, vr22
.else
    STORE8x8_B
    add.d               t3,     t3,     a1
.endif

    //Row 24 - 31
    vld                 vr0,    t0,     -26
    vbsrl.v             vr17,   vr0,    5 //offset = -21
    vreplvei.h          vr2,    vr23,   0
    vshuf.b             vr3,    vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr4,    vr3,    vr2

    vbsrl.v             vr17,   vr0,    4 //offset = -22
    vreplvei.h          vr5,    vr23,   1
    vshuf.b             vr9,    vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr7,    vr9,    vr5

    vreplvei.h          vr8,    vr23,   2
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vbsrl.v             vr17,   vr0,    3 //offset = -23
    vreplvei.h          vr11,   vr23,   3
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr3,    vr2
    vmaddwod.h.bu.b     vr7,    vr9,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vbsrl.v             vr17,   vr0,    2 //offset = -24
    vreplvei.h          vr2,    vr23,   4
    vshuf.b             vr3,    vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr5,    vr3,    vr2

    vbsrl.v             vr17,   vr0,    1 //offset = -25
    vreplvei.h          vr6,    vr23,   5
    vshuf.b             vr12,   vr17,   vr17,   vr20
    vmulwev.h.bu.b      vr9,    vr12,   vr6

    vshuf.b             vr16,   vr0,    vr0,    vr20
    vreplvei.h          vr11,   vr23,   6
    vmulwev.h.bu.b      vr14,   vr16,   vr11

    vmaddwod.h.bu.b     vr5,    vr3,    vr2
    vmaddwod.h.bu.b     vr9,    vr12,   vr6
    vmaddwod.h.bu.b     vr14,   vr16,   vr11

    vssrarni.bu.h       vr5,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr9,    vr7,    5
    vssrarni.bu.h       vr14,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5
    vilvl.d             vr17,   vr0,    vr13

.if \mode == 17
    STORE8x16_B_TRANSPOSE
    alsl.d              a0,     a1,     t3,   3
    addi.d              t3,     a0,     0
.else
    STORE8x8_B
    addi.d              t3,     a0,     8
    addi.d              a0,     a0,     8
.endif
    addi.d              t0,     t0,     8
    addi.w              t7,     t7,     -1
    bnez                t7,     1b
    addi.d              sp,     sp,     64
endfunc
.endm

INTRA_PRED_ANG32_17 17
INTRA_PRED_ANG32_17 19

function x265_intra_pred_ang32_18_lsx
    vld                 vr0,    a2,     0  //0  1  2 .. 15
    vld                 vr1,    a2,     16 // 16 17 .. 31

    vld                 vr2,    a2,     64+1 //1 2 ... 16
    vld                 vr3,    a2,     64+1+16 // 17 18 .. 32
    //Row 0
    vst                 vr0,    a0,     0
    vst                 vr1,    a0,     16
    add.d               a0,     a0,     a1

    //Row 1-16
.irp i, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,\
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
    vbsll.v             vr1,    vr1,    1
    vextrins.b          vr1,    vr0,    0x0f
    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr2,    \i
    vst                 vr0,    a0,     0
    vst                 vr1,    a0,     16
    add.d               a0,     a0,     a1
.endr
    //Row 17-31
.irp i, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,\
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e
    vbsll.v             vr1,    vr1,    1
    vextrins.b          vr1,    vr0,    0x0f
    vbsll.v             vr0,    vr0,    1
    vextrins.b          vr0,    vr3,    \i
    vst                 vr0,    a0,     0
    vst                 vr1,    a0,     16
    add.d               a0,     a0,     a1
.endr
endfunc

function x265_intra_pred_ang4_2_lsx
    vld                 vr0,    a2,     8+2
    vbsrl.v             vr1,    vr0,    1
    vbsrl.v             vr2,    vr0,    2
    vbsrl.v             vr3,    vr0,    3
    fst.s               f0,     a0,     0
    fstx.s              f1,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    fst.s               f2,     a0,     0
    fstx.s              f3,     a0,     a1
endfunc

function x265_intra_pred_ang4_34_lsx
    vld                 vr0,    a2,     0+2
    vbsrl.v             vr1,    vr0,    1
    vbsrl.v             vr2,    vr0,    2
    vbsrl.v             vr3,    vr0,    3
    fst.s               f0,     a0,     0
    fstx.s              f1,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    fst.s               f2,     a0,     0
    fstx.s              f3,     a0,     a1
endfunc

.macro TRANSPOSE4x4_B_STORE
    vilvl.b    vr4,    vr7,    vr4
    vilvl.b    vr10,   vr13,   vr10
    vilvl.h    vr4,    vr10,   vr4
    vstelm.w   vr4,    a0,     0,    0
    add.d      a0,     a0,     a1
    vstelm.w   vr4,    a0,     0,    1
    add.d      a0,     a0,     a1
    vstelm.w   vr4,    a0,     0,    2
    add.d      a0,     a0,     a1
    vstelm.w   vr4,    a0,     0,    3
.endm

.macro STORE4x4_B
    fst.s      f4,     a0,     0
    fstx.s     f7,     a0,     a1
    alsl.d     a0,     a1,     a0,   1
    fst.s      f10,    a0,     0
    fstx.s     f13,    a0,     a1
.endm

.macro INTRA_PRED_ANG4_3 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 3
    vld                 vr0,    a2,     8+1
.else
    vld                 vr0,    a2,     0+1
.endif
    la.local            t1,     ang8_mode3
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0 //offset = 0
    vmulwev.h.bu.b      vr4,    vr0,    vr2 //(32 - fraction) * ref[offset + x]

    vbsrl.v             vr6,    vr0,    2 //offset = 1
    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vbsrl.v             vr9,    vr0,    4 //offset = 2
    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr9,    vr8

    vbsrl.v             vr12,   vr0,    6 //offset = 3
    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr0,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr9,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 3
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_3 3
INTRA_PRED_ANG4_3 33

.macro INTRA_PRED_ANG4_4 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 4
    vld                 vr0,    a2,     8+1
.else
    vld                 vr0,    a2,     0+1
.endif
    la.local            t1,     ang8_mode4
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0 //offset = 0
    vmulwev.h.bu.b      vr4,    vr0,    vr2 //(32 - fraction) * ref[offset + x]

    vbsrl.v             vr6,    vr0,    2 //offset = 1
    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr12,   vr0,    4 //offset = 2
    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr0,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 4
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_4 4
INTRA_PRED_ANG4_4 32

.macro INTRA_PRED_ANG4_5 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 5
    vld                 vr0,    a2,     8+1
.else
    vld                 vr0,    a2,     0+1
.endif
    la.local            t1,     ang8_mode5
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0 //offset = 0
    vmulwev.h.bu.b      vr4,    vr0,    vr2 //(32 - fraction) * ref[offset + x]

    vbsrl.v             vr6,    vr0,    2 //offset = 1
    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vbsrl.v             vr12,   vr0,    4 //offset = 2
    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr12,   vr11

    vmaddwod.h.bu.b     vr4,    vr0,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr12,   vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 5
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_5 5
INTRA_PRED_ANG4_5 31

.macro INTRA_PRED_ANG4_6 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 6
    vld                 vr0,    a2,     8+1
.else
    vld                 vr0,    a2,     0+1
.endif
    la.local            t1,     ang8_mode6
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0 //offset = 0
    vmulwev.h.bu.b      vr4,    vr0,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr0,    vr5

    vbsrl.v             vr6,    vr0,    2 //offset = 1
    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr6,    vr11

    vmaddwod.h.bu.b     vr4,    vr0,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr0,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr6,    vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 6
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_6 6
INTRA_PRED_ANG4_6 30

.macro INTRA_PRED_ANG4_7 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 7
    vld                 vr0,    a2,     8+1
.else
    vld                 vr0,    a2,     0+1
.endif
    la.local            t1,     ang8_mode7
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0 //offset = 0
    vmulwev.h.bu.b      vr4,    vr0,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr0,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr0,    vr8

    vbsrl.v             vr6,    vr0,    2 //offset = 1
    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr6,    vr11

    vmaddwod.h.bu.b     vr4,    vr0,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr0,    vr5
    vmaddwod.h.bu.b     vr10,   vr0,    vr8
    vmaddwod.h.bu.b     vr13,   vr6,    vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 7
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_7 7
INTRA_PRED_ANG4_7 29

.macro INTRA_PRED_ANG4_8 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 8
    vld                 vr0,    a2,     8+1
.else
    vld                 vr0,    a2,     0+1
.endif
    la.local            t1,     ang8_mode8
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0 //offset = 0
    vmulwev.h.bu.b      vr4,    vr0,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr0,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr0,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr0,    vr11

    vmaddwod.h.bu.b     vr4,    vr0,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr0,    vr5
    vmaddwod.h.bu.b     vr10,   vr0,    vr8
    vmaddwod.h.bu.b     vr13,   vr0,    vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 8
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_8 8
INTRA_PRED_ANG4_8 28

.macro INTRA_PRED_ANG4_9 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 9
    vld                 vr0,    a2,     8+1
.else
    vld                 vr0,    a2,     0+1
.endif
    la.local            t1,     ang8_mode9
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0 //offset = 0
    vmulwev.h.bu.b      vr4,    vr0,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr0,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr0,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr0,    vr11

    vmaddwod.h.bu.b     vr4,    vr0,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr0,    vr5
    vmaddwod.h.bu.b     vr10,   vr0,    vr8
    vmaddwod.h.bu.b     vr13,   vr0,    vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 9
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_9 9
INTRA_PRED_ANG4_9 27

function x265_intra_pred_ang4_10_lsx
    vld                 vr0,    a2,     8+1
    //transpose
    vreplvei.b          vr4,    vr0,    0
    vreplvei.b          vr7,    vr0,    1
    vreplvei.b          vr10,   vr0,    2
    vreplvei.b          vr13,   vr0,    3
    //bFilter != 0
    beqz                a4,     1f
    vldrepl.b           vr5,    a2,     0
    vsllwil.hu.bu       vr5,    vr5,    0
    vreplvei.h          vr6,    vr5,    0 //topLeft
    vsllwil.hu.bu       vr17,   vr4,    0
    vreplvei.h          vr17,   vr17,   0 //top
    vld                 vr8,    a2,     0+1
    vsllwil.hu.bu       vr8,    vr8,    0
    vsub.h              vr8,    vr8,    vr6
    vsrai.h             vr8,    vr8,    1
    vadd.h              vr8,    vr8,    vr17
    vssrani.bu.h        vr4,    vr8,    0
1:
    STORE4x4_B
endfunc

function x265_intra_pred_ang4_26_lsx
    vld                 vr1,    a2,     0
    vbsrl.v             vr0,    vr1,    1
    addi.d              t0,     a0,     0
    fst.s               f0,     a0,     0
    fstx.s              f0,     a0,     a1
    alsl.d              a0,     a1,     a0,    1
    fst.s               f0,     a0,     0
    fstx.s              f0,     a0,     a1
    //bFilter != 0
    beqz                a4,     1f
    vsllwil.hu.bu       vr5,    vr1,    0
    vreplvei.h          vr6,    vr5,    0 //topLeft
    vreplvei.h          vr7,    vr5,    1 //top
    vbsrl.v             vr8,    vr1,    8+1 ////srcPix[width2 + 1 + y]
    vsllwil.hu.bu       vr8,    vr8,    0
    vsub.h              vr8,    vr8,    vr6
    vsrai.h             vr8,    vr8,    1
    vadd.h              vr8,    vr8,    vr7
    vssrani.bu.h        vr4,    vr8,    0
    //Update Col 0
    vstelm.b            vr4,    t0,     0,    0
    add.d               t0,     t0,     a1
    vstelm.b            vr4,    t0,     0,    1
    add.d               t0,     t0,     a1
    vstelm.b            vr4,    t0,     0,    2
    add.d               t0,     t0,     a1
    vstelm.b            vr4,    t0,     0,    3
1:
endfunc

.macro INTRA_PRED_ANG4_11 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 11
    vld                 vr1,    a2,     0
    vbsrl.v             vr0,    vr1,    8
    vextrins.b          vr0,    vr1,    0x00
.else
    vld                 vr0,    a2,     0
.endif
    la.local            t1,     ang8_mode11
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0 //offset = -1
    vmulwev.h.bu.b      vr4,    vr0,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr0,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr0,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr0,    vr11

    vmaddwod.h.bu.b     vr4,    vr0,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr0,    vr5
    vmaddwod.h.bu.b     vr10,   vr0,    vr8
    vmaddwod.h.bu.b     vr13,   vr0,    vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 11
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_11 11
INTRA_PRED_ANG4_11 25

.macro INTRA_PRED_ANG4_12 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 12
    vld                 vr1,    a2,     0
    vbsrl.v             vr0,    vr1,    8
    vextrins.b          vr0,    vr1,    0x00
.else
    vld                 vr0,    a2,     0
.endif
    la.local            t1,     ang8_mode12
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0 //offset = -1
    vmulwev.h.bu.b      vr4,    vr0,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr0,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr0,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr0,    vr11

    vmaddwod.h.bu.b     vr4,    vr0,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr0,    vr5
    vmaddwod.h.bu.b     vr10,   vr0,    vr8
    vmaddwod.h.bu.b     vr13,   vr0,    vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 12
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_12 12
INTRA_PRED_ANG4_12 24

.macro INTRA_PRED_ANG4_13 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 13
    vld                 vr1,    a2,     0
    vbsrl.v             vr0,    vr1,    7
    vextrins.b          vr0,    vr1,    0x04
    vextrins.b          vr0,    vr1,    0x10
.else
    vld                 vr0,    a2,     0-1
    vextrins.b          vr0,    vr0,    0x0d
.endif
    la.local            t1,     ang8_mode13
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0
    vbsrl.v             vr6,    vr0,    2 //offset = -1
    vmulwev.h.bu.b      vr4,    vr6,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr6,    vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr0,    vr11 //offset = -2

    vmaddwod.h.bu.b     vr4,    vr6,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr6,    vr8
    vmaddwod.h.bu.b     vr13,   vr0,    vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 13
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_13 13
INTRA_PRED_ANG4_13 23

.macro INTRA_PRED_ANG4_14 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 14
    vld                 vr1,    a2,     0
    vbsrl.v             vr0,    vr1,    7
    vextrins.b          vr0,    vr1,    0x02
    vextrins.b          vr0,    vr1,    0x10
.else
    vld                 vr0,    a2,     0-1
    vextrins.b          vr0,    vr0,    0x0b
.endif
    la.local            t1,     ang8_mode14
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0
    vbsrl.v             vr6,    vr0,    2 //offset = -1
    vmulwev.h.bu.b      vr4,    vr6,    vr2 //(32 - fraction) * ref[offset + x]

    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr6,    vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr0,    vr8 //offset = -2

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr0,    vr11

    vmaddwod.h.bu.b     vr4,    vr6,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr6,    vr5
    vmaddwod.h.bu.b     vr10,   vr0,    vr8
    vmaddwod.h.bu.b     vr13,   vr0,    vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 14
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_14 14
INTRA_PRED_ANG4_14 22

.macro INTRA_PRED_ANG4_15 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 15
    vld                 vr1,    a2,     0
    vbsrl.v             vr0,    vr1,    6
    vextrins.b          vr0,    vr1,    0x04
    vextrins.b          vr0,    vr1,    0x12
    vextrins.b          vr0,    vr1,    0x20
.else
    vld                 vr0,    a2,     0-2
    vextrins.b          vr0,    vr0,    0x0e
    vextrins.b          vr0,    vr0,    0x1c
.endif
    la.local            t1,     ang8_mode15
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0
    vbsrl.v             vr6,    vr0,    4 //offset = -1
    vmulwev.h.bu.b      vr4,    vr6,    vr2 //(32 - fraction) * ref[offset + x]

    vbsrl.v             vr16,   vr0,    2 //offset = -2
    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr16,   vr8 //offset = -2

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr0,    vr11 //offset = -3

    vmaddwod.h.bu.b     vr4,    vr6,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr0,    vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 15
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_15 15
INTRA_PRED_ANG4_15 21

.macro INTRA_PRED_ANG4_16 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 16
    vld                 vr1,    a2,     0
    vbsrl.v             vr0,    vr1,    6
    vextrins.b          vr0,    vr1,    0x03
    vextrins.b          vr0,    vr1,    0x12
    vextrins.b          vr0,    vr1,    0x20
.else
    vld                 vr0,    a2,     0-2
    vextrins.b          vr0,    vr0,    0x0d
    vextrins.b          vr0,    vr0,    0x1c
.endif
    la.local            t1,     ang8_mode16
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0
    vbsrl.v             vr6,    vr0,    4 //offset = -1
    vmulwev.h.bu.b      vr4,    vr6,    vr2 //(32 - fraction) * ref[offset + x]

    vbsrl.v             vr16,   vr0,    2 //offset = -2
    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr16,   vr8 //offset = -2

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr0,    vr11 //offset = -3

    vmaddwod.h.bu.b     vr4,    vr6,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr16,   vr8
    vmaddwod.h.bu.b     vr13,   vr0,    vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 16
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_16 16
INTRA_PRED_ANG4_16 20

.macro INTRA_PRED_ANG4_17 mode
function x265_intra_pred_ang4_\mode\()_lsx
.if \mode == 17
    vld                 vr1,    a2,     0
    vbsrl.v             vr0,    vr1,    5
    vextrins.b          vr0,    vr1,    0x04
    vextrins.b          vr0,    vr1,    0x12
    vextrins.b          vr0,    vr1,    0x21
    vextrins.b          vr0,    vr1,    0x30
.else
    vld                 vr0,    a2,     0-3
    vextrins.b          vr0,    vr0,    0x0f
    vextrins.b          vr0,    vr0,    0x1d
    vextrins.b          vr0,    vr0,    0x2c
.endif
    la.local            t1,     ang8_mode17
    vld                 vr1,    t1,     16 //fraction

    vreplvei.h          vr2,    vr1,    0
    vbsrl.v             vr23,   vr0,    1
    vilvl.b             vr0,    vr23,   vr0
    vbsrl.v             vr6,    vr0,    6 //offset = -1
    vmulwev.h.bu.b      vr4,    vr6,    vr2 //(32 - fraction) * ref[offset + x]

    vbsrl.v             vr16,   vr0,    4 //offset = -2
    vreplvei.h          vr5,    vr1,    1
    vmulwev.h.bu.b      vr7,    vr16,   vr5

    vbsrl.v             vr12,   vr0,    2 //offset = -3
    vreplvei.h          vr8,    vr1,    2
    vmulwev.h.bu.b      vr10,   vr12,   vr8

    vreplvei.h          vr11,   vr1,    3
    vmulwev.h.bu.b      vr13,   vr0,    vr11 //offset = -4

    vmaddwod.h.bu.b     vr4,    vr6,    vr2 //fraction * ref[offset + x + 1]
    vmaddwod.h.bu.b     vr7,    vr16,   vr5
    vmaddwod.h.bu.b     vr10,   vr12,   vr8
    vmaddwod.h.bu.b     vr13,   vr0,    vr11

    vssrarni.bu.h       vr4,    vr4,    5 //+ 16) >> 5)
    vssrarni.bu.h       vr7,    vr7,    5
    vssrarni.bu.h       vr10,   vr10,   5
    vssrarni.bu.h       vr13,   vr13,   5

.if \mode == 17
    TRANSPOSE4x4_B_STORE
.else
    STORE4x4_B
.endif
endfunc
.endm

INTRA_PRED_ANG4_17 17
INTRA_PRED_ANG4_17 19

function x265_intra_pred_ang4_18_lsx
    vld                 vr13,   a2,     0-3
    vextrins.b          vr13,   vr13,   0x0e
    vextrins.b          vr13,   vr13,   0x1d
    vextrins.b          vr13,   vr13,   0x2c
    vbsrl.v             vr4,    vr13,   3
    vbsrl.v             vr7,    vr13,   2
    vbsrl.v             vr10,   vr13,   1
    STORE4x4_B
endfunc

/*
 * void x265_intra_pred_dc4_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter)
 */
function x265_intra_pred_dc4_lsx
   vld                  vr0,    a2,     1
   vbsrl.v              vr1,    vr0,    8
   vilvl.b              vr2,    vr1,    vr0
   vhaddw.hu.bu         vr2,    vr2,    vr2
   vhaddw.wu.hu         vr2,    vr2,    vr2
   vhaddw.du.wu         vr2,    vr2,    vr2
   vaddi.hu             vr2,    vr2,    4
   vreplvei.h           vr2,    vr2,    0
   vssrani.bu.h         vr3,    vr2,    3 //dcVal
   or                   t0,     zero,   a0
   //store dc 4x4
   fst.s                f3,     a0,     0
   fstx.s               f3,     a0,     a1
   alsl.d               a0,     a1,     a0,   1
   fst.s                f3,     a0,     0
   fstx.s               f3,     a0,     a1
   beqz                 a4,     1f
   //do dc filter
   vsllwil.hu.bu        vr2,    vr3,    0   //dst[x]
   vsllwil.hu.bu        vr4,    vr0,    0   //above[x]
   vslli.h              vr5,    vr2,    1
   vadd.h               vr5,    vr5,    vr2 //3 * dst[x]
   vadd.h               vr6,    vr5,    vr4
   vaddi.hu             vr6,    vr6,    2
   vssrani.bu.h         vr6,    vr6,    2   //dst[x] = (pixel)((above[x] +  3 * dst[x] + 2) >> 2);
   vbsrl.v              vr8,    vr0,    8   //left[y]
   vextrins.b           vr0,    vr0,    0x18
   vhaddw.hu.bu         vr0,    vr0,    vr0 //above[0] + left[0]
   vslli.h              vr7,    vr2,    1   //2 * dst[0]
   vadd.h               vr7,    vr7,    vr0
   vaddi.hu             vr7,    vr7,    2
   vssrani.bu.h         vr7,    vr7,    2   //dst[0] = (pixel)((above[0] + left[0] + 2 * dst[0] + 2) >> 2);
   vextrins.b           vr6,    vr7,    0x00
   fst.s                f6,     t0,     0   //filter row 0
   vsllwil.hu.bu        vr8,    vr8,    0
   vadd.h               vr8,    vr8,    vr5
   vaddi.hu             vr8,    vr8,    2
   vssrani.bu.h         vr8,    vr8,    2
   //filter col 0
   add.d                t0,     t0,     a1
   vstelm.b             vr8,    t0,     0,    1
   add.d                t0,     t0,     a1
   vstelm.b             vr8,    t0,     0,    2
   add.d                t0,     t0,     a1
   vstelm.b             vr8,    t0,     0,    3
1:
endfunc

function x265_intra_pred_dc8_lsx
   vld                  vr0,    a2,     1
   vld                  vr1,    a2,     17
   vilvl.b              vr2,    vr1,    vr0
   vhaddw.hu.bu         vr2,    vr2,    vr2
   vhaddw.wu.hu         vr2,    vr2,    vr2
   vhaddw.du.wu         vr2,    vr2,    vr2
   vhaddw.qu.du         vr2,    vr2,    vr2
   vaddi.hu             vr2,    vr2,    8
   vreplvei.h           vr2,    vr2,    0
   vssrani.bu.h         vr3,    vr2,    4 //dcVal
   or                   t0,     zero,   a0
   //store dc 8x8
   .rept 3
   fst.d                f3,     a0,     0
   fstx.d               f3,     a0,     a1
   alsl.d               a0,     a1,     a0,   1
   .endr
   fst.d                f3,     a0,     0
   fstx.d               f3,     a0,     a1
   beqz                 a4,     1f
   //do dc filter
   vsllwil.hu.bu        vr2,    vr3,    0   //dst[x]
   vsllwil.hu.bu        vr4,    vr0,    0   //above[x]
   vslli.h              vr5,    vr2,    1
   vadd.h               vr5,    vr5,    vr2 //3 * dst[x]
   vadd.h               vr6,    vr5,    vr4
   vaddi.hu             vr6,    vr6,    2
   vssrani.bu.h         vr6,    vr6,    2   //dst[x] = (pixel)((above[x] +  3 * dst[x] + 2) >> 2);
   vextrins.b           vr0,    vr1,    0x10
   vhaddw.hu.bu         vr0,    vr0,    vr0 //above[0] + left[0]
   vslli.h              vr7,    vr2,    1   //2 * dst[0]
   vadd.h               vr7,    vr7,    vr0
   vaddi.hu             vr7,    vr7,    2
   vssrani.bu.h         vr7,    vr7,    2   //dst[0] = (pixel)((above[0] + left[0] + 2 * dst[0] + 2) >> 2);
   vextrins.b           vr6,    vr7,    0x00
   fst.d                f6,     t0,     0   //filter row 0
   vsllwil.hu.bu        vr8,    vr1,    0
   vadd.h               vr8,    vr8,    vr5
   vaddi.hu             vr8,    vr8,    2
   vssrani.bu.h         vr8,    vr8,    2
   //filter col 0
   add.d                t0,     t0,     a1
   .irp i, 1,2,3,4,5,6
   vstelm.b             vr8,    t0,     0,    \i
   add.d                t0,     t0,     a1
   .endr
   vstelm.b             vr8,    t0,     0,    7
1:
endfunc

function x265_intra_pred_dc16_lsx
   vld                  vr0,    a2,     1
   vld                  vr1,    a2,     33
   vilvh.b              vr3,    vr1,    vr0
   vilvl.b              vr2,    vr1,    vr0
   vhaddw.hu.bu         vr2,    vr2,    vr2
   vhaddw.hu.bu         vr3,    vr3,    vr3
   vadd.h               vr3,    vr3,    vr2
   vhaddw.wu.hu         vr3,    vr3,    vr3
   vhaddw.du.wu         vr3,    vr3,    vr3
   vhaddw.qu.du         vr3,    vr3,    vr3
   vaddi.hu             vr3,    vr3,    16
   vssrani.bu.h         vr3,    vr3,    5 //dcVal
   vreplvei.b           vr3,    vr3,    0
   or                   t0,     zero,   a0
   //store dc 16x16
   .rept 7
   vst                  vr3,    a0,     0
   vstx                 vr3,    a0,     a1
   alsl.d               a0,     a1,     a0,   1
   .endr
   vst                  vr3,    a0,     0
   vstx                 vr3,    a0,     a1
   beqz                 a4,     1f
   //do dc filter
   vsllwil.hu.bu        vr4,    vr3,    0   //dst[x]
   vsllwil.hu.bu        vr6,    vr0,    0   //above[x]
   vexth.hu.bu          vr7,    vr0
   vslli.h              vr8,    vr4,    1
   vadd.h               vr8,    vr8,    vr4 //3 * dst[x]
   vadd.h               vr9,    vr8,    vr6
   vaddi.hu             vr9,    vr9,    2
   vadd.h               vr10,   vr8,    vr7
   vaddi.hu             vr10,   vr10,   2
   vssrani.bu.h         vr10,   vr9,    2   //dst[x] = (pixel)((above[x] +  3 * dst[x] + 2) >> 2);
   vextrins.b           vr0,    vr1,    0x10
   vhaddw.hu.bu         vr0,    vr0,    vr0 //above[0] + left[0]
   vslli.h              vr11,   vr4,    1   //2 * dst[0]
   vadd.h               vr11,   vr11,   vr0
   vaddi.hu             vr11,   vr11,   2
   vssrani.bu.h         vr11,   vr11,   2   //dst[0] = (pixel)((above[0] + left[0] + 2 * dst[0] + 2) >> 2);
   vextrins.b           vr10,   vr11,   0x00
   vst                  vr10,   t0,     0   //filter row 0
   vsllwil.hu.bu        vr12,   vr1,    0   //left[y]
   vexth.hu.bu          vr13,   vr1
   vadd.h               vr12,   vr12,   vr8 //left[y] + 3 * *dst
   vaddi.hu             vr12,   vr12,   2
   vadd.h               vr13,   vr13,   vr8
   vaddi.hu             vr13,   vr13,   2
   vssrani.bu.h         vr13,   vr12,   2
   //filter col 0
   add.d                t0,     t0,     a1
   .irp i, 1,2,3,4,5,6,7,8,9,10,11,12,13,14
   vstelm.b             vr13,   t0,     0,    \i
   add.d                t0,     t0,     a1
   .endr
   vstelm.b             vr13,   t0,     0,    15
1:
endfunc

function x265_intra_pred_dc32_lsx
   vld                  vr0,    a2,     1
   vld                  vr4,    a2,     1+16
   vld                  vr1,    a2,     65
   vld                  vr5,    a2,     65+16
   vilvh.b              vr3,    vr1,    vr0
   vilvl.b              vr2,    vr1,    vr0
   vilvh.b              vr7,    vr5,    vr4
   vilvl.b              vr6,    vr5,    vr4
   vhaddw.hu.bu         vr2,    vr2,    vr2
   vhaddw.hu.bu         vr3,    vr3,    vr3
   vhaddw.hu.bu         vr6,    vr6,    vr6
   vhaddw.hu.bu         vr7,    vr7,    vr7
   vadd.h               vr3,    vr3,    vr2
   vadd.h               vr3,    vr3,    vr6
   vadd.h               vr3,    vr3,    vr7
   vhaddw.wu.hu         vr3,    vr3,    vr3
   vhaddw.du.wu         vr3,    vr3,    vr3
   vhaddw.qu.du         vr3,    vr3,    vr3
   vaddi.hu             vr3,    vr3,    16
   vaddi.hu             vr3,    vr3,    16
   vssrani.bu.h         vr3,    vr3,    6 //dcVal
   vreplvei.b           vr3,    vr3,    0
   //store dc 32x32
   .rept 31
   vst                  vr3,    a0,     0
   vst                  vr3,    a0,     16
   add.d                a0,     a0,     a1
   .endr
   vst                  vr3,    a0,     0
   vst                  vr3,    a0,     16
endfunc

function x265_intra_pred_dc32_lasx
   xvld                 xr0,    a2,     1
   xvld                 xr1,    a2,     65
   xvilvh.b             xr3,    xr1,    xr0
   xvilvl.b             xr2,    xr1,    xr0
   xvhaddw.hu.bu        xr2,    xr2,    xr2
   xvhaddw.hu.bu        xr3,    xr3,    xr3
   xvadd.h              xr3,    xr3,    xr2
   xvhaddw.wu.hu        xr3,    xr3,    xr3
   xvhaddw.du.wu        xr3,    xr3,    xr3
   xvhaddw.qu.du        xr3,    xr3,    xr3
   xvpermi.q            xr4,    xr3,    0x01
   vadd.h               vr3,    vr3,    vr4
   vaddi.hu             vr3,    vr3,    16
   vaddi.hu             vr3,    vr3,    16
   vssrani.bu.h         vr3,    vr3,    6 //dcVal
   xvreplve0.b          xr3,    xr3
   //store dc 32x32
   .rept 15
   xvst                 xr3,    a0,     0
   xvstx                xr3,    a0,     a1
   alsl.d               a0,     a1,     a0,   1
   .endr
   xvst                 xr3,    a0,     0
   xvstx                xr3,    a0,     a1
endfunc
