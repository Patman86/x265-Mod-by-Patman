/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Changsheng Wu <wu.changsheng@sanechips.com.cn>
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

#include "asm.S"

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4
.text

.macro IDCT_MUL_1V4IMM m1, m2, m3, m4, src, d1, d2, d3, d4
    li              t1, \m1
    li              t2, \m2
    li              t3, \m3
    li              t4, \m4
    vmul.vx         \d1, \src, t1
    vmul.vx         \d2, \src, t2
    vmul.vx         \d3, \src, t3
    vmul.vx         \d4, \src, t4
.endm

.macro IDCT_MULADD_1V4IMM m1, m2, m3, m4, src, d1, d2, d3, d4
    li              t1, \m1
    li              t2, \m2
    li              t3, \m3
    li              t4, \m4
    vmacc.vx        \d1, t1, \src
    vmacc.vx        \d2, t2, \src
    vmacc.vx        \d3, t3, \src
    vmacc.vx        \d4, t4, \src
.endm

.macro IDCT_MULADD_2x2V2IMM m1, m2, m3, m4, src1, src2, dst1, dst2
    li              t1, \m1
    li              t2, \m2
    li              t3, \m3
    li              t4, \m4
    vmul.vx         \dst1, \src1, t1
    vmacc.vx        \dst1, t2, \src2
    vmul.vx         \dst2, \src1, t3
    vmacc.vx        \dst2, t4, \src2
.endm

.macro IDCT_SUMSUB_SHIFT dst1, dst2, src1, src2, shift
    vadd.vv \dst1, \src1, \src2
    vsub.vv \dst2, \src1, \src2
    vsll.vi \dst1, \dst1, \shift
    vsll.vi \dst2, \dst2, \shift
.endm

.macro TEMP_STORE len, r1, r2, r3, r4
    addi            sp, sp, -4 * \len
    addi            t1, sp, \len
    addi            t2, sp, \len * 2
    addi            t3, sp, \len * 3
    addi            a4, a4, 4 * \len
    vse32.v         \r1, (sp)
    vse32.v         \r2, (t1)
    vse32.v         \r3, (t2)
    vse32.v         \r4, (t3)
.endm

.macro TEMP_LOAD len, num, r1, r2, r3, r4
    addi            t1, sp, \num * \len
    addi            t2, sp, (\num + 1) * \len
    addi            t3, sp, (\num + 2) * \len
    addi            t4, sp, (\num + 3) * \len
    vle32.v         \r1, (t1)
    vle32.v         \r2, (t2)
    vle32.v         \r3, (t3)
    vle32.v         \r4, (t4)
.endm

.macro PBFI32_STORE8REG dst, off1, off2, shift, strided
    vnclip.wi       v17, v16, \shift
    vnclip.wi       v19, v18, \shift
    vnclip.wi       v21, v20, \shift
    vnclip.wi       v23, v22, \shift

    addi            t0, \dst, \off1
    slli            t2, \strided, 1
    add             t3, t2, \strided
    add             t1, t0, \strided
    add             t2, t2, t0
    add             t3, t3, t0

    vmv.v.i         v0, 5
    TRANSPOSE_4x4   16, v16, v18, v20, v22, v17, v19, v21, v23
    vsetivli        zero, 4, e16, mf2, ta, ma
    vse16.v         v16, (t0)
    vse16.v         v18, (t1)
    vse16.v         v20, (t2)
    vse16.v         v22, (t3)

    vnclip.wi       v17, v24, \shift
    vnclip.wi       v19, v26, \shift
    vnclip.wi       v21, v28, \shift
    vnclip.wi       v23, v30, \shift

    addi            t0, \dst, \off2
    slli            t2, \strided, 1
    add             t3, t2, \strided
    add             t1, t0, \strided
    add             t2, t2, t0
    add             t3, t3, t0

    TRANSPOSE_4x4   16, v16, v18, v20, v22, v17, v19, v21, v23
    vsetivli        zero, 4, e16, mf2, ta, ma
    vse16.v         v16, (t0)
    vse16.v         v18, (t1)
    vse16.v         v20, (t2)
    vse16.v         v22, (t3)
.endm

.macro PBFIM_STORE8REG4 dst, off1, shift, strided
    addi            t0, \dst, \off1
    li              t1, 0x55
    vmv.v.x         v0, t1
    vnclip.wi       v1, v16, \shift
    vnclip.wi       v2, v18, \shift
    vnclip.wi       v3, v20, \shift
    vnclip.wi       v4, v22, \shift
    vnclip.wi       v5, v24, \shift
    vnclip.wi       v6, v26, \shift
    vnclip.wi       v7, v28, \shift
    vnclip.wi       v8, v30, \shift

    TRANSPOSE_4x4   16, v16, v17, v18, v19, v1, v2, v3, v4
    vsetivli        zero, 4, e16, mf2, ta, ma
    TRANSPOSE_4x4   16, v20, v21, v22, v23, v5, v6, v7, v8

    vsetivli        zero, 8, e16, m1, ta, ma
    vslideup.vi     v16, v20, 4
    vslideup.vi     v17, v21, 4
    vslideup.vi     v18, v22, 4
    vslideup.vi     v19, v23, 4

    slli            t2, \strided, 1
    add             t1, t0, \strided
    add             t2, t2, t0
    add             t3, t2, \strided
    vse16.v         v16, (t0)
    vse16.v         v17, (t1)
    vse16.v         v18, (t2)
    vse16.v         v19, (t3)
.endm

.macro PBFIM_STORE8REG8 dst, off1, shift, strided
    addi            t0, \dst, \off1
    li              t1, 0x55
    vmv.v.x         v0, t1
    vnclip.wi       v1, v16, \shift
    vnclip.wi       v2, v18, \shift
    vnclip.wi       v3, v20, \shift
    vnclip.wi       v4, v22, \shift
    vnclip.wi       v5, v24, \shift
    vnclip.wi       v6, v26, \shift
    vnclip.wi       v7, v28, \shift
    vnclip.wi       v8, v30, \shift

    TRANSPOSE_8x8   16, v16, v17, v18, v19, v20, v21, v22, v23, v1, v2, v3, v4, v5, v6, v7, v8
    vsetivli        zero, 8, e16, m1, ta, ma

    slli            t4, \strided, 2
    slli            t2, \strided, 1
    add             t1, t0, \strided
    add             t2, t2, t0
    add             t3, t2, \strided
    vse16.v         v16, (t0)
    vse16.v         v17, (t1)
    vse16.v         v18, (t2)
    vse16.v         v19, (t3)

    add             t0, t0, t4
    add             t1, t1, t4
    add             t2, t2, t4
    add             t3, t3, t4
    vse16.v         v20, (t0)
    vse16.v         v21, (t1)
    vse16.v         v22, (t2)
    vse16.v         v23, (t3)
.endm

.macro IDCT32_LOAD_LINE strided, src, l1, l2, l3, l4, d1, d2, d3, d4
    addi            t1, \src, (\l1 * \strided)
    addi            t2, \src, (\l2 * \strided)
    addi            t3, \src, (\l3 * \strided)
    addi            t4, \src, (\l4 * \strided)
    vle16.v         \d1, (t1)
    vle16.v         \d2, (t2)
    vle16.v         \d3, (t3)
    vle16.v         \d4, (t4)
.endm

.macro IDCT32_SEXT32 d1, d2, d3, d4, s1, s2, s3, s4
    vsext.vf2       \d1, \s1
    vsext.vf2       \d2, \s2
    vsext.vf2       \d3, \s3
    vsext.vf2       \d4, \s4
.endm

// void partialButterflyInverse32(const int16_t* src)
.macro PARTIAL_BUTTER_FLY_INVERSE_32 src, dst, shift, strided, label
    li              a5, 32
    slli            a3, \strided, 2
\label:
    li              a4, 0
    vsetivli        zero, 4, e16, mf2, ta, ma
    IDCT32_LOAD_LINE 64, \src, 0, 8, 16, 24, v5, v7, v6, v8
    IDCT32_LOAD_LINE 64, \src, 4, 12, 20, 28, v13, v14, v15, v16
    vsetivli        zero, 4, e32, m1, ta, ma
    IDCT32_SEXT32   v1, v2, v3, v4, v5, v6, v7, v8

    // v5~6:EEEE[0~1] v8~9:EEEO[0~1] v1~v4:EEE[0~3]
    IDCT_SUMSUB_SHIFT v5, v6, v1, v2, 6
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v3, v4, v8, v9
    SUMSUB_ABCD     v1, v4, v2, v3, v5, v8, v6, v9

    vsetivli        zero, 4, e16, mf2, ta, ma
    IDCT32_LOAD_LINE 64, \src, 2, 6, 10, 14, v24, v25, v26, v27
    IDCT32_LOAD_LINE 64, \src, 18, 22, 26, 30, v28, v29, v30, v31
    vsetivli        zero, 4, e32, m1, ta, ma
    IDCT32_SEXT32   v7, v8, v9, v10, v13, v14, v15, v16

    // v12~15:EEO[0~3]
    IDCT_MUL_1V4IMM     89, 75, 50, 18, v7, v12, v13, v14, v15
    IDCT_MULADD_1V4IMM  75, -18, -89, -50, v8, v12, v13, v14, v15
    IDCT_MULADD_1V4IMM  50, -89, 18, 75, v9, v12, v13, v14, v15
    IDCT_MULADD_1V4IMM  18, -50, 75, -89, v10, v12, v13, v14, v15

    // v16~23:EE[0~7]
    SUMSUB_ABCD     v16, v23, v17, v22, v1, v12, v2, v13
    SUMSUB_ABCD     v18, v21, v19, v20, v3, v14, v4, v15

    // load v9~15 7
    vsetivli        zero, 4, e32, m1, ta, ma
    IDCT32_SEXT32   v9, v10, v11, v12, v24, v25, v26, v27
    IDCT32_SEXT32   v13, v14, v15, v7, v28, v29, v30, v31

    // v24~31 EO[0~7]
    IDCT_MUL_1V4IMM      90, 87, 80, 70, v9, v24, v25, v26, v27
    IDCT_MUL_1V4IMM      57, 43, 25, 9, v9, v28, v29, v30, v31
    IDCT_MULADD_1V4IMM  87, 57, 9, -43, v10, v24, v25, v26, v27
    IDCT_MULADD_1V4IMM  -80, -90, -70, -25, v10, v28, v29, v30, v31
    IDCT_MULADD_1V4IMM  80, 9, -70, -87, v11, v24, v25, v26, v27
    IDCT_MULADD_1V4IMM  -25, 57, 90, 43, v11, v28, v29, v30, v31
    IDCT_MULADD_1V4IMM  70, -43, -87, 9, v12, v24, v25, v26, v27
    IDCT_MULADD_1V4IMM  90, 25, -80, -57, v12, v28, v29, v30, v31
    IDCT_MULADD_1V4IMM  57, -80, -25, 90, v13, v24, v25, v26, v27
    IDCT_MULADD_1V4IMM  -9, -87, 43, 70, v13, v28, v29, v30, v31
    IDCT_MULADD_1V4IMM  43, -90, 57, 25, v14, v24, v25, v26, v27
    IDCT_MULADD_1V4IMM  -87, 70, 9, -80, v14, v28, v29, v30, v31
    IDCT_MULADD_1V4IMM  25, -70, 90, -80, v15, v24, v25, v26, v27
    IDCT_MULADD_1V4IMM  43, 9, -57, 87, v15, v28, v29, v30, v31
    IDCT_MULADD_1V4IMM  9, -25, 43, -57, v7, v24, v25, v26, v27
    IDCT_MULADD_1V4IMM  70, -80, 87, -90, v7, v28, v29, v30, v31

    // v0~15 E[0~15]
    SUMSUB_ABCD     v0, v15, v1, v14, v16, v24, v17, v25
    SUMSUB_ABCD     v2, v13, v3, v12, v18, v26, v19, v27
    SUMSUB_ABCD     v4, v11, v5, v10, v20, v28, v21, v29
    SUMSUB_ABCD     v6, v9, v7, v8, v22, v30, v23, v31

    // v17 19 21 23 O[0~3]
    vsetivli        zero, 4, e16, mf2, ta, ma
    IDCT32_LOAD_LINE 64, \src, 1, 3, 5, 7, v24, v25, v26, v27
    IDCT32_LOAD_LINE 64, \src, 9, 11, 13, 15, v28, v29, v30, v31
    vsetivli        zero, 4, e32, m1, ta, ma
    IDCT32_SEXT32   v16, v18, v20, v22, v24, v25, v26, v27
    TEMP_STORE      16, v16, v18, v20, v22
    IDCT_MUL_1V4IMM     90, 90, 88, 85, v16, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  90, 82, 67, 46, v18, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  88, 67, 31, -13, v20, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  85, 46, -13, -67, v22, v17, v19, v21, v23

    vsetivli        zero, 4, e16, mf2, ta, ma
    IDCT32_LOAD_LINE 64, \src, 17, 19, 21, 23, v24, v25, v26, v27
    vsetivli        zero, 4, e32, m1, ta, ma
    IDCT32_SEXT32   v16, v18, v20, v22, v28, v29, v30, v31
    TEMP_STORE      16, v16, v18, v20, v22
    IDCT_MULADD_1V4IMM  82, 22, -54, -90, v16, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  78, -4, -82, -73, v18, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  73, -31, -90, -22, v20, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  67, -54, -78, 38, v22, v17, v19, v21, v23

    vsetivli        zero, 4, e16, mf2, ta, ma
    IDCT32_LOAD_LINE 64, \src, 25, 27, 29, 31, v28, v29, v30, v31
    vsetivli        zero, 4, e32, m1, ta, ma
    IDCT32_SEXT32   v16, v18, v20, v22, v24, v25, v26, v27
    TEMP_STORE      16, v16, v18, v20, v22
    IDCT_MULADD_1V4IMM  61, -73, -46, 82, v16, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  54, -85, -4, 88, v18, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  46, -90, 38, 54, v20, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  38, -88, 73, -4, v22, v17, v19, v21, v23

    vsetivli        zero, 4, e32, m1, ta, ma
    IDCT32_SEXT32   v16, v18, v20, v22, v28, v29, v30, v31
    TEMP_STORE      16, v16, v18, v20, v22
    IDCT_MULADD_1V4IMM  31, -78, 90, -61, v16, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  22, -61, 85, -90, v18, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  13, -38, 61, -78, v20, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  4, -13, 22, -31, v22, v17, v19, v21, v23

    TEMP_LOAD       16, 12, v25, v27, v29, v31
    //v16 18 20 22 24 26 28 30 dst0~3 28~31
    SUMSUB_ABCD v16, v30, v18, v28, v0, v17, v1, v19
    SUMSUB_ABCD v20, v26, v22, v24, v2, v21, v3, v23
    vsetivli        zero, 4, e16, mf2, ta, ma
    PBFI32_STORE8REG \dst, 0, 56, \shift, \strided
    vsetivli        zero, 4, e32, m1, ta, ma

    // v0~3 O[4~7]
    TEMP_LOAD       16, 8, v16, v17, v18, v19
    IDCT_MUL_1V4IMM     82, 78, 73, 67, v25, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  22, -4, -31, -54, v27, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -54, -82, -90, -78, v29, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -90, -73, -22, 38, v31, v0, v1, v2, v3

    TEMP_LOAD       16, 4, v25, v27, v29, v31
    IDCT_MULADD_1V4IMM  -61, 13, 78, 85, v16, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  13, 85, 67, -22, v17, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  78, 67, -38, -90, v18, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  85, -22, -90, 4, v19, v0, v1, v2, v3

    TEMP_LOAD       16, 0, v16, v17, v18, v19
    IDCT_MULADD_1V4IMM  31, -88, -13, 90, v25, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -46, -61, 82, 13, v27, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -90, 31, 61, -88, v29, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -67, 90, -46, -31, v31, v0, v1, v2, v3

    IDCT_MULADD_1V4IMM  4, 54, -88, 82, v16, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  73, -38, -4, 46, v17, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  88, -90, 85, -73, v18, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  38, -46, 54, -61, v19, v0, v1, v2, v3

    TEMP_LOAD       16, 12, v25, v27, v29, v31
    //v16 18 20 22 24 26 28 30 dst4~7 24~27
    SUMSUB_ABCD v16, v30, v18, v28, v4, v0, v5, v1
    SUMSUB_ABCD v20, v26, v22, v24, v6, v2, v7, v3
    vsetivli        zero, 4, e16, mf2, ta, ma
    PBFI32_STORE8REG \dst, 8, 48, \shift, \strided
    vsetivli        zero, 4, e32, m1, ta, ma

    // v0~7 O[8~15]
    TEMP_LOAD       16, 8, v16, v17, v18, v19
    IDCT_MUL_1V4IMM     61, 54, 46, 38, v25, v0, v1, v2, v3
    IDCT_MUL_1V4IMM     31, 22, 13, 4, v25, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  -73, -85, -90, -88, v27, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -78, -61, -38, -13, v27, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  -46, -4, 38, 73, v29, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  90, 85, 61, 22, v29, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  82, 88, 54, -4, v31, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -61, -90, -78, -31, v31, v4, v5, v6, v7

    TEMP_LOAD       16, 4, v25, v27, v29, v31
    IDCT_MULADD_1V4IMM  31, -46, -90, -67, v16, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  4, 73, 88, 38, v16, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  -88, -61, 31, 90, v17, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  54, -38, -90, -46, v17, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  -13, 82, 61, -46, v18, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -88, -4, 85, 54, v18, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  90, 13, -88, -31, v19, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  82, 46, -73, -61, v19, v4, v5, v6, v7

    TEMP_LOAD       16, 0, v16, v17, v18, v19
    IDCT_MULADD_1V4IMM  -4, -90, 22, 85, v25, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -38, -78, 54, 67, v25, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  -90, 38, 67, -78, v27, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -22, 90, -31, -73, v27, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  22, 67, -85, 13, v29, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  73, -82, 4, 78, v29, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  85, -78, 13, 61, v31, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -90, 54, 22, -82, v31, v4, v5, v6, v7

    IDCT_MULADD_1V4IMM  -38, -22, 73, -90, v16, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  67, -13, -46, 85, v16, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  -78, 90, -82, 54, v17, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -13, -31, 67, -88, v17, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  54, -31, 4, 22, v18, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -46, 67, -82, 90, v18, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  67, -73, 78, -82, v19, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  85, -88, 90, -90, v19, v4, v5, v6, v7

    //v16 18 20 22 24 26 28 30 dst8~11 20~23
    SUMSUB_ABCD v16, v30, v18, v28, v8, v0, v9, v1
    SUMSUB_ABCD v20, v26, v22, v24, v10, v2, v11, v3
    vsetivli        zero, 4, e16, mf2, ta, ma
    PBFI32_STORE8REG \dst, 16, 40, \shift, \strided
    vsetivli        zero, 4, e32, m1, ta, ma

    //v16 18 20 22 24 26 28 30 dst12~15 16~19
    vsetivli         zero, 4, e32, m1, ta, ma
    SUMSUB_ABCD v16, v30, v18, v28, v12, v4, v13, v5
    SUMSUB_ABCD v20, v26, v22, v24, v14, v6, v15, v7
    vsetivli        zero, 4, e16, mf2, ta, ma
    PBFIM_STORE8REG4 \dst, 24, \shift, \strided

    add             sp, sp, a4
    addi            a5, a5, -4
    addi            \src, \src, 8
    add             \dst, \dst, a3
    bgtz            a5, \label
    slli            t1, a3, 3
    sub             \dst, \dst, t1
.endm

.macro PBFI32_STORE4REG dst, off1, off2, shift, strided, m1, m2, m3, m4, s1, s2, s3, s4
    addi            t0, \dst, \off1
    addi            t1, \dst, \off2
    vnclip.wi       \m1, \s1, \shift
    vnclip.wi       \m2, \s2, \shift
    vnclip.wi       \m3, \s3, \shift
    vnclip.wi       \m4, \s4, \shift
    vssseg2e16.v    \m1, (t0), \strided
    vssseg2e16.v    \m3, (t1), \strided
.endm

.macro PARTIAL_BUTTER_FLY_INVERSE_16 src, dst, shift, strided, label
    li              a5, 16
    slli            a3, \strided, 3
\label:
    li              a4, 0
    vsetivli        zero, 8, e16, m1, ta, ma
    IDCT32_LOAD_LINE 32, \src, 0, 4, 8, 12, v8, v10, v9, v11
    IDCT32_LOAD_LINE 32, \src, 2, 6, 10, 14, v24, v25, v26, v27
    vsetivli        zero, 8, e32, m2, ta, ma
    IDCT32_SEXT32   v0, v2, v4, v6, v8, v9, v10, v11

    // v8~10:EEE[0~1] v12~14:EEO[0~1] v0~v7:EE[0~3]
    IDCT_SUMSUB_SHIFT v8, v10, v0, v2, 6
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v4, v6, v12, v14
    SUMSUB_ABCD     v0, v6, v2, v4, v8, v12, v10, v14

    // v20~27:EO[0~3]
    IDCT32_SEXT32   v12, v14, v16, v18, v24, v25, v26, v27
    IDCT_MUL_1V4IMM     89, 75, 50, 18, v12, v20, v22, v24, v26
    IDCT_MULADD_1V4IMM  75, -18, -89, -50, v14, v20, v22, v24, v26
    IDCT_MULADD_1V4IMM  50, -89, 18, 75, v16, v20, v22, v24, v26
    IDCT_MULADD_1V4IMM  18, -50, 75, -89, v18, v20, v22, v24, v26

    // v8~v19 v28-v31:E[0~7]
    SUMSUB_ABCD     v8, v30, v10, v28, v0, v20, v2, v22
    SUMSUB_ABCD     v12, v18, v14, v16, v4, v24, v6, v26
    TEMP_STORE      32, v8, v10, v12, v14
    TEMP_STORE      32, v16, v18, v28, v30

    vsetivli        zero, 8, e16, m1, ta, ma
    IDCT32_LOAD_LINE 32, \src, 1, 3, 5, 7, v0, v1, v2, v3
    IDCT32_LOAD_LINE 32, \src, 9, 11, 13, 15, v4, v5, v6, v7
    vsetivli        zero, 8, e32, m2, ta, ma
    IDCT32_SEXT32   v16, v18, v20, v22, v0, v1, v2, v3
    IDCT32_SEXT32   v24, v26, v28, v30, v4, v5, v6, v7

    // v0~15 O[0~7]
    IDCT_MUL_1V4IMM      90, 87, 80, 70, v16, v0, v2, v4, v6
    IDCT_MUL_1V4IMM      57, 43, 25, 9, v16, v8, v10, v12, v14
    IDCT_MULADD_1V4IMM  87, 57, 9, -43, v18, v0, v2, v4, v6
    IDCT_MULADD_1V4IMM  -80, -90, -70, -25, v18, v8, v10, v12, v14
    IDCT_MULADD_1V4IMM  80, 9, -70, -87, v20, v0, v2, v4, v6
    IDCT_MULADD_1V4IMM  -25, 57, 90, 43, v20, v8, v10, v12, v14
    IDCT_MULADD_1V4IMM  70, -43, -87, 9, v22, v0, v2, v4, v6
    IDCT_MULADD_1V4IMM  90, 25, -80, -57, v22, v8, v10, v12, v14
    IDCT_MULADD_1V4IMM  57, -80, -25, 90, v24, v0, v2, v4, v6
    IDCT_MULADD_1V4IMM  -9, -87, 43, 70, v24, v8, v10, v12, v14
    IDCT_MULADD_1V4IMM  43, -90, 57, 25, v26, v0, v2, v4, v6
    IDCT_MULADD_1V4IMM  -87, 70, 9, -80, v26, v8, v10, v12, v14
    IDCT_MULADD_1V4IMM  25, -70, 90, -80, v28, v0, v2, v4, v6
    IDCT_MULADD_1V4IMM  43, 9, -57, 87, v28, v8, v10, v12, v14
    IDCT_MULADD_1V4IMM  9, -25, 43, -57, v30, v0, v2, v4, v6
    IDCT_MULADD_1V4IMM  70, -80, 87, -90, v30, v8, v10, v12, v14

    // dst[0~1,14~15] dst[2~3,12~13]
    TEMP_LOAD       32, 4, v16, v18, v20, v22
    SUMSUB_ABCD     v24, v30, v26, v28, v16, v0, v18, v2
    vsetivli        zero, 8, e16, m1, ta, ma
    PBFI32_STORE4REG \dst, 0, 28, \shift, \strided, v0, v1, v2, v3, v24, v26, v28, v30
    vsetivli        zero, 8, e32, m2, ta, ma
    SUMSUB_ABCD     v24, v30, v26, v28, v20, v4, v22, v6
    vsetivli        zero, 8, e16, m1, ta, ma
    PBFI32_STORE4REG \dst, 4, 24, \shift, \strided, v0, v1, v2, v3, v24, v26, v28, v30
    vsetivli        zero, 8, e32, m2, ta, ma

    // v16~v31 dst[4~7,8~11]
    TEMP_LOAD       32, 0, v0, v2, v4, v6
    SUMSUB_ABCD     v16, v30, v18, v28, v0, v8, v2, v10
    SUMSUB_ABCD     v20, v26, v22, v24, v4, v12, v6, v14
    vsetivli        zero, 8, e16, m1, ta, ma
    PBFIM_STORE8REG8 \dst, 8, \shift, \strided

    add             sp, sp, a4
    addi            a5, a5, -8
    addi            \src, \src, 16
    add             \dst, \dst, a3
    bgtz            a5, \label
    slli            t1, a3, 1
    sub             \dst, \dst, t1
.endm

.macro PARTIAL_BUTTER_FLY_INVERSE_8 src, dst, shift, strided, label
    vsetivli        zero, 8, e16, m1, ta, ma
    IDCT32_LOAD_LINE 16, \src, 0, 2, 4, 6, v8, v10, v9, v11
    IDCT32_LOAD_LINE 16, \src, 1, 3, 5, 7, v24, v25, v26, v27
    vsetivli        zero, 8, e32, m2, ta, ma
    IDCT32_SEXT32   v0, v2, v4, v6, v8, v9, v10, v11

    // v8~v11:EE[0~1] v12~15:EO[0~1] v0~7:E[0~3]
    IDCT_SUMSUB_SHIFT v8, v10, v0, v2, 6
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v4, v6, v12, v14
    SUMSUB_ABCD v0, v6, v2, v4, v8, v12, v10, v14

    // v8~15 O[0~3]
    IDCT32_SEXT32   v16, v18, v20, v22, v24, v25, v26, v27
    IDCT_MUL_1V4IMM     89, 75, 50, 18, v16, v8, v10, v12, v14
    IDCT_MULADD_1V4IMM  75, -18, -89, -50, v18, v8, v10, v12, v14
    IDCT_MULADD_1V4IMM  50, -89, 18, 75, v20, v8, v10, v12, v14
    IDCT_MULADD_1V4IMM  18, -50, 75, -89, v22, v8, v10, v12, v14

    // v16~v31  dst[0~7]
    SUMSUB_ABCD     v16, v30, v18, v28, v0, v8, v2, v10
    SUMSUB_ABCD     v20, v26, v22, v24, v4, v12, v6, v14
    vsetivli        zero, 8, e16, m1, ta, ma
    PBFIM_STORE8REG8 \dst, 0, \shift, \strided
.endm

.macro PARTIAL_BUTTER_FLY_INVERSE_4 src, dst, shift, strided, label
    vsetivli        zero, 8, e16, m1, ta, ma
    addi            t1, \src, 16
    vle16.v         v8, (\src)
    vle16.v         v9, (t1)
    vslidedown.vi   v10, v8, 4
    vslidedown.vi   v11, v9, 4
    vsetivli        zero, 4, e32, m1, ta, ma
    IDCT32_SEXT32   v0, v2, v4, v6, v8, v9, v10, v11

    // v8~11:E[0~1] v12~15:O[0~1]
    IDCT_SUMSUB_SHIFT v8, v10, v0, v2, 6
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v4, v6, v12, v14

    // v16~v23  dst[0~3]
    SUMSUB_ABCD     v16, v22, v18, v20, v8, v12, v10, v14

    vsetivli        zero, 4, e16, mf2, ta, ma
    vnclip.wi       v4, v16, \shift
    vnclip.wi       v5, v18, \shift
    vnclip.wi       v6, v20, \shift
    vnclip.wi       v7, v22, \shift

    vmv.v.i         v0, 5
    TRANSPOSE_4x4   16, v8, v9, v10, v11, v4, v5, v6, v7
    vsetivli        zero, 4, e16, mf2, ta, ma
    slli            t2, \strided, 1
    add             t1, \dst, \strided
    add             t2, t2, \dst
    add             t3, t2, \strided
    vse16.v         v8, (\dst)
    vse16.v         v9, (t1)
    vse16.v         v10, (t2)
    vse16.v         v11, (t3)
.endm

.macro IDCT_N size
function PFX(idct\size\()_v)
.if \size == 32
    addi    sp, sp, -1024
    addi    sp, sp, -1024
.else
    addi    sp, sp, -(\size * \size * 2)
.endif
    mv      a7, sp
    slli    a2, a2, 1
    li      a6, (\size * 2)

    PARTIAL_BUTTER_FLY_INVERSE_\size a0, a7, 7, a6, lidct\size\()_1
    PARTIAL_BUTTER_FLY_INVERSE_\size a7, a1, (12 - (BIT_DEPTH - 8)), a2, lidct\size\()_2

.if \size == 32
    addi    sp, sp, 1024
    addi    sp, sp, 1024
.else
    addi    sp, sp, (\size * \size * 2)
.endif
    ret
endfunc
.endm

// void idct16(const int16_t* src, int16_t* dst, intptr_t dstStride)
IDCT_N 32
IDCT_N 16
IDCT_N 8
IDCT_N 4

.macro INVERSE_DST4 src, dst, shift, strided
    // load v0~3
    vsetivli        zero, 4, e16, mf2, ta, ma
    IDCT32_LOAD_LINE 8, \src, 0, 1, 2, 3, v5, v6, v7, v8
    vsetivli        zero, 4, e32, m1, ta, ma
    IDCT32_SEXT32   v0, v1, v2, v3, v5, v6, v7, v8

    li          t0, 29
    li          t1, 55
    li          t2, -29
    li          t3, 74

    // v4~7 c0~3
    vadd.vv     v4, v0, v2
    vadd.vv     v5, v2, v3
    vsub.vv     v6, v0, v3
    vmul.vx     v7, v1, t3

    // v8 10 12 14 dst[0~3]
    vmv.v.v     v8, v7
    vmv.v.v     v10, v7
    vmacc.vx    v8, t0, v4
    vmacc.vx    v8, t1, v5
    vmacc.vx    v10, t1, v6
    vmacc.vx    v10, t2, v5
    vadd.vv     v12, v0, v3
    vsub.vv     v12, v12, v2
    vmul.vx     v12, v12, t3
    vrsub.vi    v14, v7, 0
    vmacc.vx    v14, t1, v4
    vmacc.vx    v14, t0, v6

    slli            t2, \strided, 1
    add             t3, t2, \strided
    add             t1, \dst, \strided
    add             t2, t2, \dst
    add             t3, t3, \dst

    // store
    vsetivli        zero, 4, e16, mf2, ta, ma
    vmv.v.i         v0, 5
    vnclip.wi       v1, v8, \shift
    vnclip.wi       v2, v10, \shift
    vnclip.wi       v3, v12, \shift
    vnclip.wi       v4, v14, \shift

    TRANSPOSE_4x4   16, v5, v6, v7, v8, v1, v2, v3, v4
    vsetivli        zero, 4, e16, mf2, ta, ma
    vse16.v         v5, (\dst)
    vse16.v         v6, (t1)
    vse16.v         v7, (t2)
    vse16.v         v8, (t3)
.endm

function PFX(idst4_v)
    addi    sp, sp, -32
    slli    a2, a2, 1
    li      t5, 8

    INVERSE_DST4 a0, sp, 7, t5
    INVERSE_DST4 sp, a1, (12 - (BIT_DEPTH - 8)), a2

    addi    sp, sp, 32
    ret
endfunc

.macro TEMP_STORE2 r1, r2
    addi            sp, sp, -32 * 2
    addi            t1, sp, 32
    addi            a4, a4, 64
    vse32.v         \r1, (sp)
    vse32.v         \r2, (t1)
.endm

.macro TEMP_LOAD2 num, r1, r2
    addi            t1, sp, \num * 32
    addi            t2, sp, (\num + 1) * 32
    vle32.v         \r1, (t1)
    vle32.v         \r2, (t2)
.endm

.macro DCT_STORE_2L dst, strided, line1, line2, shift, m1, m2, s1, s2
    vnclip.wi       \m1, \s1, \shift
    vnclip.wi       \m2, \s2, \shift
    addi            t1, \dst, \strided * \line1
    addi            t2, \dst, \strided * \line2
    vse16.v         \m1, (t1)
    vse16.v         \m2, (t2)
.endm

.macro DCT32_STORE_2L dst, strided, line1, line2, shift, m1, m2, s1, s2
    vadd.vv         \s1, \s1, \m1
    vadd.vv         \s2, \s2, \m2
    vsetivli        zero, 8, e16, m1, ta, ma
    DCT_STORE_2L    \dst, \strided, \line1, \line2, \shift, \m1, \m2, \s1, \s2
.endm

.macro DCTS_STORE_4L dst, strided, l1, l2, l3, l4, shift, m1, m2, m3, m4, s1, s2, s3, s4
    vnsrl.wi        \m1, \s1, \shift
    vnsrl.wi        \m2, \s2, \shift
    vnsrl.wi        \m3, \s3, \shift
    vnsrl.wi        \m4, \s4, \shift
    addi            t1, \dst, \strided * \l1
    addi            t2, \dst, \strided * \l2
    addi            t3, \dst, \strided * \l3
    addi            t4, \dst, \strided * \l4
    vse16.v         \m1, (t1)
    vse16.v         \m2, (t2)
    vse16.v         \m3, (t3)
    vse16.v         \m4, (t4)
.endm

.macro DCT_STORE_4L dst, strided, l1, l2, l3, l4, shift, m1, m2, m3, m4, s1, s2, s3, s4
    vnclip.wi       \m1, \s1, \shift
    vnclip.wi       \m2, \s2, \shift
    vnclip.wi       \m3, \s3, \shift
    vnclip.wi       \m4, \s4, \shift
    addi            t1, \dst, \strided * \l1
    addi            t2, \dst, \strided * \l2
    addi            t3, \dst, \strided * \l3
    addi            t4, \dst, \strided * \l4
    vse16.v         \m1, (t1)
    vse16.v         \m2, (t2)
    vse16.v         \m3, (t3)
    vse16.v         \m4, (t4)
.endm

.macro DCT32_STORE_4L dst, strided, l1, l2, l3, l4, shift, m1, m2, m3, m4, s1, s2, s3, s4
    vadd.vv         \s1, \s1, \m1
    vadd.vv         \s2, \s2, \m2
    vadd.vv         \s3, \s3, \m3
    vadd.vv         \s4, \s4, \m4
    vsetivli        zero, 8, e16, m1, ta, ma
    DCT_STORE_4L    \dst, \strided, \l1, \l2, \l3, \l4, \shift, \m1, \m2, \m3, \m4, \s1, \s2, \s3, \s4
.endm

.macro DCT32_LOAD16_LINE src, strided, i1, i2, i3, i4, d1, d2, d3, d4
    addi            t1, \src, \i1 * 2
    addi            t2, \src, \i2 * 2
    addi            t3, \src, \i3 * 2
    addi            t4, \src, \i4 * 2
    vlse16.v        \d1, (t1), \strided
    vlse16.v        \d2, (t2), \strided
    vlse16.v        \d3, (t3), \strided
    vlse16.v        \d4, (t4), \strided
.endm

.macro DCT32_SUB d1, d2, d3, d4, src1, src2, src3, src4, src5, src6, src7, src8
    vsub.vv         \d1, \src1, \src2
    vsub.vv         \d2, \src3, \src4
    vsub.vv         \d3, \src5, \src6
    vsub.vv         \d4, \src7, \src8
.endm

.macro DCT32_ADD w, d1, d2, d3, d4, src1, src2, src3, src4, src5, src6, src7, src8
    \w\()add.vv     \d1, \src1, \src2
    \w\()add.vv     \d2, \src3, \src4
    \w\()add.vv     \d3, \src5, \src6
    \w\()add.vv     \d4, \src7, \src8
.endm

.macro DCT32_MUL_1V4IMM w, m1, m2, m3, m4, src, d1, d2, d3, d4
    li              t1, \m1
    li              t2, \m2
    li              t3, \m3
    li              t4, \m4
    \w\()mul.vx     \d1, \src, t1
    \w\()mul.vx     \d2, \src, t2
    \w\()mul.vx     \d3, \src, t3
    \w\()mul.vx     \d4, \src, t4
.endm

.macro DCT32_MUL_2x1V2IMM w, m1, m2, m3, m4, s1, s2, d1, d2
    li              t1, \m1
    li              t2, \m2
    li              t3, \m3
    li              t4, \m4
    \w\()mul.vx     \d1, \s1, t1
    \w\()mul.vx     \d2, \s1, t2
    \w\()macc.vx    \d1, t3, \s2
    \w\()macc.vx    \d2, t4, \s2
.endm

.macro DCT32_MULADD_1V4IMM w, m1, m2, m3, m4, src, d1, d2, d3, d4
    li              t1, \m1
    li              t2, \m2
    li              t3, \m3
    li              t4, \m4
    \w\()macc.vx    \d1, t1, \src
    \w\()macc.vx    \d2, t2, \src
    \w\()macc.vx    \d3, t3, \src
    \w\()macc.vx    \d4, t4, \src
.endm

.macro DCT32_MULADD_2x1V2IMM w, m1, m2, m3, m4, s1, s2, d1, d2
    li              t1, \m1
    li              t2, \m2
    li              t3, \m3
    li              t4, \m4
    \w\()macc.vx    \d1, t1, \s1
    \w\()macc.vx    \d2, t2, \s1
    \w\()macc.vx    \d1, t3, \s2
    \w\()macc.vx    \d2, t4, \s2
.endm


.macro PARTIAL_BUTTER_FLY_32 src, dst, shift, strided, label
    li              a5, 32
    slli            a6, a2, 3
\label:
    li              a4, 0
    vsetivli        zero, 8, e16, m1, ta, ma
    DCT32_LOAD16_LINE \src, \strided, 0, 3, 4, 7, v0, v1, v2, v3
    DCT32_LOAD16_LINE \src, \strided, 8, 11, 12, 15, v4, v5, v6, v7
    DCT32_LOAD16_LINE \src, \strided, 16, 19, 20, 23, v8, v9, v10, v11
    DCT32_LOAD16_LINE \src, \strided, 24, 27, 28, 31, v12, v13, v14, v15

    // O dstline 1 3 5 7
    DCT32_SUB  v16, v17, v18, v19, v0, v15, v1, v14, v2, v13, v3, v12
    DCT32_SUB  v28, v29, v30, v31, v4, v11, v5, v10, v6, v9, v7, v8
    DCT32_MUL_1V4IMM vw, 90, 90, 88, 85, v16, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 85, 46, -13, -67, v17, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 82, 22, -54, -90, v18, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 67, -54, -78, 38, v19, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 61, -73, -46, 82, v28, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 38, -88, 73, -4, v29, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 31, -78, 90, -61, v30, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 4, -13, 22, -31, v31, v20, v22, v24, v26
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_STORE      32, v20, v22, v24, v26
    vsetivli        zero, 8, e16, m1, ta, ma

    // O dstline 9 11 13 15
    DCT32_MUL_1V4IMM vw, 82, 78, 73, 67, v16, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, -90, -73, -22, 38, v17, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, -61, 13, 78, 85, v18, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 85, -22, -90,  4, v19, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 31, -88, -13, 90, v28, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, -67, 90, -46, -31, v29, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 4, 54, -88, 82, v30, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 38, -46, 54, -61, v31, v20, v22, v24, v26
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_STORE      32, v20, v22, v24, v26
    vsetivli        zero, 8, e16, m1, ta, ma

    // O dstline 17 19 21 23
    DCT32_MUL_1V4IMM vw, 61, 54, 46, 38, v16, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 82, 88, 54, -4, v17, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 31, -46, -90, -67, v18, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 90, 13, -88, -31, v19, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, -4, -90, 22, 85, v28, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 85, -78, 13, 61, v29, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, -38, -22, 73, -90, v30, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 67, -73, 78, -82, v31, v20, v22, v24, v26
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_STORE      32, v20, v22, v24, v26
    vsetivli        zero, 8, e16, m1, ta, ma

    // O dstline 25 27 29 31
    DCT32_MUL_1V4IMM vw, 31, 22, 13, 4, v16, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, -61, -90, -78, -31, v17, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 4, 73, 88, 38, v18, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 82, 46, -73, -61, v19, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, -38, -78, 54, 67, v28, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, -90, 54, 22, -82, v29, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 67, -13, -46, 85, v30, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 85, -88, 90, -90, v31, v20, v22, v24, v26
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_STORE      32, v20, v22, v24, v26
    vsetivli        zero, 8, e16, m1, ta, ma

    // E  0 3 4 7 8 11 12 15
    DCT32_ADD   vw, v16, v18, v20, v22, v0, v15, v1, v14, v2, v13, v3, v12
    DCT32_ADD   vw, v24, v26, v28, v30, v4, v11, v5, v10, v6, v9, v7, v8

    // EE/EO 0 3 4 7
    vsetivli        zero, 8, e32, m2, ta, ma
    SUMSUB_ABCD v0, v8, v2, v10, v16, v30, v18, v28
    SUMSUB_ABCD v4, v12, v6, v14, v20, v26, v22, v24

    // EO dst 2 6 10 14 18 22 26 30
    DCT32_MUL_1V4IMM v, 90, 87, 80, 70, v8, v16, v18, v20, v22
    DCT32_MUL_1V4IMM v, 57, 43, 25, 9, v8, v24, v26, v28, v30
    DCT32_MULADD_1V4IMM v, 70, -43, -87, 9, v10, v16, v18, v20, v22
    DCT32_MULADD_1V4IMM v, 90, 25, -80, -57, v10, v24, v26, v28, v30
    DCT32_MULADD_1V4IMM v, 57, -80, -25, 90, v12, v16, v18, v20, v22
    DCT32_MULADD_1V4IMM v, -9, -87, 43, 70, v12, v24, v26, v28, v30
    DCT32_MULADD_1V4IMM v, 9, -25, 43, -57, v14, v16, v18, v20, v22
    DCT32_MULADD_1V4IMM v, 70, -80, 87, -90, v14, v24, v26, v28, v30
    TEMP_STORE  32, v16, v18, v20, v22
    TEMP_STORE  32, v24, v26, v28, v30

    // EEE/EEO 0 3
    SUMSUB_ABCD v8, v12, v10, v14, v0, v6, v2, v4

    // EEO dst 4 12 20 28
    DCT32_MUL_1V4IMM v, 89, 75, 50, 18, v12, v0, v2, v4, v6
    DCT32_MULADD_1V4IMM v, 18, -50, 75, -89, v14, v0, v2, v4, v6
    TEMP_STORE  32, v0, v2, v4, v6

    // EEEE/EEEO 0
    vadd.vv         v16, v8, v10
    vsub.vv         v18, v8, v10
    TEMP_STORE2     v16, v18

    vsetivli        zero, 8, e16, m1, ta, ma
    DCT32_LOAD16_LINE \src, \strided, 1, 2, 5, 6, v0, v1, v2, v3
    DCT32_LOAD16_LINE \src, \strided, 9, 10, 13, 14, v4, v5, v6, v7
    DCT32_LOAD16_LINE \src, \strided, 17, 18, 21, 22, v8, v9, v10, v11
    DCT32_LOAD16_LINE \src, \strided, 25, 26, 29, 30, v12, v13, v14, v15

    // O dstline 1 3
    DCT32_SUB  v16, v17, v18, v19, v0, v15, v1, v14, v2, v13, v3, v12
    DCT32_SUB  v28, v29, v30, v31, v4, v11, v5, v10, v6, v9, v7, v8
    DCT32_MUL_2x1V2IMM vw, 90, 82, 88, 67, v16, v17, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 78, -4, 73, -31, v18, v19, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 54, -85, 46, -90, v28, v29, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 22, -61, 13, -38, v30, v31, v20, v22
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_LOAD2      26, v24, v26
    DCT32_STORE_2L  \dst, 64, 1, 3, \shift, v24, v26, v20, v22

    // O dstline 5 7
    DCT32_MUL_2x1V2IMM vw, 67, 46, 31, -13, v16, v17, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -82, -73, -90, -22, v18, v19, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -4, 88, 38, 54, v28, v29, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 85, -90, 61, -78, v30, v31, v20, v22
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_LOAD2      28, v24, v26
    DCT32_STORE_2L  \dst, 64, 5, 7, \shift, v24, v26, v20, v22

    // O dstline 9 11
    DCT32_MUL_2x1V2IMM vw, 22, -4, -54, -82, v16, v17, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 13, 85, 78, 67, v18, v19, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -46, -61, -90, 31, v28, v29, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 73, -38, 88, -90, v30, v31, v20, v22
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_LOAD2      22, v24, v26
    DCT32_STORE_2L  \dst, 64, 9, 11, \shift, v24, v26, v20, v22

    // O dstline 13 15
    DCT32_MUL_2x1V2IMM vw, -31, -54, -90, -78, v16, v17, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 67, -22, -38, -90, v18, v19, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 82, 13, 61, -88, v28, v29, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -4, 46, 85, -73, v30, v31, v20, v22
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_LOAD2      24, v24, v26
    DCT32_STORE_2L  \dst, 64, 13, 15, \shift, v24, v26, v20, v22

    // O dstline 17 19
    DCT32_MUL_2x1V2IMM vw, -73, -85, -46, -4, v16, v17, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -88, -61, -13, 82, v18, v19, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -90, 38, 22, 67, v28, v29, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -78, 90, 54, -31, v30, v31, v20, v22
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_LOAD2      18, v24, v26
    DCT32_STORE_2L  \dst, 64, 17, 19, \shift, v24, v26, v20, v22

    // O dstline 21 23
    DCT32_MUL_2x1V2IMM vw, -90, -88, 38, 73, v16, v17, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 31, 90, 61, -46, v18, v19, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 67, -78, -85, 13, v28, v29, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -82, 54, 4, 22, v30, v31, v20, v22
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_LOAD2      20, v24, v26
    DCT32_STORE_2L  \dst, 64, 21, 23, \shift, v24, v26, v20, v22

    // O dstline 25 27
    DCT32_MUL_2x1V2IMM vw, -78, -61, 90, 85, v16, v17, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 54, -38, -88, -4, v18, v19, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -22, 90, 73, -82, v28, v29, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -13, -31, -46, 67, v30, v31, v20, v22
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_LOAD2      14, v24, v26
    DCT32_STORE_2L  \dst, 64, 25, 27, \shift, v24, v26, v20, v22

    // O dstline 29 31
    DCT32_MUL_2x1V2IMM vw, -38, -13, 61, 22, v16, v17, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -90, -46, 85, 54, v18, v19, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -31, -73, 4, 78, v28, v29, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 67, -88, -82, 90, v30, v31, v20, v22
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_LOAD2      16, v24, v26
    DCT32_STORE_2L  \dst, 64, 29, 31, \shift, v24, v26, v20, v22

    // E  1 2 5 6 9 10 13 14
    DCT32_ADD   vw, v16, v18, v20, v22, v0, v15, v1, v14, v2, v13, v3, v12
    DCT32_ADD   vw, v24, v26, v28, v30, v4, v11, v5, v10, v6, v9, v7, v8

    // EE/EO 1 2 5 6
    vsetivli        zero, 8, e32, m2, ta, ma
    SUMSUB_ABCD v0, v8, v2, v10, v16, v30, v18, v28
    SUMSUB_ABCD v4, v12, v6, v14, v20, v26, v22, v24

    // EO dst 2 6 10 14 18 22 26 30
    DCT32_MUL_1V4IMM v, 87, 57,  9, -43, v8, v16, v18, v20, v22
    DCT32_MUL_1V4IMM v, -80, -90, -70, -25, v8, v24, v26, v28, v30
    DCT32_MULADD_1V4IMM v, 80,  9, -70, -87, v10, v16, v18, v20, v22
    DCT32_MULADD_1V4IMM v, -25, 57, 90, 43, v10, v24, v26, v28, v30
    DCT32_MULADD_1V4IMM v, 43, -90, 57, 25, v12, v16, v18, v20, v22
    DCT32_MULADD_1V4IMM v, -87, 70, 9, -80, v12, v24, v26, v28, v30
    DCT32_MULADD_1V4IMM v, 25, -70, 90, -80, v14, v16, v18, v20, v22
    DCT32_MULADD_1V4IMM v, 43, 9, -57, 87, v14, v24, v26, v28, v30
    TEMP_LOAD       32, 10, v8, v10, v12, v14
    DCT32_STORE_4L  \dst, 64, 2, 6, 10, 14, \shift, v8, v10, v12, v14, v16, v18, v20, v22
    vsetivli        zero, 8, e32, m2, ta, ma
    TEMP_LOAD       32, 6, v8, v10, v12, v14
    DCT32_STORE_4L  \dst, 64, 18, 22, 26, 30, \shift, v8, v10, v12, v14, v24, v26, v28, v30
    vsetivli        zero, 8, e32, m2, ta, ma

    // EEE/EEO 1 2
    SUMSUB_ABCD v8, v12, v10, v14, v0, v6, v2, v4

    // EEO dst 4 12 20 28
    DCT32_MUL_1V4IMM v, 75, -18, -89, -50, v12, v0, v2, v4, v6
    DCT32_MULADD_1V4IMM v, 50, -89, 18, 75, v14, v0, v2, v4, v6
    TEMP_LOAD       32, 2, v16, v18, v20, v22
    DCT32_STORE_4L  \dst, 64, 4, 12, 20, 28, \shift, v16, v18, v20, v22, v0, v2, v4, v6
    vsetivli        zero, 8, e32, m2, ta, ma

    // EEEE/EEEO 1
    vadd.vv         v16, v8, v10
    vsub.vv         v18, v8, v10
    TEMP_LOAD2      0, v20, v22

    // dst 0 16 8 24
    IDCT_SUMSUB_SHIFT v2, v4, v20, v16, 6
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v22, v18, v6, v8
    vsetivli        zero, 8, e16, m1, ta, ma
    DCT_STORE_4L \dst, 64, 0, 8, 16, 24, \shift, v10, v12, v14, v16, v2, v6, v4, v8

    add             sp, sp, a4
    addi            a5, a5, -8
    add             \src, \src, a6
    addi            \dst, \dst, 16
    bgtz            a5, \label
    addi            \dst, \dst, -64
.endm

.macro PARTIAL_BUTTER_FLY_16 src, dst, shift, strided, label
    li              a5, 16
    slli            a6, a2, 3
\label:
    vsetvli         t6, a5, e16, m1, ta, ma
    slli            t5, t6, 1
    DCT32_LOAD16_LINE \src, \strided, 0, 1, 2, 3, v0, v1, v2, v3
    DCT32_LOAD16_LINE \src, \strided, 4, 5, 6, 7, v4, v5, v6, v7
    DCT32_LOAD16_LINE \src, \strided, 8, 9, 10, 11, v8, v9, v10, v11
    DCT32_LOAD16_LINE \src, \strided, 12, 13, 14, 15, v12, v13, v14, v15

    // O 0~7
    DCT32_SUB  v16, v17, v18, v19, v0, v15, v1, v14, v2, v13, v3, v12
    DCT32_SUB  v28, v29, v30, v31, v4, v11, v5, v10, v6, v9, v7, v8

    // dst 1 3
    DCT32_MUL_2x1V2IMM vw, 90, 87, 87, 57, v16, v17, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 80, 9, 70, -43, v18, v19, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 57, -80, 43, -90, v28, v29, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 25, -70, 9, -25, v30, v31, v20, v22
    DCT_STORE_2L  \dst, 32, 1, 3, \shift, v24, v26, v20, v22

    // dst 5 7
    DCT32_MUL_2x1V2IMM vw, 80, 70, 9, -43, v16, v17, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -70, -87, -87, 9, v18, v19, v20, v22
    DCT32_MULADD_2x1V2IMM vw, -25, 90, 57, 25, v28, v29, v20, v22
    DCT32_MULADD_2x1V2IMM vw, 90, -80, 43, -57, v30, v31, v20, v22
    DCT_STORE_2L  \dst, 32, 5, 7, \shift, v24, v26, v20, v22

    // dst 9 11 13 15
    DCT32_MUL_1V4IMM vw, 57, 43, 25, 9, v16, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, -80, -90, -70, -25, v17, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, -25, 57, 90, 43, v18, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 90, 25, -80, -57, v19, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, -9, -87, 43, 70, v28, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, -87, 70, 9, -80, v29, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 43, 9, -57, 87, v30, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 70, -80, 87, -90, v31, v20, v22, v24, v26
    DCT_STORE_4L \dst, 32, 9, 11, 13, 15, \shift, v16, v18, v28, v30, v20, v22, v24, v26

    // E  0~7
    DCT32_ADD   vw, v16, v18, v20, v22, v0, v15, v1, v14, v2, v13, v3, v12
    DCT32_ADD   vw, v24, v26, v28, v30, v4, v11, v5, v10, v6, v9, v7, v8

    // EE/EO 0~3
    vsetvli         zero, t6, e32, m2, ta, ma
    SUMSUB_ABCD v0, v8, v2, v10, v16, v30, v18, v28
    SUMSUB_ABCD v4, v12, v6, v14, v20, v26, v22, v24

    // dst 2 6 10 14
    DCT32_MUL_1V4IMM v, 89, 75, 50, 18, v8, v16, v18, v20, v22
    DCT32_MULADD_1V4IMM v, 75, -18, -89, -50, v10, v16, v18, v20, v22
    DCT32_MULADD_1V4IMM v, 50, -89, 18, 75, v12, v16, v18, v20, v22
    DCT32_MULADD_1V4IMM v, 18, -50, 75, -89, v14, v16, v18, v20, v22
    vsetvli         zero, t6, e16, m1, ta, ma
    DCT_STORE_4L \dst, 32, 2, 6, 10, 14, \shift, v24, v26, v28, v30, v16, v18, v20, v22
    vsetvli         zero, t6, e32, m2, ta, ma

    // EEE/EEO 1 2
    SUMSUB_ABCD v8, v12, v10, v14, v0, v6, v2, v4

    // dst 0 8 4 12
    IDCT_SUMSUB_SHIFT v16, v18, v8, v10, 6
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v12, v14, v20, v22
    vsetvli         zero, t6, e16, m1, ta, ma
    DCT_STORE_4L \dst, 32, 0, 4, 8, 12, \shift, v8, v10, v12, v14, v16, v20, v18, v22

    sub             a5, a5, t6
    add             \src, \src, a6
    add             \dst, \dst, t5
    bgtz            a5, \label
    addi            \dst, \dst, -32
.endm

.macro PARTIAL_BUTTER_FLY_8 src, dst, shift, strided, label
    li              t1, \shift - 1
    li              t2, 1
    sll             t3, t2, t1
    vsetivli        zero, 8, e32, m2, ta, ma
    vmv.v.x         v30, t3

    vsetivli        zero, 8, e16, m1, ta, ma
    DCT32_LOAD16_LINE \src, \strided, 0, 1, 2, 3, v0, v1, v2, v3
    DCT32_LOAD16_LINE \src, \strided, 4, 5, 6, 7, v4, v5, v6, v7

    // O 0~3
    DCT32_SUB  v16, v17, v18, v19, v0, v7, v1, v6, v2, v5, v3, v4

    // dst 1 3 5 7
    DCT32_MUL_1V4IMM vw, 89, 75, 50, 18, v16, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 75, -18, -89, -50, v17, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 50, -89, 18, 75, v18, v20, v22, v24, v26
    DCT32_MULADD_1V4IMM vw, 18, -50, 75, -89, v19, v20, v22, v24, v26
    vsetivli        zero, 8, e32, m2, ta, ma
    vadd.vv         v20, v20, v30
    vadd.vv         v22, v22, v30
    vadd.vv         v24, v24, v30
    vadd.vv         v26, v26, v30
    vsetivli        zero, 8, e16, m1, ta, ma
    DCTS_STORE_4L \dst, 16, 1, 3, 5, 7, \shift, v16, v18, v10, v12, v20, v22, v24, v26

    // E  0~3
    DCT32_ADD     vw, v16, v18, v20, v22, v0, v7, v1, v6, v2, v5, v3, v4

    // EE/EO 0~1
    vsetivli        zero, 8, e32, m2, ta, ma
    SUMSUB_ABCD v0, v4, v2, v6, v16, v22, v18, v20

    // dst 0 4 2 6
    IDCT_SUMSUB_SHIFT v20, v22, v0, v2, 6
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v4, v6, v24, v26
    vadd.vv         v20, v20, v30
    vadd.vv         v22, v22, v30
    vadd.vv         v24, v24, v30
    vadd.vv         v26, v26, v30
    vsetivli        zero, 8, e16, m1, ta, ma
    DCTS_STORE_4L \dst, 16, 0, 2, 4, 6, \shift, v16, v18, v10, v12, v20, v24, v22, v26
.endm

.macro PARTIAL_BUTTER_FLY_4 src, dst, shift, strided, label
    li              t1, \shift - 1
    li              t2, 1
    sll             t3, t2, t1
    vsetivli        zero, 4, e32, m1, ta, ma
    vmv.v.x         v30, t3

    vsetivli        zero, 4, e16, m1, ta, ma
    DCT32_LOAD16_LINE \src, \strided, 0, 1, 2, 3, v0, v1, v2, v3

    // E O 0~1
    vwadd.vv        v4, v0, v3
    vwsub.vv        v8, v0, v3
    vwadd.vv        v6, v1, v2
    vwsub.vv        v10, v1, v2

    // dst 0~3
    vsetivli        zero, 4, e32, m1, ta, ma
    IDCT_SUMSUB_SHIFT v20, v22, v4, v6, 6
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v8, v10, v24, v26
    vadd.vv         v20, v20, v30
    vadd.vv         v22, v22, v30
    vadd.vv         v24, v24, v30
    vadd.vv         v26, v26, v30
    vsetivli        zero, 4, e16, m1, ta, ma
    DCTS_STORE_4L \dst, 8, 0, 1, 2, 3, \shift, v16, v18, v10, v12, v20, v24, v22, v26
.endm

.macro DCT_N size, shiftbase
function PFX(dct\size\()_v)
.if \size == 32
    addi    sp, sp, -1024
    addi    sp, sp, -1024
.else
    addi    sp, sp, -(\size * \size * 2)
.endif
    mv      a7, sp
    slli    a2, a2, 1

    PARTIAL_BUTTER_FLY_\size a0, a7, (\shiftbase + BIT_DEPTH - 8), a2, ldct\size\()_1
    li      a2, (\size * 2)
    PARTIAL_BUTTER_FLY_\size a7, a1, (\shiftbase + 7), a2, ldct\size\()_2

.if \size == 32
    addi    sp, sp, 1024
    addi    sp, sp, 1024
.else
    addi    sp, sp, (\size * \size * 2)
.endif
    ret
endfunc
.endm

//void dct16_neon(const int16_t *src, int16_t *dst, intptr_t srcStride)
DCT_N 32 4
DCT_N 16 3
DCT_N 8 2
DCT_N 4 1

.macro FASTFORWARD_DST4 src, dst, shift, srcStride
    // load v0~3 src0~4
    vsetivli    zero, 4, e16, m1, ta, ma
    addi        t1, \src, 0
    addi        t2, \src, 2
    addi        t3, \src, 4
    addi        t4, \src, 6
    addi        \src, \src, 8
    vlse16.v    v28, (t1), \srcStride
    vlse16.v    v29, (t2), \srcStride
    vlse16.v    v30, (t3), \srcStride
    vlse16.v    v31, (t4), \srcStride

    vsetivli    zero, 4, e32, m1, ta, ma
    vsext.vf2   v0, v28
    vsext.vf2   v1, v29
    vsext.vf2   v2, v30
    vsext.vf2   v3, v31
    addi        \src, \src, -8

    // v4~7 c0~3
    vadd.vv     v4, v0, v3
    vadd.vv     v5, v1, v3
    vsub.vv     v6, v0, v1
    li          t1, 74
    vmul.vx     v7, v2, t1

    //v8 10 12 14 dst 0 1 2 3
    vmv.v.v     v8, v7
    vmv.v.v     v14, v7
    li          t1, 29
    li          t2, 55
    li          t3, 74
    li          t4, -29
    vmacc.vx    v8, t1, v4
    vmacc.vx    v8, t2, v5
    vadd.vv     v10, v0, v1
    vsub.vv     v10, v10, v3
    vmul.vx     v10, v10, t3
    vmul.vx     v12, v6, t1
    vmacc.vx    v12, t2, v4
    vsub.vv     v12, v12, v7
    vmacc.vx    v14, t2, v6
    vmacc.vx    v14, t4, v5

    vsetivli    zero, 4, e16, m1, ta, ma
    vnclip.wi   v20, v8, \shift
    vnclip.wi   v21, v10, \shift
    vnclip.wi   v22, v12, \shift
    vnclip.wi   v23, v14, \shift
    addi        t1, \dst, 8 * 0
    addi        t2, \dst, 8 * 1
    addi        t3, \dst, 8 * 2
    addi        t4, \dst, 8 * 3
    vse16.v     v20, (t1)
    vse16.v     v21, (t2)
    vse16.v     v22, (t3)
    vse16.v     v23, (t4)
.endm

//void dst4_neon(const int16_t *src, int16_t *dst, intptr_t srcStride)
function PFX(dst4_v)
    addi        sp, sp, -32
    slli        a2, a2, 1
    mv          a5, sp
    li          a3, 8

    FASTFORWARD_DST4 a0, a5, (1 + BIT_DEPTH - 8), a2
    FASTFORWARD_DST4 a5, a1, 8, a3

    addi        sp, sp, 32
    ret
endfunc

//void denoiseDct_c(int16_t* dctCoef, uint32_t* resSum, const uint16_t* offset, int numCoeff)
function PFX(denoiseDct_v)
ldenoiseDct:
    vsetvli         t0, a3, e16, m1, ta, ma
    slli            t5, t0, 1
    slli            t6, t0, 2
    vle16.v         v0, (a0)
    vle16.v         v1, (a2)

    // v2 level
    vsetvli         zero, t0, e32, m2, ta, ma
    vsext.vf2       v2, v0
    // v4 sign
    vsra.vi         v4, v2, 16
    vsra.vi         v4, v4, 15
    vadd.vv         v2, v2, v4
    vxor.vv         v2, v2, v4
    //v6 resSum
    vle32.v         v6, (a1)
    vadd.vv         v6, v6, v2
    vse32.v         v6, (a1)

    //v8 offset
    vzext.vf2       v8, v1
    vsub.vv         v2, v2, v8

    //v12 dctCoef
    vmsge.vi        v0, v2, 0
    vxor.vv         v10, v2, v4
    vsub.vv         v10, v10, v4
    vmv.v.i         v12, 0
    vsetvli         zero, t0, e32, m2, ta, mu
    vadd.vv         v12, v12, v10, v0.t
    vsetvli         zero, t0, e16, m1, ta, ma
    vnsra.wi        v14, v12, 0
    vse16.v         v14, (a0)

    sub             a3, a3, t0
    add             a0, a0, t5
    add             a1, a1, t6
    add             a2, a2, t5
    bgtz            a3, ldenoiseDct
    ret
endfunc

// void nonPsyRdoQuant_c(int16_t *m_resiDctCoeff, int64_t *costUncoded, int64_t *totalUncodedCost, int64_t *totalRdCost, uint32_t blkPos)
.macro NONPSYRDOQUANT_N log2TrSize
function PFX(nonPsyRdoQuant\log2TrSize\()_v)
    li              t0, 1
    slli            t0, t0, \log2TrSize + 1
    slli            a4, a4, 1
    add             t1, a0, a4
    add             t2, t1, t0
    add             t3, t2, t0
    add             t4, t3, t0
    vsetivli        zero, 4, e16, m1, ta, ma
    // v1~4 signCoef
    vle16.v         v1, (t1)
    vle16.v         v2, (t2)
    vle16.v         v3, (t3)
    vle16.v         v4, (t4)
    vwmul.vv        v6, v1, v1
    vwmul.vv        v8, v2, v2
    vwmul.vv        v10, v3, v3
    vwmul.vv        v12, v4, v4
    vsetivli        zero, 4, e64, m2, ta, ma
    vsext.vf2       v20, v6
    vsext.vf2       v22, v8
    vsext.vf2       v24, v10
    vsext.vf2       v26, v12
.if 2 * BIT_DEPTH + 2 * \log2TrSize - 15 > 15
    li              t5, 2 * BIT_DEPTH + 2 * \log2TrSize - 15
    vmv.v.x         v2, t5
    vsll.vv         v20, v20, v2
    vsll.vv         v22, v22, v2
    vsll.vv         v24, v24, v2
    vsll.vv         v26, v26, v2
.else
    vsll.vi         v20, v20, 2 * BIT_DEPTH + 2 * \log2TrSize - 15
    vsll.vi         v22, v22, 2 * BIT_DEPTH + 2 * \log2TrSize - 15
    vsll.vi         v24, v24, 2 * BIT_DEPTH + 2 * \log2TrSize - 15
    vsll.vi         v26, v26, 2 * BIT_DEPTH + 2 * \log2TrSize - 15
.endif
    slli            a4, a4, 2
    slli            t0, t0, 2
    add             t1, a1, a4
    add             t2, t1, t0
    add             t3, t2, t0
    add             t4, t3, t0
    vse64.v         v20, (t1)
    vse64.v         v22, (t2)
    vse64.v         v24, (t3)
    vse64.v         v26, (t4)
    vadd.vv         v20, v20, v22
    vadd.vv         v20, v20, v24
    vadd.vv         v20, v20, v26
    vmv.v.i         v2, 0
    vredsum.vs      v20, v20, v2
    vmv.x.s         t1, v20
    ld              t2, (a2)
    ld              t3, (a3)
    add             t2, t2, t1
    add             t3, t3, t1
    sd              t2, (a2)
    sd              t3, (a3)

    ret
endfunc
.endm

NONPSYRDOQUANT_N 2
NONPSYRDOQUANT_N 3
NONPSYRDOQUANT_N 4
NONPSYRDOQUANT_N 5

// void psyRdoQuant_c(int16_t *m_resiDctCoeff, int16_t *m_fencDctCoeff, int64_t *costUncoded, int64_t *totalUncodedCost, int64_t *totalRdCost, int64_t *psyScale, uint32_t blkPos)
.macro PSYRDOQUANT_N log2TrSize
function PFX(PsyRdoQuant\log2TrSize\()_v)
    li              t0, 1
    slli            t0, t0, \log2TrSize + 1
    slli            a6, a6, 1
    vsetivli        zero, 4, e16, m1, ta, ma
    // v even(20~26) int16 signCoef
    add             t1, a0, a6
    add             t2, t1, t0
    add             t3, t2, t0
    add             t4, t3, t0
    vle16.v         v1, (t1)
    vle16.v         v2, (t2)
    vle16.v         v3, (t3)
    vle16.v         v4, (t4)

    // v even(12~18) int16 predictedCoef
    sub             t5, a1, a0
    add             t1, t1, t5
    add             t2, t2, t5
    add             t3, t3, t5
    add             t4, t4, t5
    vle16.v         v5, (t1)
    vle16.v         v6, (t2)
    vle16.v         v7, (t3)
    vle16.v         v8, (t4)
    vwsub.vv        v12, v5, v1
    vwsub.vv        v14, v6, v2
    vwsub.vv        v16, v7, v3
    vwsub.vv        v18, v8, v4

    vwmul.vv        v20, v1, v1
    vwmul.vv        v22, v2, v2
    vwmul.vv        v24, v3, v3
    vwmul.vv        v26, v4, v4

    // v0~7 signCoef v8~15 predictedCoef
    vsetivli        zero, 4, e64, m2, ta, ma
    vsext.vf2       v0, v20
    vsext.vf2       v2, v22
    vsext.vf2       v4, v24
    vsext.vf2       v6, v26
    vsext.vf2       v8, v12
    vsext.vf2       v10, v14
    vsext.vf2       v12, v16
    vsext.vf2       v14, v18

.if 2 * BIT_DEPTH + 2 * \log2TrSize - 15 > 15
    li              t5, 2 * BIT_DEPTH + 2 * \log2TrSize - 15
    vsetivli        zero, 4, e64, m2, ta, ma
    vmv.v.x         v16, t5
    vsll.vv         v0, v0, v16
    vsll.vv         v2, v2, v16
    vsll.vv         v4, v4, v16
    vsll.vv         v6, v6, v16
.else
    vsll.vi         v0, v0, 2 * BIT_DEPTH + 2 * \log2TrSize - 15
    vsll.vi         v2, v2, 2 * BIT_DEPTH + 2 * \log2TrSize - 15
    vsll.vi         v4, v4, 2 * BIT_DEPTH + 2 * \log2TrSize - 15
    vsll.vi         v6, v6, 2 * BIT_DEPTH + 2 * \log2TrSize - 15
.endif

    ld              t1, (a5)
    vmv.v.x         v16, t1
    vmul.vv         v8, v8, v16
    vmul.vv         v10, v10, v16
    vmul.vv         v12, v12, v16
    vmul.vv         v14, v14, v16

    li              t1, 2 * (15 - BIT_DEPTH - \log2TrSize) + 1
    blez            t1, lPsyRdoQuant\log2TrSize\()1
    vmv.v.x         v16, t1
    vsra.vv         v8, v8, v16
    vsra.vv         v10, v10, v16
    vsra.vv         v12, v12, v16
    vsra.vv         v14, v14, v16

lPsyRdoQuant\log2TrSize\()1:
    vsub.vv         v0, v0, v8
    vsub.vv         v2, v2, v10
    vsub.vv         v4, v4, v12
    vsub.vv         v6, v6, v14

    slli            t0, t0, 2
    slli            a6, a6, 2
    add             t1, a2, a6
    add             t2, t1, t0
    add             t3, t2, t0
    add             t4, t3, t0
    vse64.v         v0, (t1)
    vse64.v         v2, (t2)
    vse64.v         v4, (t3)
    vse64.v         v6, (t4)

    vadd.vv         v0, v0, v2
    vadd.vv         v0, v0, v4
    vadd.vv         v0, v0, v6
    vmv.v.i         v2, 0
    vredsum.vs      v0, v0, v2 
    vmv.x.s         t1, v0
    ld              t2, (a3)
    ld              t3, (a4)
    add             t2, t2, t1
    add             t3, t3, t1
    sd              t2, (a3)
    sd              t3, (a4)

    ret
endfunc
.endm

PSYRDOQUANT_N 2
PSYRDOQUANT_N 3
PSYRDOQUANT_N 4
PSYRDOQUANT_N 5

