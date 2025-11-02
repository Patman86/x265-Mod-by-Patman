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

.macro IDCT_LOAD4xN_EXT32 size, src, str1, str2, str3, str4, m1, m2, m3, m4, d1, d2, d3, d4
    vsetivli        zero, 4, e16, m1, ta, ma
    addi            t1, \src, (\str1 * \size * 2)
    addi            t2, \src, (\str2 * \size * 2)
    addi            t3, \src, (\str3 * \size * 2)
    addi            t4, \src, (\str4 * \size * 2)
    vle16.v         \m1, (t1)
    vle16.v         \m2, (t2)
    vle16.v         \m3, (t3)
    vle16.v         \m4, (t4)

    vsetivli        zero, 4, e32, m1, ta, ma
    vsext.vf2       \d1, \m1
    vsext.vf2       \d2, \m2
    vsext.vf2       \d3, \m3
    vsext.vf2       \d4, \m4
.endm

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

// void partialButterflyInverse4(const int16_t* src)
function PFX(partialButterflyInverse4), export=0
    // load v1~4
    IDCT_LOAD4xN_EXT32 4, a0, 0, 2, 1, 3, v5, v6, v7, v8, v1, v2, v3, v4

    // v10~11:E[0~1]
    IDCT_SUMSUB_SHIFT v10, v11, v1, v2, 6

    // v8~9:O[0~1]
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v3, v4, v8, v9

    // v 2 4 6 8  dst[0~3]
    SUMSUB_ABCD     v2, v8, v4, v6, v10, v8, v11, v9

    ret
endfunc

// void partialButterflyInverse8(const int16_t* src)
function PFX(partialButterflyInverse8), export=0
    // load v1~4
    IDCT_LOAD4xN_EXT32 8, a0, 0, 4, 2, 6, v5, v6, v7, v8, v1, v2, v3, v4

    // v5~6:EE[0~1]
    IDCT_SUMSUB_SHIFT v5, v6, v1, v2, 6

    // v8~9:EO[0~1]
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v3, v4, v8, v9

    // v17~20:E[0~3]
    SUMSUB_ABCD v17, v20, v18, v19, v5, v8, v6, v9

    // load v7~10
    IDCT_LOAD4xN_EXT32 8, a0, 1, 3, 5, 7, v13, v14, v15, v16, v7, v8, v9, v10

    // v21~24 O[0~3]
    IDCT_MUL_1V4IMM     89, 75, 50, 18, v7, v21, v22, v23, v24
    IDCT_MULADD_1V4IMM  75, -18, -89, -50, v8, v21, v22, v23, v24
    IDCT_MULADD_1V4IMM  50, -89, 18, 75, v9, v21, v22, v23, v24
    IDCT_MULADD_1V4IMM  18, -50, 75, -89, v10, v21, v22, v23, v24

    // v even(2~16)  dst[0~7]
    SUMSUB_ABCD     v2, v16, v4, v14, v17, v21, v18, v22
    SUMSUB_ABCD     v6, v12, v8, v10, v19, v23, v20, v24

    ret
endfunc

// void partialButterflyInverse16(const int16_t* src)
function PFX(partialButterflyInverse16), export=0
    // load v1~4
    IDCT_LOAD4xN_EXT32 16, a0, 0, 8, 4, 12, v5, v6, v7, v8, v1, v2, v3, v4

    // v5~6:EEE[0~1]
    IDCT_SUMSUB_SHIFT v5, v6, v1, v2, 6

    // v8~9:EEO[0~1]
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v3, v4, v8, v9

    // v1~v4:EE[0~3]
    SUMSUB_ABCD     v1, v4, v2, v3, v5, v8, v6, v9

    // load v7~10
    IDCT_LOAD4xN_EXT32 16, a0, 2, 6, 10, 14, v13, v14, v15, v16, v7, v8, v9, v10

    // v13~16:EO[0~3]
    IDCT_MUL_1V4IMM     89, 75, 50, 18, v7, v13, v14, v15, v16
    IDCT_MULADD_1V4IMM  75, -18, -89, -50, v8, v13, v14, v15, v16
    IDCT_MULADD_1V4IMM  50, -89, 18, 75, v9, v13, v14, v15, v16
    IDCT_MULADD_1V4IMM  18, -50, 75, -89, v10, v13, v14, v15, v16

    // v7,9,11,17,19,21,23,25:E[0~7]
    SUMSUB_ABCD     v7, v25, v9, v23, v1, v13, v2, v14
    SUMSUB_ABCD     v11, v21, v17, v19, v3, v15, v4, v16

    // load v2, 4, 6, 8, 10, 12, 14, 16
    IDCT_LOAD4xN_EXT32 16, a0, 1, 3, 5, 7, v18, v20, v22, v24, v2, v4, v6, v8
    IDCT_LOAD4xN_EXT32 16, a0, 9, 11, 13, 15, v18, v20, v22, v24, v10, v12, v14, v16

    // v1,3,5,13,15,27,29,31 O[0~7]
    IDCT_MUL_1V4IMM      90, 87, 80, 70, v2, v1, v3, v5, v13
    IDCT_MUL_1V4IMM      57, 43, 25, 9, v2, v15, v27, v29, v31
    IDCT_MULADD_1V4IMM  87, 57, 9, -43, v4, v1, v3, v5, v13
    IDCT_MULADD_1V4IMM  -80, -90, -70, -25, v4, v15, v27, v29, v31
    IDCT_MULADD_1V4IMM  80, 9, -70, -87, v6, v1, v3, v5, v13
    IDCT_MULADD_1V4IMM  -25, 57, 90, 43, v6, v15, v27, v29, v31
    IDCT_MULADD_1V4IMM  70, -43, -87, 9, v8, v1, v3, v5, v13
    IDCT_MULADD_1V4IMM  90, 25, -80, -57, v8, v15, v27, v29, v31
    IDCT_MULADD_1V4IMM  57, -80, -25, 90, v10, v1, v3, v5, v13
    IDCT_MULADD_1V4IMM  -9, -87, 43, 70, v10, v15, v27, v29, v31
    IDCT_MULADD_1V4IMM  43, -90, 57, 25, v12, v1, v3, v5, v13
    IDCT_MULADD_1V4IMM  -87, 70, 9, -80, v12, v15, v27, v29, v31
    IDCT_MULADD_1V4IMM  25, -70, 90, -80, v14, v1, v3, v5, v13
    IDCT_MULADD_1V4IMM  43, 9, -57, 87, v14, v15, v27, v29, v31
    IDCT_MULADD_1V4IMM  9, -25, 43, -57, v16, v1, v3, v5, v13
    IDCT_MULADD_1V4IMM  70, -80, 87, -90, v16, v15, v27, v29, v31

    // v even(2~30),0 dst[0~15]
    SUMSUB_ABCD     v2, v0, v4, v30, v7, v1, v9, v3
    SUMSUB_ABCD     v6, v28, v8, v26, v11, v5, v17, v13
    SUMSUB_ABCD     v10, v24, v12, v22, v19, v15, v21, v27
    SUMSUB_ABCD     v14, v20, v16, v18, v23, v29, v25, v31
    ret
endfunc

.macro PBFI32_STORE8REG dst, off1, off2, shift, strided
    addi            t0, \dst, \off1
    addi            t1, \dst, \off2
    vsetivli        zero, 4, e16, m1, ta, ma
    vnclip.wi       v0, v16, \shift
    vnclip.wi       v1, v18, \shift
    vnclip.wi       v2, v20, \shift
    vnclip.wi       v3, v22, \shift
    vnclip.wi       v16, v24, \shift
    vnclip.wi       v17, v26, \shift
    vnclip.wi       v18, v28, \shift
    vnclip.wi       v19, v30, \shift
    vssseg4e16.v    v0, (t0), \strided
    vssseg4e16.v    v16, (t1), \strided
.endm

// void partialButterflyInverse32(const int16_t* src)
.macro PARTIAL_BUTTER_FLY_INVERSE_32 src, dst, shift, strided
    // load v1~4
    IDCT_LOAD4xN_EXT32 32, \src, 0, 16, 8, 24, v5, v6, v7, v8, v1, v2, v3, v4

    // v5~6:EEEE[0~1]
    IDCT_SUMSUB_SHIFT v5, v6, v1, v2, 6

    // v8~9:EEEO[0~1]
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v3, v4, v8, v9

    // v1~v4:EEE[0~3]
    SUMSUB_ABCD     v1, v4, v2, v3, v5, v8, v6, v9

    // load v7~10
    IDCT_LOAD4xN_EXT32 32, \src, 4, 12, 20, 28, v13, v14, v15, v16, v7, v8, v9, v10

    // v12~15:EEO[0~3]
    IDCT_MUL_1V4IMM     89, 75, 50, 18, v7, v12, v13, v14, v15
    IDCT_MULADD_1V4IMM  75, -18, -89, -50, v8, v12, v13, v14, v15
    IDCT_MULADD_1V4IMM  50, -89, 18, 75, v9, v12, v13, v14, v15
    IDCT_MULADD_1V4IMM  18, -50, 75, -89, v10, v12, v13, v14, v15

    // v16~23:EE[0~7]
    SUMSUB_ABCD     v16, v23, v17, v22, v1, v12, v2, v13
    SUMSUB_ABCD     v18, v21, v19, v20, v3, v14, v4, v15

    // load v9~15 7
    IDCT_LOAD4xN_EXT32 32, \src, 2, 6, 10, 14, v1, v2, v3, v4, v9, v10, v11, v12
    IDCT_LOAD4xN_EXT32 32, \src, 18, 22, 26, 30, v1, v2, v3, v4, v13, v14, v15, v7

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
    IDCT_LOAD4xN_EXT32 32, \src, 1, 3, 5, 7, v24, v25, v26, v27, v16, v18, v20, v22
    IDCT_MUL_1V4IMM     90, 90, 88, 85, v16, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  90, 82, 67, 46, v18, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  88, 67, 31, -13, v20, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  85, 46, -13, -67, v22, v17, v19, v21, v23

    IDCT_LOAD4xN_EXT32 32, \src, 9, 11, 13, 15, v24, v25, v26, v27, v16, v18, v20, v22
    IDCT_MULADD_1V4IMM  82, 22, -54, -90, v16, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  78, -4, -82, -73, v18, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  73, -31, -90, -22, v20, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  67, -54, -78, 38, v22, v17, v19, v21, v23

    IDCT_LOAD4xN_EXT32 32, \src, 17, 19, 21, 23, v24, v25, v26, v27, v16, v18, v20, v22
    IDCT_MULADD_1V4IMM  61, -73, -46, 82, v16, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  54, -85, -4, 88, v18, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  46, -90, 38, 54, v20, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  38, -88, 73, -4, v22, v17, v19, v21, v23

    IDCT_LOAD4xN_EXT32 32, \src, 25, 27, 29, 31, v24, v25, v26, v27, v16, v18, v20, v22
    IDCT_MULADD_1V4IMM  31, -78, 90, -61, v16, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  22, -61, 85, -90, v18, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  13, -38, 61, -78, v20, v17, v19, v21, v23
    IDCT_MULADD_1V4IMM  4, -13, 22, -31, v22, v17, v19, v21, v23

    //v16 18 20 22 24 26 28 30 dst0~3 28~31
    SUMSUB_ABCD v16, v30, v18, v28, v0, v17, v1, v19
    SUMSUB_ABCD v20, v26, v22, v24, v2, v21, v3, v23
    PBFI32_STORE8REG \dst, 0, 56, \shift, \strided

    // v0~3 O[4~7]
    IDCT_LOAD4xN_EXT32 32, \src, 1, 3, 5, 7, v20, v21, v22, v23, v16, v17, v18, v19
    IDCT_MUL_1V4IMM     82, 78, 73, 67, v16, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  22, -4, -31, -54, v17, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -54, -82, -90, -78, v18, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -90, -73, -22, 38, v19, v0, v1, v2, v3

    IDCT_LOAD4xN_EXT32 32, \src, 9, 11, 13, 15, v20, v21, v22, v23, v16, v17, v18, v19
    IDCT_MULADD_1V4IMM  -61, 13, 78, 85, v16, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  13, 85, 67, -22, v17, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  78, 67, -38, -90, v18, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  85, -22, -90, 4, v19, v0, v1, v2, v3

    IDCT_LOAD4xN_EXT32 32, \src, 17, 19, 21, 23, v20, v21, v22, v23, v16, v17, v18, v19
    IDCT_MULADD_1V4IMM  31, -88, -13, 90, v16, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -46, -61, 82, 13, v17, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -90, 31, 61, -88, v18, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -67, 90, -46, -31, v19, v0, v1, v2, v3

    IDCT_LOAD4xN_EXT32 32, \src, 25, 27, 29, 31, v20, v21, v22, v23, v16, v17, v18, v19
    IDCT_MULADD_1V4IMM  4, 54, -88, 82, v16, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  73, -38, -4, 46, v17, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  88, -90, 85, -73, v18, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  38, -46, 54, -61, v19, v0, v1, v2, v3

    //v16 18 20 22 24 26 28 30 dst4~7 24~27
    SUMSUB_ABCD v16, v30, v18, v28, v4, v0, v5, v1
    SUMSUB_ABCD v20, v26, v22, v24, v6, v2, v7, v3
    PBFI32_STORE8REG \dst, 8, 48, \shift, \strided

    // v0~7 O[8~15]
    IDCT_LOAD4xN_EXT32 32, \src, 1, 3, 5, 7, v20, v21, v22, v23, v16, v17, v18, v19
    IDCT_MUL_1V4IMM     61, 54, 46, 38, v16, v0, v1, v2, v3
    IDCT_MUL_1V4IMM     31, 22, 13, 4, v16, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  -73, -85, -90, -88, v17, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -78, -61, -38, -13, v17, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  -46, -4, 38, 73, v18, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  90, 85, 61, 22, v18, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  82, 88, 54, -4, v19, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -61, -90, -78, -31, v19, v4, v5, v6, v7

    IDCT_LOAD4xN_EXT32 32, \src, 9, 11, 13, 15, v20, v21, v22, v23, v16, v17, v18, v19
    IDCT_MULADD_1V4IMM  31, -46, -90, -67, v16, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  4, 73, 88, 38, v16, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  -88, -61, 31, 90, v17, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  54, -38, -90, -46, v17, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  -13, 82, 61, -46, v18, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -88, -4, 85, 54, v18, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  90, 13, -88, -31, v19, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  82, 46, -73, -61, v19, v4, v5, v6, v7

    IDCT_LOAD4xN_EXT32 32, \src, 17, 19, 21, 23, v20, v21, v22, v23, v16, v17, v18, v19
    IDCT_MULADD_1V4IMM  -4, -90, 22, 85, v16, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -38, -78, 54, 67, v16, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  -90, 38, 67, -78, v17, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -22, 90, -31, -73, v17, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  22, 67, -85, 13, v18, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  73, -82, 4, 78, v18, v4, v5, v6, v7
    IDCT_MULADD_1V4IMM  85, -78, 13, 61, v19, v0, v1, v2, v3
    IDCT_MULADD_1V4IMM  -90, 54, 22, -82, v19, v4, v5, v6, v7

    IDCT_LOAD4xN_EXT32 32, \src, 25, 27, 29, 31, v20, v21, v22, v23, v16, v17, v18, v19
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
    PBFI32_STORE8REG \dst, 16, 40, \shift, \strided

    //v16 18 20 22 24 26 28 30 dst12~15 16~19
    vsetivli        zero, 4, e32, m1, ta, ma
    SUMSUB_ABCD v16, v30, v18, v28, v12, v4, v13, v5
    SUMSUB_ABCD v20, v26, v22, v24, v14, v6, v15, v7
    PBFI32_STORE8REG \dst, 24, 32, \shift, \strided

.endm

.macro STORE_PARTIAL_BUTTERFLY_INVERSE size, dst, shift, strided
    // v1~16 dst[0~15]
    vsetivli        zero, 4, e16, m1, ta, ma
    vnclip.wi       v1, v2, \shift
    vnclip.wi       v2, v4, \shift
    vnclip.wi       v3, v6, \shift
    vnclip.wi       v4, v8, \shift
.if \size > 4
    vnclip.wi       v5, v10, \shift
    vnclip.wi       v6, v12, \shift
    vnclip.wi       v7, v14, \shift
    vnclip.wi       v8, v16, \shift
.endif
.if \size > 8
    vnclip.wi       v9,  v18, \shift
    vnclip.wi       v10, v20, \shift
    vnclip.wi       v11, v22, \shift
    vnclip.wi       v12, v24, \shift
    vnclip.wi       v13, v26, \shift
    vmv1r.v         v20, v0
    vnclip.wi       v14, v28, \shift
    vnclip.wi       v15, v30, \shift
    vnclip.wi       v16, v20, \shift
.endif

.if \size == 4
    vssseg4e16.v    v1, (\dst), \strided 
.elseif \size == 8
    vssseg8e16.v    v1, (\dst), \strided
.elseif \size == 16
    addi            t1, \dst, 16
    vssseg8e16.v    v1, (\dst), \strided
    vssseg8e16.v    v9, (t1), \strided
.endif
.endm

.macro PARTIAL_BUTTERFLY_INVERSE size, src, dst, shift, strided, label
    li              t5, (\size / 4)
    mv              a0, \src
    mv              a1, \dst
    mv              a3, \strided
    slli            a7, \strided, 2
\label:
.if \size <= 16
    jal             PFX(partialButterflyInverse\size\())
    STORE_PARTIAL_BUTTERFLY_INVERSE \size, a1, \shift, a3
.else
    PARTIAL_BUTTER_FLY_INVERSE_32 a0, a1, \shift, a3
.endif
    addi            a0, a0, 8
    add             a1, a1, a7
    addi            t5, t5, -1
    bgtz            t5, \label
.endm

.macro IDCT_N size
function PFX(idct\size\()_v)
    addi    sp, sp, -16
    sd      ra, 8(sp)
.if \size == 32
    addi    sp, sp, -1024
    addi    sp, sp, -1024
.else
    addi    sp, sp, -(\size * \size * 2)
.endif
    mv      a5, a1
    slli    a6, a2, 1
    li      t6, (\size * 2)

    PARTIAL_BUTTERFLY_INVERSE \size, a0, sp, 7, t6, lidct\size\()_1
    PARTIAL_BUTTERFLY_INVERSE \size, sp, a5, (12 - (BIT_DEPTH - 8)), a6, lidct\size\()_2

.if \size == 32
    addi    sp, sp, 1024
    addi    sp, sp, 1024
.else
    addi    sp, sp, (\size * \size * 2)
.endif
    ld      ra, 8(sp)
    addi    sp, sp, 16
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
    IDCT_LOAD4xN_EXT32 4, \src, 0, 1, 2, 3, v5, v6, v7, v8, v0, v1, v2, v3

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

    // store
    vsetivli        zero, 4, e16, m1, ta, ma
    vnclip.wi       v0, v8, \shift
    vnclip.wi       v1, v10, \shift
    vnclip.wi       v2, v12, \shift
    vnclip.wi       v3, v14, \shift
    vssseg4e16.v    v0, (\dst), \strided
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

.macro DCT_LOAD16_LINE src, strided, m1, m2, m3, m4, d1, d2, d3, d4
    vsetivli        zero, 4, e16, m1, ta, ma
    addi            t1, \src, 0
    addi            t2, \src, 2
    addi            t3, \src, 4
    addi            t4, \src, 6
    addi            \src, \src, 8
    vlse16.v        \m1, (t1), \strided
    vlse16.v        \m2, (t2), \strided
    vlse16.v        \m3, (t3), \strided
    vlse16.v        \m4, (t4), \strided

    vsetivli        zero, 4, e32, m1, ta, ma
    vsext.vf2       \d1, \m1
    vsext.vf2       \d2, \m2
    vsext.vf2       \d3, \m3
    vsext.vf2       \d4, \m4
.endm

.macro DCT_MULADD_4V4IMM first, m1, m2, m3, m4, s1, s2, s3, s4, dst
    li              t1, \m1
    li              t2, \m2
    li              t3, \m3
    li              t4, \m4
.if \first == 1
    vmul.vx         \dst, \s1, t1
.else
    vmacc.vx        \dst, t1, \s1
.endif
    vmacc.vx        \dst, t2, \s2
    vmacc.vx        \dst, t3, \s3
    vmacc.vx        \dst, t4, \s4
.endm

function PFX(partialButterfly4), export=0
    // load v0~3
    DCT_LOAD16_LINE a0, a2, v16, v17, v18, v19, v0, v1, v2, v3
    addi            a0, a0, -8

    // v9 11 E[0~1]  v5 7 O[0~1]
    SUMSUB_ABCD v9, v5, v11, v7, v0, v3, v1, v2

    // v2 6 4 8 dst 0 2 1 3
    IDCT_SUMSUB_SHIFT v2, v6, v9, v11, 6
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v5, v7, v4, v8

    ret
endfunc

function PFX(partialButterfly8), export=0
    // load v0~4 24 6 26
    DCT_LOAD16_LINE a0, a2, v16, v17, v18, v19, v0, v1, v2, v3
    DCT_LOAD16_LINE a0, a2, v16, v17, v18, v19, v4, v24, v6, v26
    addi            a0, a0, -16

    // v16~19 E[0~3]  v5 7 13 15 O[0~3]
    SUMSUB_ABCD v16, v5, v17, v7, v0, v26, v1, v6
    SUMSUB_ABCD v18, v13, v19, v15, v2, v24, v3, v4

    // v1 3 EE[0~1] v9 11 EO [0~1]
    SUMSUB_ABCD v1, v9, v3, v11, v16, v19, v17, v18

    // v2 10 6 14 dst 0 4 2 6
    IDCT_SUMSUB_SHIFT v2, v10, v1, v3, 6
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v9, v11, v6, v14

    // v4 8 12 16 dst 1 3 5 7
    DCT_MULADD_4V4IMM 1, 89, 75, 50, 18, v5, v7, v13, v15, v4
    DCT_MULADD_4V4IMM 1, 75, -18, -89, -50, v5, v7, v13, v15, v8
    DCT_MULADD_4V4IMM 1, 50, -89, 18, 75, v5, v7, v13, v15, v12
    DCT_MULADD_4V4IMM 1, 18, -50, 75, -89, v5, v7, v13, v15, v16

    ret
endfunc

function PFX(partialButterfly16), export=0
    // load v0~4 24 6 26 8~12 28 14 30
    DCT_LOAD16_LINE a0, a2, v16, v17, v18, v19, v0, v1, v2, v3
    DCT_LOAD16_LINE a0, a2, v16, v17, v18, v19, v4, v24, v6, v26
    DCT_LOAD16_LINE a0, a2, v16, v17, v18, v19, v8, v9, v10, v11
    DCT_LOAD16_LINE a0, a2, v16, v17, v18, v19, v12, v28, v14, v30
    addi            a0, a0, -32

    // v16~23 E[0~7]  v5 7 13 15 25 27 29 31 O[0~7]
    SUMSUB_ABCD v16, v5, v17, v7, v0, v30, v1, v14
    SUMSUB_ABCD v18, v13, v19, v15, v2, v28, v3, v12
    SUMSUB_ABCD v20, v25, v21, v27, v4, v11, v24, v10
    SUMSUB_ABCD v22, v29, v23, v31, v6, v9, v26, v8

    // v0~3 EE[0~3] v24 8 28 12 EO [0~3]
    SUMSUB_ABCD v0, v24, v1, v8, v16, v23, v17, v22
    SUMSUB_ABCD v2, v28, v3, v12, v18, v21, v19, v20

    // v16 4 EEE[0~1] V20 22 EEO[0~1]
    SUMSUB_ABCD v16, v20, v4, v22, v0, v3, v1, v2

    // v2 18 10 26 dst 0 8 4 12
    IDCT_SUMSUB_SHIFT v2, v18, v16, v4, 6
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v20, v22, v10, v26

    // v6 14 22 30 dst 2 6 10 14
    DCT_MULADD_4V4IMM 1, 89, 75, 50, 18, v24, v8, v28, v12, v6
    DCT_MULADD_4V4IMM 1, 75, -18, -89, -50, v24, v8, v28, v12, v14
    DCT_MULADD_4V4IMM 1, 50, -89, 18, 75, v24, v8, v28, v12, v22
    DCT_MULADD_4V4IMM 1, 18, -50, 75, -89, v24, v8, v28, v12, v30

    // v4 8 12 16 20 24 28 0 dst odd(1~15)
    DCT_MULADD_4V4IMM 1, 90, 87, 80, 70, v5, v7, v13, v15, v4
    DCT_MULADD_4V4IMM 0, 57, 43, 25, 9, v25, v27, v29, v31, v4
    DCT_MULADD_4V4IMM 1, 87, 57, 9, -43, v5, v7, v13, v15, v8
    DCT_MULADD_4V4IMM 0, -80, -90, -70, -25, v25, v27, v29, v31, v8
    DCT_MULADD_4V4IMM 1, 80, 9, -70, -87, v5, v7, v13, v15, v12
    DCT_MULADD_4V4IMM 0, -25, 57, 90, 43, v25, v27, v29, v31, v12
    DCT_MULADD_4V4IMM 1, 70, -43, -87, 9, v5, v7, v13, v15, v16
    DCT_MULADD_4V4IMM 0, 90, 25, -80, -57, v25, v27, v29, v31, v16
    DCT_MULADD_4V4IMM 1, 57, -80, -25, 90, v5, v7, v13, v15, v20
    DCT_MULADD_4V4IMM 0, -9, -87, 43, 70, v25, v27, v29, v31, v20
    DCT_MULADD_4V4IMM 1, 43, -90, 57, 25, v5, v7, v13, v15, v24
    DCT_MULADD_4V4IMM 0, -87, 70, 9, -80, v25, v27, v29, v31, v24
    DCT_MULADD_4V4IMM 1, 25, -70, 90, -80, v5, v7, v13, v15, v28
    DCT_MULADD_4V4IMM 0, 43,  9, -57, 87, v25, v27, v29, v31, v28
    DCT_MULADD_4V4IMM 1, 9, -25, 43, -57, v5, v7, v13, v15, v0
    DCT_MULADD_4V4IMM 0, 70, -80, 87, -90, v25, v27, v29, v31, v0

    ret
endfunc

.macro STORE_PARTIAL_BUTTERFLY size, dst, shift
    // v1~16 dst[0~15]
    vsetivli        zero, 4, e16, m1, ta, ma
    vnclip.wi       v1, v2, \shift
    vnclip.wi       v2, v4, \shift
    vnclip.wi       v3, v6, \shift
    vnclip.wi       v4, v8, \shift
    addi            t1, \dst, \size * 2 * 0
    addi            t2, \dst, \size * 2 * 1
    addi            t3, \dst, \size * 2 * 2
    addi            t4, \dst, \size * 2 * 3
    vse16.v         v1, (t1)
    vse16.v         v2, (t2)
    vse16.v         v3, (t3)
    vse16.v         v4, (t4)
.if \size > 4
    vnclip.wi       v5, v10, \shift
    vnclip.wi       v6, v12, \shift
    vnclip.wi       v7, v14, \shift
    vnclip.wi       v8, v16, \shift
    addi            t1, \dst, \size * 2 * 4
    addi            t2, \dst, \size * 2 * 5
    addi            t3, \dst, \size * 2 * 6
    addi            t4, \dst, \size * 2 * 7
    vse16.v         v5, (t1)
    vse16.v         v6, (t2)
    vse16.v         v7, (t3)
    vse16.v         v8, (t4)
.endif
.if \size > 8
    vnclip.wi       v9,  v18, \shift
    vnclip.wi       v10, v20, \shift
    vnclip.wi       v11, v22, \shift
    vnclip.wi       v12, v24, \shift
    addi            t1, \dst, \size * 2 * 8
    addi            t2, \dst, \size * 2 * 9
    addi            t3, \dst, \size * 2 * 10
    addi            t4, \dst, \size * 2 * 11
    vse16.v         v9, (t1)
    vse16.v         v10, (t2)
    vse16.v         v11, (t3)
    vse16.v         v12, (t4)

    vnclip.wi       v13, v26, \shift
    vmv1r.v         v20, v0
    vnclip.wi       v14, v28, \shift
    vnclip.wi       v15, v30, \shift
    vnclip.wi       v16, v20, \shift
    addi            t1, \dst, \size * 2 * 12
    addi            t2, \dst, \size * 2 * 13
    addi            t3, \dst, \size * 2 * 14
    addi            t4, \dst, \size * 2 * 15
    vse16.v         v13, (t1)
    vse16.v         v14, (t2)
    vse16.v         v15, (t3)
    vse16.v         v16, (t4)
.endif
.endm

.macro TEMP_LOADSTORE lors, dst, r1, r2, r3, r4
    li              t1, 128
    addi            t2, \dst, 4 * 128
    addi            t3, \dst, 8 * 128
    addi            t4, \dst, 12 * 128
    v\lors\()se32.v \r1, (\dst), t1
    v\lors\()se32.v \r2, (t2), t1
    v\lors\()se32.v \r3, (t3), t1
    v\lors\()se32.v \r4, (t4), t1
.endm

.macro DCT32_CALO dst, t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16
    DCT_MULADD_4V4IMM 1, \t1, \t2, \t3, \t4, v11, v27, v1, v31, \dst
    DCT_MULADD_4V4IMM 0, \t5, \t6, \t7, \t8, v15, v3, v29, v5, \dst
    DCT_MULADD_4V4IMM 0, \t9, \t10, \t11, \t12, v25, v7, v23, v17, \dst
    DCT_MULADD_4V4IMM 0, \t13, \t14, \t15, \t16, v9, v21, v13, v19, \dst
.endm

.macro DCT32_CALEO dst, t1,t2,t3,t4,t5,t6,t7,t8
    DCT_MULADD_4V4IMM 1, \t1, \t2, \t3, \t4, v1, v3, v5, v7, \dst
    DCT_MULADD_4V4IMM 0, \t5, \t6, \t7, \t8, v11, v10, v30, v8, \dst
.endm

.macro DCT32_STORE_2L dst, line1, line2, shift, m1, m2, s1, s2
    vsetivli        zero, 4, e16, m1, ta, ma
    vnclip.wi       \m1, \s1, \shift
    vnclip.wi       \m2, \s2, \shift
    addi            t1, \dst, 32 * 2 * \line1
    addi            t2, \dst, 32 * 2 * \line2
    vse16.v         \m1, (t1)
    vse16.v         \m2, (t2)
    vsetivli        zero, 4, e32, m1, ta, ma
.endm

.macro DCT_STORE_4L dst, strided, l1, l2, l3, l4, shift, m1, m2, m3, m4, s1, s2, s3, s4
    vsetivli        zero, 4, e16, m1, ta, ma
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
    vsetivli        zero, 4, e32, m1, ta, ma
.endm

.macro PARTIAL_BUTTER_FLY_32 src, dst, shift
    // load v0~31 src0~31
    DCT_LOAD16_LINE a0, a2, v28, v29, v30, v31, v0, v1, v2, v3
    DCT_LOAD16_LINE a0, a2, v28, v29, v30, v31, v4, v5, v6, v7
    DCT_LOAD16_LINE a0, a2, v28, v29, v30, v31, v8, v9, v10, v11
    DCT_LOAD16_LINE a0, a2, v28, v29, v30, v31, v12, v13, v14, v15
    DCT_LOAD16_LINE a0, a2, v28, v29, v30, v31, v16, v17, v18, v19
    DCT_LOAD16_LINE a0, a2, v28, v29, v30, v31, v20, v21, v22, v23
    DCT_LOAD16_LINE a0, a2, v28, v29, v30, v31, v24, v25, v26, v27
    TEMP_LOADSTORE  s, \dst, v4, v11, v20, v27
    DCT_LOAD16_LINE a0, a2, v4, v11, v20, v27, v28, v29, v30, v31
    addi            a0, a0, -64

    // 4 20 0 30 2 28 6 26 8 24 10 22 12 18 E[0~3 5~10 12~15]
    // 11 27 1 31 3 29 5 25 7 23 9 21 13 19 O[0~3 5~10 12~15]
    SUMSUB_ABCD v4, v11, v20, v27, v0, v31, v1, v30
    SUMSUB_ABCD v0, v1, v30, v31, v2, v29, v3, v28
    SUMSUB_ABCD v2, v3, v28, v29, v5, v26, v6, v25
    SUMSUB_ABCD v6, v5, v26, v25, v7, v24, v8, v23
    SUMSUB_ABCD v8, v7, v24, v23, v9, v22, v10, v21
    SUMSUB_ABCD v10, v9, v22, v21, v12, v19, v13, v18
    SUMSUB_ABCD v12, v13, v18, v19, v14, v17, v15, v16

    // 14 16 4 18 0 22 2 EE[0~3 5~7] 15 17 12 20 10 30 8 EO[0~3 5~7]
    SUMSUB_ABCD v14, v15, v16, v17, v4, v18, v20, v12
    SUMSUB_ABCD v4, v12, v18, v20, v0, v22, v30, v10
    SUMSUB_ABCD v0, v10, v22, v30, v2, v24, v28, v8
    SUMSUB_AB   v2, v8, v6, v26

    // load v6 24 26 28 src4 11 20 27 store EO[0~3]
    TEMP_LOADSTORE l, \dst, v6, v24, v26, v28
    TEMP_LOADSTORE s, \dst, v15, v17, v12, v20

    //v12 20 E[4 11] v15 17 O[4 11]
    SUMSUB_ABCD v12, v15, v20, v17, v6, v28, v24, v26

    // v 6 28 dst odd(1~31)
    DCT32_CALO v6, 90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38,31, 22, 13, 4
    DCT32_CALO v28, 90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13
    DCT32_STORE_2L \dst, 1, 3, \shift, v24, v26, v6, v28
    DCT32_CALO v6, 88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22
    DCT32_CALO v28, 85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31
    DCT32_STORE_2L \dst, 5, 7, \shift, v24, v26, v6, v28
    DCT32_CALO v6, 82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67,  4, 73, 88, 38
    DCT32_CALO v28, 78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46
    DCT32_STORE_2L \dst, 9, 11, \shift, v24, v26, v6, v28
    DCT32_CALO v6, 73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54
    DCT32_CALO v28, 67, -54, -78, 38, 85, -22, -90,  4, 90, 13, -88, -31, 82, 46, -73, -61
    DCT32_STORE_2L \dst, 13, 15, \shift, v24, v26, v6, v28
    DCT32_CALO v6, 61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67
    DCT32_CALO v28, 54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73
    DCT32_STORE_2L \dst, 17, 19, \shift, v24, v26, v6, v28
    DCT32_CALO v6, 46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82,  4, 78
    DCT32_CALO v28, 38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82
    DCT32_STORE_2L \dst, 21, 23, \shift, v24, v26, v6, v28
    DCT32_CALO v6, 31, -78, 90, -61,  4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85
    DCT32_CALO v28, 22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88
    DCT32_STORE_2L \dst, 25, 27, \shift, v24, v26, v6, v28
    DCT32_CALO v6, 13, -38, 61, -78, 88, -90, 85, -73, 54, -31,  4, 22, -46, 67, -82, 90
    DCT32_CALO v28, 4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90
    DCT32_STORE_2L \dst, 29, 31, \shift, v24, v26, v6, v28

    // load v1 3 5 7 EO[0~3]
    TEMP_LOADSTORE l, \dst, v1, v3, v5, v7

    // v9 EE4 v11 EO4
    SUMSUB_AB   v9, v11, v12, v20

    // v6 12 20 24 EEE[0~3]  v13 15 17 19 EEO[0~3]
    SUMSUB_ABCD v6, v13, v12, v15, v14, v2, v16, v22
    SUMSUB_ABCD v20, v17, v24, v19, v4, v0, v18, v9

    //v21 23 EEEE[0~1]  v25 27 EEEO[0~1]
    SUMSUB_ABCD v21, v25, v23, v27, v6, v24, v12, v20

    //v2 4 6 12 dst 0 16 8 24
    IDCT_SUMSUB_SHIFT v2, v4, v21, v23, 6
    IDCT_MULADD_2x2V2IMM 83, 36, 36, -83, v25, v27, v6, v12
    DCT_STORE_4L \dst, 64, 0, 16, 8, 24, \shift, v20, v21, v22, v23, v2, v4, v6, v12

    //v2 4 6 12 dst 4 12 20 28
    DCT_MULADD_4V4IMM 1, 89, 75, 50, 18, v13, v15, v17, v19, v2
    DCT_MULADD_4V4IMM 1, 75, -18, -89, -50, v13, v15, v17, v19, v4
    DCT_MULADD_4V4IMM 1, 50, -89, 18, 75, v13, v15, v17, v19, v6
    DCT_MULADD_4V4IMM 1, 18, -50, 75, -89, v13, v15, v17, v19, v12
    DCT_STORE_4L \dst, 64, 4, 12, 20, 28, \shift, v20, v21, v22, v23, v2, v4, v6, v12

    //v2 4 6 12 dst 2 6 10 14 18 22 26 30
    DCT32_CALEO v2, 90, 87, 80, 70, 57, 43, 25,  9
    DCT32_CALEO v4, 87, 57,  9, -43, -80, -90, -70, -25
    DCT32_CALEO v6, 80,  9, -70, -87, -25, 57, 90, 43
    DCT32_CALEO v12, 70, -43, -87,  9, 90, 25, -80, -57
    DCT_STORE_4L \dst, 64, 2, 6, 10, 14, \shift, v20, v21, v22, v23, v2, v4, v6, v12
    DCT32_CALEO v2, 57, -80, -25, 90, -9, -87, 43, 70
    DCT32_CALEO v4, 43, -90, 57, 25, -87, 70,  9, -80
    DCT32_CALEO v6, 25, -70, 90, -80, 43,  9, -57, 87
    DCT32_CALEO v12, 9, -25, 43, -57, 70, -80, 87, -90
    DCT_STORE_4L \dst, 64, 18, 22, 26, 30, \shift, v20, v21, v22, v23, v2, v4, v6, v12
.endm

.macro PARTIAL_BUTTERFLY size, src, dst, shift, strided, label
    li              t5, (\size / 4)
    li              t1, 4
    mv              a0, \src
    mv              a1, \dst
    mv              a2, \strided
    slli            a7, \strided, 2

\label:
.if \size <= 16
    jal             PFX(partialButterfly\size\())
    STORE_PARTIAL_BUTTERFLY \size, a1, \shift
.else
    PARTIAL_BUTTER_FLY_32 a0, a1, \shift
.endif
    add             a0, a0, a7
    addi            a1, a1, 8
    addi            t5, t5, -1
    bgtz            t5, \label
.endm

.macro DCT_N size, shiftbase
function PFX(dct\size\()_v)
    addi    sp, sp, -16
    sd      ra, 8(sp)
.if \size == 32
    addi    sp, sp, -1024
    addi    sp, sp, -1024
.else
    addi    sp, sp, -(\size * \size * 2)
.endif
    mv      a5, a1
    slli    a2, a2, 1
    li      a4, (\size * 2)

    PARTIAL_BUTTERFLY \size, a0, sp, (\shiftbase + BIT_DEPTH - 8), a2, ldct\size\()_1
    PARTIAL_BUTTERFLY \size, sp, a5, (\shiftbase + 7), a4, ldct\size\()_2

.if \size == 32
    addi    sp, sp, 1024
    addi    sp, sp, 1024
.else
    addi    sp, sp, (\size * \size * 2)
.endif
    ld      ra, 8(sp)
    addi    sp, sp, 16
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
    DCT_LOAD16_LINE \src, \srcStride, v28, v29, v30, v31, v0, v1, v2, v3
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

    DCT_STORE_4L \dst, 8, 0, 1, 2, 3, \shift, v20, v21, v22, v23, v8, v10, v12, v14
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

