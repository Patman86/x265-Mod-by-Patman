/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Jia Yuan <yuan.jia@sanechips.com.cn>
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

.section .rodata
.align 4

tbl_const_1to5_3to0:
    .byte 1, 2, 3, 4
    .byte 3, 2, 1, 0

.align 4
tbl_const_1to8_7to0:
    .byte 1, 2, 3, 4, 5, 6, 7, 8          // coefficient of (x+1)*TR
    .byte 7, 6, 5, 4, 3, 2, 1, 0          // coefficient of (blksize-1-x)*L[y]

.align 4
tbl_const_1to16_15to0:
    .byte 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    .byte 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

.section .text
.align 4


function PFX(intra_pred_planar4_rvv)
    li t0, 4
    vsetvli t0, t0, e8, mf2, ta, ma

    addi t1, a2, 1
    vle8.v v0, (t1)           # v0 = above[0..3]

    lbu t2, 5(a2)
    vmv.v.x v2, t2            # v2 = topRight_b_broadcast

    lbu t3, 13(a2)
    vsetvli zero, zero, e16, m1, ta, ma
    vmv.v.x v3, t3            # v3 = bottomLeft_h_broadcast

    la t2, tbl_const_1to5_3to0
    vsetvli zero, zero, e8, mf2, ta, ma
    vle8.v v4, (t2)           # v4 = [1,2,3,4] (x+1)
    addi t2, t2, 4
    vle8.v v5, (t2)           # v5 = [3,2,1,0] (3-x)

    vsetvli zero, zero, e16, m1, ta, ma
    vzext.vf2 v6, v0          # zero extension to 16 bits
    vsll.vi v6, v6, 2         # v6 = 4 * above[x]

    vsetvli zero, zero, e8, mf2, ta, ma
    vwmulu.vv v10, v4, v2      # v10 = (x+1) * topRight

    vsetvli zero, zero, e16, m1, ta, ma
    vadd.vv v6, v6, v10        # v6 = 4 * above[x] + (x+1) * topRight

    vzext.vf2 v12, v0          # v12 = above (zero extension to 16 bits)
    vsub.vv v8, v3, v12        # v8 = bottomLeft - above[x], as a correction

    csrwi  vxrm, 0
    li t4, 0                  # y = 0
    mv t5, a0                 # t5 = current dst
    mv t6, a1                 # t6 = dstStride

.rept 4
    add t1, a2, t4
    addi t1, t1, 9           # left = srcPix + 9 + y
    lbu t2, 0(t1)             # t2 = left[y]

    vsetvli zero, zero, e8, mf2, ta, ma
    vmv.v.x v9, t2            # v9 = left[y]_broadcast

    vsetvli zero, zero, e16, m1, ta, ma
    vadd.vv v6, v6, v8        # v6 = (blkSize - 1 -y) * above[x] + (x+1) * topRight + (y+1) * bottomLeft

    vmv.v.v v16, v6           # v16 = v6

    vsetvli zero, zero, e8, mf2, ta, ma
    vwmulu.vv v14, v5, v9                  # v14 = (3-x) * left[y]

    vsetvli zero, zero, e16, m1, ta, ma
    vadd.vv v16, v16, v14                  # v16 += (3-x) * left[y]

    vsetvli zero, zero, e8, mf2, ta, ma
    vnclipu.wi v18, v16, 3

    vse8.v v18, (t5)

    add     t5, t5, t6          # dst += dstStride
    addi    t4, t4, 1           # y++
.endr
    ret
endfunc


function PFX(intra_pred_planar8_rvv)
// Register map
// a0  = dst
// a1  = dstStride
// a2  = *srcPix
// a3  = left[x]
// a4  = tmp
// v0  = above[7:0]
// v1  = left[7:0]
// v2  = topRight = rep(above[blkSize])
// v3  = bottomLeft = rep(left[blkSize])
// v4  = const[1 2 3 4 5 6 7 8]
// v5  = const[7 6 5 4 3 2 1 0]

    li t0, 8
    vsetvli t0, t0, e8, mf2, ta, ma

    addi t1, a2, 1
    vle8.v v0, (t1)           # v0 = above[0..7]

    lbu t2, 9(a2)
    vmv.v.x v2, t2            # v2 = topRight_b_broadcast

    lbu t3, 25(a2)
    vsetvli zero, zero, e16, m1, ta, ma
    vmv.v.x v3, t3            # v3 = bottomLeft_h_broadcast

    la t2, tbl_const_1to8_7to0
    vsetvli zero, zero, e8, mf2, ta, ma
    vle8.v v4, (t2)           # v4 = [1,2,3,4,5,6,7,8] (x+1)
    addi t2, t2, 8
    vle8.v v5, (t2)           # v5 = [7,6,5,4,3,2,1,0] (7-x)

    vsetvli zero, zero, e16, m1, ta, ma
    vzext.vf2 v6, v0          # zero extension to 16 bits
    vsll.vi v6, v6, 3         # v6 = 8 * above[x]

    vsetvli zero, zero, e8, mf2, ta, ma
    vwmulu.vv v10, v4, v2      # v10 = (x+1) * topRight

    vsetvli zero, zero, e16, m1, ta, ma
    vadd.vv v6, v6, v10        # v6 = 8 * above[x] + (x+1) * topRight

    vzext.vf2 v12, v0          # v12 = above (zero extension to 16 bits)
    vsub.vv v8, v3, v12        # v8 = bottomLeft - above[x], as a correction

    csrwi  vxrm, 0
    li t3, 8
    li t4, 0                  # y = 0
    mv t5, a0                 # t5 = current dst
    mv t6, a1                 # t6 = dstStride

loop_y_8x8:
    add t1, a2, t4
    addi t1, t1, 17           # left = srcPix + 17 + y
    lbu t2, 0(t1)             # t2 = left[y]

    vsetvli zero, zero, e8, mf2, ta, ma
    vmv.v.x v9, t2            # v9 = left[y]_broadcast

    vsetvli zero, zero, e16, m1, ta, ma
    vadd.vv v6, v6, v8        # v6 = (blkSize - 1 -y) * above[x] + (x+1) * topRight + (y+1) * bottomLeft

    vmv.v.v v16, v6           # v16 = v6

    vsetvli zero, zero, e8, mf2, ta, ma
    vwmulu.vv v14, v5, v9                  # v14 = (7-x) * left[y]

    vsetvli zero, zero, e16, m1, ta, ma
    vadd.vv v16, v16, v14                  # v16 += (7-x) * left[y]

    vsetvli zero, zero, e8, mf2, ta, ma
    vnclipu.wi v18, v16, 4

    vse8.v v18, (t5)

    add     t5, t5, t6          # dst += dstStride
    addi    t4, t4, 1           # y++
    addi    t3, t3, -1
    bnez    t3, loop_y_8x8

    ret
endfunc


function PFX(intra_pred_planar16_rvv)
    li t0, 16
    vsetvli t0, t0, e8, m1, ta, ma

    addi t1, a2, 1
    vle8.v v0, (t1)           # v0 = above[0..15]

    lbu t2, 17(a2)
    vmv.v.x v2, t2            # topRight_b_broadcast

    lbu t3, 49(a2)
    vsetvli zero, zero, e16, m2, ta, ma
    vmv.v.x v4, t3            # {v4, v5} = bottomLeft_h_broadcast

    la t2, tbl_const_1to16_15to0
    vsetvli zero, zero, e8, m1, ta, ma
    vle8.v v6, (t2)           # v6 = [1,2,3,...,16] (x+1)
    addi t2, t2, 16
    vle8.v v7, (t2)           # v7 = [15,14,13,...,0] (15-x)

    vsetvli zero, zero, e16, m2, ta, ma
    vzext.vf2 v8, v0
    vsll.vi v8, v8, 4         # v8 = 16 * above[x]

    vsetvli zero, zero, e8, m1, ta, ma
    vwmulu.vv v10, v6, v2     # v10 = (x+1) * topRight

    vsetvli zero, zero, e16, m2, ta, ma
    vadd.vv v8, v8, v10       # {v8, v9} = 16 * above[x] + (x+1) * topRight

    vzext.vf2 v24, v0
    vsub.vv v12, v4, v24      # {v12, v13} = bottomLeft - above[x]

    csrwi  vxrm, 0
    li t3, 16
    li t4, 0
    mv t5, a0
    mv t6, a1

loop_y_16x16:
    add t1, a2, t4
    addi t1, t1, 33           # left = srcPix + 33 + y
    lbu t2, 0(t1)             # t2 = left[y]

    vsetvli zero, zero, e8, m1, ta, ma
    vmv.v.x v14, t2           # v14 = left[y]_broadcast

    vsetvli zero, zero, e16, m2, ta, ma
    vadd.vv v8, v8, v12       # {v8, v9} +=  = (blkSize - 1 -y) * above[x] + (x+1) * topRight + (y+1) * bottomLeft

    vmv.v.v v16, v8           # v16 = v8

    vsetvli zero, zero, e8, m1, ta, ma
    vwmulu.vv v18, v7, v14    # v18 = (15-x) * left[y]

    vsetvli zero, zero, e16, m2, ta, ma
    vadd.vv v16, v16, v18     # v16 += (15-x) * left[y]

    vsetvli zero, zero, e8, m1, ta, ma
    vnclipu.wi v20, v16, 5

    vse8.v v20, (t5)

    add     t5, t5, t6          # dst += dstStride
    addi    t4, t4, 1           # y++
    addi    t3, t3, -1
    bnez    t3, loop_y_16x16

    ret
endfunc
