/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Yujiao He <he.yujiao@sanechips.com.cn>
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

#if HIGH_BIT_DEPTH
# if BIT_DEPTH == 10
#  define P2S_SHIFT 4
# elif BIT_DEPTH == 12
#  define P2S_SHIFT 2
# endif
#else // if !HIGH_BIT_DEPTH
# define P2S_SHIFT 6
#endif // HIGH_BIT_DEPTH

.macro p2s_x1 lmul1, lmul2
#if HIGH_BIT_DEPTH
    vle16.v         v8, (a0)
    vsll.vi         v16, v8, P2S_SHIFT
#else
    vsetvli         zero, t0, e8, \lmul2, ta, ma
    vle8.v          v8, (a0)
    vsetvli         zero, t0, e16, \lmul1, ta, ma
    vzext.vf2       v16, v8
    vsll.vi         v16, v16, P2S_SHIFT
#endif
    vadd.vv         v16, v16, v0
    vse16.v         v16, (a2)
.endm

.macro P2S_FUNC w, h, lmul1, lmul2
function PFX(filterPixelToShort_\w\()x\h\()_rvv)
    li              t0, \w
    li              t2, 0xe000
    vsetvli         zero, t0, e16, \lmul1, ta, ma
    vmv.v.x         v0, t2
#if HIGH_BIT_DEPTH
    slli            a1, a1, 1
#endif
    slli            a3, a3, 1
.if \h <= 16 /* height <= 16, fully unroll*/
    p2s_x1          \lmul1, \lmul2
.rept \h -1
    add             a0, a0, a1
    add             a2, a2, a3
    p2s_x1          \lmul1, \lmul2
.endr
.else  /* height > 16, 4 rows unroll*/
    li              t1, \h
1:
.rept 4
    p2s_x1          \lmul1, \lmul2
    add             a0, a0, a1
    add             a2, a2, a3
.endr
    addi            t1, t1, -4
    bnez            t1, 1b
.endif
    ret
endfunc
.endm

P2S_FUNC 4, 2, m1, m1
P2S_FUNC 4, 4, m1, m1
P2S_FUNC 4, 8, m1, m1
P2S_FUNC 4, 16, m1, m1
P2S_FUNC 4, 32, m1, m1

P2S_FUNC 6, 8, m1, m1
P2S_FUNC 6, 16, m1, m1

P2S_FUNC 8, 2, m1, m1
P2S_FUNC 8, 4, m1, m1
P2S_FUNC 8, 8, m1, m1
P2S_FUNC 8, 6, m1, m1
P2S_FUNC 8, 12, m1, m1
P2S_FUNC 8, 16, m1, m1
P2S_FUNC 8, 32, m1, m1
P2S_FUNC 8, 64, m1, m1

P2S_FUNC 12, 16, m2, m1
P2S_FUNC 12, 32, m2, m1

P2S_FUNC 16, 4, m2, m1
P2S_FUNC 16, 8, m2, m1
P2S_FUNC 16, 12, m2, m1
P2S_FUNC 16, 16, m2, m1
P2S_FUNC 16, 24, m2, m1
P2S_FUNC 16, 32, m2, m1
P2S_FUNC 16, 64, m2, m1

P2S_FUNC 24, 32, m4, m2
P2S_FUNC 24, 64, m4, m2

P2S_FUNC 32, 8, m4, m2
P2S_FUNC 32, 12, m4, m2
P2S_FUNC 32, 16, m4, m2
P2S_FUNC 32, 24, m4, m2
P2S_FUNC 32, 32, m4, m2
P2S_FUNC 32, 48, m4, m2
P2S_FUNC 32, 64, m4, m2

P2S_FUNC 48, 64, m8, m4

P2S_FUNC 64, 16, m8, m4
P2S_FUNC 64, 32, m8, m4
P2S_FUNC 64, 48, m8, m4
P2S_FUNC 64, 64, m8, m4
