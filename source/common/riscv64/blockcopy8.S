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

#include "asm.S"

.align 4

.text

// uint32_t copy_count(int16_t* coeff, const int16_t* residual, intptr_t resiStride)
.macro COPY_CNT_N_V tr, lmul
function PFX(copy_cnt_\tr\()_v)
    slli            a2, a2, 1
    li              a3, \tr
    vsetvli         zero, a3, e16, m\lmul, ta, ma
    vmv.v.i         v20, 0
    vmv.v.i         v24, 1
    vmv.v.i         v28, 0

.rept \tr / 2
    vle16.v         v4, (a1)
    add             a1, a1, a2
    vle16.v         v8, (a1)
    add             a1, a1, a2
    vse16.v         v4, (a0)
    addi            a0, a0, \tr * 2
    vse16.v         v8, (a0)
    addi            a0, a0, \tr * 2

    vmsne.vi        v0, v4, 0
    vsetvli         zero, a3, e16, m\lmul, ta, mu
    vadd.vv         v20, v20, v24, v0.t
    vmsne.vi        v0, v8, 0
    vadd.vv         v20, v20, v24, v0.t
    vsetvli         zero, a3, e16, m\lmul, ta, ma
.endr

    vredsum.vs      v20, v20, v28
    vmv.x.s         a0, v20
    ret
endfunc
.endm

COPY_CNT_N_V 4, 1
COPY_CNT_N_V 8, 1
COPY_CNT_N_V 16, 2
COPY_CNT_N_V 32, 4

// int  count_nonzero_c(const int16_t* quantCoeff)
.macro COUNT_NONZERO_N_V tr, lmul
function PFX(count_nonzero_\tr\()_v)
    li              a3, \tr
    vsetvli         zero, a3, e16, m\lmul, ta, ma
    vmv.v.i         v12, 0
    vmv.v.i         v16, 1
    vmv.v.i         v20, 0

.rept \tr / 2
    vle16.v         v4, (a0)
    addi            a0, a0, \tr * 2
    vle16.v         v8, (a0)
    addi            a0, a0, \tr * 2
    vmsne.vi        v0, v4, 0
    vsetvli         zero, a3, e16, m\lmul, ta, mu
    vadd.vv         v12, v12, v16, v0.t
    vmsne.vi        v0, v8, 0
    vadd.vv         v12, v12, v16, v0.t
    vsetvli         zero, a3, e16, m\lmul, ta, ma
.endr

    vredsum.vs      v12, v12, v20
    vmv.x.s         a0, v12
    ret
endfunc
.endm

COUNT_NONZERO_N_V 4, 1
COUNT_NONZERO_N_V 8, 1
COUNT_NONZERO_N_V 16, 2
COUNT_NONZERO_N_V 32, 4
