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

.align 4

.text

#if BIT_DEPTH == 8
// void pixelavg_pp(pixel *dst, intptr_t dstride, const pixel *src0, intptr_t sstride0,
//                  const pixel *src1, intptr_t sstride1, int);
.macro AVG_PP_LOOP_FUNC w, h, lmul
function PFX(pixel_avg_pp_\w\()x\h\()_rvv)
    csrwi        vxrm, 0
    li           t0, \h
    li           t1, \w
1:
    addi         t0, t0, -4
    vsetvli      zero, t1, e8, m\()\lmul, ta, ma

    vle8.v       v0, (a2)
    vle8.v       v16, (a4)
    add          a2, a2, a3
    add          a4, a4, a5
    vaaddu.vv    v0, v0, v16

    vle8.v       v4, (a2)
    vle8.v       v20, (a4)
    add          a2, a2, a3
    add          a4, a4, a5
    vaaddu.vv    v4, v4, v20

    vle8.v       v8, (a2)
    vle8.v       v24, (a4)
    add          a2, a2, a3
    add          a4, a4, a5
    vaaddu.vv    v8, v8, v24

    vle8.v       v12, (a2)
    vle8.v       v28, (a4)
    add          a2, a2, a3
    add          a4, a4, a5
    vaaddu.vv    v12, v12, v28

    vse8.v       v0, (a0)
    add          a0, a0, a1
    vse8.v       v4, (a0)
    add          a0, a0, a1
    vse8.v       v8, (a0)
    add          a0, a0, a1
    vse8.v       v12, (a0)
    add          a0, a0, a1
    bnez         t0, 1b
    ret
endfunc
.endm

AVG_PP_LOOP_FUNC  4,  4,  1
AVG_PP_LOOP_FUNC  4,  8,  1
AVG_PP_LOOP_FUNC  4,  16,  1

AVG_PP_LOOP_FUNC  8,  4,  1
AVG_PP_LOOP_FUNC  8,  8,  1
AVG_PP_LOOP_FUNC  8,  16,  1
AVG_PP_LOOP_FUNC  8,  32,  1

AVG_PP_LOOP_FUNC  12,  16,  1

AVG_PP_LOOP_FUNC  16,  4,  1
AVG_PP_LOOP_FUNC  16,  8,  1
AVG_PP_LOOP_FUNC  16,  12,  1
AVG_PP_LOOP_FUNC  16,  16,  1
AVG_PP_LOOP_FUNC  16,  32,  1
AVG_PP_LOOP_FUNC  16,  64,  1

AVG_PP_LOOP_FUNC  24,  32,  2

AVG_PP_LOOP_FUNC  32,  8,  2
AVG_PP_LOOP_FUNC  32,  16,  2
AVG_PP_LOOP_FUNC  32,  24,  2
AVG_PP_LOOP_FUNC  32,  32,  2
AVG_PP_LOOP_FUNC  32,  64,  2

AVG_PP_LOOP_FUNC  48,  64,  4

AVG_PP_LOOP_FUNC  64,  16,  4
AVG_PP_LOOP_FUNC  64,  32,  4
AVG_PP_LOOP_FUNC  64,  48,  4
AVG_PP_LOOP_FUNC  64,  64,  4

#endif