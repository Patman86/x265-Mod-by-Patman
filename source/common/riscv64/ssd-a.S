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

/* sse_ss, 1 row */
.macro SSE_SS_x1 s0,s1
    vle16.v             v8, (\s0)
    vle16.v             v12, (\s1)
    vsub.vv             v16, v8, v12
    vwmacc.vv           v0, v16, v16     
.endm

/* sse_ss_func, width <= 16, fully unrolled */
.macro SSE_SS_FUNC_SMALL w,h,lmul1,lmul2
function PFX(pixel_sse_ss_\w\()x\h\()_rvv)
    li                  t0, \w
    slli                a1, a1, 1
    slli                a3, a3, 1
    vsetvli             zero, t0, e32, \lmul1, ta, ma
    vmv.v.i             v0, 0

    vsetvli             zero, t0, e16, \lmul2, ta, ma
    SSE_SS_x1           a0, a2
.rept \h - 1
    add                 a0, a0, a1
    add                 a2, a2, a3
    SSE_SS_x1           a0, a2
.endr
    vsetvli             zero, t0, e32, \lmul1, ta, ma
    vmv.v.i             v8, 0
#if BIT_DEPTH == 12
    vwredsum.vs         v8, v0, v8
    vsetvli             zero, t0, e64, \lmul1, ta, ma
#else
    vredsum.vs          v8, v0, v8
#endif
    vmv.x.s             a0, v8
    ret
endfunc
.endm

SSE_SS_FUNC_SMALL 4, 4, m1, m1
SSE_SS_FUNC_SMALL 8, 8, m2, m1
SSE_SS_FUNC_SMALL 16, 16, m4, m2

/* ssd_s_func, width <= 16, fully unrolled */
.macro SSD_S_FUNC_SMALL w,h,lmul1,lmul2
function PFX(pixel_ssd_s_\w\()x\h\()_rvv)
    li                  t0, \w
    slli                a1, a1, 1

    vsetvli             zero, t0, e32, \lmul1, ta, ma
    vmv.v.i             v0, 0

    vsetvli             zero, t0, e16, \lmul2, ta, ma
    vle16.v             v8, (a0)
    vwmacc.vv           v0, v8, v8
.rept \h - 1
    add                 a0, a0, a1
    vle16.v             v8, (a0)
    vwmacc.vv           v0, v8, v8
.endr
    vsetvli             zero, t0, e32, \lmul1, ta, ma
    vmv.v.i             v8, 0
    vredsum.vs          v8, v0, v8
    vmv.x.s             a0, v8
    ret
endfunc
.endm

SSD_S_FUNC_SMALL 4, 4, m1, m1
SSD_S_FUNC_SMALL 8, 8, m2, m1
SSD_S_FUNC_SMALL 16, 16, m4, m2

# if !HIGH_BIT_DEPTH

/* sse_pp, 1 row */
.macro SSE_PP_x1 s0,s1,lmul1,lmul2
    vsetvli             zero, t0, e8, \lmul2, ta, ma
    vle8.v              v8, (\s0)
    vle8.v              v10, (\s1)
    VABDU               v16, v8, v10, v12

    vsetvli             zero, t0, e32, \lmul1, ta, ma
    vzext.vf4           v24, v16
    vmacc.vv            v0, v24, v24
.endm

/* sse_pp_func */
.macro SSE_PP_FUNC w,h,lmul1,lmul2,len
function PFX(pixel_sse_pp_\w\()x\h\()_rvv)
    li                  t0, \len
    vsetvli             zero, t0, e32, \lmul1, ta, ma
    vmv.v.i             v0, 0

.if \w <=16 /* width <=16, fully unrolled */
    SSE_PP_x1           a0, a2,\lmul1,\lmul2
.rept \h - 1
    add                 a0, a0, a1
    add                 a2, a2, a3
    SSE_PP_x1           a0, a2, \lmul1, \lmul2
.endr

.else /* width >16, loop by row */
    li                  t1, \h
1:
    SSE_PP_x1           a0, a2, \lmul1, \lmul2

.if \w ==64
    addi                t2, a0, 32
    addi                t3, a2, 32
    SSE_PP_x1           t2, t3, \lmul1, \lmul2
.endif
    addi                t1, t1, -1
    add                 a0, a0, a1
    add                 a2, a2, a3
    bnez                t1, 1b
.endif
    vmv.v.i             v8, 0
    vredsum.vs          v8, v0, v8
    vmv.x.s             a0, v8
    ret
endfunc
.endm

SSE_PP_FUNC 4, 4, m1, m1, 4
SSE_PP_FUNC 4, 8, m1, m1, 4
SSE_PP_FUNC 8, 8, m2, m1, 8
SSE_PP_FUNC 8, 16, m2, m1, 8
SSE_PP_FUNC 16, 16 ,m4, m1, 16
SSE_PP_FUNC 16, 32, m4, m1, 16
SSE_PP_FUNC 32, 32, m8, m2, 32
SSE_PP_FUNC 32, 64, m8, m2, 32
SSE_PP_FUNC 64, 64, m8, m2, 32

/* sse_ss, width >16, loop by row */
.macro SSE_SS_FUNC_BIG w,h,lmul1,lmul2
function PFX(pixel_sse_ss_\w\()x\h\()_rvv)
    li                  t0, 32
    slli                a1, a1, 1
    slli                a3, a3, 1
    li                  t1, \h
    vsetvli             zero, t0, e32, \lmul1, ta, ma
    vmv.v.i             v0, 0

    vsetvli             zero, t0, e16, \lmul2, ta, ma       
1:
    SSE_SS_x1           a0, a2
.if \w ==64
    addi                t2, a0, 64
    addi                t3, a2, 64
    SSE_SS_x1           t2, t3
.endif
    addi                t1, t1, -1
    add                 a0, a0, a1
    add                 a2, a2, a3
    bnez                t1, 1b

    vsetvli             zero, t0, e32, \lmul1, ta, ma
    vmv.v.i             v8, 0
    vredsum.vs          v8, v0, v8
    vmv.x.s             a0, v8
    ret
endfunc
.endm

SSE_SS_FUNC_BIG 32, 32, m8, m4
SSE_SS_FUNC_BIG 64, 64, m8, m4

/* ssd_s, width >16, loop by row */
.macro SSD_S_FUNC_BIG w,h,lmul1,lmul2,len
function PFX(pixel_ssd_s_\w\()x\h\()_rvv)
    li                  t0, 32
    slli                a1, a1, 1
    li                  t1, \h

    vsetvli             zero, t0, e32, \lmul1, ta, ma
    vmv.v.i             v0, 0

    vsetvli             zero, t0, e16, \lmul2, ta, ma
1:
    vle16.v             v8, (a0)           
    vwmacc.vv           v0, v8, v8
.if \w == 64
    addi                t2, a0, 64
    vle16.v             v8, (t2)
    vwmacc.vv           v0, v8, v8
.endif      
    addi                t1, t1, -1
    add                 a0, a0, a1
    bnez                t1, 1b

    vsetvli             zero, t0, e32, \lmul1, ta, ma
    vmv.v.i             v8, 0
    vredsum.vs          v8, v0, v8
    vmv.x.s             a0, v8
    ret
endfunc
.endm

SSD_S_FUNC_BIG 32, 32, m8, m4
SSD_S_FUNC_BIG 64, 64, m8, m4

#else // !HIGH BIT DEPTH

/* sse_pp, 1 row, width <=16 */
.macro SSE_PP_x1_SMALL s0,s1,lmul1,lmul2
    vle16.v             v8, (\s0)
    vle16.v             v10, (\s1)
    VABDU               v16, v8, v10, v12
    vwmacc.vv           v0, v16, v16
.endm

/* sse_pp_func, width <=16 */
.macro SSE_PP_FUNC_SMALL w,h,lmul1,lmul2
function PFX(pixel_sse_pp_\w\()x\h\()_rvv)
    li                  t0, \w
    slli                a1, a1, 1
    slli                a3, a3, 1

    vsetvli             zero, t0, e32, \lmul1, ta, ma
    vmv.v.i             v0, 0

    vsetvli             zero, t0, e16, \lmul2, ta, ma
    SSE_PP_x1_SMALL     a0, a2, \lmul1, \lmul2
.rept \h - 1
    add                 a0, a0, a1
    add                 a2, a2, a3
    SSE_PP_x1_SMALL     a0, a2, \lmul1, \lmul2
.endr
    vsetvli             zero, t0, e32, \lmul1, ta, ma
    vmv.v.i             v8, 0
#if BIT_DEPTH == 12
    vwredsum.vs         v8, v0, v8
    vsetvli             zero, t0, e64, \lmul1, ta, ma
#else
    vredsum.vs          v8, v0, v8
#endif
    vmv.x.s             a0, v8
    ret
endfunc
.endm

SSE_PP_FUNC_SMALL 4, 4, m1, m1
SSE_PP_FUNC_SMALL 4, 8, m1, m1
SSE_PP_FUNC_SMALL 8, 8, m2, m1
SSE_PP_FUNC_SMALL 8, 16, m2, m1
SSE_PP_FUNC_SMALL 16, 16, m4, m2
SSE_PP_FUNC_SMALL 16, 32, m4, m2

/* sse_pp, 1 row, width > 16 */
.macro SSE_PP_x1_BIG s0,s1,lmul1,lmul2
    vsetvli             zero, t0, e16, \lmul2, ta, ma
    vle16.v             v8, (\s0)
    vle16.v             v10, (\s1)
    VABDU               v16, v8, v10, v12

    vsetvli             zero, t0, e64, \lmul1, ta, ma
    vzext.vf4           v24, v16
    vmacc.vv            v0, v24, v24
.endm

/* sse_pp_func,  width >16 */
.macro SSE_PP_FUNC_BIG w,h,lmul1,lmul2
function PFX(pixel_sse_pp_\w\()x\h\()_rvv)
    li                  t0, 16
    slli                a1, a1, 1
    slli                a3, a3, 1
    li                  t1, \h
    vsetvli             zero, t0, e64, \lmul1, ta, ma
    vmv.v.i             v0, 0
1:
    mv                  t2, a0
    mv                  t3, a2
    SSE_PP_x1_BIG       a0, a2, \lmul1, \lmul2
.rept (\w / 16) -1
    addi                t2, t2, 32
    addi                t3, t3, 32
    SSE_PP_x1_BIG       t2, t3, \lmul1, \lmul2
.endr
    addi                t1, t1, -1
    add                 a0, a0, a1
    add                 a2, a2, a3
    bnez                t1, 1b

    vmv.v.i             v8, 0
    vredsum.vs          v8, v0, v8
    vmv.x.s             a0, v8
    ret
endfunc
.endm

SSE_PP_FUNC_BIG 32, 32, m8, m2
SSE_PP_FUNC_BIG 32, 64, m8, m2
SSE_PP_FUNC_BIG 64, 64, m8, m2

/* sse_ss, 1 row, width >16 */
.macro SSE_SS_x1_BIG s0,s1,lmul1,lmul2
    vsetvli             zero, t0, e16, \lmul2, ta, ma
    vle16.v             v8, (\s0)
    vle16.v             v10, (\s1)
    vsub.vv             v16, v8, v10

    vsetvli             zero, t0, e64, \lmul1, ta, ma
    vsext.vf4           v8, v16
    vmacc.vv            v0, v8, v8
.endm

/* sse_ss_func, width > 16, loop by row */
.macro SSE_SS_FUNC_BIG w,h,lmul1,lmul2
function PFX(pixel_sse_ss_\w\()x\h\()_rvv)
    li                  t0, 16
    slli                a1, a1, 1
    slli                a3, a3, 1

    vsetvli             zero, t0, e64, \lmul1, ta, ma
    vmv.v.i             v0, 0
    li                  t1, \h
1:
    mv                  t2, a0
    mv                  t3, a2
    SSE_SS_x1_BIG       a0, a2, \lmul1, \lmul2
.rept (\w / 16) -1
    addi                t2, t2, 32
    addi                t3, t3, 32
    SSE_SS_x1_BIG       t2, t3, \lmul1, \lmul2
.endr
    addi                t1, t1, -1
    add                 a0, a0, a1
    add                 a2, a2, a3
    bnez                t1, 1b

    vmv.v.i             v8, 0
    vredsum.vs          v8, v0, v8
    vmv.x.s             a0, v8
    ret
endfunc
.endm

SSE_SS_FUNC_BIG 32, 32, m8, m2
SSE_SS_FUNC_BIG 64, 64, m8, m2

/* ssd_s, 1 row, width >16 */
.macro SSD_S_x1_BIG s0,lmul1,lmul2
    vsetvli             zero, t0, e16, \lmul2, ta, ma
    vle16.v             v8, (\s0)            

    vsetvli             zero, t0, e64, \lmul1, ta, ma
    vsext.vf4           v16, v8
    vmacc.vv            v0, v16, v16        
.endm

/* ssd_s_func, width > 16, loop by row */
.macro SSD_S_FUNC_BIG w,h,lmul1,lmul2
function PFX(pixel_ssd_s_\w\()x\h\()_rvv)
    li                  t0, 16
    slli                a1, a1, 1
    li                  t1, \h
    vsetvli             zero, t0, e64, \lmul1, ta, ma
    vmv.v.i             v0, 0
1:
    mv                  t2, a0
    SSD_S_x1_BIG        a0, \lmul1, \lmul2
.rept (\w / 16) -1
    addi                t2, t2, 32
    SSD_S_x1_BIG        t2, \lmul1, \lmul2
.endr     
    addi                t1, t1, -1
    add                 a0, a0, a1
    bnez                t1, 1b

    vmv.v.i             v8, 0
    vredsum.vs          v8, v0, v8
    vmv.x.s             a0, v8
    ret
endfunc
.endm

SSD_S_FUNC_BIG 32, 32, m8, m2
SSD_S_FUNC_BIG 64, 64, m8, m2

#endif
