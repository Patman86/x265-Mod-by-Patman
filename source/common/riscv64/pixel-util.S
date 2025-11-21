/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Changsheng Wu <wu.changsheng@sanechips.com.cn>
 *          Yujiao He <he.yujiao@sanechips.com.cn>
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

// uint64_t pixel_var(const pixel* pix, intptr_t i_stride)
.macro PIXEL_VAR size, lmul, lmul2, lmul4
function PFX(pixel_var_\size\()x\size\()_v)
    li              t1, \size
    li              t3, \size
#if HIGH_BIT_DEPTH
    slli            t2, a1, 1
#else
    mv              t2, a1
#endif
    vsetvli         zero, t3, e32, m\lmul4, ta, ma
    vmv.v.i         v8, 0
    vmv.v.i         v16, 0

loop_pixelvar\size:
#if HIGH_BIT_DEPTH
    vsetvli         zero, t3, e16, m\lmul2, ta, ma
    vle16.v         v0, (a0)
    vwaddu.wv       v8, v8, v0
    vwmulu.vv       v24, v0, v0
    vsetvli         zero, t3, e32, m\lmul4, ta, ma
    vadd.vv         v16, v16, v24
#else
    vsetvli         zero, t3, e8, m\lmul, ta, ma
    vle8.v          v0, (a0)
    vwmulu.vv       v4, v0, v0
    vsetvli         zero, t3, e16, m\lmul2, ta, ma
    vzext.vf2       v24, v0
    vwaddu.wv       v8, v8, v24
    vwaddu.wv       v16, v16, v4
#endif
    addi            t1, t1, -1
    add             a0, a0, t2
    bgtz            t1, loop_pixelvar\size

    vsetvli         zero, t3, e32, m1, ta, ma
    vmv.v.i         v0, 0
    vsetvli         zero, t3, e32, m\lmul4, ta, ma
    vredsum.vs      v1, v8, v0
    vredsum.vs      v2, v16, v0
    vmv.x.s         t1, v1
    vmv.x.s         t2, v2
    slli            t2, t2, 32
    add             a0, t1, t2
    ret
endfunc
.endm

.macro PIXEL_VAR2 size, lmul, lmul2, lmul4
function PFX(pixel_var_\size\()x\size\()_v)
    li              t1, \size
    li              t3, \size
    slli            a2, a1, 1
    slli            a3, a2, 1
    add             a4, a3, a2
    add             a5, a2, a1
    vsetvli         zero, t3, e32, m\lmul4, ta, ma
    vmv.v.i         v8, 0
    vmv.v.i         v16, 0

loop_pixelvar\size:
#if HIGH_BIT_DEPTH
    add             t4, a0, a2
    add             t5, a0, a3
    add             t6, a0, a4
    vsetvli         zero, t3, e16, m\lmul2, ta, ma
    vle16.v         v0, (a0)
    vle16.v         v2, (t4)
    vle16.v         v4, (t5)
    vle16.v         v6, (t6)
    add             a0, t6, a2
    vwaddu.wv       v8, v8, v0
    vwaddu.wv       v8, v8, v2
    vwaddu.wv       v8, v8, v4
    vwaddu.wv       v8, v8, v6
    vwmulu.vv       v24, v0, v0
    vwmulu.vv       v12, v2, v2
    vwmulu.vv       v20, v4, v4
    vwmulu.vv       v28, v6, v6
    vsetvli         zero, t3, e32, m\lmul4, ta, ma
    vadd.vv         v16, v16, v24
    vadd.vv         v16, v16, v12
    vadd.vv         v16, v16, v20
    vadd.vv         v16, v16, v28
#else
    add             t4, a0, a1
    add             t5, a0, a2
    add             t6, a0, a5
    vsetvli         zero, t3, e8, m\lmul, ta, ma
    vle8.v          v20, (a0)
    vle8.v          v21, (t4)
    vle8.v          v22, (t5)
    vle8.v          v23, (t6)
    add             a0, t6, a1
    vwmulu.vv       v24, v20, v20
    vwmulu.vv       v26, v21, v21
    vwmulu.vv       v28, v22, v22
    vwmulu.vv       v30, v23, v23
    vsetvli         zero, t3, e16, m\lmul2, ta, ma
    vzext.vf2       v0, v20
    vzext.vf2       v2, v21
    vzext.vf2       v4, v22
    vzext.vf2       v6, v23
    vwaddu.wv       v8, v8, v0
    vwaddu.wv       v8, v8, v2
    vwaddu.wv       v8, v8, v4
    vwaddu.wv       v8, v8, v6
    vwaddu.wv       v16, v16, v24
    vwaddu.wv       v16, v16, v26
    vwaddu.wv       v16, v16, v28
    vwaddu.wv       v16, v16, v30
#endif
    addi            t1, t1, -4
    bgtz            t1, loop_pixelvar\size

    vsetvli         zero, t3, e32, m1, ta, ma
    vmv.v.i         v0, 0
    vsetvli         zero, t3, e32, m\lmul4, ta, ma
    vredsum.vs      v1, v8, v0
    vredsum.vs      v2, v16, v0
    vmv.x.s         t1, v1
    vmv.x.s         t2, v2
    slli            t2, t2, 32
    add             a0, t1, t2
    ret
endfunc
.endm

PIXEL_VAR2 4, 1, 1, 1
PIXEL_VAR2 8, 1, 1, 2
PIXEL_VAR2 16, 1, 2, 4
PIXEL_VAR  32, 2, 4, 8

// uint64_t pixel_var(const pixel* pix, intptr_t i_stride)
function PFX(pixel_var_64x64_v)
    li              t1, 64
#if HIGH_BIT_DEPTH
    slli            t2, a1, 1
#else
    mv              t2, a1
#endif
    vsetvli         zero, t1, e32, m8, ta, ma
    vmv.v.i         v8, 0
    vmv.v.i         v16, 0

loop_pixelvar64:
    li              t3, 64
    mv              t4, a0
loop_pixelvar264:
#if HIGH_BIT_DEPTH
    vsetvli         t5, t3, e16, m4, ta, ma
    vle16.v         v0, (t4)
#else
    vsetvli         t5, t3, e8, m2, ta, ma
    vle8.v          v4, (t4)
    vsetvli         zero, t5, e16, m4, ta, ma
    vzext.vf2       v0, v4
#endif
    vwaddu.wv       v8, v8, v0
    vwmulu.vv       v24, v0, v0
    vsetvli         zero, t5, e32, m8, ta, ma
    vadd.vv         v16, v16, v24
#if HIGH_BIT_DEPTH
    slli            t6, t5, 1
    add             t4, t4, t6
#else
    add             t4, t4, t5
#endif
    sub             t3, t3, t5
    bgtz            t3, loop_pixelvar264
    addi            t1, t1, -1
    add             a0, a0, t2
    bgtz            t1, loop_pixelvar64

    li              t1, 64
    vsetvli         zero, t1, e32, m1, ta, ma
    vmv.v.i         v0, 0
    vsetvli         zero, t1, e32, m8, ta, ma
    vredsum.vs      v1, v8, v0
    vredsum.vs      v2, v16, v0
    vmv.x.s         t1, v1
    vmv.x.s         t2, v2
    slli            t2, t2, 32
    add             a0, t1, t2
    ret
endfunc

// void getResidual(const pixel* fenc, const pixel* pred, int16_t* residual, intptr_t stride)
.macro GET_RESIDUAL blocksize, lmul, lmul2
function PFX(getResidual\blocksize\()_v)
    li              t1, \blocksize
    slli            a4, a3, 1
#if HIGH_BIT_DEPTH
    vsetvli         zero, t1, e16, m\lmul2, ta, ma
#else
    vsetvli         zero, t1, e8, m\lmul, ta, ma
#endif
loop_residual\blocksize:
    add             t4, a2, a4
#if HIGH_BIT_DEPTH
    add             t2, a0, a4
    add             t3, a1, a4
    vle16.v         v0, (a0)
    vle16.v         v12, (t2)
    vle16.v         v4, (a1)
    vle16.v         v16, (t3)
    add             a0, t2, a4
    add             a1, t3, a4
    vsub.vv         v8, v0, v4
    vsub.vv         v20, v12, v16
#else
    add             t2, a0, a3
    add             t3, a1, a3
    vle8.v          v0, (a0)
    vle8.v          v12, (t2)
    vle8.v          v4, (a1)
    vle8.v          v16, (t3)
    add             a0, t2, a3
    add             a1, t3, a3
    vwsubu.vv       v8, v0, v4
    vwsubu.vv       v20, v12, v16
#endif
    vse16.v         v8, (a2)
    vse16.v         v20, (t4)
    add             a2, t4, a4
    addi            t1, t1, -2
    bgtz            t1, loop_residual\blocksize
    ret
endfunc
.endm

.macro GET_RESIDUAL2 blocksize, lmul, lmul2
function PFX(getResidual\blocksize\()_v)
    li              t1, \blocksize
    slli            a4, a3, 1
    slli            a5, a4, 1
    add             a6, a5, a4
    add             a7, a4, a3
#if HIGH_BIT_DEPTH
    vsetvli         zero, t1, e16, m\lmul2, ta, ma
#else
    vsetvli         zero, t1, e8, m\lmul, ta, ma
#endif
loop_residual\blocksize:
#if HIGH_BIT_DEPTH
    add             t2, a0, a4
    add             t3, a1, a4
    add             t4, a0, a5
    add             t5, a1, a5
    add             t0, a0, a6
    add             t6, a1, a6
    vle16.v         v0, (a0)
    vle16.v         v12, (t2)
    vle16.v         v2, (t4)
    vle16.v         v14, (t0)
    vle16.v         v4, (a1)
    vle16.v         v16, (t3)
    vle16.v         v6, (t5)
    vle16.v         v18, (t6)
    add             a0, t0, a4
    add             a1, t6, a4
    vsub.vv         v8, v0, v4
    vsub.vv         v20, v12, v16
    vsub.vv         v10, v2, v6
    vsub.vv         v22, v14, v18
#else
    add             t2, a0, a3
    add             t3, a1, a3
    add             t4, a0, a4
    add             t5, a1, a4
    add             t0, a0, a7
    add             t6, a1, a7
    vle8.v          v0, (a0)
    vle8.v          v12, (t2)
    vle8.v          v2, (t4)
    vle8.v          v14, (t0)
    vle8.v          v4, (a1)
    vle8.v          v16, (t3)
    vle8.v          v6, (t5)
    vle8.v          v18, (t6)
    add             a0, t0, a3
    add             a1, t6, a3
    vwsubu.vv       v8, v0, v4
    vwsubu.vv       v20, v12, v16
    vwsubu.vv       v10, v2, v6
    vwsubu.vv       v22, v14, v18
#endif
    add             t2, a2, a4
    add             t3, a2, a5
    add             t4, a2, a6
    vse16.v         v8, (a2)
    vse16.v         v20, (t2)
    vse16.v         v10, (t3)
    vse16.v         v22, (t4)
    add             a2, t4, a4
    addi            t1, t1, -4
    bgtz            t1, loop_residual\blocksize
    ret
endfunc
.endm

GET_RESIDUAL2 4,  1, 1
GET_RESIDUAL2 8,  1, 1
GET_RESIDUAL2 16, 1, 2
GET_RESIDUAL  32, 2, 4

// void pixel_sub_ps_c(int16_t* a, intptr_t dstride, const pixel* b0, const pixel* b1, intptr_t sstride0, intptr_t sstride1)
.macro SUB_PS xsize, ysize, lmul, lmul2
function PFX(pixel_sub_ps_\xsize\()x\ysize\()_v)
    li              t0, \xsize
    li              t1, \ysize
    slli            a1, a1, 1
#if HIGH_BIT_DEPTH
    slli            a4, a4, 1
    slli            a5, a5, 1
#endif
loop_subps_\xsize\()x\ysize\():
#if HIGH_BIT_DEPTH
    vsetvli         zero, t0, e16, m\lmul2, ta, ma
    vle16.v         v0, (a2)
    vle16.v         v8, (a3)
    vsub.vv         v16, v0, v8
#else
    vsetvli         zero, t0, e8, m\lmul, ta, ma
    vle8.v          v0, (a2)
    vle8.v          v4, (a3)
    vwsubu.vv       v16, v0, v4
    vsetvli         zero, t0, e16, m\lmul2, ta, ma
#endif
    vse16.v         v16, (a0)
    add             a0, a0, a1
    add             a2, a2, a4
    add             a3, a3, a5
    addi            t1, t1, -1
    bgtz            t1, loop_subps_\xsize\()x\ysize\()
    ret
endfunc
.endm

.macro SUB_PS2 xsize, ysize, lmul, lmul2
function PFX(pixel_sub_ps_\xsize\()x\ysize\()_v)
    li              t0, \xsize
    li              t1, \ysize
    slli            a1, a1, 1
#if HIGH_BIT_DEPTH
    slli            a4, a4, 1
    slli            a5, a5, 1
#endif
loop_subps_\xsize\()x\ysize\():
    add             t3, a2, a4
    add             t4, a3, a5
#if HIGH_BIT_DEPTH
    vsetvli         zero, t0, e16, m\lmul2, ta, ma
    vle16.v         v0, (a2)
    vle16.v         v4, (t3)
    vle16.v         v8, (a3)
    vle16.v         v12, (t4)
    vsub.vv         v16, v0, v8
    vsub.vv         v20, v4, v12
#else
    vsetvli         zero, t0, e8, m\lmul, ta, ma
    vle8.v          v0, (a2)
    vle8.v          v2, (t3)
    vle8.v          v4, (a3)
    vle8.v          v6, (t4)
    vwsubu.vv       v16, v0, v4
    vwsubu.vv       v20, v2, v6
    vsetvli         zero, t0, e16, m\lmul2, ta, ma
#endif
    add             a2, t3, a4
    add             a3, t4, a5
    add             t5, a0, a1
    vse16.v         v16, (a0)
    vse16.v         v20, (t5)
    add             a0, t5, a1
    addi            t1, t1, -2
    bgtz            t1, loop_subps_\xsize\()x\ysize\()
    ret
endfunc
.endm

SUB_PS2  4,  4,  1, 1
SUB_PS2  4,  8,  1, 1
SUB_PS2  8,  8,  1, 1
SUB_PS2  8,  16, 1, 1
SUB_PS2  16, 16, 1, 2
SUB_PS2  16, 32, 1, 2
SUB_PS2  32, 32, 2, 4
SUB_PS2  32, 64, 2, 4
SUB_PS   64, 64, 4, 8

// void pixel_add_ps_c(pixel* a, intptr_t dstride, const pixel* b0, const int16_t* b1, intptr_t sstride0, intptr_t sstride1)
.macro ADD_PS xsize, ysize, lmul, lmul2
function PFX(pixel_add_ps_\xsize\()x\ysize\()_v)
    li              t0, \xsize
    li              t1, \ysize
    li              t2, (1 << BIT_DEPTH) - 1
    slli            a5, a5, 1
#if HIGH_BIT_DEPTH
    slli            a1, a1, 1
    slli            a4, a4, 1
#endif
    vsetvli         zero, t0, e16, m\lmul2, ta, ma
    vmv.v.i         v24, 0
.if \lmul2 <= 4
    vmv.v.x         v28, t2
.endif
loop_addps_\xsize\()x\ysize\():
    vsetvli         zero, t0, e16, m\lmul2, ta, ma
    vle16.v         v8, (a3)
#if HIGH_BIT_DEPTH
    vle16.v         v0, (a2)
    vadd.vv         v16, v0, v8
#else
    vsetvli         zero, t0, e8, m\lmul, ta, ma
    vle8.v          v0, (a2)
    vwaddu.wv       v16, v8, v0
    vsetvli         zero, t0, e16, m\lmul2, ta, ma
#endif
    vmax.vv         v8, v16, v24
.if \lmul2 <= 4
    vminu.vv        v16, v8, v28
.else
    vmv.v.x         v0, t2
    vminu.vv        v16, v8, v0
.endif
#if HIGH_BIT_DEPTH
    vse16.v         v16, (a0)
#else
    vsetvli         zero, t0, e8, m\lmul, ta, ma
    vnsrl.wi        v8, v16, 0
    vse8.v          v8, (a0)
#endif
    addi            t1, t1, -1
    add             a0, a0, a1
    add             a2, a2, a4
    add             a3, a3, a5
    bgtz            t1, loop_addps_\xsize\()x\ysize\()
    ret
endfunc
.endm

.macro ADD_PS2 xsize, ysize, lmul, lmul2
function PFX(pixel_add_ps_\xsize\()x\ysize\()_v)
    li              t0, \xsize
    li              t1, \ysize
    li              t2, (1 << BIT_DEPTH) - 1
    slli            a5, a5, 1
#if HIGH_BIT_DEPTH
    slli            a1, a1, 1
    slli            a4, a4, 1
#endif
    vsetvli         zero, t0, e16, m\lmul2, ta, ma
    vmv.v.i         v24, 0
    vmv.v.x         v28, t2
loop_addps_\xsize\()x\ysize\():
    vsetvli         zero, t0, e16, m\lmul2, ta, ma
    add             t3, a3, a5
    add             t4, a2, a4
    vle16.v         v8, (a3)
    vle16.v         v12, (t3)
#if HIGH_BIT_DEPTH
    vle16.v         v0, (a2)
    vle16.v         v4, (t4)
    vadd.vv         v16, v0, v8
    vadd.vv         v20, v4, v12
#else
    vsetvli         zero, t0, e8, m\lmul, ta, ma
    vle8.v          v0, (a2)
    vle8.v          v4, (t4)
    vwaddu.wv       v16, v8, v0
    vwaddu.wv       v20, v12, v4
    vsetvli         zero, t0, e16, m\lmul2, ta, ma
#endif
    add             a2, t4, a4
    add             a3, t3, a5
    vmax.vv         v16, v16, v24
    vmax.vv         v20, v20, v24
    vminu.vv        v16, v16, v28
    vminu.vv        v20, v20, v28
    add             t3, a0, a1
#if HIGH_BIT_DEPTH
    vse16.v         v16, (a0)
    vse16.v         v20, (t3)
#else
    vsetvli         zero, t0, e8, m\lmul, ta, ma
    vnsrl.wi        v0, v16, 0
    vnsrl.wi        v4, v20, 0
    vse8.v          v0, (a0)
    vse8.v          v4, (t3)
#endif
    addi            t1, t1, -2
    add             a0, t3, a1
    bgtz            t1, loop_addps_\xsize\()x\ysize\()
    ret
endfunc
.endm

ADD_PS2 4,  4,  1, 1
ADD_PS2 4,  8,  1, 1
ADD_PS2 8,  8,  1, 1
ADD_PS2 8,  16, 1, 1
ADD_PS2 16, 16, 1, 2
ADD_PS2 16, 32, 1, 2
ADD_PS2 32, 64, 2, 4
ADD_PS2 32, 32, 2, 4
ADD_PS  64, 64, 4, 8

// void scale1D_128to64(pixel *dst, const pixel *src)
function PFX(scale1D_128to64_v)
    li              t1, 64
#if HIGH_BIT_DEPTH
    vsetvli         zero, t1, e16, m8, ta, ma
    li              t2, 4
    addi            t3, a1, 2
    addi            t4, a1, 256
    addi            t5, a1, 258
    addi            t6, a0, 128
    vlse16.v        v0, (a1), t2
    vlse16.v        v8, (t3), t2
    vlse16.v        v16, (t4), t2
    vlse16.v        v24, (t5), t2
    vaaddu.vv       v0, v0, v8
    vaaddu.vv       v8, v16, v24
    vse16.v         v0, (a0)
    vse16.v         v8, (t6)
#else
    vsetvli         zero, t1, e8, m4, ta, ma
    li              t2, 2
    addi            t3, a1, 1
    addi            t4, a1, 128
    addi            t5, a1, 129
    addi            t6, a0, 64
    vlse8.v         v0, (a1), t2
    vlse8.v         v8, (t3), t2
    vlse8.v         v16, (t4), t2
    vlse8.v         v24, (t5), t2
    vaaddu.vv       v0, v0, v8
    vaaddu.vv       v8, v16, v24
    vse8.v          v0, (a0)
    vse8.v          v8, (t6)
#endif
    ret
endfunc

// void scale2D_64to32(pixel* dst, const pixel* src, intptr_t stride)
function PFX(scale2D_64to32_v)
    li              t1, 32
    li              t2, 32

#if HIGH_BIT_DEPTH
    vsetvli         zero, t1, e16, m4, ta, ma
    li              t3, 4
    slli            a2, a2, 1
    slli            a3, a2, 1
loop_scale2D_64to32:
    addi            t4, a1, 2
    add             t5, a1, a2
    add             t6, t4, a2
    vlse16.v        v0, (a1), t3
    vlse16.v        v4, (t4), t3
    vlse16.v        v8, (t5), t3
    vlse16.v        v12, (t6), t3
    vadd.vv         v0, v0, v4
    vadd.vv         v0, v0, v8
    vadd.vv         v0, v0, v12
    vadd.vi         v0, v0, 2
    vsrl.vi         v0, v0, 2
    vse16.v         v0, (a0)
    addi            a0, a0, 64
#else
    vsetvli         zero, t1, e8, m2, ta, ma
    li              t3, 2
    slli            a3, a2, 1
loop_scale2D_64to32:
    addi            t4, a1, 1
    add             t5, a1, a2
    add             t6, t4, a2
    vlse8.v         v0, (a1), t3
    vlse8.v         v4, (t4), t3
    vlse8.v         v8, (t5), t3
    vlse8.v         v12, (t6), t3
    vwaddu.vv       v16, v0, v4
    vwaddu.vv       v20, v8, v12
    vsetvli         zero, t1, e16, m4, ta, ma
    vadd.vv         v4, v16, v20
    vadd.vi         v4, v4, 2
    vsetvli         zero, t1, e8, m2, ta, ma
    vnsrl.wi        v0, v4, 2
    vse8.v          v0, (a0)
    addi            a0, a0, 32
#endif
    addi            t2, t2, -1
    add             a1, a1, a3
    bgtz            t2, loop_scale2D_64to32

    ret
endfunc

// void dequant_scaling_c(const int16_t* quantCoef, const int32_t* deQuantCoef, int16_t* coef, int num, int per, int shift)
function PFX(dequant_scaling_v)
    addi            a5, a5, 4
    bge             a4, a5, dequant_scaling2
    sub             a5, a5, a4
loop_dequant_scaling1:
    vsetvli         t1, a3, e32, m8, ta, ma
    vle16.v         v0, (a0)
    vle32.v         v16, (a1)
    vsext.vf2       v8, v0
    vmul.vv         v0, v8, v16
    vsetvli         zero, t1, e16, m4, ta, ma
    vnclip.wx       v8, v0, a5
    vse16.v         v8, (a2)
    sub             a3, a3, t1
    slli            t2, t1, 1
    slli            t3, t1, 2
    add             a0, a0, t2
    add             a2, a2, t2
    add             a1, a1, t3
    bgtz            a3, loop_dequant_scaling1
    ret
    
dequant_scaling2:
    sub             a5, a4, a5
loop_dequant_scaling2:
    vsetvli         t1, a3, e32, m8, ta, ma
    vle16.v         v0, (a0)
    vle32.v         v16, (a1)
    vsext.vf2       v8, v0
    vmul.vv         v0, v8, v16
    vsetvli         zero, t1, e16, m4, ta, ma
    vnclip.wi       v8, v0, 0
    vsetvli         zero, t1, e32, m8, ta, ma
    vsext.vf2       v0, v8
    vsll.vx         v0, v0, a5
    vsetvli         zero, t1, e16, m4, ta, ma
    vnclip.wi       v8, v0, 0
    vse16.v         v8, (a2)
    sub             a3, a3, t1
    slli            t2, t1, 1
    slli            t3, t1, 2
    add             a0, a0, t2
    add             a2, a2, t2
    add             a1, a1, t3
    bgtz            a3, loop_dequant_scaling2
    ret
endfunc

// void dequant_normal_c(const int16_t* quantCoef, int16_t* coef, int num, int scale, int shift)
function PFX(dequant_normal_v)
loop_dequant_normal:
    vsetvli         t1, a2, e16, m4, ta, ma
    slli            t2, t1, 1
    vle16.v         v0, (a0)
    vsetvli         zero, t1, e32, m8, ta, ma
    vsext.vf2       v16, v0
    vmul.vx         v8, v16, a3
    vsetvli         zero, t1, e16, m4, ta, ma
    vnclip.wx       v0, v8, a4
    vse16.v         v0, (a1)
    sub             a2, a2, t1
    add             a0, a0, t2
    add             a1, a1, t2
    bgtz            a2, loop_dequant_normal
    ret
endfunc

// void ssim_4x4x2_core(const pixel* pix1, intptr_t stride1, const pixel* pix2, intptr_t stride2, int sums[2][4])
function PFX(ssim_4x4x2_core_v)
    vsetivli        zero, 8, e16, m1, ta, ma
    vmv.v.i         v4, 0
    vmv.v.i         v6, 0
    vsetivli        zero, 8, e32, m2, ta, ma
    vmv.v.i         v8, 0
    vmv.v.i         v10, 0
    vmv.v.i         v20, 0
    li              t1, 4
#if HIGH_BIT_DEPTH
    slli            a1, a1, 1
    slli            a3, a3, 1
#endif

loop_ssim_4x4x2_core:
#if HIGH_BIT_DEPTH
    vsetivli        zero, 8, e16, m1, ta, ma
    vle16.v         v0, (a0)
    vle16.v         v1, (a2)
    vadd.vv         v4, v4, v0
    vadd.vv         v6, v6, v1
    vwmaccu.vv      v8, v0, v0
    vwmaccu.vv      v8, v1, v1
    vwmaccu.vv      v10, v0, v1
#else
    vsetivli        zero, 8, e8, m1, ta, ma
    vle8.v          v0, (a0)
    vle8.v          v1, (a2)
    vwaddu.wv       v4, v4, v0
    vwaddu.wv       v6, v6, v1
    vwmulu.vv       v12, v0, v0
    vwmulu.vv       v14, v1, v1
    vwmulu.vv       v16, v0, v1
    vsetivli        zero, 8, e16, m1, ta, ma
    vwaddu.wv       v8, v8, v12
    vwaddu.wv       v8, v8, v14
    vwaddu.wv       v10, v10, v16
#endif
    add             a0, a0, a1
    add             a2, a2, a3
    addi            t1, t1, -1
    bgtz            t1, loop_ssim_4x4x2_core

    vsetivli        zero, 4, e16, m1, ta, ma
    vslidedown.vi   v0, v4, 4
    vslidedown.vi   v1, v6, 4
    vredsum.vs      v12, v4, v20
    vredsum.vs      v13, v6, v20
    vredsum.vs      v14, v0, v20
    vredsum.vs      v15, v1, v20
    vsetivli        zero, 4, e32, m1, ta, ma
    vzext.vf2       v0, v12
    vzext.vf2       v1, v13
    vzext.vf2       v2, v14
    vzext.vf2       v3, v15
    vmv.x.s         t1, v0
    vmv.x.s         t2, v1
    vmv.x.s         t3, v2
    vmv.x.s         t4, v3
    sw              t1, (a4)
    sw              t2, 4(a4)
    sw              t3, 16(a4)
    sw              t4, 20(a4)
    vsetivli        zero, 8, e32, m2, ta, ma
    vslidedown.vi   v12, v8, 4
    vslidedown.vi   v14, v10, 4
    vsetivli        zero, 4, e32, m1, ta, ma
    vredsum.vs      v0, v8, v20
    vredsum.vs      v1, v10, v20
    vredsum.vs      v2, v12, v20
    vredsum.vs      v3, v14, v20
    vmv.x.s         t1, v0
    vmv.x.s         t2, v1
    vmv.x.s         t3, v2
    vmv.x.s         t4, v3
    sw              t1, 8(a4)
    sw              t2, 12(a4)
    sw              t3, 24(a4)
    sw              t4, 28(a4)

    ret
endfunc

// uint32_t quant_c(const int16_t* coef, const int32_t* quantCoeff, int32_t* deltaU, int16_t* qCoef, int qBits, int add, int numCoeff)
function PFX(quant_v)
    vsetvli         zero, a6, e32, m4, ta, ma
    vmv.v.i         v24, 0
    vmv.v.i         v28, 0
    mv              t6, a6
    addi            a7, a4, -8
loop_quant:
    vsetvli         t1, a6, e16, m2, ta, ma
    vle16.v         v4, (a0)
    VABS            v8, v4, v6
    vsetvli         zero, t1, e32, m4, ta, ma
    vzext.vf2       v12, v8
    vsext.vf2       v16, v4
    vmslt.vi        v0, v16, 0
    vle32.v         v20, (a1)
    vmul.vv         v12, v12, v20
    vadd.vx         v8, v12, a5
    vsra.vx         v8, v8, a4
    vsll.vx         v4, v8, a4
    vsub.vv         v4, v12, v4
    vsra.vx         v4, v4, a7
    vse32.v         v4, (a2)
    vrsub.vi        v16, v8, 0
    vmerge.vvm      v20, v8, v16, v0
    vmsne.vi        v0, v8, 0
    vmerge.vim      v12, v28, 1, v0
    vsetvli         zero, t1, e32, m4, tu, ma
    vadd.vv         v24, v24, v12
    vsetvli         zero, t1, e16, m2, ta, ma
    vnclip.wi       v4, v20, 0
    vse16.v         v4, (a3)
    sub             a6, a6, t1
    slli            t2, t1, 1
    slli            t3, t1, 2
    add             a0, a0, t2
    add             a3, a3, t2
    add             a1, a1, t3
    add             a2, a2, t3
    bgtz            a6, loop_quant
    
    vsetvli         zero, t6, e32, m4, ta, ma
    vredsum.vs      v16, v24, v28
    vmv.x.s         a0, v16
    ret
endfunc

// uint32_t nquant_c(const int16_t* coef, const int32_t* quantCoeff, int16_t* qCoef, int qBits, int add, int numCoeff)
function PFX(nquant_v)
    vsetvli         zero, a5, e32, m4, ta, ma
    vmv.v.i         v24, 0
    vmv.v.i         v28, 0
    mv              t6, a5
loop_nquant:
    vsetvli         t1, a5, e16, m2, ta, ma
    vle16.v         v4, (a0)
    VABS            v8, v4, v6
    vsetvli         zero, t1, e32, m4, ta, ma
    vzext.vf2       v12, v8
    vsext.vf2       v16, v4
    vmslt.vi        v0, v16, 0
    vle32.v         v20, (a1)
    vmul.vv         v12, v12, v20
    vadd.vx         v12, v12, a4
    vsra.vx         v12, v12, a3
    vrsub.vi        v16, v12, 0
    vmerge.vvm      v20, v12, v16, v0
    vmsne.vi        v0, v12, 0
    vmerge.vim      v8, v28, 1, v0
    vsetvli         zero, t1, e32, m4, tu, ma
    vadd.vv         v24, v24, v8
    vsetvli         zero, t1, e16, m2, ta, ma
    vnclip.wi       v4, v20, 0
    VABS            v12, v4, v8
    vse16.v         v12, (a2)
    sub             a5, a5, t1
    slli            t2, t1, 1
    slli            t3, t1, 2
    add             a0, a0, t2
    add             a2, a2, t2
    add             a1, a1, t3
    bgtz            a5, loop_nquant
    
    vsetvli         zero, t6, e32, m4, ta, ma
    vredsum.vs      v16, v24, v28
    vmv.x.s         a0, v16
    ret
endfunc

// void ssimDist_c(const pixel* fenc, uint32_t fStride, const pixel* recon, intptr_t rstride, uint64_t *ssBlock, int shift, uint64_t *ac_k)
.macro SSIM_DIST size, lmul, lmul2, lmul4
function PFX(ssimDist\size\()_v)
    li              t1, \size
    li              t2, \size
    vsetvli         zero, t2, e32, m\lmul4, ta, ma
    vmv.v.i         v8, 0
    vmv.v.i         v16, 0
    vsetvli         zero, t2, e8, m\lmul, ta, ma
    vmv.v.x         v28, a5
    vmv1r.v         v31, v8
loop_ssimDist\size\():
    vsetvli         zero, t2, e8, m\lmul, ta, ma
    vle8.v          v0, (a0)
    vle8.v          v2, (a2)
    vwsubu.vv       v4, v0, v2
    vsrl.vv         v0, v0, v28
    vsetvli         zero, t2, e16, m\lmul2, ta, ma
    vwmacc.vv       v8, v4, v4
    vzext.vf2       v4, v0
    vwmacc.vv       v16, v4, v4
    add             a0, a0, a1
    add             a2, a2, a3
    addi            t1, t1, -1
    bgtz            t1, loop_ssimDist\size
    vsetvli         zero, t2, e32, m\lmul4, ta, ma
    vredsum.vs      v0, v8, v31
    vredsum.vs      v1, v16, v31
    vmv.x.s         t1, v0
    vmv.x.s         t2, v1
    sd              t1, (a4)
    sd              t2, (a6)
    ret
endfunc
.endm

.macro SSIM_DIST2 size, lmul, lmul2, lmul4
function PFX(ssimDist\size\()_v)
    li              t1, \size
    li              t2, \size
    vsetvli         zero, t2, e32, m\lmul4, ta, ma
    vmv.v.i         v8, 0
    vmv.v.i         v16, 0
    vsetvli         zero, t2, e8, m\lmul, ta, ma
    vmv.v.x         v30, a5
    vmv1r.v         v31, v8
loop_ssimDist\size\():
    vsetvli         zero, t2, e8, m\lmul, ta, ma
    add             t3, a0, a1
    add             t4, a2, a3
    vle8.v          v0, (a0)
    vle8.v          v4, (t3)
    vle8.v          v2, (a2)
    vle8.v          v6, (t4)
    add             a0, t3, a1
    add             a2, t4, a3
    vwsubu.vv       v12, v0, v2
    vwsubu.vv       v20, v4, v6
    vsrl.vv         v0, v0, v30
    vsrl.vv         v2, v4, v30
    vsetvli         zero, t2, e16, m\lmul2, ta, ma
    vwmacc.vv       v8, v12, v12
    vwmacc.vv       v8, v20, v20
    vzext.vf2       v4, v0
    vzext.vf2       v24, v2
    vwmacc.vv       v16, v4, v4
    vwmacc.vv       v16, v24, v24
    addi            t1, t1, -2
    bgtz            t1, loop_ssimDist\size
    vsetvli         zero, t2, e32, m\lmul4, ta, ma
    vredsum.vs      v0, v8, v31
    vredsum.vs      v1, v16, v31
    vmv.x.s         t1, v0
    vmv.x.s         t2, v1
    sd              t1, (a4)
    sd              t2, (a6)
    ret
endfunc
.endm

SSIM_DIST2 4,  1, 1, 1
SSIM_DIST2 8,  1, 1, 2
SSIM_DIST2 16, 1, 2, 4
SSIM_DIST  32, 2, 4, 8

// void ssimDist_c(const pixel* fenc, uint32_t fStride, const pixel* recon, intptr_t rstride, uint64_t *ssBlock, int shift, uint64_t *ac_k)
function PFX(ssimDist64_v)
    li              t1, 64
    vsetvli         zero, t1, e32, m8, ta, ma
    vmv.v.i         v8, 0
    vmv.v.i         v16, 0
    vmv1r.v         v31, v8
loop_ssimDist642:
    li              t2, 64
    mv              t4, a0
    mv              t5, a2
loop_ssimDist641:
    vsetvli         t3, t2, e8, m2, ta, ma
    vle8.v          v0, (t4)
    vle8.v          v2, (t5)
    vwsubu.vv       v4, v0, v2
    vsrl.vx         v0, v0, a5
    vsetvli         zero, t3, e16, m4, ta, ma
    vwmacc.vv       v8, v4, v4
    vzext.vf2       v4, v0
    vwmacc.vv       v16, v4, v4
    add             t4, t4, t3
    add             t5, t5, t3
    sub             t2, t2, t3
    bgtz            t2, loop_ssimDist641
    add             a0, a0, a1
    add             a2, a2, a3
    addi            t1, t1, -1
    bgtz            t1, loop_ssimDist642
    vsetvli         zero, t3, e32, m8, ta, ma
    vredsum.vs      v0, v8, v31
    vredsum.vs      v1, v16, v31
    vmv.x.s         t1, v0
    vmv.x.s         t2, v1
    sd              t1, (a4)
    sd              t2, (a6)
    ret
endfunc

// void normFact_c(const pixel* src, uint32_t blockSize, int shift, uint64_t *z_k)
function PFX(normFact_v)
    mul             a1, a1, a1
    vsetvli         zero, a1, e32, m8, ta, ma
    vmv.v.i         v24, 0
    vmv1r.v         v20, v24
loop_normFact:
    vsetvli         t1, a1, e8, m2, ta, ma
    vle8.v          v0, (a0)
    vsrl.vx         v0, v0, a2
    vwmulu.vv       v4, v0, v0
    vsetvli         zero, t1, e16, m4, ta, ma
    vwaddu.wv       v24, v24, v4
    sub             a1, a1, t1
    add             a0, a0, t1
    bgtz            a1, loop_normFact
    vsetvli         zero, t1, e32, m8, ta, ma
    vredsum.vs      v8, v24, v20
    vmv.x.s         t2, v8
    sd              t2, (a3)
    ret
endfunc

// int scanPosLast_c(const uint16_t *scan, const coeff_t *coeff, uint16_t *coeffSign,
//    uint16_t *coeffFlag, uint8_t *coeffNum, int numSig, const uint16_t*, const int)
function PFX(scanPosLast_v)
    li              t1, 64
    li              t6, 0
    vsetvli         zero, t1, e16, m8, ta, ma
    vmv.v.i         v16, 0
    vse16.v         v16, (a2)
    vse16.v         v16, (a3)
    vse8.v          v16, (a4)
    li              t3, 0x8000
loop_scanPosLast:
    vsetivli        zero, 16, e16, m2, ta, ma
    vle16.v         v2, (a0)
    vsll.vi         v2, v2, 1
    vluxei16.v      v4, (a1), v2
    vmsne.vi        v0, v4, 0
    viota.m         v30, v0
    vcpop.m         t2, v0
    vmerge.vim      v12, v16, 1, v0
    vid.v           v14
    vsrl.vi         v4, v4, 15
    vsll.vv         v4, v4, v30
    bge             t2, a5, tail_scanPosLast
    vrsub.vi        v14, v14, 15
    vsll.vv         v12, v12, v14
    vredor.vs       v2, v12, v16
    vredsum.vs      v10, v4, v16
    vmv.x.s         t5, v2
    vmv.x.s         t4, v10
    sh              t4, (a2)
    sb              t2, (a4)
    sh              t5, (a3)
    sub             a5, a5, t2
    addi            a0, a0, 32
    addi            a2, a2, 2
    addi            a3, a3, 2
    addi            a4, a4, 1
    addi            t6, t6, 16
    bgtz            a5, loop_scanPosLast
tail_scanPosLast:
    li              t0, 16
    vmv.x.s         t5, v0
    and             t4, t3, t5
    sub             t1, t2, a5
    add             t1, t1, t4
    beq             t1, t3, finish_scanPosLast
    vmseq.vx        v28, v30, a5
    vmsbf.m         v0, v28
    vcpop.m         t0, v0
finish_scanPosLast:
    vsetvli         zero, t0, e16, m2, ta, ma
    addi            t1, t0, -1
    vrsub.vx        v14, v14, t1
    vsll.vv         v12, v12, v14
    vredor.vs       v2, v12, v16
    vredsum.vs      v10, v4, v16
    vmv.x.s         t4, v10
    vmv.x.s         t5, v2
    sh              t4, (a2)
    sh              t5, (a3)
    sb              a5, (a4)
    add             a0, t6, t1
    ret
endfunc

//void weight_pp_c(const pixel* src, pixel* dst, intptr_t stride, int width, int height, int w0, int round, int shift, int offset)
function PFX(weight_pp_v)
    ld              t0, (sp)
    li              t6, (1 << BIT_DEPTH) - 1
    addi            a7, a7, BIT_DEPTH - 14
    addi            t1, a7, -1
    li              t2, 1
    sll             a6, t2, t1
#if HIGH_BIT_DEPTH
    slli            a2, a2, 1
    vsetvli         zero, a3, e16, m1, ta, ma
    vmv.v.x         v1, a5
    vsetvli         zero, a3, e32, m2, ta, ma
#else
    sll             t3, t2, a7
    addi            t3, t3, -1
    and             t3, t3, a5
    beqz            t3, fast_weight_pp
    vsetvli         zero, a3, e8, m1, ta, ma
    vmv.v.x         v1, a5
    vsetvli         zero, a3, e32, m4, ta, ma
#endif
    vmv.v.i         v12, 0
    vmv.v.x         v16, t6
    vmv.v.x         v20, a6
    vmv.v.x         v24, a7
    vmv.v.x         v28, t0
loop_weight_pp_h:
    mv              t1, a0
    mv              t2, a1
    mv              t3, a3
loop_weight_pp_w:
#if HIGH_BIT_DEPTH
    vsetvli         t4, t3, e16, m1, ta, ma
    vle16.v         v0, (t1)
    vwmul.vv        v4, v0, v1
    vsetvli         zero, t4, e32, m2, ta, ma
#else
    vsetvli         t4, t3, e8, m1, ta, ma
    vle8.v          v0, (t1)
    vwmulu.vv       v2, v0, v1
    vsetvli         zero, t4, e32, m4, ta, ma
    vzext.vf2       v4, v2
#endif
    vadd.vv         v4, v4, v20
    vsra.vv         v4, v4, v24
    vadd.vv         v4, v4, v28
    vmax.vv         v4, v4, v12
    vmin.vv         v4, v4, v16
#if HIGH_BIT_DEPTH
    vsetvli         zero, t4, e16, m1, ta, ma
    vnsrl.wi        v8, v4, 0
    vse16.v         v8, (t2)
#else
    vsetvli         zero, t4, e16, m2, ta, ma
    vnsrl.wi        v8, v4, 0
    vsetvli         zero, t4, e8, m1, ta, ma
    vnsrl.wi        v4, v8, 0
    vse8.v          v4, (t2)
#endif
    sub             t3, t3, t4
#if HIGH_BIT_DEPTH
    slli            t4, t4, 1
#endif
    add             t1, t1, t4
    add             t2, t2, t4
    bgtz            t3, loop_weight_pp_w
    addi            a4, a4, -1
    add             a0, a0, a2
    add             a1, a1, a2
    bgtz            a4, loop_weight_pp_h
    ret
fast_weight_pp:
    add             a5, a5, a6
    sra             a5, a5, a7
    vsetvli         zero, a3, e8, m1, ta, ma
    vmv.v.x         v1, a5
    vsetvli         zero, a3, e16, m2, ta, ma
    vmv.v.i         v12, 0
    vmv.v.x         v14, t6
    vmv.v.x         v16, t0
loop_weight_pp_fh:
    mv              t1, a0
    mv              t2, a1
    mv              t3, a3
loop_weight_pp_fw:
    vsetvli         t4, t3, e8, m1, ta, ma
    vle8.v          v0, (t1)
    vwmulu.vv       v2, v0, v1
    vsetvli         zero, t4, e16, m2, ta, ma
    vadd.vv         v2, v2, v16
    vmax.vv         v2, v2, v12
    vmin.vv         v2, v2, v14
    vsetvli         zero, t4, e8, m1, ta, ma
    vnsrl.wi        v4, v2, 0
    vse8.v          v4, (t2)
    sub             t3, t3, t4
    add             t1, t1, t4
    add             t2, t2, t4
    bgtz            t3, loop_weight_pp_fw
    addi            a4, a4, -1
    add             a0, a0, a2
    add             a1, a1, a2
    bgtz            a4, loop_weight_pp_fh
    ret
endfunc

//void weight_sp_c(const int16_t* src, pixel* dst, intptr_t srcStride, intptr_t dstStride, int width, int height, int w0, int round, int shift, int offset)
function PFX(weight_sp_v)
    ld              t0, (sp)
    ld              t1, 8(sp)
    li              t6, (1 << BIT_DEPTH) - 1
    li              t2, 1 << 13
    mul             t2, t2, a6
    add             a7, a7, t2
    slli            a2, a2, 1
#if HIGH_BIT_DEPTH
    slli            a3, a3, 1
#endif
    vsetvli         zero, a3, e16, m1, ta, ma
    vmv.v.x         v1, a6
    vsetvli         zero, a3, e32, m2, ta, ma
    vmv.v.i         v12, 0
    vmv.v.x         v16, t6
    vmv.v.x         v20, a7
    vmv.v.x         v24, t0
    vmv.v.x         v28, t1
loop_weight_sp_h:
    mv              t1, a0
    mv              t2, a1
    mv              t3, a4
loop_weight_sp_w:
    vsetvli         t4, t3, e16, m1, ta, ma
    vle16.v         v0, (t1)
    vwmul.vv        v4, v0, v1
    vsetvli         zero, t4, e32, m2, ta, ma
    vadd.vv         v4, v4, v20
    vsra.vv         v4, v4, v24
    vadd.vv         v4, v4, v28
    vmax.vv         v4, v4, v12
    vmin.vv         v4, v4, v16
    vsetvli         zero, t4, e16, m1, ta, ma
    vnsrl.wi        v8, v4, 0
#if HIGH_BIT_DEPTH
    vse16.v         v8, (t2)
#else
    vsetvli         zero, t4, e8, m1, ta, ma
    vnsrl.wi        v4, v8, 0
    vse8.v          v4, (t2)
#endif
    sub             t3, t3, t4
    slli            t5, t4, 1
#if HIGH_BIT_DEPTH
    add             t2, t2, t5
#else
    add             t2, t2, t4
#endif
    add             t1, t1, t5
    bgtz            t3, loop_weight_sp_w
    addi            a5, a5, -1
    add             a0, a0, a2
    add             a1, a1, a3
    bgtz            a5, loop_weight_sp_h
    ret
endfunc

//void addAvg(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride)
.macro ADDAVG_X_Y bx, by, lmul1, lmul2, lmul3
function PFX(addAvg_\bx\()x\by\()_v)
#if HIGH_BIT_DEPTH
    slli            a5, a5, 1
#endif
    slli            a3, a3, 1
    slli            a4, a4, 1
    li              a6, \by
    li              a7, (1 << (14 - BIT_DEPTH)) + 2 * (1 << 13)
    li              t6, (1 << BIT_DEPTH) - 1
    li              t1, \bx
    vsetvli         zero, t1, e32, m\lmul3, ta, ma
    vmv.v.i         v28, 0
    vmv.v.x         v24, t6
    vmv.v.x         v20, a7
loop_addAvg_\bx\()_\by\()_y:
    li              t1, \bx
    mv              t2, a0
    mv              t3, a1
    mv              t4, a2
loop_addAvg_\bx\()_\by\()_x:
    vsetvli         t5, t1, e16, m\lmul2, ta, ma
    vle16.v         v0, (t2)
    vle16.v         v8, (t3)
    vwadd.vv        v4, v0, v8
    vsetvli         zero, t5, e32, m\lmul3, ta, ma
    vadd.vv         v4, v4, v20
    vsra.vi         v4, v4, (15 - BIT_DEPTH)
    vmax.vv         v4, v4, v28
    vmin.vv         v4, v4, v24
    vsetvli         zero, t5, e16, m\lmul2, ta, ma
    vnsrl.wi        v8, v4, 0
#if HIGH_BIT_DEPTH
    vse16.v         v8, (t4)
#else
    vsetvli         zero, t5, e8, m\lmul1, ta, ma
    vnsrl.wi        v4, v8, 0
    vse8.v          v4, (t4)
#endif
    slli            t6, t5, 1
    sub             t1, t1, t5
    add             t2, t2, t6
    add             t3, t3, t6
#if HIGH_BIT_DEPTH
    add             t4, t4, t6
#else
    add             t4, t4, t5
#endif
    bgtz            t1, loop_addAvg_\bx\()_\by\()_x
    add             a0, a0, a3
    add             a1, a1, a4
    add             a2, a2, a5
    addi            a6, a6, -1
    bgtz            a6, loop_addAvg_\bx\()_\by\()_y
    ret
endfunc
.endm

.macro ADDAVG_X_Y2 bx, by, lmul1, lmul2, lmul3
function PFX(addAvg_\bx\()x\by\()_v)
#if HIGH_BIT_DEPTH
    slli            a5, a5, 1
#endif
    slli            a3, a3, 1
    slli            a4, a4, 1
    li              a6, \by
    li              a7, (1 << (14 - BIT_DEPTH)) + 2 * (1 << 13)
    li              t6, (1 << BIT_DEPTH) - 1
    li              t1, \bx
    vsetvli         zero, t1, e32, m\lmul3, ta, ma
    vmv.v.i         v28, 0
    vmv.v.x         v24, t6
    vmv.v.x         v20, a7
loop_addAvg_\bx\()_\by\()_y:
    vsetvli         t5, t1, e16, m\lmul2, ta, ma
    add             t2, a0, a3
    add             t3, a1, a4
    vle16.v         v0, (a0)
    vle16.v         v4, (t2)
    vle16.v         v8, (a1)
    vle16.v         v12, (t3)
    add             a0, t2, a3
    add             a1, t3, a4
    vwadd.vv        v16, v0, v8
    vwadd.vv        v8, v4, v12
    vsetvli         zero, t5, e32, m\lmul3, ta, ma
    vadd.vv         v16, v16, v20
    vadd.vv         v8, v8, v20
    vsra.vi         v16, v16, (15 - BIT_DEPTH)
    vsra.vi         v8, v8, (15 - BIT_DEPTH)
    vmax.vv         v16, v16, v28
    vmax.vv         v8, v8, v28
    vmin.vv         v16, v16, v24
    vmin.vv         v8, v8, v24
    vsetvli         zero, t5, e16, m\lmul2, ta, ma
    vnsrl.wi        v0, v16, 0
    vnsrl.wi        v4, v8, 0
    add             t3, a2, a5
#if HIGH_BIT_DEPTH
    vse16.v         v0, (a2)
    vse16.v         v4, (t3)
#else
    vsetvli         zero, t5, e8, m\lmul1, ta, ma
    vnsrl.wi        v8, v0, 0
    vnsrl.wi        v12, v4, 0
    vse8.v          v8, (a2)
    vse8.v          v12, (t3)
#endif
    add             a2, t3, a5
    addi            a6, a6, -2
    bgtz            a6, loop_addAvg_\bx\()_\by\()_y
    ret
endfunc
.endm

ADDAVG_X_Y2  4, 2, 1, 1, 1
ADDAVG_X_Y2  4, 4, 1, 1, 1
ADDAVG_X_Y2  4, 8, 1, 1, 1
ADDAVG_X_Y2  4, 16, 1, 1, 1
ADDAVG_X_Y2  4, 32, 1, 1, 1
ADDAVG_X_Y2  6, 8, 1, 1, 2
ADDAVG_X_Y2  6, 16, 1, 1, 2
ADDAVG_X_Y2  8, 2, 1, 1, 2
ADDAVG_X_Y2  8, 4, 1, 1, 2
ADDAVG_X_Y2  8, 6, 1, 1, 2
ADDAVG_X_Y2  8, 8, 1, 1, 2
ADDAVG_X_Y2  8, 12, 1, 1, 2
ADDAVG_X_Y2  8, 16, 1, 1, 2
ADDAVG_X_Y2  8, 32, 1, 1, 2
ADDAVG_X_Y2  8, 64, 1, 1, 2
ADDAVG_X_Y2  12, 16, 1, 2, 4
ADDAVG_X_Y2  12, 32, 1, 2, 4
ADDAVG_X_Y2  16, 4, 1, 2, 4
ADDAVG_X_Y2  16, 8, 1, 2, 4
ADDAVG_X_Y2  16, 12, 1, 2, 4
ADDAVG_X_Y2  16, 16, 1, 2, 4
ADDAVG_X_Y2  16, 24, 1, 2, 4
ADDAVG_X_Y2  16, 32, 1, 2, 4
ADDAVG_X_Y2  16, 64, 1, 2, 4
ADDAVG_X_Y  24, 32, 1, 2, 4
ADDAVG_X_Y  24, 64, 1, 2, 4
ADDAVG_X_Y  32, 8, 1, 2, 4
ADDAVG_X_Y  32, 12, 1, 2, 4
ADDAVG_X_Y  32, 16, 1, 2, 4
ADDAVG_X_Y  32, 24, 1, 2, 4
ADDAVG_X_Y  32, 32, 1, 2, 4
ADDAVG_X_Y  32, 48, 1, 2, 4
ADDAVG_X_Y  32, 64, 1, 2, 4
ADDAVG_X_Y  48, 64, 1, 2, 4
ADDAVG_X_Y  64, 16, 1, 2, 4
ADDAVG_X_Y  64, 32, 1, 2, 4
ADDAVG_X_Y  64, 48, 1, 2, 4
ADDAVG_X_Y  64, 64, 1, 2, 4

//void blockfill_s_c(int16_t* dst, intptr_t dstride, int16_t val)
.macro BLOCKFILL_S size, lmul
function PFX(blockfill_s_\size\()_v)
    slli            a1, a1, 1
    li              t1, \size
    slli            a4, a1, 1
    add             a5, a4, a1
    vsetvli         zero, t1, e16, m\lmul, ta, ma
    vmv.v.x         v0, a2
loop_blockfill_s_\size:
    add             t3, a0, a1
    add             t4, a0, a4
    add             t5, a0, a5
    vse16.v         v0, (a0)
    vse16.v         v0, (t3)
    vse16.v         v0, (t4)
    vse16.v         v0, (t5)
    add             a0, t5, a1
    addi            t1, t1, -4
    bgtz            t1, loop_blockfill_s_\size
    ret
endfunc
.endm

BLOCKFILL_S 4, 1
BLOCKFILL_S 8, 1
BLOCKFILL_S 16, 2
BLOCKFILL_S 32, 4
BLOCKFILL_S 64, 8

//void cpy2Dto1D_shl(int16_t* dst, const int16_t* src, intptr_t srcStride, int shift)
.macro CPY2DTO1D_SHL size, lmul
function PFX(cpy2Dto1D_shl_\size\()_v)
    slli            a2, a2, 1
    li              t1, \size
    vsetvli         zero, t1, e16, m\lmul, ta, ma
    vmv.v.x         v0, a3
cpy2Dto1D_shl_\size:
    vle16.v         v8, (a1)
    vsll.vv         v8, v8, v0
    vse16.v         v8, (a0)
    addi            a0, a0, 2 * \size
    add             a1, a1, a2
    addi            t1, t1, -1
    bgtz            t1, cpy2Dto1D_shl_\size
    ret
endfunc
.endm

.macro CPY2DTO1D_SHL2 size, lmul
function PFX(cpy2Dto1D_shl_\size\()_v)
    slli            a2, a2, 1
    li              t1, \size
    slli            a4, a2, 1
    add             a5, a4, a2
    vsetvli         zero, t1, e16, m\lmul, ta, ma
    vmv.v.x         v0, a3
cpy2Dto1D_shl_\size:
    add             t3, a1, a2
    add             t4, a1, a4
    add             t5, a1, a5
    vle16.v         v4, (a1)
    vle16.v         v8, (t3)
    vle16.v         v12, (t4)
    vle16.v         v16, (t5)
    add             a1, t5, a2
    vsll.vv         v4, v4, v0
    vsll.vv         v8, v8, v0
    vsll.vv         v12, v12, v0
    vsll.vv         v16, v16, v0
    addi            t3, a0, 2 * \size
    addi            t4, a0, 4 * \size
    addi            t5, a0, 6 * \size
    vse16.v         v4, (a0)
    vse16.v         v8, (t3)
    vse16.v         v12, (t4)
    vse16.v         v16, (t5)
    addi            a0, t5, 2 * \size
    addi            t1, t1, -4
    bgtz            t1, cpy2Dto1D_shl_\size
    ret
endfunc
.endm

CPY2DTO1D_SHL2 4, 1
CPY2DTO1D_SHL2 8, 1
CPY2DTO1D_SHL2 16, 2
CPY2DTO1D_SHL2 32, 4
CPY2DTO1D_SHL  64, 8

//void cpy2Dto1D_shr(int16_t* dst, const int16_t* src, intptr_t srcStride, int shift)
.macro CPY2DTO1D_SHR size, lmul
function PFX(cpy2Dto1D_shr_\size\()_v)
    slli            a2, a2, 1
    li              t1, \size
    addi            t2, a3, -1
    li              t3, 1
    sll             t3, t3, t2
    vsetvli         zero, t1, e16, m\lmul, ta, ma
    vmv.v.x         v0, a3
    vmv.v.x         v24, t3
cpy2Dto1D_shr_\size:
    vle16.v         v8, (a1)
    vadd.vv         v8, v8, v24
    vsra.vv         v8, v8, v0
    vse16.v         v8, (a0)
    addi            a0, a0, 2 * \size
    add             a1, a1, a2
    addi            t1, t1, -1
    bgtz            t1, cpy2Dto1D_shr_\size
    ret
endfunc
.endm

.macro CPY2DTO1D_SHR2 size, lmul
function PFX(cpy2Dto1D_shr_\size\()_v)
    slli            a2, a2, 1
    li              t1, \size
    addi            t2, a3, -1
    li              t3, 1
    sll             t3, t3, t2
    slli            a4, a2, 1
    vsetvli         zero, t1, e16, m\lmul, ta, ma
    vmv.v.x         v0, a3
    vmv.v.x         v24, t3
    add             a5, a4, a2
cpy2Dto1D_shr_\size:
    add             t3, a1, a2
    add             t4, a1, a4
    add             t5, a1, a5
    vle16.v         v4, (a1)
    vle16.v         v8, (t3)
    vle16.v         v12, (t4)
    vle16.v         v16, (t5)
    add             a1, t5, a2
    vadd.vv         v4, v4, v24
    vadd.vv         v8, v8, v24
    vadd.vv         v12, v12, v24
    vadd.vv         v16, v16, v24
    vsra.vv         v4, v4, v0
    vsra.vv         v8, v8, v0
    vsra.vv         v12, v12, v0
    vsra.vv         v16, v16, v0
    addi            t3, a0, 2 * \size
    addi            t4, a0, 4 * \size
    addi            t5, a0, 6 * \size
    vse16.v         v4, (a0)
    vse16.v         v8, (t3)
    vse16.v         v12, (t4)
    vse16.v         v16, (t5)
    addi            a0, t5, 2 * \size
    addi            t1, t1, -4
    bgtz            t1, cpy2Dto1D_shr_\size
    ret
endfunc
.endm

CPY2DTO1D_SHR2 4, 1
CPY2DTO1D_SHR2 8, 1
CPY2DTO1D_SHR2 16, 2
CPY2DTO1D_SHR2 32, 4
CPY2DTO1D_SHR  64, 8

//void cpy1Dto2D_shl(int16_t* dst, const int16_t* src, intptr_t dstStride, int shift)
.macro CPY1DTO2D_SHL size, lmul
function PFX(cpy1Dto2D_shl_\size\()_v)
    slli            a2, a2, 1
    li              t1, \size
    vsetvli         zero, t1, e16, m\lmul, ta, ma
    vmv.v.x         v0, a3
cpy1Dto2D_shl_\size:
    vle16.v         v8, (a1)
    vsll.vv         v8, v8, v0
    vse16.v         v8, (a0)
    addi            a1, a1, 2 * \size
    add             a0, a0, a2
    addi            t1, t1, -1
    bgtz            t1, cpy1Dto2D_shl_\size
    ret
endfunc
.endm

.macro CPY1DTO2D_SHL2 size, lmul
function PFX(cpy1Dto2D_shl_\size\()_v)
    slli            a2, a2, 1
    li              t1, \size
    slli            a4, a2, 1
    add             a5, a4, a2
    vsetvli         zero, t1, e16, m\lmul, ta, ma
    vmv.v.x         v0, a3
cpy1Dto2D_shl_\size:
    addi            t3, a1, 2 * \size
    addi            t4, a1, 4 * \size
    addi            t5, a1, 6 * \size
    vle16.v         v4, (a1)
    vle16.v         v8, (t3)
    vle16.v         v12, (t4)
    vle16.v         v16, (t5)
    addi            a1, t5, 2 * \size
    vsll.vv         v4, v4, v0
    vsll.vv         v8, v8, v0
    vsll.vv         v12, v12, v0
    vsll.vv         v16, v16, v0
    add             t3, a0, a2
    add             t4, a0, a4
    add             t5, a0, a5
    vse16.v         v4, (a0)
    vse16.v         v8, (t3)
    vse16.v         v12, (t4)
    vse16.v         v16, (t5)
    add             a0, t5, a2
    addi            t1, t1, -4
    bgtz            t1, cpy1Dto2D_shl_\size
    ret
endfunc
.endm

CPY1DTO2D_SHL2 4, 1
CPY1DTO2D_SHL2 8, 1
CPY1DTO2D_SHL2 16, 2
CPY1DTO2D_SHL2 32, 4
CPY1DTO2D_SHL  64, 8

//void cpy1Dto2D_shr(int16_t* dst, const int16_t* src, intptr_t dstStride, int shift)
.macro CPY1DTO2D_SHR size, lmul
function PFX(cpy1Dto2D_shr_\size\()_v)
    slli            a2, a2, 1
    li              t1, \size
    addi            t2, a3, -1
    li              t3, 1
    sll             t3, t3, t2
    vsetvli         zero, t1, e16, m\lmul, ta, ma
    vmv.v.x         v0, a3
    vmv.v.x         v24, t3
cpy1Dto2D_shr_\size:
    vle16.v         v8, (a1)
    vadd.vv         v8, v8, v24
    vsra.vv         v8, v8, v0
    vse16.v         v8, (a0)
    addi            a1, a1, 2 * \size
    add             a0, a0, a2
    addi            t1, t1, -1
    bgtz            t1, cpy1Dto2D_shr_\size
    ret
endfunc
.endm

.macro CPY1DTO2D_SHR2 size, lmul
function PFX(cpy1Dto2D_shr_\size\()_v)
    slli            a2, a2, 1
    li              t1, \size
    addi            t2, a3, -1
    li              t3, 1
    sll             t3, t3, t2
    slli            a4, a2, 1
    vsetvli         zero, t1, e16, m\lmul, ta, ma
    vmv.v.x         v0, a3
    vmv.v.x         v24, t3
    add             a5, a4, a2
cpy1Dto2D_shr_\size:
    addi            t3, a1, 2 * \size
    addi            t4, a1, 4 * \size
    addi            t5, a1, 6 * \size
    vle16.v         v4, (a1)
    vle16.v         v8, (t3)
    vle16.v         v12, (t4)
    vle16.v         v16, (t5)
    addi            a1, t5, 2 * \size
    vadd.vv         v4, v4, v24
    vadd.vv         v8, v8, v24
    vadd.vv         v12, v12, v24
    vadd.vv         v16, v16, v24
    vsra.vv         v4, v4, v0
    vsra.vv         v8, v8, v0
    vsra.vv         v12, v12, v0
    vsra.vv         v16, v16, v0
    add             t3, a0, a2
    add             t4, a0, a4
    add             t5, a0, a5
    vse16.v         v4, (a0)
    vse16.v         v8, (t3)
    vse16.v         v12, (t4)
    vse16.v         v16, (t5)
    add             a0, t5, a2
    addi            t1, t1, -4
    bgtz            t1, cpy1Dto2D_shr_\size
    ret
endfunc
.endm

CPY1DTO2D_SHR2 4, 1
CPY1DTO2D_SHR2 8, 1
CPY1DTO2D_SHR2 16, 2
CPY1DTO2D_SHR2 32, 4
CPY1DTO2D_SHR  64, 8

//void blockcopy_pp_c(pixel* a, intptr_t stridea, const pixel* b, intptr_t strideb)
.macro BLOCKCOPY_PP bx, by, lmul
function PFX(blockcopy_pp_\bx\()x\by\()_v)
    li              t1, \by
    li              t2, \bx
#if HIGH_BIT_DEPTH
    slli            a1, a1, 1
    slli            a3, a3, 1
    vsetvli         zero, t2, e16, m\lmul, ta, ma
#else
    vsetvli         zero, t2, e8, m\lmul, ta, ma
#endif
loop_blockcopy_pp_\bx\()x\by:
#if HIGH_BIT_DEPTH
    vle16.v         v0, (a2)
    vse16.v         v0, (a0)
#else
    vle8.v          v0, (a2)
    vse8.v          v0, (a0)
#endif
    add             a0, a0, a1
    add             a2, a2, a3
    addi            t1, t1, -1
    bgtz            t1, loop_blockcopy_pp_\bx\()x\by
    ret
endfunc
.endm

.macro BLOCKCOPY_PP2 bx, by, lmul
function PFX(blockcopy_pp_\bx\()x\by\()_v)
    li              t1, \by
    li              t2, \bx
#if HIGH_BIT_DEPTH
    slli            a1, a1, 1
    slli            a3, a3, 1
    vsetvli         zero, t2, e16, m\lmul, ta, ma
#else
    vsetvli         zero, t2, e8, m\lmul, ta, ma
#endif
    slli            a4, a1, 1
    slli            a6, a3, 1
    add             a5, a4, a1
    add             a7, a6, a3
loop_blockcopy_pp_\bx\()x\by:
    add             t3, a2, a3
    add             t4, a2, a6
    add             t5, a2, a7
#if HIGH_BIT_DEPTH
    vle16.v         v0, (a2)
    vle16.v         v4, (t3)
    vle16.v         v8, (t4)
    vle16.v         v12, (t5)
#else
    vle8.v          v0, (a2)
    vle8.v          v4, (t3)
    vle8.v          v8, (t4)
    vle8.v          v12, (t5)
#endif
    add             a2, t5, a3
    add             t3, a0, a1
    add             t4, a0, a4
    add             t5, a0, a5
#if HIGH_BIT_DEPTH
    vse16.v         v0, (a0)
    vse16.v         v4, (t3)
    vse16.v         v8, (t4)
    vse16.v         v12, (t5)
#else
    vse8.v          v0, (a0)
    vse8.v          v4, (t3)
    vse8.v          v8, (t4)
    vse8.v          v12, (t5)
#endif
    add             a0, t5, a1
    addi            t1, t1, -4
    bgtz            t1, loop_blockcopy_pp_\bx\()x\by
    ret
endfunc
.endm

.macro BLOCKCOPY_PP3 bx, by, lmul
function PFX(blockcopy_pp_\bx\()x\by\()_v)
    li              t1, \by
    li              t2, \bx
#if HIGH_BIT_DEPTH
    slli            a1, a1, 1
    slli            a3, a3, 1
    vsetvli         zero, t2, e16, m\lmul, ta, ma
#else
    vsetvli         zero, t2, e8, m\lmul, ta, ma
#endif
loop_blockcopy_pp_\bx\()x\by:
    add             t3, a2, a3
#if HIGH_BIT_DEPTH
    vle16.v         v0, (a2)
    vle16.v         v4, (t3)
#else
    vle8.v          v0, (a2)
    vle8.v          v4, (t3)
#endif
    add             a2, t3, a3
    add             t4, a0, a1
#if HIGH_BIT_DEPTH
    vse16.v         v0, (a0)
    vse16.v         v4, (t4)
#else
    vse8.v          v0, (a0)
    vse8.v          v4, (t4)
#endif
    add             a0, t4, a1
    addi            t1, t1, -2
    bgtz            t1, loop_blockcopy_pp_\bx\()x\by
    ret
endfunc
.endm

BLOCKCOPY_PP3  4, 2, 1
BLOCKCOPY_PP2  4, 4, 1
BLOCKCOPY_PP2  4, 8, 1
BLOCKCOPY_PP2  4, 16, 1
BLOCKCOPY_PP2  4, 32, 1
BLOCKCOPY_PP2  6, 8, 1
BLOCKCOPY_PP2  6, 16, 1
BLOCKCOPY_PP3  8, 2, 1
BLOCKCOPY_PP2  8, 4, 1
BLOCKCOPY_PP3  8, 6, 1
BLOCKCOPY_PP2  8, 8, 1
BLOCKCOPY_PP2  8, 12, 1
BLOCKCOPY_PP2  8, 16, 1
BLOCKCOPY_PP2  8, 32, 1
BLOCKCOPY_PP2  8, 64, 1
#if HIGH_BIT_DEPTH
BLOCKCOPY_PP2  12, 16, 2
BLOCKCOPY_PP2  12, 32, 2
BLOCKCOPY_PP2  16, 4, 2
BLOCKCOPY_PP2  16, 8, 2
BLOCKCOPY_PP2  16, 12, 2
BLOCKCOPY_PP2  16, 16, 2
BLOCKCOPY_PP2  16, 24, 2
BLOCKCOPY_PP2  16, 32, 2
BLOCKCOPY_PP2  16, 64, 2
BLOCKCOPY_PP2  24, 32, 4
BLOCKCOPY_PP2  24, 64, 4
BLOCKCOPY_PP2  32, 8, 4
BLOCKCOPY_PP2  32, 12, 4
BLOCKCOPY_PP2  32, 16, 4
BLOCKCOPY_PP2  32, 24, 4
BLOCKCOPY_PP2  32, 32, 4
BLOCKCOPY_PP2  32, 48, 4
BLOCKCOPY_PP2  32, 64, 4
BLOCKCOPY_PP   48, 64, 8
BLOCKCOPY_PP   64, 16, 8
BLOCKCOPY_PP   64, 32, 8
BLOCKCOPY_PP   64, 48, 8
BLOCKCOPY_PP   64, 64, 8
#else
BLOCKCOPY_PP2  12, 16, 1
BLOCKCOPY_PP2  12, 32, 1
BLOCKCOPY_PP2  16, 4, 1
BLOCKCOPY_PP2  16, 8, 1
BLOCKCOPY_PP2  16, 12, 1
BLOCKCOPY_PP2  16, 16, 1
BLOCKCOPY_PP2  16, 24, 1
BLOCKCOPY_PP2  16, 32, 1
BLOCKCOPY_PP2  16, 64, 1
BLOCKCOPY_PP2  24, 32, 2
BLOCKCOPY_PP2  24, 64, 2
BLOCKCOPY_PP2  32, 8, 2
BLOCKCOPY_PP2  32, 12, 2
BLOCKCOPY_PP2  32, 16, 2
BLOCKCOPY_PP2  32, 24, 2
BLOCKCOPY_PP2  32, 32, 2
BLOCKCOPY_PP2  32, 48, 2
BLOCKCOPY_PP2  32, 64, 2
BLOCKCOPY_PP2  48, 64, 4
BLOCKCOPY_PP2  64, 16, 4
BLOCKCOPY_PP2  64, 32, 4
BLOCKCOPY_PP2  64, 48, 4
BLOCKCOPY_PP2  64, 64, 4
#endif

//void blockcopy_ss_c(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb)
.macro BLOCKCOPY_SS bx, by, lmul
function PFX(blockcopy_ss_\bx\()x\by\()_v)
    li              t1, \by
    li              t2, \bx
    slli            a1, a1, 1
    slli            a3, a3, 1
    vsetvli         zero, t2, e16, m\lmul, ta, ma
loop_blockcopy_ss_\bx\()x\by:
    vle16.v         v0, (a2)
    vse16.v         v0, (a0)
    add             a0, a0, a1
    add             a2, a2, a3
    addi            t1, t1, -1
    bgtz            t1, loop_blockcopy_ss_\bx\()x\by
    ret
endfunc
.endm

.macro BLOCKCOPY_SS2 bx, by, lmul
function PFX(blockcopy_ss_\bx\()x\by\()_v)
    li              t1, \by
    li              t2, \bx
    slli            a1, a1, 1
    slli            a3, a3, 1
    slli            a4, a1, 1
    slli            a6, a3, 1
    add             a5, a4, a1
    add             a7, a6, a3
    vsetvli         zero, t2, e16, m\lmul, ta, ma
loop_blockcopy_ss_\bx\()x\by:
    add             t3, a2, a3
    add             t4, a2, a6
    add             t5, a2, a7
    vle16.v         v0, (a2)
    vle16.v         v4, (t3)
    vle16.v         v8, (t4)
    vle16.v         v12, (t5)
    add             a2, t5, a3
    add             t3, a0, a1
    add             t4, a0, a4
    add             t5, a0, a5
    vse16.v         v0, (a0)
    vse16.v         v4, (t3)
    vse16.v         v8, (t4)
    vse16.v         v12, (t5)
    add             a0, t5, a1
    addi            t1, t1, -4
    bgtz            t1, loop_blockcopy_ss_\bx\()x\by
    ret
endfunc
.endm

BLOCKCOPY_SS2  4, 4, 1
BLOCKCOPY_SS2  4, 8, 1
BLOCKCOPY_SS2  8, 8, 1
BLOCKCOPY_SS2  8, 16, 1
BLOCKCOPY_SS2  16, 16, 2
BLOCKCOPY_SS2  16, 24, 2
BLOCKCOPY_SS2  16, 32, 2
BLOCKCOPY_SS2  32, 32, 4
BLOCKCOPY_SS2  32, 64, 4
BLOCKCOPY_SS   64, 64, 8

//void blockcopy_sp_c(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb)
.macro BLOCKCOPY_SP bx, by, lmul, lmul2
function PFX(blockcopy_sp_\bx\()x\by\()_v)
#if HIGH_BIT_DEPTH
    slli            a1, a1, 1
#endif
    slli            a3, a3, 1
    li              t1, \by
    li              t2, \bx
    vsetvli         zero, t2, e16, m\lmul, ta, ma
loop_blockcopy_sp_\bx\()x\by:
    vle16.v         v0, (a2)
#if HIGH_BIT_DEPTH
    vse16.v         v0, (a0)
#else
    vsetvli         zero, t2, e8, m\lmul2, ta, ma
    vnsrl.wi        v8, v0, 0
    vse8.v          v8, (a0)
#endif
    add             a0, a0, a1
    add             a2, a2, a3
    addi            t1, t1, -1
    bgtz            t1, loop_blockcopy_sp_\bx\()x\by
    ret
endfunc
.endm

.macro BLOCKCOPY_SP2 bx, by, lmul, lmul2
function PFX(blockcopy_sp_\bx\()x\by\()_v)
#if HIGH_BIT_DEPTH
    slli            a1, a1, 1
#endif
    slli            a3, a3, 1
    slli            a4, a1, 1
    slli            a6, a3, 1
    add             a5, a4, a1
    add             a7, a6, a3
    li              t1, \by
    li              t2, \bx
    vsetvli         zero, t2, e16, m\lmul, ta, ma
loop_blockcopy_sp_\bx\()x\by:
    add             t3, a2, a3
    add             t4, a2, a6
    add             t5, a2, a7
    vle16.v         v0, (a2)
    vle16.v         v4, (t3)
    vle16.v         v8, (t4)
    vle16.v         v12, (t5)
    add             a2, t5, a3
#if HIGH_BIT_DEPTH
    vse16.v         v0, (a0)
#else
    vsetvli         zero, t2, e8, m\lmul2, ta, ma
    add             t3, a0, a1
    add             t4, a0, a4
    add             t5, a0, a5
    vnsrl.wi        v16, v0, 0
    vnsrl.wi        v20, v4, 0
    vnsrl.wi        v24, v8, 0
    vnsrl.wi        v28, v12, 0
    vse8.v          v16, (a0)
    vse8.v          v20, (t3)
    vse8.v          v24, (t4)
    vse8.v          v28, (t5)
#endif
    add             a0, t5, a1
    addi            t1, t1, -4
    bgtz            t1, loop_blockcopy_sp_\bx\()x\by
    ret
endfunc
.endm

BLOCKCOPY_SP2  4, 4, 1, 1
BLOCKCOPY_SP2  4, 8, 1, 1
BLOCKCOPY_SP2  8, 8, 1, 1
BLOCKCOPY_SP2  8, 16, 1, 1
BLOCKCOPY_SP2  16, 16, 2, 1
BLOCKCOPY_SP2  16, 24, 2, 1
BLOCKCOPY_SP2  16, 32, 2, 1
BLOCKCOPY_SP2  32, 32, 4, 2
BLOCKCOPY_SP2  32, 64, 4, 2
BLOCKCOPY_SP   64, 64, 8, 4

//void blockcopy_ps_c(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb)
.macro BLOCKCOPY_PS bx, by, lmul, lmul2
function PFX(blockcopy_ps_\bx\()x\by\()_v)
    li              t1, \by
    li              t2, \bx
    slli            a1, a1, 1
#if HIGH_BIT_DEPTH
    slli            a3, a3, 1
    vsetvli         zero, t2, e16, m\lmul, ta, ma
#else
    vsetvli         zero, t2, e8, m\lmul2, ta, ma
#endif
loop_blockcopy_ps_\bx\()x\by:
#if HIGH_BIT_DEPTH
    vle16.v         v0, (a2)
    vse16.v         v0, (a0)
#else
    vsetvli         zero, t2, e8, m\lmul2, ta, ma
    vle8.v          v0, (a2)
    vsetvli         zero, t2, e16, m\lmul, ta, ma
    vzext.vf2       v8, v0
    vse16.v         v8, (a0)
#endif
    add             a0, a0, a1
    add             a2, a2, a3
    addi            t1, t1, -1
    bgtz            t1, loop_blockcopy_ps_\bx\()x\by
    ret
endfunc
.endm

.macro BLOCKCOPY_PS2 bx, by, lmul, lmul2
function PFX(blockcopy_ps_\bx\()x\by\()_v)
    li              t1, \by
    li              t2, \bx
    slli            a1, a1, 1
#if HIGH_BIT_DEPTH
    slli            a3, a3, 1
    vsetvli         zero, t2, e16, m\lmul, ta, ma
#endif
    slli            a4, a1, 1
    slli            a6, a3, 1
    add             a5, a4, a1
    add             a7, a6, a3
loop_blockcopy_ps_\bx\()x\by:
    add             t3, a2, a3
    add             t4, a2, a6
    add             t5, a2, a7
#if HIGH_BIT_DEPTH
    vle16.v         v0, (a2)
    vle16.v         v4, (t3)
    vle16.v         v8, (t4)
    vle16.v         v12, (t5)
#else
    vsetvli         zero, t2, e8, m\lmul2, ta, ma
    vle8.v          v16, (a2)
    vle8.v          v20, (t3)
    vle8.v          v24, (t4)
    vle8.v          v28, (t5)
    vsetvli         zero, t2, e16, m\lmul, ta, ma
    vzext.vf2       v0, v16
    vzext.vf2       v4, v20
    vzext.vf2       v8, v24
    vzext.vf2       v12, v28
#endif
    add             a2, t5, a3
    add             t3, a0, a1
    add             t4, a0, a4
    add             t5, a0, a5
    vse16.v         v0, (a0)
    vse16.v         v4, (t3)
    vse16.v         v8, (t4)
    vse16.v         v12, (t5)
    add             a0, t5, a1
    addi            t1, t1, -4
    bgtz            t1, loop_blockcopy_ps_\bx\()x\by
    ret
endfunc
.endm

BLOCKCOPY_PS2  4, 4, 1, 1
BLOCKCOPY_PS2  4, 8, 1, 1
BLOCKCOPY_PS2  8, 8, 1, 1
BLOCKCOPY_PS2  8, 16, 1, 1
BLOCKCOPY_PS2  16, 16, 2, 1
BLOCKCOPY_PS2  16, 24, 2, 1
BLOCKCOPY_PS2  16, 32, 2, 1
BLOCKCOPY_PS2  32, 32, 4, 2
BLOCKCOPY_PS2  32, 64, 4, 2
BLOCKCOPY_PS   64, 64, 8, 4

//void planecopy_cp_c(const uint8_t* src, intptr_t srcStride, pixel* dst, intptr_t dstStride, int width, int height, int shift)
function PFX(planecopy_cp_v)
#if HIGH_BIT_DEPTH
    slli            a3, a3, 1
    vsetvli         zero, a4, e16, m2, ta, ma
#else
    vsetvli         zero, a4, e8, m1, ta, ma
#endif
    vmv.v.x         v30, a6

loop_planecopy_cp_y:
    mv              t0, a0
    mv              t1, a2
    mv              t2, a4
loop_planecopy_cp_x:
#if HIGH_BIT_DEPTH
    vsetvli         t3, t2, e8, m1, ta, ma
    slli            t4, t3, 1
    vle8.v          v0, (t0)
    vsetvli         zero, t3, e16, m2, ta, ma
    vzext.vf2       v2, v0
    vsll.vv         v2, v2, v30
    vse16.v         v2, (t1)
#else
    vsetvli         t3, t2, e8, m1, ta, ma
    vle8.v          v0, (t0)
    vsll.vv         v2, v0, v30
    vse8.v          v2, (t1)
#endif
    sub             t2, t2, t3
    add             t0, t0, t3
#if HIGH_BIT_DEPTH
    add             t1, t1, t4
#else
    add             t1, t1, t3
#endif
    bgtz            t2, loop_planecopy_cp_x
    add             a0, a0, a1
    add             a2, a2, a3
    addi            a5, a5, -1
    bgtz            a5, loop_planecopy_cp_y
    ret
endfunc

//void planecopy_sp_c(const uint16_t* src, intptr_t srcStride, pixel* dst, intptr_t dstStride, int width, int height, int shift, uint16_t mask)
function PFX(planecopy_sp_v)
    slli            a1, a1, 1
#if HIGH_BIT_DEPTH
    slli            a3, a3, 1
#endif
    vsetvli         zero, a4, e16, m1, ta, ma
    vmv.v.x         v30, a6
    vmv.v.x         v28, a7

loop_planecopy_sp_y:
    mv              t0, a0
    mv              t1, a2
    mv              t2, a4
loop_planecopy_sp_x:
    vsetvli         t3, t2, e16, m1, ta, ma
    slli            t4, t3, 1
    vle16.v         v0, (t0)
    vsrl.vv         v0, v0, v30
    vand.vv         v0, v0, v28
#if HIGH_BIT_DEPTH
    vse16.v         v0, (t1)
#else
    vsetvli         zero, t3, e8, m1, ta, ma
    vnsrl.wi        v2, v0, 0
    vse8.v          v2, (t1)
#endif
    sub             t2, t2, t3
    add             t0, t0, t4
#if HIGH_BIT_DEPTH
    add             t1, t1, t4
#else
    add             t1, t1, t3
#endif
    bgtz            t2, loop_planecopy_sp_x
    add             a0, a0, a1
    add             a2, a2, a3
    addi            a5, a5, -1
    bgtz            a5, loop_planecopy_sp_y
    ret
endfunc

//void planecopy_pp_shr_c(const pixel* src, intptr_t srcStride, pixel* dst, intptr_t dstStride, int width, int height, int shift)
function PFX(planecopy_pp_shr_v)
#if HIGH_BIT_DEPTH
    slli            a1, a1, 1
    slli            a3, a3, 1
    vsetvli         zero, a4, e16, m1, ta, ma
#else
    vsetvli         zero, a4, e8, m1, ta, ma
#endif
    vmv.v.x         v30, a6

loop_planecopy_pp_shr_y:
    mv              t0, a0
    mv              t1, a2
    mv              t2, a4
loop_planecopy_pp_shr_x:
#if HIGH_BIT_DEPTH
    vsetvli         t3, t2, e16, m1, ta, ma
    slli            t4, t3, 1
    vle16.v         v0, (t0)
    vsrl.vv         v0, v0, v30
    vse16.v         v0, (t1)
#else
    vsetvli         t3, t2, e8, m1, ta, ma
    vle8.v          v0, (t0)
    vsrl.vv         v0, v0, v30
    vse8.v          v0, (t1)
#endif
    sub             t2, t2, t3
#if HIGH_BIT_DEPTH
    add             t0, t0, t4
    add             t1, t1, t4
#else
    add             t0, t0, t3
    add             t1, t1, t3
#endif
    bgtz            t2, loop_planecopy_pp_shr_x
    add             a0, a0, a1
    add             a2, a2, a3
    addi            a5, a5, -1
    bgtz            a5, loop_planecopy_pp_shr_y
    ret
endfunc

//void planecopy_sp_shl_c(const uint16_t* src, intptr_t srcStride, pixel* dst, intptr_t dstStride, int width, int height, int shift, uint16_t mask)
function PFX(planecopy_sp_shl_v)
    slli            a1, a1, 1
#if HIGH_BIT_DEPTH
    slli            a3, a3, 1
#endif
    vsetvli         zero, a4, e16, m1, ta, ma
    vmv.v.x         v30, a6
    vmv.v.x         v28, a7

loop_planecopy_sp_shl_y:
    mv              t0, a0
    mv              t1, a2
    mv              t2, a4
loop_planecopy_sp_shl_x:
    vsetvli         t3, t2, e16, m1, ta, ma
    slli            t4, t3, 1
    vle16.v         v0, (t0)
    vsll.vv         v0, v0, v30
    vand.vv         v0, v0, v28
#if HIGH_BIT_DEPTH
    vse16.v         v0, (t1)
#else
    vsetvli         zero, t3, e8, m1, ta, ma
    vnsrl.wi        v2, v0, 0
    vse8.v          v2, (t1)
#endif
    sub             t2, t2, t3
    add             t0, t0, t4
#if HIGH_BIT_DEPTH
    add             t1, t1, t4
#else
    add             t1, t1, t3
#endif
    bgtz            t2, loop_planecopy_sp_shl_x
    add             a0, a0, a1
    add             a2, a2, a3
    addi            a5, a5, -1
    bgtz            a5, loop_planecopy_sp_shl_y
    ret
endfunc

// uint32_t costCoeffNxN(
//    uint16_t *scan,        // a0
//    coeff_t *coeff,        // a1
//    intptr_t trSize,       // a2
//    uint16_t *absCoeff,    // a3
//    uint8_t *tabSigCtx,    // a4
//    uint16_t scanFlagMask, // a5
//    uint8_t *baseCtx,      // a6
//    int offset,            // a7
//    int scanPosSigOff,     // sp
//    int subPosBase)        // sp + 8, or sp + 4 on APPLE
function PFX(costCoeffNxN_rvv)
    // load coeff
    slli            a2, a2, 1
    vsetivli        zero, 4, e16, m1, ta, ma
    add             t0, a1, a2
    add             t1, t0, a2
    add             t2, t1, a2
    vle16.v         v16, (a1)
    vle16.v         v18, (t0)
    vle16.v         v17, (t1)
    vle16.v         v19, (t2)

    vsetivli        zero, 16, e16, m1, ta, ma
    vslideup.vi     v16, v18, 4
    vslideup.vi     v17, v19, 4
    vslideup.vi     v16, v17, 8

    // loading tabSigCtx
    vsetivli        zero, 16, e8, m1, ta, ma
    vle8.v          v20, (a4)

    vsetivli        zero, 16, e16, m2, ta, ma
    vzext.vf2       v12, v20
    VABS            v6, v16, v30

    // load scan table
    lw              a2, 0(sp)
    xori            t6, a2, 15
    slli            t0, t6, 1
    add             t1, a0, t0
    vle16.v         v8, (t1)
    vmv.v.i         v10, 15
    vxor.vv         v4, v8, v10

    // reorder abs(coeff)
    vrgatherei16.vv  v2, v6, v4

    // reorder tabSigCtx
    vrgatherei16.vv  v14, v12, v4
    vadd.vx          v0, v14, a7

    // register mapping
    // a0 - sum
    // a1 - entropyStateBits
    // {v0, v1} - sigCtx
    // {v2, v3} - abs(coeff)
    // a2 - scanPosSigOff
    // a3 - absCoeff
    // a4 - numNonZero
    // a5 - scanFlagMask
    // a6 - baseCtx
    // a7 - scanPosSigOff & 15
    mv              a7, t6
    li              a0, 0
    la              a1, PFX(entropyStateBits)
    li              a4, 0
    beqz            a2, idx_zero

loop_nonzero:
    addi            a2, a2, -1
    slli            t0, a4, 1
    add             t0, a3, t0
    vmv.x.s         t1, v2
    vmv.x.s         t6, v0              // t6 = ctxSig
    vslidedown.vi   v2, v2, 1
    vslidedown.vi   v0, v0, 1
    sh              t1, 0(t0)           // absCoeff[numNonZero] = tmpCoeff[blkPos]

    andi            t5, a5, 1           // t5  sig = scanFlagMask & 1
    srli            a5, a5, 1           // scanFlagMask >>= 1;
    add             a4, a4, t5          // numNonZero += sig

    add             t0, a6, t6
    lbu             t4, 0(t0)           // mstate = baseCtx[ctxSig] load uint8
    andi            t3, t4, 1           // t3 mps = mstate & 1
    xor             t4, t4, t5          // t4 = mstate ^ sig

    slli            t0, t4, 2
    add             t0, a1, t0
    lw              t1, 0(t0)           // stateBits = PFX(entropyStateBits)[mstate ^ sig]
    add             a0, a0, t1          // sum += stateBits
    srli            t1, t1, 24
    add             t2, t1, t3          // t2 nextState = (stateBits >> 24) + mps

    li              t0, 1
    bne             t4, t0, 1f          // if ((mstate ^ sig) == 1)
    mv              t2, t5              // nextState = sig
1:
    add             t0, a6, t6
    sb              t2, 0(t0)           // baseCtx[ctxSig] = nextState
    bnez            a2, loop_nonzero

idx_zero:
    slli            t0, a4, 1
    add             t0, a3, t0
    vmv.x.s         t1, v2
    vmv.x.s         t6, v0
    sh              t1, 0(t0)           // absCoeff[numNonZero] = tmpCoeff[blkPos]

    lw              t0, 8(sp)
    li              t1, 0xFFFF
    and             t0, t0, t1          // subPosBase &= 0xFFFF
    seqz            a2, t0

    add             a4, a4, a7
    add             a4, a4, a2
    beqz            a4, 2f              // subPosBase != 0 && numNonZero == 0,exit

    addi            a2, a2, -1
    and             t6, t6, a2          // ctxSig &= posZeroMask

    andi            t5, a5, 1           // t5, sig = scanFlagMask & 1

    add             t0, a6, t6
    lbu             t4, 0(t0)           // mstate = baseCtx[ctxSig] load uint8
    andi            t3, t4, 1           // t3 mps = mstate & 1
    xor             t4, t4, t5          // t4 = mstate ^ sig

    slli            t0, t4, 2
    add             t0, a1, t0
    lw              t1, 0(t0)           // stateBits = PFX(entropyStateBits)[mstate ^ sig]
    add             a0, a0, t1          // sum += stateBits
    srli            t1, t1, 24
    add             t2, t1, t3          // t2 nextState = (stateBits >> 24) + mps

    li              t0, 1
    bne             t4, t0, 3f          // if ((mstate ^ sig) == 1)
    mv              t2, t5              // nextState = sig
3:
    add             t0, a6, t6
    sb              t2, 0(t0)           // baseCtx[ctxSig] = nextState
2:
    li              t0, 0xFFFFFF
    and             a0, a0, t0          //sum &=0xFFFFFF
    ret
endfunc
