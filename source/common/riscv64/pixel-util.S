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

PIXEL_VAR 8, 1, 1, 2
PIXEL_VAR 16, 1, 2, 4
PIXEL_VAR 32, 2, 4, 8

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
    slli            t3, a3, 1
#if HIGH_BIT_DEPTH
    vsetvli         zero, t1, e16, m\lmul2, ta, ma
    slli            t2, a3, 1
#else
    vsetvli         zero, t1, e8, m\lmul, ta, ma
    mv              t2, a3
#endif

loop_residual\blocksize:
#if HIGH_BIT_DEPTH
    vle16.v         v0, (a0)
    vle16.v         v4, (a1)
    vsub.vv         v8, v0, v4
#else
    vle8.v          v0, (a0)
    vle8.v          v4, (a1)
    vwsubu.vv       v8, v0, v4
#endif
    vse16.v         v8, (a2)

    addi            t1, t1, -1
    add             a0, a0, t2
    add             a1, a1, t2
    add             a2, a2, t3
    bgtz            t1, loop_residual\blocksize

    ret
endfunc
.endm

GET_RESIDUAL 4,  1, 1
GET_RESIDUAL 8,  1, 1
GET_RESIDUAL 16, 1, 2
GET_RESIDUAL 32, 2, 4

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

SUB_PS  4,  4,  1, 1
SUB_PS  4,  8,  1, 1
SUB_PS  8,  8,  1, 1
SUB_PS  8,  16, 1, 1
SUB_PS  16, 16, 1, 2
SUB_PS  16, 32, 1, 2
SUB_PS  32, 32, 2, 4
SUB_PS  32, 64, 2, 4
SUB_PS  64, 64, 4, 8

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
    vmslt.vi        v0, v16, 0
    vmerge.vim      v8, v16, 0, v0
#if HIGH_BIT_DEPTH
    vmsgtu.vx       v0, v8, t2
    vmerge.vxm      v16, v8, t2, v0
    vse16.v         v16, (a0)
#else
    vsetvli         zero, t0, e8, m\lmul, ta, ma
    vnclipu.wi      v16, v8, 0
    vse8.v          v16, (a0)
#endif
    addi            t1, t1, -1
    add             a0, a0, a1
    add             a2, a2, a4
    add             a3, a3, a5
    bgtz            t1, loop_addps_\xsize\()x\ysize\()

    ret
endfunc
.endm

ADD_PS 4,  4,  1, 1
ADD_PS 4,  8,  1, 1
ADD_PS 8,  8,  1, 1
ADD_PS 8,  16, 1, 1
ADD_PS 16, 16, 1, 2
ADD_PS 16, 32, 1, 2
ADD_PS 32, 64, 2, 4
ADD_PS 32, 32, 2, 4
ADD_PS 64, 64, 4, 8

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
    vrsub.vi        v6, v4, 0
    vmax.vv         v8, v4, v6
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
    vrsub.vi        v6, v4, 0
    vmax.vv         v8, v4, v6
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
    vrsub.vi        v8, v4, 0
    vmax.vv         v12, v4, v8
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
    vmv1r.v         v31, v8
loop_ssimDist\size\():
    vsetvli         zero, t2, e8, m\lmul, ta, ma
    vle8.v          v0, (a0)
    vle8.v          v2, (a2)
    vwsubu.vv       v4, v0, v2
    vsrl.vx         v0, v0, a5
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

SSIM_DIST 4,  1, 1, 1
SSIM_DIST 8,  1, 1, 2
SSIM_DIST 16, 1, 2, 4
SSIM_DIST 32, 2, 4, 8

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
