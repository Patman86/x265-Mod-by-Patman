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

#ifndef ASM_S_  // #include guards
#define ASM_S_

.option arch, +v

#define PFX3(prefix, name) prefix ## _ ## name
#define PFX2(prefix, name) PFX3(prefix, name)
#define PFX(name)          PFX2(X265_NS, name)

#if defined __clang__
#define EXTERN_ASM
#define HAVE_AS_FUNC 0
#define PREFIX 1
#else
#define EXTERN_ASM
#define HAVE_AS_FUNC 1
#endif

#ifdef __ELF__
#define ELF
#else
#ifdef PREFIX
#define ELF #
#else
#define ELF @
#endif
#endif

#if HAVE_AS_FUNC
#define FUNC
#else
#ifdef PREFIX
#define FUNC #
#else
#define FUNC @
#endif
#endif

#define GLUE(a, b) a ## b
#define JOIN(a, b) GLUE(a, b)

#define PFX_C(name)        JOIN(JOIN(JOIN(EXTERN_ASM, X265_NS), _), name)

#define FENC_STRIDE 64
#define FDNC_STRIDE 128

// Alignment of stack arguments of size less than 8 bytes.
#define STACK_ARG_ALIGNMENT 8

// Get offset from SP of stack argument at index `idx`.
#define STACK_ARG_OFFSET(idx) (idx * STACK_ARG_ALIGNMENT)

.macro function name, export=1
    .macro endfunc
ELF     .size   \name, . - \name
FUNC    .endfunc
        .purgem endfunc
    .endm
        .align  2
.if \export == 1
        .global EXTERN_ASM\name
ELF     .hidden EXTERN_ASM\name
ELF     .type   EXTERN_ASM\name, %function
FUNC    .func   EXTERN_ASM\name
EXTERN_ASM\name:
.else
ELF     .hidden \name
ELF     .type   \name, %function
FUNC    .func   \name
\name:
.endif
.endm

.macro  const   name, align=2
    .macro endconst
ELF     .size   \name, . - \name
        .purgem endconst
    .endm
#ifdef __MACH__
    .const_data
#else
    .section .rodata
#endif
    .align          \align
\name:
.endm

.macro SUMSUB_AB sum, diff, a, b
    vadd.vv         \sum,  \a, \b
    vsub.vv         \diff, \a, \b
.endm

.macro SUMSUB_ABCD s1, d1, s2, d2, a, b, c, d
    SUMSUB_AB       \s1, \d1, \a, \b
    SUMSUB_AB       \s2, \d2, \c, \d
.endm

.macro HADAMARD4 r1, r2, r3, r4, t1, t2, t3, t4
    SUMSUB_ABCD     \t1, \t2, \t3, \t4, \r1, \r2, \r3, \r4
    SUMSUB_ABCD     \r1, \r3, \r2, \r4, \t1, \t3, \t2, \t4
.endm

.macro SIGNOF x, t1, t2
    srai            \t1, \x, 31
    neg             \x, \x
    li              \t2, 63
    srl             \x, \x, \t2
    or              \x, \x, \t1
.endm

.macro SIGNOF_RVV  vd, va, vb
    vmv.v.i         \vd, 0
    vmsltu.vv       v0, \va, \vb
    vmerge.vim      \vd, \vd, -1, v0
    vmsgtu.vv       v0, \va, \vb
    vmerge.vim      \vd, \vd, 1, v0
.endm

.macro VABS vd, va, vb
#if HAVE_RVV_ZF
    vabs.v          \vd, \va
#else
    vrsub.vi        \vb, \va, 0
    vmax.vv         \vd, \va, \vb
#endif
.endm

.macro VABD d0, s0, s1, t0
#if HAVE_RVV_ZF
    vabd.vv         \d0, \s0, \s1
#else
    vmax.vv         \d0, \s0, \s1
    vmin.vv         \t0, \s0, \s1
    vsub.vv         \d0, \d0, \t0
#endif
.endm

.macro VABDU d0, s0, s1, t0
#if HAVE_RVV_ZF
    vabdu.vv        \d0, \s0, \s1
#else
    vmaxu.vv        \d0, \s0, \s1
    vminu.vv        \t0, \s0, \s1
    vsub.vv         \d0, \d0, \t0
#endif
.endm

.macro UWABDU d0, s0, s1, t0, t1
    vmaxu.vv        \t1, \s0, \s1
    vminu.vv        \t0, \s0, \s1
    vwsubu.vv       \d0, \t1, \t0
.endm

.macro UWADBDACCU d0, s0, s1, t0, t1
#if HAVE_RVV_ZF
    vwabdaccu.vv    \d0, \s0, \s1
#else
    vmaxu.vv        \t1, \s0, \s1
    vminu.vv        \t0, \s0, \s1
    vsub.vv         \t1, \t1, \t0
    vwaddu.wv       \d0, \d0, \t1
#endif
.endm

// v0 must be 0x55 as mask
.macro TRANSPOSE_8x8 es, d0, d1, d2, d3, d4, d5, d6, d7, r0, r1, r2, r3, r4, r5, r6, r7
    vslideup.vi     \d0, \r1, 1
    vslidedown.vi   \d1, \r0, 1
    vslideup.vi     \d2, \r3, 1
    vslidedown.vi   \d3, \r2, 1
    vslideup.vi     \d4, \r5, 1
    vslidedown.vi   \d5, \r4, 1
    vslideup.vi     \d6, \r7, 1
    vslidedown.vi   \d7, \r6, 1
    vmerge.vvm      \d0, \d0, \r0, v0
    vmerge.vvm      \d1, \r1, \d1, v0
    vmerge.vvm      \d2, \d2, \r2, v0
    vmerge.vvm      \d3, \r3, \d3, v0
    vmerge.vvm      \d4, \d4, \r4, v0
    vmerge.vvm      \d5, \r5, \d5, v0
    vmerge.vvm      \d6, \d6, \r6, v0
    vmerge.vvm      \d7, \r7, \d7, v0

.if \es == 16
    vsetivli        zero, 4, e32, m1, ta, ma
.elseif \es == 8
    vsetivli        zero, 4, e16, m1, ta, ma
.endif

    vslideup.vi     \r0, \d2, 1
    vslidedown.vi   \r1, \d0, 1
    vslideup.vi     \r2, \d3, 1
    vslidedown.vi   \r3, \d1, 1
    vslideup.vi     \r4, \d6, 1
    vslidedown.vi   \r5, \d4, 1
    vslideup.vi     \r6, \d7, 1
    vslidedown.vi   \r7, \d5, 1
    vmerge.vvm      \r0, \r0, \d0, v0
    vmerge.vvm      \r1, \d2, \r1, v0
    vmerge.vvm      \r2, \r2, \d1, v0
    vmerge.vvm      \r3, \d3, \r3, v0
    vmerge.vvm      \r4, \r4, \d4, v0
    vmerge.vvm      \r5, \d6, \r5, v0
    vmerge.vvm      \r6, \r6, \d5, v0
    vmerge.vvm      \r7, \d7, \r7, v0

.if \es == 16
    vsetivli        zero, 2, e64, m1, ta, ma
.elseif \es == 8
    vsetivli        zero, 2, e32, m1, ta, ma
.endif

    vslideup.vi     \d0, \r4, 1
    vslidedown.vi   \d4, \r0, 1
    vslideup.vi     \d1, \r6, 1
    vslidedown.vi   \d5, \r2, 1
    vslideup.vi     \d2, \r5, 1
    vslidedown.vi   \d6, \r1, 1
    vslideup.vi     \d3, \r7, 1
    vslidedown.vi   \d7, \r3, 1
    vmerge.vvm      \d0, \d0, \r0, v0
    vmerge.vvm      \d4, \r4, \d4, v0
    vmerge.vvm      \d1, \d1, \r2, v0
    vmerge.vvm      \d5, \r6, \d5, v0
    vmerge.vvm      \d2, \d2, \r1, v0
    vmerge.vvm      \d6, \r5, \d6, v0
    vmerge.vvm      \d3, \d3, \r3, v0
    vmerge.vvm      \d7, \r7, \d7, v0
.endm

// v0 = 0x5555
.macro TRN_2REG d1, d2, vn, vm
    vslideup.vi     \d1, \vm, 1
    vslidedown.vi   \d2, \vn, 1
    vmerge.vvm      \d1, \d1, \vn, v0
    vmerge.vvm      \d2, \vm, \d2, v0
.endm

// v0 = 0x5555
.macro TRN_4REG d1, d2, d3, d4, s1, s2, s3, s4
    vslideup.vi     \d1, \s2, 1
    vslideup.vi     \d3, \s4, 1
    vslidedown.vi   \d2, \s1, 1
    vslidedown.vi   \d4, \s3, 1
    vmerge.vvm      \d1, \d1, \s1, v0
    vmerge.vvm      \d2, \s2, \d2, v0
    vmerge.vvm      \d3, \d3, \s3, v0
    vmerge.vvm      \d4, \s4, \d4, v0
.endm

// v0 = 0x5555
.macro TRN_8REG d1, d2, d3, d4, d5, d6, d7, d8, s1, s2, s3, s4, s5, s6, s7, s8
    vslideup.vi     \d1, \s2, 1
    vslideup.vi     \d3, \s4, 1
    vslideup.vi     \d5, \s6, 1
    vslideup.vi     \d7, \s8, 1
    vslidedown.vi   \d2, \s1, 1
    vslidedown.vi   \d4, \s3, 1
    vslidedown.vi   \d6, \s5, 1
    vslidedown.vi   \d8, \s7, 1
    vmerge.vvm      \d1, \d1, \s1, v0
    vmerge.vvm      \d2, \s2, \d2, v0
    vmerge.vvm      \d3, \d3, \s3, v0
    vmerge.vvm      \d4, \s4, \d4, v0
    vmerge.vvm      \d5, \d5, \s5, v0
    vmerge.vvm      \d6, \s6, \d6, v0
    vmerge.vvm      \d7, \d7, \s7, v0
    vmerge.vvm      \d8, \s8, \d8, v0
.endm

.macro STRIDE_SHIFT t0, t1, stride0, stride1, shift0, shift1
    slli     \t0, \stride0, \shift0
    slli     \t1, \stride1, \shift1
.endm

.macro ADDPTR t0, t1, s0, s1, stride0, stride1, shift0, shift1
    stride_shift \t0, \t1, \stride0, \stride1, \shift0, \shift1
    add      \s0, \s0, \t0
    add      \s1, \s1, \t1
.endm

#endif
