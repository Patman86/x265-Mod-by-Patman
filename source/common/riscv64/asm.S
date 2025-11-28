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

.macro HADAMARD4_V r1, r2, r3, r4, t1, t2, t3, t4
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
