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

//void saoCuStatsE0_c(const int16_t *diff, const pixel *rec, intptr_t stride, int endX, int endY, int32_t *stats, int32_t *count)
function PFX(saoCuStatsE0_rvv)
    addi            sp, sp, -64
    sd              s0, (sp)
    sd              s1, 8(sp)
    sd              s2, 16(sp)
    sd              s3, 24(sp)
    sd              s4, 32(sp)
    sd              s5, 40(sp)
    sd              s6, 48(sp)
    li              s0, 0
    li              s1, 0
    li              s2, 0
    li              s3, 0
    li              s4, 0
    li              s6, 1 << 7
#if HIGH_BIT_DEPTH
    slli            a2, a2, 1
    vsetvli         t1, zero, e32, m2, ta, ma
#else
    vsetvli         t1, zero, e32, m4, ta, ma
#endif
    vmv.v.i         v4, 0
    vmv.v.i         v8, 0
    vmv.v.i         v12, 0
    vmv.v.i         v16, 0
    vmv.v.i         v20, 0
#if HIGH_BIT_DEPTH
    vsetvli         t1, zero, e16, m1, ta, ma
#else
    vsetvli         t1, zero, e16, m2, ta, ma
#endif
    vmv.v.i         v30, 0

loop_saoCuStatsE0_y:
    mv              t0, a0
    mv              t1, a1
    mv              t2, a3
#if HIGH_BIT_DEPTH
    lhu             t3, -2(a1)
    lhu             t4, (a1)
#else
    lbu             t3, -1(a1)
    lbu             t4, (a1)
#endif
    sub             t3, t3, t4
    SIGNOF          t3, t4, t5
#if HIGH_BIT_DEPTH
    vsetvli         zero, a3, e16, m1, ta, ma
#else
    vsetvli         zero, a3, e8, m1, ta, ma
#endif
    vmv.s.x         v1, t3
loop_saoCuStatsE0_x:
#if HIGH_BIT_DEPTH
    addi            t5, t1, 2
    vsetvli         t3, t2, e16, m1, ta, ma
    vle16.v         v2, (t1)
    vle16.v         v3, (t5)
#else
    addi            t5, t1, 1
    vsetvli         t3, t2, e8, m1, ta, ma
    vle8.v          v2, (t1)
    vle8.v          v3, (t5)
#endif
    addi            t4, t3, -1
    SIGNOF_RVV      v24, v2, v3
    vslideup.vi     v1, v24, 1
    vsub.vv         v25, v24, v1
    vslidedown.vx   v1, v24, t4

    vmseq.vi        v0, v25, -2
    vmseq.vi        v2, v25, -1
    vmseq.vi        v3, v25, 0
    vmseq.vi        v24, v25, 1
    vmseq.vi        v25, v25, 2

    vcpop.m         t4, v0
    vcpop.m         t5, v2
    vcpop.m         t6, v3
    vcpop.m         a7, v24
    vcpop.m         s5, v25
    add             s0, s0, t4
    add             s1, s1, t5
    add             s2, s2, t6
    add             s3, s3, a7
    add             s4, s4, s5

#if HIGH_BIT_DEPTH
    vle16.v         v26, (t0)
    vmerge.vvm      v6, v30, v26, v0
    vmv1r.v         v0, v2
    vmerge.vvm      v10, v30, v26, v0
    vmv1r.v         v0, v3
    vmerge.vvm      v14, v30, v26, v0
    vmv1r.v         v0, v24
    vmerge.vvm      v18, v30, v26, v0
    vmv1r.v         v0, v25
    vmerge.vvm      v22, v30, v26, v0
    vsetvli         zero, t3, e16, m1, tu, ma
    vwadd.wv        v4, v4, v6
    vwadd.wv        v8, v8, v10
    vwadd.wv        v12, v12, v14
    vwadd.wv        v16, v16, v18
    vwadd.wv        v20, v20, v22
#else
    vsetvli         zero, t3, e16, m2, ta, ma
    vle16.v         v26, (t0)
    vmerge.vvm      v28, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v4, v4, v28
    vsetvli         zero, t3, e16, m2, ta, ma
    vmv1r.v         v0, v2
    vmerge.vvm      v28, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v8, v8, v28
    vsetvli         zero, t3, e16, m2, ta, ma
    vmv1r.v         v0, v3
    vmerge.vvm      v28, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v12, v12, v28
    vsetvli         zero, t3, e16, m2, ta, ma
    vmv1r.v         v0, v24
    vmerge.vvm      v28, v30, v26, v0
    vmv1r.v         v0, v25
    vmerge.vvm      v2, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v16, v16, v28
    vwadd.wv        v20, v20, v2
#endif

    slli            t4, t3, 1
#if HIGH_BIT_DEPTH
    add             t1, t1, t4
#else
    add             t1, t1, t3
#endif
    sub             t2, t2, t3
    add             t0, t0, t4
    bgtz            t2, loop_saoCuStatsE0_x
    add             a0, a0, s6
    add             a1, a1, a2
    addi            a4, a4, -1
    bgtz            a4, loop_saoCuStatsE0_y

    lw              t0, (a6)
    lw              t1, 4(a6)
    lw              t2, 8(a6)
    lw              t3, 12(a6)
    lw              t4, 16(a6)
    add             t1, t1, s0
    add             t2, t2, s1
    add             t0, t0, s2
    add             t3, t3, s3
    add             t4, t4, s4
    sw              t0, (a6)
    sw              t1, 4(a6)
    sw              t2, 8(a6)
    sw              t3, 12(a6)
    sw              t4, 16(a6)

#if HIGH_BIT_DEPTH
    vsetvli         zero, a3, e32, m2, ta, ma
#else
    vsetvli         zero, a3, e32, m4, ta, ma
#endif
    vredsum.vs      v24, v4, v30
    vredsum.vs      v25, v8, v30
    vredsum.vs      v26, v12, v30
    vredsum.vs      v27, v16, v30
    vredsum.vs      v28, v20, v30
    vsetvli         zero, a3, e32, m1, ta, ma
    vmv.x.s         s0, v24
    vmv.x.s         s1, v25
    vmv.x.s         s2, v26
    vmv.x.s         s3, v27
    vmv.x.s         s4, v28
    lw              t0, (a5)
    lw              t1, 4(a5)
    lw              t2, 8(a5)
    lw              t3, 12(a5)
    lw              t4, 16(a5)
    add             t1, t1, s0
    add             t2, t2, s1
    add             t0, t0, s2
    add             t3, t3, s3
    add             t4, t4, s4
    sw              t0, (a5)
    sw              t1, 4(a5)
    sw              t2, 8(a5)
    sw              t3, 12(a5)
    sw              t4, 16(a5)

    ld              s0, (sp)
    ld              s1, 8(sp)
    ld              s2, 16(sp)
    ld              s3, 24(sp)
    ld              s4, 32(sp)
    ld              s5, 40(sp)
    ld              s6, 48(sp)
    addi            sp, sp, 64
    ret
endfunc

//void saoCuStatsE1_c(const int16_t *diff, const pixel *rec, intptr_t stride, int8_t *upBuff1, int endX, int endY, int32_t *stats, int32_t *count)
function PFX(saoCuStatsE1_rvv)
    addi            sp, sp, -64
    sd              s0, (sp)
    sd              s1, 8(sp)
    sd              s2, 16(sp)
    sd              s3, 24(sp)
    sd              s4, 32(sp)
    sd              s5, 40(sp)
    sd              s6, 48(sp)
    sd              s7, 56(sp)
    li              s0, 0
    li              s1, 0
    li              s2, 0
    li              s3, 0
    li              s4, 0
    li              s6, 1 << 7
#if HIGH_BIT_DEPTH
    slli            a2, a2, 1
    vsetvli         t1, zero, e32, m2, ta, ma
#else
    vsetvli         t1, zero, e32, m4, ta, ma
#endif
    vmv.v.i         v4, 0
    vmv.v.i         v8, 0
    vmv.v.i         v12, 0
    vmv.v.i         v16, 0
    vmv.v.i         v20, 0
#if HIGH_BIT_DEPTH
    vsetvli         t1, zero, e16, m1, ta, ma
#else
    vsetvli         t1, zero, e16, m2, ta, ma
#endif
    vmv.v.i         v30, 0

loop_saoCuStatsE1_y:
    mv              t0, a0
    mv              t1, a1
    mv              t2, a4
    mv              t4, a3
loop_saoCuStatsE1_x:
    add             t5, t1, a2
#if HIGH_BIT_DEPTH
    vsetvli         t3, t2, e16, m1, ta, ma
    vle16.v         v2, (t1)
    vle16.v         v3, (t5)
    vle8.v          v1, (t4)
    vsext.vf2       v26, v1
    SIGNOF_RVV      v24, v2, v3
    vadd.vv         v25, v24, v26
    vsetvli         zero, t3, e8, m1, ta, ma
    vnsrl.wi        v26, v24, 0
    vrsub.vi        v1, v26, 0
    vse8.v          v1, (t4)
    vsetvli         zero, t3, e16, m1, ta, ma
#else
    vsetvli         t3, t2, e8, m1, ta, ma
    vle8.v          v2, (t1)
    vle8.v          v3, (t5)
    vle8.v          v1, (t4)
    SIGNOF_RVV      v24, v2, v3
    vadd.vv         v25, v24, v1
    vrsub.vi        v1, v24, 0
    vse8.v          v1, (t4)
#endif

    vmseq.vi        v0, v25, -2
    vmseq.vi        v2, v25, -1
    vmseq.vi        v3, v25, 0
    vmseq.vi        v24, v25, 1
    vmseq.vi        v25, v25, 2

    vcpop.m         s7, v0
    add             s0, s0, s7
    vcpop.m         t5, v2
    vcpop.m         t6, v3
    vcpop.m         s7, v24
    vcpop.m         s5, v25
    add             s1, s1, t5
    add             s2, s2, t6
    add             s3, s3, s7
    add             s4, s4, s5

#if HIGH_BIT_DEPTH
    vle16.v         v26, (t0)
    vmerge.vvm      v6, v30, v26, v0
    vmv1r.v         v0, v2
    vmerge.vvm      v10, v30, v26, v0
    vmv1r.v         v0, v3
    vmerge.vvm      v14, v30, v26, v0
    vmv1r.v         v0, v24
    vmerge.vvm      v18, v30, v26, v0
    vmv1r.v         v0, v25
    vmerge.vvm      v22, v30, v26, v0
    vsetvli         zero, t3, e16, m1, tu, ma
    vwadd.wv        v4, v4, v6
    vwadd.wv        v8, v8, v10
    vwadd.wv        v12, v12, v14
    vwadd.wv        v16, v16, v18
    vwadd.wv        v20, v20, v22
#else
    vsetvli         zero, t3, e16, m2, ta, ma
    vle16.v         v26, (t0)
    vmerge.vvm      v28, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v4, v4, v28
    vsetvli         zero, t3, e16, m2, ta, ma
    vmv1r.v         v0, v2
    vmerge.vvm      v28, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v8, v8, v28
    vsetvli         zero, t3, e16, m2, ta, ma
    vmv1r.v         v0, v3
    vmerge.vvm      v28, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v12, v12, v28
    vsetvli         zero, t3, e16, m2, ta, ma
    vmv1r.v         v0, v24
    vmerge.vvm      v28, v30, v26, v0
    vmv1r.v         v0, v25
    vmerge.vvm      v2, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v16, v16, v28
    vwadd.wv        v20, v20, v2
#endif

    slli            t5, t3, 1
#if HIGH_BIT_DEPTH
    add             t1, t1, t5
#else
    add             t1, t1, t3
#endif
    add             t4, t4, t3
    sub             t2, t2, t3
    add             t0, t0, t5
    bgtz            t2, loop_saoCuStatsE1_x
    add             a0, a0, s6
    add             a1, a1, a2
    addi            a5, a5, -1
    bgtz            a5, loop_saoCuStatsE1_y

    lw              t0, (a7)
    lw              t1, 4(a7)
    lw              t2, 8(a7)
    lw              t3, 12(a7)
    lw              t4, 16(a7)
    add             t1, t1, s0
    add             t2, t2, s1
    add             t0, t0, s2
    add             t3, t3, s3
    add             t4, t4, s4
    sw              t0, (a7)
    sw              t1, 4(a7)
    sw              t2, 8(a7)
    sw              t3, 12(a7)
    sw              t4, 16(a7)

#if HIGH_BIT_DEPTH
    vsetvli         zero, a3, e32, m2, ta, ma
#else
    vsetvli         zero, a3, e32, m4, ta, ma
#endif
    vredsum.vs      v24, v4, v30
    vredsum.vs      v25, v8, v30
    vredsum.vs      v26, v12, v30
    vredsum.vs      v27, v16, v30
    vredsum.vs      v28, v20, v30
    vsetvli         zero, a3, e32, m1, ta, ma
    vmv.x.s         s0, v24
    vmv.x.s         s1, v25
    vmv.x.s         s2, v26
    vmv.x.s         s3, v27
    vmv.x.s         s4, v28
    lw              t0, (a6)
    lw              t1, 4(a6)
    lw              t2, 8(a6)
    lw              t3, 12(a6)
    lw              t4, 16(a6)
    add             t1, t1, s0
    add             t2, t2, s1
    add             t0, t0, s2
    add             t3, t3, s3
    add             t4, t4, s4
    sw              t0, (a6)
    sw              t1, 4(a6)
    sw              t2, 8(a6)
    sw              t3, 12(a6)
    sw              t4, 16(a6)

    ld              s0, (sp)
    ld              s1, 8(sp)
    ld              s2, 16(sp)
    ld              s3, 24(sp)
    ld              s4, 32(sp)
    ld              s5, 40(sp)
    ld              s6, 48(sp)
    ld              s7, 56(sp)
    addi            sp, sp, 64
    ret
endfunc

//void saoCuStatsE2_c(const int16_t *diff, const pixel *rec, intptr_t stride, int8_t *upBuff1, int8_t *upBufft, int endX, int endY, int32_t *stats, int32_t *count)
function PFX(saoCuStatsE2_rvv)
    ld              t6, (sp)
    addi            sp, sp, -80
    sd              s0, (sp)
    sd              s1, 8(sp)
    sd              s2, 16(sp)
    sd              s3, 24(sp)
    sd              s4, 32(sp)
    sd              s5, 40(sp)
    sd              s6, 48(sp)
    sd              s7, 56(sp)
    sd              s8, 64(sp)
    sd              s9, 72(sp)
    li              s0, 0
    li              s1, 0
    li              s2, 0
    li              s3, 0
    li              s4, 0
    li              s6, 1 << 7
#if HIGH_BIT_DEPTH
    slli            a2, a2, 1
    addi            s5, a2, 2
    vsetvli         t1, zero, e32, m2, ta, ma
#else
    addi            s5, a2, 1
    vsetvli         t1, zero, e32, m4, ta, ma
#endif
    vmv.v.i         v4, 0
    vmv.v.i         v8, 0
    vmv.v.i         v12, 0
    vmv.v.i         v16, 0
    vmv.v.i         v20, 0
#if HIGH_BIT_DEPTH
    vsetvli         t1, zero, e16, m1, ta, ma
#else
    vsetvli         t1, zero, e16, m2, ta, ma
#endif
    vmv.v.i         v30, 0

loop_saoCuStatsE2_y:
    mv              t0, a0
    mv              t1, a1
    mv              t2, a5
    mv              t4, a3
    mv              t5, a4
    add             s7, a1, a2
#if HIGH_BIT_DEPTH
    lhu             s8, (s7)
    lhu             s9, -2(a1)
#else
    lbu             s8, (s7)
    lbu             s9, -1(a1)
#endif
    sub             s8, s8, s9
    SIGNOF          s8, s7, s9
    sb              s8, (a4)
loop_saoCuStatsE2_x:
    add             s7, t1, s5
    addi            s8, t5, 1
#if HIGH_BIT_DEPTH
    vsetvli         t3, t2, e16, m1, ta, ma
    vle16.v         v2, (t1)
    vle16.v         v3, (s7)
    vle8.v          v1, (t4)
    vsext.vf2       v26, v1
    SIGNOF_RVV      v24, v2, v3
    vadd.vv         v25, v24, v26
    vrsub.vi        v24, v24, 0
    vsetvli         zero, t3, e8, m1, ta, ma
    vnsrl.wi        v26, v24, 0
    vse8.v          v26, (s8)
    vsetvli         zero, t3, e16, m1, ta, ma
#else
    vsetvli         t3, t2, e8, m1, ta, ma
    vle8.v          v2, (t1)
    vle8.v          v3, (s7)
    vle8.v          v1, (t4)
    SIGNOF_RVV      v24, v2, v3
    vadd.vv         v25, v24, v1
    vrsub.vi        v24, v24, 0
    vse8.v          v24, (s8)
#endif

    vmseq.vi        v0, v25, -2
    vmseq.vi        v2, v25, -1
    vmseq.vi        v3, v25, 0
    vmseq.vi        v24, v25, 1
    vmseq.vi        v25, v25, 2

    vcpop.m         s7, v0
    add             s0, s0, s7
    vcpop.m         s8, v2
    add             s1, s1, s8
    vcpop.m         s7, v3
    vcpop.m         s8, v24
    vcpop.m         s9, v25
    add             s2, s2, s7
    add             s3, s3, s8
    add             s4, s4, s9

#if HIGH_BIT_DEPTH
    vle16.v         v26, (t0)
    vmerge.vvm      v6, v30, v26, v0
    vmv1r.v         v0, v2
    vmerge.vvm      v10, v30, v26, v0
    vmv1r.v         v0, v3
    vmerge.vvm      v14, v30, v26, v0
    vmv1r.v         v0, v24
    vmerge.vvm      v18, v30, v26, v0
    vmv1r.v         v0, v25
    vmerge.vvm      v22, v30, v26, v0
    vsetvli         zero, t3, e16, m1, tu, ma
    vwadd.wv        v4, v4, v6
    vwadd.wv        v8, v8, v10
    vwadd.wv        v12, v12, v14
    vwadd.wv        v16, v16, v18
    vwadd.wv        v20, v20, v22
#else
    vsetvli         zero, t3, e16, m2, ta, ma
    vle16.v         v26, (t0)
    vmerge.vvm      v28, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v4, v4, v28
    vsetvli         zero, t3, e16, m2, ta, ma
    vmv1r.v         v0, v2
    vmerge.vvm      v28, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v8, v8, v28
    vsetvli         zero, t3, e16, m2, ta, ma
    vmv1r.v         v0, v3
    vmerge.vvm      v28, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v12, v12, v28
    vsetvli         zero, t3, e16, m2, ta, ma
    vmv1r.v         v0, v24
    vmerge.vvm      v28, v30, v26, v0
    vmv1r.v         v0, v25
    vmerge.vvm      v2, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v16, v16, v28
    vwadd.wv        v20, v20, v2
#endif

    slli            s7, t3, 1
#if HIGH_BIT_DEPTH
    add             t1, t1, s7
#else
    add             t1, t1, t3
#endif
    add             t4, t4, t3
    add             t5, t5, t3
    sub             t2, t2, t3
    add             t0, t0, s7
    bgtz            t2, loop_saoCuStatsE2_x
    mv              s8, a4
    add             a0, a0, s6
    add             a1, a1, a2
    mv              a4, a3
    addi            a6, a6, -1
    mv              a3, s8
    bgtz            a6, loop_saoCuStatsE2_y

    lw              t0, (t6)
    lw              t1, 4(t6)
    lw              t2, 8(t6)
    lw              t3, 12(t6)
    lw              t4, 16(t6)
    add             t1, t1, s0
    add             t2, t2, s1
    add             t0, t0, s2
    add             t3, t3, s3
    add             t4, t4, s4
    sw              t0, (t6)
    sw              t1, 4(t6)
    sw              t2, 8(t6)
    sw              t3, 12(t6)
    sw              t4, 16(t6)

#if HIGH_BIT_DEPTH
    vsetvli         zero, a3, e32, m2, ta, ma
#else
    vsetvli         zero, a3, e32, m4, ta, ma
#endif
    vredsum.vs      v24, v4, v30
    vredsum.vs      v25, v8, v30
    vredsum.vs      v26, v12, v30
    vredsum.vs      v27, v16, v30
    vredsum.vs      v28, v20, v30
    vsetvli         zero, a3, e32, m1, ta, ma
    vmv.x.s         s0, v24
    vmv.x.s         s1, v25
    vmv.x.s         s2, v26
    vmv.x.s         s3, v27
    vmv.x.s         s4, v28
    lw              t0, (a7)
    lw              t1, 4(a7)
    lw              t2, 8(a7)
    lw              t3, 12(a7)
    lw              t4, 16(a7)
    add             t1, t1, s0
    add             t2, t2, s1
    add             t0, t0, s2
    add             t3, t3, s3
    add             t4, t4, s4
    sw              t0, (a7)
    sw              t1, 4(a7)
    sw              t2, 8(a7)
    sw              t3, 12(a7)
    sw              t4, 16(a7)

    ld              s0, (sp)
    ld              s1, 8(sp)
    ld              s2, 16(sp)
    ld              s3, 24(sp)
    ld              s4, 32(sp)
    ld              s5, 40(sp)
    ld              s6, 48(sp)
    ld              s7, 56(sp)
    ld              s8, 64(sp)
    ld              s9, 72(sp)
    addi            sp, sp, 80
    ret
endfunc

//void saoCuStatsE3_c(const int16_t *diff, const pixel *rec, intptr_t stride, int8_t *upBuff1, int endX, int endY, int32_t *stats, int32_t *count)
function PFX(saoCuStatsE3_rvv)
    addi            sp, sp, -64
    sd              s0, (sp)
    sd              s1, 8(sp)
    sd              s2, 16(sp)
    sd              s3, 24(sp)
    sd              s4, 32(sp)
    sd              s5, 40(sp)
    sd              s6, 48(sp)
    sd              s7, 56(sp)
    li              s0, 0
    li              s1, 0
    li              s2, 0
    li              s3, 0
    li              s4, 0
    li              s6, 1 << 7
#if HIGH_BIT_DEPTH
    slli            a2, a2, 1
    addi            s5, a2, -2
    vsetvli         t1, zero, e32, m2, ta, ma
#else
    addi            s5, a2, -1
    vsetvli         t1, zero, e32, m4, ta, ma
#endif
    vmv.v.i         v4, 0
    vmv.v.i         v8, 0
    vmv.v.i         v12, 0
    vmv.v.i         v16, 0
    vmv.v.i         v20, 0
#if HIGH_BIT_DEPTH
    vsetvli         t1, zero, e16, m1, ta, ma
#else
    vsetvli         t1, zero, e16, m2, ta, ma
#endif
    vmv.v.i         v30, 0

loop_saoCuStatsE3_y:
    mv              t0, a0
    mv              t1, a1
    mv              t2, a4
    mv              t4, a3
loop_saoCuStatsE3_x:
    add             t5, t1, s5
#if HIGH_BIT_DEPTH
    vsetvli         t3, t2, e16, m1, ta, ma
    vle16.v         v2, (t1)
    vle16.v         v3, (t5)
    vle8.v          v1, (t4)
    vsext.vf2       v26, v1
    SIGNOF_RVV      v24, v2, v3
    vadd.vv         v25, v24, v26
    vsetvli         zero, t3, e8, m1, ta, ma
    vnsrl.wi        v26, v24, 0
    vrsub.vi        v1, v26, 0
    addi            t6, t4, -1
    vse8.v          v1, (t6)
    vsetvli         zero, t3, e16, m1, ta, ma
#else
    vsetvli         t3, t2, e8, m1, ta, ma
    vle8.v          v2, (t1)
    vle8.v          v3, (t5)
    vle8.v          v1, (t4)
    SIGNOF_RVV      v24, v2, v3
    vadd.vv         v25, v24, v1
    vrsub.vi        v1, v24, 0
    addi            t6, t4, -1
    vse8.v          v1, (t6)
#endif

    vmseq.vi        v0, v25, -2
    vmseq.vi        v2, v25, -1
    vmseq.vi        v3, v25, 0
    vmseq.vi        v24, v25, 1
    vmseq.vi        v25, v25, 2

    vcpop.m         s7, v0
    add             s0, s0, s7
    vcpop.m         t5, v2
    add             s1, s1, t5
    vcpop.m         t6, v3
    vcpop.m         s7, v24
    vcpop.m         t5, v25
    add             s2, s2, t6
    add             s3, s3, s7
    add             s4, s4, t5

#if HIGH_BIT_DEPTH
    vle16.v         v26, (t0)
    vmerge.vvm      v6, v30, v26, v0
    vmv1r.v         v0, v2
    vmerge.vvm      v10, v30, v26, v0
    vmv1r.v         v0, v3
    vmerge.vvm      v14, v30, v26, v0
    vmv1r.v         v0, v24
    vmerge.vvm      v18, v30, v26, v0
    vmv1r.v         v0, v25
    vmerge.vvm      v22, v30, v26, v0
    vsetvli         zero, t3, e16, m1, tu, ma
    vwadd.wv        v4, v4, v6
    vwadd.wv        v8, v8, v10
    vwadd.wv        v12, v12, v14
    vwadd.wv        v16, v16, v18
    vwadd.wv        v20, v20, v22
#else
    vsetvli         zero, t3, e16, m2, ta, ma
    vle16.v         v26, (t0)
    vmerge.vvm      v28, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v4, v4, v28
    vsetvli         zero, t3, e16, m2, ta, ma
    vmv1r.v         v0, v2
    vmerge.vvm      v28, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v8, v8, v28
    vsetvli         zero, t3, e16, m2, ta, ma
    vmv1r.v         v0, v3
    vmerge.vvm      v28, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v12, v12, v28
    vsetvli         zero, t3, e16, m2, ta, ma
    vmv1r.v         v0, v24
    vmerge.vvm      v28, v30, v26, v0
    vmv1r.v         v0, v25
    vmerge.vvm      v2, v30, v26, v0
    vsetvli         zero, t3, e16, m2, tu, ma
    vwadd.wv        v16, v16, v28
    vwadd.wv        v20, v20, v2
#endif

    slli            t5, t3, 1
#if HIGH_BIT_DEPTH
    add             t1, t1, t5
#else
    add             t1, t1, t3
#endif
    add             t4, t4, t3
    sub             t2, t2, t3
    add             t0, t0, t5
    bgtz            t2, loop_saoCuStatsE3_x
#if HIGH_BIT_DEPTH
    slli            t5, a4, 1
    add             t5, t5, a1
    lhu             t6, (t5)
    add             t5, t5, a2
    lhu             s7, -2(t5)
#else
    add             t5, a1, a4
    lbu             t6, (t5)
    add             t5, t5, a2
    lbu             s7, -1(t5)
#endif
    sub             t5, s7, t6
    SIGNOF          t5, s7, t6
    add             t6, a3, a4
    sb              t5, -1(t6)
    add             a0, a0, s6
    add             a1, a1, a2
    addi            a5, a5, -1
    bgtz            a5, loop_saoCuStatsE3_y

    lw              t0, (a7)
    lw              t1, 4(a7)
    lw              t2, 8(a7)
    lw              t3, 12(a7)
    lw              t4, 16(a7)
    add             t1, t1, s0
    add             t2, t2, s1
    add             t0, t0, s2
    add             t3, t3, s3
    add             t4, t4, s4
    sw              t0, (a7)
    sw              t1, 4(a7)
    sw              t2, 8(a7)
    sw              t3, 12(a7)
    sw              t4, 16(a7)

#if HIGH_BIT_DEPTH
    vsetvli         zero, a3, e32, m2, ta, ma
#else
    vsetvli         zero, a3, e32, m4, ta, ma
#endif
    vredsum.vs      v24, v4, v30
    vredsum.vs      v25, v8, v30
    vredsum.vs      v26, v12, v30
    vredsum.vs      v27, v16, v30
    vredsum.vs      v28, v20, v30
    vsetvli         zero, a3, e32, m1, ta, ma
    vmv.x.s         s0, v24
    vmv.x.s         s1, v25
    vmv.x.s         s2, v26
    vmv.x.s         s3, v27
    vmv.x.s         s4, v28
    lw              t0, (a6)
    lw              t1, 4(a6)
    lw              t2, 8(a6)
    lw              t3, 12(a6)
    lw              t4, 16(a6)
    add             t1, t1, s0
    add             t2, t2, s1
    add             t0, t0, s2
    add             t3, t3, s3
    add             t4, t4, s4
    sw              t0, (a6)
    sw              t1, 4(a6)
    sw              t2, 8(a6)
    sw              t3, 12(a6)
    sw              t4, 16(a6)

    ld              s0, (sp)
    ld              s1, 8(sp)
    ld              s2, 16(sp)
    ld              s3, 24(sp)
    ld              s4, 32(sp)
    ld              s5, 40(sp)
    ld              s6, 48(sp)
    ld              s7, 56(sp)
    addi            sp, sp, 64
    ret
endfunc
