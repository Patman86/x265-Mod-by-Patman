/*****************************************************************************
 * Copyright (C) 2024 MulticoreWare, Inc
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

#include <riscv_vector.h>
#include <stdint.h>
#include "sao.h"
#include "primitives.h"

static inline void compute_eo_stats(const vint8m1_t edge_type, const int16_t *diff,
    int32_t *stats, int32_t *count, int vl)
{
    (void)__riscv_vsetvl_e16m2(vl);
    vint16m2_t tmp_0 = __riscv_vmv_v_x_i16m2(0, vl);
    (void)__riscv_vsetvl_e32m1(vl);
    vint32m1_t tmp_01 = __riscv_vmv_v_x_i32m1(0, vl);

    // Create a mask for each edge type.
    (void)__riscv_vsetvl_e8m1(vl);
    vbool8_t mask0 = __riscv_vmseq_vx_i8m1_b8(edge_type, -2, vl);
    vbool8_t mask1 = __riscv_vmseq_vx_i8m1_b8(edge_type, -1, vl);
    vbool8_t mask2 = __riscv_vmseq_vx_i8m1_b8(edge_type, 0, vl);
    vbool8_t mask3 = __riscv_vmseq_vx_i8m1_b8(edge_type, 1, vl);
    vbool8_t mask4 = __riscv_vmseq_vx_i8m1_b8(edge_type, 2, vl);

    count[1] += __riscv_vcpop_m_b8(mask0, vl);
    count[2] += __riscv_vcpop_m_b8(mask1, vl);
    count[0] += __riscv_vcpop_m_b8(mask2, vl);
    count[3] += __riscv_vcpop_m_b8(mask3, vl);
    count[4] += __riscv_vcpop_m_b8(mask4, vl);

    // Widen the masks to 16-bit.
    (void)__riscv_vsetvl_e16m2(vl);
    vint16m2_t load_diff = __riscv_vle16_v_i16m2(diff, vl);
    vint16m2_t temp_add0 = __riscv_vmerge_vvm_i16m2(tmp_0, load_diff, mask0, vl);
    vint16m2_t temp_add1 = __riscv_vmerge_vvm_i16m2(tmp_0, load_diff, mask1, vl);
    vint16m2_t temp_add2 = __riscv_vmerge_vvm_i16m2(tmp_0, load_diff, mask2, vl);
    vint16m2_t temp_add3 = __riscv_vmerge_vvm_i16m2(tmp_0, load_diff, mask3, vl);
    vint16m2_t temp_add4 = __riscv_vmerge_vvm_i16m2(tmp_0, load_diff, mask4, vl);

    vint32m1_t temp_stats0 = __riscv_vwredsum_vs_i16m2_i32m1(temp_add0, tmp_01, vl);
    vint32m1_t temp_stats1 = __riscv_vwredsum_vs_i16m2_i32m1(temp_add1, tmp_01, vl);
    vint32m1_t temp_stats2 = __riscv_vwredsum_vs_i16m2_i32m1(temp_add2, tmp_01, vl);
    vint32m1_t temp_stats3 = __riscv_vwredsum_vs_i16m2_i32m1(temp_add3, tmp_01, vl);
    vint32m1_t temp_stats4 = __riscv_vwredsum_vs_i16m2_i32m1(temp_add4, tmp_01, vl);

    stats[1] += __riscv_vmv_x_s_i32m1_i32(temp_stats0);
    stats[2] += __riscv_vmv_x_s_i32m1_i32(temp_stats1);
    stats[0] += __riscv_vmv_x_s_i32m1_i32(temp_stats2);
    stats[3] += __riscv_vmv_x_s_i32m1_i32(temp_stats3);
    stats[4] += __riscv_vmv_x_s_i32m1_i32(temp_stats4);
}

static inline vint8m1_t signOf_rvv(const pixel *a, const pixel *b, int vl)
{
#if HIGH_BIT_DEPTH
    vl = __riscv_vsetvl_e16m2(vl);
    vuint16m2_t s0 = __riscv_vle16_v_u16m2(a, vl);
    vuint16m2_t s1 = __riscv_vle16_v_u16m2(b, vl);

    vbool8_t bgt = __riscv_vmsgtu_vv_u16m2_b8(s0, s1, vl);
    vbool8_t blt = __riscv_vmsltu_vv_u16m2_b8(s0, s1, vl);

    // a > b : 1   a == b : 0  a < b : -1
    vl = __riscv_vsetvl_e8m1(vl);
    vint8m1_t ret = __riscv_vmv_v_x_i8m1(0, vl);
    ret = __riscv_vmerge_vxm_i8m1(ret, 1, bgt, vl);
    ret = __riscv_vmerge_vxm_i8m1(ret, -1, blt, vl);
#else // HIGH_BIT_DEPTH
    vl = __riscv_vsetvl_e8m1(vl);
    vint8m1_t ret = __riscv_vmv_v_x_i8m1(0, vl);

    vuint8m1_t s0 = __riscv_vle8_v_u8m1(a, vl);
    vuint8m1_t s1 = __riscv_vle8_v_u8m1(b, vl);

    // a > b : 1   a == b : 0  a < b : -1
    vbool8_t bgt = __riscv_vmsgtu_vv_u8m1_b8(s0, s1, vl);
    ret = __riscv_vmerge_vxm_i8m1(ret, 1, bgt, vl);
    vbool8_t blt = __riscv_vmsltu_vv_u8m1_b8(s0, s1, vl);
    ret = __riscv_vmerge_vxm_i8m1(ret, -1, blt, vl);
#endif // HIGH_BIT_DEPTH

    return ret;
}

namespace X265_NS {
void saoCuStatsBO_rvv(const int16_t *diff, const pixel *rec, intptr_t stride, int endX, int endY, int32_t *stats, int32_t *count)
{
#if HIGH_BIT_DEPTH
    const int n_elem = 4;
    const int elem_width = 16;
#else
    const int n_elem = 8;
    const int elem_width = 8;
#endif

    // Additional temporary buffer for accumulation.
    int32_t stats_tmp[32] = { 0 };
    int32_t count_tmp[32] = { 0 };

    // Byte-addressable pointers to buffers, to optimise address calculation.
    uint8_t *stats_b[2] = {
        reinterpret_cast<uint8_t *>(stats),
        reinterpret_cast<uint8_t *>(stats_tmp),
    };
    uint8_t *count_b[2] = {
        reinterpret_cast<uint8_t *>(count),
        reinterpret_cast<uint8_t *>(count_tmp),
    };

    // Combine shift for index calculation with shift for address calculation.
    const int right_shift = X265_DEPTH - X265_NS::SAO::SAO_BO_BITS;
    const int left_shift = 2;
    const int shift = right_shift - left_shift;
    // Mask out bits 7, 1 & 0 to account for combination of shifts.
    const int mask = 0x7c;

    // Compute statistics into temporary buffers.
    for (int y = 0; y < endY; y++)
    {
        int x = 0;
        for (; x + n_elem < endX; x += n_elem)
        {
            uint64_t class_idx_64 =
                *reinterpret_cast<const uint64_t *>(rec + x) >> shift;

            for (int i = 0; i < n_elem; ++i)
            {
                const int idx = i & 1;
                const int off  = (class_idx_64 >> (i * elem_width)) & mask;
                *reinterpret_cast<uint32_t*>(stats_b[idx] + off) += diff[x + i];
                *reinterpret_cast<uint32_t*>(count_b[idx] + off) += 1;
            }
        }

        if (x < endX)
        {
            uint64_t class_idx_64 =
                *reinterpret_cast<const uint64_t *>(rec + x) >> shift;

            for (int i = 0; (i + x) < endX; ++i)
            {
                const int idx = i & 1;
                const int off  = (class_idx_64 >> (i * elem_width)) & mask;
                *reinterpret_cast<uint32_t*>(stats_b[idx] + off) += diff[x + i];
                *reinterpret_cast<uint32_t*>(count_b[idx] + off) += 1;
            }
        }

        diff += MAX_CU_SIZE;
        rec += stride;
    }

    // Reduce temporary buffers to destination using Neon.
    for (int i = 0; i < 32;)
    {
        int vl = __riscv_vsetvl_e32m2(32 - i);
        vint32m2_t s0 = __riscv_vle32_v_i32m2(stats_tmp + i, vl);
        vint32m2_t s1 = __riscv_vle32_v_i32m2(stats + i, vl);
        vint32m2_t ss = __riscv_vadd_vv_i32m2(s0, s1, vl);
        __riscv_vse32_v_i32m2(stats + i, ss, vl);

        vint32m2_t c0 = __riscv_vle32_v_i32m2(count_tmp + i, vl);
        vint32m2_t c1 = __riscv_vle32_v_i32m2(count + i, vl);
        vint32m2_t cs = __riscv_vadd_vv_i32m2(c0, c1, vl);
        __riscv_vse32_v_i32m2(count + i, cs, vl);

        i += vl;
    }
}

void saoCuStatsE0_rvv(const int16_t *diff, const pixel *rec, intptr_t stride, int endX, int endY, int32_t *stats, int32_t *count)
{
    for (int y = 0; y < endY; y++)
    {
        // Calculate negated sign_left(x) directly, to save negation when
        // reusing sign_right(x) as sign_left(x + 1).
        int vl = __riscv_vsetvl_e8m1(endX);
        vint8m1_t neg_sign_left = __riscv_vmv_v_x_i8m1(x265_signOf(rec[-1] - rec[0]), vl);
        for (int x = 0; x < endX; x += vl)
        {
            vl = __riscv_vsetvl_e8m1(endX - x);
            vint8m1_t sign_right = signOf_rvv(rec + x, rec + x + 1, vl);

            // neg_sign_left(x) = sign_right(x + 1), reusing one from previous
            // iteration.
            neg_sign_left = __riscv_vslideup_vx_i8m1(neg_sign_left, sign_right, 1, vl);

            // Subtract instead of add, as sign_left is negated.
            vint8m1_t edge_type = __riscv_vsub_vv_i8m1(sign_right, neg_sign_left, vl);

            // For reuse in the next iteration.
            neg_sign_left = __riscv_vslidedown_vx_i8m1(sign_right, vl - 1, vl);

            compute_eo_stats(edge_type, diff + x, stats, count, vl);
        }

        diff += MAX_CU_SIZE;
        rec += stride;
    }
}


void setupSaoPrimitives_rvv(EncoderPrimitives &p)
{
    p.saoCuStatsBO = saoCuStatsBO_rvv;
    p.saoCuStatsE0 = saoCuStatsE0_rvv;
}

}// namespace X265_NS
