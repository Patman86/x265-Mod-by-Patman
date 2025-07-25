/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Li Zhang <li.zhang2@arm.com>
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

#include "pixel-prim.h"
#include "mem-neon.h"
#include "neon-sve-bridge.h"

#include <arm_neon.h>

namespace
{
#if HIGH_BIT_DEPTH
template<int size>
uint64_t pixel_var_sve(const uint16_t *pix, intptr_t i_stride)
{
    if (size > 16)
    {
        uint64x2_t sum[2] = { vdupq_n_u64(0), vdupq_n_u64(0) };
        uint64x2_t sqr[2] = { vdupq_n_u64(0), vdupq_n_u64(0) };

        for (int h = 0; h < size; ++h)
        {
            for (int w = 0; w + 16 <= size; w += 16)
            {
                uint16x8_t s[2];
                load_u16x8xn<2>(pix + w, 8, s);

                sum[0] = x265_udotq_u16(sum[0], s[0], vdupq_n_u16(1));
                sum[1] = x265_udotq_u16(sum[1], s[1], vdupq_n_u16(1));

                sqr[0] = x265_udotq_u16(sqr[0], s[0], s[0]);
                sqr[1] = x265_udotq_u16(sqr[1], s[1], s[1]);
            }

            pix += i_stride;
        }

        sum[0] = vaddq_u64(sum[0], sum[1]);
        sqr[0] = vaddq_u64(sqr[0], sqr[1]);

        return vaddvq_u64(sum[0]) + (vaddvq_u64(sqr[0]) << 32);
    }
    if (size == 16)
    {
        uint16x8_t sum[2] = { vdupq_n_u16(0), vdupq_n_u16(0) };
        uint64x2_t sqr[2] = { vdupq_n_u64(0), vdupq_n_u64(0) };

        for (int h = 0; h < size; ++h)
        {
            uint16x8_t s[2];
            load_u16x8xn<2>(pix, 8, s);

            sum[0] = vaddq_u16(sum[0], s[0]);
            sum[1] = vaddq_u16(sum[1], s[1]);

            sqr[0] = x265_udotq_u16(sqr[0], s[0], s[0]);
            sqr[1] = x265_udotq_u16(sqr[1], s[1], s[1]);

            pix += i_stride;
        }

        uint32x4_t sum_u32 = vpaddlq_u16(sum[0]);
        sum_u32 = vpadalq_u16(sum_u32, sum[1]);
        sqr[0] = vaddq_u64(sqr[0], sqr[1]);

        return vaddvq_u32(sum_u32) + (vaddvq_u64(sqr[0]) << 32);
    }
    if (size == 8)
    {
        uint16x8_t sum = vdupq_n_u16(0);
        uint64x2_t sqr = vdupq_n_u64(0);

        for (int h = 0; h < size; ++h)
        {
            uint16x8_t s = vld1q_u16(pix);

            sum = vaddq_u16(sum, s);
            sqr = x265_udotq_u16(sqr, s, s);

            pix += i_stride;
        }

        return vaddlvq_u16(sum) + (vaddvq_u64(sqr) << 32);
    }
    if (size == 4) {
        uint16x4_t sum = vdup_n_u16(0);
        uint32x4_t sqr = vdupq_n_u32(0);

        for (int h = 0; h < size; ++h)
        {
            uint16x4_t s = vld1_u16(pix);

            sum = vadd_u16(sum, s);
            sqr = vmlal_u16(sqr, s, s);

            pix += i_stride;
        }

        return vaddv_u16(sum) + (vaddlvq_u32(sqr) << 32);
    }
}
#endif // HIGH_BIT_DEPTH
}

namespace X265_NS
{
#if HIGH_BIT_DEPTH
void setupPixelPrimitives_sve(EncoderPrimitives &p)
{
    p.cu[BLOCK_4x4].var   = pixel_var_sve<4>;
    p.cu[BLOCK_8x8].var   = pixel_var_sve<8>;
    p.cu[BLOCK_16x16].var = pixel_var_sve<16>;
    p.cu[BLOCK_32x32].var = pixel_var_sve<32>;
    p.cu[BLOCK_64x64].var = pixel_var_sve<64>;
}
#else // !HIGH_BIT_DEPTH
void setupPixelPrimitives_sve(EncoderPrimitives &)
{
}
#endif // HIGH_BIT_DEPTH
}
