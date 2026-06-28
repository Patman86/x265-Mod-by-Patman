/*****************************************************************************
* Copyright (C) 2013-2021 MulticoreWare, Inc
*
 * Authors: gunasrij <gunasri.jayakumar@multicorewareinc.com>
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

/* MCTF SIMD kernels - Provides AVX2 implementations of MCSTF primitives
 * declared in temporalfilter.h (MCTFPrimitives).
 *
 * Entry point: setupMCTFPrimitives_x86(MCTFPrimitives&, int cpuMask)
 * Called from primitives.cpp after setupMCTFPrimitives_scalar().
 * Overrides only the function pointers whose ISA requirement is met.
 */

#if ENABLE_ASSEMBLY && X265_ARCH_X86

#include <emmintrin.h>   /* SSE2   */
#include <tmmintrin.h>   /* SSSE3  */
#include <smmintrin.h>   /* SSE4.1 */
#include <immintrin.h>   /* AVX2   */
#include <cmath>

#include "common.h"
#include "primitives.h"
#include "temporalfilter.h"
#include "mv.h"
#include "cpu.h"


#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#  define MCSTF_TARGET_AVX2
#else
#  define MCSTF_TARGET_AVX2 __attribute__((target("avx2")))
#endif

namespace X265_NS {

    /* Shared helpers */

    static inline int hsum_epi32(__m128i v)
    {
        __m128i hi = _mm_srli_si128(v, 8);
        v = _mm_add_epi32(v, hi);
        hi = _mm_srli_si128(v, 4);
        v = _mm_add_epi32(v, hi);
        return _mm_cvtsi128_si32(v);
    }
	
	MCSTF_TARGET_AVX2
    static inline int hsum_epi32_avx(__m256i v)
    {
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);
        __m128i sum = _mm_add_epi32(lo, hi);

        __m128i hi2 = _mm_srli_si128(sum, 8);
        sum = _mm_add_epi32(sum, hi2);
        __m128i hi3 = _mm_srli_si128(sum, 4);
        sum = _mm_add_epi32(sum, hi3);

        return _mm_cvtsi128_si32(sum);
    }

    MCSTF_TARGET_AVX2
        static inline __m256i load8px_epi32(const pixel* p)
    {
#if X265_DEPTH > 8
        return _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p));
#else
        return _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)p));
#endif
    }

    MCSTF_TARGET_AVX2
        static inline __m256i load4px_epi32(const pixel* p)
    {
#if X265_DEPTH > 8
        return _mm256_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*)p));
#else
        return _mm256_cvtepu8_epi32(_mm_cvtsi32_si128(*(const int*)p));
#endif
    }

    MCSTF_TARGET_AVX2
        static inline void store8px(pixel* dst, __m256i v)
    {
#if X265_DEPTH > 8
        /* _mm256_packus_epi32(v,v) → [lane0-3, lane0-3 | lane4-7, lane4-7].
         * permute4x64(0x08) moves qwords 0 and 2 into the lower 128 bits.  */
        __m256i packed = _mm256_packus_epi32(v, v);
        __m128i out = _mm256_castsi256_si128(
            _mm256_permute4x64_epi64(packed, 0x08));
        _mm_storeu_si128((__m128i*)dst, out);
#else
        __m256i as16 = _mm256_packs_epi32(v, v);
        __m128i lo = _mm256_castsi256_si128(as16);
        __m128i hi = _mm256_extracti128_si256(as16, 1);
        __m128i merged = _mm_unpacklo_epi64(lo, hi);
        __m128i out8 = _mm_packus_epi16(merged, merged);
        _mm_storel_epi64((__m128i*)dst, out8);
#endif
    }

    MCSTF_TARGET_AVX2
        static inline void store4px(pixel* dst, __m256i v)
    {
        __m128i lo = _mm256_castsi256_si128(v);  /* only lower 4 lanes valid */
#if X265_DEPTH > 8
        __m128i out = _mm_packus_epi32(lo, _mm_setzero_si128());
        _mm_storel_epi64((__m128i*)dst, out);
#else
        __m128i as16 = _mm_packs_epi32(lo, _mm_setzero_si128());
        __m128i out8 = _mm_packus_epi16(as16, _mm_setzero_si128());
        *(int*)dst = _mm_cvtsi128_si32(out8);
#endif
    }

    MCSTF_TARGET_AVX2
        static int motionErrorLumaFrac_AVX2(
            const pixel* origOrigin, intptr_t origStride,
            const pixel* buffOrigin, intptr_t buffStride,
            int x, int y, int dx, int dy,
            int bs, int besterror, int bitDepth, int errorMode)
    {
        const int* xFilter = s_interpolationFilter[dx & 0xF];
        const int* yFilter = s_interpolationFilter[dy & 0xF];

        int tempArray[64 + 8][64];
        const int int_dx = dx >> 4;
        const int int_dy = dy >> 4;

        const __m128i xf12_128 = _mm_unpacklo_epi16(
            _mm_set1_epi16((int16_t)xFilter[1]),
            _mm_set1_epi16((int16_t)xFilter[2]));
        const __m128i xf34_128 = _mm_unpacklo_epi16(
            _mm_set1_epi16((int16_t)xFilter[3]),
            _mm_set1_epi16((int16_t)xFilter[4]));
        const __m128i xf56_128 = _mm_unpacklo_epi16(
            _mm_set1_epi16((int16_t)xFilter[5]),
            _mm_set1_epi16((int16_t)xFilter[6]));

        const __m256i xf12 = _mm256_set_m128i(xf12_128, xf12_128);
        const __m256i xf34 = _mm256_set_m128i(xf34_128, xf34_128);
        const __m256i xf56 = _mm256_set_m128i(xf56_128, xf56_128);

        for (int y1 = 1; y1 < bs + 7; y1++)
        {
            const pixel* rowStart0 = buffOrigin
                + (y + y1 + int_dy - 3) * buffStride
                + (x + int_dx - 3);

            for (int x1 = 0; x1 < bs; x1 += 8)
            {
                const pixel* rowStart = rowStart0 + x1;


#if X265_DEPTH > 8
                 /* uint16 pixels: load 8× uint16 directly into 128-bit reg.
                  * _mm256_set_m128i places lo_half in [127:0] and hi_half in
                  * [255:128].  We build the 256-bit interleaved pairs:
                  *   pairs_ab[255:0] = { unpackhi(s_a, s_b) | unpacklo(s_a, s_b) }
                  * so madd lane i gives  s_a[i]*f_a + s_b[i]*f_b.                 */
                __m128i s1 = _mm_loadu_si128((const __m128i*) & rowStart[1]);
                __m128i s2 = _mm_loadu_si128((const __m128i*) & rowStart[2]);
                __m128i s3 = _mm_loadu_si128((const __m128i*) & rowStart[3]);
                __m128i s4 = _mm_loadu_si128((const __m128i*) & rowStart[4]);
                __m128i s5 = _mm_loadu_si128((const __m128i*) & rowStart[5]);
                __m128i s6 = _mm_loadu_si128((const __m128i*) & rowStart[6]);
#else
                __m128i s1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) & rowStart[1]));
                __m128i s2 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) & rowStart[2]));
                __m128i s3 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) & rowStart[3]));
                __m128i s4 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) & rowStart[4]));
                __m128i s5 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) & rowStart[5]));
                __m128i s6 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) & rowStart[6]));
#endif

                __m256i pairs12 = _mm256_set_m128i(
                    _mm_unpackhi_epi16(s1, s2),
                    _mm_unpacklo_epi16(s1, s2));
                __m256i pairs34 = _mm256_set_m128i(
                    _mm_unpackhi_epi16(s3, s4),
                    _mm_unpacklo_epi16(s3, s4));
                __m256i pairs56 = _mm256_set_m128i(
                    _mm_unpackhi_epi16(s5, s6),
                    _mm_unpacklo_epi16(s5, s6));

                __m256i h_out = _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_madd_epi16(pairs12, xf12),
                        _mm256_madd_epi16(pairs34, xf34)),
                    _mm256_madd_epi16(pairs56, xf56));

                _mm256_storeu_si256((__m256i*) & tempArray[y1][x1], h_out);
            }
        }

        const __m256i yt1 = _mm256_set1_epi32(yFilter[1]);
        const __m256i yt2 = _mm256_set1_epi32(yFilter[2]);
        const __m256i yt3 = _mm256_set1_epi32(yFilter[3]);
        const __m256i yt4 = _mm256_set1_epi32(yFilter[4]);
        const __m256i yt5 = _mm256_set1_epi32(yFilter[5]);
        const __m256i yt6 = _mm256_set1_epi32(yFilter[6]);
        const __m256i vmax = _mm256_set1_epi32((1 << bitDepth) - 1);
        const __m256i vmin = _mm256_setzero_si256();
        const __m256i round_v = _mm256_set1_epi32(1 << 11);

        __m256i xerror = _mm256_setzero_si256();

        for (int outY = 0; outY < bs; outY++)
        {
            const pixel* origRowBase = origOrigin + (y + outY) * origStride + x;

            for (int x1 = 0; x1 < bs; x1 += 8)
            {
                __m256i t1 = _mm256_loadu_si256((const __m256i*) & tempArray[outY + 1][x1]);
                __m256i t2 = _mm256_loadu_si256((const __m256i*) & tempArray[outY + 2][x1]);
                __m256i t3 = _mm256_loadu_si256((const __m256i*) & tempArray[outY + 3][x1]);
                __m256i t4 = _mm256_loadu_si256((const __m256i*) & tempArray[outY + 4][x1]);
                __m256i t5 = _mm256_loadu_si256((const __m256i*) & tempArray[outY + 5][x1]);
                __m256i t6 = _mm256_loadu_si256((const __m256i*) & tempArray[outY + 6][x1]);

                __m256i v = _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_add_epi32(_mm256_mullo_epi32(t1, yt1),
                            _mm256_mullo_epi32(t2, yt2)),
                        _mm256_add_epi32(_mm256_mullo_epi32(t3, yt3),
                            _mm256_mullo_epi32(t4, yt4))),
                    _mm256_add_epi32(_mm256_mullo_epi32(t5, yt5),
                        _mm256_mullo_epi32(t6, yt6)));

                v = _mm256_srai_epi32(_mm256_add_epi32(v, round_v), 12);
                v = _mm256_min_epi32(_mm256_max_epi32(v, vmin), vmax);

#if X265_DEPTH > 8
                __m256i orig = _mm256_cvtepu16_epi32(
                    _mm_loadu_si128((const __m128i*) & origRowBase[x1]));
#else
                __m128i xorig = _mm_loadl_epi64((const __m128i*) & origRowBase[x1]);
                __m256i orig = _mm256_set_m128i(
                    _mm_cvtepu8_epi32(_mm_srli_si128(xorig, 4)),
                    _mm_cvtepu8_epi32(xorig));
#endif

                __m256i diff = _mm256_sub_epi32(v, orig);

                __m256i row_err;
                if (errorMode == 0)  /* SAD */
                    row_err = _mm256_abs_epi32(diff);
                else                 /* SSD */
                    row_err = _mm256_mullo_epi32(diff, diff);

                xerror = _mm256_add_epi32(xerror, row_err);
            }

            int error = hsum_epi32_avx(xerror);
            if (error > besterror)
                return error;
        }

        return hsum_epi32_avx(xerror);
    }

    /* Per-block separable 6-tap filter worker */
    MCSTF_TARGET_AVX2
        static void applyMotionBlock_AVX2(
            const pixel* pSrcImage, int srcStride,
            pixel* pDstImage, int dstStride,
            int x, int y,
            int blockSizeX, int blockSizeY,
            int xInt, int yInt,
            const int* xFilter,
            const int* yFilter)
    {
        int tempArray[15][8];

        const __m256i xf1 = _mm256_set1_epi32(xFilter[1]);
        const __m256i xf2 = _mm256_set1_epi32(xFilter[2]);
        const __m256i xf3 = _mm256_set1_epi32(xFilter[3]);
        const __m256i xf4 = _mm256_set1_epi32(xFilter[4]);
        const __m256i xf5 = _mm256_set1_epi32(xFilter[5]);
        const __m256i xf6 = _mm256_set1_epi32(xFilter[6]);

        const __m256i yf1 = _mm256_set1_epi32(yFilter[1]);
        const __m256i yf2 = _mm256_set1_epi32(yFilter[2]);
        const __m256i yf3 = _mm256_set1_epi32(yFilter[3]);
        const __m256i yf4 = _mm256_set1_epi32(yFilter[4]);
        const __m256i yf5 = _mm256_set1_epi32(yFilter[5]);
        const __m256i yf6 = _mm256_set1_epi32(yFilter[6]);

        const __m256i vround = _mm256_set1_epi32(1 << 11);
        const __m256i vmin = _mm256_setzero_si256();
        const __m256i vmax = _mm256_set1_epi32((1 << X265_DEPTH) - 1);

        const int hColBase = x + xInt - 3;
        const int hRowEnd = blockSizeY + 6;

        /* H-PASS */
        if (blockSizeX == 8)
        {
            for (int by = 1; by <= hRowEnd; by++)
            {
                const pixel* srcRow = pSrcImage + (y + by + yInt - 3) * srcStride;

                __m256i s1 = load8px_epi32(srcRow + hColBase + 1);
                __m256i s2 = load8px_epi32(srcRow + hColBase + 2);
                __m256i s3 = load8px_epi32(srcRow + hColBase + 3);
                __m256i s4 = load8px_epi32(srcRow + hColBase + 4);
                __m256i s5 = load8px_epi32(srcRow + hColBase + 5);
                __m256i s6 = load8px_epi32(srcRow + hColBase + 6);

                __m256i acc = _mm256_mullo_epi32(s1, xf1);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s2, xf2));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s3, xf3));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s4, xf4));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s5, xf5));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s6, xf6));

                _mm256_storeu_si256((__m256i*) & tempArray[by][0], acc);
            }
        }
        else
        {
            for (int by = 1; by <= hRowEnd; by++)
            {
                const pixel* srcRow = pSrcImage + (y + by + yInt - 3) * srcStride;

                __m256i s1 = load4px_epi32(srcRow + hColBase + 1);
                __m256i s2 = load4px_epi32(srcRow + hColBase + 2);
                __m256i s3 = load4px_epi32(srcRow + hColBase + 3);
                __m256i s4 = load4px_epi32(srcRow + hColBase + 4);
                __m256i s5 = load4px_epi32(srcRow + hColBase + 5);
                __m256i s6 = load4px_epi32(srcRow + hColBase + 6);

                __m256i acc = _mm256_mullo_epi32(s1, xf1);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s2, xf2));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s3, xf3));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s4, xf4));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s5, xf5));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s6, xf6));

                _mm_storeu_si128((__m128i*) & tempArray[by][0],
                    _mm256_castsi256_si128(acc));
            }
        }

        /* V-PASS */
        if (blockSizeX == 8)
        {
            for (int by = 0; by < blockSizeY; by++)
            {
                pixel* pDstPel = pDstImage + (y + by) * dstStride + x;

                __m256i t1 = _mm256_loadu_si256((const __m256i*) & tempArray[by + 1][0]);
                __m256i t2 = _mm256_loadu_si256((const __m256i*) & tempArray[by + 2][0]);
                __m256i t3 = _mm256_loadu_si256((const __m256i*) & tempArray[by + 3][0]);
                __m256i t4 = _mm256_loadu_si256((const __m256i*) & tempArray[by + 4][0]);
                __m256i t5 = _mm256_loadu_si256((const __m256i*) & tempArray[by + 5][0]);
                __m256i t6 = _mm256_loadu_si256((const __m256i*) & tempArray[by + 6][0]);

                __m256i acc = _mm256_mullo_epi32(t1, yf1);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(t2, yf2));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(t3, yf3));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(t4, yf4));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(t5, yf5));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(t6, yf6));

                acc = _mm256_add_epi32(acc, vround);
                acc = _mm256_srai_epi32(acc, 12);

                acc = _mm256_max_epi32(acc, vmin);
                acc = _mm256_min_epi32(acc, vmax);

                store8px(pDstPel, acc);
            }
        }
        else
        {
            for (int by = 0; by < blockSizeY; by++)
            {
                pixel* pDstPel = pDstImage + (y + by) * dstStride + x;

                __m256i t1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) & tempArray[by + 1][0]));
                __m256i t2 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) & tempArray[by + 2][0]));
                __m256i t3 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) & tempArray[by + 3][0]));
                __m256i t4 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) & tempArray[by + 4][0]));
                __m256i t5 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) & tempArray[by + 5][0]));
                __m256i t6 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) & tempArray[by + 6][0]));

                __m256i acc = _mm256_mullo_epi32(t1, yf1);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(t2, yf2));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(t3, yf3));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(t4, yf4));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(t5, yf5));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(t6, yf6));

                acc = _mm256_add_epi32(acc, vround);
                acc = _mm256_srai_epi32(acc, 12);
                acc = _mm256_max_epi32(acc, vmin);
                acc = _mm256_min_epi32(acc, vmax);

                store4px(pDstPel, acc);
            }
        }
    }

    MCSTF_TARGET_AVX2
        static void applyMotion_AVX2(
            const pixel* pSrcImage, int srcStride,
            pixel* pDstImage, int dstStride,
            int width, int height,
            int blockSizeX, int blockSizeY,
            uint32_t mvsStride, const MV* mvs,
            int csx, int csy,
            int blockRow, int rowSize, int vShift)
    {
        const int blkRowStart = (blockRow * rowSize) >> vShift;
        const int blkRowEnd = X265_MIN((blockRow * rowSize + rowSize) >> vShift, height);
        const int rowStart = (!rowSize) ? 0 : blkRowStart;
        const int rowEnd = (!rowSize) ? height : blkRowEnd;
        int       blockNumY = (!rowSize) ? 0 : blkRowStart / blockSizeY;

        for (int y = rowStart;
            y + blockSizeY <= rowEnd;
            y += blockSizeY, blockNumY++)
        {
            for (int x = 0, blockNumX = 0;
                x + blockSizeX <= width;
                x += blockSizeX, blockNumX++)
            {
                const int  mvIdx = blockNumY * (int)mvsStride + blockNumX;
                const MV& mv = mvs[mvIdx];

                const int dx = mv.x >> csx;
                const int dy = mv.y >> csy;
                const int xInt = mv.x >> (4 + csx);
                const int yInt = mv.y >> (4 + csy);

                const int* xFilter = s_interpolationFilter[dx & 0xf];
                const int* yFilter = s_interpolationFilter[dy & 0xf];

                applyMotionBlock_AVX2(
                    pSrcImage, srcStride,
                    pDstImage, dstStride,
                    x, y,
                    blockSizeX, blockSizeY,
                    xInt, yInt,
                    xFilter, yFilter);
            }
        }
    }

    /* Dispatch - overrides scalar defaults with SIMD variants at runtime */
    void setupMCTFPrimitives_x86(MCSTFPrimitives & p, int cpuMask)
    {
        if (cpuMask & X265_CPU_AVX2)
        {
            p.motionErrorLumaFrac = motionErrorLumaFrac_AVX2;
            p.applyMotion = applyMotion_AVX2;
        }
    }
} /* namespace X265_NS */

#endif /* ENABLE_ASSEMBLY && X265_ARCH_X86 */