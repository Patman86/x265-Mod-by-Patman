/*****************************************************************************
 * Copyright (C) 2013-2021 MulticoreWare, Inc
 *
 * Authors: Kirithika Kalirathnam <kirithika@multicorewareinc.com>

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

#include "common.h"
#include "mcstfharness.h"

using namespace X265_NS;

/* Fixed interior position/margins used by every check below: the buffers are
 * TEST_STRIDE x TEST_STRIDE with CENTER chosen so that the widest kernel
 * (bs = 16, +/- MV_PIXEL_RANGE integer MV, +/- 3 tap padding) never reads
 * outside the allocated buffers. */
static const int CENTER = 40;
static const int MV_PIXEL_RANGE = 4; /* max integer part of a test MV, in pixels */
static const int PAD = 16;           /* margin reserved for applyMotion's plane origin */

MCSTFHarness::MCSTFHarness()
{
    for (int i = 0; i < TEST_BUF_SIZE; i++)
    {
        m_origBuf[i] = (pixel)(rand() % (PIXEL_MAX + 1));
        m_refBuf[i]  = (pixel)(rand() % (PIXEL_MAX + 1));
    }
}

bool MCSTFHarness::check_motionErrorLumaFrac(const MCSTFPrimitives& ref, const MCSTFPrimitives& opt)
{
    const int bsValues[] = { 8, 16 };
    const int bestErrorValues[] = { (8 * 8 * 1024 * 1024), 100 };

    for (size_t bi = 0; bi < sizeof(bsValues) / sizeof(int); bi++)
    {
        int bs = bsValues[bi];

        for (int i = 0; i < ITERS; i++)
        {
            int dx = (rand() % (2 * MV_PIXEL_RANGE * 16 + 1)) - MV_PIXEL_RANGE * 16;
            int dy = (rand() % (2 * MV_PIXEL_RANGE * 16 + 1)) - MV_PIXEL_RANGE * 16;
            int besterror = bestErrorValues[i % 2];

            int resRef = ref.motionErrorLumaFrac(
                m_origBuf, TEST_STRIDE, m_refBuf, TEST_STRIDE,
                CENTER, CENTER, dx, dy, bs, besterror, X265_DEPTH);
            int resOpt = opt.motionErrorLumaFrac(
                m_origBuf, TEST_STRIDE, m_refBuf, TEST_STRIDE,
                CENTER, CENTER, dx, dy, bs, besterror, X265_DEPTH);

            if (resRef != resOpt)
            {
                printf("motionErrorLumaFrac[bs=%d]: failed (dx=%d dy=%d besterror=%d ref=%d opt=%d)\n",
                    bs, dx, dy, besterror, resRef, resOpt);
                return false;
            }
        }
    }
    return true;
}

bool MCSTFHarness::check_applyMotion(const MCSTFPrimitives& ref, const MCSTFPrimitives& opt)
{
    struct Cfg { int blockSizeX, blockSizeY, csx, csy; };
    const Cfg cfgs[] = { { 8, 8, 0, 0 }, { 4, 4, 1, 1 } };

    const int width = 64, height = 64;
    pixel* planeOrigin = m_origBuf + PAD * TEST_STRIDE + PAD;

    for (size_t ci = 0; ci < sizeof(cfgs) / sizeof(Cfg); ci++)
    {
        const Cfg& cfg = cfgs[ci];
        const int mvStride = width / cfg.blockSizeX;
        const int mvRows   = height / cfg.blockSizeY;

        MV mvs[256]; /* worst case: 64/4 x 64/4 chroma grid */
        for (int i = 0; i < mvStride * mvRows; i++)
            mvs[i] = MV((rand() % (2 * MV_PIXEL_RANGE * 16 + 1)) - MV_PIXEL_RANGE * 16,
                        (rand() % (2 * MV_PIXEL_RANGE * 16 + 1)) - MV_PIXEL_RANGE * 16);

        /* whole-frame call (rowSize = 0) and a row-batched call, matching the
         * two ways TemporalFilter::applyMotion is actually driven */
        const struct { int blockRow, rowSize, vShift; } rowCfgs[] =
        {
            { 0, 0, 0 },
            { 1, 16, cfg.csy },
        };

        for (size_t ri = 0; ri < sizeof(rowCfgs) / sizeof(rowCfgs[0]); ri++)
        {
            memset(m_dstBufRef, 0, sizeof(m_dstBufRef));
            memset(m_dstBufOpt, 0, sizeof(m_dstBufOpt));
            pixel* dstOriginRef = m_dstBufRef + PAD * TEST_STRIDE + PAD;
            pixel* dstOriginOpt = m_dstBufOpt + PAD * TEST_STRIDE + PAD;

            ref.applyMotion(planeOrigin, TEST_STRIDE, dstOriginRef, TEST_STRIDE,
                width, height, cfg.blockSizeX, cfg.blockSizeY, mvStride, mvs,
                cfg.csx, cfg.csy, rowCfgs[ri].blockRow, rowCfgs[ri].rowSize, rowCfgs[ri].vShift);
            opt.applyMotion(planeOrigin, TEST_STRIDE, dstOriginOpt, TEST_STRIDE,
                width, height, cfg.blockSizeX, cfg.blockSizeY, mvStride, mvs,
                cfg.csx, cfg.csy, rowCfgs[ri].blockRow, rowCfgs[ri].rowSize, rowCfgs[ri].vShift);

            if (memcmp(m_dstBufRef, m_dstBufOpt, sizeof(m_dstBufRef)))
            {
                printf("applyMotion[bx=%d by=%d rowSize=%d]: failed\n",
                    cfg.blockSizeX, cfg.blockSizeY, rowCfgs[ri].rowSize);
                return false;
            }
        }
    }
    return true;
}

bool MCSTFHarness::check_computeBlockStats(const MCSTFPrimitives& ref, const MCSTFPrimitives& opt)
{
    const int blkSizes[] = { 4, 8 };

    for (size_t bi = 0; bi < sizeof(blkSizes) / sizeof(int); bi++)
    {
        int blkSize = blkSizes[bi];

        for (int i = 0; i < ITERS; i++)
        {
            int varRef, diffRef, varOpt, diffOpt;
            ref.computeBlockStats(m_origBuf + CENTER * TEST_STRIDE + CENTER, TEST_STRIDE,
                m_refBuf + CENTER * TEST_STRIDE + CENTER, TEST_STRIDE, blkSize, &varRef, &diffRef);
            opt.computeBlockStats(m_origBuf + CENTER * TEST_STRIDE + CENTER, TEST_STRIDE,
                m_refBuf + CENTER * TEST_STRIDE + CENTER, TEST_STRIDE, blkSize, &varOpt, &diffOpt);

            if (varRef != varOpt || diffRef != diffOpt)
            {
                printf("computeBlockStats[bs=%d]: failed (variance %d vs %d, diffsum %d vs %d)\n",
                    blkSize, varRef, varOpt, diffRef, diffOpt);
                return false;
            }
        }
    }
    return true;
}

bool MCSTFHarness::check_bilateralFilter(const MCSTFPrimitives& ref, const MCSTFPrimitives& opt)
{
    const int blkSizes[] = { 4, 8 };
    const int numRefsValues[] = { 1, 2, 4 };
    const double bdw = 1024.0 / (PIXEL_MAX + 1);
    const double maxSample = PIXEL_MAX;

    for (size_t bi = 0; bi < sizeof(blkSizes) / sizeof(int); bi++)
    {
        int blkSize = blkSizes[bi];

        for (size_t ni = 0; ni < sizeof(numRefsValues) / sizeof(int); ni++)
        {
            int numRefs = numRefsValues[ni];

            const pixel* refBlks[4];
            intptr_t refStrides[4];
            double vww[4], vsw[4];
            for (int i = 0; i < numRefs; i++)
            {
                refBlks[i] = m_refBuf + CENTER * TEST_STRIDE + CENTER + i * 4;
                refStrides[i] = TEST_STRIDE;
                vww[i] = 0.1 + (rand() % 100) / 100.0;
                vsw[i] = 50.0 + (rand() % 500);
            }

            const pixel* srcBlk = m_origBuf + CENTER * TEST_STRIDE + CENTER;

            memset(m_dstBufRef, 0, sizeof(m_dstBufRef));
            memset(m_dstBufOpt, 0, sizeof(m_dstBufOpt));

            ref.bilateralFilter(srcBlk, TEST_STRIDE, numRefs, refBlks, refStrides,
                vww, vsw, bdw, maxSample, blkSize, m_dstBufRef, TEST_STRIDE);
            opt.bilateralFilter(srcBlk, TEST_STRIDE, numRefs, refBlks, refStrides,
                vww, vsw, bdw, maxSample, blkSize, m_dstBufOpt, TEST_STRIDE);

            for (int y = 0; y < blkSize; y++)
            {
                if (memcmp(m_dstBufRef + y * TEST_STRIDE, m_dstBufOpt + y * TEST_STRIDE, blkSize * sizeof(pixel)))
                {
                    printf("bilateralFilter[bs=%d numRefs=%d]: failed\n", blkSize, numRefs);
                    return false;
                }
            }
        }
    }
    return true;
}

bool MCSTFHarness::testCorrectness(const MCSTFPrimitives& ref, const MCSTFPrimitives& opt)
{
    if (opt.motionErrorLumaFrac && !check_motionErrorLumaFrac(ref, opt))
        return false;
    if (opt.applyMotion && !check_applyMotion(ref, opt))
        return false;
    if (opt.computeBlockStats && !check_computeBlockStats(ref, opt))
        return false;
    if (opt.bilateralFilter && !check_bilateralFilter(ref, opt))
        return false;
    return true;
}

void MCSTFHarness::measureSpeed(const MCSTFPrimitives& ref, const MCSTFPrimitives& opt)
{
    if (opt.motionErrorLumaFrac)
    {
        printf("motionErrorLumaFrac[bs=16]\t");
        // cppcheck-suppress unsignedPositive
        REPORT_SPEEDUP(opt.motionErrorLumaFrac, ref.motionErrorLumaFrac,
            m_origBuf, TEST_STRIDE, m_refBuf, TEST_STRIDE, CENTER, CENTER, 20, 5, 16, INT_MAX, X265_DEPTH);
    }

    if (opt.applyMotion)
    {
        static MV mvs[64];
        pixel* planeOrigin = m_origBuf + PAD * TEST_STRIDE + PAD;
        printf("applyMotion[8x8]\t");
        // cppcheck-suppress unsignedPositive
        REPORT_SPEEDUP(opt.applyMotion, ref.applyMotion,
            planeOrigin, TEST_STRIDE, m_dstBufOpt, TEST_STRIDE, 64, 64, 8, 8, 8, mvs, 0, 0, 0, 0, 0);
    }

    if (opt.computeBlockStats)
    {
        int variance, diffsum;
        printf("computeBlockStats[bs=8]\t");
        // cppcheck-suppress unsignedPositive
        REPORT_SPEEDUP(opt.computeBlockStats, ref.computeBlockStats,
            m_origBuf + CENTER * TEST_STRIDE + CENTER, TEST_STRIDE,
            m_refBuf + CENTER * TEST_STRIDE + CENTER, TEST_STRIDE, 8, &variance, &diffsum);
    }

    if (opt.bilateralFilter)
    {
        const pixel* refBlks[4];
        intptr_t refStrides[4];
        double vww[4], vsw[4];
        for (int i = 0; i < 4; i++)
        {
            refBlks[i] = m_refBuf + CENTER * TEST_STRIDE + CENTER + i * 4;
            refStrides[i] = TEST_STRIDE;
            vww[i] = 1.0;
            vsw[i] = 200.0;
        }
        const pixel* srcBlk = m_origBuf + CENTER * TEST_STRIDE + CENTER;
        const double bdw = 1024.0 / (PIXEL_MAX + 1);

        printf("bilateralFilter[bs=8,refs=4]\t");
        // cppcheck-suppress unsignedPositive
        REPORT_SPEEDUP(opt.bilateralFilter, ref.bilateralFilter,
            srcBlk, TEST_STRIDE, 4, refBlks, refStrides, vww, vsw, bdw, (double)PIXEL_MAX, 8, m_dstBufOpt, TEST_STRIDE);
    }
}
