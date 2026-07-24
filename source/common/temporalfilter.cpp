/*****************************************************************************
* Copyright (C) 2013-2021 MulticoreWare, Inc
*
 * Authors: Ashok Kumar Mishra <ashok@multicorewareinc.com>
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
#include "common.h"
#include "temporalfilter.h"
#include "primitives.h"
#include "threadpool.h"
#include "frame.h"
#include "slice.h"
#include "framedata.h"
#include "analysis.h"

using namespace X265_NS;

namespace X265_NS {

    /* MCSTF scalar fallbacks - used when the runtime CPU lacks the required ISA.*/

    static int motionErrorLumaFrac_c(
        const pixel* origOrigin, intptr_t origStride,
        const pixel* buffOrigin, intptr_t buffStride,
        int x, int y, int dx, int dy,
        int bs, int besterror, int bitDepth)
    {
        const int* xFilter = s_interpolationFilter[dx & 0xF];
        const int* yFilter = s_interpolationFilter[dy & 0xF];
        int tempArray[64 + 8][64];
        int error = 0;

        for (int y1 = 1; y1 < bs + 7; y1++)
        {
            const int yOffset = y + y1 + (dy >> 4) - 3;
            const pixel* sourceRow = buffOrigin + yOffset * buffStride;
            for (int x1 = 0; x1 < bs; x1++)
            {
                int iBase = x + x1 + (dx >> 4) - 3;
                const pixel* rowStart = sourceRow + iBase;
                int iSum = 0;
                iSum += xFilter[1] * rowStart[1];
                iSum += xFilter[2] * rowStart[2];
                iSum += xFilter[3] * rowStart[3];
                iSum += xFilter[4] * rowStart[4];
                iSum += xFilter[5] * rowStart[5];
                iSum += xFilter[6] * rowStart[6];
                tempArray[y1][x1] = iSum;
            }
        }

        const int maxSampleValue = (1 << bitDepth) - 1;
        for (int y1 = 0; y1 < bs; y1++)
        {
            const pixel* origRow = origOrigin + (y + y1) * origStride;
            for (int x1 = 0; x1 < bs; x1++)
            {
                int iSum = 0;
                iSum += yFilter[1] * tempArray[y1 + 1][x1];
                iSum += yFilter[2] * tempArray[y1 + 2][x1];
                iSum += yFilter[3] * tempArray[y1 + 3][x1];
                iSum += yFilter[4] * tempArray[y1 + 4][x1];
                iSum += yFilter[5] * tempArray[y1 + 5][x1];
                iSum += yFilter[6] * tempArray[y1 + 6][x1];
                iSum = (iSum + (1 << 11)) >> 12;
                iSum = iSum < 0 ? 0 : (iSum > maxSampleValue ? maxSampleValue : iSum);

                int diff = iSum - origRow[x + x1];
                error += diff * diff;
            }
            if (error > besterror)
                return error;
        }
        return error;
    }
    static void applyMotion_c(
        const pixel* pSrcImage, int srcStride,
        pixel* pDstImage, int dstStride,
        int width, int height,
        int blockSizeX, int blockSizeY,
        uint32_t mvsStride, const MV* mvs,
        int csx, int csy,
        int blockRow, int rowSize, int vShift)
    {
        static const int numFilterTaps = 7;
        static const int centreTapOffset = 3;
        const int maxValue = (1 << X265_DEPTH) - 1;

        const int blkRowStart = (!rowSize) ? 0 : (blockRow * rowSize) >> vShift;
        const int blkRowEnd   = (!rowSize) ? height : X265_MIN((blockRow * rowSize + rowSize) >> vShift, height);
        int       blockNumY   = (!rowSize) ? 0 : blkRowStart / blockSizeY;

        for (int y = blkRowStart;
            y + blockSizeY <= blkRowEnd;
            y += blockSizeY, blockNumY++)
        {
            for (int x = 0, blockNumX = 0;
                x + blockSizeX <= width;
                x += blockSizeX, blockNumX++)
            {
                const int mvIdx = blockNumY * (int)mvsStride + blockNumX;
                const MV& mv = mvs[mvIdx];

                const int dx = mv.x >> csx;
                const int dy = mv.y >> csy;
                const int xInt = mv.x >> (4 + csx);
                const int yInt = mv.y >> (4 + csy);

                const int* xFilter = s_interpolationFilter[dx & 0xf];
                const int* yFilter = s_interpolationFilter[dy & 0xf];

                int tempArray[8 + numFilterTaps][8];

                for (int by = 1; by < blockSizeY + numFilterTaps; by++)
                {
                    const int yOffset = y + by + yInt - centreTapOffset;
                    const pixel* sourceRow = pSrcImage + yOffset * srcStride;

                    for (int bx = 0; bx < blockSizeX; bx++)
                    {
                        int iBase = x + bx + xInt - centreTapOffset;
                        const pixel* rowStart = sourceRow + iBase;

                        int iSum = 0;
                        iSum += xFilter[1] * rowStart[1];
                        iSum += xFilter[2] * rowStart[2];
                        iSum += xFilter[3] * rowStart[3];
                        iSum += xFilter[4] * rowStart[4];
                        iSum += xFilter[5] * rowStart[5];
                        iSum += xFilter[6] * rowStart[6];

                        tempArray[by][bx] = iSum;
                    }
                }

                pixel* pDstRow = pDstImage + y * dstStride;

                for (int by = 0; by < blockSizeY; by++, pDstRow += dstStride)
                {
                    pixel* pDstPel = pDstRow + x;

                    for (int bx = 0; bx < blockSizeX; bx++, pDstPel++)
                    {
                        int iSum = 0;

                        iSum += yFilter[1] * tempArray[by + 1][bx];
                        iSum += yFilter[2] * tempArray[by + 2][bx];
                        iSum += yFilter[3] * tempArray[by + 3][bx];
                        iSum += yFilter[4] * tempArray[by + 4][bx];
                        iSum += yFilter[5] * tempArray[by + 5][bx];
                        iSum += yFilter[6] * tempArray[by + 6][bx];

                        iSum = (iSum + (1 << 11)) >> 12;
                        iSum = iSum < 0 ? 0 : (iSum > maxValue ? maxValue : iSum);

                        *pDstPel = (pixel)iSum;
                    }
                }
            }
        }
    }

    static void computeBlockStats_c(
        const pixel* srcPel, intptr_t srcStride,
        const pixel* refPel, intptr_t refStride,
        int blkSize, int* outVariance, int* outDiffsum)
    {
        int variance = 0, diffsum = 0;
        for (int y1 = 0; y1 < blkSize; y1++)
        {
            for (int x1 = 0; x1 < blkSize; x1++)
            {
                int diff = *(srcPel + srcStride * y1 + x1)
                    - *(refPel + refStride * y1 + x1);
                variance += diff * diff;
                if (x1 != blkSize - 1)
                {
                    int diffR = *(srcPel + srcStride * y1 + x1 + 1)
                        - *(refPel + refStride * y1 + x1 + 1);
                    diffsum += (diffR - diff) * (diffR - diff);
                }
                if (y1 != blkSize - 1)
                {
                    int diffD = *(srcPel + srcStride * (y1 + 1) + x1)
                        - *(refPel + refStride * (y1 + 1) + x1);
                    diffsum += (diffD - diff) * (diffD - diff);
                }
            }
        }
        *outVariance = variance;
        *outDiffsum = diffsum;
    }

    static void bilateralFilter_c(
        const pixel* srcBlk, intptr_t srcStride,
        int             numRefs,
        const pixel* const* refBlks,
        const intptr_t* refStrides,
        const double* vww,
        const double* vsw,
        double          bdw,
        double          maxSample,
        int             blkSize,
        pixel* dstBlk, intptr_t dstStride)
    {
        for (int y = 0; y < blkSize; y++)
        {
            for (int x = 0; x < blkSize; x++)
            {
                const int orgVal = (int)srcBlk[y * srcStride + x];
                double temporalWeightSum = 1.0;
                double newVal = (double)orgVal;
                for (int i = 0; i < numRefs; i++)
                {
                    const int refVal = (int)refBlks[i][y * refStrides[i] + x];
                    double diff = (double)(refVal - orgVal) * bdw;
                    double diffSq = diff * diff;
                    const double weight = vww[i] * exp(-diffSq / vsw[i]);
                    newVal += weight * refVal;
                    temporalWeightSum += weight;
                }
                newVal /= temporalWeightSum;
                double sampleVal = round(newVal);
                sampleVal = (sampleVal < 0 ? 0 : (sampleVal > maxSample ? maxSample : sampleVal));
                dstBlk[y * dstStride + x] = (pixel)sampleVal;
            }
        }
    }

    /* Global MCSTF primitives table */
    MCSTFPrimitives mcstfPrim;

    void setupMCSTFPrimitives_scalar(MCSTFPrimitives& p)
    {
        p.motionErrorLumaFrac = motionErrorLumaFrac_c;
        p.applyMotion = applyMotion_c;
        p.computeBlockStats = computeBlockStats_c;
        p.bilateralFilter = bilateralFilter_c;
    }
}/* namespace X265_NS */

void OrigPicBuffer::addPicture(Frame* inFrame)
{
    m_mcstfPicList.pushFrontMCSTF(*inFrame);
}

void OrigPicBuffer::addEncPicture(Frame* inFrame)
{
    m_mcstfOrigPicFreeList.pushFrontMCSTF(*inFrame);
}

void OrigPicBuffer::addEncPictureToPicList(Frame* inFrame)
{
    m_mcstfOrigPicList.pushFrontMCSTF(*inFrame);
}

OrigPicBuffer::~OrigPicBuffer()
{
    while (!m_mcstfOrigPicList.empty())
    {
        Frame* curFrame = m_mcstfOrigPicList.popBackMCSTF();
        curFrame->destroy();
        delete curFrame;
    }

    while (!m_mcstfOrigPicFreeList.empty())
    {
        Frame* curFrame = m_mcstfOrigPicFreeList.popBackMCSTF();
        curFrame->destroy();
        delete curFrame;
    }
}

void OrigPicBuffer::setOrigPicList(Frame* inFrame, int frameCnt)
{
    Slice* slice = inFrame->m_encData->m_slice;
    uint8_t j = 0;
    for (int iterPOC = (inFrame->m_poc - inFrame->m_mcstf->m_range);
        iterPOC <= (inFrame->m_poc + inFrame->m_mcstf->m_range); iterPOC++)
    {
        if (iterPOC != inFrame->m_poc)
        {
            if (iterPOC < 0)
                continue;
            if (iterPOC >= frameCnt)
                break;

            Frame *iterFrame = m_mcstfPicList.getPOCMCSTF(iterPOC);
            X265_CHECK(iterFrame, "Reference frame not found in OPB");
            if (iterFrame != NULL)
            {
                slice->m_mcstfRefFrameList[1][j] = iterFrame;
                iterFrame->m_refPicCnt[1]--;
            }

            iterFrame = m_mcstfOrigPicList.getPOCMCSTF(iterPOC);
            if (iterFrame != NULL)
            {

                slice->m_mcstfRefFrameList[1][j] = iterFrame;

                iterFrame->m_refPicCnt[1]--;
                Frame *cFrame = m_mcstfOrigPicList.getPOCMCSTF(inFrame->m_poc);
                X265_CHECK(cFrame, "Reference frame not found in encoded OPB");
                cFrame->m_refPicCnt[1]--;
            }
            j++;
        }
    }
}

void OrigPicBuffer::recycleOrigPicList()
{
    Frame *iterFrame = m_mcstfPicList.first();

    while (iterFrame)
    {
        Frame *curFrame = iterFrame;
        iterFrame = iterFrame->m_nextMCSTF;
        if (!curFrame->m_refPicCnt[1])
        {
            m_mcstfPicList.removeMCSTF(*curFrame);
            iterFrame = m_mcstfPicList.first();
        }
    }

    iterFrame = m_mcstfOrigPicList.first();

    while (iterFrame)
    {
        Frame *curFrame = iterFrame;
        iterFrame = iterFrame->m_nextMCSTF;
        if (!curFrame->m_refPicCnt[1])
        {
            m_mcstfOrigPicList.removeMCSTF(*curFrame);
            *curFrame->m_isSubSampled = false;
            m_mcstfOrigPicFreeList.pushFrontMCSTF(*curFrame);
            iterFrame = m_mcstfOrigPicList.first();
        }
    }
}

void OrigPicBuffer::addPictureToFreelist(Frame* inFrame)
{
    m_mcstfOrigPicFreeList.pushBack(*inFrame);
}

TemporalFilter::TemporalFilter()
{
    m_sourceWidth = 0;
    m_sourceHeight = 0,
    m_QP = 0;
    m_sliceTypeConfig = 3;
    m_numRef = 0;

    m_range = 2;
    m_chromaFactor = 0.55;
    m_sigmaMultiplier = 9.0;
    m_sigmaZeroPoint = 10.0;
    m_overallStrength = 0.95;
}

TemporalFilter::~TemporalFilter()
{
    if (m_metld)
        delete m_metld;
}

void TemporalFilter::init(const x265_param* param)
{
    m_param = param;
    m_bitDepth = param->internalBitDepth;
    m_sourceWidth = param->sourceWidth;
    m_sourceHeight = param->sourceHeight;
    m_internalCsp = param->internalCsp;
    m_numComponents = (m_internalCsp != X265_CSP_I400) ? MAX_NUM_COMPONENT : 1;

    m_metld = new MotionEstimatorTLD;
}

int TemporalFilter::createRefPicInfo(TemporalFilterRefPicInfo* refFrame, x265_param* param)
{
    CHECKED_MALLOC_ZERO(refFrame->mvs, MV, sizeof(MV)* ((m_sourceWidth ) / 4) * ((m_sourceHeight ) / 4));
    refFrame->mvsStride = m_sourceWidth / 4;
    CHECKED_MALLOC_ZERO(refFrame->mvs0, MV, sizeof(MV)* ((m_sourceWidth ) / 16) * ((m_sourceHeight ) / 16));
    refFrame->mvsStride0 = m_sourceWidth / 16;
    CHECKED_MALLOC_ZERO(refFrame->mvs1, MV, sizeof(MV)* ((m_sourceWidth ) / 16) * ((m_sourceHeight ) / 16));
    refFrame->mvsStride1 = m_sourceWidth / 16;
    CHECKED_MALLOC_ZERO(refFrame->mvs2, MV, sizeof(MV)* ((m_sourceWidth ) / 16)*((m_sourceHeight ) / 16));
    refFrame->mvsStride2 = m_sourceWidth / 16;

    CHECKED_MALLOC_ZERO(refFrame->noise, int, sizeof(int) * ((m_sourceWidth) / 4) * ((m_sourceHeight) / 4));
    CHECKED_MALLOC_ZERO(refFrame->error, int, sizeof(int) * ((m_sourceWidth) / 4) * ((m_sourceHeight) / 4));

    refFrame->slicetype = X265_TYPE_AUTO;

    refFrame->compensatedPic = new PicYuv;
    refFrame->compensatedPic->create(param, true);

    return 1;
fail:
    return 0;
}

int MotionEstimatorTLD::motionErrorLumaSSD(pixel* src,
    int stride,
    pixel* buf,
    int x,
    int y,
    int dx,
    int dy,
    int bs,
    int besterror)
{

    pixel* origOrigin = src;
    intptr_t origStride = stride;
    pixel *buffOrigin = buf;
    intptr_t buffStride = stride;
    int error = 0;// dx * 10 + dy * 10;
    if (((dx | dy) & 0xF) == 0)
    {
        dx /= m_motionVectorFactor;
        dy /= m_motionVectorFactor;

        const pixel* bufferRowStart = buffOrigin + (y + dy) * buffStride + (x + dx);
#if 0
        const pixel* origRowStart = origOrigin + y * origStride + x;

        for (int y1 = 0; y1 < bs; y1++)
        {
            for (int x1 = 0; x1 < bs; x1++)
            {
                int diff = origRowStart[x1] - bufferRowStart[x1];
                error += diff * diff;
            }

            origRowStart += origStride;
            bufferRowStart += buffStride;
        }
#else
        int partEnum = partitionFromSizes(bs, bs);
        /* copy PU block into cache */
        primitives.pu[partEnum].copy_pp(predPUYuv.m_buf[0], FENC_STRIDE, bufferRowStart, buffStride);

        error = (int)primitives.cu[partEnum].sse_pp(me.fencPUYuv.m_buf[0], FENC_STRIDE, predPUYuv.m_buf[0], FENC_STRIDE);

#endif
        if (error > besterror)
        {
            return error;
        }
    }
    else
    {
        error = mcstfPrim.motionErrorLumaFrac(
            origOrigin, origStride, buffOrigin, buffStride,
            x, y, dx, dy, bs, besterror, m_bitDepth);
        if (error > besterror) return error;
    }
    return error;
}

void TemporalFilter::applyMotion(MV *mvs, uint32_t mvsStride, PicYuv *input, PicYuv *output, const int blockRow, const int rowSize)
{
    static const int lumaBlockSize = 8;
    int srcStride = 0;
    int dstStride = 0;
    int csx = 0, csy = 0;
    for (int c = 0; c < m_numComponents; c++)
    {

        const pixel* pSrcImage = input->m_picOrg[c];
        pixel* pDstImage = output->m_picOrg[c];

        if (!c)
        {
            srcStride = (int)input->m_stride;
            dstStride = (int)output->m_stride;
        }
        else
        {
            srcStride = (int)input->m_strideC;
            dstStride = (int)output->m_strideC;
            csx = CHROMA_H_SHIFT(m_internalCsp);
            csy = CHROMA_V_SHIFT(m_internalCsp);
        }
        const int blockSizeX = lumaBlockSize >> csx;
        const int blockSizeY = lumaBlockSize >> csy;
        const int height = input->m_picHeight >> csy;
        const int width = input->m_picWidth >> csx;

        const int vShift = (!c) ? 0 : csy;

        mcstfPrim.applyMotion(pSrcImage, srcStride, pDstImage, dstStride, width, height, blockSizeX, blockSizeY, mvsStride, mvs, csx, csy, blockRow, rowSize, vShift);
    }
}

void TemporalFilter::bilateralFilterCore(Frame* frame, TemporalFilterRefPicInfo* m_mcstfRefList, int numRefs, int blockRow, int rowSize)
{

    int refStrengthRow = 0;

    const double lumaSigmaSq = (m_QP - m_sigmaZeroPoint) * (m_QP - m_sigmaZeroPoint) * m_sigmaMultiplier;
    const double chromaSigmaSq = 30 * 30;
    PicYuv* orgPic = frame->m_fencPic;

    for (int i = 0; i < numRefs; i++)
    {
        TemporalFilterRefPicInfo* ref = &m_mcstfRefList[i];
        applyMotion(m_mcstfRefList[i].mvs, m_mcstfRefList[i].mvsStride, m_mcstfRefList[i].picBuffer, ref->compensatedPic, blockRow, rowSize);
    }

    for (int c = 0; c < m_numComponents; c++)
    {
        const int csx = (!c) ? 0 : CHROMA_H_SHIFT(m_internalCsp);
        const int csy = (!c) ? 0 : CHROMA_V_SHIFT(m_internalCsp);
        const int height = (!c) ? orgPic->m_picHeight : orgPic->m_picHeight >> csy;
        const int width = (!c) ? orgPic->m_picWidth : orgPic->m_picWidth >> csx;
        pixel* srcPelPlane = orgPic->m_picOrg[c];
        const intptr_t srcStride = (!c) ? orgPic->m_stride : (intptr_t)orgPic->m_strideC;

        const double sigmaSq = (!c) ? lumaSigmaSq : chromaSigmaSq;
        const double weightScaling = m_overallStrength * ((!c) ? 0.4 : m_chromaFactor);
        const double maxSampleValue = (1 << m_bitDepth) - 1;
        const double bitDepthDiffWeighting = 1024.0 / (maxSampleValue + 1);
        const int blkSize = (!c) ? 8 : 4;

        const int vShift = (!c) ? 0 : csy;
        const int planeRowStart = (!rowSize) ? 0 : (blockRow * rowSize) >> vShift;
        const int planeRowEnd =  (!rowSize) ? height : X265_MIN((blockRow * rowSize + rowSize) >> vShift, height);
        const int blkRowStart = (planeRowStart / blkSize) * blkSize;
        const int blkRowEnd = X265_MIN(((planeRowEnd + blkSize - 1) / blkSize) * blkSize, height);

        for (int by = blkRowStart; by + blkSize <= blkRowEnd; by += blkSize)
        {
            for (int bx = 0; bx + blkSize <= width; bx += blkSize)
            {
                const pixel* srcPel = srcPelPlane + by * srcStride + bx;

                // Step 1: noise computation via SIMD primitive
                double minError = DBL_MAX;
                for (int i = 0; i < numRefs; i++)
                {
                    TemporalFilterRefPicInfo* refPicInfo = &m_mcstfRefList[i];
                    const intptr_t refStride = (!c) ? refPicInfo->compensatedPic->m_stride
                        : refPicInfo->compensatedPic->m_strideC;
                    const pixel* refPel = refPicInfo->compensatedPic->m_picOrg[c]
                        + by * refStride + bx;

                    int iVariance = 0, iDiffsum = 0;
                    mcstfPrim.computeBlockStats(
                        srcPel, srcStride, refPel, refStride,
                        blkSize, &iVariance, &iDiffsum);

                    const int cntV = blkSize * blkSize;
                    const int cntD = 2 * cntV - blkSize - blkSize;
                    refPicInfo->noise[(by / blkSize) * refPicInfo->mvsStride + (bx / blkSize)] =
                        (int)round((15.0 * cntD / cntV * iVariance + 5.0) / (iDiffsum + 5.0));
                    minError = X265_MIN(minError,
                        (double)refPicInfo->error[(by / blkSize) * refPicInfo->mvsStride + (bx / blkSize)]);
                }

                // Step 2: pre-compute vww / vsw (block-level)
                double vww[MCSTF_MAX_REFS] = {};
                double vsw[MCSTF_MAX_REFS] = {};
                for (int i = 0; i < numRefs; i++)
                {
                    TemporalFilterRefPicInfo* refPicInfo = &m_mcstfRefList[i];
                    const int error = refPicInfo->error[(by / blkSize) * refPicInfo->mvsStride + (bx / blkSize)];
                    const int noise = refPicInfo->noise[(by / blkSize) * refPicInfo->mvsStride + (bx / blkSize)];
                    const int index = X265_MIN(3, std::abs(refPicInfo->origOffset) - 1);
                    double ww = 1, sw = 1;
                    ww *= (noise < 25) ? 1 : 1.2;
                    sw *= (noise < 25) ? 1.3 : 0.8;
                    ww *= (error < 50) ? 1.2 : ((error > 100) ? 0.8 : 1);
                    sw *= (error < 50) ? 1.3 : 1;
                    ww *= ((minError + 1) / (error + 1));
                    vww[i] = weightScaling * s_refStrengths[refStrengthRow][index] * ww;
                    vsw[i] = 2 * sw * sigmaSq;
                }

                //  Step 3: pixel filtering via SIMD primitive
                const pixel* refBlkPtrs[MCSTF_MAX_REFS];
                intptr_t     refBlkStrides[MCSTF_MAX_REFS];
                for (int i = 0; i < numRefs; i++)
                {
                    TemporalFilterRefPicInfo* refPicInfo = &m_mcstfRefList[i];
                    refBlkStrides[i] = (!c) ? refPicInfo->compensatedPic->m_stride
                        : refPicInfo->compensatedPic->m_strideC;
                    refBlkPtrs[i] = refPicInfo->compensatedPic->m_picOrg[c]
                        + by * refBlkStrides[i] + bx;
                }
                mcstfPrim.bilateralFilter(srcPel, srcStride, numRefs, refBlkPtrs, refBlkStrides, vww, vsw, bitDepthDiffWeighting, maxSampleValue,
                    blkSize, srcPelPlane + by * srcStride + bx, srcStride);
            }
        }
    }
}

//  Splits the frame into 64-row blocks, dispatches jobs to the threadpool via BilateralFilterGroup, then returns.
void TemporalFilter::bilateralFilter(Frame* curFrame, TemporalFilterRefPicInfo* mcstfRefList, ThreadPool* pool)
{
    const int numRef       = curFrame->m_mcstf->m_numRef;
    const int rowSize      = 64;
    const int frameHeight  = curFrame->m_fencPic->m_picHeight;
    const int numBlockRows = (frameHeight + rowSize - 1) / rowSize;

    if (numRef == 0)
        return;

    if (!pool)
    {
        bilateralFilterCore(curFrame, mcstfRefList, numRef, 0, 0);
        return;
    }

    BilateralFilterGroup filterGroup(*this, pool);

    for (int row = 0; row < numBlockRows; row++)
        filterGroup.add(curFrame, mcstfRefList, numRef, row, rowSize);

    filterGroup.finishBatch();
}

void MotionEstimatorTLD::motionEstimationLuma(MV *mvs, uint32_t mvStride, pixel* src,int stride, int height, int width, pixel* buf,
                                              int row, const int rowSize, const MV* previous, uint32_t prevMvStride, int factor)
{

    int range = m_searchRange;
    int stepSize, blockSize;

    stepSize = blockSize = m_blockSize;

    const int origWidth = width;
    const int origHeight = height;
    int rowStart = row * rowSize;

    if (rowStart > height)
        return;

    int rowEnd = (!rowSize) ? height : X265_MIN(rowStart + rowSize, height);
    int error;

    for (int blockY = rowStart; blockY + blockSize <= rowEnd; blockY += stepSize)
    {
        for (int blockX = 0; blockX + blockSize <= origWidth; blockX += stepSize)
        {
            const intptr_t pelOffset = blockY * stride + blockX;
            me.setSourcePU(src, stride, pelOffset, blockSize, blockSize, X265_HEX_SEARCH, 1);


            MV best(0, 0);
            int leastError = INT_MAX;

            if (previous == NULL)
            {
                range = m_searchRange;
            }
            else
            {

                for (int py = -1; py <= 1; py++)
                {
                    int testy = blockY / (2 * blockSize) + py;

                    for (int px = -1; px <= 1; px++)
                    {

                        int testx = blockX / (2 * blockSize) + px;
                        if ((testx >= 0) && (testx < origWidth / (2 * blockSize)) && (testy >= 0) && (testy < origHeight / (2 * blockSize)))
                        {
                            int mvIdx = testy * prevMvStride + testx;
                            MV old = previous[mvIdx];

                            error = motionErrorLumaSSD(src, stride, buf, blockX, blockY, old.x * factor, old.y * factor, blockSize, leastError);

                            if (error < leastError)
                            {
                                best.set(old.x * factor, old.y * factor);
                                leastError = error;
                            }
                        }
                    }
                }

                error = motionErrorLumaSSD(src, stride, buf, blockX, blockY, 0, 0, blockSize, leastError);

                if (error < leastError)
                {
                    best.set(0, 0);
                    leastError = error;
                }

            }

            MV prevBest = best;
            for (int y2 = prevBest.y / m_motionVectorFactor - range; y2 <= prevBest.y / m_motionVectorFactor + range; y2++)
            {
                for (int x2 = prevBest.x / m_motionVectorFactor - range; x2 <= prevBest.x / m_motionVectorFactor + range; x2++)
                {
                    error = motionErrorLumaSSD(src, stride, buf, blockX, blockY, x2 * m_motionVectorFactor, y2 * m_motionVectorFactor, blockSize, leastError);

                    if (error < leastError)
                    {
                        best.set(x2 * m_motionVectorFactor, y2 * m_motionVectorFactor);
                        leastError = error;
                    }
                }
            }

            /* Removed above block's Motion estimation dependency as the atomicity cost outweighs quality benefit */

            if (blockX > 0)
            {
                int idx = ((blockY / stepSize) * mvStride + (blockX - stepSize) / stepSize);
                MV leftMV = mvs[idx];

                error = motionErrorLumaSSD(src, stride, buf, blockX, blockY, leftMV.x, leftMV.y, blockSize, leastError);

                if (error < leastError)
                {
                    best.set(leftMV.x, leftMV.y);
                    leastError = error;
                }
            }

            int mvIdx = (blockY / stepSize) * mvStride + (blockX / stepSize);
            mvs[mvIdx] = best;
        }
    }
}


void MotionEstimatorTLD::motionEstimationLumaDoubleRes(MV *mvs, uint32_t mvStride, PicYuv *orig, PicYuv *buffer,
                                                       const MV *previous, uint32_t prevMvStride, int factor, int* minError, int row, const int rowSize)
{

    int range = 0;
    int stepSize, blockSize;

    stepSize = blockSize = m_blockSize / 2;

    const int origWidth  = orig->m_picWidth;
    const int origHeight = orig->m_picHeight;
    int rowStart         = row * rowSize;

    if (rowStart > origHeight)
        return;   // row beyond frame edge — nothing to do

    int rowEnd = (!rowSize) ? origHeight : X265_MIN(rowStart + rowSize, origHeight);

    int error;

    for (int blockY = rowStart; blockY + blockSize <= rowEnd; blockY += stepSize)
    {
        for (int blockX = 0; blockX + blockSize <= origWidth; blockX += stepSize)
        {

            const intptr_t pelOffset = blockY * orig->m_stride + blockX;
            me.setSourcePU(orig->m_picOrg[0], orig->m_stride, pelOffset, blockSize, blockSize, X265_HEX_SEARCH, 1);

            MV best(0, 0);
            int leastError = INT_MAX;

            if (previous == NULL)
            {
                range = 8;
            }
            else
            {

                for (int py = -1; py <= 1; py++)
                {
                    int testy = blockY / (2 * blockSize) + py;

                    for (int px = -1; px <= 1; px++)
                    {

                        int testx = blockX / (2 * blockSize) + px;
                        if ((testx >= 0) && (testx < origWidth / (2 * blockSize)) && (testy >= 0) && (testy < origHeight / (2 * blockSize)))
                        {
                            int mvIdx = testy * prevMvStride + testx;
                            MV old = previous[mvIdx];

                            error = motionErrorLumaSSD(orig->m_picOrg[0], (int)orig->m_stride, buffer->m_picOrg[0], blockX, blockY, old.x * factor, old.y * factor, blockSize, leastError);

                            if (error < leastError)
                            {
                                best.set(old.x * factor, old.y * factor);
                                leastError = error;
                            }
                        }
                    }
                }

                error = motionErrorLumaSSD(orig->m_picOrg[0], (int)orig->m_stride, buffer->m_picOrg[0], blockX, blockY, 0, 0, blockSize, leastError);

                if (error < leastError)
                {
                    best.set(0, 0);
                    leastError = error;
                }

            }

            MV prevBest = best;
            for (int y2 = prevBest.y / m_motionVectorFactor - range; y2 <= prevBest.y / m_motionVectorFactor + range; y2++)
            {
                for (int x2 = prevBest.x / m_motionVectorFactor - range; x2 <= prevBest.x / m_motionVectorFactor + range; x2++)
                {
                    error = motionErrorLumaSSD(orig->m_picOrg[0], (int)orig->m_stride, buffer->m_picOrg[0], blockX, blockY, x2 * m_motionVectorFactor, y2 * m_motionVectorFactor, blockSize, leastError);

                    if (error < leastError)
                    {
                        best.set(x2 * m_motionVectorFactor, y2 * m_motionVectorFactor);
                        leastError = error;
                    }
                }
            }

            prevBest = best;
            int doubleRange = 3;
            for (int y2 = prevBest.y - doubleRange; y2 <= prevBest.y + doubleRange; y2++)
            {
                for (int x2 = prevBest.x - doubleRange; x2 <= prevBest.x + doubleRange; x2++)
                {
                    error = motionErrorLumaSSD(orig->m_picOrg[0], (int)orig->m_stride, buffer->m_picOrg[0], blockX, blockY, x2, y2, blockSize, leastError);

                    if (error < leastError)
                    {
                        best.set(x2, y2);
                        leastError = error;
                    }
                }
            }

            /* Using Above block's Motion vector only when above block is available within the same thread */
            if (blockY != rowStart)
            {
                int idx = ((blockY - stepSize) / stepSize) * mvStride + (blockX / stepSize);
                MV aboveMV = mvs[idx];

                error = motionErrorLumaSSD(orig->m_picOrg[0], (int)orig->m_stride, buffer->m_picOrg[0], blockX, blockY, aboveMV.x, aboveMV.y, blockSize, leastError);

                if (error < leastError)
                {
                    best.set(aboveMV.x, aboveMV.y);
                    leastError = error;
                }
            }

            if (blockX > 0)
            {
                int idx = ((blockY / stepSize) * mvStride + (blockX - stepSize) / stepSize);
                MV leftMV = mvs[idx];

                error = motionErrorLumaSSD(orig->m_picOrg[0], (int)orig->m_stride, buffer->m_picOrg[0], blockX, blockY, leftMV.x, leftMV.y, blockSize, leastError);

                if (error < leastError)
                {
                    best.set(leftMV.x, leftMV.y);
                    leastError = error;
                }
            }

            const pixel* blkOrigin = orig->m_picOrg[0] + blockY * orig->m_stride + blockX;
            uint64_t sumSqr = primitives.cu[partitionFromSizes(blockSize, blockSize)].var(blkOrigin, orig->m_stride);
            uint32_t sum = (uint32_t)sumSqr;
            uint32_t sqr = (uint32_t)(sumSqr >> 32);
            const int N = blockSize * blockSize;
            double variance = (double)sqr - ((double)sum * sum) / N;

            leastError = (int)(20 * ((leastError + 5.0) / (variance + 5.0)) + (leastError / (blockSize * blockSize)) / 50);

            int mvIdx = (blockY / stepSize) * mvStride + (blockX / stepSize);
            mvs[mvIdx] = best;
            minError[mvIdx] = leastError;
        }
    }
}

void TemporalFilter::destroyRefPicInfo(TemporalFilterRefPicInfo* curFrame)
{
    if (curFrame)
    {
        if (curFrame->compensatedPic)
        {
            curFrame->compensatedPic->destroy();
            delete curFrame->compensatedPic;
        }

        if (curFrame->mvs)
            X265_FREE(curFrame->mvs);
        if (curFrame->mvs0)
            X265_FREE(curFrame->mvs0);
        if (curFrame->mvs1)
            X265_FREE(curFrame->mvs1);
        if (curFrame->mvs2)
            X265_FREE(curFrame->mvs2);
        if (curFrame->noise)
            X265_FREE(curFrame->noise);
        if (curFrame->error)
            X265_FREE(curFrame->error);
    }
}
