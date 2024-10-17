/*****************************************************************************
 * Copyright (C) 2013-2020 MulticoreWare, Inc
 *
 * Authors: Steve Borho <steve@borho.org>
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
#include "frame.h"
#include "piclist.h"
#include "picyuv.h"
#include "slice.h"

using namespace X265_NS;

#if ENABLE_MULTIVIEW
void Slice::createInterLayerReferencePictureSet(PicList& picList, PicList& refPicSetInterLayer0, PicList& refPicSetInterLayer1)
{

    for (int i = 0; i < 1; i++)
    {
        Frame* refPic = picList.getPOC(m_poc, 0);
        int viewIdCur = 0;
        int viewIdZero = 1;
        int viewIdRef = 1;

        if ((viewIdCur <= viewIdZero && viewIdCur <= viewIdRef) || (viewIdCur >= viewIdZero && viewIdCur >= viewIdRef))
        {
            refPicSetInterLayer0.pushBackSubDPB(*refPic);
        }
        else
        {
            refPicSetInterLayer1.pushBackSubDPB(*refPic);
        }
    }
}
#endif

#if ENABLE_MULTIVIEW
void Slice::setRefPicList(PicList& picList, int sLayerId, PicList& refPicSetInterLayer0, PicList& refPicSetInterLayer1)
#else
void Slice::setRefPicList(PicList& picList, int sLayerId)
#endif
{
    if (m_sliceType == I_SLICE)
    {
        memset(m_refFrameList, 0, sizeof(m_refFrameList));
        memset(m_refReconPicList, 0, sizeof(m_refReconPicList));
        memset(m_refPOCList, 0, sizeof(m_refPOCList));
        m_numRefIdx[1] = m_numRefIdx[0] = 0;

#if ENABLE_SCC_EXT
        bool checkNumPocTotalCurr = m_param->bEnableSCC ? false : true;
        if (!checkNumPocTotalCurr)
        {
            if (m_rps.numberOfPictures == 0)
            {
                Frame* prevPic = picList.getPOC(X265_MAX(0, m_poc - 1));
                if (prevPic->m_poc != X265_MAX(0, m_poc - 1))
                {
                    prevPic = picList.getPOC(m_poc);
                }
                m_lastEncPic = prevPic;
            }
            return;
        }
#endif

        return;
    }

#if ENABLE_SCC_EXT || ENABLE_MULTIVIEW || ENABLE_ALPHA
    /*Reset the number of references for I-slice marked as P-slice*/
    if ((m_param->bEnableSCC || sLayerId) && m_sliceType != m_origSliceType)
    {
        memset(m_refFrameList, 0, sizeof(m_refFrameList));
        memset(m_refReconPicList, 0, sizeof(m_refReconPicList));
        memset(m_refPOCList, 0, sizeof(m_refPOCList));
        m_numRefIdx[0] = 1;
    }
#endif

#if ENABLE_SCC_EXT
    bool checkNumPocTotalCurr = m_param->bEnableSCC ? false : true;
    if (!checkNumPocTotalCurr && m_rps.numberOfPictures == 0)
    {
        Frame* prevPic = picList.getPOC(X265_MAX(0, m_poc - 1));
        if (prevPic->m_poc != X265_MAX(0, m_poc - 1))
        {
            prevPic = picList.getPOC(m_poc);

        }
        m_lastEncPic = prevPic;
    }
#endif

    Frame* refPic = NULL;
    Frame* refPicSetStCurr0[MAX_NUM_REF];
    Frame* refPicSetStCurr1[MAX_NUM_REF];
    Frame* refPicSetLtCurr[MAX_NUM_REF];
    int numPocStCurr0 = 0;
    int numPocStCurr1 = 0;
    int numPocLtCurr = 0;
    int i;

    for (i = 0; i < m_rps.numberOfNegativePictures; i++)
    {
        if (m_rps.bUsed[i] && m_origSliceType != I_SLICE)
        {
            refPic = picList.getPOC(m_poc + m_rps.deltaPOC[i], m_rps.deltaPOC[i] ? sLayerId : 0);
            refPicSetStCurr0[numPocStCurr0] = refPic;
            numPocStCurr0++;
        }
    }

    for (; i < m_rps.numberOfNegativePictures + m_rps.numberOfPositivePictures; i++)
    {
        if (m_rps.bUsed[i] && m_origSliceType != I_SLICE)
        {
            refPic = picList.getPOC(m_poc + m_rps.deltaPOC[i], m_rps.deltaPOC[i] ? sLayerId : 0);
            refPicSetStCurr1[numPocStCurr1] = refPic;
            numPocStCurr1++;
        }
    }

    X265_CHECK(m_rps.numberOfPictures == m_rps.numberOfNegativePictures + m_rps.numberOfPositivePictures,
               "unexpected picture in RPS\n");

    // ref_pic_list_init
    Frame* rpsCurrList0[MAX_NUM_REF + 1];
    Frame* rpsCurrList1[MAX_NUM_REF + 1];
#if ENABLE_MULTIVIEW
    int numPocTotalCurr = numPocStCurr0 + numPocStCurr1 + numPocLtCurr + refPicSetInterLayer0.size() + refPicSetInterLayer1.size();
#else
    int numPocTotalCurr = numPocStCurr0 + numPocStCurr1 + numPocLtCurr;
#endif

#if ENABLE_SCC_EXT
    if (m_param->bEnableSCC)
        numPocTotalCurr++;
#endif

    int cIdx = 0;
    for (i = 0; i < numPocStCurr0; i++, cIdx++)
        rpsCurrList0[cIdx] = refPicSetStCurr0[i];

#if ENABLE_MULTIVIEW
    if (m_param->numViews > 1)
        for (i = 0; i < refPicSetInterLayer0.size(); i++, cIdx++)
            rpsCurrList0[cIdx] = refPicSetInterLayer0.getPOC(m_poc, 0);
#endif

    for (i = 0; i < numPocStCurr1; i++, cIdx++)
        rpsCurrList0[cIdx] = refPicSetStCurr1[i];

    for (i = 0; i < numPocLtCurr; i++, cIdx++)
        rpsCurrList0[cIdx] = refPicSetLtCurr[i];

#if ENABLE_MULTIVIEW
    if (m_param->numViews > 1)
        for (i = 0; i < refPicSetInterLayer1.size(); i++, cIdx++)
            rpsCurrList0[cIdx] = refPicSetInterLayer1.getPOC(m_poc, 0);
#endif

#if ENABLE_SCC_EXT
    if (m_param->bEnableSCC)
        rpsCurrList0[cIdx++] = picList.getPOC(m_poc);
#endif

    X265_CHECK(cIdx == numPocTotalCurr, "RPS index check fail\n");

    if (m_sliceType == B_SLICE)
    {
        cIdx = 0;
        for (i = 0; i < numPocStCurr1; i++, cIdx++)
            rpsCurrList1[cIdx] = refPicSetStCurr1[i];

#if ENABLE_MULTIVIEW
        if (m_param->numViews > 1)
            for (i = 0; i < refPicSetInterLayer1.size(); i++, cIdx++)
                rpsCurrList1[cIdx] = refPicSetInterLayer1.getPOC(m_poc, 0);
#endif

        for (i = 0; i < numPocStCurr0; i++, cIdx++)
            rpsCurrList1[cIdx] = refPicSetStCurr0[i];

        for (i = 0; i < numPocLtCurr; i++, cIdx++)
            rpsCurrList1[cIdx] = refPicSetLtCurr[i];

#if ENABLE_MULTIVIEW
        if (m_param->numViews > 1)
            for (i = 0; i < refPicSetInterLayer0.size(); i++, cIdx++)
                rpsCurrList1[cIdx] = refPicSetInterLayer0.getPOC(m_poc, 0);
#endif

#if  ENABLE_SCC_EXT
        if (m_param->bEnableSCC)
            rpsCurrList1[cIdx++] = picList.getPOC(m_poc);
#endif

        X265_CHECK(cIdx == numPocTotalCurr, "RPS index check fail\n");
    }

    for (int rIdx = 0; rIdx < m_numRefIdx[0]; rIdx++)
    {
        cIdx = rIdx % numPocTotalCurr;
        X265_CHECK(cIdx >= 0 && cIdx < numPocTotalCurr, "RPS index check fail\n");
        m_refFrameList[0][rIdx] = rpsCurrList0[cIdx];
#if ENABLE_MULTIVIEW
        m_refFrameList[0][rIdx] = rpsCurrList0[cIdx];
#endif
    }

#if  ENABLE_SCC_EXT
    if (m_param->bEnableSCC && numPocTotalCurr > m_numRefIdx[0])
    {
        m_refFrameList[0][m_numRefIdx[0] - 1] = picList.getPOC(m_poc);
    }
#endif

    if (m_sliceType != B_SLICE)
    {
        m_numRefIdx[1] = 0;
        memset(m_refFrameList[1], 0, sizeof(m_refFrameList[1]));
    }
    else
    {
        for (int rIdx = 0; rIdx < m_numRefIdx[1]; rIdx++)
        {
            cIdx = rIdx % numPocTotalCurr;
            X265_CHECK(cIdx >= 0 && cIdx < numPocTotalCurr, "RPS index check fail\n");
            m_refFrameList[1][rIdx] = rpsCurrList1[cIdx];
#if ENABLE_MULTIVIEW
            m_refFrameList[1][rIdx] = rpsCurrList1[cIdx];
#endif
        }
    }

    for (int dir = 0; dir < 2; dir++)
        for (int numRefIdx = 0; numRefIdx < m_numRefIdx[dir]; numRefIdx++)
            m_refPOCList[dir][numRefIdx] = m_refFrameList[dir][numRefIdx]->m_poc;
}

void Slice::disableWeights()
{
    for (int l = 0; l < 2; l++)
        for (int i = 0; i < MAX_NUM_REF; i++)
            for (int yuv = 0; yuv < 3; yuv++)
            {
                WeightParam& wp = m_weightPredTable[l][i][yuv];
                wp.wtPresent = 0;
                wp.log2WeightDenom = 0;
                wp.inputWeight = 1;
                wp.inputOffset = 0;
            }
}

#if  ENABLE_SCC_EXT
bool Slice::isOnlyCurrentPictureAsReference() const
{
    if (m_sliceType == I_SLICE)
    {
        return true;
    }

    for (int i = 0; i < m_numRefIdx[0]; i++)
    {
        if (m_refFrameList[0][i]->m_poc != m_poc)
        {
            return false;
        }
    }

    for (int i = 0; i < m_numRefIdx[1]; i++)
    {
        if (m_refFrameList[1][i]->m_poc != m_poc)
        {
            return false;
        }
    }

    return true;
}
#endif

/* Sorts the deltaPOC and Used by current values in the RPS based on the
 * deltaPOC values.  deltaPOC values are sorted with -ve values before the +ve
 * values.  -ve values are in decreasing order.  +ve values are in increasing
 * order */
void RPS::sortDeltaPOC()
{
    // sort in increasing order (smallest first)
    for (int j = 1; j < numberOfPictures; j++)
    {
        int dPOC = deltaPOC[j];
        bool used = bUsed[j];
        for (int k = j - 1; k >= 0; k--)
        {
            int temp = deltaPOC[k];
            if (dPOC < temp)
            {
                deltaPOC[k + 1] = temp;
                bUsed[k + 1] = bUsed[k];
                deltaPOC[k] = dPOC;
                bUsed[k] = used;
            }
        }
    }

    // flip the negative values to largest first
    int numNegPics = numberOfNegativePictures;
    for (int j = 0, k = numNegPics - 1; j < numNegPics >> 1; j++, k--)
    {
        int dPOC = deltaPOC[j];
        bool used = bUsed[j];
        deltaPOC[j] = deltaPOC[k];
        bUsed[j] = bUsed[k];
        deltaPOC[k] = dPOC;
        bUsed[k] = used;
    }
}

uint32_t Slice::realEndAddress(uint32_t endCUAddr) const
{
    // Calculate end address
    uint32_t internalAddress = (endCUAddr - 1) % m_param->num4x4Partitions;
    uint32_t externalAddress = (endCUAddr - 1) / m_param->num4x4Partitions;
    uint32_t xmax = m_sps->picWidthInLumaSamples - (externalAddress % m_sps->numCuInWidth) * m_param->maxCUSize;
    uint32_t ymax = m_sps->picHeightInLumaSamples - (externalAddress / m_sps->numCuInWidth) * m_param->maxCUSize;

    while (g_zscanToPelX[internalAddress] >= xmax || g_zscanToPelY[internalAddress] >= ymax)
        internalAddress--;

    internalAddress++;
    if (internalAddress == m_param->num4x4Partitions)
    {
        internalAddress = 0;
        externalAddress++;
    }

    return externalAddress * m_param->num4x4Partitions + internalAddress;
}


