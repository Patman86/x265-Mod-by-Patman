/*****************************************************************************
 * Copyright (C) 2013-2020 MulticoreWare, Inc
 *
 * Authors: Steve Borho <steve@borho.org>
 *          Min Chen <chenm003@163.com>
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
#include "framedata.h"
#include "picyuv.h"
#include "slice.h"

#include "dpb.h"

using namespace X265_NS;

DPB::~DPB()
{
    while (!m_freeList.empty())
    {
        Frame* curFrame = m_freeList.popFront();
        curFrame->destroy();
        delete curFrame;
    }

    while (!m_picList.empty())
    {
        Frame* curFrame = m_picList.popFront();
        curFrame->destroy();
        delete curFrame;
    }

    while (m_frameDataFreeList)
    {
        FrameData* next = m_frameDataFreeList->m_freeListNext;
        m_frameDataFreeList->destroy();

        m_frameDataFreeList->m_reconPic[0]->destroy();
        delete m_frameDataFreeList->m_reconPic[0];

        delete m_frameDataFreeList;
        m_frameDataFreeList = next;
    }
}

// move unreferenced pictures from picList to freeList for recycle
void DPB::recycleUnreferenced()
{
    Frame *iterFrame = m_picList.first();

    while (iterFrame)
    {
        Frame *curFrame = iterFrame;
        iterFrame = iterFrame->m_next;
        bool isMCSTFReferenced = false;

        if (curFrame->m_param->bEnableTemporalFilter)
            isMCSTFReferenced =!!(curFrame->m_refPicCnt[1]);

        if (curFrame->m_valid && !curFrame->m_encData->m_bHasReferences && !curFrame->m_countRefEncoders && !isMCSTFReferenced)
        {
            curFrame->m_bChromaExtended = false;

            if (curFrame->m_param->bEnableTemporalFilter)
                *curFrame->m_isSubSampled = false;

            // Reset column counter
            X265_CHECK(curFrame->m_reconRowFlag != NULL, "curFrame->m_reconRowFlag check failure");
            X265_CHECK(curFrame->m_reconColCount != NULL, "curFrame->m_reconColCount check failure");
            X265_CHECK(curFrame->m_numRows > 0, "curFrame->m_numRows check failure");

            for(int32_t row = 0; row < curFrame->m_numRows; row++)
            {
                curFrame->m_reconRowFlag[row].set(0);
                curFrame->m_reconColCount[row].set(0);
            }

            // iterator is invalidated by remove, restart scan
            m_picList.remove(*curFrame);
#if ENABLE_MULTIVIEW
            if (curFrame->m_param->numViews > 1 && !curFrame->m_viewId && m_picList.getPOC(curFrame->m_poc, 1) && curFrame == m_picList.getPOC(curFrame->m_poc, 1)->refPicSetInterLayer0.getPOC(curFrame->m_poc, curFrame->m_viewId))
            {
                m_picList.getPOC(curFrame->m_poc, 1)->refPicSetInterLayer0.removeSubDPB(*curFrame);
            }
#endif
            iterFrame = m_picList.first();

            m_freeList.pushBack(*curFrame);
            curFrame->m_encData->m_freeListNext = m_frameDataFreeList;
            m_frameDataFreeList = curFrame->m_encData;
            for (int i = 0; i < INTEGRAL_PLANE_NUM; i++)
            {
                if (curFrame->m_encData->m_meBuffer[i] != NULL)
                {
                    X265_FREE(curFrame->m_encData->m_meBuffer[i]);
                    curFrame->m_encData->m_meBuffer[i] = NULL;
                }
            }
            if (curFrame->m_ctuInfo != NULL)
            {
                uint32_t widthInCU = (curFrame->m_param->sourceWidth + curFrame->m_param->maxCUSize - 1) >> curFrame->m_param->maxLog2CUSize;
                uint32_t heightInCU = (curFrame->m_param->sourceHeight + curFrame->m_param->maxCUSize - 1) >> curFrame->m_param->maxLog2CUSize;
                uint32_t numCUsInFrame = widthInCU * heightInCU;
                for (uint32_t i = 0; i < numCUsInFrame; i++)
                {
                    X265_FREE((*curFrame->m_ctuInfo + i)->ctuInfo);
                    (*curFrame->m_ctuInfo + i)->ctuInfo = NULL;
                }
                X265_FREE(*curFrame->m_ctuInfo);
                *(curFrame->m_ctuInfo) = NULL;
                X265_FREE(curFrame->m_ctuInfo);
                curFrame->m_ctuInfo = NULL;
                X265_FREE(curFrame->m_prevCtuInfoChange);
                curFrame->m_prevCtuInfoChange = NULL;
            }
            curFrame->m_encData = NULL;
            for (int i = 0; i < !!curFrame->m_param->bEnableSCC + 1; i++)
                curFrame->m_reconPic[i] = NULL;
        }
    }
}

void DPB::prepareEncode(Frame *newFrame)
{
    Slice* slice = newFrame->m_encData->m_slice;
    slice->m_poc = newFrame->m_poc;
    slice->m_fieldNum = newFrame->m_fieldNum;

    int pocCurr = slice->m_poc;
    int type = newFrame->m_lowres.sliceType;
    bool bIsKeyFrame = newFrame->m_lowres.bKeyframe;
    slice->m_nalUnitType = getNalUnitType(pocCurr, bIsKeyFrame);
    if (slice->m_nalUnitType == NAL_UNIT_CODED_SLICE_IDR_W_RADL || slice->m_nalUnitType == NAL_UNIT_CODED_SLICE_IDR_N_LP)
        m_lastIDR = pocCurr;
    slice->m_lastIDR = m_lastIDR;
    slice->m_sliceType = IS_X265_TYPE_B(type) ? B_SLICE : (type == X265_TYPE_P) ? P_SLICE : I_SLICE;
#if ENABLE_SCC_EXT
    if (slice->m_param->bEnableSCC)        slice->m_origSliceType = slice->m_sliceType;
    if (slice->m_param->bEnableSCC && IS_X265_TYPE_I(type))
        slice->m_sliceType = P_SLICE;
#endif

    if (type == X265_TYPE_B)
    {
        newFrame->m_encData->m_bHasReferences = false;

        newFrame->m_tempLayer = (newFrame->m_param->bEnableTemporalSubLayers && !m_bTemporalSublayer) ? 1 : newFrame->m_tempLayer;
        // Adjust NAL type for unreferenced B frames (change from _R "referenced"
        // to _N "non-referenced" NAL unit type)
        switch (slice->m_nalUnitType)
        {
        case NAL_UNIT_CODED_SLICE_TRAIL_R:
            slice->m_nalUnitType = newFrame->m_param->bEnableTemporalSubLayers ? NAL_UNIT_CODED_SLICE_TSA_N : NAL_UNIT_CODED_SLICE_TRAIL_N;
            break;
        case NAL_UNIT_CODED_SLICE_RADL_R:
            slice->m_nalUnitType = NAL_UNIT_CODED_SLICE_RADL_N;
            break;
        case NAL_UNIT_CODED_SLICE_RASL_R:
            slice->m_nalUnitType = NAL_UNIT_CODED_SLICE_RASL_N;
            break;
        default:
            break;
        }
    }
    else
    {
        /* m_bHasReferences starts out as true for non-B pictures, and is set to false
         * once no more pictures reference it */
        newFrame->m_encData->m_bHasReferences = true;
    }

    m_picList.pushFront(*newFrame);

    int layer = slice->m_param->numViews > 1 ? newFrame->m_viewId : (slice->m_param->numScalableLayers > 1) ? newFrame->m_sLayerId : 0;
    if (m_bTemporalSublayer && getTemporalLayerNonReferenceFlag(layer))
    {
        switch (slice->m_nalUnitType)
        {
        case NAL_UNIT_CODED_SLICE_TRAIL_R:
            slice->m_nalUnitType =  NAL_UNIT_CODED_SLICE_TRAIL_N;
            break;
        case NAL_UNIT_CODED_SLICE_RADL_R:
            slice->m_nalUnitType = NAL_UNIT_CODED_SLICE_RADL_N;
            break;
        case NAL_UNIT_CODED_SLICE_RASL_R:
            slice->m_nalUnitType = NAL_UNIT_CODED_SLICE_RASL_N;
            break;
        default:
            break;
        }
    }
    // Do decoding refresh marking if any
    decodingRefreshMarking(pocCurr, slice->m_nalUnitType, layer);

    uint32_t maxDecBuffer = (slice->m_sps->maxDecPicBuffering[newFrame->m_tempLayer] >= 8 && slice->m_param->bEnableSCC) ? 7 : slice->m_sps->maxDecPicBuffering[newFrame->m_tempLayer];
    computeRPS(pocCurr, newFrame->m_tempLayer, slice->isIRAP(), &slice->m_rps, maxDecBuffer, layer);
    bool isTSAPic = ((slice->m_nalUnitType == 2) || (slice->m_nalUnitType == 3)) ? true : false;
    // Mark pictures in m_piclist as unreferenced if they are not included in RPS
    applyReferencePictureSet(&slice->m_rps, pocCurr, newFrame->m_tempLayer, isTSAPic, layer);


    if (m_bTemporalSublayer && newFrame->m_tempLayer > 0
        && !(slice->m_nalUnitType == NAL_UNIT_CODED_SLICE_RADL_N     // Check if not a leading picture
            || slice->m_nalUnitType == NAL_UNIT_CODED_SLICE_RADL_R
            || slice->m_nalUnitType == NAL_UNIT_CODED_SLICE_RASL_N
            || slice->m_nalUnitType == NAL_UNIT_CODED_SLICE_RASL_R)
        )
    {
        if (isTemporalLayerSwitchingPoint(pocCurr, newFrame->m_tempLayer, layer) || (slice->m_sps->maxTempSubLayers == 1))
        {
            if (getTemporalLayerNonReferenceFlag(layer))
            {
                slice->m_nalUnitType = NAL_UNIT_CODED_SLICE_TSA_N;
            }
            else
            {
                slice->m_nalUnitType = NAL_UNIT_CODED_SLICE_TSA_R;
            }
        }
        else if (isStepwiseTemporalLayerSwitchingPoint(&slice->m_rps, pocCurr, newFrame->m_tempLayer, layer))
        {
            bool isSTSA = true;
            int id = newFrame->m_gopOffset % x265_gop_ra_length[newFrame->m_gopId];
            for (int ii = id; (ii < x265_gop_ra_length[newFrame->m_gopId] && isSTSA == true); ii++)
            {
                int tempIdRef = x265_gop_ra[newFrame->m_gopId][ii].layer;
                if (tempIdRef == newFrame->m_tempLayer)
                {
                    for (int jj = 0; jj < slice->m_rps.numberOfPositivePictures + slice->m_rps.numberOfNegativePictures; jj++)
                    {
                        if (slice->m_rps.bUsed[jj])
                        {
                            int refPoc = x265_gop_ra[newFrame->m_gopId][ii].poc_offset + slice->m_rps.deltaPOC[jj];
                            int kk = 0;
                            for (kk = 0; kk < x265_gop_ra_length[newFrame->m_gopId]; kk++)
                            {
                                if (x265_gop_ra[newFrame->m_gopId][kk].poc_offset == refPoc)
                                {
                                    break;
                                }
                            }
                            if (x265_gop_ra[newFrame->m_gopId][kk].layer >= newFrame->m_tempLayer)
                            {
                                isSTSA = false;
                                break;
                            }
                        }
                    }
                }
            }
            if (isSTSA == true)
            {
                if (getTemporalLayerNonReferenceFlag(layer))
                {
                    slice->m_nalUnitType = NAL_UNIT_CODED_SLICE_STSA_N;
                }
                else
                {
                    slice->m_nalUnitType = NAL_UNIT_CODED_SLICE_STSA_R;
                }
            }
        }
    }

#if ENABLE_MULTIVIEW
    if (newFrame->m_viewId)
        slice->createInterLayerReferencePictureSet(m_picList, newFrame->refPicSetInterLayer0, newFrame->refPicSetInterLayer1);
#endif
    int numRef = slice->m_param->bEnableSCC ? slice->m_rps.numberOfNegativePictures + 1 : slice->m_rps.numberOfNegativePictures;
    if (slice->m_sliceType != I_SLICE)
        slice->m_numRefIdx[0] = x265_clip3(1, newFrame->m_param->maxNumReferences, numRef + newFrame->refPicSetInterLayer0.size() + newFrame->refPicSetInterLayer1.size());
    else
        slice->m_numRefIdx[0] = X265_MIN(newFrame->m_param->maxNumReferences, numRef); // Ensuring L0 contains just the -ve POC
#if ENABLE_MULTIVIEW || ENABLE_SCC_EXT
    if(slice->m_param->numViews > 1 || !!slice->m_param->bEnableSCC)
        slice->m_numRefIdx[1] = X265_MIN(newFrame->m_param->bBPyramid ? 3 : 2, slice->m_rps.numberOfPositivePictures + newFrame->refPicSetInterLayer0.size() + newFrame->refPicSetInterLayer1.size());
    else
#endif
        slice->m_numRefIdx[1] = X265_MIN(newFrame->m_param->bBPyramid ? 2 : 1, slice->m_rps.numberOfPositivePictures);
#if ENABLE_MULTIVIEW
    slice->setRefPicList(m_picList, layer, newFrame->refPicSetInterLayer0, newFrame->refPicSetInterLayer1);
#else
    slice->setRefPicList(m_picList, layer);
#endif

    X265_CHECK(slice->m_sliceType != B_SLICE || slice->m_numRefIdx[1], "B slice without L1 references (non-fatal)\n");

    if (slice->m_sliceType == B_SLICE)
    {
        /* TODO: the lookahead should be able to tell which reference picture
         * had the least motion residual.  We should be able to use that here to
         * select a colocation reference list and index */

        bool bLowDelay = true;
        int  iCurrPOC = slice->m_poc;
        int iRefIdx = 0;

        for (iRefIdx = 0; iRefIdx < slice->m_numRefIdx[0] && bLowDelay; iRefIdx++)
        {
            if (slice->m_refPOCList[0][iRefIdx] > iCurrPOC)
            {
                bLowDelay = false;
            }
        }
        for (iRefIdx = 0; iRefIdx < slice->m_numRefIdx[1] && bLowDelay; iRefIdx++)
        {
            if (slice->m_refPOCList[1][iRefIdx] > iCurrPOC)
            {
                bLowDelay = false;
            }
        }

        slice->m_bCheckLDC = bLowDelay;
        slice->m_colFromL0Flag = bLowDelay;
        slice->m_colRefIdx = 0;
    }
    else
    {
        slice->m_bCheckLDC = true;
        slice->m_colFromL0Flag = true;
        slice->m_colRefIdx = 0;
    }

    slice->m_bTemporalMvp = slice->m_sps->bTemporalMVPEnabled;
#if ENABLE_SCC_EXT
    bool bGPBcheck = false;
    if (slice->m_sliceType == B_SLICE)
    {
        if (slice->m_param->bEnableSCC)
        {
            if (slice->m_numRefIdx[0] - 1 == slice->m_numRefIdx[1])
            {
                bGPBcheck = true;
                for (int i = 0; i < slice->m_numRefIdx[1]; i++)
                {
                    if (slice->m_refPOCList[1][i] != slice->m_refPOCList[0][i])
                    {
                        bGPBcheck = false;
                        break;
                    }
                }
            }
        }
        else if (slice->m_numRefIdx[0] == slice->m_numRefIdx[1])
        {
            bGPBcheck = true;
            int i;
            for (i = 0; i < slice->m_numRefIdx[1]; i++)
            {
                if (slice->m_refPOCList[1][i] != slice->m_refPOCList[0][i])
                {
                    bGPBcheck = false;
                    break;
                }
            }
        }
    }
    if (bGPBcheck)
    {
        slice->m_bLMvdL1Zero = true;
    }
    else
    {
        slice->m_bLMvdL1Zero = false;
    }

    if (!slice->isIntra() && slice->m_param->bEnableTemporalMvp)
    {
        const Frame* colPic = slice->m_refFrameList[slice->isInterB() && !slice->m_colFromL0Flag][slice->m_colRefIdx];
        if (colPic->m_poc == slice->m_poc)
            slice->m_bTemporalMvp = false;
        else
            slice->m_bTemporalMvp = true;
    }
#endif

    // Disable Loopfilter in bound area, because we will do slice-parallelism in future
    slice->m_sLFaseFlag = (newFrame->m_param->maxSlices > 1) ? false : ((SLFASE_CONSTANT & (1 << (pocCurr % 31))) > 0);

    /* Increment reference count of all motion-referenced frames to prevent them
     * from being recycled. These counts are decremented at the end of
     * compressFrame() */
    int numPredDir = slice->isInterP() ? 1 : slice->isInterB() ? 2 : 0;
    for (int l = 0; l < numPredDir; l++)
    {
        for (int ref = 0; ref < slice->m_numRefIdx[l]; ref++)
        {
            Frame *refpic = slice->m_refFrameList[l][ref];
            ATOMIC_INC(&refpic->m_countRefEncoders);
        }
    }
}

void DPB::computeRPS(int curPoc, int tempId, bool isRAP, RPS * rps, unsigned int maxDecPicBuffer, int scalableLayerId)
{
    unsigned int poci = 0, numNeg = 0, numPos = 0;

    Frame* iterPic = m_picList.first();

    while (iterPic && (poci < maxDecPicBuffer - 1))
    {
        int layer = iterPic->m_param->numViews > 1 ? iterPic->m_viewId : (iterPic->m_param->numScalableLayers > 1) ? iterPic->m_sLayerId : 0;
        if (iterPic->m_valid && (iterPic->m_poc != curPoc) && iterPic->m_encData->m_bHasReferences && layer == scalableLayerId)
        {
            if ((!m_bTemporalSublayer || (iterPic->m_tempLayer <= tempId)) && ((m_lastIDR >= curPoc) || (m_lastIDR <= iterPic->m_poc)))
            {
#if ENABLE_MULTIVIEW
                    if (iterPic->m_param->numViews > 1 && layer && numNeg == (uint8_t)(iterPic->m_param->maxNumReferences - 1) && (iterPic->m_poc - curPoc) < 0)
                    {
                        iterPic = iterPic->m_next;
                        continue;
                    }
#endif
                    rps->poc[poci] = iterPic->m_poc;
                    rps->deltaPOC[poci] = rps->poc[poci] - curPoc;
                    (rps->deltaPOC[poci] < 0) ? numNeg++ : numPos++;
                    rps->bUsed[poci] = !isRAP;
                    poci++;
            }
        }
        iterPic = iterPic->m_next;
    }

    rps->numberOfPictures = poci;
    rps->numberOfPositivePictures = numPos;
    rps->numberOfNegativePictures = numNeg;

    rps->sortDeltaPOC();
}

bool DPB::getTemporalLayerNonReferenceFlag(int scalableLayerId)
{
    Frame* curFrame = m_picList.first();
    int layer = curFrame->m_param->numViews > 1 ? curFrame->m_viewId : (curFrame->m_param->numScalableLayers > 1) ? curFrame->m_sLayerId : 0;
    if (curFrame->m_valid && curFrame->m_encData->m_bHasReferences && layer == scalableLayerId)
    {
        curFrame->m_sameLayerRefPic = true;
        return false;
    }
    else
        return true;
}

/* Marking reference pictures when an IDR/CRA is encountered. */
void DPB::decodingRefreshMarking(int pocCurr, NalUnitType nalUnitType, int scalableLayerId)
{
    if (nalUnitType == NAL_UNIT_CODED_SLICE_IDR_W_RADL || nalUnitType == NAL_UNIT_CODED_SLICE_IDR_N_LP)
    {
        /* If the nal_unit_type is IDR, all pictures in the reference picture
         * list are marked as "unused for reference" */
        Frame* iterFrame = m_picList.first();
        while (iterFrame)
        {
            int layer = iterFrame->m_param->numViews > 1 ? iterFrame->m_viewId : (iterFrame->m_param->numScalableLayers > 1) ? iterFrame->m_sLayerId : 0;
            if (iterFrame->m_valid && iterFrame->m_poc != pocCurr && layer == scalableLayerId)
                iterFrame->m_encData->m_bHasReferences = false;
            iterFrame = iterFrame->m_next;
        }
    }
    else // CRA or No DR
    {
        if (m_bRefreshPending && pocCurr > m_pocCRA)
        {
            /* If the bRefreshPending flag is true (a deferred decoding refresh
             * is pending) and the current temporal reference is greater than
             * the temporal reference of the latest CRA picture (pocCRA), mark
             * all reference pictures except the latest CRA picture as "unused
             * for reference" and set the bRefreshPending flag to false */
            Frame* iterFrame = m_picList.first();
            while (iterFrame)
            {
                int layer = iterFrame->m_param->numViews > 1 ? iterFrame->m_viewId : (iterFrame->m_param->numScalableLayers > 1) ? iterFrame->m_sLayerId : 0;
                if (iterFrame->m_valid && iterFrame->m_poc != pocCurr && iterFrame->m_poc != m_pocCRA && layer == scalableLayerId)
                    iterFrame->m_encData->m_bHasReferences = false;
                iterFrame = iterFrame->m_next;
            }

            if (scalableLayerId == m_picList.first()->m_param->numLayers - 1)
                m_bRefreshPending = false;
        }
        if (nalUnitType == NAL_UNIT_CODED_SLICE_CRA)
        {
            /* If the nal_unit_type is CRA, set the bRefreshPending flag to true
             * and pocCRA to the temporal reference of the current picture */
            m_bRefreshPending = true;
            m_pocCRA = pocCurr;
        }
    }

    /* Note that the current picture is already placed in the reference list and
     * its marking is not changed.  If the current picture has a nal_ref_idc
     * that is not 0, it will remain marked as "used for reference" */
}

/** Function for applying picture marking based on the Reference Picture Set */
void DPB::applyReferencePictureSet(RPS *rps, int curPoc, int tempId, bool isTSAPicture, int scalableLayerId)
{
    // loop through all pictures in the reference picture buffer
    Frame* iterFrame = m_picList.first();
    while (iterFrame)
    {
        int layer = iterFrame->m_param->numViews > 1 ? iterFrame->m_viewId : (iterFrame->m_param->numScalableLayers > 1) ? iterFrame->m_sLayerId : 0;
        if (iterFrame->m_valid && iterFrame->m_poc != curPoc && iterFrame->m_encData->m_bHasReferences && layer == scalableLayerId)
        {
            // loop through all pictures in the Reference Picture Set
            // to see if the picture should be kept as reference picture
            bool referenced = false;
            for (int i = 0; i < rps->numberOfPositivePictures + rps->numberOfNegativePictures; i++)
            {
                if (iterFrame->m_poc == curPoc + rps->deltaPOC[i])
                {
                    referenced = true;
                    break;
                }
            }
            if (!referenced)
                iterFrame->m_encData->m_bHasReferences = false;

            if (m_bTemporalSublayer)
            {
                //check that pictures of higher temporal layers are not used
                assert(referenced == 0 || iterFrame->m_encData->m_bHasReferences == false || iterFrame->m_tempLayer <= tempId);

                //check that pictures of higher or equal temporal layer are not in the RPS if the current picture is a TSA picture
                if (isTSAPicture)
                {
                    assert(referenced == 0 || iterFrame->m_tempLayer < tempId);
                }
                //check that pictures marked as temporal layer non-reference pictures are not used for reference
                if (iterFrame->m_tempLayer == tempId)
                {
                    assert(referenced == 0 || iterFrame->m_sameLayerRefPic == true);
                }
            }
        }
        iterFrame = iterFrame->m_next;
    }
}

bool DPB::isTemporalLayerSwitchingPoint(int curPoc, int tempId, int scalableLayerId)
{
    // loop through all pictures in the reference picture buffer
    Frame* iterFrame = m_picList.first();
    while (iterFrame)
    {
        int layer = iterFrame->m_param->numViews > 1 ? iterFrame->m_viewId : (iterFrame->m_param->numScalableLayers > 1) ? iterFrame->m_sLayerId : 0;
        if (iterFrame->m_valid && iterFrame->m_poc != curPoc && iterFrame->m_encData->m_bHasReferences && layer == scalableLayerId)
        {
            if (iterFrame->m_tempLayer >= tempId)
            {
                return false;
            }
        }
        iterFrame = iterFrame->m_next;
    }
    return true;
}

bool DPB::isStepwiseTemporalLayerSwitchingPoint(RPS *rps, int curPoc, int tempId, int scalableLayerId)
{
    // loop through all pictures in the reference picture buffer
    Frame* iterFrame = m_picList.first();
    while (iterFrame)
    {
        int layer = iterFrame->m_param->numViews > 1 ? iterFrame->m_viewId : (iterFrame->m_param->numScalableLayers > 1) ? iterFrame->m_sLayerId : 0;
        if (iterFrame->m_valid && iterFrame->m_poc != curPoc && iterFrame->m_encData->m_bHasReferences && layer == scalableLayerId)
        {
            for (int i = 0; i < rps->numberOfPositivePictures + rps->numberOfNegativePictures; i++)
            {
                if ((iterFrame->m_poc == curPoc + rps->deltaPOC[i]) && rps->bUsed[i])
                {
                    if (iterFrame->m_tempLayer >= tempId)
                    {
                        return false;
                    }
                }
            }
        }
        iterFrame = iterFrame->m_next;
    }
    return true;
}

/* deciding the nal_unit_type */
NalUnitType DPB::getNalUnitType(int curPOC, bool bIsKeyFrame)
{
    if (!curPOC)
        return NAL_UNIT_CODED_SLICE_IDR_N_LP;
    if (bIsKeyFrame)
        return (m_bOpenGOP || m_craNal) ? NAL_UNIT_CODED_SLICE_CRA : m_bhasLeadingPicture ? NAL_UNIT_CODED_SLICE_IDR_W_RADL : NAL_UNIT_CODED_SLICE_IDR_N_LP;
    if (m_pocCRA && curPOC < m_pocCRA)
        // All leading pictures are being marked as TFD pictures here since
        // current encoder uses all reference pictures while encoding leading
        // pictures. An encoder can ensure that a leading picture can be still
        // decodable when random accessing to a CRA/CRANT/BLA/BLANT picture by
        // controlling the reference pictures used for encoding that leading
        // picture. Such a leading picture need not be marked as a TFD picture.
        return NAL_UNIT_CODED_SLICE_RASL_R;

    if (m_lastIDR && curPOC < m_lastIDR)
        return NAL_UNIT_CODED_SLICE_RADL_R;

    return NAL_UNIT_CODED_SLICE_TRAIL_R;
}
