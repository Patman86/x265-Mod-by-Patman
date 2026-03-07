/*****************************************************************************
 * Copyright (C) 2013-2025 MulticoreWare, Inc
 *
 * Authors: Shashank Pathipati <shashank.pathipati@multicorewareinc.com>
 *          Somu Vineela <somu@mutlicorewareinc.com>
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

#include "threadedme.h"
#include "encoder.h"
#include "frameencoder.h"

#include <iostream>
#include <sstream>

namespace X265_NS {
int g_puStartIdx[128][8] = {0};

bool ThreadedME::create()
{
    m_active = true;
    m_tldCount = m_pool->m_numWorkers;
    m_tld = new ThreadLocalData[m_tldCount];
    for (int i = 0; i < m_tldCount; i++)
    {
        m_tld[i].analysis.initSearch(*m_param, m_enc.m_scalingList);
        m_tld[i].analysis.create(m_tld);
    }

    initPuStartIdx();

    /* start sequence at zero */
    m_enqueueSeq = 0ULL;

    return true;
}

void ThreadedME::initPuStartIdx()
{
    int startIdx = 0;
    uint32_t ctuSize = m_param->maxCUSize;

    for (uint32_t puIdx = 0; puIdx < MAX_NUM_PU_SIZES; ++puIdx)
    {
        const PUBlock& pu = g_puLookup[puIdx];

        if (pu.width > ctuSize || pu.height > ctuSize)
            continue;

        int indexWidth = pu.isAmp ? X265_MAX(pu.width, pu.height) : pu.width;
        int indexHeight = pu.isAmp ? indexWidth : pu.height;

        int numPUs = (ctuSize / indexWidth) * (ctuSize / indexHeight);
        int partIdx = static_cast<int>(pu.partsize);

        g_puStartIdx[pu.width + pu.height][partIdx] = startIdx;

        startIdx += pu.isAmp ? 2 * numPUs : numPUs;
    }
}

void ThreadedME::enqueueCTUBlock(int row, int col, int width, int height, int layer, FrameEncoder* frameEnc)
{
    frameEnc->m_tmeTasksLock.acquire();

    Frame* frame = frameEnc->m_frame[layer];

    CTUTask task;
    task.seq = ATOMIC_ADD(&m_enqueueSeq, 1ULL);
    task.row = row;
    task.col = col;
    task.width = width;
    task.height = height;
    task.layer = layer;

    task.frame = frame;
    task.frameEnc = frameEnc;

    frameEnc->m_tmeTasks.push(task);
    frameEnc->m_tmeTasksLock.release();

    m_taskEvent.trigger();
}

void ThreadedME::enqueueReadyRows(int row, int layer, FrameEncoder* frameEnc)
{
    int bufRow = X265_MIN(row + m_param->tmeNumBufferRows, static_cast<int>(frameEnc->m_numRows));

    for (int r = 0; r < bufRow; r++)
    {
        if (frameEnc->m_tmeDeps[r].isQueued)
            continue;

        bool isInitialRow = r < m_param->tmeNumBufferRows;
        bool isExternalDepResolved = frameEnc->m_tmeDeps[r].external;

        int prevRow = X265_MAX(0, r - m_param->tmeNumBufferRows);
        bool isInternalDepResolved = frameEnc->m_tmeDeps[prevRow].internal;

        if ((isInitialRow && isExternalDepResolved) ||
            (!isInitialRow && isExternalDepResolved && isInternalDepResolved))
        {
            int cols = static_cast<int>(frameEnc->m_numCols);
            for (int c = 0; c < cols; c += m_param->tmeTaskBlockSize)
            {
                int blockWidth = X265_MIN(m_param->tmeTaskBlockSize, cols - c);
                enqueueCTUBlock(r, c, blockWidth, 1, layer, frameEnc);
            }
            frameEnc->m_tmeDeps[r].isQueued = true;
        }
    }
}

void ThreadedME::threadMain()
{
    while (m_active)
    {
        int newCTUsPushed = 0;

        for (int i = 0; i < m_param->frameNumThreads; i++)
        {
            FrameEncoder* frameEnc = m_enc.m_frameEncoder[i];
            frameEnc->m_tmeTasksLock.acquire();

            while (!frameEnc->m_tmeTasks.empty())
            {
                CTUTask task = frameEnc->m_tmeTasks.front();
                frameEnc->m_tmeTasks.pop();

                m_taskQueueLock.acquire();
                m_taskQueue.push(task);
                m_taskQueueLock.release();

                newCTUsPushed++;
                tryWakeOne();
            }

            frameEnc->m_tmeTasksLock.release();
        }

        if (newCTUsPushed == 0)
            m_taskEvent.wait();
    }
}

void ThreadedME::findJob(int workerThreadId)
{
    m_taskQueueLock.acquire();
    if (m_taskQueue.empty())
    {
        m_helpWanted = false;
        m_taskQueueLock.release();
        return;
    }
    
    m_helpWanted = true;
    int64_t stime = x265_mdate();

#ifdef DETAILED_CU_STATS
    ScopedElapsedTime tmeTime(m_tld[workerThreadId].analysis.m_stats[m_jpId].tmeTime);
    m_tld[workerThreadId].analysis.m_stats[m_jpId].countTmeTasks++;
#endif

    CTUTask task = m_taskQueue.top();
    m_taskQueue.pop();
    m_taskQueueLock.release();

    int numCols = (m_param->sourceWidth + m_param->maxCUSize - 1) / m_param->maxCUSize;
    Frame* frame = task.frame;

    for (int i = 0; i < task.height; i++)
    {
        for (int j = 0; j < task.width; j++)
        {

            int ctuAddr = (task.row + i) * numCols + (task.col + j);
            CUData* ctu = frame->m_encData->getPicCTU(ctuAddr);
            ctu->m_slice = frame->m_encData->m_slice;

            task.ctu = ctu;
            task.geom = &task.frameEnc->m_cuGeoms[task.frameEnc->m_ctuGeomMap[ctuAddr]];

            frame->m_encData->m_cuStat[ctuAddr].baseQp = frame->m_encData->m_avgQpRc;
            initCTU(*ctu, task.row + i, task.col + j, task);

            task.frame->m_ctuMEFlags[ctuAddr].set(0);
            m_tld[workerThreadId].analysis.deriveMVsForCTU(*task.ctu, *task.geom, *frame);

            task.frame->m_ctuMEFlags[ctuAddr].set(1);
        }
    }

    if (m_param->csvLogLevel >= 2)
    {
        int64_t etime = x265_mdate();
        ATOMIC_ADD(&task.frameEnc->m_totalThreadedMETime[task.layer], etime - stime);
    }

    m_taskEvent.trigger();
}


void ThreadedME::stopJobs()
{
    this->m_active = false;
    m_taskEvent.trigger();
}

void ThreadedME::destroy()
{
    for (int i = 0; i < m_tldCount; i++)
        m_tld[i].destroy();
    delete[] m_tld;
}

void ThreadedME::collectStats()
{
#ifdef DETAILED_CU_STATS
    for (int i = 0; i < m_tldCount; i++)
        m_cuStats.accumulate(m_tld[i].analysis.m_stats[m_jpId], *m_param);
#endif
}

void initCTU(CUData& ctu, int row, int col, CTUTask& task)
{
    Frame& frame = *task.frame;
    FrameEncoder& frameEnc = *task.frameEnc;

    int numRows = frameEnc.m_numRows;
    int numCols = frameEnc.m_numCols;
    Slice *slice = frame.m_encData->m_slice;
    CTURow& ctuRow = frameEnc.m_rows[row];

    const uint32_t bFirstRowInSlice = ((row == 0) || (frameEnc.m_rows[row - 1].sliceId != ctuRow.sliceId)) ? 1 : 0;
    const uint32_t bLastRowInSlice = ((row == numRows - 1) || (frameEnc.m_rows[row + 1].sliceId != ctuRow.sliceId)) ? 1 : 0;

    const uint32_t bLastCuInSlice = (bLastRowInSlice & (col == numCols - 1)) ? 1 : 0;

    int ctuAddr = (numCols * row) + col;

    ctu.initCTU(frame, ctuAddr, slice->m_sliceQp, bFirstRowInSlice, bLastRowInSlice, bLastCuInSlice);
}

}