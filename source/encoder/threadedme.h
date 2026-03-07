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

#ifndef THREADED_ME_H
#define THREADED_ME_H

#include "common.h"
#include "threading.h"
#include "threadpool.h"
#include "cudata.h"
#include "lowres.h"
#include "frame.h"
#include "analysis.h"
#include "mv.h"

#include <queue>
#include <vector>
#include <fstream>

namespace X265_NS {

extern int g_puStartIdx[128][8];

class Encoder;
class Analysis;
class FrameEncoder;

struct PUBlock {
    uint32_t width;
    uint32_t height;
    PartSize partsize;
    bool isAmp;
};

const PUBlock g_puLookup[MAX_NUM_PU_SIZES] = {
    { 8,   4, SIZE_2NxN,  0 },
    { 4,   8, SIZE_Nx2N,  0 },
    { 8,   8, SIZE_2Nx2N, 0 },
    { 16,  4, SIZE_2NxnU, 1 },
    { 16, 12, SIZE_2NxnD, 1 },
    { 4,  16, SIZE_nLx2N, 1 },
    { 12, 16, SIZE_nRx2N, 1 },
    { 16,  8, SIZE_2NxN,  0 },
    { 8,  16, SIZE_Nx2N,  0 },
    { 16, 16, SIZE_2Nx2N, 0 },
    { 32,  8, SIZE_2NxnU, 1 },
    { 32, 24, SIZE_2NxnD, 1 },
    { 8,  32, SIZE_nLx2N, 1 },
    { 24, 32, SIZE_nRx2N, 1 },
    { 32, 16, SIZE_2NxN,  0 },
    { 16, 32, SIZE_Nx2N,  0 },
    { 32, 32, SIZE_2Nx2N, 0 },
    { 64, 16, SIZE_2NxnU, 1 },
    { 64, 48, SIZE_2NxnD, 1 },
    { 16, 64, SIZE_nLx2N, 1 },
    { 48, 64, SIZE_nRx2N, 1 },
    { 64, 32, SIZE_2NxN,  0 },
    { 32, 64, SIZE_Nx2N,  0 },
    { 64, 64, SIZE_2Nx2N, 0 }
};

struct CTUTaskData
{
    CUData& ctuData;
    CUGeom& ctuGeom;
    Frame& frame;
};

struct CTUBlockTask
{
    int row;
    int col;
    int width;
    int height;
    Frame* frame;
    class FrameEncoder* frameEnc;
    unsigned long long seq; /* monotonic sequence to preserve enqueue order */
};

struct PUData
{
    PartSize part;
    const CUGeom* cuGeom;
    int puOffset;
    int areaId;
    int finalIdx;
    int qp;
};

struct MEData
{
    MV       mv[2];
    MV       mvp[2];
    uint32_t mvCost[2];
    int      ref[2];
    int      bits;
    uint32_t cost;
};

struct CTUTask
{
    uint64_t seq;
    int row;
    int col;
    int width;
    int height;
    int layer;

    CUData* ctu;
    CUGeom* geom;
    Frame* frame;
    FrameEncoder* frameEnc;
};


struct CompareCTUTask {
    bool operator()(const CTUTask& a, const CTUTask& b) const {
        if (a.frame->m_poc == b.frame->m_poc)
        {
            int a_pos = a.row + a.col;
            int b_pos = b.row + b.col;
            if (a_pos != b_pos) return a_pos > b_pos;
        }

        /* Compare by sequence number to preserve FIFO enqueue order.
         * priority_queue in C++ is a max-heap, so return true when a.seq > b.seq
         * to make smaller seq (earlier enqueue) the top() element. */
        return a.seq > b.seq;
    }
};

/**
 * @brief Threaded motion-estimation module that schedules CTU blocks across worker threads.
 *
 * Owns per-worker analysis state (ThreadLocalData), manages the CTU task queues,
 * and exposes a JobProvider interface for the thread pool to execute MVP
 * derivation and ME searches in parallel.
 */
class ThreadedME: public JobProvider, public Thread
{
public:
    x265_param*             m_param;
    Encoder&                m_enc;

    std::priority_queue<CTUTask, std::vector<CTUTask>, CompareCTUTask>  m_taskQueue;
    Lock                    m_taskQueueLock;
    Event                   m_taskEvent;

    volatile bool           m_active;
    unsigned long long      m_enqueueSeq;

    ThreadLocalData*        m_tld;
    int                     m_tldCount;

#ifdef DETAILED_CU_STATS
    CUStats                 m_cuStats;
#endif

    /**
     * @brief Construct the ThreadedME manager; call create() before use.
     */
    ThreadedME(x265_param* param, Encoder& enc): m_param(param), m_enc(enc) {};
    
    /**
     * @brief Creates threadpool, thread local data and registers itself as a job provider
     */
    bool create();

    /**
     * @brief Initialize lookup table used to index PU offsets for all valid CTU sizes.
     */
    void initPuStartIdx();

    /**
     * @brief Enqueue a block of CTUs for motion estimation.
     *
     * Blocks are queued per FrameEncoder and later moved into the global
     * priority queue consumed by worker threads.
     */
    void enqueueCTUBlock(int row, int col, int width, int height, int layer, FrameEncoder* frameEnc);

    /**
     * @brief Inspect dependency state and enqueue newly-unblocked CTU rows.
     *
     * Uses external (row-level) and internal (buffered-row) dependencies to
     * decide when a row can be split into CTU block tasks.
     */
    void enqueueReadyRows(int row, int layer, FrameEncoder* frameEnc);

    /**
     * @brief Main dispatcher thread that transfers per-frame tasks into the global queue.
     */
    void threadMain();

    /**
     * @brief Dequeue a CTU task, derive MVs, and run ME over all supported PU shapes.
     *
     * Called by worker threads via JobProvider; processes an entire CTU block.
     */
    void findJob(int workerThreadId);

    /**
     * @brief Stops worker threads
     */
    void stopJobs();

    /**
     * @brief Cleanup allocated resources
     */
    void destroy();

    /**
     * @brief Accumulate detailed CU statistics from worker thread local data.
     */
    void collectStats();
};

// Utils

/**
 * @brief A workaround to init CTUs before processRowEncoder does the same,
 * since the CUData is needed before the FrameEncoder initializes it
 */
void initCTU(CUData& ctu, int row, int col, CTUTask& task);

};
    
#endif
