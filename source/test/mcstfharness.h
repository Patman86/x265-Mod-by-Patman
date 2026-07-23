/*****************************************************************************
 * Copyright (C) 2013-2021 MulticoreWare, Inc
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

#ifndef _MCSTFHARNESS_H_1
#define _MCSTFHARNESS_H_1 1

#include "testharness.h"
#include "temporalfilter.h"

/* MCSTFPrimitives is a standalone function-pointer table (not part of
 * EncoderPrimitives), and it only ever has one SIMD tier (AVX2), so this
 * harness does not fit the TestHarness(EncoderPrimitives) interface used by
 * the rest of the test bench. It is driven directly from testbench.cpp
 * instead of through the generic harness[] loop. */
class MCSTFHarness
{
protected:

    enum { TEST_STRIDE = 96 };
    enum { TEST_BUF_SIZE = TEST_STRIDE * TEST_STRIDE };
    enum { ITERS = 32 };

    ALIGN_VAR_32(pixel, m_origBuf[TEST_BUF_SIZE]);
    ALIGN_VAR_32(pixel, m_refBuf[TEST_BUF_SIZE]);
    ALIGN_VAR_32(pixel, m_dstBufRef[TEST_BUF_SIZE]);
    ALIGN_VAR_32(pixel, m_dstBufOpt[TEST_BUF_SIZE]);

    bool check_motionErrorLumaFrac(const MCSTFPrimitives& ref, const MCSTFPrimitives& opt);
    bool check_applyMotion(const MCSTFPrimitives& ref, const MCSTFPrimitives& opt);
    bool check_computeBlockStats(const MCSTFPrimitives& ref, const MCSTFPrimitives& opt);
    bool check_bilateralFilter(const MCSTFPrimitives& ref, const MCSTFPrimitives& opt);

public:

    MCSTFHarness();

    const char *getName() const { return "mcstf"; }

    bool testCorrectness(const MCSTFPrimitives& ref, const MCSTFPrimitives& opt);

    void measureSpeed(const MCSTFPrimitives& ref, const MCSTFPrimitives& opt);
};

#endif // ifndef _MCSTFHARNESS_H_1
