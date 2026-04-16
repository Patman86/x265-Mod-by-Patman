/*****************************************************************************
 * Copyright (C) 2013-2020 MulticoreWare, Inc
 *
 * Authors: Deepthi Nandakumar <deepthi@multicorewareinc.com>
 *          Min Chen <min.chen@multicorewareinc.com>
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
#include "slice.h"
#include "threading.h"
#include "param.h"
#include "cpu.h"
#include "x265.h"
#include "svt.h"
#include <locale.h>

#if _MSC_VER
#pragma warning(disable: 4996) // POSIX functions are just fine, thanks
#pragma warning(disable: 4706) // assignment within conditional
#pragma warning(disable: 4127) // conditional expression is constant
#endif

#if _WIN32
#define strcasecmp _stricmp
#endif

#if !defined(HAVE_STRTOK_R)

/*
 * adapted from public domain strtok_r() by Charlie Gordon
 *
 *   from comp.lang.c  9/14/2007
 *
 *      http://groups.google.com/group/comp.lang.c/msg/2ab1ecbb86646684
 *
 *     (Declaration that it's public domain):
 *      http://groups.google.com/group/comp.lang.c/msg/7c7b39328fefab9c
 */

#undef strtok_r
static char* strtok_r(char* str, const char* delim, char** nextp)
{
    if (!str)
        str = *nextp;

    str += strspn(str, delim);

    if (!*str)
        return NULL;

    char *ret = str;

    str += strcspn(str, delim);

    if (*str)
        *str++ = '\0';

    *nextp = str;

    return ret;
}

#endif // if !defined(HAVE_STRTOK_R)

#if EXPORT_C_API

/* these functions are exported as C functions (default) */
using namespace X265_NS;
extern "C" {

#else

/* these functions exist within private namespace (multilib) */
namespace X265_NS {

#endif

x265_param *x265_param_alloc()
{
    x265_param* param = (x265_param*)x265_malloc(sizeof(x265_param));
    memset(param, 0, sizeof(x265_param));
#ifdef SVT_HEVC
    param->svtHevcParam = (EB_H265_ENC_CONFIGURATION*)x265_malloc(sizeof(EB_H265_ENC_CONFIGURATION));
#endif
    return param;
}

void x265_param_free(x265_param* p)
{
    x265_zone_free(p);
#ifdef SVT_HEVC
     x265_free(p->svtHevcParam);
#endif
    x265_free(p);
}

#if ENABLE_SCC_EXT
enum SCCProfileName
{
    NONE = 0,
    // The following are SCC profiles, which would map to the MAINSCC profile idc.
    // The enumeration indicates the bit-depth constraint in the bottom 2 digits
    //                           the chroma format in the next digit
    //                           the intra constraint in the next digit
    //                           If it is a SCC profile there is a '2' for the next digit.
    //                           If it is a highthroughput , there is a '2' for the top digit else '1' for the top digit
    SCC_MAIN = 121108,
    SCC_MAIN_10 = 121110,
    SCC_MAIN_444 = 121308,
    SCC_MAIN_444_10 = 121310,
};

static const SCCProfileName validSCCProfileNames[1][4/* bit depth constraint 8=0, 10=1, 12=2, 14=3*/][4/*chroma format*/] =
{
   {
        { NONE,         SCC_MAIN,      NONE,      SCC_MAIN_444                     }, // 8-bit  intra for 400, 420, 422 and 444
        { NONE,         SCC_MAIN_10,   NONE,      SCC_MAIN_444_10                  }, // 10-bit intra for 400, 420, 422 and 444
        { NONE,         NONE,          NONE,      NONE                             }, // 12-bit intra for 400, 420, 422 and 444
        { NONE,         NONE,          NONE,      NONE                             }  // 16-bit intra for 400, 420, 422 and 444
    },
};
#endif

void x265_param_default(x265_param* param)
{
#ifdef SVT_HEVC
    EB_H265_ENC_CONFIGURATION* svtParam = (EB_H265_ENC_CONFIGURATION*)param->svtHevcParam;
#endif

    memset(param, 0, sizeof(x265_param));

    /* Applying default values to all elements in the param structure */
    param->cpuid = X265_NS::cpu_detect(false);
    param->bEnableWavefront = 1;
    param->frameNumThreads = 0;

    param->logLevel = X265_LOG_INFO;
    param->logfn[0] = 0;
    param->logfLevel = X265_LOG_INFO;
    param->csvLogLevel = 0;
    param->csvfn[0] = 0;
    param->rc.lambdaFileName[0] = 0;
    param->bLogCuStats = 0;
    param->decodedPictureHashSEI = 0;

    /* Quality Measurement Metrics */
    param->bEnablePsnr = 0;
    param->bEnableSsim = 0;

    /* Source specifications */
    param->internalBitDepth = X265_DEPTH;
    param->sourceBitDepth = 8;
    param->internalCsp = X265_CSP_I420;
    param->levelIdc = 0; //Auto-detect level
    param->uhdBluray = 0;
    param->bHighTier = 1; //Allow high tier by default
    param->interlaceMode = 0;
    param->bField = 0;
    param->bAnnexB = 1;
    param->bRepeatHeaders = 0;
    param->bEnableAccessUnitDelimiters = 0;
    param->bEnableEndOfBitstream = 0;
    param->bEnableEndOfSequence = 0;
    param->bEmitHRDSEI = 0;
    param->bEmitInfoSEI = 1;
    param->bEmitHDRSEI = 0; /*Deprecated*/
    param->bEmitHDR10SEI = 0;
    param->bEmitIDRRecoverySEI = 0;

    /* CU definitions */
    param->maxCUSize = 64;
    param->minCUSize = 8;
    param->tuQTMaxInterDepth = 1;
    param->tuQTMaxIntraDepth = 1;
    param->maxTUSize = 32;

    /* Coding Structure */
    param->keyframeMin = 0;
    param->keyframeMax = 250;
    param->gopLookahead = 0;
    param->bOpenGOP = 1;
    param->craNal = 0;
    param->bframes = 4;
    param->lookaheadDepth = 20;
    param->bFrameAdaptive = X265_B_ADAPT_TRELLIS;
    param->bBPyramid = 1;
    param->scenecutThreshold = 40; /* Magic number pulled in from x264 */
    param->bHistBasedSceneCut = 0;
    param->lookaheadSlices = 8;
    param->lookaheadThreads = 0;
    param->scenecutBias = 5.0;
    param->radl = 0;
    param->chunkStart = 0;
    param->chunkEnd = 0;
    param->bEnableHRDConcatFlag = 0;
    param->bEnableFades = 0;
    param->bEnableSceneCutAwareQp = 0;
    param->fwdMaxScenecutWindow = 1200;
    param->bwdMaxScenecutWindow = 600;
    param->mcstfFrameRange = 2;
    for (int i = 0; i < 6; i++)
    {
        int deltas[6] = { 5, 4, 3, 2, 1, 0 };

        param->fwdScenecutWindow[i] = 200;
        param->fwdRefQpDelta[i] = deltas[i];
        param->fwdNonRefQpDelta[i] = param->fwdRefQpDelta[i] + (SLICE_TYPE_DELTA * param->fwdRefQpDelta[i]);

        param->bwdScenecutWindow[i] = 100;
        param->bwdRefQpDelta[i] = -1;
        param->bwdNonRefQpDelta[i] = -1;
    }

    /* Intra Coding Tools */
    param->bEnableConstrainedIntra = 0;
    param->bEnableStrongIntraSmoothing = 1;
    param->bEnableFastIntra = 0;
    param->bEnableSplitRdSkip = 0;

    /* Inter Coding tools */
    param->searchMethod = X265_HEX_SEARCH;
    param->subpelRefine = 2;
    param->searchRange = 57;
    param->maxNumMergeCand = 3;
    param->limitReferences = 1;
    param->limitModes = 0;
    param->bEnableWeightedPred = 1;
    param->bEnableWeightedBiPred = 0;
    param->bEnableEarlySkip = 1;
    param->recursionSkipMode = 1;
    param->edgeVarThreshold = 0.05f;
    param->bEnableAMP = 0;
    param->bEnableRectInter = 0;
    param->rdLevel = 3;
    param->rdoqLevel = 0;
    param->bEnableSignHiding = 1;
    param->bEnableTransformSkip = 0;
    param->bEnableTSkipFast = 0;
    param->maxNumReferences = 3;
    param->bEnableTemporalMvp = 1;
    param->bEnableHME = 0;
    param->hmeSearchMethod[0] = X265_HEX_SEARCH;
    param->hmeSearchMethod[1] = param->hmeSearchMethod[2] = X265_UMH_SEARCH;
    param->hmeRange[0] = 16;
    param->hmeRange[1] = 32;
    param->hmeRange[2] = 48;
    param->bSourceReferenceEstimation = 0;
    param->limitTU = 0;
    param->dynamicRd = 0;

    /* Loop Filter */
    param->bEnableLoopFilter = 1;

    /* SAO Loop Filter */
    param->bEnableSAO = 1;
    param->bSaoNonDeblocked = 0;
    param->bLimitSAO = 0;
    param->selectiveSAO = 0;

    /* Coding Quality */
    param->cbQpOffset = 0;
    param->crQpOffset = 0;
    param->rdPenalty = 0;
    param->psyRd = 2.0;
    param->psyRdoq = 0.0;
    param->analysisReuseMode = 0; /*DEPRECATED*/
    param->analysisMultiPassRefine = 0;
    param->analysisMultiPassDistortion = 0;
    param->analysisReuseFileName[0] = 0;
    param->analysisSave[0] = 0;
    param->analysisLoad[0] = 0;
    param->bIntraInBFrames = 1;
    param->bLossless = 0;
    param->bCULossless = 0;
    param->bEnableTemporalSubLayers = 0;
    param->bEnableRdRefine = 0;
    param->bIntraRDRefine = 0;
    param->bMultiPassOptRPS = 0;
    param->bSsimRd = 0;

    /* Rate control options */
    param->rc.vbvMaxBitrate = 0;
    param->rc.vbvBufferSize = 0;
    param->rc.vbvBufferInit = 0.9;
    param->vbvBufferEnd = 0;
    param->vbvEndFrameAdjust = 0;
    param->minVbvFullness = 50;
    param->maxVbvFullness = 80;
    param->rc.rfConstant = 28;
    param->rc.bitrate = 0;
    param->rc.qCompress = 0.6;
    param->rc.ipFactor = 1.4f;
    param->rc.pbFactor = 1.3f;
    param->rc.qpStep = 4;
    param->rc.rateControlMode = X265_RC_CRF;
    param->rc.qp = 32;
    param->rc.aqMode = X265_AQ_AUTO_VARIANCE;
    param->rc.hevcAq = 0;
    param->rc.qgSize = 32;
    param->rc.aqStrength = 1.0;
    param->rc.aqBiasStrength = 1.0;
    param->rc.qpAdaptationRange = 1.0;
    param->rc.cuTree = 1;
    param->rc.rfConstantMax = 0;
    param->rc.rfConstantMin = 0;
    param->rc.bStatRead = 0;
    param->rc.bStatWrite = 0;
    param->rc.dataShareMode = X265_SHARE_MODE_FILE;
    param->rc.statFileName[0] = 0;
    param->rc.sharedMemName[0] = 0;
    param->rc.bEncFocusedFramesOnly = 0;
    param->rc.complexityBlur = 20;
    param->rc.qblur = 0.5;
    param->rc.zoneCount = 0;
    param->rc.zonefileCount = 0;
    param->rc.zones = NULL;
    param->rc.bEnableSlowFirstPass = 1;
    param->rc.bStrictCbr = 0;
    param->rc.bEnableGrain = 0;
    param->rc.qpMin = 0;
    param->rc.qpMax = QP_MAX_MAX;
    param->rc.bEnableConstVbv = 0;
    param->bResetZoneConfig = 1;
    param->reconfigWindowSize = 0;
    param->rc.bAutoAq = 0;
    param->decoderVbvMaxRate = 0;
    param->bliveVBV2pass = 0;

    /* Video Usability Information (VUI) */
    param->vui.aspectRatioIdc = 0;
    param->vui.sarWidth = 0;
    param->vui.sarHeight = 0;
    param->vui.bEnableOverscanAppropriateFlag = 0;
    param->vui.bEnableVideoSignalTypePresentFlag = 0;
    param->vui.videoFormat = 5;
    param->vui.bEnableVideoFullRangeFlag = 0;
    param->vui.bEnableColorDescriptionPresentFlag = 0;
    param->vui.colorPrimaries = 2;
    param->vui.transferCharacteristics = 2;
    param->vui.matrixCoeffs = 2;
    param->vui.bEnableChromaLocInfoPresentFlag = 0;
    param->vui.chromaSampleLocTypeTopField = 0;
    param->vui.chromaSampleLocTypeBottomField = 0;
    param->vui.bEnableDefaultDisplayWindowFlag = 0;
    param->vui.defDispWinLeftOffset = 0;
    param->vui.defDispWinRightOffset = 0;
    param->vui.defDispWinTopOffset = 0;
    param->vui.defDispWinBottomOffset = 0;
    param->maxCLL = 0;
    param->maxFALL = 0;
    param->minLuma = 0;
    param->maxLuma = PIXEL_MAX;
    param->log2MaxPocLsb = 8;
    param->maxSlices = 1;
    param->videoSignalTypePreset[0] = 0;

    /*Conformance window*/
    param->confWinRightOffset = 0;
    param->confWinBottomOffset = 0;

    param->bEmitVUITimingInfo   = 1;
    param->bEmitVUIHRDInfo      = 1;
    param->bOptQpPPS            = 0;
    param->bOptRefListLengthPPS = 0;
    param->bOptCUDeltaQP        = 0;
    param->bAQMotion = 0;
    param->bHDROpt = 0; /*DEPRECATED*/
    param->bHDR10Opt = 0;
    param->analysisReuseLevel = 0;  /*DEPRECATED*/
    param->analysisSaveReuseLevel = 0;
    param->analysisLoadReuseLevel = 0;
    param->toneMapFile[0] = 0;
    param->bDhdr10opt = 0;
    param->dolbyProfile = 0;
    param->bCTUInfo = 0;
    param->bUseRcStats = 0;
    param->scaleFactor = 0;
    param->intraRefine = 0;
    param->interRefine = 0;
    param->bDynamicRefine = 0;
    param->mvRefine = 1;
    param->ctuDistortionRefine = 0;
    param->bUseAnalysisFile = 1;
    param->csvfpt = NULL;
    param->forceFlush = 0;
    param->bDisableLookahead = 0;
    param->bCopyPicToFrame = 1;
    param->maxAUSizeFactor = 1;
    param->naluFile[0] = 0;

    /* DCT Approximations */
    param->bLowPassDct = 0;
    param->bAnalysisType = 0;
    param->bSingleSeiNal = 0;

    /* SEI messages */
    param->preferredTransferCharacteristics = -1;
    param->pictureStructure = -1;
    param->bEmitCLL = 1;

    param->bEnableFrameDuplication = 0;
    param->dupThreshold = 70;

    /* SVT Hevc Encoder specific params */
    param->bEnableSvtHevc = 0;
    param->svtHevcParam = NULL;

    /* MCSTF */
    param->bEnableTemporalFilter = 0;
    param->temporalFilterStrength = 0.95;
    param->searchRangeForLayer0 = 3;
    param->searchRangeForLayer1 = 3;
    param->searchRangeForLayer2 = 3;

    /* Threaded ME */
    param->tmeTaskBlockSize = 1;
    param->tmeNumBufferRows = 10;

    /*Alpha Channel Encoding*/
    param->bEnableAlpha = 0;
    param->numScalableLayers = 1;

#ifdef SVT_HEVC
    param->svtHevcParam = svtParam;
    svt_param_default(param);
#endif
    /* Film grain characteristics model filename */
    param->filmGrain = NULL;
    param->aomFilmGrain = NULL;
    param->bEnableSBRC = 0;

    /* Multi-View Encoding*/
    param->numViews = 1;
    param->format = 0;

    param->numLayers = 1;

    /* SCC */
    param->bEnableSCC = 0;

    param->bConfigRCFrame = 0;
}

int x265_param_default_preset(x265_param* param, const char* preset, const char* tune)
{
#if EXPORT_C_API
    ::x265_param_default(param);
#else
    X265_NS::x265_param_default(param);
#endif

    if (preset)
    {
        char *end;
        int i = strtol(preset, &end, 10);
        if (*end == 0 && i >= 0 && i < (int)(sizeof(x265_preset_names) / sizeof(*x265_preset_names) - 1))
            preset = x265_preset_names[i];

        if (!strcmp(preset, "ultrafast"))
        {
            param->mcstfFrameRange = 1;
            param->maxNumMergeCand = 2;
            param->bIntraInBFrames = 0;
            param->lookaheadDepth = 5;
            param->scenecutThreshold = 0; // disable lookahead
            param->maxCUSize = 32;
            param->minCUSize = 16;
            param->bframes = 3;
            param->bFrameAdaptive = 0;
            param->subpelRefine = 0;
            param->searchMethod = X265_DIA_SEARCH;
            param->bEnableSAO = 0;
            param->bEnableSignHiding = 0;
            param->bEnableWeightedPred = 0;
            param->rdLevel = 2;
            param->maxNumReferences = 1;
            param->limitReferences = 0;
            param->rc.aqStrength = 0.0;
            param->rc.aqMode = X265_AQ_NONE;
            param->rc.hevcAq = 0;
            param->rc.qgSize = 32;
            param->bEnableFastIntra = 1;
        }
        else if (!strcmp(preset, "superfast"))
        {
            param->mcstfFrameRange = 1;
            param->maxNumMergeCand = 2;
            param->bIntraInBFrames = 0;
            param->lookaheadDepth = 10;
            param->maxCUSize = 32;
            param->bframes = 3;
            param->bFrameAdaptive = 0;
            param->subpelRefine = 1;
            param->bEnableWeightedPred = 0;
            param->rdLevel = 2;
            param->maxNumReferences = 1;
            param->limitReferences = 0;
            param->rc.aqStrength = 0.0;
            param->rc.aqMode = X265_AQ_NONE;
            param->rc.hevcAq = 0;
            param->rc.qgSize = 32;
            param->bEnableSAO = 0;
            param->bEnableFastIntra = 1;
        }
        else if (!strcmp(preset, "veryfast"))
        {
            param->mcstfFrameRange = 1;
            param->maxNumMergeCand = 2;
            param->limitReferences = 3;
            param->bIntraInBFrames = 0;
            param->lookaheadDepth = 15;
            param->bFrameAdaptive = 0;
            param->subpelRefine = 1;
            param->rdLevel = 2;
            param->maxNumReferences = 2;
            param->rc.qgSize = 32;
            param->bEnableFastIntra = 1;
        }
        else if (!strcmp(preset, "faster"))
        {
            param->mcstfFrameRange = 1;
            param->maxNumMergeCand = 2;
            param->limitReferences = 3;
            param->bIntraInBFrames = 0;
            param->lookaheadDepth = 15;
            param->bFrameAdaptive = 0;
            param->rdLevel = 2;
            param->maxNumReferences = 2;
            param->bEnableFastIntra = 1;
        }
        else if (!strcmp(preset, "fast"))
        {
            param->mcstfFrameRange = 1;
            param->maxNumMergeCand = 2;
            param->limitReferences = 3;
            param->bEnableEarlySkip = 0;
            param->bIntraInBFrames = 0;
            param->lookaheadDepth = 15;
            param->bFrameAdaptive = 0;
            param->rdLevel = 2;
            param->maxNumReferences = 3;
            param->bEnableFastIntra = 1;
        }
        else if (!strcmp(preset, "medium"))
        {
            param->mcstfFrameRange = 1;
            /* defaults */
        }
        else if (!strcmp(preset, "slow"))
        {
            param->limitReferences = 3;
            param->bEnableEarlySkip = 0;
            param->bIntraInBFrames = 0;
            param->bEnableRectInter = 1;
            param->lookaheadDepth = 25;
            param->rdLevel = 4;
            param->rdoqLevel = 2;
            param->psyRdoq = 1.0;
            param->subpelRefine = 3;
            param->searchMethod = X265_STAR_SEARCH;
            param->maxNumReferences = 4;
            param->limitModes = 1;
            param->lookaheadSlices = 4; // limit parallelism as already enough work exists
        }
        else if (!strcmp(preset, "slower"))
        {
            param->bEnableEarlySkip = 0;
            param->bEnableWeightedBiPred = 1;
            param->bEnableAMP = 1;
            param->bEnableRectInter = 1;
            param->lookaheadDepth = 40;
            param->bframes = 8;
            param->tuQTMaxInterDepth = 3;
            param->tuQTMaxIntraDepth = 3;
            param->rdLevel = 6;
            param->rdoqLevel = 2;
            param->psyRdoq = 1.0;
            param->subpelRefine = 4;
            param->maxNumMergeCand = 4;
            param->searchMethod = X265_STAR_SEARCH;
            param->maxNumReferences = 5;
            param->limitModes = 1;
            param->lookaheadSlices = 0; // disabled for best quality
            param->limitTU = 4;
        }
        else if (!strcmp(preset, "veryslow"))
        {
            param->bEnableEarlySkip = 0;
            param->bEnableWeightedBiPred = 1;
            param->bEnableAMP = 1;
            param->bEnableRectInter = 1;
            param->lookaheadDepth = 40;
            param->bframes = 8;
            param->tuQTMaxInterDepth = 3;
            param->tuQTMaxIntraDepth = 3;
            param->rdLevel = 6;
            param->rdoqLevel = 2;
            param->psyRdoq = 1.0;
            param->subpelRefine = 4;
            param->maxNumMergeCand = 5;
            param->searchMethod = X265_STAR_SEARCH;
            param->maxNumReferences = 5;
            param->limitReferences = 0;
            param->limitModes = 0;
            param->lookaheadSlices = 0; // disabled for best quality
            param->limitTU = 0;
        }
        else if (!strcmp(preset, "placebo"))
        {
            param->bEnableEarlySkip = 0;
            param->bEnableWeightedBiPred = 1;
            param->bEnableAMP = 1;
            param->bEnableRectInter = 1;
            param->lookaheadDepth = 60;
            param->searchRange = 92;
            param->bframes = 8;
            param->tuQTMaxInterDepth = 4;
            param->tuQTMaxIntraDepth = 4;
            param->rdLevel = 6;
            param->rdoqLevel = 2;
            param->psyRdoq = 1.0;
            param->subpelRefine = 5;
            param->maxNumMergeCand = 5;
            param->searchMethod = X265_STAR_SEARCH;
            param->bEnableTransformSkip = 1;
            param->recursionSkipMode = 0;
            param->maxNumReferences = 5;
            param->limitReferences = 0;
            param->lookaheadSlices = 0; // disabled for best quality
            // TODO: optimized esa
        }
        else
            return -1;
    }
    if (tune)
    {
        param->tune = tune;
        if (!strcmp(tune, "psnr"))
        {
            param->rc.aqStrength = 0.0;
            param->psyRd = 0.0;
            param->psyRdoq = 0.0;
        }
        else if (!strcmp(tune, "ssim"))
        {
            param->rc.aqMode = X265_AQ_AUTO_VARIANCE;
            param->psyRd = 0.0;
            param->psyRdoq = 0.0;
        }
        else if (!strcmp(tune, "fastdecode") ||
                 !strcmp(tune, "fast-decode"))
        {
            param->bEnableLoopFilter = 0;
            param->bEnableSAO = 0;
            param->bEnableWeightedPred = 0;
            param->bEnableWeightedBiPred = 0;
            param->bIntraInBFrames = 0;
        }
        else if (!strcmp(tune, "zerolatency") ||
                 !strcmp(tune, "zero-latency"))
        {
            param->bFrameAdaptive = 0;
            param->bframes = 0;
            param->lookaheadDepth = 0;
            param->scenecutThreshold = 0;
            param->bHistBasedSceneCut = 0;
            param->rc.cuTree = 0;
            param->frameNumThreads = 1;
        }
        else if (!strcmp(tune, "grain"))
        {
            param->rc.ipFactor = 1.1;
            param->rc.pbFactor = 1.0;
            param->rc.cuTree = 0;
            param->rc.aqMode = 0;
            param->rc.hevcAq = 0;
            param->rc.qpStep = 1;
            param->rc.bEnableGrain = 1;
            param->recursionSkipMode = 0;
            param->psyRd = 4.0;
            param->psyRdoq = 10.0;
            param->bEnableSAO = 0;
            param->rc.bEnableConstVbv = 1;
        }
        else if (!strcmp(tune, "animation"))
        {
            param->bframes = (param->bframes + 2) >= param->lookaheadDepth? param->bframes : param->bframes + 2;
            param->psyRd = 0.4;
            param->rc.aqStrength = 0.4;
            param->deblockingFilterBetaOffset = 1;
            param->deblockingFilterTCOffset = 1;
        }
        else if (!strcmp(tune, "vmaf"))  /*Adding vmaf for x265 + SVT-HEVC integration support*/
        {
            /*vmaf is under development, currently x265 won't support vmaf*/
        }
        else
            return -1;
    }

#ifdef SVT_HEVC
    if (svt_set_preset(param, preset))
        return -1;
#endif

    return 0;
}

static int x265_atobool(const char* str, bool& bError)
{
    if (!strcmp(str, "1") ||
        !strcmp(str, "true") ||
        !strcmp(str, "yes"))
        return 1;
    if (!strcmp(str, "0") ||
        !strcmp(str, "false") ||
        !strcmp(str, "no"))
        return 0;
    bError = true;
    return 0;
}

static int parseName(const char* arg, const char* const* names, bool& bError)
{
    for (int i = 0; names[i]; i++)
        if (!strcmp(arg, names[i]))
            return i;

    return x265_atoi(arg, bError);
}
/* internal versions of string-to-int with additional error checking */
#undef atoi
#undef atof
#define atoi(str) x265_atoi(str, bError)
#define atof(str) x265_atof(str, bError)
#define atobool(str) (x265_atobool(str, bError))

int x265_scenecut_aware_qp_param_parse(x265_param* p, const char* name, const char* value)
{
    bool bError = false;
    char nameBuf[64];
    if (!name)
        return X265_PARAM_BAD_NAME;
    // skip -- prefix if provided
    if (name[0] == '-' && name[1] == '-')
        name += 2;
    // s/_/-/g
    if (strlen(name) + 1 < sizeof(nameBuf) && strchr(name, '_'))
    {
        char *c;
        strcpy(nameBuf, name);
        while ((c = strchr(nameBuf, '_')) != 0)
            *c = '-';
        name = nameBuf;
    }
    if (!value)
        value = "true";
    else if (value[0] == '=')
        value++;
#define OPT(STR) else if (!strcmp(name, STR))
    if (0);
    OPT("scenecut-aware-qp") p->bEnableSceneCutAwareQp = x265_atoi(value, bError);
    OPT("masking-strength") bError = parseMaskingStrength(p, value);
    else
        return X265_PARAM_BAD_NAME;
#undef OPT
    return bError ? X265_PARAM_BAD_VALUE : 0;
}


/* internal versions of string-to-int with additional error checking */
#undef atoi
#undef atof
#define atoi(str) x265_atoi(str, bError)
#define atof(str) x265_atof(str, bError)
#define atobool(str) (x265_atobool(str, bError))

int x265_zone_param_parse(x265_param* p, const char* name, const char* value)
{
    bool bError = false;
    char nameBuf[64];

    if (!name)
        return X265_PARAM_BAD_NAME;

    // skip -- prefix if provided
    if (name[0] == '-' && name[1] == '-')
        name += 2;

    // s/_/-/g
    if (strlen(name) + 1 < sizeof(nameBuf) && strchr(name, '_'))
    {
        char *c;
        strcpy(nameBuf, name);
        while ((c = strchr(nameBuf, '_')) != 0)
            *c = '-';

        name = nameBuf;
    }

    if (!strncmp(name, "no-", 3))
    {
        name += 3;
        value = !value || x265_atobool(value, bError) ? "false" : "true";
    }
    else if (!strncmp(name, "no", 2))
    {
        name += 2;
        value = !value || x265_atobool(value, bError) ? "false" : "true";
    }
    else if (!value)
        value = "true";
    else if (value[0] == '=')
        value++;

#define OPT(STR) else if (!strcmp(name, STR))
#define OPT2(STR1, STR2) else if (!strcmp(name, STR1) || !strcmp(name, STR2))

    if (0);
    OPT("ref") p->maxNumReferences = atoi(value);
    OPT("fast-intra") p->bEnableFastIntra = atobool(value);
    OPT("early-skip") p->bEnableEarlySkip = atobool(value);
    OPT("rskip") p->recursionSkipMode = atoi(value);
    OPT("rskip-edge-threshold") p->edgeVarThreshold = atoi(value)/100.0f;
    OPT("me") p->searchMethod = parseName(value, x265_motion_est_names, bError);
    OPT("subme") p->subpelRefine = atoi(value);
    OPT("merange") p->searchRange = atoi(value);
    OPT("rect") p->bEnableRectInter = atobool(value);
    OPT("amp") p->bEnableAMP = atobool(value);
    OPT("max-merge") p->maxNumMergeCand = (uint32_t)atoi(value);
    OPT("rd") p->rdLevel = atoi(value);
    OPT("radl") p->radl = atoi(value);
    OPT2("rdoq", "rdoq-level")
    {
        int bval = atobool(value);
        if (bError || bval)
        {
            bError = false;
            p->rdoqLevel = atoi(value);
        }
        else
            p->rdoqLevel = 0;
    }
    OPT("b-intra") p->bIntraInBFrames = atobool(value);
    OPT("scaling-list") snprintf(p->scalingLists, X265_MAX_STRING_SIZE, "%s", value);
    OPT("crf")
    {
        p->rc.rfConstant = atof(value);
        p->rc.rateControlMode = X265_RC_CRF;
    }
    OPT("qp")
    {
        p->rc.qp = atoi(value);
        p->rc.rateControlMode = X265_RC_CQP;
    }
    OPT("bitrate")
    {
        p->rc.bitrate = atoi(value);
        p->rc.rateControlMode = X265_RC_ABR;
    }
    OPT("aq-mode") p->rc.aqMode = atoi(value);
    OPT("aq-strength") p->rc.aqStrength = atof(value);
    OPT("aq-bias-strength") p->rc.aqBiasStrength = atof(value);
    OPT("nr-intra") p->noiseReductionIntra = atoi(value);
    OPT("nr-inter") p->noiseReductionInter = atoi(value);
    OPT("limit-modes") p->limitModes = atobool(value);
    OPT("splitrd-skip") p->bEnableSplitRdSkip = atobool(value);
    OPT("cu-lossless") p->bCULossless = atobool(value);
    OPT("rd-refine") p->bEnableRdRefine = atobool(value);
    OPT("limit-tu") p->limitTU = atoi(value);
    OPT("tskip") p->bEnableTransformSkip = atobool(value);
    OPT("tskip-fast") p->bEnableTSkipFast = atobool(value);
    OPT("rdpenalty") p->rdPenalty = atoi(value);
    OPT("dynamic-rd") p->dynamicRd = atof(value);
    else
        return X265_PARAM_BAD_NAME;

#undef OPT
#undef OPT2

    return bError ? X265_PARAM_BAD_VALUE : 0;
}

#undef atobool
#undef atoi
#undef atof

/* internal versions of string-to-int with additional error checking */
#undef atoi
#undef atof
#define atoi(str) x265_atoi(str, bError)
#define atof(str) x265_atof(str, bError)
#define atobool(str) (bNameWasBool = true, x265_atobool(str, bError))


struct ParseContext
{
    x265_param* p;
    const char* name;
    const char* value;
    bool bError;
    bool bNameWasBool;
    bool bValueWasNull;
    char nameBuf[64];
};

static int normalizeParseArgs(ParseContext& ctx)
{
    if (!ctx.name)
        return X265_PARAM_BAD_NAME;

    if (ctx.name[0] == '-' && ctx.name[1] == '-')
        ctx.name += 2;

    if (strlen(ctx.name) + 1 < sizeof(ctx.nameBuf) && strchr(ctx.name, '_'))
    {
        char* c;
        strcpy(ctx.nameBuf, ctx.name);
        while ((c = strchr(ctx.nameBuf, '_')) != 0)
            *c = '-';
        ctx.name = ctx.nameBuf;
    }

    if (!strncmp(ctx.name, "no-", 3))
    {
        ctx.name += 3;
        ctx.value = !ctx.value || x265_atobool(ctx.value, ctx.bError) ? "false" : "true";
    }
    else if (!strncmp(ctx.name, "no", 2))
    {
        ctx.name += 2;
        ctx.value = !ctx.value || x265_atobool(ctx.value, ctx.bError) ? "false" : "true";
    }
    else if (!ctx.value)
        ctx.value = "true";
    else if (ctx.value[0] == '=')
        ctx.value++;

    return 0;
}

static int parseLogLevelValue(ParseContext& ctx, int& dst)
{
    dst = x265_atoi(ctx.value, ctx.bError);
    if (ctx.bError)
    {
        ctx.bError = false;
        dst = parseName(ctx.value, logLevelNames, ctx.bError) - 1;
    }
    return ctx.bError ? X265_PARAM_BAD_VALUE : 0;
}

static int parseRdoqValue(ParseContext& ctx, int& dst)
{
    int bval = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    if (ctx.bError || bval)
    {
        ctx.bError = false;
        dst = x265_atoi(ctx.value, ctx.bError);
    }
    else
        dst = 0;
    return ctx.bError ? X265_PARAM_BAD_VALUE : 0;
}

static int parsePsyValue(ParseContext& ctx, double& dst)
{
    int bval = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    if (ctx.bError || bval)
    {
        ctx.bError = false;
        dst = x265_atof(ctx.value, ctx.bError);
    }
    else
        dst = 0.0;
    return ctx.bError ? X265_PARAM_BAD_VALUE : 0;
}

static int parseLoggingStatsOpts(ParseContext& ctx)
{
#define OPT(STR) else if (!strcmp(ctx.name, STR))
    if (0) ;
    OPT("log-level") return parseLogLevelValue(ctx, ctx.p->logLevel);
    OPT("log") return parseLogLevelValue(ctx, ctx.p->logLevel);
    OPT("cu-stats") ctx.p->bLogCuStats = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("log-file") snprintf(ctx.p->logfn, X265_MAX_STRING_SIZE, "%s", ctx.value);
    OPT("log-file-level") return parseLogLevelValue(ctx, ctx.p->logfLevel);
    OPT("csv") snprintf(ctx.p->csvfn, X265_MAX_STRING_SIZE, "%s", ctx.value);
    OPT("csv-log-level") ctx.p->csvLogLevel = x265_atoi(ctx.value, ctx.bError);
    OPT("ssim") ctx.p->bEnableSsim = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("psnr") ctx.p->bEnablePsnr = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("hash") ctx.p->decodedPictureHashSEI = x265_atoi(ctx.value, ctx.bError);
    else return X265_PARAM_BAD_NAME;
#undef OPT
    return ctx.bError ? X265_PARAM_BAD_VALUE : 0;
}

static int parseInputOutputOpts(ParseContext& ctx)
{
#define OPT(STR) else if (!strcmp(ctx.name, STR))
#define OPT2(A,B) else if (!strcmp(ctx.name, A) || !strcmp(ctx.name, B))
    if (0) ;
    OPT("fps")
    {
        if (sscanf(ctx.value, "%u/%u", &ctx.p->fpsNum, &ctx.p->fpsDenom) != 2)
        {
            float fps = (float)x265_atof(ctx.value, ctx.bError);
            if (!ctx.bError && fps > 0 && fps < INT_MAX / 1000)
            {
                ctx.p->fpsNum = (int)(fps * 1000 + .5);
                ctx.p->fpsDenom = 1000;
            }
            else
            {
                ctx.bError = false;
                ctx.p->fpsNum = x265_atoi(ctx.value, ctx.bError);
                ctx.p->fpsDenom = 1;
            }
        }
    }
    OPT("total-frames") ctx.p->totalFrames = x265_atoi(ctx.value, ctx.bError);
    OPT("input-res") ctx.bError = sscanf(ctx.value, "%dx%d", &ctx.p->sourceWidth, &ctx.p->sourceHeight) != 2;
    OPT("input-csp") ctx.p->internalCsp = parseName(ctx.value, x265_source_csp_names, ctx.bError);
    OPT("interlace")
    {
        ctx.bNameWasBool = true;
        ctx.p->interlaceMode = x265_atobool(ctx.value, ctx.bError);
        if (ctx.bError)
        {
            ctx.bError = false;
            ctx.p->interlaceMode = parseName(ctx.value, x265_interlace_names, ctx.bError);
        }
    }
    OPT("field") ctx.p->bField = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("chunk-start") ctx.p->chunkStart = x265_atoi(ctx.value, ctx.bError);
    OPT("chunk-end") ctx.p->chunkEnd = x265_atoi(ctx.value, ctx.bError);
    OPT("frame-dup") ctx.p->bEnableFrameDuplication = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("dup-threshold") ctx.p->dupThreshold = x265_atoi(ctx.value, ctx.bError);
    else return X265_PARAM_BAD_NAME;
#undef OPT
#undef OPT2
    return ctx.bError ? X265_PARAM_BAD_VALUE : 0;
}

static int parsePerformanceOpts(ParseContext& ctx)
{
#define OPT(STR) else if (!strcmp(ctx.name, STR))
#define OPT2(A,B) else if (!strcmp(ctx.name, A) || !strcmp(ctx.name, B))
    if (0) ;
    OPT("asm")
    {
#if X265_ARCH_X86
        if (!strcasecmp(ctx.value, "avx512"))
            ctx.p->cpuid = X265_NS::cpu_detect(true);
        else if (!ctx.bValueWasNull)
            ctx.p->cpuid = parseCpuName(ctx.value, ctx.bError, false);
        else
            ctx.p->cpuid = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
#else
        if (!ctx.bValueWasNull)
            ctx.p->cpuid = parseCpuName(ctx.value, ctx.bError, false);
        else
            ctx.p->cpuid = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
#endif
    }
    OPT("frame-threads") ctx.p->frameNumThreads = x265_atoi(ctx.value, ctx.bError);
    OPT("pmode") ctx.p->bDistributeModeAnalysis = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("pme") ctx.p->bDistributeMotionEstimation = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT2("pools", "numa-pools") snprintf(ctx.p->numaPools, X265_MAX_STRING_SIZE, "%s", ctx.value);
    OPT("wpp") ctx.p->bEnableWavefront = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("lookahead-threads") ctx.p->lookaheadThreads = x265_atoi(ctx.value, ctx.bError);
    OPT("lookahead-slices") ctx.p->lookaheadSlices = x265_atoi(ctx.value, ctx.bError);
    OPT("slices") ctx.p->maxSlices = x265_atoi(ctx.value, ctx.bError);
    OPT("copy-pic") ctx.p->bCopyPicToFrame = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("threaded-me") ctx.p->bThreadedME = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    else return X265_PARAM_BAD_NAME;
#undef OPT
#undef OPT2
    return ctx.bError ? X265_PARAM_BAD_VALUE : 0;
}

static int parseProfileLevelTierOpts(ParseContext& ctx)
{
#define OPT(STR) else if (!strcmp(ctx.name, STR))
#define OPT2(A,B) else if (!strcmp(ctx.name, A) || !strcmp(ctx.name, B))
    if (0) ;
    OPT2("level-idc", "level")
    {
        double val = x265_atof(ctx.value, ctx.bError);
        if (!ctx.bError && val < 10)
            ctx.p->levelIdc = (int)(10 * val + .5);
        else
        {
            ctx.bError = false;
            int ival = x265_atoi(ctx.value, ctx.bError);
            if (!ctx.bError && ival <= 100)
                ctx.p->levelIdc = ival;
            else
                ctx.bError = true;
        }
    }
    OPT("high-tier") ctx.p->bHighTier = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("allow-non-conformance") ctx.p->bAllowNonConformance = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("ref") ctx.p->maxNumReferences = x265_atoi(ctx.value, ctx.bError);
    OPT("uhd-bd") ctx.p->uhdBluray = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    else return X265_PARAM_BAD_NAME;
#undef OPT
#undef OPT2
    return ctx.bError ? X265_PARAM_BAD_VALUE : 0;
}

static int parseAnalysisOpts(ParseContext& ctx)
{
#define OPT(STR) else if (!strcmp(ctx.name, STR))
#define OPT2(A,B) else if (!strcmp(ctx.name, A) || !strcmp(ctx.name, B))
    if (0) ;
    OPT("ctu") ctx.p->maxCUSize = (uint32_t)x265_atoi(ctx.value, ctx.bError);
    OPT("min-cu-size") ctx.p->minCUSize = (uint32_t)x265_atoi(ctx.value, ctx.bError);
    OPT("tu-intra-depth") ctx.p->tuQTMaxIntraDepth = (uint32_t)x265_atoi(ctx.value, ctx.bError);
    OPT("tu-inter-depth") ctx.p->tuQTMaxInterDepth = (uint32_t)x265_atoi(ctx.value, ctx.bError);
    OPT("max-tu-size") ctx.p->maxTUSize = (uint32_t)x265_atoi(ctx.value, ctx.bError);
    OPT("me") ctx.p->searchMethod = parseName(ctx.value, x265_motion_est_names, ctx.bError);
    OPT("subme") ctx.p->subpelRefine = x265_atoi(ctx.value, ctx.bError);
    OPT("merange") ctx.p->searchRange = x265_atoi(ctx.value, ctx.bError);
    OPT("rect") ctx.p->bEnableRectInter = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("amp") ctx.p->bEnableAMP = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("max-merge") ctx.p->maxNumMergeCand = (uint32_t)x265_atoi(ctx.value, ctx.bError);
    OPT("temporal-mvp") ctx.p->bEnableTemporalMvp = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("early-skip") ctx.p->bEnableEarlySkip = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("rskip") ctx.p->recursionSkipMode = x265_atoi(ctx.value, ctx.bError);
    OPT("rskip-edge-threshold") ctx.p->edgeVarThreshold = x265_atoi(ctx.value, ctx.bError) / 100.0f;
    OPT("rdpenalty") ctx.p->rdPenalty = x265_atoi(ctx.value, ctx.bError);
    OPT("tskip") ctx.p->bEnableTransformSkip = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("tskip-fast") ctx.p->bEnableTSkipFast = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("strong-intra-smoothing") ctx.p->bEnableStrongIntraSmoothing = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("lossless") ctx.p->bLossless = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("cu-lossless") ctx.p->bCULossless = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT2("constrained-intra", "cip") ctx.p->bEnableConstrainedIntra = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("fast-intra") ctx.p->bEnableFastIntra = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("open-gop") ctx.p->bOpenGOP = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("intra-refresh") ctx.p->bIntraRefresh = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("scenecut")
    {
        ctx.bNameWasBool = true;
        ctx.p->scenecutThreshold = x265_atobool(ctx.value, ctx.bError);
        if (ctx.bError)
        {
            ctx.bError = false;
            ctx.p->scenecutThreshold = x265_atoi(ctx.value, ctx.bError);
        }
    }
    OPT("temporal-layers") ctx.p->bEnableTemporalSubLayers = x265_atoi(ctx.value, ctx.bError);
    OPT("keyint") ctx.p->keyframeMax = x265_atoi(ctx.value, ctx.bError);
    OPT("min-keyint") ctx.p->keyframeMin = x265_atoi(ctx.value, ctx.bError);
    OPT("rc-lookahead") ctx.p->lookaheadDepth = x265_atoi(ctx.value, ctx.bError);
    OPT("bframes") ctx.p->bframes = x265_atoi(ctx.value, ctx.bError);
    OPT("bframe-bias") ctx.p->bFrameBias = x265_atoi(ctx.value, ctx.bError);
    OPT("b-adapt")
    {
        ctx.bNameWasBool = true;
        ctx.p->bFrameAdaptive = x265_atobool(ctx.value, ctx.bError);
        if (ctx.bError)
        {
            ctx.bError = false;
            ctx.p->bFrameAdaptive = x265_atoi(ctx.value, ctx.bError);
        }
    }
    OPT("limit-refs") ctx.p->limitReferences = x265_atoi(ctx.value, ctx.bError);
    OPT("limit-modes") ctx.p->limitModes = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("weightp") ctx.p->bEnableWeightedPred = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("weightb") ctx.p->bEnableWeightedBiPred = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("rd") ctx.p->rdLevel = x265_atoi(ctx.value, ctx.bError);
    OPT2("rdoq", "rdoq-level") return parseRdoqValue(ctx, ctx.p->rdoqLevel);
    OPT("psy-rd") return parsePsyValue(ctx, ctx.p->psyRd);
    OPT("psy-rdoq") return parsePsyValue(ctx, ctx.p->psyRdoq);
    OPT("rd-refine") ctx.p->bEnableRdRefine = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("intra-rd-refine") ctx.p->bIntraRDRefine = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("signhide") ctx.p->bEnableSignHiding = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("b-intra") ctx.p->bIntraInBFrames = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("limit-tu") ctx.p->limitTU = x265_atoi(ctx.value, ctx.bError);
    OPT("splitrd-skip") ctx.p->bEnableSplitRdSkip = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("dynamic-rd") ctx.p->dynamicRd = x265_atof(ctx.value, ctx.bError);
    OPT("analyze-src-pics") ctx.p->bSourceReferenceEstimation = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("hme") ctx.p->bEnableHME = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("hme-search")
    {
        char search[3][5] = {{0}};
        if (3 == sscanf(ctx.value, "%4[^,],%4[^,],%4s", search[0], search[1], search[2]))
        {
            for (int level = 0; level < 3; level++)
                ctx.p->hmeSearchMethod[level] = parseName(search[level], x265_motion_est_names, ctx.bError);
        }
        else if (sscanf(ctx.value, "%4s", search[0]) == 1)
        {
            ctx.p->hmeSearchMethod[0] = parseName(search[0], x265_motion_est_names, ctx.bError);
            ctx.p->hmeSearchMethod[1] = ctx.p->hmeSearchMethod[2] = ctx.p->hmeSearchMethod[0];
            ctx.p->bEnableHME = true;
        }
        else ctx.bError = true;
    }
    OPT("hme-range")
    {
        if (sscanf(ctx.value, "%d,%d,%d", &ctx.p->hmeRange[0], &ctx.p->hmeRange[1], &ctx.p->hmeRange[2]) != 3)
            ctx.bError = true;
        else
            ctx.p->bEnableHME = true;
    }
    OPT("mcstf") ctx.p->bEnableTemporalFilter = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("gop-lookahead") ctx.p->gopLookahead = x265_atoi(ctx.value, ctx.bError);
    OPT("radl") ctx.p->radl = x265_atoi(ctx.value, ctx.bError);
    OPT("lowpass-dct") ctx.p->bLowPassDct = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    else return X265_PARAM_BAD_NAME;
#undef OPT
#undef OPT2
    return ctx.bError ? X265_PARAM_BAD_VALUE : 0;
}

static int parseRateControlOpts(ParseContext& ctx)
{
#define OPT(STR) else if (!strcmp(ctx.name, STR))
#define OPT2(A,B) else if (!strcmp(ctx.name, A) || !strcmp(ctx.name, B))
    if (0) ;
    OPT2("ipratio", "ip-factor") ctx.p->rc.ipFactor = x265_atof(ctx.value, ctx.bError);
    OPT2("pbratio", "pb-factor") ctx.p->rc.pbFactor = x265_atof(ctx.value, ctx.bError);
    OPT("qcomp") ctx.p->rc.qCompress = x265_atof(ctx.value, ctx.bError);
    OPT("qpstep") ctx.p->rc.qpStep = x265_atoi(ctx.value, ctx.bError);
    OPT("cplxblur") ctx.p->rc.complexityBlur = x265_atof(ctx.value, ctx.bError);
    OPT("qblur") ctx.p->rc.qblur = x265_atof(ctx.value, ctx.bError);
    OPT("aq-mode") ctx.p->rc.aqMode = x265_atoi(ctx.value, ctx.bError);
    OPT("aq-strength") ctx.p->rc.aqStrength = x265_atof(ctx.value, ctx.bError);
    OPT("aq-bias-strength") ctx.p->rc.aqBiasStrength = x265_atof(ctx.value, ctx.bError);
    OPT("aq-motion") ctx.p->bAQMotion = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("auto-aq") ctx.p->rc.bAutoAq = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("vbv-maxrate") ctx.p->rc.vbvMaxBitrate = x265_atoi(ctx.value, ctx.bError);
    OPT("vbv-bufsize") ctx.p->rc.vbvBufferSize = x265_atoi(ctx.value, ctx.bError);
    OPT("vbv-init") ctx.p->rc.vbvBufferInit = x265_atof(ctx.value, ctx.bError);
    OPT("vbv-end") ctx.p->vbvBufferEnd = x265_atof(ctx.value, ctx.bError);
    OPT("vbv-end-fr-adj") ctx.p->vbvEndFrameAdjust = x265_atof(ctx.value, ctx.bError);
    OPT("crf-max") ctx.p->rc.rfConstantMax = x265_atof(ctx.value, ctx.bError);
    OPT("crf-min") ctx.p->rc.rfConstantMin = x265_atof(ctx.value, ctx.bError);
    OPT("qpmax") ctx.p->rc.qpMax = x265_atoi(ctx.value, ctx.bError);
    OPT("qpmin") ctx.p->rc.qpMin = x265_atoi(ctx.value, ctx.bError);
    OPT("crf") { ctx.p->rc.rfConstant = x265_atof(ctx.value, ctx.bError); ctx.p->rc.rateControlMode = X265_RC_CRF; }
    OPT("bitrate") { ctx.p->rc.bitrate = x265_atoi(ctx.value, ctx.bError); ctx.p->rc.rateControlMode = X265_RC_ABR; }
    OPT("qp") { ctx.p->rc.qp = x265_atoi(ctx.value, ctx.bError); ctx.p->rc.rateControlMode = X265_RC_CQP; }
    OPT("rc-grain") ctx.p->rc.bEnableGrain = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("cutree") ctx.p->rc.cuTree = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("pass")
    {
        int pass = x265_clip3(0, 3, x265_atoi(ctx.value, ctx.bError));
        ctx.p->rc.bStatWrite = pass & 1;
        ctx.p->rc.bStatRead = pass & 2;
        ctx.p->rc.dataShareMode = X265_SHARE_MODE_FILE;
    }
    OPT("stats") snprintf(ctx.p->rc.statFileName, X265_MAX_STRING_SIZE, "%s", ctx.value);
    OPT("slow-firstpass") ctx.p->rc.bEnableSlowFirstPass = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("strict-cbr") { ctx.p->rc.bStrictCbr = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError)); ctx.p->rc.pbFactor = 1.0; }
    OPT("zones")
    {
        ctx.p->rc.zoneCount = 1;
        for (const char* c = ctx.value; *c; c++)
            ctx.p->rc.zoneCount += (*c == '/');
        ctx.p->rc.zones = X265_MALLOC(x265_zone, ctx.p->rc.zoneCount);
        const char* c = ctx.value;
        for (int i = 0; i < ctx.p->rc.zoneCount; i++)
        {
            int len;
            if (3 == sscanf(c, "%d,%d,q=%d%n", &ctx.p->rc.zones[i].startFrame, &ctx.p->rc.zones[i].endFrame, &ctx.p->rc.zones[i].qp, &len))
                ctx.p->rc.zones[i].bForceQp = 1;
            else if (3 == sscanf(c, "%d,%d,b=%f%n", &ctx.p->rc.zones[i].startFrame, &ctx.p->rc.zones[i].endFrame, &ctx.p->rc.zones[i].bitrateFactor, &len))
                ctx.p->rc.zones[i].bForceQp = 0;
            else { ctx.bError = true; break; }
            c += len + 1;
        }
    }
    OPT("analysis-reuse-mode") ctx.p->analysisReuseMode = parseName(ctx.value, x265_analysis_names, ctx.bError);
    OPT("analysis-reuse-file") snprintf(ctx.p->analysisReuseFileName, X265_MAX_STRING_SIZE, "%s", ctx.value);
    OPT("analysis-save") snprintf(ctx.p->analysisSave, X265_MAX_STRING_SIZE, "%s", ctx.value);
    OPT("analysis-load") snprintf(ctx.p->analysisLoad, X265_MAX_STRING_SIZE, "%s", ctx.value);
    OPT("analysis-reuse-level")
    {
        int level = x265_atoi(ctx.value, ctx.bError);
        ctx.p->analysisReuseLevel = level;
        ctx.p->analysisSaveReuseLevel = level;
        ctx.p->analysisLoadReuseLevel = level;
    }
    OPT("analysis-save-reuse-level") ctx.p->analysisSaveReuseLevel = x265_atoi(ctx.value, ctx.bError);
    OPT("analysis-load-reuse-level") ctx.p->analysisLoadReuseLevel = x265_atoi(ctx.value, ctx.bError);
    OPT("multi-pass-opt-analysis") ctx.p->analysisMultiPassRefine = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("multi-pass-opt-distortion") ctx.p->analysisMultiPassDistortion = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("multi-pass-opt-rps") ctx.p->bMultiPassOptRPS = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("scenecut-bias") ctx.p->scenecutBias = x265_atof(ctx.value, ctx.bError);
    OPT("hist-scenecut") ctx.p->bHistBasedSceneCut = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("scenecut-aware-qp") ctx.p->bEnableSceneCutAwareQp = x265_atoi(ctx.value, ctx.bError);
    OPT("masking-strength") ctx.bError = parseMaskingStrength(ctx.p, ctx.value);
    OPT("const-vbv") ctx.p->rc.bEnableConstVbv = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("hevc-aq") ctx.p->rc.hevcAq = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("qp-adaptation-range") ctx.p->rc.qpAdaptationRange = x265_atof(ctx.value, ctx.bError);
    OPT("cra-nal") ctx.p->craNal = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("nr-intra") ctx.p->noiseReductionIntra = x265_atoi(ctx.value, ctx.bError);
    OPT("nr-inter") ctx.p->noiseReductionInter = x265_atoi(ctx.value, ctx.bError);
    OPT("qg-size") ctx.p->rc.qgSize = x265_atoi(ctx.value, ctx.bError);
    OPT("lambda-file") snprintf(ctx.p->rc.lambdaFileName, X265_MAX_STRING_SIZE, "%s", ctx.value);
    OPT("vbv-live-multi-pass") ctx.p->bliveVBV2pass = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("min-vbv-fullness") ctx.p->minVbvFullness = x265_atof(ctx.value, ctx.bError);
    OPT("max-vbv-fullness") ctx.p->maxVbvFullness = x265_atof(ctx.value, ctx.bError);
    OPT("frame-rc") ctx.p->bConfigRCFrame = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    else return X265_PARAM_BAD_NAME;
#undef OPT
#undef OPT2
    return ctx.bError ? X265_PARAM_BAD_VALUE : 0;
}

static int parseVuiSeiHdrOpts(ParseContext& ctx)
{
#define OPT(STR) else if (!strcmp(ctx.name, STR))
#define OPT2(A,B) else if (!strcmp(ctx.name, A) || !strcmp(ctx.name, B))
    if (0) ;
    OPT("annexb") ctx.p->bAnnexB = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("repeat-headers") ctx.p->bRepeatHeaders = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("aud") ctx.p->bEnableAccessUnitDelimiters = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("info") ctx.p->bEmitInfoSEI = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("hrd") ctx.p->bEmitHRDSEI = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("sar")
    {
        ctx.p->vui.aspectRatioIdc = parseName(ctx.value, x265_sar_names, ctx.bError);
        if (ctx.bError)
        {
            ctx.bError = false;
            ctx.p->vui.aspectRatioIdc = X265_EXTENDED_SAR;
            ctx.bError = sscanf(ctx.value, "%d:%d", &ctx.p->vui.sarWidth, &ctx.p->vui.sarHeight) != 2;
        }
    }
    OPT("overscan")
    {
        if (!strcmp(ctx.value, "show"))
            ctx.p->vui.bEnableOverscanInfoPresentFlag = 1;
        else if (!strcmp(ctx.value, "crop"))
        {
            ctx.p->vui.bEnableOverscanInfoPresentFlag = 1;
            ctx.p->vui.bEnableOverscanAppropriateFlag = 1;
        }
        else if (!strcmp(ctx.value, "unknown"))
            ctx.p->vui.bEnableOverscanInfoPresentFlag = 0;
        else
            ctx.bError = true;
    }
    OPT("videoformat") { ctx.p->vui.bEnableVideoSignalTypePresentFlag = 1; ctx.p->vui.videoFormat = parseName(ctx.value, x265_video_format_names, ctx.bError); }
    OPT("range") { ctx.p->vui.bEnableVideoSignalTypePresentFlag = 1; ctx.p->vui.bEnableVideoFullRangeFlag = parseName(ctx.value, x265_fullrange_names, ctx.bError); }
    OPT("colorprim") { ctx.p->vui.bEnableVideoSignalTypePresentFlag = 1; ctx.p->vui.bEnableColorDescriptionPresentFlag = 1; ctx.p->vui.colorPrimaries = parseName(ctx.value, x265_colorprim_names, ctx.bError); }
    OPT("transfer") { ctx.p->vui.bEnableVideoSignalTypePresentFlag = 1; ctx.p->vui.bEnableColorDescriptionPresentFlag = 1; ctx.p->vui.transferCharacteristics = parseName(ctx.value, x265_transfer_names, ctx.bError); }
    OPT("colormatrix") { ctx.p->vui.bEnableVideoSignalTypePresentFlag = 1; ctx.p->vui.bEnableColorDescriptionPresentFlag = 1; ctx.p->vui.matrixCoeffs = parseName(ctx.value, x265_colmatrix_names, ctx.bError); }
    OPT("chromaloc")
    {
        ctx.p->vui.bEnableChromaLocInfoPresentFlag = 1;
        ctx.p->vui.chromaSampleLocTypeTopField = x265_atoi(ctx.value, ctx.bError);
        ctx.p->vui.chromaSampleLocTypeBottomField = ctx.p->vui.chromaSampleLocTypeTopField;
    }
    OPT2("display-window", "crop-rect")
    {
        ctx.p->vui.bEnableDefaultDisplayWindowFlag = 1;
        ctx.bError = sscanf(ctx.value, "%d,%d,%d,%d", &ctx.p->vui.defDispWinLeftOffset, &ctx.p->vui.defDispWinTopOffset, &ctx.p->vui.defDispWinRightOffset, &ctx.p->vui.defDispWinBottomOffset) != 4;
    }
    OPT("master-display") snprintf(ctx.p->masteringDisplayColorVolume, X265_MAX_STRING_SIZE, "%s", ctx.value);
    OPT("max-cll") ctx.bError = sscanf(ctx.value, "%hu,%hu", &ctx.p->maxCLL, &ctx.p->maxFALL) != 2;
    OPT("min-luma") ctx.p->minLuma = (uint16_t)x265_atoi(ctx.value, ctx.bError);
    OPT("max-luma") ctx.p->maxLuma = (uint16_t)x265_atoi(ctx.value, ctx.bError);
    OPT("vui-timing-info") ctx.p->bEmitVUITimingInfo = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("vui-hrd-info") ctx.p->bEmitVUIHRDInfo = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("log2-max-poc-lsb") ctx.p->log2MaxPocLsb = x265_atoi(ctx.value, ctx.bError);
    OPT("hdr") ctx.p->bEmitHDR10SEI = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("hdr10") ctx.p->bEmitHDR10SEI = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("hdr-opt") ctx.p->bHDR10Opt = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("hdr10-opt") ctx.p->bHDR10Opt = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("dhdr10-info") snprintf(ctx.p->toneMapFile, X265_MAX_STRING_SIZE, "%s", ctx.value);
    OPT("dhdr10-opt") ctx.p->bDhdr10opt = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("idr-recovery-sei") ctx.p->bEmitIDRRecoverySEI = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("single-sei") ctx.p->bSingleSeiNal = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("atc-sei") ctx.p->preferredTransferCharacteristics = x265_atoi(ctx.value, ctx.bError);
    OPT("pic-struct") ctx.p->pictureStructure = x265_atoi(ctx.value, ctx.bError);
    OPT("nalu-file") snprintf(ctx.p->naluFile, X265_MAX_STRING_SIZE, "%s", ctx.value);
    OPT("hrd-concat") ctx.p->bEnableHRDConcatFlag = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("eob") ctx.p->bEnableEndOfBitstream = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("eos") ctx.p->bEnableEndOfSequence = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("cll") ctx.p->bEmitCLL = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("video-signal-type-preset") snprintf(ctx.p->videoSignalTypePreset, X265_MAX_STRING_SIZE, "%s", ctx.value);
    OPT("dolby-vision-profile")
    {
        double val = x265_atof(ctx.value, ctx.bError);
        if (!ctx.bError && val < 10)
            ctx.p->dolbyProfile = (int)(10 * val + .5);
        else
        {
            ctx.bError = false;
            int ival = x265_atoi(ctx.value, ctx.bError);
            if (!ctx.bError && ival <= 100)
                ctx.p->dolbyProfile = ival;
            else
                ctx.bError = true;
        }
    }
    else return X265_PARAM_BAD_NAME;
#undef OPT
#undef OPT2
    return ctx.bError ? X265_PARAM_BAD_VALUE : 0;
}

static int parseExtendedOpts(ParseContext& ctx)
{
#define OPT(STR) else if (!strcmp(ctx.name, STR))
    if (0) ;
    OPT("opt-qp-pps") ctx.p->bOptQpPPS = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("opt-ref-list-length-pps") ctx.p->bOptRefListLengthPPS = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("opt-cu-delta-qp") ctx.p->bOptCUDeltaQP = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("limit-sao") ctx.p->bLimitSAO = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("ctu-info") ctx.p->bCTUInfo = x265_atoi(ctx.value, ctx.bError);
    OPT("scale-factor") ctx.p->scaleFactor = x265_atoi(ctx.value, ctx.bError);
    OPT("refine-intra") ctx.p->intraRefine = x265_atoi(ctx.value, ctx.bError);
    OPT("refine-inter") ctx.p->interRefine = x265_atoi(ctx.value, ctx.bError);
    OPT("refine-mv") ctx.p->mvRefine = x265_atoi(ctx.value, ctx.bError);
    OPT("force-flush") ctx.p->forceFlush = x265_atoi(ctx.value, ctx.bError);
    OPT("refine-analysis-type")
    {
        if (!strcmp(ctx.value, "avc")) ctx.p->bAnalysisType = AVC_INFO;
        else if (!strcmp(ctx.value, "hevc")) ctx.p->bAnalysisType = HEVC_INFO;
        else if (!strcmp(ctx.value, "off")) ctx.p->bAnalysisType = DEFAULT;
        else ctx.bError = true;
    }
    OPT("max-ausize-factor") ctx.p->maxAUSizeFactor = x265_atof(ctx.value, ctx.bError);
    OPT("dynamic-refine") ctx.p->bDynamicRefine = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("refine-ctu-distortion") ctx.p->ctuDistortionRefine = x265_atoi(ctx.value, ctx.bError);
    OPT("selective-sao") ctx.p->selectiveSAO = x265_atoi(ctx.value, ctx.bError);
    OPT("fades") ctx.p->bEnableFades = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
    OPT("film-grain") ctx.p->filmGrain = (char*)ctx.value;
    OPT("aom-film-grain") ctx.p->aomFilmGrain = (char*)ctx.value;
    OPT("sbrc") ctx.p->bEnableSBRC = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
#if ENABLE_ALPHA
    OPT("alpha")
    {
        if ((ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError)))
        {
            ctx.p->bEnableAlpha = 1;
            ctx.p->numScalableLayers = 2;
            ctx.p->numLayers = 2;
        }
    }
#endif
#if ENABLE_MULTIVIEW
    OPT("format") ctx.p->format = x265_atoi(ctx.value, ctx.bError);
    OPT("num-views") ctx.p->numViews = x265_atoi(ctx.value, ctx.bError);
#endif
#if ENABLE_SCC_EXT
    OPT("scc") ctx.p->bEnableSCC = x265_atoi(ctx.value, ctx.bError);
#endif
#ifdef SVT_HEVC
    OPT("svt") ctx.p->bEnableSvtHevc = (ctx.bNameWasBool = true, x265_atobool(ctx.value, ctx.bError));
#endif
    else return X265_PARAM_BAD_NAME;
#undef OPT
    return ctx.bError ? X265_PARAM_BAD_VALUE : 0;
}

int x265_param_parse(x265_param* p, const char* name, const char* value)
{
    ParseContext ctx;
    ctx.p = p;
    ctx.name = name;
    ctx.value = value;
    ctx.bError = false;
    ctx.bNameWasBool = false;
    ctx.bValueWasNull = !value;
    ctx.nameBuf[0];
#ifdef SVT_HEVC
    static int count;
    count++;
#endif

    int ret = normalizeParseArgs(ctx);
    if (ret)
        return ret;

#ifdef SVT_HEVC
    if (p->bEnableSvtHevc)
    {
        if (svt_param_parse(p, ctx.name, ctx.value))
            return X265_PARAM_BAD_VALUE;
        return 0;
    }
#endif

    if ((ret = parseLoggingStatsOpts(ctx))     != X265_PARAM_BAD_NAME) return ret;
    if ((ret = parseInputOutputOpts(ctx))      != X265_PARAM_BAD_NAME) return ret;
    if ((ret = parsePerformanceOpts(ctx))      != X265_PARAM_BAD_NAME) return ret;
    if ((ret = parseProfileLevelTierOpts(ctx)) != X265_PARAM_BAD_NAME) return ret;
    if ((ret = parseAnalysisOpts(ctx))         != X265_PARAM_BAD_NAME) return ret;
    if ((ret = parseRateControlOpts(ctx))      != X265_PARAM_BAD_NAME) return ret;
    if ((ret = parseVuiSeiHdrOpts(ctx))        != X265_PARAM_BAD_NAME) return ret;
    if ((ret = parseExtendedOpts(ctx))         != X265_PARAM_BAD_NAME) return ret;

    return X265_PARAM_BAD_NAME;
}

} /* end extern "C" or namespace */

namespace X265_NS {
// internal encoder functions

int x265_atoi(const char* str, bool& bError)
{
    char *end;
    int v = strtol(str, &end, 0);

    if (end == str || *end != '\0')
        bError = true;
    return v;
}

double x265_atof(const char* str, bool& bError)
{
    char *end;
    double v = strtod(str, &end);

    if (end == str || *end != '\0')
        bError = true;
    return v;
}

/* cpu name can be:
 *   auto || true - x265::cpu_detect()
 *   false || no  - disabled
 *   integer bitmap value
 *   comma separated list of SIMD names, eg: SSE4.1,XOP */
int parseCpuName(const char* value, bool& bError, bool bEnableavx512)
{
    if (!value)
    {
        bError = 1;
        return 0;
    }
    int cpu;
    if (isdigit(value[0]))
        cpu = x265_atoi(value, bError);
    else
        cpu = !strcmp(value, "auto") || x265_atobool(value, bError) ? X265_NS::cpu_detect(bEnableavx512) : 0;

    if (bError)
    {
        char *buf = strdup(value);
        char *tok, *saveptr = NULL, *init;
        bError = 0;
        cpu = 0;
        for (init = buf; (tok = strtok_r(init, ",", &saveptr)); init = NULL)
        {
            int i;
            for (i = 0; X265_NS::cpu_names[i].flags && strcasecmp(tok, X265_NS::cpu_names[i].name); i++)
            {
            }

            cpu |= X265_NS::cpu_names[i].flags;
            if (!X265_NS::cpu_names[i].flags)
                bError = 1;
        }

        free(buf);
#if X265_ARCH_X86
        if ((cpu & X265_CPU_SSSE3) && !(cpu & X265_CPU_SSE2_IS_SLOW))
            cpu |= X265_CPU_SSE2_IS_FAST;
#endif
    }

    return cpu;
}

static const int fixedRatios[][2] =
{
    { 1,  1 },
    { 12, 11 },
    { 10, 11 },
    { 16, 11 },
    { 40, 33 },
    { 24, 11 },
    { 20, 11 },
    { 32, 11 },
    { 80, 33 },
    { 18, 11 },
    { 15, 11 },
    { 64, 33 },
    { 160, 99 },
    { 4, 3 },
    { 3, 2 },
    { 2, 1 },
};

void setParamAspectRatio(x265_param* p, int width, int height)
{
    p->vui.aspectRatioIdc = X265_EXTENDED_SAR;
    p->vui.sarWidth = width;
    p->vui.sarHeight = height;
    for (size_t i = 0; i < sizeof(fixedRatios) / sizeof(fixedRatios[0]); i++)
    {
        if (width == fixedRatios[i][0] && height == fixedRatios[i][1])
        {
            p->vui.aspectRatioIdc = (int)i + 1;
            return;
        }
    }
}

void getParamAspectRatio(x265_param* p, int& width, int& height)
{
    if (!p->vui.aspectRatioIdc)
        width = height = 0;
    else if ((size_t)p->vui.aspectRatioIdc <= sizeof(fixedRatios) / sizeof(fixedRatios[0]))
    {
        width  = fixedRatios[p->vui.aspectRatioIdc - 1][0];
        height = fixedRatios[p->vui.aspectRatioIdc - 1][1];
    }
    else if (p->vui.aspectRatioIdc == X265_EXTENDED_SAR)
    {
        width  = p->vui.sarWidth;
        height = p->vui.sarHeight;
    }
    else
        width = height = 0;
}

static inline int _confirm(x265_param* param, bool bflag, const char* message)
{
    if (!bflag)
        return 0;

    x265_log(param, X265_LOG_ERROR, "%s\n", message);
    return 1;
}

int x265_check_params(x265_param* param)
{
#define CHECK(expr, msg) check_failed |= _confirm(param, expr, msg)
    int check_failed = 0; /* abort if there is a fatal configuration problem */
    CHECK((uint64_t)param->sourceWidth * param->sourceHeight > 142606336,
          "Input video resolution exceeds the maximum supported luma samples 142,606,336 (16384x8704) of Level 7.2.");
    CHECK(param->uhdBluray == 1 && (X265_DEPTH != 10 || param->internalCsp != 1 || param->interlaceMode != 0),
        "uhd-bd: bit depth, chroma subsample, source picture type must be 10, 4:2:0, progressive");
    CHECK(param->maxCUSize != 64 && param->maxCUSize != 32 && param->maxCUSize != 16,
          "max cu size must be 16, 32, or 64");
    if (check_failed == 1)
        return check_failed;

    uint32_t maxLog2CUSize = (uint32_t)g_log2Size[param->maxCUSize];
    uint32_t tuQTMaxLog2Size = X265_MIN(maxLog2CUSize, 5);
    uint32_t tuQTMinLog2Size = 2; //log2(4)

    CHECK((param->maxSlices > 1) && !param->bEnableWavefront,
        "Multiple-Slices mode must be enable Wavefront Parallel Processing (--wpp)");
    CHECK(param->internalBitDepth != X265_DEPTH,
          "internalBitDepth must match compiled bit depth");
    CHECK(param->minCUSize != 32 && param->minCUSize != 16 && param->minCUSize != 8,
          "minimim CU size must be 8, 16 or 32");
    CHECK(param->minCUSize > param->maxCUSize,
          "min CU size must be less than or equal to max CU size");
    CHECK(param->rc.qp < -6 * (param->internalBitDepth - 8) || param->rc.qp > QP_MAX_SPEC,
          "QP exceeds supported range (-QpBDOffsety to 51)");
    CHECK(param->fpsNum == 0 || param->fpsDenom == 0,
          "Frame rate numerator and denominator must be specified");
    CHECK(param->interlaceMode < 0 || param->interlaceMode > 2,
          "Interlace mode must be 0 (progressive) 1 (top-field first) or 2 (bottom field first)");
    CHECK(param->searchMethod < 0 || param->searchMethod > X265_FULL_SEARCH,
          "Search method is not supported value (0:DIA 1:HEX 2:UMH 3:HM 4:SEA 5:FULL)");
    CHECK(param->searchRange < 0,
          "Search Range must be more than 0");
    CHECK(param->searchRange >= 32768,
          "Search Range must be less than 32768");
    CHECK(param->subpelRefine > X265_MAX_SUBPEL_LEVEL,
          "subme must be less than or equal to X265_MAX_SUBPEL_LEVEL (7)");
    CHECK(param->subpelRefine < 0,
          "subme must be greater than or equal to 0");
    CHECK(param->limitReferences > 3,
          "limitReferences must be 0, 1, 2 or 3");
    CHECK(param->limitModes > 1,
          "limitRectAmp must be 0, 1");
    CHECK(param->frameNumThreads < 0 || param->frameNumThreads > X265_MAX_FRAME_THREADS,
          "frameNumThreads (--frame-threads) must be [0 .. X265_MAX_FRAME_THREADS)");
    CHECK(param->cbQpOffset < -12, "Min. Chroma Cb QP Offset is -12");
    CHECK(param->cbQpOffset >  12, "Max. Chroma Cb QP Offset is  12");
    CHECK(param->crQpOffset < -12, "Min. Chroma Cr QP Offset is -12");
    CHECK(param->crQpOffset >  12, "Max. Chroma Cr QP Offset is  12");

    CHECK(tuQTMaxLog2Size > maxLog2CUSize,
          "QuadtreeTULog2MaxSize must be log2(maxCUSize) or smaller.");

    CHECK(param->tuQTMaxInterDepth < 1 || param->tuQTMaxInterDepth > 4,
          "QuadtreeTUMaxDepthInter must be greater than 0 and less than 5");
    CHECK(maxLog2CUSize < tuQTMinLog2Size + param->tuQTMaxInterDepth - 1,
          "QuadtreeTUMaxDepthInter must be less than or equal to the difference between log2(maxCUSize) and QuadtreeTULog2MinSize plus 1");
    CHECK(param->tuQTMaxIntraDepth < 1 || param->tuQTMaxIntraDepth > 4,
          "QuadtreeTUMaxDepthIntra must be greater 0 and less than 5");
    CHECK(maxLog2CUSize < tuQTMinLog2Size + param->tuQTMaxIntraDepth - 1,
          "QuadtreeTUMaxDepthInter must be less than or equal to the difference between log2(maxCUSize) and QuadtreeTULog2MinSize plus 1");
    CHECK((param->maxTUSize != 32 && param->maxTUSize != 16 && param->maxTUSize != 8 && param->maxTUSize != 4),
          "max TU size must be 4, 8, 16, or 32");
    CHECK(param->limitTU > 4, "Invalid limit-tu option, limit-TU must be between 0 and 4");
    CHECK(param->maxNumMergeCand < 1, "MaxNumMergeCand must be 1 or greater.");
    CHECK(param->maxNumMergeCand > 5, "MaxNumMergeCand must be 5 or smaller.");

    CHECK(param->maxNumReferences < 1, "maxNumReferences must be 1 or greater.");
    CHECK(param->maxNumReferences > MAX_NUM_REF, "maxNumReferences must be 16 or smaller.");

    CHECK(param->sourceWidth < (int)param->maxCUSize || param->sourceHeight < (int)param->maxCUSize,
          "Picture size must be at least one CTU");
    CHECK(param->internalCsp < X265_CSP_I400 || X265_CSP_I444 < param->internalCsp,
          "chroma subsampling must be i400 (4:0:0 monochrome), i420 (4:2:0 default), i422 (4:2:0), i444 (4:4:4)");
    CHECK(param->sourceWidth & !!CHROMA_H_SHIFT(param->internalCsp),
          "Picture width must be an integer multiple of the specified chroma subsampling");
    CHECK(param->sourceHeight & !!CHROMA_V_SHIFT(param->internalCsp),
          "Picture height must be an integer multiple of the specified chroma subsampling");

    CHECK(param->rc.rateControlMode > X265_RC_CRF || param->rc.rateControlMode < X265_RC_ABR,
          "Rate control mode is out of range");
    CHECK(param->rdLevel < 1 || param->rdLevel > 6,
          "RD Level is out of range");
    CHECK(param->rdoqLevel < 0 || param->rdoqLevel > 2,
          "RDOQ Level is out of range");
    CHECK(param->dynamicRd < 0 || param->dynamicRd > x265_ADAPT_RD_STRENGTH,
          "Dynamic RD strength must be between 0 and 4");
    CHECK(param->recursionSkipMode > 2 || param->recursionSkipMode < 0,
          "Invalid Recursion skip mode. Valid modes 0,1,2");
    if (param->recursionSkipMode == EDGE_BASED_RSKIP)
    {
        CHECK(param->edgeVarThreshold < 0.0f || param->edgeVarThreshold > 1.0f,
              "Minimum edge density percentage for a CU should be an integer between 0 to 100");
    }
    CHECK(param->bframes && (param->bEnableTemporalFilter ? (param->bframes > param->lookaheadDepth) : (param->bframes >= param->lookaheadDepth)) && !param->rc.bStatRead,
          "Lookahead depth must be greater than the max consecutive bframe count");
    CHECK(param->bframes < 0,
          "bframe count should be greater than zero");
    CHECK(param->bframes > X265_BFRAME_MAX,
          "max consecutive bframe count must be 16 or smaller");
    CHECK(param->lookaheadDepth > X265_LOOKAHEAD_MAX,
          "Lookahead depth must be less than 256");
    CHECK(param->lookaheadSlices > 16 || param->lookaheadSlices < 0,
          "Lookahead slices must between 0 and 16");
    CHECK(param->rc.aqMode < X265_AQ_NONE || param->rc.aqMode > X265_AQ_EDGE_BIASED,
          "Aq-Mode is out of range");
    CHECK(param->rc.aqStrength < 0 || param->rc.aqStrength > 3,
          "Aq-Strength is out of range");
    CHECK(param->rc.qpAdaptationRange < 1.0f || param->rc.qpAdaptationRange > 6.0f,
        "qp adaptation range is out of range");
    CHECK(param->deblockingFilterTCOffset < -6 || param->deblockingFilterTCOffset > 6,
          "deblocking filter tC offset must be in the range of -6 to +6");
    CHECK(param->deblockingFilterBetaOffset < -6 || param->deblockingFilterBetaOffset > 6,
          "deblocking filter Beta offset must be in the range of -6 to +6");
    CHECK(param->psyRd < 0 || 5.0 < param->psyRd, "Psy-rd strength must be between 0 and 5.0");
    CHECK(param->psyRdoq < 0 || 50.0 < param->psyRdoq, "Psy-rdoq strength must be between 0 and 50.0");
    CHECK(param->bEnableWavefront < 0, "WaveFrontSynchro cannot be negative");
    CHECK((param->vui.aspectRatioIdc < 0
           || param->vui.aspectRatioIdc > 16)
          && param->vui.aspectRatioIdc != X265_EXTENDED_SAR,
          "Sample Aspect Ratio must be 0-16 or 255");
    CHECK(param->vui.aspectRatioIdc == X265_EXTENDED_SAR && param->vui.sarWidth <= 0,
          "Sample Aspect Ratio width must be greater than 0");
    CHECK(param->vui.aspectRatioIdc == X265_EXTENDED_SAR && param->vui.sarHeight <= 0,
          "Sample Aspect Ratio height must be greater than 0");
    CHECK(param->vui.videoFormat < 0 || param->vui.videoFormat > 5,
          "Video Format must be component,"
          " pal, ntsc, secam, mac or unknown");
    CHECK(param->vui.colorPrimaries < 0
          || param->vui.colorPrimaries > 12
          || param->vui.colorPrimaries == 3,
          "Color Primaries must be unknown, bt709, bt470m,"
          " bt470bg, smpte170m, smpte240m, film, bt2020, smpte-st-428, smpte-rp-431 or smpte-eg-432");
    CHECK(param->vui.transferCharacteristics < 0
          || param->vui.transferCharacteristics > 18
          || param->vui.transferCharacteristics == 3,
          "Transfer Characteristics must be unknown, bt709, bt470m, bt470bg,"
          " smpte170m, smpte240m, linear, log100, log316, iec61966-2-4, bt1361e,"
          " iec61966-2-1, bt2020-10, bt2020-12, smpte-st-2084, smpte-st-428 or arib-std-b67");
    CHECK(param->vui.matrixCoeffs < 0
          || param->vui.matrixCoeffs > 15
          || param->vui.matrixCoeffs == 3,
          "Matrix Coefficients must be unknown, bt709, fcc, bt470bg, smpte170m,"
          " smpte240m, gbr, ycgco, bt2020nc, bt2020c, smpte-st-2085, chroma-nc, chroma-c, ictcp or ipt-pq-c2");
    CHECK(param->vui.chromaSampleLocTypeTopField < 0
          || param->vui.chromaSampleLocTypeTopField > 5,
          "Chroma Sample Location Type Top Field must be 0-5");
    CHECK(param->vui.chromaSampleLocTypeBottomField < 0
          || param->vui.chromaSampleLocTypeBottomField > 5,
          "Chroma Sample Location Type Bottom Field must be 0-5");
    CHECK(param->vui.defDispWinLeftOffset < 0,
          "Default Display Window Left Offset must be 0 or greater");
    CHECK(param->vui.defDispWinRightOffset < 0,
          "Default Display Window Right Offset must be 0 or greater");
    CHECK(param->vui.defDispWinTopOffset < 0,
          "Default Display Window Top Offset must be 0 or greater");
    CHECK(param->vui.defDispWinBottomOffset < 0,
          "Default Display Window Bottom Offset must be 0 or greater");
    CHECK(param->rc.rfConstant < -6 * (param->internalBitDepth - 8) || param->rc.rfConstant > 51,
          "Valid quality based range: -qpBDOffsetY to 51");
    CHECK(param->rc.rfConstantMax < -6 * (param->internalBitDepth - 8) || param->rc.rfConstantMax > 51,
          "Valid quality based range: -qpBDOffsetY to 51");
    CHECK(param->rc.rfConstantMin < -6 * (param->internalBitDepth - 8) || param->rc.rfConstantMin > 51,
          "Valid quality based range: -qpBDOffsetY to 51");
    CHECK(param->bFrameAdaptive < 0 || param->bFrameAdaptive > 2,
          "Valid adaptive b scheduling values 0 - none, 1 - fast, 2 - full");
    CHECK(param->logLevel<-1 || param->logLevel> X265_LOG_FULL,
          "Valid Logging level -1:none 0:error 1:warning 2:info 3:debug 4:full");
    CHECK(param->scenecutThreshold < 0,
          "scenecutThreshold must be greater than 0");
    CHECK(param->scenecutBias < 0 || 100 < param->scenecutBias,
            "scenecut-bias must be between 0 and 100");
    CHECK(param->radl < 0 || param->radl > param->bframes,
          "radl must be between 0 and bframes");
    CHECK(param->rdPenalty < 0 || param->rdPenalty > 2,
          "Valid penalty for 32x32 intra TU in non-I slices. 0:disabled 1:RD-penalty 2:maximum");
    CHECK(param->keyframeMax < -1,
          "Invalid max IDR period in frames. value should be greater than -1");
    CHECK(param->gopLookahead < -1,
          "GOP lookahead must be greater than -1");
    CHECK(param->decodedPictureHashSEI < 0 || param->decodedPictureHashSEI > 3,
          "Invalid hash option. Decoded Picture Hash SEI 0: disabled, 1: MD5, 2: CRC, 3: Checksum");
    CHECK(param->rc.vbvBufferSize < 0,
          "Size of the vbv buffer can not be less than zero");
    CHECK(param->rc.vbvMaxBitrate < 0,
          "Maximum local bit rate can not be less than zero");
    CHECK(param->rc.vbvBufferInit < 0,
          "Valid initial VBV buffer occupancy must be a fraction 0 - 1, or size in kbits");
    CHECK(param->vbvBufferEnd < 0,
        "Valid final VBV buffer emptiness must be a fraction 0 - 1, or size in kbits");
    CHECK(param->vbvEndFrameAdjust < 0,
        "Valid vbv-end-fr-adj must be a fraction 0 - 1");
    if ((param->rc.vbvBufferSize > 0 || param->rc.vbvMaxBitrate > 0) && param->bThreadedME)
    {
        param->bThreadedME = 0;
        x265_log(param, X265_LOG_WARNING, "VBV and threaded-me both enabled. Disabling threaded-me\n");
    }
    CHECK(param->minVbvFullness < 0 && param->minVbvFullness > 100,
        "min-vbv-fullness must be a fraction 0 - 100");
    CHECK(param->maxVbvFullness < 0 && param->maxVbvFullness > 100,
        "max-vbv-fullness must be a fraction 0 - 100");
    CHECK(param->rc.bitrate < 0,
          "Target bitrate can not be less than zero");
    CHECK(param->rc.qCompress < 0.5 || param->rc.qCompress > 1.0,
          "qCompress must be between 0.5 and 1.0");
    if (param->noiseReductionIntra)
        CHECK(0 > param->noiseReductionIntra || param->noiseReductionIntra > 2000, "Valid noise reduction range 0 - 2000");
    if (param->noiseReductionInter)
        CHECK(0 > param->noiseReductionInter || param->noiseReductionInter > 2000, "Valid noise reduction range 0 - 2000");
    CHECK(param->rc.rateControlMode == X265_RC_CQP && param->rc.bStatRead,
          "Constant QP is incompatible with 2pass");
    CHECK(param->rc.bStrictCbr && (param->rc.bitrate <= 0 || param->rc.vbvBufferSize <=0),
          "Strict-cbr cannot be applied without specifying both target bitrate and vbv bufsize");
    CHECK(strlen(param->analysisSave) && (param->analysisSaveReuseLevel < 0 || param->analysisSaveReuseLevel > 10),
        "Invalid analysis save refine level. Value must be between 1 and 10 (inclusive)");
    CHECK(strlen(param->analysisLoad) && (param->analysisLoadReuseLevel < 0 || param->analysisLoadReuseLevel > 10),
        "Invalid analysis load refine level. Value must be between 1 and 10 (inclusive)");
    CHECK(strlen(param->analysisLoad) && (param->mvRefine < 1 || param->mvRefine > 3),
        "Invalid mv refinement level. Value must be between 1 and 3 (inclusive)");
    CHECK(param->scaleFactor > 2, "Invalid scale-factor. Supports factor <= 2");
    CHECK(param->rc.qpMax < QP_MIN || param->rc.qpMax > QP_MAX_MAX,
        "qpmax exceeds supported range (0 to 69)");
    CHECK(param->rc.qpMin < QP_MIN || param->rc.qpMin > QP_MAX_MAX,
        "qpmin exceeds supported range (0 to 69)");
    CHECK(param->log2MaxPocLsb < 4 || param->log2MaxPocLsb > 16,
        "Supported range for log2MaxPocLsb is 4 to 16");
    CHECK(param->bCTUInfo < 0 || (param->bCTUInfo != 0 && param->bCTUInfo != 1 && param->bCTUInfo != 2 && param->bCTUInfo != 4 && param->bCTUInfo != 6) || param->bCTUInfo > 6,
        "Supported values for bCTUInfo are 0, 1, 2, 4, 6");
    CHECK(param->interRefine > 3 || param->interRefine < 0,
        "Invalid refine-inter value, refine-inter levels 0 to 3 supported");
    CHECK(param->intraRefine > 4 || param->intraRefine < 0,
        "Invalid refine-intra value, refine-intra levels 0 to 3 supported");
    CHECK(param->ctuDistortionRefine < 0 || param->ctuDistortionRefine > 1,
        "Invalid refine-ctu-distortion value, must be either 0 or 1");
    CHECK(param->maxAUSizeFactor < 0.5 || param->maxAUSizeFactor > 1.0,
        "Supported factor for controlling max AU size is from 0.5 to 1");
    CHECK((param->dolbyProfile != 0) && (param->dolbyProfile != 50) && (param->dolbyProfile != 81) && (param->dolbyProfile != 82) && (param->dolbyProfile != 84),
        "Unsupported Dolby Vision profile, only profile 5, profile 8.1, profile 8.2 and profile 8.4 enabled");
    CHECK(param->dupThreshold < 1 || 99 < param->dupThreshold,
        "Invalid frame-duplication threshold. Value must be between 1 and 99.");
    if (param->dolbyProfile)
    {
        CHECK((param->rc.vbvMaxBitrate <= 0 || param->rc.vbvBufferSize <= 0), "Dolby Vision requires VBV settings to enable HRD.\n");
        CHECK((param->internalBitDepth != 10), "Dolby Vision profile - 5, profile - 8.1, profile - 8.2 and profile - 8.4 are Main10 only\n");
        CHECK((param->internalCsp != X265_CSP_I420), "Dolby Vision profile - 5, profile - 8.1, profile - 8.2 and profile - 8.4 requires YCbCr 4:2:0 color space\n");
        if (param->dolbyProfile == 81)
            CHECK(param->masteringDisplayColorVolume[0] == 0, "Dolby Vision profile - 8.1 requires Mastering display color volume information\n");
    }
    if (param->bField && param->interlaceMode)
    {
        CHECK( (param->bFrameAdaptive==0), "Adaptive B-frame decision method should be closed for field feature.\n" );
        // to do
    }
    CHECK(param->selectiveSAO < 0 || param->selectiveSAO > 4,
        "Invalid SAO tune level. Value must be between 0 and 4 (inclusive)");
    if (param->bEnableSceneCutAwareQp)
    {
        if (param->bEnableSceneCutAwareQp != FORWARD && !param->rc.bStatRead)
        {
            param->bEnableSceneCutAwareQp = 0;
            x265_log(param, X265_LOG_WARNING, "Disabling Scenecut Aware Frame Quantizer Selection since single pass only works with Forward masking\n");
        }
        else
        {
            CHECK(param->bEnableSceneCutAwareQp < 0 || param->bEnableSceneCutAwareQp > 3,
            "Invalid masking direction. Value must be between 0 and 3(inclusive)");
            for (int i = 0; i < 6; i++)
            {
                CHECK(param->fwdScenecutWindow[i] < 0 || param->fwdScenecutWindow[i] > 1000,
                    "Invalid forward scenecut Window duration. Value must be between 0 and 1000(inclusive)");
                CHECK(param->fwdRefQpDelta[i] < 0 || param->fwdRefQpDelta[i] > 20,
                    "Invalid fwdRefQpDelta value. Value must be between 0 and 20 (inclusive)");
                CHECK(param->fwdNonRefQpDelta[i] < 0 || param->fwdNonRefQpDelta[i] > 20,
                    "Invalid fwdNonRefQpDelta value. Value must be between 0 and 20 (inclusive)");

                CHECK(param->bwdScenecutWindow[i] < 0 || param->bwdScenecutWindow[i] > 1000,
                    "Invalid backward scenecut Window duration. Value must be between 0 and 1000(inclusive)");
                CHECK(param->bwdRefQpDelta[i] < -1 || param->bwdRefQpDelta[i] > 20,
                    "Invalid bwdRefQpDelta value. Value must be between 0 and 20 (inclusive)");
                CHECK(param->bwdNonRefQpDelta[i] < -1 || param->bwdNonRefQpDelta[i] > 20,
                    "Invalid bwdNonRefQpDelta value. Value must be between 0 and 20 (inclusive)");
            }
        }
    }
    if (param->bEnableHME)
    {
        for (int level = 0; level < 3; level++)
            CHECK(param->hmeRange[level] < 0 || param->hmeRange[level] >= 32768,
                "Search Range for HME levels must be between 0 and 32768");
    }
#if !X86_64 && !X265_ARCH_ARM64 && !X265_ARCH_RISCV64
    CHECK(param->searchMethod == X265_SEA && (param->sourceWidth > 840 || param->sourceHeight > 480),
        "SEA motion search does not support resolutions greater than 480p in 32 bit build");
#endif

    if (strlen(param->masteringDisplayColorVolume) || param->maxFALL || param->maxCLL)
        param->bEmitHDR10SEI = 1;

    bool isSingleSEI = (param->bRepeatHeaders
                     || param->bEmitHRDSEI
                     || param->bEmitInfoSEI
                     || param->bEmitHDR10SEI
                     || param->bEmitIDRRecoverySEI
                   || !!param->interlaceMode
                     || param->preferredTransferCharacteristics > 1
                     || strlen(param->toneMapFile)
                     || strlen(param->naluFile));

    if (!isSingleSEI && param->bSingleSeiNal)
    {
        param->bSingleSeiNal = 0;
        x265_log(param, X265_LOG_WARNING, "None of the SEI messages are enabled. Disabling Single SEI NAL\n");
    }
    CHECK(param->confWinRightOffset < 0, "Conformance Window Right Offset must be 0 or greater");
    CHECK(param->confWinBottomOffset < 0, "Conformance Window Bottom Offset must be 0 or greater");
    CHECK(param->decoderVbvMaxRate < 0, "Invalid Decoder Vbv Maxrate. Value can not be less than zero");
    if (param->bliveVBV2pass)
    {
        CHECK((param->rc.bStatRead == 0), "Live VBV in multi pass option requires rate control 2 pass to be enabled");
        if ((param->rc.vbvMaxBitrate <= 0 || param->rc.vbvBufferSize <= 0))
        {
            param->bliveVBV2pass = 0;
            x265_log(param, X265_LOG_WARNING, "Live VBV enabled without VBV settings.Disabling live VBV in 2 pass\n");
        }
    }
    CHECK(param->rc.dataShareMode != X265_SHARE_MODE_FILE && param->rc.dataShareMode != X265_SHARE_MODE_SHAREDMEM, "Invalid data share mode. It must be one of the X265_DATA_SHARE_MODES enum values\n" );
#if ENABLE_ALPHA
    if (param->bEnableAlpha)
    {
        CHECK((param->internalCsp != X265_CSP_I420), "Alpha encode supported only with i420a colorspace");
        CHECK((param->internalBitDepth > 10), "BitDepthConstraint must be 8 and 10  for Scalable main profile");
        CHECK((param->analysisMultiPassDistortion || param->analysisMultiPassRefine), "Alpha encode doesnot support multipass feature");
        CHECK((strlen(param->analysisSave) || strlen(param->analysisLoad)), "Alpha encode doesnot support analysis save and load  feature");
    }
#endif
#if ENABLE_MULTIVIEW
    CHECK((param->numViews > 2), "Multi-View Encoding currently support only 2 views");
    if (param->numViews > 1)
    {
        CHECK(param->internalBitDepth != 8, "BitDepthConstraint must be 8 for Multiview main profile");
        CHECK(param->analysisMultiPassDistortion || param->analysisMultiPassRefine, "Multiview encode doesnot support multipass feature");
        CHECK(strlen(param->analysisSave) || strlen(param->analysisLoad), "Multiview encode doesnot support analysis save and load feature");
        CHECK(param->isAbrLadderEnable, "Multiview encode and Abr-Ladder feature can't be enabled together");
    }
#endif
#if ENABLE_SCC_EXT
    bool checkValid = false;

    if (!!param->bEnableSCC)
    {
        checkValid = param->keyframeMax <= 1 || param->totalFrames == 1;
        if (checkValid)     x265_log(param, X265_LOG_WARNING, "intra constraint flag must be 0 for SCC profiles. Disabling SCC  \n");
        checkValid = param->totalFrames == 1;
        if (checkValid)     x265_log(param, X265_LOG_WARNING, "one-picture-only constraint flag shall be 0 for SCC profiles. Disabling SCC  \n");
        const uint32_t bitDepthIdx = (param->internalBitDepth == 8 ? 0 : (param->internalBitDepth == 10 ? 1 : (param->internalBitDepth == 12 ? 2 : (param->internalBitDepth == 16 ? 3 : 4))));
        const uint32_t chromaFormatIdx = uint32_t(param->internalCsp);
        checkValid = !((bitDepthIdx > 2 || chromaFormatIdx > 3) ? false : (validSCCProfileNames[0][bitDepthIdx][chromaFormatIdx] != NONE));
        if (checkValid)     x265_log(param, X265_LOG_WARNING, "Invalid intra constraint flag, bit depth constraint flag and chroma format constraint flag combination for a RExt profile. Disabling SCC \n");
        if (checkValid)
            param->bEnableSCC = 0;
    }
    if (!!param->bEnableSCC)
    {
        if (param->bEnableRdRefine && param->bDynamicRefine)
        {
            param->bEnableRdRefine = 0;
            x265_log(param, X265_LOG_WARNING, "Disabling rd-refine as it can not be used with scc and dynamic-refine\n");
        }
        if (param->bEnableRdRefine && param->interRefine > 0)
        {
            param->bEnableRdRefine = 0;
            x265_log(param, X265_LOG_WARNING, "Disabling rd-refine as it can not be used with scc and inter-refine\n");
        }
    }
    CHECK(!!param->bEnableSCC&& param->rdLevel != 6, "Enabling scc extension in x265 requires rdlevel of 6 ");
#endif

    return check_failed;
}

void x265_param_apply_fastfirstpass(x265_param* param)
{
    /* Set faster options in case of turbo firstpass */
    if (param->rc.bStatWrite && !param->rc.bStatRead)
    {
        param->maxNumReferences = 1;
        param->maxNumMergeCand = 1;
        param->bEnableRectInter = 0;
        param->bEnableFastIntra = 1;
        param->bEnableAMP = 0;
        param->searchMethod = X265_DIA_SEARCH;
        param->subpelRefine = X265_MIN(2, param->subpelRefine);
        param->bEnableEarlySkip = 1;
        param->rdLevel = X265_MIN(2, param->rdLevel);
    }
}

static void appendtool(x265_param* param, char* buf, size_t size, const char* toolstr)
{
    static const int overhead = (int)strlen("x265 [info]: tools: ");

    if (strlen(buf) + strlen(toolstr) + overhead >= size)
    {
        x265_log(param, X265_LOG_INFO, "tools:%s\n", buf);
        snprintf(buf, size, " %s", toolstr);
    }
    else
    {
        strcat(buf, " ");
        strcat(buf, toolstr);
    }
}

void x265_print_params(x265_param* param)
{
    setlocale(LC_NUMERIC, "en_US.utf8");
    if (param->logLevel < X265_LOG_INFO)
        return;

    if (param->interlaceMode)
        x265_log(param, X265_LOG_INFO, "Interlaced field inputs                 : %s\n", x265_interlace_names[param->interlaceMode]);

        x265_log(param, X265_LOG_INFO, "Coding QT: max CU size, min CU size     : %d / %d\n", param->maxCUSize, param->minCUSize);

    if (param->bThreadedME)
        x265_log(param, X265_LOG_INFO, "ThreadedME: task block / buf rows       : %d / %d\n", param->tmeTaskBlockSize, param->tmeNumBufferRows);

    x265_log(param, X265_LOG_INFO, "Residual QT: max TU size, max depth     : %d / %d inter / %d intra\n",
             param->maxTUSize, param->tuQTMaxInterDepth, param->tuQTMaxIntraDepth);

    if (param->bEnableHME)
        x265_log(param, X265_LOG_INFO, "HME L0,1,2 / range / subpel / merge     : %s, %s, %s / %d / %d / %d\n",
            x265_motion_est_names[param->hmeSearchMethod[0]], x265_motion_est_names[param->hmeSearchMethod[1]], x265_motion_est_names[param->hmeSearchMethod[2]], param->searchRange, param->subpelRefine, param->maxNumMergeCand);
    else
        x265_log(param, X265_LOG_INFO, "ME / range / subpel / merge             : %s / %d / %d / %d\n",
            x265_motion_est_names[param->searchMethod], param->searchRange, param->subpelRefine, param->maxNumMergeCand);

    if (param->scenecutThreshold && param->keyframeMax != INT_MAX)
        x265_log(param, X265_LOG_INFO, "Keyframe min / max / scenecut / bias    : %d / %d / %d / %.2lf \n",
                 param->keyframeMin, param->keyframeMax, param->scenecutThreshold, param->scenecutBias * 100);
    else if (param->bHistBasedSceneCut && param->keyframeMax != INT_MAX)
        x265_log(param, X265_LOG_INFO, "Keyframe min / max / scenecut           : %d / %d / %d\n",
                 param->keyframeMin, param->keyframeMax, param->bHistBasedSceneCut);
    else if (param->keyframeMax == INT_MAX)
        x265_log(param, X265_LOG_INFO, "Keyframe min / max / scenecut           : disabled\n");

    if (param->cbQpOffset || param->crQpOffset)
        x265_log(param, X265_LOG_INFO, "Cb/Cr QP Offset                         : %d / %d\n", param->cbQpOffset, param->crQpOffset);

    if (param->rdPenalty)
        x265_log(param, X265_LOG_INFO, "Intra 32x32 TU penalty type             : %d\n", param->rdPenalty);

    x265_log(param, X265_LOG_INFO, "Lookahead / bframes / badapt            : %d / %d / %d\n", param->lookaheadDepth, param->bframes, param->bFrameAdaptive);
    x265_log(param, X265_LOG_INFO, "b-pyramid / weightp / weightb           : %d / %d / %d\n",
             param->bBPyramid, param->bEnableWeightedPred, param->bEnableWeightedBiPred);
    x265_log(param, X265_LOG_INFO, "References / ref-limit  cu / depth      : %d / %s / %s\n",
             param->maxNumReferences, (param->limitReferences & X265_REF_LIMIT_CU) ? "on" : "off",
             (param->limitReferences & X265_REF_LIMIT_DEPTH) ? "on" : "off");

    if (param->rc.aqMode && !param->rc.bAutoAq)
        x265_log(param, X265_LOG_INFO, "AQ: mode / str / qg-size / cu-tree      : %d / %0.1f / %d / %d\n", param->rc.aqMode,
                 param->rc.aqStrength, param->rc.qgSize, param->rc.cuTree);
    else if (param->rc.bAutoAq)
        x265_log(param, X265_LOG_INFO, "AQ: mode / str / qg-size / cu-tree      : auto / %0.1f / %d / %d\n", param->rc.aqStrength, param->rc.qgSize, param->rc.cuTree);

    if (param->bLossless)
        x265_log(param, X265_LOG_INFO, "Rate Control                            : Lossless\n");
    else switch (param->rc.rateControlMode)
    {
    case X265_RC_ABR:
        x265_log(param, X265_LOG_INFO, "Rate Control / qCompress                : ABR-%d kbps / %0.2f\n", param->rc.bitrate, param->rc.qCompress); break;
    case X265_RC_CQP:
        x265_log(param, X265_LOG_INFO, "Rate Control                            : CQP-%d\n", param->rc.qp); break;
    case X265_RC_CRF:
        x265_log(param, X265_LOG_INFO, "Rate Control / qCompress                : CRF-%0.1f / %0.2f\n", param->rc.rfConstant, param->rc.qCompress); break;
    }

    if (param->rc.vbvBufferSize)
    {
        if (param->vbvBufferEnd)
            x265_log(param, X265_LOG_INFO, "VBV buffer / maxrate / init / end / adj : %d / %d / %.3f / %.3f / %.3f\n",
            param->rc.vbvBufferSize, param->rc.vbvMaxBitrate, param->rc.vbvBufferInit, param->vbvBufferEnd, param->vbvEndFrameAdjust);
        else
            x265_log(param, X265_LOG_INFO, "VBV buffer / maxrate / init             : %d / %d / %.3f\n",
            param->rc.vbvBufferSize, param->rc.vbvMaxBitrate, param->rc.vbvBufferInit);
    }

    char buf[80] = { 0 };
    char tmp[40];
#define TOOLOPT(FLAG, STR) if (FLAG) appendtool(param, buf, sizeof(buf), STR);
#define TOOLVAL(VAL, STR)  if (VAL) { snprintf(tmp, sizeof(tmp), STR, VAL); appendtool(param, buf, sizeof(buf), tmp); }
    TOOLOPT(param->bEnableRectInter, "rect");
    TOOLOPT(param->bEnableAMP, "amp");
    TOOLOPT(param->limitModes, "limit-modes");
    TOOLVAL(param->rdLevel, "rd=%d");
    TOOLVAL(param->dynamicRd, "dynamic-rd=%.2f");
    TOOLOPT(param->bSsimRd, "ssim-rd");
    TOOLVAL(param->psyRd, "psy-rd=%.2lf");
    TOOLVAL(param->rdoqLevel, "rdoq=%d");
    TOOLVAL(param->psyRdoq, "psy-rdoq=%.2lf");
    TOOLOPT(param->bEnableRdRefine, "rd-refine");
    TOOLOPT(param->bEnableEarlySkip, "early-skip");
    TOOLVAL(param->recursionSkipMode, "rskip mode=%d");
    if (param->recursionSkipMode == EDGE_BASED_RSKIP)
        TOOLVAL(param->edgeVarThreshold, "rskip-edge-threshold=%.2f");
    TOOLOPT(param->bEnableSplitRdSkip, "splitrd-skip");
    TOOLVAL(param->noiseReductionIntra, "nr-intra=%d");
    TOOLVAL(param->noiseReductionInter, "nr-inter=%d");
    TOOLOPT(param->bEnableTSkipFast, "tskip-fast");
    TOOLOPT(!param->bEnableTSkipFast && param->bEnableTransformSkip, "tskip");
    TOOLVAL(param->limitTU , "limit-tu=%d");
    TOOLOPT(param->bCULossless, "cu-lossless");
    TOOLOPT(param->bEnableSignHiding, "signhide");
    TOOLOPT(param->bEnableTemporalMvp, "tmvp");
    TOOLOPT(param->bEnableConstrainedIntra, "cip");
    TOOLOPT(param->bIntraInBFrames, "b-intra");
    TOOLOPT(param->bEnableFastIntra, "fast-intra");
    TOOLOPT(param->bEnableStrongIntraSmoothing, "strong-intra-smoothing");
    TOOLVAL(param->lookaheadSlices, "lslices=%d");
    TOOLVAL(param->lookaheadThreads, "lthreads=%d")
    TOOLVAL(param->bCTUInfo, "ctu-info=%d");
    if (param->bAnalysisType == AVC_INFO)
    {
        TOOLOPT(param->bAnalysisType, "refine-analysis-type=avc");
    }
    else if (param->bAnalysisType == HEVC_INFO)
        TOOLOPT(param->bAnalysisType, "refine-analysis-type=hevc");
    TOOLOPT(param->bDynamicRefine, "dynamic-refine");
    if (param->maxSlices > 1)
        TOOLVAL(param->maxSlices, "slices=%d");
    if (param->bEnableLoopFilter)
    {
        if (param->deblockingFilterBetaOffset || param->deblockingFilterTCOffset)
        {
            snprintf(tmp, sizeof(tmp), "deblock(tC=%d:B=%d)", param->deblockingFilterTCOffset, param->deblockingFilterBetaOffset);
            appendtool(param, buf, sizeof(buf), tmp);
        }
        else
            TOOLOPT(param->bEnableLoopFilter, "deblock");
    }
    TOOLOPT(param->bSaoNonDeblocked, "sao-non-deblock");
    TOOLOPT(!param->bSaoNonDeblocked && param->bEnableSAO, "sao");
    if (param->selectiveSAO && param->selectiveSAO != 4)
        TOOLOPT(param->selectiveSAO, "selective-sao");
    TOOLOPT(param->rc.bStatWrite, "stats-write");
    TOOLOPT(param->rc.bStatRead,  "stats-read");
    TOOLOPT(param->bSingleSeiNal, "single-sei");
#if ENABLE_ALPHA
    TOOLOPT(param->numScalableLayers > 1, "alpha");
#endif
#if ENABLE_MULTIVIEW
    TOOLOPT(param->numViews > 1, "multi-view");
#endif
#if ENABLE_HDR10_PLUS
    TOOLOPT(param->toneMapFile != NULL, "dhdr10-info");
#endif
    if(param->bEnableTemporalFilter)
        TOOLOPT(param->bEnableTemporalFilter, "mcstf");
    x265_log(param, X265_LOG_INFO, "tools:%s\n", buf);
    fflush(stderr);
}

char *x265_param2string(x265_param* p, int padx, int pady)
{
    char *buf, *s;
    size_t bufSize = 4000 + p->rc.zoneCount * 64;
    if (strlen(p->numaPools))
        bufSize += strlen(p->numaPools);
    if (strlen(p->masteringDisplayColorVolume))
        bufSize += strlen(p->masteringDisplayColorVolume);
    if (strlen(p->videoSignalTypePreset))
        bufSize += strlen(p->videoSignalTypePreset);

    buf = s = X265_MALLOC(char, bufSize);
    if (!buf)
        return NULL;
#define BOOL(param, cliopt) \
    s += snprintf(s, bufSize - (s - buf), " %s", (param) ? cliopt : "no-" cliopt);

    s += snprintf(s, bufSize - (s - buf), "cpuid=%d", p->cpuid);
    s += snprintf(s, bufSize - (s - buf), " frame-threads=%d", p->frameNumThreads);
    if (strlen(p->numaPools))
        s += snprintf(s, bufSize - (s - buf), " numa-pools=%s", p->numaPools);
    BOOL(p->bEnableWavefront, "wpp");
    BOOL(p->bDistributeModeAnalysis, "pmode");
    BOOL(p->bDistributeMotionEstimation, "pme");
    BOOL(p->bEnablePsnr, "psnr");
    BOOL(p->bEnableSsim, "ssim");
    s += snprintf(s, bufSize - (s - buf), " log-level=%d", p->logLevel);
    if (strlen(p->csvfn))
        s += snprintf(s, bufSize - (s - buf), " csv csv-log-level=%d", p->csvLogLevel);
    s += snprintf(s, bufSize - (s - buf), " bitdepth=%d", p->internalBitDepth);
    s += snprintf(s, bufSize - (s - buf), " input-csp=%d", p->internalCsp);
    s += snprintf(s, bufSize - (s - buf), " fps=%u/%u", p->fpsNum, p->fpsDenom);
    s += snprintf(s, bufSize - (s - buf), " input-res=%dx%d", p->sourceWidth - padx, p->sourceHeight - pady);
    s += snprintf(s, bufSize - (s - buf), " interlace=%d", p->interlaceMode);
    s += snprintf(s, bufSize - (s - buf), " total-frames=%d", p->totalFrames);
    if (p->chunkStart)
        s += snprintf(s, bufSize - (s - buf), " chunk-start=%d", p->chunkStart);
    if (p->chunkEnd)
        s += snprintf(s, bufSize - (s - buf), " chunk-end=%d", p->chunkEnd);
    s += snprintf(s, bufSize - (s - buf), " level-idc=%d", p->levelIdc);
    s += snprintf(s, bufSize - (s - buf), " high-tier=%d", p->bHighTier);
    s += snprintf(s, bufSize - (s - buf), " uhd-bd=%d", p->uhdBluray);
    s += snprintf(s, bufSize - (s - buf), " ref=%d", p->maxNumReferences);
    BOOL(p->bAllowNonConformance, "allow-non-conformance");
    BOOL(p->bRepeatHeaders, "repeat-headers");
    BOOL(p->bAnnexB, "annexb");
    BOOL(p->bEnableAccessUnitDelimiters, "aud");
    BOOL(p->bEnableEndOfBitstream, "eob");
    BOOL(p->bEnableEndOfSequence, "eos");
    BOOL(p->bEmitHRDSEI, "hrd");
    BOOL(p->bEmitInfoSEI, "info");
    s += snprintf(s, bufSize - (s - buf), " hash=%d", p->decodedPictureHashSEI);
    s += snprintf(s, bufSize - (s - buf), " temporal-layers=%d", p->bEnableTemporalSubLayers);
    BOOL(p->bOpenGOP, "open-gop");
    s += snprintf(s, bufSize - (s - buf), " min-keyint=%d", p->keyframeMin);
    s += snprintf(s, bufSize - (s - buf), " keyint=%d", p->keyframeMax);
    s += snprintf(s, bufSize - (s - buf), " gop-lookahead=%d", p->gopLookahead);
    s += snprintf(s, bufSize - (s - buf), " bframes=%d", p->bframes);
    s += snprintf(s, bufSize - (s - buf), " b-adapt=%d", p->bFrameAdaptive);
    BOOL(p->bBPyramid, "b-pyramid");
    s += snprintf(s, bufSize - (s - buf), " bframe-bias=%d", p->bFrameBias);
    s += snprintf(s, bufSize - (s - buf), " rc-lookahead=%d", p->lookaheadDepth);
    s += snprintf(s, bufSize - (s - buf), " lookahead-slices=%d", p->lookaheadSlices);
    s += snprintf(s, bufSize - (s - buf), " scenecut=%d", p->scenecutThreshold);
    BOOL(p->bHistBasedSceneCut, "hist-scenecut");
    s += snprintf(s, bufSize - (s - buf), " radl=%d", p->radl);
    BOOL(p->bEnableHRDConcatFlag, "splice");
    BOOL(p->bIntraRefresh, "intra-refresh");
    s += snprintf(s, bufSize - (s - buf), " ctu=%d", p->maxCUSize);
    s += snprintf(s, bufSize - (s - buf), " min-cu-size=%d", p->minCUSize);
    BOOL(p->bEnableRectInter, "rect");
    BOOL(p->bEnableAMP, "amp");
    s += snprintf(s, bufSize - (s - buf), " max-tu-size=%d", p->maxTUSize);
    s += snprintf(s, bufSize - (s - buf), " tu-inter-depth=%d", p->tuQTMaxInterDepth);
    s += snprintf(s, bufSize - (s - buf), " tu-intra-depth=%d", p->tuQTMaxIntraDepth);
    s += snprintf(s, bufSize - (s - buf), " limit-tu=%d", p->limitTU);
    s += snprintf(s, bufSize - (s - buf), " rdoq-level=%d", p->rdoqLevel);
    s += snprintf(s, bufSize - (s - buf), " dynamic-rd=%.2f", p->dynamicRd);
    BOOL(p->bSsimRd, "ssim-rd");
    BOOL(p->bEnableSignHiding, "signhide");
    BOOL(p->bEnableTransformSkip, "tskip");
    s += snprintf(s, bufSize - (s - buf), " nr-intra=%d", p->noiseReductionIntra);
    s += snprintf(s, bufSize - (s - buf), " nr-inter=%d", p->noiseReductionInter);
    BOOL(p->bEnableConstrainedIntra, "constrained-intra");
    BOOL(p->bEnableStrongIntraSmoothing, "strong-intra-smoothing");
    s += snprintf(s, bufSize - (s - buf), " max-merge=%d", p->maxNumMergeCand);
    s += snprintf(s, bufSize - (s - buf), " limit-refs=%d", p->limitReferences);
    BOOL(p->limitModes, "limit-modes");
    s += snprintf(s, bufSize - (s - buf), " me=%d", p->searchMethod);
    s += snprintf(s, bufSize - (s - buf), " subme=%d", p->subpelRefine);
    s += snprintf(s, bufSize - (s - buf), " merange=%d", p->searchRange);
    BOOL(p->bEnableTemporalMvp, "temporal-mvp");
    BOOL(p->bEnableFrameDuplication, "frame-dup");
    if(p->bEnableFrameDuplication)
        s += snprintf(s, bufSize - (s - buf), " dup-threshold=%d", p->dupThreshold);
    BOOL(p->bEnableHME, "hme");
    if (p->bEnableHME)
    {
        s += snprintf(s, bufSize - (s - buf), " Level 0,1,2=%d,%d,%d", p->hmeSearchMethod[0], p->hmeSearchMethod[1], p->hmeSearchMethod[2]);
        s += snprintf(s, bufSize - (s - buf), " merange L0,L1,L2=%d,%d,%d", p->hmeRange[0], p->hmeRange[1], p->hmeRange[2]);
    }
    BOOL(p->bEnableWeightedPred, "weightp");
    BOOL(p->bEnableWeightedBiPred, "weightb");
    BOOL(p->bSourceReferenceEstimation, "analyze-src-pics");
    BOOL(p->bEnableLoopFilter, "deblock");
    if (p->bEnableLoopFilter)
        s += snprintf(s, bufSize - (s - buf), "=%d:%d", p->deblockingFilterTCOffset, p->deblockingFilterBetaOffset);
    BOOL(p->bEnableSAO, "sao");
    BOOL(p->bSaoNonDeblocked, "sao-non-deblock");
    s += snprintf(s, bufSize - (s - buf), " rd=%d", p->rdLevel);
    s += snprintf(s, bufSize - (s - buf), " selective-sao=%d", p->selectiveSAO);
    BOOL(p->bEnableEarlySkip, "early-skip");
    BOOL(p->recursionSkipMode, "rskip");
    if (p->recursionSkipMode == EDGE_BASED_RSKIP)
        s += snprintf(s, bufSize - (s - buf), " rskip-edge-threshold=%f", p->edgeVarThreshold);

    BOOL(p->bEnableFastIntra, "fast-intra");
    BOOL(p->bEnableTSkipFast, "tskip-fast");
    BOOL(p->bCULossless, "cu-lossless");
    BOOL(p->bIntraInBFrames, "b-intra");
    BOOL(p->bEnableSplitRdSkip, "splitrd-skip");
    s += snprintf(s, bufSize - (s - buf), " rdpenalty=%d", p->rdPenalty);
    s += snprintf(s, bufSize - (s - buf), " psy-rd=%.2f", p->psyRd);
    s += snprintf(s, bufSize - (s - buf), " psy-rdoq=%.2f", p->psyRdoq);
    BOOL(p->bEnableRdRefine, "rd-refine");
    BOOL(p->bIntraRDRefine, "intra-rd-refine");
    BOOL(p->bLossless, "lossless");
    s += snprintf(s, bufSize - (s - buf), " cbqpoffs=%d", p->cbQpOffset);
    s += snprintf(s, bufSize - (s - buf), " crqpoffs=%d", p->crQpOffset);
    s += snprintf(s, bufSize - (s - buf), " rc=%s", p->rc.rateControlMode == X265_RC_ABR ? (
         p->rc.bitrate == p->rc.vbvMaxBitrate ? "cbr" : "abr")
         : p->rc.rateControlMode == X265_RC_CRF ? "crf" : "cqp");
    if (p->rc.rateControlMode == X265_RC_ABR || p->rc.rateControlMode == X265_RC_CRF)
    {
        if (p->rc.rateControlMode == X265_RC_CRF)
            s += snprintf(s, bufSize - (s - buf), " crf=%.1f", p->rc.rfConstant);
        else
            s += snprintf(s, bufSize - (s - buf), " bitrate=%d", p->rc.bitrate);
        s += snprintf(s, bufSize - (s - buf), " qcomp=%.2f qpstep=%d", p->rc.qCompress, p->rc.qpStep);
        s += snprintf(s, bufSize - (s - buf), " stats-write=%d", p->rc.bStatWrite);
        s += snprintf(s, bufSize - (s - buf), " stats-read=%d", p->rc.bStatRead);
        if (p->rc.bStatRead)
            s += snprintf(s, bufSize - (s - buf), " cplxblur=%.1f qblur=%.1f",
            p->rc.complexityBlur, p->rc.qblur);
        if (p->rc.bStatWrite && !p->rc.bStatRead)
            BOOL(p->rc.bEnableSlowFirstPass, "slow-firstpass");
        if (p->rc.vbvBufferSize)
        {
            s += snprintf(s, bufSize - (s - buf), " vbv-maxrate=%d vbv-bufsize=%d vbv-init=%.1f min-vbv-fullness=%.1f max-vbv-fullness=%.1f",
                p->rc.vbvMaxBitrate, p->rc.vbvBufferSize, p->rc.vbvBufferInit, p->minVbvFullness, p->maxVbvFullness);
            if (p->vbvBufferEnd)
                s += snprintf(s, bufSize - (s - buf), " vbv-end=%.1f vbv-end-fr-adj=%.1f", p->vbvBufferEnd, p->vbvEndFrameAdjust);
            if (p->rc.rateControlMode == X265_RC_CRF)
                s += snprintf(s, bufSize - (s - buf), " crf-max=%.1f crf-min=%.1f", p->rc.rfConstantMax, p->rc.rfConstantMin);
        }
    }
    else if (p->rc.rateControlMode == X265_RC_CQP)
        s += snprintf(s, bufSize - (s - buf), " qp=%d", p->rc.qp);
    if (!(p->rc.rateControlMode == X265_RC_CQP && p->rc.qp == 0))
    {
        s += snprintf(s, bufSize - (s - buf), " ipratio=%.2f", p->rc.ipFactor);
        if (p->bframes)
            s += snprintf(s, bufSize - (s - buf), " pbratio=%.2f", p->rc.pbFactor);
    }
    if (!p->rc.bAutoAq)
        s += snprintf(s, bufSize - (s - buf), " aq-mode=%d", p->rc.aqMode);
    if (p->rc.bAutoAq)
        BOOL(p->rc.bAutoAq, "auto-aq");
    s += snprintf(s, bufSize - (s - buf), " aq-strength=%.2f", p->rc.aqStrength);
    s += snprintf(s, bufSize - (s - buf), " aq-bias-strength=%.2f", p->rc.aqBiasStrength);
    BOOL(p->rc.cuTree, "cutree");
    s += snprintf(s, bufSize - (s - buf), " zone-count=%d", p->rc.zoneCount);
    if (p->rc.zoneCount)
    {
        for (int i = 0; i < p->rc.zoneCount; ++i)
        {
            s += snprintf(s, bufSize - (s - buf), " zones: start-frame=%d end-frame=%d",
                 p->rc.zones[i].startFrame, p->rc.zones[i].endFrame);
            if (p->rc.zones[i].bForceQp)
                s += snprintf(s, bufSize - (s - buf), " qp=%d", p->rc.zones[i].qp);
            else
                s += snprintf(s, bufSize - (s - buf), " bitrate-factor=%f", p->rc.zones[i].bitrateFactor);
        }
    }
    BOOL(p->rc.bStrictCbr, "strict-cbr");
    s += snprintf(s, bufSize - (s - buf), " qg-size=%d", p->rc.qgSize);
    BOOL(p->rc.bEnableGrain, "rc-grain");
    s += snprintf(s, bufSize - (s - buf), " qpmax=%d qpmin=%d", p->rc.qpMax, p->rc.qpMin);
    BOOL(p->rc.bEnableConstVbv, "const-vbv");
    s += snprintf(s, bufSize - (s - buf), " sar=%d", p->vui.aspectRatioIdc);
    if (p->vui.aspectRatioIdc == X265_EXTENDED_SAR)
        s += snprintf(s, bufSize - (s - buf), " sar-width : sar-height=%d:%d", p->vui.sarWidth, p->vui.sarHeight);
    s += snprintf(s, bufSize - (s - buf), " overscan=%d", p->vui.bEnableOverscanInfoPresentFlag);
    if (p->vui.bEnableOverscanInfoPresentFlag)
        s += snprintf(s, bufSize - (s - buf), " overscan-crop=%d", p->vui.bEnableOverscanAppropriateFlag);
    s += snprintf(s, bufSize - (s - buf), " videoformat=%d", p->vui.videoFormat);
    s += snprintf(s, bufSize - (s - buf), " range=%d", p->vui.bEnableVideoFullRangeFlag);
    s += snprintf(s, bufSize - (s - buf), " colorprim=%d", p->vui.colorPrimaries);
    s += snprintf(s, bufSize - (s - buf), " transfer=%d", p->vui.transferCharacteristics);
    s += snprintf(s, bufSize - (s - buf), " colormatrix=%d", p->vui.matrixCoeffs);
    s += snprintf(s, bufSize - (s - buf), " chromaloc=%d", p->vui.bEnableChromaLocInfoPresentFlag);
    if (p->vui.bEnableChromaLocInfoPresentFlag)
        s += snprintf(s, bufSize - (s - buf), " chromaloc-top=%d chromaloc-bottom=%d",
        p->vui.chromaSampleLocTypeTopField, p->vui.chromaSampleLocTypeBottomField);
    s += snprintf(s, bufSize - (s - buf), " display-window=%d", p->vui.bEnableDefaultDisplayWindowFlag);
    if (p->vui.bEnableDefaultDisplayWindowFlag)
        s += snprintf(s, bufSize - (s - buf), " left=%d top=%d right=%d bottom=%d",
        p->vui.defDispWinLeftOffset, p->vui.defDispWinTopOffset,
        p->vui.defDispWinRightOffset, p->vui.defDispWinBottomOffset);
    if (strlen(p->masteringDisplayColorVolume))
        s += snprintf(s, bufSize - (s - buf), " master-display=%s", p->masteringDisplayColorVolume);
    if (p->bEmitCLL)
        s += snprintf(s, bufSize - (s - buf), " cll=%hu,%hu", p->maxCLL, p->maxFALL);
    s += snprintf(s, bufSize - (s - buf), " min-luma=%hu", p->minLuma);
    s += snprintf(s, bufSize - (s - buf), " max-luma=%hu", p->maxLuma);
    s += snprintf(s, bufSize - (s - buf), " log2-max-poc-lsb=%d", p->log2MaxPocLsb);
    BOOL(p->bEmitVUITimingInfo, "vui-timing-info");
    BOOL(p->bEmitVUIHRDInfo, "vui-hrd-info");
    s += snprintf(s, bufSize - (s - buf), " slices=%d", p->maxSlices);
    BOOL(p->bOptQpPPS, "opt-qp-pps");
    BOOL(p->bOptRefListLengthPPS, "opt-ref-list-length-pps");
    BOOL(p->bMultiPassOptRPS, "multi-pass-opt-rps");
    s += snprintf(s, bufSize - (s - buf), " scenecut-bias=%.2f", p->scenecutBias);
    BOOL(p->bOptCUDeltaQP, "opt-cu-delta-qp");
    BOOL(p->bAQMotion, "aq-motion");
    BOOL(p->bEmitHDR10SEI, "hdr10");
    BOOL(p->bHDR10Opt, "hdr10-opt");
    BOOL(p->bDhdr10opt, "dhdr10-opt");
    BOOL(p->bEmitIDRRecoverySEI, "idr-recovery-sei");
    if (strlen(p->analysisSave))
        s += snprintf(s, bufSize - (s - buf), " analysis-save");
    if (strlen(p->analysisLoad))
        s += snprintf(s, bufSize - (s - buf), " analysis-load");
    s += snprintf(s, bufSize - (s - buf), " analysis-reuse-level=%d", p->analysisReuseLevel);
    s += snprintf(s, bufSize - (s - buf), " analysis-save-reuse-level=%d", p->analysisSaveReuseLevel);
    s += snprintf(s, bufSize - (s - buf), " analysis-load-reuse-level=%d", p->analysisLoadReuseLevel);
    s += snprintf(s, bufSize - (s - buf), " scale-factor=%d", p->scaleFactor);
    s += snprintf(s, bufSize - (s - buf), " refine-intra=%d", p->intraRefine);
    s += snprintf(s, bufSize - (s - buf), " refine-inter=%d", p->interRefine);
    s += snprintf(s, bufSize - (s - buf), " refine-mv=%d", p->mvRefine);
    s += snprintf(s, bufSize - (s - buf), " refine-ctu-distortion=%d", p->ctuDistortionRefine);
    BOOL(p->bLimitSAO, "limit-sao");
    s += snprintf(s, bufSize - (s - buf), " ctu-info=%d", p->bCTUInfo);
    BOOL(p->bLowPassDct, "lowpass-dct");
    s += snprintf(s, bufSize - (s - buf), " refine-analysis-type=%d", p->bAnalysisType);
    s += snprintf(s, bufSize - (s - buf), " copy-pic=%d", p->bCopyPicToFrame);
    s += snprintf(s, bufSize - (s - buf), " max-ausize-factor=%.1f", p->maxAUSizeFactor);
    BOOL(p->bDynamicRefine, "dynamic-refine");
    BOOL(p->bSingleSeiNal, "single-sei");
    BOOL(p->rc.hevcAq, "hevc-aq");
    BOOL(p->bEnableSvtHevc, "svt");
    BOOL(p->bField, "field");
    s += snprintf(s, bufSize - (s - buf), " qp-adaptation-range=%.2f", p->rc.qpAdaptationRange);
    s += snprintf(s, bufSize - (s - buf), " scenecut-aware-qp=%d", p->bEnableSceneCutAwareQp);
    if (p->bEnableSceneCutAwareQp)
        s += snprintf(s, bufSize - (s - buf), " fwd-scenecut-window=%d fwd-ref-qp-delta=%.2f fwd-nonref-qp-delta=%.2f bwd-scenecut-window=%d bwd-ref-qp-delta=%.2f bwd-nonref-qp-delta=%.2f", p->fwdMaxScenecutWindow, p->fwdRefQpDelta[0], p->fwdNonRefQpDelta[0], p->bwdMaxScenecutWindow, p->bwdRefQpDelta[0], p->bwdNonRefQpDelta[0]);
    s += snprintf(s, bufSize - (s - buf), " conformance-window-offsets right=%d bottom=%d", p->confWinRightOffset, p->confWinBottomOffset);
    s += snprintf(s, bufSize - (s - buf), " decoder-max-rate=%d", p->decoderVbvMaxRate);
    BOOL(p->bliveVBV2pass, "vbv-live-multi-pass");
    if (p->filmGrain)
        s += snprintf(s, bufSize - (s - buf), " film-grain=%s", p->filmGrain); // Film grain characteristics model filename
    if (p->aomFilmGrain)
        s += snprintf(s, bufSize - (s - buf), " aom-film-grain=%s", p->aomFilmGrain);
    BOOL(p->bEnableTemporalFilter, "mcstf");
#if ENABLE_ALPHA
    BOOL(p->bEnableAlpha, "alpha");
#endif
#if ENABLE_MULTIVIEW
    s += snprintf(s, bufSize - (s - buf), " num-views=%d", p->numViews);
    s += snprintf(s, bufSize - (s - buf), " format=%d", p->format);
#endif
#if ENABLE_SCC_EXT
    s += snprintf(s, bufSize - (s - buf), "scc=%d", p->bEnableSCC);
#endif
    BOOL(p->bEnableSBRC, "sbrc");
    BOOL(p->bConfigRCFrame, "frame-rc");
#undef BOOL
    return buf;
}

bool parseLambdaFile(x265_param* param)
{
    if (!strlen(param->rc.lambdaFileName))
        return false;

    FILE *lfn = x265_fopen(param->rc.lambdaFileName, "r");
    if (!lfn)
    {
        x265_log_file(param, X265_LOG_ERROR, "unable to read lambda file <%s>\n", param->rc.lambdaFileName);
        return true;
    }

    char line[2048];
    char *toksave = NULL, *tok = NULL, *buf = NULL;

    for (int t = 0; t < 3; t++)
    {
        double *table = t ? x265_lambda2_tab : x265_lambda_tab;

        for (int i = 0; i < QP_MAX_MAX + 1; i++)
        {
            double value;

            do
            {
                if (!tok)
                {
                    /* consume a line of text file */
                    if (!fgets(line, sizeof(line), lfn))
                    {
                        fclose(lfn);

                        if (t < 2)
                        {
                            x265_log(param, X265_LOG_ERROR, "lambda file is incomplete\n");
                            return true;
                        }
                        else
                            return false;
                    }

                    /* truncate at first hash */
                    char *hash = strchr(line, '#');
                    if (hash) *hash = 0;
                    buf = line;
                }

                tok = strtok_r(buf, " ,", &toksave);
                buf = NULL;
                if (tok && sscanf(tok, "%lf", &value) == 1)
                    break;
            }
            while (1);

            if (t == 2)
            {
                x265_log(param, X265_LOG_ERROR, "lambda file contains too many values\n");
                fclose(lfn);
                return true;
            }
            else
                x265_log(param, X265_LOG_DEBUG, "lambda%c[%d] = %lf\n", t ? '2' : ' ', i, value);
            table[i] = value;
        }
    }

    fclose(lfn);
    return false;
}

bool parseMaskingStrength(x265_param* p, const char* value)
{
    bool bError = false;
    int window1[6];
    double refQpDelta1[6], nonRefQpDelta1[6];
    if (p->bEnableSceneCutAwareQp == FORWARD)
    {
        if (3 == sscanf(value, "%d,%lf,%lf", &window1[0], &refQpDelta1[0], &nonRefQpDelta1[0]))
        {
            if (window1[0] > 0)
                p->fwdMaxScenecutWindow = window1[0];
            if (refQpDelta1[0] >= 0)
                p->fwdRefQpDelta[0] = refQpDelta1[0];
            if (nonRefQpDelta1[0] >= 0)
                p->fwdNonRefQpDelta[0] = nonRefQpDelta1[0];

            p->fwdScenecutWindow[0] = p->fwdMaxScenecutWindow / 6;
            for (int i = 1; i < 6; i++)
            {
                p->fwdScenecutWindow[i] = p->fwdMaxScenecutWindow / 6;
                p->fwdRefQpDelta[i] = p->fwdRefQpDelta[i - 1] - (0.15 * p->fwdRefQpDelta[i - 1]);
                p->fwdNonRefQpDelta[i] = p->fwdNonRefQpDelta[i - 1] - (0.15 * p->fwdNonRefQpDelta[i - 1]);
            }
        }
        else if (18 == sscanf(value, "%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf"
            , &window1[0], &refQpDelta1[0], &nonRefQpDelta1[0], &window1[1], &refQpDelta1[1], &nonRefQpDelta1[1]
            , &window1[2], &refQpDelta1[2], &nonRefQpDelta1[2], &window1[3], &refQpDelta1[3], &nonRefQpDelta1[3]
            , &window1[4], &refQpDelta1[4], &nonRefQpDelta1[4], &window1[5], &refQpDelta1[5], &nonRefQpDelta1[5]))
        {
            p->fwdMaxScenecutWindow = 0;
            for (int i = 0; i < 6; i++)
            {
                p->fwdScenecutWindow[i] = window1[i];
                p->fwdRefQpDelta[i] = refQpDelta1[i];
                p->fwdNonRefQpDelta[i] = nonRefQpDelta1[i];
                p->fwdMaxScenecutWindow += p->fwdScenecutWindow[i];
            }
        }
        else
        {
            x265_log(NULL, X265_LOG_ERROR, "Specify all the necessary offsets for masking-strength \n");
            bError = true;
        }
    }
    else if (p->bEnableSceneCutAwareQp == BACKWARD)
    {
        if (3 == sscanf(value, "%d,%lf,%lf", &window1[0], &refQpDelta1[0], &nonRefQpDelta1[0]))
        {
            if (window1[0] > 0)
                p->bwdMaxScenecutWindow = window1[0];
            if (refQpDelta1[0] >= 0)
                p->bwdRefQpDelta[0] = refQpDelta1[0];
            if (nonRefQpDelta1[0] >= 0)
                p->bwdNonRefQpDelta[0] = nonRefQpDelta1[0];

            p->bwdScenecutWindow[0] = p->bwdMaxScenecutWindow / 6;
            for (int i = 1; i < 6; i++)
            {
                p->bwdScenecutWindow[i] = p->bwdMaxScenecutWindow / 6;
                p->bwdRefQpDelta[i] = p->bwdRefQpDelta[i - 1] - (0.15 * p->bwdRefQpDelta[i - 1]);
                p->bwdNonRefQpDelta[i] = p->bwdNonRefQpDelta[i - 1] - (0.15 * p->bwdNonRefQpDelta[i - 1]);
            }
        }
        else if (18 == sscanf(value, "%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf"
            , &window1[0], &refQpDelta1[0], &nonRefQpDelta1[0], &window1[1], &refQpDelta1[1], &nonRefQpDelta1[1]
            , &window1[2], &refQpDelta1[2], &nonRefQpDelta1[2], &window1[3], &refQpDelta1[3], &nonRefQpDelta1[3]
            , &window1[4], &refQpDelta1[4], &nonRefQpDelta1[4], &window1[5], &refQpDelta1[5], &nonRefQpDelta1[5]))
        {
            p->bwdMaxScenecutWindow = 0;
            for (int i = 0; i < 6; i++)
            {
                p->bwdScenecutWindow[i] = window1[i];
                p->bwdRefQpDelta[i] = refQpDelta1[i];
                p->bwdNonRefQpDelta[i] = nonRefQpDelta1[i];
                p->bwdMaxScenecutWindow += p->bwdScenecutWindow[i];
            }
        }
        else
        {
            x265_log(NULL, X265_LOG_ERROR, "Specify all the necessary offsets for masking-strength \n");
            bError = true;
        }
    }
    else if (p->bEnableSceneCutAwareQp == BI_DIRECTIONAL)
    {
        int window2[6];
        double refQpDelta2[6], nonRefQpDelta2[6];
        if (6 == sscanf(value, "%d,%lf,%lf,%d,%lf,%lf", &window1[0], &refQpDelta1[0], &nonRefQpDelta1[0], &window2[0], &refQpDelta2[0], &nonRefQpDelta2[0]))
        {
            if (window1[0] > 0)
                p->fwdMaxScenecutWindow = window1[0];
            if (refQpDelta1[0] >= 0)
                p->fwdRefQpDelta[0] = refQpDelta1[0];
            if (nonRefQpDelta1[0] >= 0)
                p->fwdNonRefQpDelta[0] = nonRefQpDelta1[0];
            if (window2[0] > 0)
                p->bwdMaxScenecutWindow = window2[0];
            if (refQpDelta2[0] >= 0)
                p->bwdRefQpDelta[0] = refQpDelta2[0];
            if (nonRefQpDelta2[0] >= 0)
                p->bwdNonRefQpDelta[0] = nonRefQpDelta2[0];

            p->fwdScenecutWindow[0] = p->fwdMaxScenecutWindow / 6;
            p->bwdScenecutWindow[0] = p->bwdMaxScenecutWindow / 6;
            for (int i = 1; i < 6; i++)
            {
                p->fwdScenecutWindow[i] = p->fwdMaxScenecutWindow / 6;
                p->bwdScenecutWindow[i] = p->bwdMaxScenecutWindow / 6;
                p->fwdRefQpDelta[i] = p->fwdRefQpDelta[i - 1] - (0.15 * p->fwdRefQpDelta[i - 1]);
                p->fwdNonRefQpDelta[i] = p->fwdNonRefQpDelta[i - 1] - (0.15 * p->fwdNonRefQpDelta[i - 1]);
                p->bwdRefQpDelta[i] = p->bwdRefQpDelta[i - 1] - (0.15 * p->bwdRefQpDelta[i - 1]);
                p->bwdNonRefQpDelta[i] = p->bwdNonRefQpDelta[i - 1] - (0.15 * p->bwdNonRefQpDelta[i - 1]);
            }
        }
        else if (36 == sscanf(value, "%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf,%d,%lf,%lf"
            , &window1[0], &refQpDelta1[0], &nonRefQpDelta1[0], &window1[1], &refQpDelta1[1], &nonRefQpDelta1[1]
            , &window1[2], &refQpDelta1[2], &nonRefQpDelta1[2], &window1[3], &refQpDelta1[3], &nonRefQpDelta1[3]
            , &window1[4], &refQpDelta1[4], &nonRefQpDelta1[4], &window1[5], &refQpDelta1[5], &nonRefQpDelta1[5]
            , &window2[0], &refQpDelta2[0], &nonRefQpDelta2[0], &window2[1], &refQpDelta2[1], &nonRefQpDelta2[1]
            , &window2[2], &refQpDelta2[2], &nonRefQpDelta2[2], &window2[3], &refQpDelta2[3], &nonRefQpDelta2[3]
            , &window2[4], &refQpDelta2[4], &nonRefQpDelta2[4], &window2[5], &refQpDelta2[5], &nonRefQpDelta2[5]))
        {
            p->fwdMaxScenecutWindow = 0;
            p->bwdMaxScenecutWindow = 0;
            for (int i = 0; i < 6; i++)
            {
                p->fwdScenecutWindow[i] = window1[i];
                p->fwdRefQpDelta[i] = refQpDelta1[i];
                p->fwdNonRefQpDelta[i] = nonRefQpDelta1[i];
                p->bwdScenecutWindow[i] = window2[i];
                p->bwdRefQpDelta[i] = refQpDelta2[i];
                p->bwdNonRefQpDelta[i] = nonRefQpDelta2[i];
                p->fwdMaxScenecutWindow += p->fwdScenecutWindow[i];
                p->bwdMaxScenecutWindow += p->bwdScenecutWindow[i];
            }
        }
        else
        {
            x265_log(NULL, X265_LOG_ERROR, "Specify all the necessary offsets for masking-strength \n");
            bError = true;
        }
    }
    return bError;
}

void x265_copy_params(x265_param* dst, x265_param* src)
{
    dst->mcstfFrameRange = src->mcstfFrameRange;
    dst->cpuid = src->cpuid;
    dst->frameNumThreads = src->frameNumThreads;
    if (strlen(src->numaPools)) snprintf(dst->numaPools, X265_MAX_STRING_SIZE, "%s", src->numaPools);
    else dst->numaPools[0] = 0;

    dst->tune = src->tune;
    dst->bEnableWavefront = src->bEnableWavefront;
    dst->bDistributeModeAnalysis = src->bDistributeModeAnalysis;
    dst->bDistributeMotionEstimation = src->bDistributeMotionEstimation;
    dst->bLogCuStats = src->bLogCuStats;
    dst->bEnablePsnr = src->bEnablePsnr;
    dst->bEnableSsim = src->bEnableSsim;
    dst->logLevel = src->logLevel;
    if (strlen(src->logfn)) snprintf(dst->logfn, X265_MAX_STRING_SIZE, "%s", src->logfn);
    else dst->logfn[0] = 0;
    dst->logfLevel = src->logfLevel;
    dst->csvLogLevel = src->csvLogLevel;
    if (strlen(src->csvfn)) snprintf(dst->csvfn, X265_MAX_STRING_SIZE, "%s", src->csvfn);
    else dst->csvfn[0] = 0;
    dst->internalBitDepth = src->internalBitDepth;
    dst->sourceBitDepth = src->sourceBitDepth;
    dst->internalCsp = src->internalCsp;
    dst->fpsNum = src->fpsNum;
    dst->fpsDenom = src->fpsDenom;
    dst->sourceHeight = src->sourceHeight;
    dst->sourceWidth = src->sourceWidth;
    dst->interlaceMode = src->interlaceMode;
    dst->totalFrames = src->totalFrames;
    dst->levelIdc = src->levelIdc;
    dst->bHighTier = src->bHighTier;
    dst->uhdBluray = src->uhdBluray;
    dst->maxNumReferences = src->maxNumReferences;
    dst->bAllowNonConformance = src->bAllowNonConformance;
    dst->bRepeatHeaders = src->bRepeatHeaders;
    dst->bAnnexB = src->bAnnexB;
    dst->bEnableAccessUnitDelimiters = src->bEnableAccessUnitDelimiters;
    dst->bEnableEndOfBitstream = src->bEnableEndOfBitstream;
    dst->bEnableEndOfSequence = src->bEnableEndOfSequence;
    dst->bEmitInfoSEI = src->bEmitInfoSEI;
    dst->decodedPictureHashSEI = src->decodedPictureHashSEI;
    dst->bEnableTemporalSubLayers = src->bEnableTemporalSubLayers;
    dst->bOpenGOP = src->bOpenGOP;
    dst->craNal = src->craNal;
    dst->keyframeMax = src->keyframeMax;
    dst->keyframeMin = src->keyframeMin;
    dst->bframes = src->bframes;
    dst->bFrameAdaptive = src->bFrameAdaptive;
    dst->bFrameBias = src->bFrameBias;
    dst->bBPyramid = src->bBPyramid;
    dst->lookaheadDepth = src->lookaheadDepth;
    dst->lookaheadSlices = src->lookaheadSlices;
    dst->lookaheadThreads = src->lookaheadThreads;
    dst->scenecutThreshold = src->scenecutThreshold;
    dst->bHistBasedSceneCut = src->bHistBasedSceneCut;
    dst->bIntraRefresh = src->bIntraRefresh;
    dst->maxCUSize = src->maxCUSize;
    dst->minCUSize = src->minCUSize;
    dst->bEnableRectInter = src->bEnableRectInter;
    dst->bEnableAMP = src->bEnableAMP;
    dst->maxTUSize = src->maxTUSize;
    dst->tuQTMaxInterDepth = src->tuQTMaxInterDepth;
    dst->tuQTMaxIntraDepth = src->tuQTMaxIntraDepth;
    dst->limitTU = src->limitTU;
    dst->rdoqLevel = src->rdoqLevel;
    dst->bEnableSignHiding = src->bEnableSignHiding;
    dst->bEnableTransformSkip = src->bEnableTransformSkip;
    dst->noiseReductionInter = src->noiseReductionInter;
    dst->noiseReductionIntra = src->noiseReductionIntra;
    if (strlen(src->scalingLists)) snprintf(dst->scalingLists, X265_MAX_STRING_SIZE, "%s", src->scalingLists);
    else dst->scalingLists[0] = 0;
    dst->bEnableStrongIntraSmoothing = src->bEnableStrongIntraSmoothing;
    dst->bEnableConstrainedIntra = src->bEnableConstrainedIntra;
    dst->maxNumMergeCand = src->maxNumMergeCand;
    dst->limitReferences = src->limitReferences;
    dst->limitModes = src->limitModes;
    dst->searchMethod = src->searchMethod;
    dst->subpelRefine = src->subpelRefine;
    dst->searchRange = src->searchRange;
    dst->bEnableTemporalMvp = src->bEnableTemporalMvp;
    dst->bEnableFrameDuplication = src->bEnableFrameDuplication;
    dst->dupThreshold = src->dupThreshold;
    dst->bEnableHME = src->bEnableHME;
    if (src->bEnableHME)
    {
        for (int level = 0; level < 3; level++)
        {
            dst->hmeSearchMethod[level] = src->hmeSearchMethod[level];
            dst->hmeRange[level] = src->hmeRange[level];
        }
    }
    dst->bEnableWeightedBiPred = src->bEnableWeightedBiPred;
    dst->bEnableWeightedPred = src->bEnableWeightedPred;
    dst->bSourceReferenceEstimation = src->bSourceReferenceEstimation;
    dst->bEnableLoopFilter = src->bEnableLoopFilter;
    dst->deblockingFilterBetaOffset = src->deblockingFilterBetaOffset;
    dst->deblockingFilterTCOffset = src->deblockingFilterTCOffset;
    dst->bEnableSAO = src->bEnableSAO;
    dst->bSaoNonDeblocked = src->bSaoNonDeblocked;
    dst->rdLevel = src->rdLevel;
    dst->bEnableEarlySkip = src->bEnableEarlySkip;
    dst->recursionSkipMode = src->recursionSkipMode;
    dst->edgeVarThreshold = src->edgeVarThreshold;
    dst->bEnableFastIntra = src->bEnableFastIntra;
    dst->bEnableTSkipFast = src->bEnableTSkipFast;
    dst->bCULossless = src->bCULossless;
    dst->bIntraInBFrames = src->bIntraInBFrames;
    dst->rdPenalty = src->rdPenalty;
    dst->psyRd = src->psyRd;
    dst->psyRdoq = src->psyRdoq;
    dst->bEnableRdRefine = src->bEnableRdRefine;
    dst->analysisReuseMode = src->analysisReuseMode;
    if (strlen(src->analysisReuseFileName)) snprintf(dst->analysisReuseFileName, X265_MAX_STRING_SIZE, "%s", src->analysisReuseFileName);
    else dst->analysisReuseFileName[0] = 0;
    dst->bLossless = src->bLossless;
    dst->cbQpOffset = src->cbQpOffset;
    dst->crQpOffset = src->crQpOffset;
    dst->preferredTransferCharacteristics = src->preferredTransferCharacteristics;
    dst->pictureStructure = src->pictureStructure;

    dst->rc.rateControlMode = src->rc.rateControlMode;
    dst->rc.qp = src->rc.qp;
    dst->rc.bitrate = src->rc.bitrate;
    dst->rc.qCompress = src->rc.qCompress;
    dst->rc.ipFactor = src->rc.ipFactor;
    dst->rc.pbFactor = src->rc.pbFactor;
    dst->rc.rfConstant = src->rc.rfConstant;
    dst->rc.qpStep = src->rc.qpStep;
    dst->rc.aqMode = src->rc.aqMode;
    dst->rc.aqStrength = src->rc.aqStrength;
    dst->rc.aqBiasStrength = src->rc.aqBiasStrength;
    dst->rc.vbvBufferSize = src->rc.vbvBufferSize;
    dst->rc.vbvMaxBitrate = src->rc.vbvMaxBitrate;

    dst->rc.vbvBufferInit = src->rc.vbvBufferInit;
    dst->minVbvFullness = src->minVbvFullness;
    dst->maxVbvFullness = src->maxVbvFullness;
    dst->rc.cuTree = src->rc.cuTree;
    dst->rc.rfConstantMax = src->rc.rfConstantMax;
    dst->rc.rfConstantMin = src->rc.rfConstantMin;
    dst->rc.bStatWrite = src->rc.bStatWrite;
    dst->rc.bStatRead = src->rc.bStatRead;
    dst->rc.dataShareMode = src->rc.dataShareMode;
    if (strlen(src->rc.statFileName)) snprintf(dst->rc.statFileName, X265_MAX_STRING_SIZE, "%s", src->rc.statFileName);
    else dst->rc.statFileName[0] = 0;
    if (strlen(src->rc.sharedMemName)) snprintf(dst->rc.sharedMemName, X265_MAX_STRING_SIZE, "%s", src->rc.sharedMemName);
    else dst->rc.sharedMemName[0] = 0;
    dst->rc.qblur = src->rc.qblur;
    dst->rc.complexityBlur = src->rc.complexityBlur;
    dst->rc.bEnableSlowFirstPass = src->rc.bEnableSlowFirstPass;
    dst->rc.zoneCount = src->rc.zoneCount;
    dst->rc.zonefileCount = src->rc.zonefileCount;
    dst->reconfigWindowSize = src->reconfigWindowSize;
    dst->bResetZoneConfig = src->bResetZoneConfig;
    dst->bNoResetZoneConfig = src->bNoResetZoneConfig;
    dst->decoderVbvMaxRate = src->decoderVbvMaxRate;

    if (src->rc.zonefileCount && src->rc.zones && src->bResetZoneConfig)
    {
        for (int i = 0; i < src->rc.zonefileCount; i++)
        {
            dst->rc.zones[i].startFrame = src->rc.zones[i].startFrame;
            dst->rc.zones[0].keyframeMax = src->rc.zones[0].keyframeMax;
            memcpy(dst->rc.zones[i].zoneParam, src->rc.zones[i].zoneParam, sizeof(x265_param));
        }
    }
    else if (src->rc.zoneCount && src->rc.zones)
    {
        for (int i = 0; i < src->rc.zoneCount; i++)
        {
            dst->rc.zones[i].startFrame = src->rc.zones[i].startFrame;
            dst->rc.zones[i].endFrame = src->rc.zones[i].endFrame;
            dst->rc.zones[i].bForceQp = src->rc.zones[i].bForceQp;
            dst->rc.zones[i].qp = src->rc.zones[i].qp;
            dst->rc.zones[i].bitrateFactor = src->rc.zones[i].bitrateFactor;
        }
    }
    else
        dst->rc.zones = NULL;

    if (strlen(src->rc.lambdaFileName)) snprintf(dst->rc.lambdaFileName, X265_MAX_STRING_SIZE, "%s", src->rc.lambdaFileName);
    else dst->rc.lambdaFileName[0] = 0;
    dst->rc.bStrictCbr = src->rc.bStrictCbr;
    dst->rc.qgSize = src->rc.qgSize;
    dst->rc.bEnableGrain = src->rc.bEnableGrain;
    dst->rc.qpMax = src->rc.qpMax;
    dst->rc.qpMin = src->rc.qpMin;
    dst->rc.bEnableConstVbv = src->rc.bEnableConstVbv;
    dst->rc.hevcAq = src->rc.hevcAq;
    dst->rc.qpAdaptationRange = src->rc.qpAdaptationRange;
    dst->rc.bAutoAq = src->rc.bAutoAq;

    dst->vui.aspectRatioIdc = src->vui.aspectRatioIdc;
    dst->vui.sarWidth = src->vui.sarWidth;
    dst->vui.sarHeight = src->vui.sarHeight;
    dst->vui.bEnableOverscanAppropriateFlag = src->vui.bEnableOverscanAppropriateFlag;
    dst->vui.bEnableOverscanInfoPresentFlag = src->vui.bEnableOverscanInfoPresentFlag;
    dst->vui.bEnableVideoSignalTypePresentFlag = src->vui.bEnableVideoSignalTypePresentFlag;
    dst->vui.videoFormat = src->vui.videoFormat;
    dst->vui.bEnableVideoFullRangeFlag = src->vui.bEnableVideoFullRangeFlag;
    dst->vui.bEnableColorDescriptionPresentFlag = src->vui.bEnableColorDescriptionPresentFlag;
    dst->vui.colorPrimaries = src->vui.colorPrimaries;
    dst->vui.transferCharacteristics = src->vui.transferCharacteristics;
    dst->vui.matrixCoeffs = src->vui.matrixCoeffs;
    dst->vui.bEnableChromaLocInfoPresentFlag = src->vui.bEnableChromaLocInfoPresentFlag;
    dst->vui.chromaSampleLocTypeTopField = src->vui.chromaSampleLocTypeTopField;
    dst->vui.chromaSampleLocTypeBottomField = src->vui.chromaSampleLocTypeBottomField;
    dst->vui.bEnableDefaultDisplayWindowFlag = src->vui.bEnableDefaultDisplayWindowFlag;
    dst->vui.defDispWinBottomOffset = src->vui.defDispWinBottomOffset;
    dst->vui.defDispWinLeftOffset = src->vui.defDispWinLeftOffset;
    dst->vui.defDispWinRightOffset = src->vui.defDispWinRightOffset;
    dst->vui.defDispWinTopOffset = src->vui.defDispWinTopOffset;

    if (strlen(src->masteringDisplayColorVolume)) snprintf(dst->masteringDisplayColorVolume, X265_MAX_STRING_SIZE, "%s", src->masteringDisplayColorVolume);
    else dst->masteringDisplayColorVolume[0] = 0;
    dst->maxLuma = src->maxLuma;
    dst->minLuma = src->minLuma;
    dst->bEmitCLL = src->bEmitCLL;
    dst->maxCLL = src->maxCLL;
    dst->maxFALL = src->maxFALL;
    dst->log2MaxPocLsb = src->log2MaxPocLsb;
    dst->bEmitVUIHRDInfo = src->bEmitVUIHRDInfo;
    dst->bEmitVUITimingInfo = src->bEmitVUITimingInfo;
    dst->maxSlices = src->maxSlices;
    dst->bOptQpPPS = src->bOptQpPPS;
    dst->bOptRefListLengthPPS = src->bOptRefListLengthPPS;
    dst->bMultiPassOptRPS = src->bMultiPassOptRPS;
    dst->scenecutBias = src->scenecutBias;
    dst->gopLookahead = src->lookaheadDepth;
    dst->bOptCUDeltaQP = src->bOptCUDeltaQP;
    dst->analysisMultiPassDistortion = src->analysisMultiPassDistortion;
    dst->analysisMultiPassRefine = src->analysisMultiPassRefine;
    dst->bAQMotion = src->bAQMotion;
    dst->bSsimRd = src->bSsimRd;
    dst->dynamicRd = src->dynamicRd;
    dst->bEmitHDR10SEI = src->bEmitHDR10SEI;
    dst->bEmitHRDSEI = src->bEmitHRDSEI;
    dst->bHDROpt = src->bHDROpt; /*DEPRECATED*/
    dst->bHDR10Opt = src->bHDR10Opt;
    dst->analysisReuseLevel = src->analysisReuseLevel;
    dst->analysisSaveReuseLevel = src->analysisSaveReuseLevel;
    dst->analysisLoadReuseLevel = src->analysisLoadReuseLevel;
    dst->bLimitSAO = src->bLimitSAO;
    if (strlen(src->toneMapFile)) snprintf(dst->toneMapFile, X265_MAX_STRING_SIZE, "%s", src->toneMapFile);
    else dst->toneMapFile[0] = 0;
    dst->bDhdr10opt = src->bDhdr10opt;
    dst->bCTUInfo = src->bCTUInfo;
    dst->bUseRcStats = src->bUseRcStats;
    dst->interRefine = src->interRefine;
    dst->intraRefine = src->intraRefine;
    dst->mvRefine = src->mvRefine;
    dst->maxLog2CUSize = src->maxLog2CUSize;
    dst->maxCUDepth = src->maxCUDepth;
    dst->unitSizeDepth = src->unitSizeDepth;
    dst->num4x4Partitions = src->num4x4Partitions;

    dst->csvfpt = src->csvfpt;
    dst->bEnableSplitRdSkip = src->bEnableSplitRdSkip;
    dst->bUseAnalysisFile = src->bUseAnalysisFile;
    dst->forceFlush = src->forceFlush;
    dst->bDisableLookahead = src->bDisableLookahead;
    dst->bLowPassDct = src->bLowPassDct;
    dst->vbvBufferEnd = src->vbvBufferEnd;
    dst->vbvEndFrameAdjust = src->vbvEndFrameAdjust;
    dst->bAnalysisType = src->bAnalysisType;
    dst->bCopyPicToFrame = src->bCopyPicToFrame;
    if (strlen(src->analysisSave)) snprintf(dst->analysisSave, X265_MAX_STRING_SIZE, "%s", src->analysisSave);
    else dst->analysisSave[0] = 0;
    if (strlen(src->analysisLoad)) snprintf(dst->analysisLoad, X265_MAX_STRING_SIZE, "%s", src->analysisLoad);
    else dst->analysisLoad[0] = 0;
    dst->gopLookahead = src->gopLookahead;
    dst->radl = src->radl;
    dst->selectiveSAO = src->selectiveSAO;
    dst->maxAUSizeFactor = src->maxAUSizeFactor;
    dst->bEmitIDRRecoverySEI = src->bEmitIDRRecoverySEI;
    dst->bDynamicRefine = src->bDynamicRefine;
    dst->bSingleSeiNal = src->bSingleSeiNal;
    dst->chunkStart = src->chunkStart;
    dst->chunkEnd = src->chunkEnd;
    if (src->naluFile[0]) snprintf(dst->naluFile, X265_MAX_STRING_SIZE, "%s", src->naluFile);
    else dst->naluFile[0] = 0;
    dst->scaleFactor = src->scaleFactor;
    dst->ctuDistortionRefine = src->ctuDistortionRefine;
    dst->bEnableHRDConcatFlag = src->bEnableHRDConcatFlag;
    dst->dolbyProfile = src->dolbyProfile;
    dst->bEnableSvtHevc = src->bEnableSvtHevc;
    dst->bThreadedME = src->bThreadedME;
    dst->tmeTaskBlockSize = src->tmeTaskBlockSize;
    dst->tmeNumBufferRows = src->tmeNumBufferRows;
    dst->bEnableFades = src->bEnableFades;
    dst->bEnableSceneCutAwareQp = src->bEnableSceneCutAwareQp;
    dst->fwdMaxScenecutWindow = src->fwdMaxScenecutWindow;
    dst->bwdMaxScenecutWindow = src->bwdMaxScenecutWindow;
    for (int i = 0; i < 6; i++)
    {
        dst->fwdScenecutWindow[i] = src->fwdScenecutWindow[i];
        dst->fwdRefQpDelta[i] = src->fwdRefQpDelta[i];
        dst->fwdNonRefQpDelta[i] = src->fwdNonRefQpDelta[i];
        dst->bwdScenecutWindow[i] = src->bwdScenecutWindow[i];
        dst->bwdRefQpDelta[i] = src->bwdRefQpDelta[i];
        dst->bwdNonRefQpDelta[i] = src->bwdNonRefQpDelta[i];
    }
    dst->bField = src->bField;
    dst->bEnableTemporalFilter = src->bEnableTemporalFilter;
    dst->temporalFilterStrength = src->temporalFilterStrength;
    dst->searchRangeForLayer0 = src->searchRangeForLayer0;
    dst->searchRangeForLayer1 = src->searchRangeForLayer1;
    dst->searchRangeForLayer2 = src->searchRangeForLayer2;
    dst->confWinRightOffset = src->confWinRightOffset;
    dst->confWinBottomOffset = src->confWinBottomOffset;
    dst->bliveVBV2pass = src->bliveVBV2pass;
#if ENABLE_ALPHA
    dst->bEnableAlpha = src->bEnableAlpha;
    dst->numScalableLayers = src->numScalableLayers;
#endif
#if ENABLE_MULTIVIEW
    dst->numViews = src->numViews;
    dst->format = src->format;
#endif
    dst->numLayers = src->numLayers;
#if ENABLE_SCC_EXT
    dst->bEnableSCC = src->bEnableSCC;
#endif

    if (strlen(src->videoSignalTypePreset)) snprintf(dst->videoSignalTypePreset, X265_MAX_STRING_SIZE, "%s", src->videoSignalTypePreset);
    else dst->videoSignalTypePreset[0] = 0;
#ifdef SVT_HEVC
    memcpy(dst->svtHevcParam, src->svtHevcParam, sizeof(EB_H265_ENC_CONFIGURATION));
#endif
    /* Film grain */
    dst->filmGrain = src->filmGrain;
    /* Aom Film grain*/
    dst->aomFilmGrain = src->aomFilmGrain;
    dst->bEnableSBRC = src->bEnableSBRC;
    dst->bConfigRCFrame = src->bConfigRCFrame;
    dst->isAbrLadderEnable = src->isAbrLadderEnable;
}

#ifdef SVT_HEVC

void svt_param_default(x265_param* param)
{
    EB_H265_ENC_CONFIGURATION* svtHevcParam = (EB_H265_ENC_CONFIGURATION*)param->svtHevcParam;

    // Channel info
    svtHevcParam->channelId = 0;
    svtHevcParam->activeChannelCount = 0;

    // GOP Structure
    svtHevcParam->intraPeriodLength = -2;
    svtHevcParam->intraRefreshType = 1;
    svtHevcParam->predStructure = 2;
    svtHevcParam->baseLayerSwitchMode = 0;
    svtHevcParam->hierarchicalLevels = 3;
    svtHevcParam->sourceWidth = 0;
    svtHevcParam->sourceHeight = 0;
    svtHevcParam->latencyMode = 0;

    //Preset & Tune
    svtHevcParam->encMode = 7;
    svtHevcParam->tune = 1;

    // Interlaced Video 
    svtHevcParam->interlacedVideo = 0;

    // Quantization
    svtHevcParam->qp = 32;
    svtHevcParam->useQpFile = 0;

    // Deblock Filter
    svtHevcParam->disableDlfFlag = 0;

    // SAO
    svtHevcParam->enableSaoFlag = 1;

    // ME Tools
    svtHevcParam->useDefaultMeHme = 1;
    svtHevcParam->enableHmeFlag = 1;

    // ME Parameters
    svtHevcParam->searchAreaWidth = 16;
    svtHevcParam->searchAreaHeight = 7;

    // MD Parameters
    svtHevcParam->constrainedIntra = 0;

    // Rate Control
    svtHevcParam->frameRate = 60;
    svtHevcParam->frameRateNumerator = 0;
    svtHevcParam->frameRateDenominator = 0;
    svtHevcParam->encoderBitDepth = 8;
    svtHevcParam->encoderColorFormat = EB_YUV420;
    svtHevcParam->compressedTenBitFormat = 0;
    svtHevcParam->rateControlMode = 0;
    svtHevcParam->sceneChangeDetection = 1;
    svtHevcParam->lookAheadDistance = (uint32_t)~0;
    svtHevcParam->framesToBeEncoded = 0;
    svtHevcParam->targetBitRate = 7000000;
    svtHevcParam->maxQpAllowed = 48;
    svtHevcParam->minQpAllowed = 10;
    svtHevcParam->bitRateReduction = 0;

    // Thresholds
    svtHevcParam->improveSharpness = 0;
    svtHevcParam->videoUsabilityInfo = 0;
    svtHevcParam->highDynamicRangeInput = 0;
    svtHevcParam->accessUnitDelimiter = 0;
    svtHevcParam->bufferingPeriodSEI = 0;
    svtHevcParam->pictureTimingSEI = 0;
    svtHevcParam->registeredUserDataSeiFlag = 0;
    svtHevcParam->unregisteredUserDataSeiFlag = 0;
    svtHevcParam->recoveryPointSeiFlag = 0;
    svtHevcParam->enableTemporalId = 1;
    svtHevcParam->profile = 1;
    svtHevcParam->tier = 0;
    svtHevcParam->level = 0;

    svtHevcParam->injectorFrameRate = 60 << 16;
    svtHevcParam->speedControlFlag = 0;

    // ASM Type
    svtHevcParam->asmType = 1;

    svtHevcParam->codeVpsSpsPps = 1;
    svtHevcParam->codeEosNal = 0;
    svtHevcParam->reconEnabled = 0;
    svtHevcParam->maxCLL = 0;
    svtHevcParam->maxFALL = 0;
    svtHevcParam->useMasteringDisplayColorVolume = 0;
    svtHevcParam->useNaluFile = 0;
    svtHevcParam->whitePointX = 0;
    svtHevcParam->whitePointY = 0;
    svtHevcParam->maxDisplayMasteringLuminance = 0;
    svtHevcParam->minDisplayMasteringLuminance = 0;
    svtHevcParam->dolbyVisionProfile = 0;
    svtHevcParam->targetSocket = -1;
    svtHevcParam->logicalProcessors = 0;
    svtHevcParam->switchThreadsToRtPriority = 1;
    svtHevcParam->fpsInVps = 0;

    svtHevcParam->tileColumnCount = 1;
    svtHevcParam->tileRowCount = 1;
    svtHevcParam->tileSliceMode = 0;
    svtHevcParam->unrestrictedMotionVector = 1;
    svtHevcParam->threadCount = 0;

    // vbv
    svtHevcParam->hrdFlag = 0;
    svtHevcParam->vbvMaxrate = 0;
    svtHevcParam->vbvBufsize = 0;
    svtHevcParam->vbvBufInit = 90;
}

int svt_set_preset(x265_param* param, const char* preset)
{
    EB_H265_ENC_CONFIGURATION* svtHevcParam = (EB_H265_ENC_CONFIGURATION*)param->svtHevcParam;
    
    if (preset)
    {
        if (!strcmp(preset, "ultrafast")) svtHevcParam->encMode = 11;
        else if (!strcmp(preset, "superfast")) svtHevcParam->encMode = 10;
        else if (!strcmp(preset, "veryfast")) svtHevcParam->encMode = 9;
        else if (!strcmp(preset, "faster")) svtHevcParam->encMode = 8;
        else if (!strcmp(preset, "fast")) svtHevcParam->encMode = 7;
        else if (!strcmp(preset, "medium")) svtHevcParam->encMode = 6;
        else if (!strcmp(preset, "slow")) svtHevcParam->encMode = 5;
        else if (!strcmp(preset, "slower")) svtHevcParam->encMode =4;
        else if (!strcmp(preset, "veryslow")) svtHevcParam->encMode = 3;
        else if (!strcmp(preset, "placebo")) svtHevcParam->encMode = 2;
        else  return -1;
    }
    return 0;
}

int svt_param_parse(x265_param* param, const char* name, const char* value)
{
    bool bError = false;
#define OPT(STR) else if (!strcmp(name, STR))

    EB_H265_ENC_CONFIGURATION* svtHevcParam = (EB_H265_ENC_CONFIGURATION*)param->svtHevcParam;
    if (0);
    OPT("input-res")  bError |= sscanf(value, "%dx%d", &svtHevcParam->sourceWidth, &svtHevcParam->sourceHeight) != 2;
    OPT("input-depth") svtHevcParam->encoderBitDepth = atoi(value);
    OPT("total-frames") svtHevcParam->framesToBeEncoded = atoi(value);
    OPT("frames") svtHevcParam->framesToBeEncoded = atoi(value);
    OPT("fps")
    {
        if (sscanf(value, "%u/%u", &svtHevcParam->frameRateNumerator, &svtHevcParam->frameRateDenominator) == 2)
            ;
        else
        {
            int fps = atoi(value);
            svtHevcParam->frameRateDenominator = 1;

            if (fps < 1000)
                svtHevcParam->frameRate = fps << 16;
            else
                svtHevcParam->frameRate = fps;
        }
    }
    OPT2("level-idc", "level")
    {
        /* allow "5.1" or "51", both converted to integer 51 */
        /* if level-idc specifies an obviously wrong value in either float or int,
        throw error consistently. Stronger level checking will be done in encoder_open() */
        if (atof(value) < 10)
            svtHevcParam->level = (int)(10 * atof(value) + .5);
        else if (atoi(value) < 100)
            svtHevcParam->level = atoi(value);
        else
            bError = true;
    }
    OPT2("pools", "numa-pools")
    {
        char *pools = strdup(value);
        char *temp1, *temp2;
        int count = 0;

        for (temp1 = strstr(pools, ","); temp1 != NULL; temp1 = strstr(temp2, ","))
        {
            temp2 = ++temp1;
            count++;
        }

        if (count > 1)
            x265_log(param, X265_LOG_WARNING, "SVT-HEVC Encoder supports pools option only upto 2 sockets \n");
        else if (count == 1)
        {
            temp1 = strtok(pools, ",");
            temp2 = strtok(NULL, ",");

            if (!strcmp(temp1, "+"))
            {
                if (!strcmp(temp2, "+")) svtHevcParam->targetSocket = -1;
                else if (!strcmp(temp2, "-")) svtHevcParam->targetSocket = 0;
                else svtHevcParam->targetSocket = -1;
            }
            else if (!strcmp(temp1, "-"))
            {
                if (!strcmp(temp2, "+")) svtHevcParam->targetSocket = 1;
                else if (!strcmp(temp2, "-")) x265_log(param, X265_LOG_ERROR, "Shouldn't exclude both sockets for pools option %s \n", pools);
                else if (!strcmp(temp2, "*")) svtHevcParam->targetSocket = 1;
                else
                {
                    svtHevcParam->targetSocket = 1;
                    svtHevcParam->logicalProcessors = atoi(temp2);
                }
            }
            else svtHevcParam->targetSocket = -1;
        }
        else
        {
            if (!strcmp(temp1, "*")) svtHevcParam->targetSocket = -1;
            else
            {
                svtHevcParam->targetSocket = 0;
                svtHevcParam->logicalProcessors = atoi(temp1);
            }
        }
        free(pools);
    }
    OPT("high-tier") svtHevcParam->tier = x265_atobool(value, bError);
    OPT("qpmin") svtHevcParam->minQpAllowed = atoi(value);
    OPT("qpmax") svtHevcParam->maxQpAllowed = atoi(value);
    OPT("rc-lookahead") svtHevcParam->lookAheadDistance = atoi(value);
    OPT("scenecut")
    {
        svtHevcParam->sceneChangeDetection = x265_atobool(value, bError);
        if (bError || svtHevcParam->sceneChangeDetection)
        {
            bError = false;
            svtHevcParam->sceneChangeDetection = 1;
        }
    }
    OPT("open-gop")
    {
        if (x265_atobool(value, bError))
            svtHevcParam->intraRefreshType = 1;
        else
            svtHevcParam->intraRefreshType = 2;
    }
    OPT("deblock")
    {
        if (strtol(value, NULL, 0))
            svtHevcParam->disableDlfFlag = 0;
        else if (x265_atobool(value, bError) == 0 && !bError)
            svtHevcParam->disableDlfFlag = 1;
    }
    OPT("sao") svtHevcParam->enableSaoFlag = (uint8_t)x265_atobool(value, bError);
    OPT("keyint") svtHevcParam->intraPeriodLength = atoi(value);
    OPT2("constrained-intra", "cip") svtHevcParam->constrainedIntra = (uint8_t)x265_atobool(value, bError);
    OPT("vui-timing-info") svtHevcParam->videoUsabilityInfo = x265_atobool(value, bError);
    OPT("hdr") svtHevcParam->highDynamicRangeInput = x265_atobool(value, bError);
    OPT("aud") svtHevcParam->accessUnitDelimiter = x265_atobool(value, bError);
    OPT("qp")
    {
        svtHevcParam->rateControlMode = 0;
        svtHevcParam->qp = atoi(value);
    }
    OPT("bitrate")
    {
        svtHevcParam->rateControlMode = 1;
        svtHevcParam->targetBitRate = atoi(value);
    }
    OPT("interlace")
    {
        svtHevcParam->interlacedVideo = (uint8_t)x265_atobool(value, bError);
        if (bError || svtHevcParam->interlacedVideo)
        {
            bError = false;
            svtHevcParam->interlacedVideo = 1;
        }
    }
    OPT("svt-hme")
    {
        svtHevcParam->enableHmeFlag = (uint8_t)x265_atobool(value, bError);
        if (svtHevcParam->enableHmeFlag) svtHevcParam->useDefaultMeHme = 1;
    }
    OPT("svt-search-width") svtHevcParam->searchAreaWidth = atoi(value);
    OPT("svt-search-height") svtHevcParam->searchAreaHeight = atoi(value);
    OPT("svt-compressed-ten-bit-format") svtHevcParam->compressedTenBitFormat = x265_atobool(value, bError);
    OPT("svt-speed-control") svtHevcParam->speedControlFlag = x265_atobool(value, bError);
    OPT("svt-preset-tuner")
    {
        if (svtHevcParam->encMode == 2)
        {
            if (!strcmp(value, "0")) svtHevcParam->encMode = 0;
            else if (!strcmp(value, "1")) svtHevcParam->encMode = 1;
            else
            {
                x265_log(param, X265_LOG_ERROR, " Unsupported value=%s for svt-preset-tuner \n", value);
                bError = true;
            }
        }
        else
            x265_log(param, X265_LOG_WARNING, " svt-preset-tuner should be used only with ultrafast preset; Ignoring it \n");
    }
    OPT("svt-hierarchical-level") svtHevcParam->hierarchicalLevels = atoi(value);
    OPT("svt-base-layer-switch-mode") svtHevcParam->baseLayerSwitchMode = atoi(value);
    OPT("svt-pred-struct") svtHevcParam->predStructure = (uint8_t)atoi(value);
    OPT("svt-fps-in-vps") svtHevcParam->fpsInVps = (uint8_t)x265_atobool(value, bError);
    OPT("master-display") svtHevcParam->useMasteringDisplayColorVolume = (uint8_t)atoi(value);
    OPT("max-cll") bError |= sscanf(value, "%hu,%hu", &svtHevcParam->maxCLL, &svtHevcParam->maxFALL) != 2;
    OPT("nalu-file") svtHevcParam->useNaluFile = (uint8_t)atoi(value);
    OPT("dolby-vision-profile")
    {
        if (atof(value) < 10)
            svtHevcParam->dolbyVisionProfile = (int)(10 * atof(value) + .5);
        else if (atoi(value) < 100)
            svtHevcParam->dolbyVisionProfile = atoi(value);
        else
            bError = true;
    }
    OPT("hrd")
        svtHevcParam->hrdFlag = (uint32_t)x265_atobool(value, bError);
    OPT("vbv-maxrate")
        svtHevcParam->vbvMaxrate = (uint32_t)x265_atoi(value, bError);
    OPT("vbv-bufsize")
        svtHevcParam->vbvBufsize = (uint32_t)x265_atoi(value, bError);
    OPT("vbv-init")
        svtHevcParam->vbvBufInit = (uint64_t)x265_atof(value, bError);
    OPT("frame-threads")
        svtHevcParam->threadCount = (uint32_t)x265_atoi(value, bError);
    else
        x265_log(param, X265_LOG_INFO, "SVT doesn't support %s param; Disabling it \n", name);


    return bError ? X265_PARAM_BAD_VALUE : 0;
}

#endif //ifdef SVT_HEVC

}
