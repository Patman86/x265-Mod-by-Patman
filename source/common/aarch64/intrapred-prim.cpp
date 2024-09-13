#include "common.h"
#include "primitives.h"


#if HAVE_NEON
#include "arm64-utils.h"
#include <arm_neon.h>

using namespace X265_NS;

namespace
{


template<int tuSize>
void intraFilter_neon(const pixel* samples, pixel* filtered) /* 1:2:1 filtering of left and top reference samples */
{
    const int tuSize2 = tuSize << 1;
    pixel topLeft = samples[0], topLast = samples[tuSize2], leftLast = samples[tuSize2 + tuSize2];

    uint16x8_t two_vec = vdupq_n_u16(2);
#if !HIGH_BIT_DEPTH
    {
        for(int i = 0; i < tuSize2 + tuSize2; i+=8)
         {
            uint16x8_t sample1 = vmovl_u8(vld1_u8(&samples[i]));
            uint16x8_t sample2 = vmovl_u8(vld1_u8(&samples[i-1]));
            uint16x8_t sample3 = vmovl_u8(vld1_u8(&samples[i+1]));

            uint16x8_t result1 = vaddq_u16(vshlq_n_u16(sample1,1), sample2 );
            uint16x8_t result2 = vaddq_u16(sample3, two_vec);
            uint16x8_t result3 = vaddq_u16(result1,result2);
            vst1_u8(&filtered[i] , vmovn_u16(vshrq_n_u16(result3, 2)));
        }
    }
#else
    {
        for(int i = 0; i < tuSize2 + tuSize2; i+=8)
        {
            uint16x8_t sample1 = vld1q_u16(&samples[i]);
            uint16x8_t sample2 = vld1q_u16(&samples[i-1]);
            uint16x8_t sample3 = vld1q_u16(&samples[i+1]);

            uint16x8_t result1 = vaddq_u16(vshlq_n_u16(sample1,1), sample2 );
            uint16x8_t result2 = vaddq_u16(sample3, two_vec);
            uint16x8_t result3 = vaddq_u16(result1,result2);
            vst1q_u16(&filtered[i] , vshrq_n_u16(result3, 2));
        }
    }
#endif
    // filtering top
    filtered[tuSize2] = topLast;

    // filtering top-left
    filtered[0] = ((topLeft << 1) + samples[1] + samples[tuSize2 + 1] + 2) >> 2;

    // filtering left
    filtered[tuSize2 + 1] = ((samples[tuSize2 + 1] << 1) + topLeft + samples[tuSize2 + 2] + 2) >> 2;
    filtered[tuSize2 + tuSize2] = leftLast;
}

template<int width>
void intra_pred_ang_neon(pixel *dst, intptr_t dstStride, const pixel *srcPix0, int dirMode, int bFilter)
{
    int width2 = width << 1;
    // Flip the neighbours in the horizontal case.
    int horMode = dirMode < 18;
    pixel neighbourBuf[129];
    const pixel *srcPix = srcPix0;

    if (horMode)
    {
        neighbourBuf[0] = srcPix[0];
        //for (int i = 0; i < width << 1; i++)
        //{
        //    neighbourBuf[1 + i] = srcPix[width2 + 1 + i];
        //    neighbourBuf[width2 + 1 + i] = srcPix[1 + i];
        //}
        memcpy(&neighbourBuf[1], &srcPix[width2 + 1], sizeof(pixel) * (width << 1));
        memcpy(&neighbourBuf[width2 + 1], &srcPix[1], sizeof(pixel) * (width << 1));
        srcPix = neighbourBuf;
    }

    // Intra prediction angle and inverse angle tables.
    const int8_t angleTable[17] = { -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32 };
    const int16_t invAngleTable[8] = { 4096, 1638, 910, 630, 482, 390, 315, 256 };

    // Get the prediction angle.
    int angleOffset = horMode ? 10 - dirMode : dirMode - 26;
    int angle = angleTable[8 + angleOffset];

    // Vertical Prediction.
    if (!angle)
    {
        for (int y = 0; y < width; y++)
        {
            memcpy(&dst[y * dstStride], srcPix + 1, sizeof(pixel)*width);
        }
        if (bFilter)
        {
            int topLeft = srcPix[0], top = srcPix[1];
            for (int y = 0; y < width; y++)
            {
                dst[y * dstStride] = x265_clip((int16_t)(top + ((srcPix[width2 + 1 + y] - topLeft) >> 1)));
            }
        }
    }
    else // Angular prediction.
    {
        // Get the reference pixels. The reference base is the first pixel to the top (neighbourBuf[1]).
        pixel refBuf[64];
        const pixel *ref;

        // Use the projected left neighbours and the top neighbours.
        if (angle < 0)
        {
            // Number of neighbours projected.
            int nbProjected = -((width * angle) >> 5) - 1;
            pixel *ref_pix = refBuf + nbProjected + 1;

            // Project the neighbours.
            int invAngle = invAngleTable[- angleOffset - 1];
            int invAngleSum = 128;
            for (int i = 0; i < nbProjected; i++)
            {
                invAngleSum += invAngle;
                ref_pix[- 2 - i] = srcPix[width2 + (invAngleSum >> 8)];
            }

            // Copy the top-left and top pixels.
            //for (int i = 0; i < width + 1; i++)
            //ref_pix[-1 + i] = srcPix[i];

            memcpy(&ref_pix[-1], srcPix, (width + 1)*sizeof(pixel));
            ref = ref_pix;
        }
        else // Use the top and top-right neighbours.
        {
            ref = srcPix + 1;
        }

        // Pass every row.
        int angleSum = 0;
        for (int y = 0; y < width; y++)
        {
            angleSum += angle;
            int offset = angleSum >> 5;
            int fraction = angleSum & 31;

            if (fraction) // Interpolate
            {
                if (width >= 8 && sizeof(pixel) == 1)
                {
                    // We have to cast to the 'real' type so that this block
                    // will compile for both low and high bitdepth.
                    const uint8_t *ref_u8 = (const uint8_t *)ref + offset;
                    uint8_t *dst_u8 = (uint8_t *)dst;

                    // f0 and f1 are unsigned (fraction is in range [0, 31]).
                    const uint8x8_t f0 = vdup_n_u8(32 - fraction);
                    const uint8x8_t f1 = vdup_n_u8(fraction);
                    for (int x = 0; x < width; x += 8)
                    {
                        uint8x8_t in0 = vld1_u8(ref_u8 + x);
                        uint8x8_t in1 = vld1_u8(ref_u8 + x + 1);
                        uint16x8_t lo = vmlal_u8(vdupq_n_u16(16), in0, f0);
                        lo = vmlal_u8(lo, in1, f1);
                        uint8x8_t res = vshrn_n_u16(lo, 5);
                        vst1_u8(dst_u8 + y * dstStride + x, res);
                    }
                }
                else if (width >= 4 && sizeof(pixel) == 2)
                {
                    // We have to cast to the 'real' type so that this block
                    // will compile for both low and high bitdepth.
                    const uint16_t *ref_u16 = (const uint16_t *)ref + offset;
                    uint16_t *dst_u16 = (uint16_t *)dst;

                    // f0 and f1 are unsigned (fraction is in range [0, 31]).
                    const uint16x4_t f0 = vdup_n_u16(32 - fraction);
                    const uint16x4_t f1 = vdup_n_u16(fraction);
                    for (int x = 0; x < width; x += 4)
                    {
                        uint16x4_t in0 = vld1_u16(ref_u16 + x);
                        uint16x4_t in1 = vld1_u16(ref_u16 + x + 1);
                        uint32x4_t lo = vmlal_u16(vdupq_n_u32(16), in0, f0);
                        lo = vmlal_u16(lo, in1, f1);
                        uint16x4_t res = vshrn_n_u32(lo, 5);
                        vst1_u16(dst_u16 + y * dstStride + x, res);
                    }
                }
                else
                {
                    for (int x = 0; x < width; x++)
                    {
                        dst[y * dstStride + x] = (pixel)(((32 - fraction) * ref[offset + x] + fraction * ref[offset + x + 1] + 16) >> 5);
                    }
                }
            }
            else // Copy.
            {
                memcpy(&dst[y * dstStride], &ref[offset], sizeof(pixel)*width);
            }
        }
    }

    // Flip for horizontal.
    if (horMode)
    {
        if (width == 8)
        {
            transpose8x8(dst, dst, dstStride, dstStride);
        }
        else if (width == 16)
        {
            transpose16x16(dst, dst, dstStride, dstStride);
        }
        else if (width == 32)
        {
            transpose32x32(dst, dst, dstStride, dstStride);
        }
        else
        {
            for (int y = 0; y < width - 1; y++)
            {
                for (int x = y + 1; x < width; x++)
                {
                    pixel tmp              = dst[y * dstStride + x];
                    dst[y * dstStride + x] = dst[x * dstStride + y];
                    dst[x * dstStride + y] = tmp;
                }
            }
        }
    }
}

#endif
template<int log2Size>
void all_angs_pred_neon(pixel *dest, pixel *refPix, pixel *filtPix, int bLuma)
{
    const int size = 1 << log2Size;
    for (int mode = 2; mode <= 34; mode++)
    {
        pixel *srcPix  = (g_intraFilterFlags[mode] & size ? filtPix  : refPix);
        pixel *out = dest + ((mode - 2) << (log2Size * 2));

        intra_pred_ang_neon<size>(out, size, srcPix, mode, bLuma);

        // Optimize code don't flip buffer
        bool modeHor = (mode < 18);

        // transpose the block if this is a horizontal mode
        if (modeHor)
        {
            if (size == 8)
            {
                transpose8x8(out, out, size, size);
            }
            else if (size == 16)
            {
                transpose16x16(out, out, size, size);
            }
            else if (size == 32)
            {
                transpose32x32(out, out, size, size);
            }
            else
            {
                for (int k = 0; k < size - 1; k++)
                {
                    for (int l = k + 1; l < size; l++)
                    {
                        pixel tmp         = out[k * size + l];
                        out[k * size + l] = out[l * size + k];
                        out[l * size + k] = tmp;
                    }
                }
            }
        }
    }
}

template<int log2Size>
void planar_pred_neon(pixel * dst, intptr_t dstStride, const pixel * srcPix, int /*dirMode*/, int /*bFilter*/)
{
    const int blkSize = 1 << log2Size;

    const pixel* above = srcPix + 1;
    const pixel* left = srcPix + (2 * blkSize + 1);

    switch (blkSize) {
    case 8:
    {
        const uint16_t log2SizePlusOne = log2Size + 1;
        uint16x8_t blkSizeVec = vdupq_n_u16(blkSize);
        uint16x8_t topRight = vdupq_n_u16(above[blkSize]);
        uint16_t bottomLeft = left[blkSize];
        uint16x8_t oneVec = vdupq_n_u16(1);
        uint16x8_t blkSizeSubOneVec = vdupq_n_u16(blkSize - 1);

        for (int y = 0; y < blkSize; y++) {
            // (blkSize - 1 - y)
            uint16x8_t vlkSizeYVec = vdupq_n_u16(blkSize - 1 - y);
            // (y + 1) * bottomLeft
            uint16x8_t bottomLeftYVec = vdupq_n_u16((y + 1) * bottomLeft);
            // left[y]
            uint16x8_t leftYVec = vdupq_n_u16(left[y]);

            for (int x = 0; x < blkSize; x += 8) {
                int idx = y * dstStride + x;
                uint16x8_t xvec = { (uint16_t)(x + 0), (uint16_t)(x + 1),
                                    (uint16_t)(x + 2), (uint16_t)(x + 3),
                                    (uint16_t)(x + 4), (uint16_t)(x + 5),
                                    (uint16_t)(x + 6), (uint16_t)(x + 7) };

                // (blkSize - 1 - y) * above[x]
                uint16x8_t aboveVec = { (uint16_t)(above[x + 0]),
                                        (uint16_t)(above[x + 1]),
                                        (uint16_t)(above[x + 2]),
                                        (uint16_t)(above[x + 3]),
                                        (uint16_t)(above[x + 4]),
                                        (uint16_t)(above[x + 5]),
                                        (uint16_t)(above[x + 6]),
                                        (uint16_t)(above[x + 7]) };

                aboveVec = vmulq_u16(aboveVec, vlkSizeYVec);

                // (blkSize - 1 - x) * left[y]
                uint16x8_t first = vsubq_u16(blkSizeSubOneVec, xvec);
                first = vmulq_u16(first, leftYVec);

                // (x + 1) * topRight
                uint16x8_t second = vaddq_u16(xvec, oneVec);
                second = vmulq_u16(second, topRight);

                uint16x8_t resVec = vaddq_u16(first, second);
                resVec = vaddq_u16(resVec, aboveVec);
                resVec = vaddq_u16(resVec, bottomLeftYVec);
                resVec = vaddq_u16(resVec, blkSizeVec);
                resVec = vshrq_n_u16(resVec, log2SizePlusOne);

                for (int i = 0; i < 8; i++)
                    dst[idx + i] = (pixel)resVec[i];
    }
}
        }
    break;
    case 4:
    case 32:
    case 16:
    {
        const uint32_t log2SizePlusOne = log2Size + 1;
        uint32x4_t blkSizeVec = vdupq_n_u32(blkSize);
        uint32x4_t topRight = vdupq_n_u32(above[blkSize]);
        uint32_t bottomLeft = left[blkSize];
        uint32x4_t oneVec = vdupq_n_u32(1);
        uint32x4_t blkSizeSubOneVec = vdupq_n_u32(blkSize - 1);

        for (int y = 0; y < blkSize; y++) {
            // (blkSize - 1 - y)
            uint32x4_t vlkSizeYVec = vdupq_n_u32(blkSize - 1 - y);
            // (y + 1) * bottomLeft
            uint32x4_t bottomLeftYVec = vdupq_n_u32((y + 1) * bottomLeft);
            // left[y]
            uint32x4_t leftYVec = vdupq_n_u32(left[y]);

            for (int x = 0; x < blkSize; x += 4) {
                int idx = y * dstStride + x;
                uint32x4_t xvec = { (uint32_t)(x + 0), (uint32_t)(x + 1),
                                    (uint32_t)(x + 2), (uint32_t)(x + 3) };

                // (blkSize - 1 - y) * above[x]
                uint32x4_t aboveVec = { (uint32_t)(above[x + 0]),
                                        (uint32_t)(above[x + 1]),
                                        (uint32_t)(above[x + 2]),
                                        (uint32_t)(above[x + 3]) };
                aboveVec = vmulq_u32(aboveVec, vlkSizeYVec);

                // (blkSize - 1 - x) * left[y]
                uint32x4_t first = vsubq_u32(blkSizeSubOneVec, xvec);
                first = vmulq_u32(first, leftYVec);

                // (x + 1) * topRight
                uint32x4_t second = vaddq_u32(xvec, oneVec);
                second = vmulq_u32(second, topRight);

                uint32x4_t resVec = vaddq_u32(first, second);
                resVec = vaddq_u32(resVec, aboveVec);
                resVec = vaddq_u32(resVec, bottomLeftYVec);
                resVec = vaddq_u32(resVec, blkSizeVec);
                resVec = vshrq_n_u32(resVec, log2SizePlusOne);

                for (int i = 0; i < 4; i++)
                    dst[idx + i] = (pixel)resVec[i];
            }
        }
    }
    break;
        }
}

static void dcPredFilter(const pixel* above, const pixel* left, pixel* dst, intptr_t dststride, int size)
{
    // boundary pixels processing
    pixel topLeft = (pixel)((above[0] + left[0] + 2 * dst[0] + 2) >> 2);
    pixel * pdst = dst;

    switch (size) {
    case 32:
    case 16:
    case 8:
    {
        uint16x8_t vconst_3 = vdupq_n_u16(3);
        uint16x8_t vconst_2 = vdupq_n_u16(2);
        for (int x = 0; x < size; x += 8) {
            uint16x8_t vabo = { (uint16_t)(above[x + 0]),
                                (uint16_t)(above[x + 1]),
                                (uint16_t)(above[x + 2]),
                                (uint16_t)(above[x + 3]),
                                (uint16_t)(above[x + 4]),
                                (uint16_t)(above[x + 5]),
                                (uint16_t)(above[x + 6]),
                                (uint16_t)(above[x + 7]) };

            uint16x8_t vdst = { (uint16_t)(dst[x + 0]),
                                (uint16_t)(dst[x + 1]),
                                (uint16_t)(dst[x + 2]),
                                (uint16_t)(dst[x + 3]),
                                (uint16_t)(dst[x + 4]),
                                (uint16_t)(dst[x + 5]),
                                (uint16_t)(dst[x + 6]),
                                (uint16_t)(dst[x + 7]) };
            //  dst[x] = (pixel)((above[x] +  3 * dst[x] + 2) >> 2);
            vdst = vmulq_u16(vdst, vconst_3);
            vdst = vaddq_u16(vdst, vabo);
            vdst = vaddq_u16(vdst, vconst_2);
            vdst = vshrq_n_u16(vdst, 2);
            for (int i = 0; i < 8; i++)
                dst[x + i] = (pixel)(vdst[i]);
        }
        dst += dststride;
        for (int y = 1; y < size; y++)
        {
            *dst = (pixel)((left[y] + 3 * *dst + 2) >> 2);
            dst += dststride;
        }
    }
    break;
    case 4:
    {
        uint16x4_t vconst_3 = vdup_n_u16(3);
        uint16x4_t vconst_2 = vdup_n_u16(2);
        uint16x4_t vabo = { (uint16_t)(above[0]),
                            (uint16_t)(above[1]),
                            (uint16_t)(above[2]),
                            (uint16_t)(above[3]) };
        uint16x4_t vdstx = { (uint16_t)(dst[0]),
                             (uint16_t)(dst[1]),
                             (uint16_t)(dst[2]),
                             (uint16_t)(dst[3]) };
        vdstx = vmul_u16(vdstx, vconst_3);
        vdstx = vadd_u16(vdstx, vabo);
        vdstx = vadd_u16(vdstx, vconst_2);
        vdstx = vshr_n_u16(vdstx, 2);
        for (int i = 0; i < 4; i++)
            dst[i] = (pixel)(vdstx[i]);

        dst += dststride;
        for (int y = 1; y < size; y++)
        {
            *dst = (pixel)((left[y] + 3 * *dst + 2) >> 2);
            dst += dststride;
        }
    }
    break;
    }

    *pdst = topLeft;
}

template<int width>
void intra_pred_dc_neon(pixel* dst, intptr_t dstStride, const pixel* srcPix, int /*dirMode*/, int bFilter)
{
    int k, l;
    int dcVal = width;

    switch (width) {
    case 32:
    case 16:
    case 8:
    {
        for (int i = 0; i < width; i += 8) {
            uint16x8_t spa = { (uint16_t)(srcPix[i + 1]),
                               (uint16_t)(srcPix[i + 2]),
                               (uint16_t)(srcPix[i + 3]),
                               (uint16_t)(srcPix[i + 4]),
                               (uint16_t)(srcPix[i + 5]),
                               (uint16_t)(srcPix[i + 6]),
                               (uint16_t)(srcPix[i + 7]),
                               (uint16_t)(srcPix[i + 8]) };
            uint16x8_t spb = { (uint16_t)(srcPix[2 * width + i + 1]),
                               (uint16_t)(srcPix[2 * width + i + 2]),
                               (uint16_t)(srcPix[2 * width + i + 3]),
                               (uint16_t)(srcPix[2 * width + i + 4]),
                               (uint16_t)(srcPix[2 * width + i + 5]),
                               (uint16_t)(srcPix[2 * width + i + 6]),
                               (uint16_t)(srcPix[2 * width + i + 7]),
                               (uint16_t)(srcPix[2 * width + i + 8]) };
            uint16x8_t vsp = vaddq_u16(spa, spb);
            dcVal += vaddlvq_u16(vsp);
        }

        dcVal = dcVal / (width + width);
        for (k = 0; k < width; k++)
            for (l = 0; l < width; l += 8) {
                uint16x8_t vdv = vdupq_n_u16((pixel)dcVal);
                for (int n = 0; n < 8; n++)
                    dst[k * dstStride + l + n] = (pixel)(vdv[n]);
            }
    }
    break;
    case 4:
    {
        uint16x4_t spa = { (uint16_t)(srcPix[1]), (uint16_t)(srcPix[2]),
                           (uint16_t)(srcPix[3]), (uint16_t)(srcPix[4]) };
        uint16x4_t spb = { (uint16_t)(srcPix[2 * width + 1]),
                           (uint16_t)(srcPix[2 * width + 2]),
                           (uint16_t)(srcPix[2 * width + 3]),
                           (uint16_t)(srcPix[2 * width + 4]) };
        uint16x4_t vsp = vadd_u16(spa, spb);
        dcVal += vaddlv_u16(vsp);

        dcVal = dcVal / (width + width);
        for (k = 0; k < width; k++) {
            uint16x4_t vdv = vdup_n_u16((pixel)dcVal);
            for (int n = 0; n < 4; n++)
                dst[k * dstStride + n] = (pixel)(vdv[n]);
        }
    }
    break;
    }

    if (bFilter)
        dcPredFilter(srcPix + 1, srcPix + (2 * width + 1), dst, dstStride, width);
}
}

namespace X265_NS
{
// x265 private namespace
extern "C" void PFX(intra_pred_planar8_neon)(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
extern "C" void PFX(intra_pred_planar16_neon)(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);

void setupIntraPrimitives_neon(EncoderPrimitives &p)
{
    p.cu[BLOCK_4x4].intra_filter = intraFilter_neon<4>;
    p.cu[BLOCK_8x8].intra_filter = intraFilter_neon<8>;
    p.cu[BLOCK_16x16].intra_filter = intraFilter_neon<16>;
    p.cu[BLOCK_32x32].intra_filter = intraFilter_neon<32>;

    for (int i = 2; i < NUM_INTRA_MODE; i++)
    {
        p.cu[BLOCK_8x8].intra_pred[i] = intra_pred_ang_neon<8>;
        p.cu[BLOCK_16x16].intra_pred[i] = intra_pred_ang_neon<16>;
        p.cu[BLOCK_32x32].intra_pred[i] = intra_pred_ang_neon<32>;
    }
    p.cu[BLOCK_4x4].intra_pred[2] = intra_pred_ang_neon<4>;
    p.cu[BLOCK_4x4].intra_pred[10] = intra_pred_ang_neon<4>;
    p.cu[BLOCK_4x4].intra_pred[18] = intra_pred_ang_neon<4>;
    p.cu[BLOCK_4x4].intra_pred[26] = intra_pred_ang_neon<4>;
    p.cu[BLOCK_4x4].intra_pred[34] = intra_pred_ang_neon<4>;

    p.cu[BLOCK_4x4].intra_pred_allangs = all_angs_pred_neon<2>;
    p.cu[BLOCK_8x8].intra_pred_allangs = all_angs_pred_neon<3>;
    p.cu[BLOCK_16x16].intra_pred_allangs = all_angs_pred_neon<4>;
    p.cu[BLOCK_32x32].intra_pred_allangs = all_angs_pred_neon<5>;

#if !HIGH_BIT_DEPTH
    p.cu[BLOCK_8x8].intra_pred[PLANAR_IDX] = PFX(intra_pred_planar8_neon);
    p.cu[BLOCK_16x16].intra_pred[PLANAR_IDX] = PFX(intra_pred_planar16_neon);
#endif

    p.cu[BLOCK_4x4].intra_pred[DC_IDX] = intra_pred_dc_neon<4>;
    p.cu[BLOCK_8x8].intra_pred[DC_IDX] = intra_pred_dc_neon<8>;
    p.cu[BLOCK_16x16].intra_pred[DC_IDX] = intra_pred_dc_neon<16>;
    p.cu[BLOCK_32x32].intra_pred[DC_IDX] = intra_pred_dc_neon<32>;
}
}



