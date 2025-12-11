/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Jia Yuan <yuan.jia@sanechips.com.cn>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at license @ x265.com.
 *****************************************************************************/

#include "common.h"
#include "primitives.h"

#include "riscv64_utils.h"
#include <riscv_vector.h>
#include <stdint.h>

namespace X265_NS
{
template<int tuSize>
void intraFilter_rvv(const pixel* samples, pixel* filtered) /* 1:2:1 filtering of left and top reference samples */
{
    const int tuSize2 = tuSize << 1;
    const int len = tuSize2 + tuSize2;

    pixel topLeft = samples[0];
    pixel topLast = samples[tuSize2];
    pixel leftLast = samples[len];

#if !HIGH_BIT_DEPTH
    {
        size_t vl = __riscv_vsetvl_e16m2(len);
        vuint16m2_t two_vec = __riscv_vmv_v_x_u16m2(2, vl);
        for(int i = 0; i < len; i+=vl) {
            vl = __riscv_vsetvl_e8m1(len - i);
            vuint8m1_t sample1_u8 = __riscv_vle8_v_u8m1(&samples[i], vl);
            vuint8m1_t sample2_u8 = __riscv_vle8_v_u8m1(&samples[i-1], vl);
            vuint8m1_t sample3_u8 = __riscv_vle8_v_u8m1(&samples[i+1], vl);

            vuint16m2_t sample1 = __riscv_vzext_vf2_u16m2(sample1_u8, vl);
            vuint16m2_t sample2 = __riscv_vzext_vf2_u16m2(sample2_u8, vl);
            vuint16m2_t sample3 = __riscv_vzext_vf2_u16m2(sample3_u8, vl);

            vuint16m2_t result1 =  __riscv_vsll_vx_u16m2(sample1, 1, vl);
            result1 = __riscv_vadd_vv_u16m2(result1, sample2, vl);
            vuint16m2_t result2 =  __riscv_vadd_vv_u16m2(sample3, two_vec, vl);
            vuint16m2_t result3 =  __riscv_vadd_vv_u16m2(result1, result2, vl);
            result3 = __riscv_vsrl_vx_u16m2(result3, 2, vl);

            vuint8m1_t result_u8 = __riscv_vnsrl_wx_u8m1(result3, 0, vl);
            __riscv_vse8_v_u8m1(&filtered[i], result_u8, vl);
        }
    }
#else
    {
        size_t vl = __riscv_vsetvl_e16m1(len);
        vuint16m1_t two_vec = __riscv_vmv_v_x_u16m1(2, vl);
        for(int i = 0; i < len; i += vl) {
            vl = __riscv_vsetvl_e16m1(len - i);
            vuint16m1_t sample1 = __riscv_vle16_v_u16m1(&samples[i], vl);
            vuint16m1_t sample2 = __riscv_vle16_v_u16m1(&samples[i-1], vl);
            vuint16m1_t sample3 = __riscv_vle16_v_u16m1(&samples[i+1], vl);

            vuint16m1_t result1 = __riscv_vsll_vx_u16m1(sample1, 1, vl);
            result1 = __riscv_vadd_vv_u16m1(result1, sample2, vl);
            vuint16m1_t result2 = __riscv_vadd_vv_u16m1(sample3, two_vec, vl);
            vuint16m1_t result3 = __riscv_vadd_vv_u16m1(result1, result2, vl);

            result3 = __riscv_vsrl_vx_u16m1(result3, 2, vl);
            __riscv_vse16_v_u16m1(&filtered[i], result3, vl);
        }
    }
#endif
    filtered[tuSize2] = topLast;
    filtered[0] = ((topLeft << 1) + samples[1] + samples[tuSize2 + 1] + 2) >> 2;
    filtered[tuSize2 + 1] = ((samples[tuSize2 + 1] << 1) + topLeft + samples[tuSize2 + 2] + 2) >> 2;
    filtered[tuSize2 + tuSize2] = leftLast;
}

template<int width>
void intra_pred_ang_rvv(pixel *dst, intptr_t dstStride, const pixel *srcPix0, int dirMode, int bFilter)
{
    int width2 = width << 1;
    int horMode = dirMode < 18;
    pixel neighbourBuf[129];
    const pixel *srcPix = srcPix0;

    if (horMode) {
        neighbourBuf[0] = srcPix[0];
        if (width <= 8 && sizeof(pixel) == 1) {
            const size_t vl = (size_t)(width << 1);
            vuint8m1_t v1 = __riscv_vle8_v_u8m1((const uint8_t *)&srcPix[width2 + 1], vl);
            vuint8m1_t v2 = __riscv_vle8_v_u8m1((const uint8_t *)&srcPix[1], vl);
            __riscv_vse8_v_u8m1((uint8_t *)&neighbourBuf[1], v1, vl);
            __riscv_vse8_v_u8m1((uint8_t *)&neighbourBuf[width2 + 1], v2, vl);
            srcPix = neighbourBuf;
        } else {
            memcpy(&neighbourBuf[1], &srcPix[width2 + 1], sizeof(pixel) * (width << 1));
            memcpy(&neighbourBuf[width2 + 1], &srcPix[1], sizeof(pixel) * (width << 1));
            srcPix = neighbourBuf;
        }

    }

    const int8_t angleTable[17] = {-32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32};
    const int16_t invAngleTable[8] = {4096, 1638, 910, 630, 482, 390, 315, 256};

    int angleOffset = horMode ? 10 - dirMode : dirMode - 26;
    int angle = angleTable[8 + angleOffset];

    if (!angle) {
        for (int y = 0; y < width; y++) {
            memcpy(&dst[y * dstStride], srcPix + 1, sizeof(pixel)*width);
        }
        if (bFilter) {
            int topLeft = srcPix[0], top = srcPix[1];
            for (int y = 0; y < width; y++) {
                dst[y * dstStride] = x265_clip((int16_t)(top + ((srcPix[width2 + 1 + y] - topLeft) >> 1)));
            }
        }
    } else {
        pixel refBuf[64];
        const pixel *ref;

        if (angle < 0) {
            int nbProjected = -((width * angle) >> 5) - 1;
            pixel *ref_pix = refBuf + nbProjected + 1;

            int invAngle = invAngleTable[- angleOffset - 1];
            int invAngleSum = 128;

            for (int i = 0; i < nbProjected; i++) {
                invAngleSum += invAngle;
                ref_pix[- 2 - i] = srcPix[width2 + (invAngleSum >> 8)];
            }
            memcpy(&ref_pix[-1], srcPix, (width + 1)*sizeof(pixel));
            ref = ref_pix;
        } else {
            ref = srcPix + 1;
        }
        int angleSum = 0;
        for (int y = 0; y < width; y++) {
            angleSum += angle;
            int offset = angleSum >> 5;
            int fraction = angleSum & 31;
            if (fraction) {
                size_t vl = width;

                #if !HIGH_BIT_DEPTH
                    const uint8_t *ref_u8 = (const uint8_t *)ref + offset;
                    uint8_t *dst_u8 = (uint8_t *)dst;
                #else
                    const uint16_t *ref_u16 = (const uint16_t *)ref + offset;
                    uint16_t *dst_u16 = (uint16_t *)dst;
                #endif

                switch (width) {
                case 32: {
                #if !HIGH_BIT_DEPTH
                    vuint8m2_t in0 = __riscv_vle8_v_u8m2(ref_u8, vl);
                    vuint8m2_t in1 = __riscv_vle8_v_u8m2(ref_u8 + 1, vl);

                    vuint8m2_t f0 = __riscv_vmv_v_x_u8m2(32 - fraction, vl);
                    vuint8m2_t f1 = __riscv_vmv_v_x_u8m2(fraction, vl);

                    vuint16m4_t temp0 = __riscv_vwmulu_vv_u16m4(in0, f0, vl);
                    vuint16m4_t temp1 = __riscv_vwmulu_vv_u16m4(in1, f1, vl);

                    vuint16m4_t sum = __riscv_vadd_vv_u16m4(temp0, temp1, vl);
                    vuint16m4_t sixteen = __riscv_vmv_v_x_u16m4(16, vl);
                    sum = __riscv_vadd_vv_u16m4(sum, sixteen, vl);

                    vuint8m2_t res = __riscv_vnsrl_wx_u8m2(sum, 5, vl);
                    __riscv_vse8_v_u8m2(dst_u8 + y * dstStride, res, vl);
                #else
                    vuint16m2_t f0 = __riscv_vmv_v_x_u16m2(32 - fraction, vl);
                    vuint16m2_t f1 = __riscv_vmv_v_x_u16m2(fraction, vl);
                    vuint32m4_t sixteen = __riscv_vmv_v_x_u32m4(16, vl);

                    for (int x = 0; x < width; x += vl) {
                        vl = __riscv_vsetvl_e16m2(width - x);
                        vuint16m2_t in0 = __riscv_vle16_v_u16m2(ref_u16 + x, vl);
                        vuint16m2_t in1 = __riscv_vle16_v_u16m2(ref_u16 + x + 1, vl);

                        vuint32m4_t temp0 = __riscv_vwmulu_vv_u32m4(in0, f0, vl);
                        vuint32m4_t temp1 = __riscv_vwmulu_vv_u32m4(in1, f1, vl);
                        vuint32m4_t sum = __riscv_vadd_vv_u32m4(temp0, temp1, vl);
                        sum = __riscv_vadd_vv_u32m4(sum, sixteen, vl);

                        vuint16m2_t res = __riscv_vnsrl_wx_u16m2(sum, 5, vl);
                        __riscv_vse16_v_u16m2(dst_u16 + y * dstStride + x, res, vl);
                    }
                #endif

                }
                break;
                case 16: {
                #if !HIGH_BIT_DEPTH
                    vuint8m1_t in0 = __riscv_vle8_v_u8m1(ref_u8, vl);
                    vuint8m1_t in1 = __riscv_vle8_v_u8m1(ref_u8 + 1, vl);

                    vuint8m1_t f0 = __riscv_vmv_v_x_u8m1(32 - fraction, vl);
                    vuint8m1_t f1 = __riscv_vmv_v_x_u8m1(fraction, vl);

                    vuint16m2_t temp0 = __riscv_vwmulu_vv_u16m2(in0, f0, vl);
                    vuint16m2_t temp1 = __riscv_vwmulu_vv_u16m2(in1, f1, vl);
                    vuint16m2_t sum = __riscv_vadd_vv_u16m2(temp0, temp1, vl);
                    vuint16m2_t sixteen = __riscv_vmv_v_x_u16m2(16, vl);
                    sum = __riscv_vadd_vv_u16m2(sum, sixteen, vl);

                    vuint8m1_t res = __riscv_vnsrl_wx_u8m1(sum, 5, vl);
                    __riscv_vse8_v_u8m1(dst_u8 + y * dstStride, res, vl);
                #else
                    vuint16m2_t in0 = __riscv_vle16_v_u16m2(ref_u16, vl);
                    vuint16m2_t in1 = __riscv_vle16_v_u16m2(ref_u16 + 1, vl);

                    vuint16m2_t f0 = __riscv_vmv_v_x_u16m2(32 - fraction, vl);
                    vuint16m2_t f1 = __riscv_vmv_v_x_u16m2(fraction, vl);

                    vuint32m4_t temp0 = __riscv_vwmulu_vv_u32m4(in0, f0, vl);
                    vuint32m4_t temp1 = __riscv_vwmulu_vv_u32m4(in1, f1, vl);

                    vuint32m4_t sum = __riscv_vadd_vv_u32m4(temp0, temp1, vl);
                    vuint16m2_t sixteen = __riscv_vmv_v_x_u16m2(16, vl);
                    sum = __riscv_vwaddu_wv_u32m4(sum, sixteen, vl);

                    vuint16m2_t res = __riscv_vnsrl_wx_u16m2(sum, 5, vl);
                    __riscv_vse16_v_u16m2(dst_u16 + y * dstStride, res, vl);
                #endif
                }
                break;
                case 8: {
                #if !HIGH_BIT_DEPTH
                    vuint8mf2_t in0 = __riscv_vle8_v_u8mf2(ref_u8, vl);
                    vuint8mf2_t in1 = __riscv_vle8_v_u8mf2(ref_u8 + 1, vl);

                    vuint8mf2_t f0 = __riscv_vmv_v_x_u8mf2(32 - fraction, vl);
                    vuint8mf2_t f1 = __riscv_vmv_v_x_u8mf2(fraction, vl);

                    vuint16m1_t temp0 = __riscv_vwmulu_vv_u16m1(in0, f0, vl);
                    vuint16m1_t temp1 = __riscv_vwmulu_vv_u16m1(in1, f1, vl);

                    vuint16m1_t sum = __riscv_vadd_vv_u16m1(temp0, temp1, vl);
                    vuint16m1_t sixteen = __riscv_vmv_v_x_u16m1(16, vl);
                    sum = __riscv_vadd_vv_u16m1(sum, sixteen, vl);

                    vuint8mf2_t res = __riscv_vnsrl_wx_u8mf2(sum, 5, vl);
                    __riscv_vse8_v_u8mf2(dst_u8 + y * dstStride, res, vl);
                #else
                    vuint16m1_t in0 = __riscv_vle16_v_u16m1(ref_u16, vl);
                    vuint16m1_t in1 = __riscv_vle16_v_u16m1(ref_u16 + 1, vl);

                    vuint16m1_t f0 = __riscv_vmv_v_x_u16m1(32 - fraction, vl);
                    vuint16m1_t f1 = __riscv_vmv_v_x_u16m1(fraction, vl);

                    vuint32m2_t temp0 = __riscv_vwmulu_vv_u32m2(in0, f0, vl);
                    vuint32m2_t temp1 = __riscv_vwmulu_vv_u32m2(in1, f1, vl);

                    vuint32m2_t sum = __riscv_vadd_vv_u32m2(temp0, temp1, vl);
                    vuint16m1_t sixteen = __riscv_vmv_v_x_u16m1(16, vl);
                    sum = __riscv_vwaddu_wv_u32m2(sum, sixteen, vl);

                    vuint16m1_t res = __riscv_vnsrl_wx_u16m1(sum, 5, vl);
                    __riscv_vse16_v_u16m1(dst_u16 + y * dstStride, res, vl);
                #endif

                }
                break;
                case 4: {
                #if !HIGH_BIT_DEPTH
                    vuint8mf4_t in0 = __riscv_vle8_v_u8mf4(ref_u8, vl);
                    vuint8mf4_t in1 = __riscv_vle8_v_u8mf4(ref_u8 + 1, vl);
                    vuint16mf2_t sixteen = __riscv_vmv_v_x_u16mf2(16, vl);

                    vuint8mf4_t f0 = __riscv_vmv_v_x_u8mf4(32 - fraction, vl);
                    vuint8mf4_t f1 = __riscv_vmv_v_x_u8mf4(fraction, vl);

                    vuint16mf2_t temp0 = __riscv_vwmulu_vv_u16mf2(in0, f0, vl);
                    vuint16mf2_t temp1 = __riscv_vwmulu_vv_u16mf2(in1, f1, vl);
                    vuint16mf2_t temp = __riscv_vadd_vv_u16mf2(temp0, temp1, vl);

                    vuint16mf2_t sum = __riscv_vadd_vv_u16mf2(temp, sixteen, vl);
                    vuint8mf4_t res = __riscv_vnsrl_wx_u8mf4(sum, 5, vl);
                    __riscv_vse8_v_u8mf4(dst_u8 + y * dstStride, res, vl);
                #else
                    vuint16mf2_t in0 = __riscv_vle16_v_u16mf2(ref_u16, vl);
                    vuint16mf2_t in1 = __riscv_vle16_v_u16mf2(ref_u16 + 1, vl);

                    vuint16mf2_t f0 = __riscv_vmv_v_x_u16mf2(32 - fraction, vl);
                    vuint16mf2_t f1 = __riscv_vmv_v_x_u16mf2(fraction, vl);

                    vuint32m1_t temp0 = __riscv_vwmulu_vv_u32m1(in0, f0, vl);
                    vuint32m1_t temp1 = __riscv_vwmulu_vv_u32m1(in1, f1, vl);

                    vuint32m1_t sum = __riscv_vadd_vv_u32m1(temp0, temp1, vl);
                    vuint16mf2_t sixteen = __riscv_vmv_v_x_u16mf2(16, vl);
                    sum = __riscv_vwaddu_wv_u32m1(sum, sixteen, vl);

                    vuint16mf2_t res = __riscv_vnsrl_wx_u16mf2(sum, 5, vl);
                    __riscv_vse16_v_u16mf2(dst_u16 + y * dstStride, res, vl);
                #endif
                }
                break;
                default: {
                    for (int x = 0; x < width; x++) {
                        dst[y * dstStride + x] = (pixel)(((32 - fraction) * ref[offset + x] + fraction * ref[offset + x + 1] + 16) >> 5);
                    }
                }
                break;
                }
            } else {
                memcpy(&dst[y * dstStride], &ref[offset], sizeof(pixel)*width);
            }
        }
    }
    if (horMode) {
        if (width == 4) {
            transpose4x4_rvv(dst, dst, dstStride, dstStride);
        } else if (width == 8) {
            transpose8x8_rvv(dst, dst, dstStride, dstStride);
        } else if (width == 16) {
            transpose16x16_rvv(dst, dst, dstStride, dstStride);
        } else if (width == 32) {
            transpose32x32_rvv(dst, dst, dstStride, dstStride);
        } else {
            for (int y = 0; y < width - 1; y++) {
                for (int x = y + 1; x < width; x++) {
                    pixel tmp = dst[y * dstStride + x];
                    dst[y * dstStride + x] = dst[x * dstStride + y];
                    dst[x * dstStride + y] = tmp;
                }
            }
        }
    }

}

template<int log2Size>
void all_angs_pred_rvv(pixel *dest, pixel *refPix, pixel *filtPix, int bLuma) {
    const int size = 1 << log2Size;
    for (int mode = 2; mode <= 34; mode++) {
        pixel *srcPix  = (g_intraFilterFlags[mode] & size ? filtPix  : refPix);
        pixel *out = dest + ((mode - 2) << (log2Size * 2));
        intra_pred_ang_rvv<size>(out, size, srcPix, mode, bLuma);

        bool modeHor = (mode < 18);
        if (modeHor) {
            if (size == 4) {
                transpose4x4_rvv(out, out, size, size);
            } else if (size == 8) {
                transpose8x8_rvv(out, out, size, size);
            } else if (size == 16) {
                transpose16x16_rvv(out, out, size, size);
            } else if (size == 32) {
                transpose32x32_rvv(out, out, size, size);
            } else {
                for (int k = 0; k < size - 1; k++) {
                    for (int l = k + 1; l < size; l++) {
                        pixel tmp = out[k * size + l];
                        out[k * size + l] = out[l * size + k];
                        out[l * size + k] = tmp;
                    }
                }
            }
        }
    }
}

#if HIGH_BIT_DEPTH
template<int log2Size>
void planar_pred_rvv(pixel * dst, intptr_t dstStride, const pixel * srcPix, int /*dirMode*/, int /*bFilter*/)
{
    const int blkSize = 1 << log2Size;

    const pixel* above = srcPix + 1;
    const pixel* left = srcPix + (2 * blkSize + 1);

    switch (blkSize) {
    case 4:
    {
        size_t vl = blkSize;
        vuint16mf2_t above0 = __riscv_vle16_v_u16mf2(above, vl);

        vuint16mf2_t topRight = __riscv_vmv_v_x_u16mf2(above[blkSize], vl);
        vuint16mf2_t bottomLeft = __riscv_vmv_v_x_u16mf2(left[blkSize], vl);

        vuint16mf2_t cx = __riscv_vmv_v_x_u16mf2(blkSize - 1, vl);
        vuint16mf2_t oneVec = __riscv_vmv_v_x_u16mf2(1, vl);
        vuint16mf2_t blkSizeVec = __riscv_vmv_v_x_u16mf2(blkSize, vl);

        // topRight constant
        vuint16mf2_t inc = __riscv_vid_v_u16mf2(vl);
        vuint16mf2_t tR0 = __riscv_vadd_vv_u16mf2(inc, oneVec, vl);

        // left constant
        vuint16mf2_t l0 = __riscv_vsub_vv_u16mf2(blkSizeVec, tR0, vl);

        vuint16mf2_t offset_bottomLeft = __riscv_vadd_vv_u16mf2(blkSizeVec, bottomLeft, vl);

        vuint16mf2_t t0 = __riscv_vmacc_vv_u16mf2(offset_bottomLeft, topRight, tR0, vl);
        t0 = __riscv_vmacc_vv_u16mf2(t0, above0, cx, vl);

        vuint16mf2_t sub_bottomLeft_above0 = __riscv_vsub_vv_u16mf2(bottomLeft, above0, vl);

        for (int y = 0; y < blkSize; y++) {
            vuint16mf2_t left_v = __riscv_vmv_v_x_u16mf2(left[y], vl);
            vuint16mf2_t r0 = __riscv_vmacc_vv_u16mf2(t0, left_v, l0, vl);
            vuint16mf2_t d0 = __riscv_vsrl_vx_u16mf2(r0, log2Size + 1, vl);
            __riscv_vse16_v_u16mf2(dst + y * dstStride, d0, vl);
            t0 = __riscv_vadd_vv_u16mf2(t0, sub_bottomLeft_above0, vl);
        }
    }
    break;
    case 8:
    {
        size_t vl = blkSize;
        vuint16m1_t above0 = __riscv_vle16_v_u16m1(above, vl);

        vuint16m1_t topRight = __riscv_vmv_v_x_u16m1(above[blkSize], vl);
        vuint16m1_t bottomLeft = __riscv_vmv_v_x_u16m1(left[blkSize], vl);
        vuint16m1_t cx = __riscv_vmv_v_x_u16m1(blkSize - 1, vl);
        vuint16m1_t oneVec = __riscv_vmv_v_x_u16m1(1, vl);
        vuint16m1_t blkSizeVec = __riscv_vmv_v_x_u16m1(blkSize, vl);

        // topRight constant
        vuint16m1_t inc = __riscv_vid_v_u16m1(vl);
        vuint16m1_t tR0 = __riscv_vadd_vv_u16m1(inc, oneVec, vl);

        // left constant
        vuint16m1_t l0 = __riscv_vsub_vv_u16m1(blkSizeVec, tR0, vl);

        vuint32m2_t offset = __riscv_vmv_v_x_u32m2(blkSize, vl);
        vuint32m2_t offset_bottomLeft = __riscv_vadd_vv_u32m2(
            offset, __riscv_vwcvtu_x_x_v_u32m2(bottomLeft, vl), vl);

        vuint32m2_t t0 = __riscv_vwmaccu_vv_u32m2(offset_bottomLeft, topRight, tR0, vl);
        t0 = __riscv_vwmaccu_vv_u32m2(t0, above0, cx, vl);

        vuint32m2_t sub_bottomLeft_above0 = __riscv_vwsubu_vv_u32m2(bottomLeft, above0, vl);

        for (int y = 0; y < blkSize; y++) {
            vuint16m1_t left_v = __riscv_vmv_v_x_u16m1(left[y], vl);
            vuint32m2_t r0 = __riscv_vwmaccu_vv_u32m2(t0, left_v, l0, vl);
            vuint16m1_t d0 = __riscv_vnsrl_wx_u16m1(r0, log2Size + 1, vl);
            __riscv_vse16_v_u16m1(dst + y * dstStride, d0, vl);
            t0 = __riscv_vadd_vv_u32m2(t0, sub_bottomLeft_above0, vl);
        }
    }
    break;
    case 16:
    {
        size_t vl = 8;

        vuint16m1_t above0 = __riscv_vle16_v_u16m1(above, vl);
        vuint16m1_t above1 = __riscv_vle16_v_u16m1(above + 8, vl);

        vuint16m1_t topRight = __riscv_vmv_v_x_u16m1(above[blkSize], vl);
        vuint16m1_t bottomLeft = __riscv_vmv_v_x_u16m1(left[blkSize], vl);

        vuint16m1_t c15 = __riscv_vmv_v_x_u16m1(15, vl);

        const uint16_t c[2][16] =
        {
            {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        };

        // left constant
        vuint16m1_t l0 = __riscv_vle16_v_u16m1(c[0], vl);
        vuint16m1_t l1 = __riscv_vle16_v_u16m1(c[0] + 8, vl);

        // topRight constant
        vuint16m1_t tR0 = __riscv_vle16_v_u16m1(c[1], vl);
        vuint16m1_t tR1 = __riscv_vle16_v_u16m1(c[1] + 8, vl);

        vuint32m2_t offset = __riscv_vmv_v_x_u32m2(blkSize, vl);
        vuint32m2_t offset_bottomLeft = __riscv_vadd_vv_u32m2(
            offset, __riscv_vwcvtu_x_x_v_u32m2(bottomLeft, vl), vl);

        vuint32m2_t t0 = __riscv_vwmaccu_vv_u32m2(offset_bottomLeft, topRight, tR0, vl);
        t0 = __riscv_vwmaccu_vv_u32m2(t0, above0, c15, vl);

        vuint32m2_t t1 = __riscv_vwmaccu_vv_u32m2(offset_bottomLeft, topRight, tR1, vl);
        t1 = __riscv_vwmaccu_vv_u32m2(t1, above1, c15, vl);

        vuint32m2_t sub_bottomLeft_above0 = __riscv_vwsubu_vv_u32m2(bottomLeft, above0, vl);
        vuint32m2_t sub_bottomLeft_above1 = __riscv_vwsubu_vv_u32m2(bottomLeft, above1, vl);

        for (int y = 0; y < 16; y++) {
            vuint16m1_t left_v = __riscv_vmv_v_x_u16m1(left[y], vl);

            vuint32m2_t r0 = __riscv_vwmaccu_vv_u32m2(t0, left_v, l0, vl);
            vuint32m2_t r1 = __riscv_vwmaccu_vv_u32m2(t1, left_v, l1, vl);

            vuint16m1_t d0 = __riscv_vnsrl_wx_u16m1(r0, log2Size + 1, vl);
            vuint16m1_t d1 = __riscv_vnsrl_wx_u16m1(r1, log2Size + 1, vl);

            __riscv_vse16_v_u16m1(dst + y * dstStride, d0, vl);
            __riscv_vse16_v_u16m1(dst + 8 + y * dstStride, d1, vl);

            t0 = __riscv_vadd_vv_u32m2(t0, sub_bottomLeft_above0, vl);
            t1 = __riscv_vadd_vv_u32m2(t1, sub_bottomLeft_above1, vl);
        }
    }
    break;
    case 32:
    {
        // Use rnu rounding mode
        const unsigned int vxrm = 0;

        size_t vl = blkSize;
        pixel topRight = above[blkSize];
        pixel bottomLeft = left[blkSize];

        vuint32m2_t topRightVec = __riscv_vmv_v_x_u32m2(topRight, vl);
        vuint32m2_t oneVec = __riscv_vmv_v_x_u32m2(1, vl);
        vuint32m2_t blkSizeSubOneVec = __riscv_vmv_v_x_u32m2(blkSize - 1, vl);

        for (int y = 0; y < blkSize; y++) {
            for (int x = 0, inc = 0; x < blkSize; x += vl, inc++) {
                vl = __riscv_vsetvl_e16m1(blkSize - x);
                // left[y]
                vuint32m2_t leftYVec = __riscv_vmv_v_x_u32m2(left[y], vl);

                // {0, 1, 2, 3, 4, 5, 6, 7 ...}
                vuint32m2_t xvec = __riscv_vadd_vv_u32m2(__riscv_vid_v_u32m2(vl), __riscv_vmv_v_x_u32m2(inc * vl, vl), vl);

                // (blkSize - 1 - x) * left[y]
                vuint32m2_t first = __riscv_vsub_vv_u32m2(blkSizeSubOneVec, xvec, vl);
                first = __riscv_vmul_vv_u32m2(first, leftYVec, vl);

                // (x + 1) * topRight
                vuint32m2_t second = __riscv_vadd_vv_u32m2(xvec, oneVec, vl);
                second = __riscv_vmul_vv_u32m2(second, topRightVec, vl);

                vuint32m2_t resVec = __riscv_vadd_vv_u32m2(first, second, vl);

                // (blkSize - 1 - y)
                vuint32m2_t vlkSizeYVec = __riscv_vmv_v_x_u32m2(blkSize - 1 - y, vl);

                // (blkSize - 1 - y) * above[x]
                vuint32m2_t aboveVec = __riscv_vzext_vf2_u32m2(__riscv_vle16_v_u16m1(above + x, vl), vl);
                aboveVec = __riscv_vmul_vv_u32m2(aboveVec, vlkSizeYVec, vl);

                resVec = __riscv_vadd_vv_u32m2(resVec, aboveVec, vl);
                // (y + 1) * bottomLeft
                vuint32m2_t bottomLeftYVec = __riscv_vmv_v_x_u32m2((y + 1) * bottomLeft, vl);
                resVec = __riscv_vadd_vv_u32m2(resVec, bottomLeftYVec, vl);

                vuint16m1_t  res =  __riscv_vnclipu_wx_u16m1(resVec, log2Size + 1, vxrm, vl);
                __riscv_vse16_v_u16m1(dst + y * dstStride + x, res, vl);
            }
        }
    }
    break;
    }
}
#endif

#if !HIGH_BIT_DEPTH
void intra_pred_planar32_rvv(pixel *dst, intptr_t dstStride, const pixel *srcPix,
                              int /*dirMode*/, int /*bFilter*/)
{
    const int log2Size = 5;
    const int blkSize = 1 << log2Size;

    const pixel *src0 = srcPix + 1;
    const pixel *src1 = srcPix + 2 * blkSize + 1;

    size_t vl = 16;

    vuint8m1_t above0 = __riscv_vle8_v_u8m1(src0, vl);
    vuint8m1_t above1 = __riscv_vle8_v_u8m1(src0 + 16, vl);

    vuint8m1_t topRight = __riscv_vmv_v_x_u8m1(src0[blkSize], vl);
    vuint8m1_t bottomLeft = __riscv_vmv_v_x_u8m1(src1[blkSize], vl);

    vuint8m1_t c31 = __riscv_vmv_v_x_u8m1(31, vl);

    const uint8_t c[2][32] =
    {
        {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
         15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0},
        {1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
    };

    // left constant
    vuint8m1_t l0 = __riscv_vle8_v_u8m1(c[0], vl);
    vuint8m1_t l1 = __riscv_vle8_v_u8m1(c[0] + 16, vl);

    // topRight constant
    vuint8m1_t tR0 = __riscv_vle8_v_u8m1(c[1], vl);
    vuint8m1_t tR1 = __riscv_vle8_v_u8m1(c[1] + 16 , vl);

    vuint16m2_t offset = __riscv_vmv_v_x_u16m2(blkSize, vl);
    vuint16m2_t offset_bottomLeft = __riscv_vadd_vv_u16m2(
        offset, __riscv_vwcvtu_x_x_v_u16m2(bottomLeft, vl), vl);

    vuint16m2_t t0 = __riscv_vwmaccu_vv_u16m2(offset_bottomLeft, topRight, tR0, vl);
    t0 = __riscv_vwmaccu_vv_u16m2(t0, above0, c31, vl);

    vuint16m2_t t1 = __riscv_vwmaccu_vv_u16m2(offset_bottomLeft, topRight, tR1, vl);
    t1 = __riscv_vwmaccu_vv_u16m2(t1, above1, c31, vl);

    vuint16m2_t sub_bottomLeft_above0 = __riscv_vwsubu_vv_u16m2(bottomLeft, above0, vl);
    vuint16m2_t sub_bottomLeft_above1 = __riscv_vwsubu_vv_u16m2(bottomLeft, above1, vl);

    for (int y = 0; y < 32; y++)
    {
        vuint8m1_t left = __riscv_vmv_v_x_u8m1(src1[y], vl);

        vuint16m2_t r0 = __riscv_vwmaccu_vv_u16m2(t0, left, l0, vl);
        vuint16m2_t r1 = __riscv_vwmaccu_vv_u16m2(t1, left, l1, vl);

        vuint8m1_t d0 = __riscv_vnsrl_wx_u8m1(r0, log2Size + 1, vl);
        vuint8m1_t d1 = __riscv_vnsrl_wx_u8m1(r1, log2Size + 1, vl);

        __riscv_vse8_v_u8m1(dst + 0 * 8 + y * dstStride, d0, vl);
        __riscv_vse8_v_u8m1(dst + 2 * 8 + y * dstStride, d1, vl);

        t0 = __riscv_vadd_vv_u16m2(t0, sub_bottomLeft_above0, vl);
        t1 = __riscv_vadd_vv_u16m2(t1, sub_bottomLeft_above1, vl);
    }
}
#endif

static void dcPredFilter_rvv(const pixel* above, const pixel* left, pixel* dst, intptr_t dststride, int size)
{
    pixel topLeft = (pixel)((above[0] + left[0] + 2 * dst[0] + 2) >> 2);
    pixel * pdst = dst;

    // Use rnu rounding mode
    const unsigned int vxrm = 0;
    size_t vl = size;

    switch (size) {
    case 32:
    {
    #if !HIGH_BIT_DEPTH
        vuint8m2_t vconst_3 = __riscv_vmv_v_x_u8m2(3, vl);
        vuint8m2_t vabo = __riscv_vle8_v_u8m2(above, vl);
        vuint8m2_t vdst = __riscv_vle8_v_u8m2(dst, vl);

        // Compute: dst[x] = (above[x] + 3 * dst[x] + 2) >> 2
        vuint16m4_t vdst_ext = __riscv_vwmulu_vv_u16m4(vdst, vconst_3, vl);
        vdst_ext = __riscv_vwaddu_wv_u16m4(vdst_ext, vabo, vl);

        vuint8m2_t res = __riscv_vnclipu_wx_u8m2(vdst_ext, 2, vxrm, vl);
        __riscv_vse8_v_u8m2(dst, res, vl);
    #else
        vuint16m4_t vconst_3 = __riscv_vmv_v_x_u16m4(3, vl);
        vuint16m4_t vabo = __riscv_vle16_v_u16m4(above, vl);
        vuint16m4_t vdst = __riscv_vle16_v_u16m4(dst, vl);

        // Compute: dst[x] = (above[x] + 3 * dst[x] + 2) >> 2
        vuint32m8_t vdst_ext = __riscv_vwmulu_vv_u32m8(vdst, vconst_3, vl);
        vdst_ext = __riscv_vwaddu_wv_u32m8(vdst_ext, vabo, vl);

        vuint16m4_t res = __riscv_vnclipu_wx_u16m4(vdst_ext, 2, vxrm, vl);
        __riscv_vse16_v_u16m4(dst, res, vl);
    #endif
    }
    break;
    case 16:
    {
    #if !HIGH_BIT_DEPTH
        vuint8m1_t vconst_3 = __riscv_vmv_v_x_u8m1(3, vl);
        vuint8m1_t vabo = __riscv_vle8_v_u8m1(above, vl);
        vuint8m1_t vdst = __riscv_vle8_v_u8m1(dst, vl);

        // Compute: dst[x] = (above[x] + 3 * dst[x] + 2) >> 2
        vuint16m2_t vdst_ext = __riscv_vwmulu_vv_u16m2(vdst, vconst_3, vl);
        vdst_ext = __riscv_vwaddu_wv_u16m2(vdst_ext, vabo, vl);

        vuint8m1_t res = __riscv_vnclipu_wx_u8m1(vdst_ext, 2, vxrm, vl);
        __riscv_vse8_v_u8m1(dst, res, vl);
    #else
        vuint16m2_t vconst_3 = __riscv_vmv_v_x_u16m2(3, vl);
        vuint16m2_t vabo = __riscv_vle16_v_u16m2(above, vl);
        vuint16m2_t vdst = __riscv_vle16_v_u16m2(dst, vl);

        // Compute: dst[x] = (above[x] + 3 * dst[x] + 2) >> 2
        vuint32m4_t vdst_ext = __riscv_vwmulu_vv_u32m4(vdst, vconst_3, vl);
        vdst_ext = __riscv_vwaddu_wv_u32m4(vdst_ext, vabo, vl);

        vuint16m2_t res = __riscv_vnclipu_wx_u16m2(vdst_ext, 2, vxrm, vl);
        __riscv_vse16_v_u16m2(dst, res, vl);
    #endif
    }
    break;
    case 8:
    {
    #if !HIGH_BIT_DEPTH
        vuint8mf2_t vconst_3 = __riscv_vmv_v_x_u8mf2(3, vl);
        vuint8mf2_t vabo = __riscv_vle8_v_u8mf2(above, vl);
        vuint8mf2_t vdst = __riscv_vle8_v_u8mf2(dst, vl);

        // Compute: dst[x] = (above[x] + 3 * dst[x] + 2) >> 2
        vuint16m1_t vdst_ext = __riscv_vwmulu_vv_u16m1(vdst, vconst_3, vl);
        vdst_ext = __riscv_vwaddu_wv_u16m1(vdst_ext, vabo, vl);

        vuint8mf2_t res = __riscv_vnclipu_wx_u8mf2(vdst_ext, 2, vxrm, vl);
        __riscv_vse8_v_u8mf2(dst, res, vl);
    #else
        vuint16m1_t vabo = __riscv_vle16_v_u16m1(above, vl);
        vuint16m1_t vdst = __riscv_vle16_v_u16m1(dst, vl);

        vuint16m1_t vconst_3 = __riscv_vmv_v_x_u16m1(3, vl);
        vuint16m1_t vconst_2 = __riscv_vmv_v_x_u16m1(2, vl);
        vuint16m1_t tmp = __riscv_vadd_vv_u16m1(vabo, vconst_2, vl);

        // Compute: dst[x] = (above[x] + 3 * dst[x] + 2) >> 2
        vuint32m2_t vdst_ext = __riscv_vwmulu_vv_u32m2(vdst, vconst_3, vl);
        vdst_ext = __riscv_vwaddu_wv_u32m2(vdst_ext, tmp, vl);

        vuint16m1_t res = __riscv_vnsrl_wx_u16m1(vdst_ext, 2, vl);
        __riscv_vse16_v_u16m1(dst, res, vl);
    #endif
    }
    break;
    case 4:
    {
    #if !HIGH_BIT_DEPTH
        vuint8mf4_t vconst_3 = __riscv_vmv_v_x_u8mf4(3, vl);
        vuint8mf4_t vabo = __riscv_vle8_v_u8mf4(above, vl);
        vuint8mf4_t vdst = __riscv_vle8_v_u8mf4(dst, vl);

        // Compute: dst[x] = (above[x] + 3 * dst[x] + 2) >> 2
        vuint16mf2_t vdst_ext = __riscv_vwmulu_vv_u16mf2(vdst, vconst_3, vl);
        vdst_ext = __riscv_vwaddu_wv_u16mf2(vdst_ext, vabo, vl);

        vuint8mf4_t res = __riscv_vnclipu_wx_u8mf4(vdst_ext, 2, vxrm, vl);
        __riscv_vse8_v_u8mf4(dst, res, vl);
    #else
        vuint16mf2_t vconst_3 = __riscv_vmv_v_x_u16mf2(3, vl);
        vuint16mf2_t vabo = __riscv_vle16_v_u16mf2(above, vl);
        vuint16mf2_t vdst = __riscv_vle16_v_u16mf2(dst, vl);

        // Compute: dst[x] = (above[x] + 3 * dst[x] + 2) >> 2
        vuint32m1_t vdst_ext = __riscv_vwmulu_vv_u32m1(vdst, vconst_3, vl);
        vdst_ext = __riscv_vwaddu_wv_u32m1(vdst_ext, vabo, vl);

        vuint16mf2_t res = __riscv_vnclipu_wx_u16mf2(vdst_ext, 2, vxrm, vl);
        __riscv_vse16_v_u16mf2(dst, res, vl);
    #endif
    }
    break;
    }
    dst += dststride;
    for (int y = 1; y < size; y++) {
        *dst = (pixel)((left[y] + 3 * *dst + 2) >> 2);
        dst += dststride;
    }
    *pdst = topLeft;
}

template<int width>
void intra_pred_dc_rvv(pixel* dst, intptr_t dstStride, const pixel* srcPix, int /*dirMode*/, int bFilter)
{
    int dcVal = width;
    size_t vl = width;

    switch (width) {
    case 32:
    {
    #if !HIGH_BIT_DEPTH
        vuint16m8_t vsp = __riscv_vmv_v_x_u16m8(0, vl);
        vuint16m1_t v_sum = __riscv_vmv_v_x_u16m1(0, 1);

        vuint8m4_t spa = __riscv_vle8_v_u8m4(srcPix + 1, vl);
        vuint8m4_t spb = __riscv_vle8_v_u8m4(srcPix + 2 * width + 1, vl);
        vuint16m8_t tmp = __riscv_vwaddu_vv_u16m8(spa, spb, vl);
        vsp = __riscv_vadd_vv_u16m8(vsp, tmp, vl);

        v_sum = __riscv_vredsum_vs_u16m8_u16m1(vsp, v_sum, width);
        dcVal += __riscv_vmv_x_s_u16m1_u16(v_sum);
        dcVal = dcVal / (width + width);

        vuint8m4_t vdc = __riscv_vmv_v_x_u8m4(dcVal, vl);
        for (int k = 0; k < width; k++) {
            pixel *row = dst + k * dstStride;
             __riscv_vse8_v_u8m4(row, vdc, vl);
        }
    #else
        vuint32m8_t vsp = __riscv_vmv_v_x_u32m8(0, vl);
        vuint32m1_t v_sum = __riscv_vmv_v_x_u32m1(0, 1);

        vuint16m4_t spa = __riscv_vle16_v_u16m4(srcPix + 1, vl);
        vuint16m4_t spb = __riscv_vle16_v_u16m4(srcPix + 2 * width + 1, vl);
        vuint32m8_t tmp = __riscv_vwaddu_vv_u32m8(spa, spb, vl);
        vsp = __riscv_vadd_vv_u32m8(vsp, tmp, vl);

        v_sum = __riscv_vredsum_vs_u32m8_u32m1(vsp, v_sum, width);
        dcVal += __riscv_vmv_x_s_u32m1_u32(v_sum);
        dcVal = dcVal / (width + width);

        vuint16m4_t vdc = __riscv_vmv_v_x_u16m4(dcVal, vl);
        for (int k = 0; k < width; k++) {
            pixel *row = dst + k * dstStride;
            __riscv_vse16_v_u16m4(row, vdc, vl);
        }
    #endif
    }
    break;
    case 16:
    {
    #if !HIGH_BIT_DEPTH

        vuint16m4_t vsp = __riscv_vmv_v_x_u16m4(0, vl);
        vuint16m1_t v_sum = __riscv_vmv_v_x_u16m1(0, 1);

        vuint8m2_t spa = __riscv_vle8_v_u8m2(srcPix + 1, vl);
        vuint8m2_t spb = __riscv_vle8_v_u8m2(srcPix + 2 * width + 1, vl);
        vuint16m4_t tmp = __riscv_vwaddu_vv_u16m4(spa, spb, vl);
        vsp = __riscv_vadd_vv_u16m4(vsp, tmp, vl);

        v_sum = __riscv_vredsum_vs_u16m4_u16m1(vsp, v_sum, width);
        dcVal += __riscv_vmv_x_s_u16m1_u16(v_sum);
        dcVal = dcVal / (width + width);

        vuint8m2_t vdc = __riscv_vmv_v_x_u8m2(dcVal, vl);
        for (int k = 0; k < width; k++) {
            pixel *row = dst + k * dstStride;
            __riscv_vse8_v_u8m2(row, vdc, vl);
        }
    #else
        vuint32m4_t vsp = __riscv_vmv_v_x_u32m4(0, vl);
        vuint32m1_t v_sum = __riscv_vmv_v_x_u32m1(0, 1);

        vuint16m2_t spa = __riscv_vle16_v_u16m2(srcPix + 1, vl);
        vuint16m2_t spb = __riscv_vle16_v_u16m2(srcPix + 2 * width + 1, vl);
        vuint32m4_t tmp = __riscv_vwaddu_vv_u32m4(spa, spb, vl);
        vsp = __riscv_vadd_vv_u32m4(vsp, tmp, vl);

        v_sum = __riscv_vredsum_vs_u32m4_u32m1(vsp, v_sum, width);
        dcVal += __riscv_vmv_x_s_u32m1_u32(v_sum);
        dcVal = dcVal / (width + width);

        vuint16m2_t vdc = __riscv_vmv_v_x_u16m2(dcVal, vl);
        for (int k = 0; k < width; k++) {
            pixel *row = dst + k * dstStride;
            __riscv_vse16_v_u16m2(row, vdc, vl);
        }
    #endif
    }
    break;
    case 8:
    {
    #if !HIGH_BIT_DEPTH
        vuint16m1_t vsp = __riscv_vmv_v_x_u16m1(0, vl);
        vuint16m1_t v_sum = __riscv_vmv_v_x_u16m1(0, 1);

        vuint8mf2_t spa = __riscv_vle8_v_u8mf2(srcPix + 1, vl);
        vuint8mf2_t spb = __riscv_vle8_v_u8mf2(srcPix + 2 * width + 1, vl);
        vuint16m1_t tmp = __riscv_vwaddu_vv_u16m1(spa, spb, vl);
        vsp = __riscv_vadd_vv_u16m1(vsp, tmp, width);

        v_sum = __riscv_vredsum_vs_u16m1_u16m1(vsp, v_sum, width);
        dcVal += __riscv_vmv_x_s_u16m1_u16(v_sum);
        dcVal = dcVal / (width + width);

        vuint8mf2_t vdc = __riscv_vmv_v_x_u8mf2(dcVal, width);
        for (int k = 0; k < width; k++) {
            pixel *row = dst + k * dstStride;
            __riscv_vse8_v_u8mf2(row, vdc, width);
    }
    #else
        vuint16m1_t spa = __riscv_vle16_v_u16m1(srcPix + 1, vl);
        vuint16m1_t spb = __riscv_vle16_v_u16m1(srcPix + 2 * width + 1, vl);

        vuint32m2_t vsp = __riscv_vmv_v_x_u32m2(0, vl);
        vuint32m1_t v_sum = __riscv_vmv_v_x_u32m1(0, 1);

        vuint32m2_t tmp = __riscv_vwaddu_vv_u32m2(spa, spb, vl);
        vsp = __riscv_vadd_vv_u32m2(vsp, tmp, vl);

        v_sum = __riscv_vredsum_vs_u32m2_u32m1(vsp, v_sum, vl);
        dcVal += __riscv_vmv_x_s_u32m1_u32(v_sum);
        dcVal = dcVal / (width + width);

        vuint16m1_t vdc = __riscv_vmv_v_x_u16m1(dcVal, width);
        for (int k = 0; k < width; k += 2) {
            __riscv_vse16_v_u16m1(dst + k * dstStride, vdc, width);
            __riscv_vse16_v_u16m1(dst + (k + 1) * dstStride, vdc, width);
        }
#endif
    }
    break;
    case 4:
    {
    #if !HIGH_BIT_DEPTH
        vuint16m1_t vsp = __riscv_vmv_v_x_u16m1(0, vl);

        vuint8mf2_t spa = __riscv_vle8_v_u8mf2(srcPix + 1, vl);
        vuint8mf2_t spb = __riscv_vle8_v_u8mf2(srcPix + 2 * width + 1, vl);
        vuint16m1_t tmp = __riscv_vwaddu_vv_u16m1(spa, spb, vl);
        vsp = __riscv_vadd_vv_u16m1(vsp, tmp, width);

        vuint16m1_t v_sum = __riscv_vmv_v_x_u16m1(0, 1);
        v_sum = __riscv_vredsum_vs_u16m1_u16m1(vsp, v_sum, width);

        dcVal += __riscv_vmv_x_s_u16m1_u16(v_sum);
        dcVal = dcVal / (width + width);

        vuint8mf2_t vdc = __riscv_vmv_v_x_u8mf2(dcVal, width);
        for (int k = 0; k < width; k++) {
            pixel *row = dst + k * dstStride;
            __riscv_vse8_v_u8mf2(row, vdc, width);
        }
    #else
        vuint16mf2_t vsp = __riscv_vmv_v_x_u16mf2(0, vl);
        vuint16m1_t v_sum = __riscv_vmv_v_x_u16m1(0, 1);

        vuint16mf2_t spa = __riscv_vle16_v_u16mf2(srcPix + 1, vl);
        vuint16mf2_t spb = __riscv_vle16_v_u16mf2(srcPix + 2 * width + 1, vl);
        vuint16mf2_t tmp = __riscv_vadd_vv_u16mf2(spa, spb, vl);
        vsp = __riscv_vadd_vv_u16mf2(vsp, tmp, vl);

        v_sum = __riscv_vredsum_vs_u16mf2_u16m1(vsp, v_sum, vl);
        dcVal += __riscv_vmv_x_s_u16m1_u16(v_sum);
        dcVal = dcVal / (width + width);

        vuint16mf2_t vdc = __riscv_vmv_v_x_u16mf2(dcVal, width);
        for (int k = 0; k < width; k++) {
            pixel *row = dst + k * dstStride;
            __riscv_vse16_v_u16mf2(row, vdc, width);
        }
    #endif
    }
    break;
    }

    if (bFilter){
        dcPredFilter_rvv(srcPix + 1, srcPix + (2 * width + 1), dst, dstStride, width);
    }
}

// x265 private namespace
void setupIntraPrimitives_rvv(EncoderPrimitives &p)
{
    p.cu[BLOCK_4x4].intra_filter = intraFilter_rvv<4>;
    p.cu[BLOCK_8x8].intra_filter = intraFilter_rvv<8>;
    p.cu[BLOCK_16x16].intra_filter = intraFilter_rvv<16>;
    p.cu[BLOCK_32x32].intra_filter = intraFilter_rvv<32>;

    for (int i = 2; i < NUM_INTRA_MODE; i++) {
        p.cu[BLOCK_4x4].intra_pred[i] = intra_pred_ang_rvv<4>;
        p.cu[BLOCK_8x8].intra_pred[i] = intra_pred_ang_rvv<8>;
        p.cu[BLOCK_16x16].intra_pred[i] = intra_pred_ang_rvv<16>;
        p.cu[BLOCK_32x32].intra_pred[i] = intra_pred_ang_rvv<32>;
    }

    //p.cu[BLOCK_4x4].intra_pred_allangs = all_angs_pred_rvv<2>;
    p.cu[BLOCK_8x8].intra_pred_allangs = all_angs_pred_rvv<3>;
    p.cu[BLOCK_16x16].intra_pred_allangs = all_angs_pred_rvv<4>;
    p.cu[BLOCK_32x32].intra_pred_allangs = all_angs_pred_rvv<5>;

#if !HIGH_BIT_DEPTH
    p.cu[BLOCK_32x32].intra_pred[PLANAR_IDX] = intra_pred_planar32_rvv;
#else
    p.cu[BLOCK_4x4].intra_pred[PLANAR_IDX] = planar_pred_rvv<2>;
    p.cu[BLOCK_8x8].intra_pred[PLANAR_IDX] = planar_pred_rvv<3>;
    p.cu[BLOCK_16x16].intra_pred[PLANAR_IDX] = planar_pred_rvv<4>;
    p.cu[BLOCK_32x32].intra_pred[PLANAR_IDX] = planar_pred_rvv<5>;
#endif

    p.cu[BLOCK_4x4].intra_pred[DC_IDX] = intra_pred_dc_rvv<4>;
    p.cu[BLOCK_8x8].intra_pred[DC_IDX] = intra_pred_dc_rvv<8>;
    p.cu[BLOCK_16x16].intra_pred[DC_IDX] = intra_pred_dc_rvv<16>;
    p.cu[BLOCK_32x32].intra_pred[DC_IDX] = intra_pred_dc_rvv<32>;

}
}
