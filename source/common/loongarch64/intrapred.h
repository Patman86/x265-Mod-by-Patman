/*****************************************************************************
 * intrapred.h: Intra Prediction metrics
 *****************************************************************************
 * Copyright (C) 2024 MulticoreWare, Inc
 *
 * Authors: jinbo <jinbo@loongson.cn>
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

#ifndef X265_LOONGARCH_INTRAPRED_H
#define X265_LOONGARCH_INTRAPRED_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void x265_intra_filter_4x4_lsx(const pixel *samples, pixel *filtered);
void x265_intra_filter_8x8_lsx(const pixel *samples, pixel *filtered);
void x265_intra_filter_16x16_lsx(const pixel *samples, pixel *filtered);
void x265_intra_filter_32x32_lsx(const pixel *samples, pixel *filtered);
void x265_intra_filter_4x4_lasx(const pixel *samples, pixel *filtered);
void x265_intra_filter_8x8_lasx(const pixel *samples, pixel *filtered);
void x265_intra_filter_16x16_lasx(const pixel *samples, pixel *filtered);
void x265_intra_filter_32x32_lasx(const pixel *samples, pixel *filtered);

void x265_intra_pred_planar_4x4_lsx(pixel *dst, intptr_t dstStride, const pixel *srcPix, int /*dirMode*/, int /*bFilter*/);
void x265_intra_pred_planar_8x8_lsx(pixel *dst, intptr_t dstStride, const pixel *srcPix, int /*dirMode*/, int /*bFilter*/);
void x265_intra_pred_planar_16x16_lsx(pixel *dst, intptr_t dstStride, const pixel *srcPix, int /*dirMode*/, int /*bFilter*/);
void x265_intra_pred_planar_32x32_lsx(pixel *dst, intptr_t dstStride, const pixel *srcPix, int /*dirMode*/, int /*bFilter*/);
void x265_intra_pred_planar_16x16_lasx(pixel *dst, intptr_t dstStride, const pixel *srcPix, int /*dirMode*/, int /*bFilter*/);
void x265_intra_pred_planar_32x32_lasx(pixel *dst, intptr_t dstStride, const pixel *srcPix, int /*dirMode*/, int /*bFilter*/);

void x265_intra_pred_ang8_2_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_3_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_4_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_5_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_6_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_7_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_8_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_9_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_10_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_11_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_12_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_13_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_14_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_15_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_16_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_17_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_18_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_26_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_34_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_2_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_3_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_4_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_5_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_6_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_7_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_8_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_9_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_10_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_11_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_12_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_13_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_14_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_15_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_16_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_17_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_18_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_19_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_20_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_21_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_22_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_23_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_24_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_25_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_26_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_27_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_28_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_29_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_30_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_31_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_32_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_33_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_2_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_3_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_4_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_5_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_6_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_6_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_6_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_7_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_8_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_9_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_10_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_11_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_12_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_13_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_14_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_15_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_16_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_17_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_18_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_19_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_20_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_21_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_22_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_23_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_24_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_25_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_26_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_27_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_28_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_29_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_30_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_31_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_32_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_33_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang32_34_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_2_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_3_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_4_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_5_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_6_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_7_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_8_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_9_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_10_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_11_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_12_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_13_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_14_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_15_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_16_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_17_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_18_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_19_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_20_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_21_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_22_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_23_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_24_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_25_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_26_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_27_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_28_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_29_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_30_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_31_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_32_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_33_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang4_34_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);

void x265_intra_pred_ang8_3_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_4_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_5_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_6_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_7_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_8_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_9_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_11_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_12_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_13_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_14_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_15_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_16_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_17_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_19_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_20_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_21_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_22_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_23_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_24_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_25_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_27_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_28_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_29_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_30_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_31_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_32_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang8_33_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_3_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_4_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_5_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_6_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_7_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_8_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_9_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_11_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_12_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_13_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_14_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_15_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_16_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_ang16_17_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);

void x265_intra_pred_dc4_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_dc8_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_dc16_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_dc32_lsx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
void x265_intra_pred_dc32_lasx(pixel* dst, intptr_t dstStride, const pixel* srcPix, int dirMode, int bFilter);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif // ifndef X265_LOONGARCH_INTRAPRED_H
