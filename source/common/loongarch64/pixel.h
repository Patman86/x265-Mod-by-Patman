/*****************************************************************************
 * pixel.h: loongarch pixel metrics
 *****************************************************************************
 * Copyright (C) 2013-2024 MulticoreWare, Inc
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

#ifndef X265_LOONGARCH_PIXEL_H
#define X265_LOONGARCH_PIXEL_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define FUNCDEF_PU(ret, name, cpu, ...) \
    ret PFX(name ## _4x4_   ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _8x8_   ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _16x16_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _32x32_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _64x64_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _8x4_   ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _4x8_   ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _16x8_  ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _8x16_  ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _16x32_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _32x16_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _64x32_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _32x64_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _16x12_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _12x16_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _16x4_  ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _4x16_  ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _32x24_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _24x32_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _32x8_  ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _8x32_  ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _64x48_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _48x64_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _64x16_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _16x64_ ## cpu)(__VA_ARGS__)

#define FUNCDEF_CHROMA_PU(ret, name, cpu, ...) \
    FUNCDEF_PU(ret, name, cpu, __VA_ARGS__); \
    ret PFX(name ## _2x2_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _4x2_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _2x4_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _8x2_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _2x8_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _8x6_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _6x8_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _8x12_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _12x8_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _6x16_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _16x6_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _2x16_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _16x2_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _4x12_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _12x4_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _32x12_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _12x32_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _32x4_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _4x32_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _32x48_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _48x32_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _16x24_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _24x16_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _8x64_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _64x8_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _64x24_ ## cpu)(__VA_ARGS__); \
    ret PFX(name ## _24x64_ ## cpu)(__VA_ARGS__);

#define DECL_PIXELS(cpu) \
    FUNCDEF_PU(void, pixel_avg, cpu, pixel* dst, intptr_t dstride, const pixel* src0, intptr_t sstride0, const pixel* src1, intptr_t sstride1, int); \
    FUNCDEF_CHROMA_PU(void, addAvg, cpu, const int16_t*, const int16_t*, pixel*, intptr_t, intptr_t, intptr_t); \
    FUNCDEF_CHROMA_PU(void, addAvg_aligned, cpu, const int16_t*, const int16_t*, pixel*, intptr_t, intptr_t, intptr_t);

DECL_PIXELS(lsx);

int x265_pixel_satd_4x4_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_4x8_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_4x16_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_4x32_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_8x4_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_8x8_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_8x16_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_8x12_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_8x32_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_8x64_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_12x16_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_12x32_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_16x4_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_16x8_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_16x12_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_16x16_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_16x24_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_16x32_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_16x64_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_24x32_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_24x64_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_32x8_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_32x16_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_32x24_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_32x32_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_32x48_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_32x64_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_48x64_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_64x16_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_64x32_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_64x48_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_64x64_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);

int x265_pixel_satd_4x8_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_4x16_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_4x32_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_8x8_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_8x16_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_8x32_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_8x64_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_12x16_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_12x32_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_16x8_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_16x16_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_16x24_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_16x32_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_16x64_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_24x32_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_24x64_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_32x8_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_32x16_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_32x24_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_32x32_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_32x48_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_32x64_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_48x64_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_64x16_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_64x32_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_64x48_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_satd_64x64_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);

FUNCDEF_PU(void, pixel_sad_x4, lsx, const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);

void x265_pixel_sad_x4_8x4_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_8x8_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_8x16_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_8x32_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_12x16_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_16x4_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_16x8_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_16x12_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_16x16_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_16x32_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_16x64_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_24x32_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_32x8_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_32x16_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_32x24_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_32x32_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_32x64_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_64x16_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_64x32_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_64x48_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_64x64_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x4_48x64_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res);

int x265_pixel_sa8d_8x8_lsx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);
int x265_pixel_sa8d_8x16_lsx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);
int x265_pixel_sa8d_16x16_lsx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);
int x265_pixel_sa8d_16x32_lsx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);
int x265_pixel_sa8d_32x32_lsx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);
int x265_pixel_sa8d_32x64_lsx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);
int x265_pixel_sa8d_64x64_lsx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);

int x265_pixel_sa8d_8x8_lasx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);
int x265_pixel_sa8d_8x16_lasx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);
int x265_pixel_sa8d_16x16_lasx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);
int x265_pixel_sa8d_16x32_lasx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);
int x265_pixel_sa8d_32x32_lasx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);
int x265_pixel_sa8d_32x64_lasx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);
int x265_pixel_sa8d_64x64_lasx(const pixel* pix1, intptr_t i_pix1, const pixel* pix2, intptr_t i_pix2);

FUNCDEF_PU(int, pixel_sad, lsx, const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);

int x265_pixel_sad_24x32_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_sad_32x8_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_sad_32x16_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_sad_32x24_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_sad_32x32_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_sad_32x64_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_sad_48x64_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_sad_64x16_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_sad_64x32_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_sad_64x48_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
int x265_pixel_sad_64x64_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);

FUNCDEF_PU(void, pixel_sad_x3, lsx, const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);

void x265_pixel_sad_x3_12x16_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_16x8_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_16x12_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_16x16_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_16x32_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_16x64_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_24x32_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_32x8_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_32x16_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_32x24_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_32x32_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_32x64_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_64x16_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_64x32_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_64x48_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_64x64_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);
void x265_pixel_sad_x3_48x64_lasx(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res);

uint32_t x265_pixel_sse_4x4_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_4x8_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_8x8_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_8x16_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_16x16_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_16x32_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_32x32_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_32x64_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_64x64_lsx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);

uint32_t x265_pixel_sse_16x16_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_16x32_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_32x32_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_32x64_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_64x64_lasx(const pixel* pix1, intptr_t stride_pix1, const pixel* pix2, intptr_t stride_pix2);

uint32_t x265_pixel_sse_ss_4x4_lsx(const int16_t* pix1, intptr_t stride_pix1, const int16_t* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_ss_8x8_lsx(const int16_t* pix1, intptr_t stride_pix1, const int16_t* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_ss_16x16_lsx(const int16_t* pix1, intptr_t stride_pix1, const int16_t* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_ss_32x32_lsx(const int16_t* pix1, intptr_t stride_pix1, const int16_t* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_ss_64x64_lsx(const int16_t* pix1, intptr_t stride_pix1, const int16_t* pix2, intptr_t stride_pix2);

uint32_t x265_pixel_sse_ss_8x8_lasx(const int16_t* pix1, intptr_t stride_pix1, const int16_t* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_ss_16x16_lasx(const int16_t* pix1, intptr_t stride_pix1, const int16_t* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_ss_32x32_lasx(const int16_t* pix1, intptr_t stride_pix1, const int16_t* pix2, intptr_t stride_pix2);
uint32_t x265_pixel_sse_ss_64x64_lasx(const int16_t* pix1, intptr_t stride_pix1, const int16_t* pix2, intptr_t stride_pix2);

void x265_weight_pp_lsx(const pixel* src, pixel* dst, intptr_t stride, int width, int height, int w0, int round, int shift, int offset);
void x265_weight_sp_lsx(const int16_t* src, pixel* dst, intptr_t srcStride, intptr_t dstStride, int width, int height, int w0, int round, int shift, int offset);

FUNCDEF_PU(void, pixel_sub_ps, lsx, int16_t* a, intptr_t dstride, const pixel* b0, const pixel* b1, intptr_t sstride0, intptr_t sstride1);

void x265_pixel_sub_ps_32x32_lasx(int16_t* a, intptr_t dstride, const pixel* b0, const pixel* b1, intptr_t sstride0, intptr_t sstride1);
void x265_pixel_sub_ps_32x64_lasx(int16_t* a, intptr_t dstride, const pixel* b0, const pixel* b1, intptr_t sstride0, intptr_t sstride1);
void x265_pixel_sub_ps_64x64_lasx(int16_t* a, intptr_t dstride, const pixel* b0, const pixel* b1, intptr_t sstride0, intptr_t sstride1);

FUNCDEF_PU(void, pixel_add_ps, lsx, pixel* a, intptr_t dstride, const pixel* b0, const int16_t* b1, intptr_t sstride0, intptr_t sstride1);

void x265_pixel_add_ps_16x16_lasx(pixel* a, intptr_t dstride, const pixel* b0, const int16_t* b1, intptr_t sstride0, intptr_t sstride1);
void x265_pixel_add_ps_16x32_lasx(pixel* a, intptr_t dstride, const pixel* b0, const int16_t* b1, intptr_t sstride0, intptr_t sstride1);
void x265_pixel_add_ps_32x32_lasx(pixel* a, intptr_t dstride, const pixel* b0, const int16_t* b1, intptr_t sstride0, intptr_t sstride1);
void x265_pixel_add_ps_32x64_lasx(pixel* a, intptr_t dstride, const pixel* b0, const int16_t* b1, intptr_t sstride0, intptr_t sstride1);
void x265_pixel_add_ps_64x64_lasx(pixel* a, intptr_t dstride, const pixel* b0, const int16_t* b1, intptr_t sstride0, intptr_t sstride1);

uint64_t x265_pixel_var_4x4_lsx(const pixel* pix, intptr_t i_stride);
uint64_t x265_pixel_var_8x8_lsx(const pixel* pix, intptr_t i_stride);
uint64_t x265_pixel_var_16x16_lsx(const pixel* pix, intptr_t i_stride);
uint64_t x265_pixel_var_32x32_lsx(const pixel* pix, intptr_t i_stride);
uint64_t x265_pixel_var_64x64_lsx(const pixel* pix, intptr_t i_stride);

uint64_t x265_pixel_var_8x8_lasx(const pixel* pix, intptr_t i_stride);
uint64_t x265_pixel_var_16x16_lasx(const pixel* pix, intptr_t i_stride);
uint64_t x265_pixel_var_32x32_lasx(const pixel* pix, intptr_t i_stride);
uint64_t x265_pixel_var_64x64_lasx(const pixel* pix, intptr_t i_stride);

uint32_t x265_pixel_ssd_s_4x4_lsx(const int16_t* a, intptr_t dstride);
uint32_t x265_pixel_ssd_s_8x8_lsx(const int16_t* a, intptr_t dstride);
uint32_t x265_pixel_ssd_s_16x16_lsx(const int16_t* a, intptr_t dstride);
uint32_t x265_pixel_ssd_s_32x32_lsx(const int16_t* a, intptr_t dstride);
uint32_t x265_pixel_ssd_s_64x64_lsx(const int16_t* a, intptr_t dstride);

uint32_t x265_pixel_ssd_s_16x16_lasx(const int16_t* a, intptr_t dstride);
uint32_t x265_pixel_ssd_s_32x32_lasx(const int16_t* a, intptr_t dstride);
uint32_t x265_pixel_ssd_s_64x64_lasx(const int16_t* a, intptr_t dstride);

void x265_pixel_getResidual_4x4_lsx(const pixel* fenc, const pixel* pred, int16_t* residual, intptr_t stride);
void x265_pixel_getResidual_8x8_lsx(const pixel* fenc, const pixel* pred, int16_t* residual, intptr_t stride);
void x265_pixel_getResidual_16x16_lsx(const pixel* fenc, const pixel* pred, int16_t* residual, intptr_t stride);
void x265_pixel_getResidual_32x32_lsx(const pixel* fenc, const pixel* pred, int16_t* residual, intptr_t stride);
void x265_pixel_getResidual_64x64_lsx(const pixel* fenc, const pixel* pred, int16_t* residual, intptr_t stride);

void x265_pixel_getResidual_32x32_lasx(const pixel* fenc, const pixel* pred, int16_t* residual, intptr_t stride);
void x265_pixel_getResidual_64x64_lasx(const pixel* fenc, const pixel* pred, int16_t* residual, intptr_t stride);

void x265_pixel_normFact_8x8_lsx(const pixel* src, uint32_t blockSize, int shift, uint64_t *z_k);
void x265_pixel_normFact_16x16_lsx(const pixel* src, uint32_t blockSize, int shift, uint64_t *z_k);
void x265_pixel_normFact_32x32_lsx(const pixel* src, uint32_t blockSize, int shift, uint64_t *z_k);
void x265_pixel_normFact_64x64_lsx(const pixel* src, uint32_t blockSize, int shift, uint64_t *z_k);

void x265_pixel_normFact_8x8_lasx(const pixel* src, uint32_t blockSize, int shift, uint64_t *z_k);
void x265_pixel_normFact_16x16_lasx(const pixel* src, uint32_t blockSize, int shift, uint64_t *z_k);
void x265_pixel_normFact_32x32_lasx(const pixel* src, uint32_t blockSize, int shift, uint64_t *z_k);
void x265_pixel_normFact_64x64_lasx(const pixel* src, uint32_t blockSize, int shift, uint64_t *z_k);

void x265_pixel_planecopy_cp_lsx(const uint8_t* src, intptr_t srcStride, pixel* dst, intptr_t dstStride, int width, int height, int shift);
void x265_pixel_planecopy_sp_lsx(const uint16_t* src, intptr_t srcStride, pixel* dst, intptr_t dstStride, int width, int height, int shift, uint16_t mask);
void x265_pixel_planecopy_sp_shl_lsx(const uint16_t* src, intptr_t srcStride, pixel* dst, intptr_t dstStride, int width, int height, int shift, uint16_t mask);
void x265_pixel_planecopy_pp_shr_lsx(const pixel* src, intptr_t srcStride, pixel* dst, intptr_t dstStride, int width, int height, int shift);

void x265_scale1D_128to64_lsx(pixel *dst, const pixel *src);
void x265_scale2D_64to32_lsx(pixel* dst, const pixel* src, intptr_t stride);

void x265_scale1D_128to64_lasx(pixel *dst, const pixel *src);
void x265_scale2D_64to32_lasx(pixel* dst, const pixel* src, intptr_t stride);

void x265_transpose_4x4_lsx(pixel* dst, const pixel* src, intptr_t stride);
void x265_transpose_8x8_lsx(pixel* dst, const pixel* src, intptr_t stride);
void x265_transpose_16x16_lsx(pixel* dst, const pixel* src, intptr_t stride);
void x265_transpose_32x32_lsx(pixel* dst, const pixel* src, intptr_t stride);
void x265_transpose_64x64_lsx(pixel* dst, const pixel* src, intptr_t stride);

void x265_ssimDist4_lsx(const pixel* fenc, uint32_t fStride, const pixel* recon, intptr_t rstride, uint64_t *ssBlock, int shift, uint64_t *ac_k);
void x265_ssimDist8_lsx(const pixel* fenc, uint32_t fStride, const pixel* recon, intptr_t rstride, uint64_t *ssBlock, int shift, uint64_t *ac_k);
void x265_ssimDist16_lsx(const pixel* fenc, uint32_t fStride, const pixel* recon, intptr_t rstride, uint64_t *ssBlock, int shift, uint64_t *ac_k);
void x265_ssimDist32_lsx(const pixel* fenc, uint32_t fStride, const pixel* recon, intptr_t rstride, uint64_t *ssBlock, int shift, uint64_t *ac_k);
void x265_ssimDist64_lsx(const pixel* fenc, uint32_t fStride, const pixel* recon, intptr_t rstride, uint64_t *ssBlock, int shift, uint64_t *ac_k);

void x265_ssimDist8_lasx(const pixel* fenc, uint32_t fStride, const pixel* recon, intptr_t rstride, uint64_t *ssBlock, int shift, uint64_t *ac_k);
void x265_ssimDist16_lasx(const pixel* fenc, uint32_t fStride, const pixel* recon, intptr_t rstride, uint64_t *ssBlock, int shift, uint64_t *ac_k);
void x265_ssimDist32_lasx(const pixel* fenc, uint32_t fStride, const pixel* recon, intptr_t rstride, uint64_t *ssBlock, int shift, uint64_t *ac_k);
void x265_ssimDist64_lasx(const pixel* fenc, uint32_t fStride, const pixel* recon, intptr_t rstride, uint64_t *ssBlock, int shift, uint64_t *ac_k);

float x265_ssim_end4_lsx(int sum0[5][4], int sum1[5][4], int width);
void x265_ssim_4x4x2_lsx(const pixel* pix1, intptr_t stride1, const pixel* pix2, intptr_t stride2, int sums[2][4]);

void x265_pixel_avg_32x8_lasx(pixel* dst, intptr_t dstride, const pixel* src0, intptr_t sstride0, const pixel* src1, intptr_t sstride1, int);
void x265_pixel_avg_32x16_lasx(pixel* dst, intptr_t dstride, const pixel* src0, intptr_t sstride0, const pixel* src1, intptr_t sstride1, int);
void x265_pixel_avg_32x24_lasx(pixel* dst, intptr_t dstride, const pixel* src0, intptr_t sstride0, const pixel* src1, intptr_t sstride1, int);
void x265_pixel_avg_32x32_lasx(pixel* dst, intptr_t dstride, const pixel* src0, intptr_t sstride0, const pixel* src1, intptr_t sstride1, int);
void x265_pixel_avg_32x64_lasx(pixel* dst, intptr_t dstride, const pixel* src0, intptr_t sstride0, const pixel* src1, intptr_t sstride1, int);
void x265_pixel_avg_48x64_lasx(pixel* dst, intptr_t dstride, const pixel* src0, intptr_t sstride0, const pixel* src1, intptr_t sstride1, int);
void x265_pixel_avg_64x16_lasx(pixel* dst, intptr_t dstride, const pixel* src0, intptr_t sstride0, const pixel* src1, intptr_t sstride1, int);
void x265_pixel_avg_64x32_lasx(pixel* dst, intptr_t dstride, const pixel* src0, intptr_t sstride0, const pixel* src1, intptr_t sstride1, int);
void x265_pixel_avg_64x48_lasx(pixel* dst, intptr_t dstride, const pixel* src0, intptr_t sstride0, const pixel* src1, intptr_t sstride1, int);
void x265_pixel_avg_64x64_lasx(pixel* dst, intptr_t dstride, const pixel* src0, intptr_t sstride0, const pixel* src1, intptr_t sstride1, int);

void x265_addAvg_24x32_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
void x265_addAvg_24x64_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
void x265_addAvg_32x8_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
void x265_addAvg_32x16_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
void x265_addAvg_32x24_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
void x265_addAvg_32x32_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
void x265_addAvg_32x48_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
void x265_addAvg_32x64_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
void x265_addAvg_48x64_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
void x265_addAvg_64x16_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
void x265_addAvg_64x32_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
void x265_addAvg_64x48_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);
void x265_addAvg_64x64_lasx(const int16_t* src0, const int16_t* src1, pixel* dst, intptr_t src0Stride, intptr_t src1Stride, intptr_t dstStride);

void x265_frame_init_lowres_core_lsx(const pixel *src0, pixel *dst0, pixel *dsth, pixel *dstv, pixel *dstc, intptr_t src_stride, intptr_t dst_stride, int width, int height);
void x265_frame_init_lowres_core_lasx(const pixel *src0, pixel *dst0, pixel *dsth, pixel *dstv, pixel *dstc, intptr_t src_stride, intptr_t dst_stride, int width, int height);
void x265_frame_subsample_luma_lsx(const pixel *src0, pixel *dst0, intptr_t src_stride, intptr_t dst_stride, int width, int height);
void x265_frame_subsample_luma_lasx(const pixel *src0, pixel *dst0, intptr_t src_stride, intptr_t dst_stride, int width, int height);
void x265_cutree_fix8_unpack_lsx(double *dst, uint16_t *src, int count);
void x265_cutree_fix8_unpack_lasx(double *dst, uint16_t *src, int count);
void x265_cutree_fix8_pack_lsx(uint16_t *dst, double *src, int count);
void x265_cutree_fix8_pack_lasx(uint16_t *dst, double *src, int count);
#undef FUNCDEF_PU
#undef DECL_PIXELS
#undef FUNCDEF_CHROMA_PU

#ifdef __cplusplus
}
#endif /*__cplusplus*/

#endif // ifndef X265_LOONGARCH_PIXEL_H
