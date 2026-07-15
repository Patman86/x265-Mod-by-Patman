/*****************************************************************************
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

#ifndef X265_LOONGARCH_BLOCKCOPY8_H
#define X265_LOONGARCH_BLOCKCOPY8_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#undef FUNCDEF_PU
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

#define FUNCDEF_TU(ret, name, cpu, ...) \
    ret PFX(name ## _4x4_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## _8x8_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## _16x16_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## _32x32_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## _64x64_ ## cpu(__VA_ARGS__))

FUNCDEF_CHROMA_PU(void, blockcopy_pp, lsx, pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);

FUNCDEF_TU(void, blockfill_s, lsx, int16_t* dst, intptr_t dstride, int16_t val);

void x265_blockcopy_ss_2x2_lsx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_2x4_lsx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_4x4_lsx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_4x8_lsx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_8x8_lsx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_8x16_lsx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_16x16_lsx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_16x32_lsx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_32x32_lsx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_32x64_lsx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_64x64_lsx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_2x2_lsx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_2x4_lsx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_4x4_lsx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_4x8_lsx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_8x8_lsx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_8x16_lsx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_16x16_lsx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_16x32_lsx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_32x32_lsx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_32x64_lsx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_64x64_lsx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ps_2x4_lsx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_4x4_lsx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_4x8_lsx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_8x8_lsx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_8x16_lsx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_16x16_lsx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_16x32_lsx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_32x32_lsx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_32x64_lsx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_64x64_lsx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
uint32_t x265_copy_count_4_lsx(int16_t* coeff, const int16_t* residual, intptr_t resiStride);
uint32_t x265_copy_count_8_lsx(int16_t* coeff, const int16_t* residual, intptr_t resiStride);
uint32_t x265_copy_count_16_lsx(int16_t* coeff, const int16_t* residual, intptr_t resiStride);
uint32_t x265_copy_count_32_lsx(int16_t* coeff, const int16_t* residual, intptr_t resiStride);

void x265_blockfill_s_16x16_lasx(int16_t* dst, intptr_t dstride, int16_t val);
void x265_blockfill_s_32x32_lasx(int16_t* dst, intptr_t dstride, int16_t val);
void x265_blockfill_s_64x64_lasx(int16_t* dst, intptr_t dstride, int16_t val);
void x265_blockcopy_pp_24x32_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_pp_24x64_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_pp_32x8_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_pp_32x16_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_pp_32x24_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_pp_32x32_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_pp_32x48_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_pp_32x64_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_pp_48x64_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_pp_64x16_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_pp_64x32_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_pp_64x48_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_pp_64x64_lasx(pixel* dst, intptr_t dstStride, const pixel* src, intptr_t srcStride);
void x265_blockcopy_ss_16x16_lasx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_16x32_lasx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_32x32_lasx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_32x64_lasx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ss_64x64_lasx(int16_t* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_16x16_lasx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_16x32_lasx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_32x32_lasx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_32x64_lasx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_sp_64x64_lasx(pixel* a, intptr_t stridea, const int16_t* b, intptr_t strideb);
void x265_blockcopy_ps_16x16_lasx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_16x32_lasx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_32x32_lasx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_32x64_lasx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
void x265_blockcopy_ps_64x64_lasx(int16_t* a, intptr_t stridea, const pixel* b, intptr_t strideb);
uint32_t x265_copy_count_16_lasx(int16_t* coeff, const int16_t* residual, intptr_t resiStride);
uint32_t x265_copy_count_32_lasx(int16_t* coeff, const int16_t* residual, intptr_t resiStride);

FUNCDEF_TU(void, cpy2Dto1D_shl, lsx, int16_t *dst, const int16_t *src, intptr_t srcStride, int shift);

void x265_cpy2Dto1D_shl_8x8_lasx(int16_t *dst, const int16_t *src, intptr_t srcStride, int shift);
void x265_cpy2Dto1D_shl_16x16_lasx(int16_t *dst, const int16_t *src, intptr_t srcStride, int shift);
void x265_cpy2Dto1D_shl_32x32_lasx(int16_t *dst, const int16_t *src, intptr_t srcStride, int shift);
void x265_cpy2Dto1D_shl_64x64_lasx(int16_t *dst, const int16_t *src, intptr_t srcStride, int shift);

FUNCDEF_TU(void, cpy2Dto1D_shr, lsx, int16_t *dst, const int16_t *src, intptr_t srcStride, int shift);

void x265_cpy2Dto1D_shr_8x8_lasx(int16_t *dst, const int16_t *src, intptr_t srcStride, int shift);
void x265_cpy2Dto1D_shr_16x16_lasx(int16_t *dst, const int16_t *src, intptr_t srcStride, int shift);
void x265_cpy2Dto1D_shr_32x32_lasx(int16_t *dst, const int16_t *src, intptr_t srcStride, int shift);
void x265_cpy2Dto1D_shr_64x64_lasx(int16_t *dst, const int16_t *src, intptr_t srcStride, int shift);

FUNCDEF_TU(void, cpy1Dto2D_shl, lsx, int16_t *dst, const int16_t *src, intptr_t dstStride, int shift);

void x265_cpy1Dto2D_shl_8x8_lasx(int16_t *dst, const int16_t *src, intptr_t dstStride, int shift);
void x265_cpy1Dto2D_shl_16x16_lasx(int16_t *dst, const int16_t *src, intptr_t dstStride, int shift);
void x265_cpy1Dto2D_shl_32x32_lasx(int16_t *dst, const int16_t *src, intptr_t dstStride, int shift);
void x265_cpy1Dto2D_shl_64x64_lasx(int16_t *dst, const int16_t *src, intptr_t dstStride, int shift);

FUNCDEF_TU(void, cpy1Dto2D_shr, lsx, int16_t *dst, const int16_t *src, intptr_t dstStride, int shift);

void x265_cpy1Dto2D_shr_8x8_lasx(int16_t *dst, const int16_t *src, intptr_t dstStride, int shift);
void x265_cpy1Dto2D_shr_16x16_lasx(int16_t *dst, const int16_t *src, intptr_t dstStride, int shift);
void x265_cpy1Dto2D_shr_32x32_lasx(int16_t *dst, const int16_t *src, intptr_t dstStride, int shift);
void x265_cpy1Dto2D_shr_64x64_lasx(int16_t *dst, const int16_t *src, intptr_t dstStride, int shift);

#undef FUNCDEF_PU
#undef FUNCDEF_CHROMA_PU

#ifdef __cplusplus
}
#endif /*__cplusplus*/

#endif //X265_LOONGARCH_BLOCKCOPY8_H
