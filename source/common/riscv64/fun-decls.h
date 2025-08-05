/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Changsheng Wu <wu.changsheng@sanechips.com.cn>
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

#ifndef _RISCV_FUNC_DECLS_
#define _RISCV_FUNC_DECLS_

#define FUNCDEF_TU_S(ret, name, cpu, ...) \
    ret PFX(name ## _4_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## _8_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## _16_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## _32_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## _64_ ## cpu(__VA_ARGS__))

#define FUNCDEF_TU_S2(ret, name, cpu, ...) \
    ret PFX(name ## 4_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## 8_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## 16_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## 32_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## 64_ ## cpu(__VA_ARGS__))

#define FUNCDEF_TU_S3(ret, name, cpu, ...) \
    ret PFX(name ## 2_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## 3_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## 4_ ## cpu(__VA_ARGS__)); \
    ret PFX(name ## 5_ ## cpu(__VA_ARGS__)); \

FUNCDEF_TU_S(uint32_t, copy_cnt, v, int16_t* dst, const int16_t* src, intptr_t srcStride);
FUNCDEF_TU_S(int, count_nonzero, v, const int16_t* quantCoeff);
FUNCDEF_TU_S2(void, idct, v, const int16_t* src, int16_t* dst, intptr_t dstStride);
FUNCDEF_TU_S2(void, dct, v, const int16_t* src, int16_t* dst, intptr_t srcStride);
FUNCDEF_TU_S3(void, nonPsyRdoQuant, v, int16_t *m_resiDctCoeff, int64_t *costUncoded, int64_t *totalUncodedCost, int64_t *totalRdCost, uint32_t blkPos);
FUNCDEF_TU_S3(void, PsyRdoQuant, v, int16_t *m_resiDctCoeff, int16_t *m_fencDctCoeff, int64_t *costUncoded, int64_t *totalUncodedCost, int64_t *totalRdCost, int64_t *psyScale, uint32_t blkPos);

void PFX(idst4_v)(const int16_t *src, int16_t *dst, intptr_t dstStride);
void PFX(dst4_v)(const int16_t *src, int16_t *dst, intptr_t srcStride);
void PFX(denoiseDct_v)(int16_t* dctCoef, uint32_t* resSum, const uint16_t* offset, int numCoeff);

#endif
