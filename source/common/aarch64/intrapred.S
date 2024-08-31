/*****************************************************************************
 * Copyright (C) 2021 MulticoreWare, Inc
 *
 * Authors: Min Chen <min.chen@multicorewareinc.com>
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

// Functions in this file:
// ***** luma_vpp *****

#include "asm.S"

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4

.text

.align 4
tbl_const_1to8_7to0:
    .byte 1, 2, 3, 4, 5, 6, 7, 8
    .byte 7, 6, 5, 4, 3, 2, 1, 0
    .byte 9, 10, 11, 12, 13, 14, 15, 16
    .byte 15, 14, 13, 12, 11, 10, 9, 8

// ***** planar_pred *****
// void planar_pred(pixel* dst, intptr_t dstStride, const pixel* srcPix, int /*dirMode*/, int /*bFilter*/)
function PFX(intra_pred_planar8_neon)
// Register map
// x0  = dst
// x1  = dstStride
// x2  = *srcPix
// x3  = left[x]
// x4  = tmp
// v0  = above[7:0]
// v1  = left[7:0]
// v2  = topRight = rep(above[blkSize])
// v3  = bottomLeft = rep(left[blkSize])
// v4  = const[8 7 6 5 4 3 2 1]
// v5  = const[7 6 5 4 3 2 1 0]

//{
//    const int blkSize = 1 << log2Size;
//    const pixel* above = srcPix + 1;
//    const pixel* left  = srcPix + (2 * blkSize + 1);
//    pixel topRight = above[blkSize];
//    pixel bottomLeft = left[blkSize];
//    for (int y = 0; y < blkSize; y++)
//        for (int x = 0; x < blkSize; x++)
//            dst[y * dstStride + x] = (pixel) (((blkSize - 1 - x) * left[y] + (blkSize - 1 -y) * above[x] + (x + 1) * topRight + (y + 1) * bottomLeft + blkSize) >> (log2Size + 1));
//}

    ldurb           w3, [x2, #(1+8)]                // topRight
    ldurb           w4, [x2, #(2*8+1+8)]            // bottomLeft
    dup             v2.8b, w3                       // v2 = topRight_b
    dup             v3.8h, w4                       // v3 = bottomLeft_h
    ldr             x3, [x2, #(2*8+1)]              // x3 = left[x]_b
    ldr             d0, [x2, #1]                    // v0 = above[x]_b

    adr             x4, tbl_const_1to8_7to0
    ldr             d4, [x4]                        // v4 = const_b[8 7 6 5 4 3 2 1]
    ldr             d5, [x4, #8]                    // v5 = const_b[7 6 5 4 3 2 1 0]

    ushll           v6.8h, v0.8b, #3                // v6 = 8 * above[x]
    usubw           v0.8h, v3.8h, v0.8b             // v0 = bottomLeft - above[x]

    umlal           v6.8h, v4.8b, v2.8b             // v6 = 8 * above[x] + (x + 1) * topRight

    mov             w4, #8

1:
    dup             v1.8b, w3
    lsr             x3, x3, #8
    add             v6.8h, v6.8h, v0.8h             // v6 = (blkSize - 1 -y=0) * above[x] + (x + 1) * topRight + (y=0 + 1) * bottomLeft
    mov             v3.16b, v6.16b
    umlal           v3.8h, v5.8b, v1.8b             // v3 = (blkSize - 1 - x) * left[y=0] + (blkSize - 1 -y=0) * above[x] + (x + 1) * topRight + (y=0 + 1) * bottomLeft
    rshrn           v3.8b, v3.8h, #4
    sub             w4, w4, #1
    st1             {v3.8b}, [x0], x1
    cbnz            w4, 1b

    ret
endfunc

// void planar_pred(pixel* dst, intptr_t dstStride, const pixel* srcPix, int /*dirMode*/, int /*bFilter*/)
function PFX(intra_pred_planar16_neon)
// Register map
// x0  = dst
// x1  = dstStride
// x2  = *srcPix
// x3  = left[x]
// x4  = tmp
// v0  = above[7:0]
// v1  = left[7:0]
// v2  = topRight = rep(above[blkSize])
// v3  = bottomLeft = rep(left[blkSize])
// v4  = const[16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1]
// v5  = const[15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0]

//{
//    const int blkSize = 1 << log2Size;
//    const pixel* above = srcPix + 1;
//    const pixel* left  = srcPix + (2 * blkSize + 1);
//    pixel topRight = above[blkSize];
//    pixel bottomLeft = left[blkSize];
//    for (int y = 0; y < blkSize; y++)
//        for (int x = 0; x < blkSize; x++)
//            dst[y * dstStride + x] = (pixel) (((blkSize - 1 - x) * left[y] + (blkSize - 1 -y) * above[x] + (x + 1) * topRight + (y + 1) * bottomLeft + blkSize) >> (log2Size + 1));
//}

    ldurb           w3, [x2, #(1+16)]               // topRight
    ldurb           w4, [x2, #(2*16+1+16)]          // bottomLeft
    ldr             q0, [x2, #(2*16+1)]             // v0 = left[x]_b
    ldr             q1, [x2, #1]                    // v1 = above[x]_b
    dup             v2.16b, w3                      // v2 = topRight_b
    dup             v3.8h, w4                       // v3 = bottomLeft_h

    adr             x4, tbl_const_1to8_7to0
    ld2             {v4.2d, v5.2d}, [x4]            // v4 = const_b[16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1]
    ext             v5.16b, v5.16b, v5.16b, #8      // v5 = const_b[15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0]

    ushll           v16.8h, v1.8b, #4               // v16,v17 = 16 * above[x]
    ushll2          v17.8h, v1.16b, #4
    usubw           v6.8h, v3.8h, v1.8b             // v6,v7 = bottomLeft - above[x]
    usubw2          v7.8h, v3.8h, v1.16b

    umlal           v16.8h, v4.8b, v2.8b            // v16,v17 = 16 * above[x] + (x + 1) * topRight
    umlal2          v17.8h, v4.16b, v2.16b

    mov             w4, #16

1:
    dup             v1.16b, v0.b[0]                 // v1 = left[x]_b
    ext             v0.16b, v0.16b, v0.16b, #1

    add             v16.8h, v16.8h, v6.8h           // v16,v17 = (blkSize - 1 -y=0) * above[x] + (x + 1) * topRight + (y=0 + 1) * bottomLeft
    add             v17.8h, v17.8h, v7.8h

    mov             v18.16b, v16.16b
    mov             v19.16b, v17.16b

    umlal           v18.8h, v5.8b, v1.8b             // v3 = (blkSize - 1 - x) * left[y=0] + (blkSize - 1 -y=0) * above[x] + (x + 1) * topRight + (y=0 + 1) * bottomLeft
    umlal2          v19.8h, v5.16b, v1.16b
    rshrn           v18.8b, v18.8h, #5
    rshrn2          v18.16b, v19.8h, #5
    st1             {v18.16b}, [x0], x1
    sub             w4, w4, #1
    cbnz            w4, 1b

    ret
endfunc
