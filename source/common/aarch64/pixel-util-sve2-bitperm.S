/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: George Steed <george.steed@arm.com>
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

#include "asm-sve.S"
#include "pixel-util-common.S"

.arch armv8-a+sve2+sve2-bitperm

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4

.text

// int scanPosLast(
//     const uint16_t *scan,      // x0
//     const coeff_t *coeff,      // x1
//     uint16_t *coeffSign,       // x2
//     uint16_t *coeffFlag,       // x3
//     uint8_t *coeffNum,         // x4
//     int numSig,                // x5
//     const uint16_t* scanCG4x4, // x6
//     const int trSize)          // x7
function PFX(scanPosLast_sve2_bitperm)
    // Convert unit of trSize stride from elements (int16) to bytes.
    add             x7, x7, x7

    // Load scan table and convert to bytes.
    ldp             q0, q1, [x6]
    uzp1            v0.16b, v0.16b, v1.16b  // v0 - Zigzag scan table.

    movrel          x10, g_SPL_and_mask
    ldr             q28, [x10]              // v28 = mask for pmovmskb.
    add             x10, x7, x7             // 2*x7
    add             x11, x7, x7, lsl #1     // 3*x7
    add             x9, x4, #1              // CG count

1:
    // Position of current CG.
    ldrh            w6, [x0], #32
    add             x6, x1, x6, lsl #1

    // Loading current CG and saturate to bytes.
    ldr             d2, [x6]
    ldr             d3, [x6, x7]
    ldr             d4, [x6, x10]
    ldr             d5, [x6, x11]
    mov             v2.d[1], v3.d[0]
    mov             v4.d[1], v5.d[0]
    sqxtn           v2.8b, v2.8h
    sqxtn2          v2.16b, v4.8h

    // Apply zigzag.
    tbl             v3.16b, {v2.16b}, v0.16b

    // Get zero/sign.
    cmeq            v5.16b, v3.16b, #0   // v5 = zero
    cmlt            v3.16b, v3.16b, #0   // v3 = negative

    //  val: v3.h[0] = pmovmskb(v3).
    // mask: v3.h[1] = pmovmskb(v4).
    and             v3.16b, v3.16b, v28.16b
    bic             v4.16b, v28.16b, v5.16b
    addp            v3.16b, v3.16b, v4.16b
    addp            v3.16b, v3.16b, v3.16b
    addp            v3.16b, v3.16b, v3.16b
    fmov            w15, s3

    // coeffNum = addv(v3 != 0) = 16 - addv(v5).
    addv            b5, v5.16b
    smov            w6, v5.b[0]
    add             w6, w6, #16
    sub             x5, x5, x6
    strb            w6, [x4], #1

    // coeffFlag = reverse_bit(w15) in 16-bit.
    rbit            w12, w15
    strh            w12, [x3], #2

    // Pack bits from z3.h[0] into z30.h[0], based on z3.h[1] mask.
    mov             h31, v3.h[1]
    bext            z30.h, z3.h, z31.h
    str             h30, [x2], #2

    cbnz            x5, 1b

    // Count trailing zeros in (reversed) coeffFlag.
    clz             w13, w15
    lsr             w12, w12, w13
    strh            w12, [x3, #-2]

    // Get last pos.
    sub             x9, x4, x9
    eor             w13, w13, #15
    add             x0, x13, x9, lsl #4
    ret
endfunc

const g_SPL_and_mask, align=8
.byte 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
endconst
