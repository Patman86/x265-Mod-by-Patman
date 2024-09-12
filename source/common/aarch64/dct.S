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
.set idct16_shift_1, 7
.set idct16_shift_2, 12-(BIT_DEPTH-8)

.set dct16_shift_1, 3+(BIT_DEPTH-8)
.set dct16_shift_2, 10

.align 4
// NOTE: Hardcoded due to asm syntax issue, don't reorder!
tbl_const_idct_0:
    .hword 64, 83, 36, 89, 75, 50, 18,  0   // v0
    .hword 90, 87, 80, 70, 57, 43, 25,  9   // v1
//    .hword 0=64, 1=83, 2=36, 3=89, 4=75, 5=50, 6=18, 7=00
//    .hword 0=90, 1=87, 2=80, 3=70, 4=57, 5=43, 6=25, 7= 9

    .hword 64, 83, 64, 36   // v0
    .hword 64, 36,-64,-83
    .hword 64,-36,-64, 83   // v1
    .hword 64,-83, 64,-36

    .hword 89, 75, 50, 18   // v2
    .hword 75,-18,-89,-50
    .hword 50,-89, 18, 75   // v3
    .hword 18,-50, 75,-89

    .hword 90,+87,+80,+70, +57,+43,+25,+ 9   // v4
    .hword 87,+57, +9,-43, -80,-90,-70,-25   // v5
    .hword 80, +9,-70,-87, -25,+57,+90,+43   // v6
    .hword 70,-43,-87, +9, +90,+25,-80,-57   // v7
    .hword 57,-80,-25,+90, - 9,-87,+43,+70   // v8
    .hword 43,-90,+57,+25, -87,+70,+ 9,-80   // v9
    .hword 25,-70,+90,-80, +43,+ 9,-57,+87   // v16
    .hword  9,-25,+43,-57, +70,-80,+87,-90   // v17

    .byte 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3  // v18

tbl_const_dct_0:
    // EE
    .hword 64,+64,+64,+64                   // v16
    .hword 83,+36,-36,-83                   // v17
    .hword 64,-64,-64,+64                   // v18
    .hword 36,-83,+83,-36                   // v19

    // EO
    .hword 89,+75,+50,+18                   // v20
    .hword 75,-18,-89,-50                   // v21
    .hword 50,-89,+18,+75                   // v22
    .hword 18,-50,+75,-89                   // v23

    // O
    .hword 90,+87,+80,+70,+57,+43,+25, +9   // v24
    .hword 87,+57, +9,-43,-80,-90,-70,-25   // v25
    .hword 80, +9,-70,-87,-25,+57,+90,+43   // v26
    .hword 70,-43,-87, +9,+90,+25,-80,-57   // v27
    .hword 57,-80,-25,+90, -9,-87,+43,+70   // v28
    .hword 43,-90,+57,+25,-87,+70, +9,-80   // v29
    .hword 25,-70,+90,-80,+43, +9,-57,+87   // v30
    .hword  9,-25,+43,-57,+70,-80,+87,-90   // v31

    .byte 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1  // v0
//    .byte 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9  // v1

    .word 64, 83, 36, 89, 75, 50, 18,  0    // v0, v1
    .word 90, 87, 80, 70, 57, 43, 25,  9    // v2, v3


// ***** idct 16x16 *****
// void idct16(const int16_t* src, int16_t* dst, intptr_t dstStride)
function PFX(idct16_neon)
// Register map
// x0  = src
// x1  = dst
// x2  = dstStride
// x8  = tbl_const_idct_0

    stp             d8, d9, [sp,#-16]!
    sub             sp, sp, #(16*16*2)

    adr             x8, tbl_const_idct_0
    ldp             q0, q1, [x8]

    mov             x5, sp
    mov             w4, #16

    // Pass1
5:
    ldr             d16, [x0, #(0*16*2)]
    ldr             d17, [x0, #(2*16*2)]
    ldr             d18, [x0, #(4*16*2)]
    ldr             d19, [x0, #(6*16*2)]
    ldr             d20, [x0, #(8*16*2)]
    ldr             d21, [x0, #(10*16*2)]
    ldr             d22, [x0, #(12*16*2)]
    ldr             d23, [x0, #(14*16*2)]

// EEE0 = 64*src[0*16+i] + 64*src[ 8*16+i];
// EEE1 = 64*src[0*16+i] - 64*src[ 8*16+i];
// EEO0 = 83*src[4*16+i] + 36*src[12*16+i];
// EEO1 = 36*src[4*16+i] - 83*src[12*16+i];
    smull           v24.4s, v16.4h, v0.h[0]         // EEE0 = 64*[0]
    smull           v26.4s, v18.4h, v0.h[1]         // EEO0 = 83*[4]
    mov             v25.16b, v24.16b                // EEE1 = 64*[0]
    smull           v27.4s, v18.4h, v0.h[2]         // EEO1 = 36*[4]

// EO0 = 89*src[ 2*16+i] + 75*src[ 6*16+i] + 50*src[10*16+i] + 18*src[14*16+i];
// EO1 = 75*src[ 2*16+i] - 18*src[ 6*16+i] - 89*src[10*16+i] - 50*src[14*16+i];
// EO2 = 50*src[ 2*16+i] - 89*src[ 6*16+i] + 18*src[10*16+i] + 75*src[14*16+i];
// EO3 = 18*src[ 2*16+i] - 50*src[ 6*16+i] + 75*src[10*16+i] - 89*src[14*16+i];
    smull           v28.4s, v17.4h, v0.h[3]         // EO0 = 89*[2]
    smull           v29.4s, v17.4h, v0.h[4]         // EO1 = 75*[2]
    smull           v30.4s, v17.4h, v0.h[5]         // EO2 = 50*[2]
    smull           v31.4s, v17.4h, v0.h[6]         // EO3 = 18*[2]

    smlal           v28.4s, v19.4h, v0.h[4]         // EO0 = 89*[2]+75*[6]
    smlsl           v29.4s, v19.4h, v0.h[6]         // EO1 = 75*[2]-18*[6]
    smlsl           v30.4s, v19.4h, v0.h[3]         // EO2 = 50*[2]-89*[6]
    smlsl           v31.4s, v19.4h, v0.h[5]         // EO3 = 18*[2]-50*[6]

    ldr             d16, [x0, #(1*16*2)]
    ldr             d17, [x0, #(3*16*2)]
    ldr             d18, [x0, #(5*16*2)]
    ldr             d19, [x0, #(7*16*2)]

    orr             v2.8b, v20.8b, v21.8b
    orr             v2.8b, v2.8b, v22.8b
    orr             v2.8b, v2.8b, v23.8b
    orr             v3.8b, v18.8b, v19.8b
    mov             x6, v2.d[0]
    mov             x7, v3.d[0]

// O0 = 90*src[ 1*16+i] + 87*src[ 3*16+i] + 80*src[ 5*16+i] + 70*src[ 7*16+i] + 57*src[ 9*16+i] + 43*src[11*16+i] + 25*src[13*16+i] +  9*src[15*16+i];
// O1 = 87*src[ 1*16+i] + 57*src[ 3*16+i] +  9*src[ 5*16+i] - 43*src[ 7*16+i] - 80*src[ 9*16+i] - 90*src[11*16+i] - 70*src[13*16+i] - 25*src[15*16+i];
// O2 = 80*src[ 1*16+i] +  9*src[ 3*16+i] - 70*src[ 5*16+i] - 87*src[ 7*16+i] - 25*src[ 9*16+i] + 57*src[11*16+i] + 90*src[13*16+i] + 43*src[15*16+i];
// O3 = 70*src[ 1*16+i] - 43*src[ 3*16+i] - 87*src[ 5*16+i] +  9*src[ 7*16+i] + 90*src[ 9*16+i] + 25*src[11*16+i] - 80*src[13*16+i] - 57*src[15*16+i];
// O4 = 57*src[ 1*16+i] - 80*src[ 3*16+i] - 25*src[ 5*16+i] + 90*src[ 7*16+i] -  9*src[ 9*16+i] - 87*src[11*16+i] + 43*src[13*16+i] + 70*src[15*16+i];
// O5 = 43*src[ 1*16+i] - 90*src[ 3*16+i] + 57*src[ 5*16+i] + 25*src[ 7*16+i] - 87*src[ 9*16+i] + 70*src[11*16+i] +  9*src[13*16+i] - 80*src[15*16+i];
// O6 = 25*src[ 1*16+i] - 70*src[ 3*16+i] + 90*src[ 5*16+i] - 80*src[ 7*16+i] + 43*src[ 9*16+i] +  9*src[11*16+i] - 57*src[13*16+i] + 87*src[15*16+i];
// O7 =  9*src[ 1*16+i] - 25*src[ 3*16+i] + 43*src[ 5*16+i] - 57*src[ 7*16+i] + 70*src[ 9*16+i] - 80*src[11*16+i] + 87*src[13*16+i] - 90*src[15*16+i];
    smull           v2.4s, v16.4h, v1.h[0]          // v2 = O0 = 90*[1]
    smull           v3.4s, v16.4h, v1.h[1]          // v3 = O1 = 87*[1]
    smull           v4.4s, v16.4h, v1.h[2]          // v4 = O2 = 80*[1]
    smull           v5.4s, v16.4h, v1.h[3]          // v5 = O3 = 70*[1]
    smull           v6.4s, v16.4h, v1.h[4]          // v6 = O4 = 57*[1]
    smull           v7.4s, v16.4h, v1.h[5]          // v7 = O5 = 43*[1]
    smull           v8.4s, v16.4h, v1.h[6]          // v8 = O6 = 25*[1]
    smull           v9.4s, v16.4h, v1.h[7]          // v9 = O7 =  9*[1]

    smlal           v2.4s, v17.4h, v1.h[1]          // v2 = O0 = 90*[1]+87*[3]
    smlal           v3.4s, v17.4h, v1.h[4]          // v3 = O1 = 87*[1]+57*[3]
    smlal           v4.4s, v17.4h, v1.h[7]          // v4 = O2 = 80*[1]+ 9*[3]
    smlsl           v5.4s, v17.4h, v1.h[5]          // v5 = O3 = 70*[1]-43*[3]
    smlsl           v6.4s, v17.4h, v1.h[2]          // v6 = O4 = 57*[1]-80*[3]
    smlsl           v7.4s, v17.4h, v1.h[0]          // v7 = O5 = 43*[1]-90*[3]
    smlsl           v8.4s, v17.4h, v1.h[3]          // v8 = O6 = 25*[1]-70*[3]
    smlsl           v9.4s, v17.4h, v1.h[6]          // v9 = O7 =  9*[1]-25*[3]

    //cmp             x7, #0
    //beq             1f
    cbz             x7, 1f

    smlal           v2.4s, v18.4h, v1.h[2]          // v2 = O0 = 90*[1]+87*[3]+80*[5]
    smlal           v3.4s, v18.4h, v1.h[7]          // v3 = O1 = 87*[1]+57*[3]+ 9*[5]
    smlsl           v4.4s, v18.4h, v1.h[3]          // v4 = O2 = 80*[1]+ 9*[3]-70*[5]
    smlsl           v5.4s, v18.4h, v1.h[1]          // v5 = O3 = 70*[1]-43*[3]-87*[5]
    smlsl           v6.4s, v18.4h, v1.h[6]          // v6 = O4 = 57*[1]-80*[3]-25*[5]
    smlal           v7.4s, v18.4h, v1.h[4]          // v7 = O5 = 43*[1]-90*[3]+57*[5]
    smlal           v8.4s, v18.4h, v1.h[0]          // v8 = O6 = 25*[1]-70*[3]+90*[5]
    smlal           v9.4s, v18.4h, v1.h[5]          // v9 = O7 =  9*[1]-25*[3]+43*[5]

    smlal           v2.4s, v19.4h, v1.h[3]          // v2 = O0 = 90*[1]+87*[3]+80*[5]+70*[7]
    smlsl           v3.4s, v19.4h, v1.h[5]          // v3 = O1 = 87*[1]+57*[3]+ 9*[5]-43*[7]
    smlsl           v4.4s, v19.4h, v1.h[1]          // v4 = O2 = 80*[1]+ 9*[3]-70*[5]-87*[7]
    smlal           v5.4s, v19.4h, v1.h[7]          // v5 = O3 = 70*[1]-43*[3]-87*[5]+ 9*[7]
    smlal           v6.4s, v19.4h, v1.h[0]          // v6 = O4 = 57*[1]-80*[3]-25*[5]+90*[7]
    smlal           v7.4s, v19.4h, v1.h[6]          // v7 = O5 = 43*[1]-90*[3]+57*[5]+25*[7]
    smlsl           v8.4s, v19.4h, v1.h[2]          // v8 = O6 = 25*[1]-70*[3]+90*[5]-80*[7]
    smlsl           v9.4s, v19.4h, v1.h[4]          // v9 = O7 =  9*[1]-25*[3]+43*[5]-57*[7]

1:
    ldr             d16, [x0, #(9*16*2)]
    ldr             d17, [x0, #(11*16*2)]
    ldr             d18, [x0, #(13*16*2)]
    ldr             d19, [x0, #(15*16*2)]

    //cmp             x6, #0
    //beq             1f
    cbz             x6, 1f

    smlal           v24.4s, v20.4h, v0.h[0]         // EEE0 = 64*[0]+64*[8]
    smlsl           v25.4s, v20.4h, v0.h[0]         // EEE1 = 64*[0]-64*[8]
    smlal           v26.4s, v22.4h, v0.h[2]         // EEO0 = 83*[0]+36*[12]
    smlsl           v27.4s, v22.4h, v0.h[1]         // EEO1 = 36*[0]-83*[12]

    smlal           v28.4s, v21.4h, v0.h[5]         // EO0 = 89*[2]+75*[6]+50*[10]
    smlsl           v29.4s, v21.4h, v0.h[3]         // EO1 = 75*[2]-18*[6]-89*[10]
    smlal           v30.4s, v21.4h, v0.h[6]         // EO2 = 50*[2]-89*[6]+18*[10]
    smlal           v31.4s, v21.4h, v0.h[4]         // EO3 = 18*[2]-50*[6]+75*[10]

    smlal           v28.4s, v23.4h, v0.h[6]         // EO0 = 89*[2]+75*[6]+50*[10]+18*[14]
    smlsl           v29.4s, v23.4h, v0.h[5]         // EO1 = 75*[2]-18*[6]-89*[10]-50*[14]
    smlal           v30.4s, v23.4h, v0.h[4]         // EO2 = 50*[2]-89*[6]+18*[10]+75*[14]
    smlsl           v31.4s, v23.4h, v0.h[3]         // EO3 = 18*[2]-50*[6]+75*[10]-89*[14]

1:
    orr             v20.8b, v16.8b, v17.8b
    orr             v21.8b, v18.8b, v19.8b
    mov             x6, v20.d[0]
    mov             x7, v21.d[0]

    add             v20.4s, v24.4s, v26.4s          // EE0 = EEE0+EEO0
    add             v21.4s, v25.4s, v27.4s          // EE1 = EEE1+EEO1
    sub             v22.4s, v25.4s, v27.4s          // EE2 = EEE1-EEO1
    sub             v23.4s, v24.4s, v26.4s          // EE3 = EEE0-EEO0

    add             v24.4s, v20.4s, v28.4s          // v24 = E0 = EE0+EO0
    sub             v25.4s, v20.4s, v28.4s          // v25 = E7 = EE0-EO0
    add             v26.4s, v21.4s, v29.4s          // v26 = E1 = EE1+EO1
    sub             v27.4s, v21.4s, v29.4s          // v27 = E6 = EE1-EO1
    add             v28.4s, v22.4s, v30.4s          // v28 = E2 = EE2+EO2
    sub             v29.4s, v22.4s, v30.4s          // v29 = E5 = EE2-EO2
    add             v30.4s, v23.4s, v31.4s          // v30 = E3 = EE3+EO3
    sub             v31.4s, v23.4s, v31.4s          // v31 = E4 = EE3-EO3

    //cmp             x6, #0
    //beq             1f
    cbz             x6, 1f

    smlal           v2.4s, v16.4h, v1.h[4]          // v2 = O0 = 90*[1]+87*[3]+80*[5]+70*[7]+57*[9]
    smlsl           v3.4s, v16.4h, v1.h[2]          // v3 = O1 = 87*[1]+57*[3]+ 9*[5]-43*[7]-80*[9]
    smlsl           v4.4s, v16.4h, v1.h[6]          // v4 = O2 = 80*[1]+ 9*[3]-70*[5]-87*[7]-25*[9]
    smlal           v5.4s, v16.4h, v1.h[0]          // v5 = O3 = 70*[1]-43*[3]-87*[5]+ 9*[7]+90*[9]
    smlsl           v6.4s, v16.4h, v1.h[7]          // v6 = O4 = 57*[1]-80*[3]-25*[5]+90*[7]- 9*[9]
    smlsl           v7.4s, v16.4h, v1.h[1]          // v7 = O5 = 43*[1]-90*[3]+57*[5]+25*[7]-87*[9]
    smlal           v8.4s, v16.4h, v1.h[5]          // v8 = O6 = 25*[1]-70*[3]+90*[5]-80*[7]+43*[9]
    smlal           v9.4s, v16.4h, v1.h[3]          // v9 = O7 =  9*[1]-25*[3]+43*[5]-57*[7]+70*[9]

    smlal           v2.4s, v17.4h, v1.h[5]          // v2 = O0 = 90*[1]+87*[3]+80*[5]+70*[7]+57*[9]+43*[11]
    smlsl           v3.4s, v17.4h, v1.h[0]          // v3 = O1 = 87*[1]+57*[3]+ 9*[5]-43*[7]-80*[9]-90*[11]
    smlal           v4.4s, v17.4h, v1.h[4]          // v4 = O2 = 80*[1]+ 9*[3]-70*[5]-87*[7]-25*[9]+57*[11]
    smlal           v5.4s, v17.4h, v1.h[6]          // v5 = O3 = 70*[1]-43*[3]-87*[5]+ 9*[7]+90*[9]+25*[11]
    smlsl           v6.4s, v17.4h, v1.h[1]          // v6 = O4 = 57*[1]-80*[3]-25*[5]+90*[7]- 9*[9]-87*[11]
    smlal           v7.4s, v17.4h, v1.h[3]          // v7 = O5 = 43*[1]-90*[3]+57*[5]+25*[7]-87*[9]+70*[11]
    smlal           v8.4s, v17.4h, v1.h[7]          // v8 = O6 = 25*[1]-70*[3]+90*[5]-80*[7]+43*[9]+ 9*[11]
    smlsl           v9.4s, v17.4h, v1.h[2]          // v9 = O7 =  9*[1]-25*[3]+43*[5]-57*[7]+70*[9]-80*[11]

1:
    //cmp             x7, #0
    //beq             1f
    cbz             x7, 1f

    smlal           v2.4s, v18.4h, v1.h[6]          // v2 = O0 = 90*[1]+87*[3]+80*[5]+70*[7]+57*[9]+43*[11]+25*[13]
    smlsl           v3.4s, v18.4h, v1.h[3]          // v3 = O1 = 87*[1]+57*[3]+ 9*[5]-43*[7]-80*[9]-90*[11]-70*[13]
    smlal           v4.4s, v18.4h, v1.h[0]          // v4 = O2 = 80*[1]+ 9*[3]-70*[5]-87*[7]-25*[9]+57*[11]+90*[13]
    smlsl           v5.4s, v18.4h, v1.h[2]          // v5 = O3 = 70*[1]-43*[3]-87*[5]+ 9*[7]+90*[9]+25*[11]-80*[13]
    smlal           v6.4s, v18.4h, v1.h[5]          // v6 = O4 = 57*[1]-80*[3]-25*[5]+90*[7]- 9*[9]-87*[11]+43*[13]
    smlal           v7.4s, v18.4h, v1.h[7]          // v7 = O5 = 43*[1]-90*[3]+57*[5]+25*[7]-87*[9]+70*[11]+ 9*[13]
    smlsl           v8.4s, v18.4h, v1.h[4]          // v8 = O6 = 25*[1]-70*[3]+90*[5]-80*[7]+43*[9]+ 9*[11]-57*[13]
    smlal           v9.4s, v18.4h, v1.h[1]          // v9 = O7 =  9*[1]-25*[3]+43*[5]-57*[7]+70*[9]-80*[11]+87*[13]

    smlal           v2.4s, v19.4h, v1.h[7]          // v2 = O0 = 90*[1]+87*[3]+80*[5]+70*[7]+57*[9]+43*[11]+25*[13]+ 9*[15]
    smlsl           v3.4s, v19.4h, v1.h[6]          // v3 = O1 = 87*[1]+57*[3]+ 9*[5]-43*[7]-80*[9]-90*[11]-70*[13]-25*[15]
    smlal           v4.4s, v19.4h, v1.h[5]          // v4 = O2 = 80*[1]+ 9*[3]-70*[5]-87*[7]-25*[9]+57*[11]+90*[13]+43*[15]
    smlsl           v5.4s, v19.4h, v1.h[4]          // v5 = O3 = 70*[1]-43*[3]-87*[5]+ 9*[7]+90*[9]+25*[11]-80*[13]-57*[15]
    smlal           v6.4s, v19.4h, v1.h[3]          // v6 = O4 = 57*[1]-80*[3]-25*[5]+90*[7]- 9*[9]-87*[11]+43*[13]+70*[15]
    smlsl           v7.4s, v19.4h, v1.h[2]          // v7 = O5 = 43*[1]-90*[3]+57*[5]+25*[7]-87*[9]+70*[11]+ 9*[13]-80*[15]
    smlal           v8.4s, v19.4h, v1.h[1]          // v8 = O6 = 25*[1]-70*[3]+90*[5]-80*[7]+43*[9]+ 9*[11]-57*[13]+87*[15]
    smlsl           v9.4s, v19.4h, v1.h[0]          // v9 = O7 =  9*[1]-25*[3]+43*[5]-57*[7]+70*[9]-80*[11]+87*[13]-90*[15]

1:
    add             v16.4s, v24.4s, v2.4s           // [ 0] = E0+O0
    sub             v17.4s, v24.4s, v2.4s           // [15] = E0-O0
    add             v18.4s, v26.4s, v3.4s           // [ 1] = E1+O1
    sub             v19.4s, v26.4s, v3.4s           // [14] = E1-O1
    add             v20.4s, v28.4s, v4.4s           // [ 2] = E2+O2
    sub             v21.4s, v28.4s, v4.4s           // [13] = E2-O2
    add             v22.4s, v30.4s, v5.4s           // [ 3] = E3+O3
    sub             v23.4s, v30.4s, v5.4s           // [12] = E3-O3
    sqrshrn         v16.4h, v16.4s, #idct16_shift_1
    sqrshrn         v17.4h, v17.4s, #idct16_shift_1
    sqrshrn         v18.4h, v18.4s, #idct16_shift_1
    sqrshrn         v19.4h, v19.4s, #idct16_shift_1
    sqrshrn         v20.4h, v20.4s, #idct16_shift_1
    sqrshrn         v21.4h, v21.4s, #idct16_shift_1
    sqrshrn         v22.4h, v22.4s, #idct16_shift_1
    sqrshrn         v23.4h, v23.4s, #idct16_shift_1
    str             d16, [x5, #( 0*16*2)]
    str             d17, [x5, #(15*16*2)]
    str             d18, [x5, #( 1*16*2)]
    str             d19, [x5, #(14*16*2)]
    str             d20, [x5, #( 2*16*2)]
    str             d21, [x5, #(13*16*2)]
    str             d22, [x5, #( 3*16*2)]
    str             d23, [x5, #(12*16*2)]

    add             v16.4s, v31.4s, v6.4s           // [ 4] = E4+O4
    sub             v17.4s, v31.4s, v6.4s           // [11] = E4-O4
    add             v18.4s, v29.4s, v7.4s           // [ 5] = E5+O5
    sub             v19.4s, v29.4s, v7.4s           // [10] = E5-O5
    add             v20.4s, v27.4s, v8.4s           // [ 6] = E6+O6
    sub             v21.4s, v27.4s, v8.4s           // [ 9] = E6-O6
    add             v22.4s, v25.4s, v9.4s           // [ 7] = E7+O7
    sub             v23.4s, v25.4s, v9.4s           // [ 8] = E7-O7
    sqrshrn         v16.4h, v16.4s, #idct16_shift_1
    sqrshrn         v17.4h, v17.4s, #idct16_shift_1
    sqrshrn         v18.4h, v18.4s, #idct16_shift_1
    sqrshrn         v19.4h, v19.4s, #idct16_shift_1
    sqrshrn         v20.4h, v20.4s, #idct16_shift_1
    sqrshrn         v21.4h, v21.4s, #idct16_shift_1
    sqrshrn         v22.4h, v22.4s, #idct16_shift_1
    sqrshrn         v23.4h, v23.4s, #idct16_shift_1
    str             d16, [x5, #( 4*16*2)]
    str             d17, [x5, #(11*16*2)]
    str             d18, [x5, #( 5*16*2)]
    str             d19, [x5, #(10*16*2)]
    str             d20, [x5, #( 6*16*2)]
    str             d21, [x5, #( 9*16*2)]
    str             d22, [x5, #( 7*16*2)]
    str             d23, [x5, #( 8*16*2)]


    add             x0, x0, #(4*2)
    add             x5, x5, #(4*2)
    sub             w4, w4, #4
    cbnz            w4, 5b

    // Pass2
    mov             x5, sp
    mov             w4, #16

    ldp             q0, q1, [x8, #(32*1)]
    ldp             q2, q3, [x8, #(32*2)]
    ldp             q4, q5, [x8, #(32*3)]
    ldp             q6, q7, [x8, #(32*4)]
    ldp             q8, q9, [x8, #(32*5)]
    ldp             q16, q17, [x8, #(32*6)]
    ldr             q18, [x8, #(32*7)]

6:
    ld2             {v30.8h, v31.8h}, [x5]          // v30 = [14 12 10 8 6 4 2 0], v31 = [15 13 11 9 7 5 3 1]
    mov             x6, v31.d[1]

    uzp1            v20.8h, v30.8h, v30.8h           // v20 = [12 8 4 0]
    uzp2            v21.8h, v30.8h, v30.8h           // v21 = [14 10 6 2]

// EE0 = 64*dst[0+dstStride*i] + 83*dst[4+dstStride*i] + 64*dst[ 8+dstStride*i] + 36*dst[12+dstStride*i];
// EE1 = 64*dst[0+dstStride*i] + 36*dst[4+dstStride*i] - 64*dst[ 8+dstStride*i] - 83*dst[12+dstStride*i];
// EE2 = 64*dst[0+dstStride*i] - 36*dst[4+dstStride*i] - 64*dst[ 8+dstStride*i] + 83*dst[12+dstStride*i];
// EE3 = 64*dst[0+dstStride*i] - 83*dst[4+dstStride*i] + 64*dst[ 8+dstStride*i] - 36*dst[12+dstStride*i];

    smull           v22.4s, v20.4h, v0.4h            // EE0
    smull2          v23.4s, v20.8h, v0.8h            // EE1
    smull           v24.4s, v20.4h, v1.4h            // EE2
    smull2          v25.4s, v20.8h, v1.8h            // EE3

// EO0 = 89*dst[ 2+dstStride*i] + 75*dst[ 6+dstStride*i] + 50*dst[10+dstStride*i] + 18*dst[14+dstStride*i];
// EO1 = 75*dst[ 2+dstStride*i] - 18*dst[ 6+dstStride*i] - 89*dst[10+dstStride*i] - 50*dst[14+dstStride*i];
// EO2 = 50*dst[ 2+dstStride*i] - 89*dst[ 6+dstStride*i] + 18*dst[10+dstStride*i] + 75*dst[14+dstStride*i];
// EO3 = 18*dst[ 2+dstStride*i] - 50*dst[ 6+dstStride*i] + 75*dst[10+dstStride*i] - 89*dst[14+dstStride*i];
    smull           v26.4s, v21.4h, v2.4h            // EO0
    smull2          v27.4s, v21.8h, v2.8h            // EO1
    smull           v28.4s, v21.4h, v3.4h            // EO2
    smull2          v29.4s, v21.8h, v3.8h            // EO3

// E0 = EE0 + EO0;
// E1 = EE1 + EO1;
// E2 = EE2 + EO2;
// E3 = EE3 + EO3;
// E4 = EE3 - EO3;
// E5 = EE2 - EO2;
// E6 = EE1 - EO1;
// E7 = EE0 - EO0;

    addp    v20.4s, v22.4s, v23.4s                  // [EE1 EE0]
    addp    v21.4s, v24.4s, v25.4s                  // [EE3 EE2]
    addp    v22.4s, v26.4s, v27.4s                  // [EO1 EO0]
    addp    v23.4s, v28.4s, v29.4s                  // [EO3 EO2]
    addp    v24.4s, v20.4s, v21.4s                  // v24 = [EE3 EE2 EE1 EE0]
    addp    v25.4s, v22.4s, v23.4s                  // v25 = [EO3 EO2 EO1 EO0]

    add     v19.4s, v24.4s, v25.4s                  // v19 = [E3 E2 E1 E0]
    sub     v20.4s, v24.4s, v25.4s                  // v20 = [E4 E5 E6 E7]
    //tbl     v21.16b, {v20.16b}, v18.16b             // v21 = [E0 E1 E2 E3]
    //tbl     v22.16b, {v21.16b}, v18.16b             // v22 = [E7 E6 E5 E4]

// O0 =   90*dst[ 1+dstStride*i] + 87*dst[ 3+dstStride*i] + 80*dst[ 5+dstStride*i] + 70*dst[ 7+dstStride*i] + 57*dst[ 9+dstStride*i] + 43*dst[11+dstStride*i] + 25*dst[13+dstStride*i] +  9*dst[15+dstStride*i];
// O1 =   87*dst[ 1+dstStride*i] + 57*dst[ 3+dstStride*i] +  9*dst[ 5+dstStride*i] - 43*dst[ 7+dstStride*i] - 80*dst[ 9+dstStride*i] - 90*dst[11+dstStride*i] - 70*dst[13+dstStride*i] - 25*dst[15+dstStride*i];
// O2 =   80*dst[ 1+dstStride*i] +  9*dst[ 3+dstStride*i] - 70*dst[ 5+dstStride*i] - 87*dst[ 7+dstStride*i] - 25*dst[ 9+dstStride*i] + 57*dst[11+dstStride*i] + 90*dst[13+dstStride*i] + 43*dst[15+dstStride*i];
// O3 =   70*dst[ 1+dstStride*i] - 43*dst[ 3+dstStride*i] - 87*dst[ 5+dstStride*i] +  9*dst[ 7+dstStride*i] + 90*dst[ 9+dstStride*i] + 25*dst[11+dstStride*i] - 80*dst[13+dstStride*i] - 57*dst[15+dstStride*i];
// O4 =   57*dst[ 1+dstStride*i] - 80*dst[ 3+dstStride*i] - 25*dst[ 5+dstStride*i] + 90*dst[ 7+dstStride*i] -  9*dst[ 9+dstStride*i] - 87*dst[11+dstStride*i] + 43*dst[13+dstStride*i] + 70*dst[15+dstStride*i];
// O5 =   43*dst[ 1+dstStride*i] - 90*dst[ 3+dstStride*i] + 57*dst[ 5+dstStride*i] + 25*dst[ 7+dstStride*i] - 87*dst[ 9+dstStride*i] + 70*dst[11+dstStride*i] +  9*dst[13+dstStride*i] - 80*dst[15+dstStride*i];
// O6 =   25*dst[ 1+dstStride*i] - 70*dst[ 3+dstStride*i] + 90*dst[ 5+dstStride*i] - 80*dst[ 7+dstStride*i] + 43*dst[ 9+dstStride*i] +  9*dst[11+dstStride*i] - 57*dst[13+dstStride*i] + 87*dst[15+dstStride*i];
// O7 =    9*dst[ 1+dstStride*i] - 25*dst[ 3+dstStride*i] + 43*dst[ 5+dstStride*i] - 57*dst[ 7+dstStride*i] + 70*dst[ 9+dstStride*i] - 80*dst[11+dstStride*i] + 87*dst[13+dstStride*i] - 90*dst[15+dstStride*i];
    // Free v21-v30
    smull   v23.4s, v31.4h, v4.4h                   // v23 = [O0]
    smull   v24.4s, v31.4h, v5.4h                   // v24 = [O1]
    smull   v25.4s, v31.4h, v6.4h                   // v25 = [O2]
    smull   v26.4s, v31.4h, v7.4h                   // v26 = [O3]
    smull   v27.4s, v31.4h, v8.4h                   // v27 = [O4]
    smull   v28.4s, v31.4h, v9.4h                   // v28 = [O5]
    smull   v29.4s, v31.4h, v16.4h                  // v29 = [O6]
    smull   v30.4s, v31.4h, v17.4h                  // v30 = [O7]

    //cmp     x6, #0
    //beq     1f
    cbz     x6, 1f

    smlal2  v23.4s, v31.8h, v4.8h
    smlal2  v24.4s, v31.8h, v5.8h
    smlal2  v25.4s, v31.8h, v6.8h
    smlal2  v26.4s, v31.8h, v7.8h
    smlal2  v27.4s, v31.8h, v8.8h
    smlal2  v28.4s, v31.8h, v9.8h
    smlal2  v29.4s, v31.8h, v16.8h
    smlal2  v30.4s, v31.8h, v17.8h

1:
//        dst[i*dstStride+ 0] = x265_clip3( -32768, 32767, (E0 + O0 + rnd) >> nShift);
//        dst[i*dstStride+ 1] = x265_clip3( -32768, 32767, (E1 + O1 + rnd) >> nShift);
//        dst[i*dstStride+ 2] = x265_clip3( -32768, 32767, (E2 + O2 + rnd) >> nShift);
//        dst[i*dstStride+ 3] = x265_clip3( -32768, 32767, (E3 + O3 + rnd) >> nShift);
//        dst[i*dstStride+ 4] = x265_clip3( -32768, 32767, (E4 + O4 + rnd) >> nShift);
//        dst[i*dstStride+ 5] = x265_clip3( -32768, 32767, (E5 + O5 + rnd) >> nShift);
//        dst[i*dstStride+ 6] = x265_clip3( -32768, 32767, (E6 + O6 + rnd) >> nShift);
//        dst[i*dstStride+ 7] = x265_clip3( -32768, 32767, (E7 + O7 + rnd) >> nShift);
//        dst[i*dstStride+ 8] = x265_clip3( -32768, 32767, (E7 - O7 + rnd) >> nShift);
//        dst[i*dstStride+ 9] = x265_clip3( -32768, 32767, (E6 - O6 + rnd) >> nShift);
//        dst[i*dstStride+10] = x265_clip3( -32768, 32767, (E5 - O5 + rnd) >> nShift);
//        dst[i*dstStride+11] = x265_clip3( -32768, 32767, (E4 - O4 + rnd) >> nShift);
//        dst[i*dstStride+12] = x265_clip3( -32768, 32767, (E3 - O3 + rnd) >> nShift);
//        dst[i*dstStride+13] = x265_clip3( -32768, 32767, (E2 - O2 + rnd) >> nShift);
//        dst[i*dstStride+14] = x265_clip3( -32768, 32767, (E1 - O1 + rnd) >> nShift);
//        dst[i*dstStride+15] = x265_clip3( -32768, 32767, (E0 - O0 + rnd) >> nShift);
    addp    v23.4s, v23.4s, v24.4s                  // [O1 O0]
    addp    v24.4s, v25.4s, v26.4s                  // [O3 O2]
    addp    v25.4s, v28.4s, v27.4s                  // [O4 O5]
    addp    v26.4s, v30.4s, v29.4s                  // [O6 O7]
    addp    v23.4s, v23.4s, v24.4s                  // v23 = [O3 O2 O1 O0]
    addp    v24.4s, v26.4s, v25.4s                  // v24 = [O4 O5 O6 O7]

    add     v26.4s, v20.4s, v24.4s                  // v26 = [4 5 6 7]
    sub     v27.4s, v19.4s, v23.4s                  // v27 = [12 13 14 15]
    add     v25.4s, v19.4s, v23.4s                  // v25 = [3 2 1 0]
    sub     v28.4s, v20.4s, v24.4s                  // v28 = [11 10 9 8]

    tbl     v26.16b, {v26.16b}, v18.16b             // v26 = [7 6 5 4]
    tbl     v27.16b, {v27.16b}, v18.16b             // v27 = [15 14 13 12]

    sqrshrn         v20.4h, v25.4s, #idct16_shift_2
    sqrshrn         v21.4h, v26.4s, #idct16_shift_2
    sqrshrn         v22.4h, v28.4s, #idct16_shift_2
    sqrshrn         v23.4h, v27.4s, #idct16_shift_2
    stp             d20, d21, [x1, #0]
    stp             d22, d23, [x1, #16]

    add             x1, x1, x2, lsl #1
    add             x5, x5, #(16*2)
    sub             w4, w4, #1
    cbnz            w4, 6b
9:
    add             sp, sp, #(16*16*2)
    ldp             d8, d9, [sp], #16
    ret
endfunc


// ***** dct 16x16 *****
// void dct16(const int16_t* src, int16_t* dst, intptr_t srcStride)
function PFX(dct16_neon)
// Register map
// x0  = src
// x1  = dst
// x2  = dstStride
// x3  = tbl_const_dct_0

    stp             d8, d9, [sp,#-16]!
    stp             d10, d11, [sp,#-16]!
    stp             d12, d13, [sp,#-16]!
    stp             d14, d15, [sp,#-16]!

    adr             x6, tbl_const_dct_0
    ld4r            {v16.2d, v17.2d, v18.2d, v19.2d}, [x6], #32
    ld4r            {v20.2d, v21.2d, v22.2d, v23.2d}, [x6], #32
    ld1             {v24.8h, v25.8h, v26.8h, v27.8h}, [x6], #64
    ld1             {v28.8h, v29.8h, v30.8h, v31.8h}, [x6], #64
    ldr             q0, [x6], #16

    add             x2, x2, x2
    mov             x5, x1
    mov             w4, #16

5:  // Pass1
    ld1             {v2.8h, v3.8h}, [x0], x2
    tbl             v3.16b, {v3.16b}, v0.16b

    add             v4.8h, v2.8h, v3.8h             // v4 = E[07 06 05 04 03 02 01 00]
    sub             v1.8h, v2.8h, v3.8h             // v1 = O[07 06 05 04 03 02 01 00]

// EE0 = E0 + E7;
// EO0 = E0 - E7;
// EE1 = E1 + E6;
// EO1 = E1 - E6;
// EE2 = E2 + E5;
// EO2 = E2 - E5;
// EE3 = E3 + E4;
// EO3 = E3 - E4;
    tbl             v2.8b, {v4.16b}, v0.8b          // v2 = E[04 05 06 07]

    add             v3.4h, v4.4h, v2.4h             // v3 = EE[03 02 01 00]
    sub             v2.4h, v4.4h, v2.4h             // v2 = EO[03 02 01 00]

// [ 0] = (64*EE0 + 64*EE1 + 64*EE2 + 64*EE3 + rnd) >> nShift;                              // v16
// [ 4] = (83*EE0 + 36*EE1 - 36*EE2 - 83*EE3 + rnd) >> nShift;                              // v17
// [ 8] = (64*EE0 - 64*EE1 - 64*EE2 + 64*EE3 + rnd) >> nShift;                              // v18
// [12] = (36*EE0 - 83*EE1 + 83*EE2 - 36*EE3 + rnd) >> nShift;                              // v19

// [ 2] = (89*EO0 + 75*EO1 + 50*EO2 + 18*EO3 + rnd) >> nShift;                              // v20
// [ 6] = (75*EO0 - 18*EO1 - 89*EO2 - 50*EO3 + rnd) >> nShift;                              // v21
// [10] = (50*EO0 - 89*EO1 + 18*EO2 + 75*EO3 + rnd) >> nShift;                              // v22
// [14] = (18*EO0 - 50*EO1 + 75*EO2 - 89*EO3 + rnd) >> nShift;                              // v23

// [ 1] = (90*O0 + 87*O1 + 80*O2 + 70*O3 + 57*O4 + 43*O5 + 25*O6 +  9*O7 + rnd) >> nShift;  // v24
// [ 3] = (87*O0 + 57*O1 +  9*O2 - 43*O3 - 80*O4 - 90*O5 - 70*O6 - 25*O7 + rnd) >> nShift;  // v25
// [ 5] = (80*O0 +  9*O1 - 70*O2 - 87*O3 - 25*O4 + 57*O5 + 90*O6 + 43*O7 + rnd) >> nShift;  // v26
// [ 7] = (70*O0 - 43*O1 - 87*O2 +  9*O3 + 90*O4 + 25*O5 - 80*O6 - 57*O7 + rnd) >> nShift;  // v27
// [ 9] = (57*O0 - 80*O1 - 25*O2 + 90*O3 -  9*O4 - 87*O5 + 43*O6 + 70*O7 + rnd) >> nShift;  // v28
// [11] = (43*O0 - 90*O1 + 57*O2 + 25*O3 - 87*O4 + 70*O5 +  9*O6 - 80*O7 + rnd) >> nShift;  // v29
// [13] = (25*O0 - 70*O1 + 90*O2 - 80*O3 + 43*O4 +  9*O5 - 57*O6 + 87*O7 + rnd) >> nShift;  // v30
// [15] = ( 9*O0 - 25*O1 + 43*O2 - 57*O3 + 70*O4 - 80*O5 + 87*O6 - 90*O7 + rnd) >> nShift;  // v31


    smull           v4.4s, v1.4h, v24.4h            // v4  = [ 1]
    smull           v5.4s, v1.4h, v25.4h            // v5  = [ 3]
    smull           v6.4s, v1.4h, v26.4h            // v6  = [ 5]
    smull           v7.4s, v1.4h, v27.4h            // v7  = [ 7]
    smull           v8.4s, v1.4h, v28.4h            // v8  = [ 9]
    smull           v9.4s, v1.4h, v29.4h            // v9  = [11]
    smull           v10.4s, v1.4h, v30.4h           // v10 = [13]
    smull           v11.4s, v1.4h, v31.4h           // v11 = [15]

    smlal2          v4.4s, v1.8h, v24.8h            // v4  = [ 1]
    smlal2          v5.4s, v1.8h, v25.8h            // v5  = [ 3]
    smlal2          v6.4s, v1.8h, v26.8h            // v6  = [ 5]
    smlal2          v7.4s, v1.8h, v27.8h            // v7  = [ 7]
    smlal2          v8.4s, v1.8h, v28.8h            // v8  = [ 9]
    smlal2          v9.4s, v1.8h, v29.8h            // v9  = [11]
    smlal2          v10.4s, v1.8h, v30.8h           // v10 = [13]
    smlal2          v11.4s, v1.8h, v31.8h           // v11 = [15]

    smull           v12.4s, v3.4h, v16.4h           // v12 = [ 0]
    smull           v13.4s, v2.4h, v20.4h           // v13 = [ 2]
    smull           v14.4s, v3.4h, v17.4h           // v14 = [ 4]
    smull           v15.4s, v2.4h, v21.4h           // v15 = [ 6]

    addp            v4.4s, v12.4s, v4.4s            // v4 = [1 0]
    addp            v5.4s, v13.4s, v5.4s            // v5 = [3 2]
    addp            v6.4s, v14.4s, v6.4s            // v6 = [5 4]
    addp            v7.4s, v15.4s, v7.4s            // v7 = [7 6]
    addp            v4.4s, v4.4s, v5.4s             // v4 = [3 2 1 0]
    addp            v5.4s, v6.4s, v7.4s             // v5 = [7 6 5 4]

    smull           v12.4s, v3.4h, v18.4h           // v12 = [ 8]
    smull           v13.4s, v2.4h, v22.4h           // v13 = [10]
    smull           v14.4s, v3.4h, v19.4h           // v14 = [12]
    smull           v15.4s, v2.4h, v23.4h           // v15 = [14]

    sqrshrn         v4.4h, v4.4s, #dct16_shift_1
    sqrshrn         v5.4h, v5.4s, #dct16_shift_1
    stp             d4, d5, [x5], #16

    addp            v6.4s, v12.4s, v8.4s            // v6 = [9 8]
    addp            v7.4s, v13.4s, v9.4s            // v7 = [11 10]
    addp            v8.4s, v14.4s, v10.4s           // v8 = [13 12]
    addp            v9.4s, v15.4s, v11.4s           // v9 = [15 14]
    addp            v6.4s, v6.4s, v7.4s             // v6 = [11 10 9 8]
    addp            v7.4s, v8.4s, v9.4s             // v7 = [15 14 13 12]

    sqrshrn         v6.4h, v6.4s, #dct16_shift_1
    sqrshrn         v7.4h, v7.4s, #dct16_shift_1
    stp             d6, d7, [x5], #16

    sub             w4, w4, #1
    cbnz            w4, 5b

    ld1             {v0.4s, v1.4s, v2.4s, v3.4s}, [x6]
    mov             w4, #16
    mov             x5, x1
6:  // Pass 2

    ldr             d16, [x5, #(16*2* 0)]
    ldr             d17, [x5, #(16*2* 1)]
    ldr             d18, [x5, #(16*2* 2)]
    ldr             d19, [x5, #(16*2* 3)]
    ldr             d20, [x5, #(16*2* 4)]
    ldr             d21, [x5, #(16*2* 5)]
    ldr             d22, [x5, #(16*2* 6)]
    ldr             d23, [x5, #(16*2* 7)]
    ldr             d24, [x5, #(16*2* 8)]
    ldr             d25, [x5, #(16*2* 9)]
    ldr             d26, [x5, #(16*2*10)]
    ldr             d27, [x5, #(16*2*11)]
    ldr             d28, [x5, #(16*2*12)]
    ldr             d29, [x5, #(16*2*13)]
    ldr             d30, [x5, #(16*2*14)]
    ldr             d31, [x5, #(16*2*15)]

    saddl           v4.4s, v16.4h, v31.4h           // v4  = E0
    saddl           v5.4s, v17.4h, v30.4h           // v5  = E1
    saddl           v6.4s, v18.4h, v29.4h           // v6  = E2
    saddl           v7.4s, v19.4h, v28.4h           // v7  = E3
    saddl           v8.4s, v20.4h, v27.4h           // v8  = E4
    saddl           v9.4s, v21.4h, v26.4h           // v9  = E5
    saddl           v10.4s, v22.4h, v25.4h          // v10 = E6
    saddl           v11.4s, v23.4h, v24.4h          // v11 = E7

// [ 1] = (90*O0 + 87*O1 + 80*O2 + 70*O3 + 57*O4 + 43*O5 + 25*O6 +  9*O7 + rnd) >> nShift;
// [ 3] = (87*O0 + 57*O1 +  9*O2 - 43*O3 - 80*O4 - 90*O5 - 70*O6 - 25*O7 + rnd) >> nShift;
// [ 5] = (80*O0 +  9*O1 - 70*O2 - 87*O3 - 25*O4 + 57*O5 + 90*O6 + 43*O7 + rnd) >> nShift;
// [ 7] = (70*O0 - 43*O1 - 87*O2 +  9*O3 + 90*O4 + 25*O5 - 80*O6 - 57*O7 + rnd) >> nShift;
// [ 9] = (57*O0 - 80*O1 - 25*O2 + 90*O3 -  9*O4 - 87*O5 + 43*O6 + 70*O7 + rnd) >> nShift;
// [11] = (43*O0 - 90*O1 + 57*O2 + 25*O3 - 87*O4 + 70*O5 +  9*O6 - 80*O7 + rnd) >> nShift;
// [13] = (25*O0 - 70*O1 + 90*O2 - 80*O3 + 43*O4 +  9*O5 - 57*O6 + 87*O7 + rnd) >> nShift;
// [15] = ( 9*O0 - 25*O1 + 43*O2 - 57*O3 + 70*O4 - 80*O5 + 87*O6 - 90*O7 + rnd) >> nShift;

    ssubl           v16.4s, v16.4h, v31.4h          // v16 = O0
    ssubl           v17.4s, v17.4h, v30.4h          // v17 = O1
    ssubl           v18.4s, v18.4h, v29.4h          // v18 = O2
    ssubl           v19.4s, v19.4h, v28.4h          // v19 = O3
    ssubl           v20.4s, v20.4h, v27.4h          // v20 = O4
    ssubl           v21.4s, v21.4h, v26.4h          // v21 = O5
    ssubl           v22.4s, v22.4h, v25.4h          // v22 = O6
    ssubl           v23.4s, v23.4h, v24.4h          // v23 = O7

    orr             v24.16b, v18.16b, v19.16b
    orr             v25.16b, v20.16b, v21.16b
    orr             v26.16b, v22.16b, v23.16b
    uqxtn           v24.4h, v24.4s
    uqxtn           v25.4h, v25.4s
    uqxtn           v26.4h, v26.4s
    mov             x0, v24.d[0]                    // x0 = zeros[O3 O2]
    mov             x2, v25.d[0]                    // x2 = zeros[O5 O4]
    mov             x6, v26.d[0]                    // x6 = zeros[O7 O6]

    mul             v24.4s, v16.4s, v2.s[0]         // v24 = [ 1] = 90*O0
    mul             v25.4s, v16.4s, v2.s[1]         // v25 = [ 3] = 87*O0
    mul             v26.4s, v16.4s, v2.s[2]         // v26 = [ 5] = 80*O0
    mul             v27.4s, v16.4s, v2.s[3]         // v27 = [ 7] = 70*O0
    mul             v28.4s, v16.4s, v3.s[0]         // v28 = [ 9] = 57*O0
    mul             v29.4s, v16.4s, v3.s[1]         // v29 = [11] = 43*O0
    mul             v30.4s, v16.4s, v3.s[2]         // v30 = [13] = 25*O0
    mul             v31.4s, v16.4s, v3.s[3]         // v31 = [15] =  9*O0

    mla             v24.4s, v17.4s, v2.s[1]         // v24 = [ 1] = 90*O0 + 87*O1
    mla             v25.4s, v17.4s, v3.s[0]         // v25 = [ 3] = 87*O0 + 57*O1
    mla             v26.4s, v17.4s, v3.s[3]         // v26 = [ 5] = 80*O0 +  9*O1
    mls             v27.4s, v17.4s, v3.s[1]         // v27 = [ 7] = 70*O0 - 43*O1
    mls             v28.4s, v17.4s, v2.s[2]         // v28 = [ 9] = 57*O0 - 80*O1
    mls             v29.4s, v17.4s, v2.s[0]         // v29 = [11] = 43*O0 - 90*O1
    mls             v30.4s, v17.4s, v2.s[3]         // v30 = [13] = 25*O0 - 70*O1
    mls             v31.4s, v17.4s, v3.s[2]         // v31 = [15] =  9*O0 - 25*O1

    cbz             x0, 1f

    mla             v24.4s, v18.4s, v2.s[2]         // v24 = [ 1] = 90*O0 + 87*O1 + 80*O2
    mla             v25.4s, v18.4s, v3.s[3]         // v25 = [ 3] = 87*O0 + 57*O1 +  9*O2
    mls             v26.4s, v18.4s, v2.s[3]         // v26 = [ 5] = 80*O0 +  9*O1 - 70*O2
    mls             v27.4s, v18.4s, v2.s[1]         // v27 = [ 7] = 70*O0 - 43*O1 - 87*O2
    mls             v28.4s, v18.4s, v3.s[2]         // v28 = [ 9] = 57*O0 - 80*O1 - 25*O2
    mla             v29.4s, v18.4s, v3.s[0]         // v29 = [11] = 43*O0 - 90*O1 + 57*O2
    mla             v30.4s, v18.4s, v2.s[0]         // v30 = [13] = 25*O0 - 70*O1 + 90*O2
    mla             v31.4s, v18.4s, v3.s[1]         // v31 = [15] =  9*O0 - 25*O1 + 43*O2

    mla             v24.4s, v19.4s, v2.s[3]         // v24 = [ 1] = 90*O0 + 87*O1 + 80*O2 + 70*O3
    mls             v25.4s, v19.4s, v3.s[1]         // v25 = [ 3] = 87*O0 + 57*O1 +  9*O2 - 43*O3
    mls             v26.4s, v19.4s, v2.s[1]         // v26 = [ 5] = 80*O0 +  9*O1 - 70*O2 - 87*O3
    mla             v27.4s, v19.4s, v3.s[3]         // v27 = [ 7] = 70*O0 - 43*O1 - 87*O2 +  9*O3
    mla             v28.4s, v19.4s, v2.s[0]         // v28 = [ 9] = 57*O0 - 80*O1 - 25*O2 + 90*O3
    mla             v29.4s, v19.4s, v3.s[2]         // v29 = [11] = 43*O0 - 90*O1 + 57*O2 + 25*O3
    mls             v30.4s, v19.4s, v2.s[2]         // v30 = [13] = 25*O0 - 70*O1 + 90*O2 - 80*O3
    mls             v31.4s, v19.4s, v3.s[0]         // v31 = [15] =  9*O0 - 25*O1 + 43*O2 - 57*O3

1:
    cbz             x2, 1f

    mla             v24.4s, v20.4s, v3.s[0]         // v24 = [ 1] = 90*O0 + 87*O1 + 80*O2 + 70*O3 + 57*O4
    mls             v25.4s, v20.4s, v2.s[2]         // v25 = [ 3] = 87*O0 + 57*O1 +  9*O2 - 43*O3 - 80*O4
    mls             v26.4s, v20.4s, v3.s[2]         // v26 = [ 5] = 80*O0 +  9*O1 - 70*O2 - 87*O3 - 25*O4
    mla             v27.4s, v20.4s, v2.s[0]         // v27 = [ 7] = 70*O0 - 43*O1 - 87*O2 +  9*O3 + 90*O4
    mls             v28.4s, v20.4s, v3.s[3]         // v28 = [ 9] = 57*O0 - 80*O1 - 25*O2 + 90*O3 -  9*O4
    mls             v29.4s, v20.4s, v2.s[1]         // v29 = [11] = 43*O0 - 90*O1 + 57*O2 + 25*O3 - 87*O4
    mla             v30.4s, v20.4s, v3.s[1]         // v30 = [13] = 25*O0 - 70*O1 + 90*O2 - 80*O3 + 43*O4
    mla             v31.4s, v20.4s, v2.s[3]         // v31 = [15] =  9*O0 - 25*O1 + 43*O2 - 57*O3 + 70*O4

    mla             v24.4s, v21.4s, v3.s[1]         // v24 = [ 1] = 90*O0 + 87*O1 + 80*O2 + 70*O3 + 57*O4 + 43*O5
    mls             v25.4s, v21.4s, v2.s[0]         // v25 = [ 3] = 87*O0 + 57*O1 +  9*O2 - 43*O3 - 80*O4 - 90*O5
    mla             v26.4s, v21.4s, v3.s[0]         // v26 = [ 5] = 80*O0 +  9*O1 - 70*O2 - 87*O3 - 25*O4 + 57*O5
    mla             v27.4s, v21.4s, v3.s[2]         // v27 = [ 7] = 70*O0 - 43*O1 - 87*O2 +  9*O3 + 90*O4 + 25*O5
    mls             v28.4s, v21.4s, v2.s[1]         // v28 = [ 9] = 57*O0 - 80*O1 - 25*O2 + 90*O3 -  9*O4 - 87*O5
    mla             v29.4s, v21.4s, v2.s[3]         // v29 = [11] = 43*O0 - 90*O1 + 57*O2 + 25*O3 - 87*O4 + 70*O5
    mla             v30.4s, v21.4s, v3.s[3]         // v30 = [13] = 25*O0 - 70*O1 + 90*O2 - 80*O3 + 43*O4 +  9*O5
    mls             v31.4s, v21.4s, v2.s[2]         // v31 = [15] =  9*O0 - 25*O1 + 43*O2 - 57*O3 + 70*O4 - 80*O5

1:
    cbz             x6, 1f

    mla             v24.4s, v22.4s, v3.s[2]         // v24 = [ 1] = 90*O0 + 87*O1 + 80*O2 + 70*O3 + 57*O4 + 43*O5 + 25*O6
    mls             v25.4s, v22.4s, v2.s[3]         // v25 = [ 3] = 87*O0 + 57*O1 +  9*O2 - 43*O3 - 80*O4 - 90*O5 - 70*O6
    mla             v26.4s, v22.4s, v2.s[0]         // v26 = [ 5] = 80*O0 +  9*O1 - 70*O2 - 87*O3 - 25*O4 + 57*O5 + 90*O6
    mls             v27.4s, v22.4s, v2.s[2]         // v27 = [ 7] = 70*O0 - 43*O1 - 87*O2 +  9*O3 + 90*O4 + 25*O5 - 80*O6
    mla             v28.4s, v22.4s, v3.s[1]         // v28 = [ 9] = 57*O0 - 80*O1 - 25*O2 + 90*O3 -  9*O4 - 87*O5 + 43*O6
    mla             v29.4s, v22.4s, v3.s[3]         // v29 = [11] = 43*O0 - 90*O1 + 57*O2 + 25*O3 - 87*O4 + 70*O5 +  9*O6
    mls             v30.4s, v22.4s, v3.s[0]         // v30 = [13] = 25*O0 - 70*O1 + 90*O2 - 80*O3 + 43*O4 +  9*O5 - 57*O6
    mla             v31.4s, v22.4s, v2.s[1]         // v31 = [15] =  9*O0 - 25*O1 + 43*O2 - 57*O3 + 70*O4 - 80*O5 + 87*O6

    mla             v24.4s, v23.4s, v3.s[3]         // v24 = [ 1] = 90*O0 + 87*O1 + 80*O2 + 70*O3 + 57*O4 + 43*O5 + 25*O6 +  9*O7
    mls             v25.4s, v23.4s, v3.s[2]         // v25 = [ 3] = 87*O0 + 57*O1 +  9*O2 - 43*O3 - 80*O4 - 90*O5 - 70*O6 - 25*O7
    mla             v26.4s, v23.4s, v3.s[1]         // v26 = [ 5] = 80*O0 +  9*O1 - 70*O2 - 87*O3 - 25*O4 + 57*O5 + 90*O6 + 43*O7
    mls             v27.4s, v23.4s, v3.s[0]         // v27 = [ 7] = 70*O0 - 43*O1 - 87*O2 +  9*O3 + 90*O4 + 25*O5 - 80*O6 - 57*O7
    mla             v28.4s, v23.4s, v2.s[3]         // v28 = [ 9] = 57*O0 - 80*O1 - 25*O2 + 90*O3 -  9*O4 - 87*O5 + 43*O6 + 70*O7
    mls             v29.4s, v23.4s, v2.s[2]         // v29 = [11] = 43*O0 - 90*O1 + 57*O2 + 25*O3 - 87*O4 + 70*O5 +  9*O6 - 80*O7
    mla             v30.4s, v23.4s, v2.s[1]         // v30 = [13] = 25*O0 - 70*O1 + 90*O2 - 80*O3 + 43*O4 +  9*O5 - 57*O6 + 87*O7
    mls             v31.4s, v23.4s, v2.s[0]         // v31 = [15] =  9*O0 - 25*O1 + 43*O2 - 57*O3 + 70*O4 - 80*O5 + 87*O6 - 90*O7

1:
    sqrshrn         v24.4h, v24.4s, #dct16_shift_2 // [1]
    sqrshrn         v25.4h, v25.4s, #dct16_shift_2 // [3]
    sqrshrn         v26.4h, v26.4s, #dct16_shift_2 // [5]
    sqrshrn         v27.4h, v27.4s, #dct16_shift_2 // [7]
    sqrshrn         v28.4h, v28.4s, #dct16_shift_2 // [9]
    sqrshrn         v29.4h, v29.4s, #dct16_shift_2 // [11]
    sqrshrn         v30.4h, v30.4s, #dct16_shift_2 // [13]
    sqrshrn         v31.4h, v31.4s, #dct16_shift_2 // [15]

    str             d24, [x5, #(16*2* 1)]
    str             d25, [x5, #(16*2* 3)]
    str             d26, [x5, #(16*2* 5)]
    str             d27, [x5, #(16*2* 7)]
    str             d28, [x5, #(16*2* 9)]
    str             d29, [x5, #(16*2*11)]
    str             d30, [x5, #(16*2*13)]
    str             d31, [x5, #(16*2*15)]

// EE0 = E0 + E7;
// EO0 = E0 - E7;
// EE1 = E1 + E6;
// EO1 = E1 - E6;
// EE2 = E2 + E5;
// EO2 = E2 - E5;
// EE3 = E3 + E4;
// EO3 = E3 - E4;
    add             v16.4s, v4.4s, v11.4s           // v16 = EE0
    sub             v17.4s, v4.4s, v11.4s           // v17 = EO0
    add             v18.4s, v5.4s, v10.4s           // v18 = EE1
    sub             v19.4s, v5.4s, v10.4s           // v19 = EO1
    add             v20.4s, v6.4s, v9.4s            // v20 = EE2
    sub             v21.4s, v6.4s, v9.4s            // v21 = EO2
    add             v22.4s, v7.4s, v8.4s            // v22 = EE3
    sub             v23.4s, v7.4s, v8.4s            // v23 = EO3

// EEE0 = EE0 + EE3;
// EEO0 = EE0 - EE3;
// EEE1 = EE1 + EE2;
// EEO1 = EE1 - EE2;

    add             v24.4s, v16.4s, v22.4s          // v24 = EEE0
    sub             v25.4s, v16.4s, v22.4s          // v25 = EEO0
    add             v26.4s, v18.4s, v20.4s          // v26 = EEE1
    sub             v27.4s, v18.4s, v20.4s          // v27 = EEO1

    orr             v28.16b, v21.16b, v23.16b
    uqxtn           v28.4h, v28.4s
    mov             x0, v28.d[0]                    // x0 = zeros[EO3 EO2]

// [ 0] = (64*EEE0 + 64*EEE1 + rnd) >> nShift;
// [ 4] = (83*EEO0 + 36*EEO1 + rnd) >> nShift;
// [ 8] = (64*EEE0 - 64*EEE1 + rnd) >> nShift;
// [12] = (36*EEO0 - 83*EEO1 + rnd) >> nShift;

    add             v28.4s, v24.4s, v26.4s          // [ 0] = EEE0+EEE1
    mul             v29.4s, v25.4s, v0.s[1]         // [ 4] = 83*EEO0
    sub             v30.4s, v24.4s, v26.4s          // [ 8] = EEE0-EEE1
    mul             v31.4s, v25.4s, v0.s[2]         // [12] = 36*EEO0

    shl             v28.4s, v28.4s, #6              // [ 0] = 64*EEE0 + 64*EEE1
    mla             v29.4s, v27.4s, v0.s[2]         // [ 4] = 83*EEO0 + 36*EEO1
    shl             v30.4s, v30.4s, #6              // [ 0] = 64*EEE0 - 64*EEE1
    mls             v31.4s, v27.4s, v0.s[1]         // [12] = 36*EEO0 - 83*EEO1

    sqrshrn         v28.4h, v28.4s, #dct16_shift_2  // [ 0]
    sqrshrn         v29.4h, v29.4s, #dct16_shift_2  // [ 4]
    sqrshrn         v30.4h, v30.4s, #dct16_shift_2  // [ 8]
    sqrshrn         v31.4h, v31.4s, #dct16_shift_2  // [12]

    str             d28, [x5, #(16*2* 0)]
    str             d29, [x5, #(16*2* 4)]
    str             d30, [x5, #(16*2* 8)]
    str             d31, [x5, #(16*2*12)]

// [ 2] = (89*EO0 + 75*EO1 + 50*EO2 + 18*EO3 + rnd) >> nShift;
// [ 6] = (75*EO0 - 18*EO1 - 89*EO2 - 50*EO3 + rnd) >> nShift;
// [10] = (50*EO0 - 89*EO1 + 18*EO2 + 75*EO3 + rnd) >> nShift;
// [14] = (18*EO0 - 50*EO1 + 75*EO2 - 89*EO3 + rnd) >> nShift;

    mul             v28.4s, v17.4s, v0.s[3]         // [ 2] = 89*EO0
    mul             v29.4s, v17.4s, v1.s[0]         // [ 6] = 75*EO0
    mul             v30.4s, v17.4s, v1.s[1]         // [10] = 50*EO0
    mul             v31.4s, v17.4s, v1.s[2]         // [14] = 18*EO0

    mla             v28.4s, v19.4s, v1.s[0]         // [ 2] = 89*EO0 + 75*EO1
    mls             v29.4s, v19.4s, v1.s[2]         // [ 6] = 75*EO0 - 18*EO1
    mls             v30.4s, v19.4s, v0.s[3]         // [10] = 50*EO0 - 89*EO1
    mls             v31.4s, v19.4s, v1.s[1]         // [14] = 18*EO0 - 50*EO1

    cbz             x0, 1f

    mla             v28.4s, v21.4s, v1.s[1]         // [ 2] = 89*EO0 + 75*EO1 + 50*EO2
    mls             v29.4s, v21.4s, v0.s[3]         // [ 6] = 75*EO0 - 18*EO1 - 89*EO2
    mla             v30.4s, v21.4s, v1.s[2]         // [10] = 50*EO0 - 89*EO1 + 18*EO2
    mla             v31.4s, v21.4s, v1.s[0]         // [14] = 18*EO0 - 50*EO1 + 75*EO2

    mla             v28.4s, v23.4s, v1.s[2]         // [ 2] = 89*EO0 + 75*EO1 + 50*EO2 + 18*EO3
    mls             v29.4s, v23.4s, v1.s[1]         // [ 6] = 75*EO0 - 18*EO1 - 89*EO2 - 50*EO3
    mla             v30.4s, v23.4s, v1.s[0]         // [10] = 50*EO0 - 89*EO1 + 18*EO2 + 75*EO3
    mls             v31.4s, v23.4s, v0.s[3]         // [14] = 18*EO0 - 50*EO1 + 75*EO2 - 89*EO3

1:

    sqrshrn         v28.4h, v28.4s, #dct16_shift_2  // [ 0]
    sqrshrn         v29.4h, v29.4s, #dct16_shift_2  // [ 4]
    sqrshrn         v30.4h, v30.4s, #dct16_shift_2  // [ 8]
    sqrshrn         v31.4h, v31.4s, #dct16_shift_2  // [12]

    str             d28, [x5, #(16*2* 2)]
    str             d29, [x5, #(16*2* 6)]
    str             d30, [x5, #(16*2*10)]
    str             d31, [x5, #(16*2*14)]

    add             x5, x5, #(4*2)
    sub             w4, w4, #4
    cbnz            w4, 6b

9:
    ldp             d14, d15, [sp], #16
    ldp             d12, d13, [sp], #16
    ldp             d10, d11, [sp], #16
    ldp             d8, d9, [sp], #16
    ret
endfunc
