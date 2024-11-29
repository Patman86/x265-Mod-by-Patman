/*****************************************************************************
 * Copyright (C) 2020-2024 MulticoreWare, Inc
 *
 * Authors: Hongbin Liu <liuhongbin1@huawei.com>
 *          Sebastian Pop <spop@amazon.com>
 *          Hari Limaye <hari.limaye@arm.com>
 *          Gerda Zsejke More <gerdazsejke.more@arm.com>
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

#include "asm.S"

#ifdef __APPLE__
.section __RODATA,__rodata
#else
.section .rodata
#endif

.align 4

.text

#if !HIGH_BIT_DEPTH
.macro SAD_START_4 f
    ldr             s0, [x0]
    ldr             s1, [x2]
    add             x0, x0, x1
    add             x2, x2, x3
    ld1             {v0.s}[1], [x0], x1
    ld1             {v1.s}[1], [x2], x3
    \f              v16.8h, v0.8b, v1.8b
.endm

.macro SAD_4 h
.rept \h / 2 - 1
    SAD_START_4 uabal
.endr
.endm

.macro SAD_START_8 f
    ld1             {v0.8b}, [x0], x1
    ld1             {v1.8b}, [x2], x3
    \f              v16.8h, v0.8b, v1.8b
.endm

.macro SAD_8 h
.rept \h - 3
    SAD_START_8 uabal
.endr
    ldr             d0, [x0]
    ldr             d1, [x2]
    uabal           v16.8h, v0.8b, v1.8b
    ldr             d0, [x0, x1]
    ldr             d1, [x2, x3]
    uabal           v16.8h, v0.8b, v1.8b
.endm

.macro SAD_START_16
    movi            v16.16b, #0
    movi            v17.16b, #0
.endm

.macro SAD_16
    ld1             {v0.16b}, [x0], x1
    ld1             {v1.16b}, [x2], x3
    ld1             {v2.16b}, [x0], x1
    ld1             {v3.16b}, [x2], x3
    uabd            v20.16b, v0.16b, v1.16b
    uadalp          v16.8h, v20.16b
    uabd            v21.16b, v2.16b, v3.16b
    uadalp          v17.8h, v21.16b
.endm

.macro SAD_END_16
    add             v16.8h, v16.8h, v17.8h
    uaddlv          s0, v16.8h
    fmov            x0, d0
    ret
.endm

.macro SAD_START_32
    movi            v16.16b, #0
    movi            v17.16b, #0
    movi            v18.16b, #0
    movi            v19.16b, #0
.endm

.macro SAD_32
    ld1             {v0.16b-v1.16b}, [x0], x1
    ld1             {v2.16b-v3.16b}, [x2], x3
    ld1             {v4.16b-v5.16b}, [x0], x1
    ld1             {v6.16b-v7.16b}, [x2], x3
    uabd            v20.16b, v0.16b, v2.16b
    uadalp          v16.8h, v20.16b
    uabd            v21.16b, v1.16b, v3.16b
    uadalp          v17.8h, v21.16b
    uabd            v22.16b, v4.16b, v6.16b
    uadalp          v18.8h, v22.16b
    uabd            v23.16b, v5.16b, v7.16b
    uadalp          v19.8h, v23.16b
.endm

.macro SAD_END_32
    add             v16.8h, v16.8h, v17.8h
    add             v17.8h, v18.8h, v19.8h
    add             v16.8h, v16.8h, v17.8h
    uaddlv          s0, v16.8h
    fmov            w0, s0
    ret
.endm

.macro SAD_START_64
    movi            v16.16b, #0
    movi            v17.16b, #0
    movi            v18.16b, #0
    movi            v19.16b, #0
.endm

.macro SAD_64
    ld1             {v0.16b-v3.16b}, [x0], x1
    ld1             {v4.16b-v7.16b}, [x2], x3
    ld1             {v24.16b-v27.16b}, [x0], x1
    ld1             {v28.16b-v31.16b}, [x2], x3
    uabd            v20.16b, v0.16b, v4.16b
    uadalp          v16.8h, v20.16b
    uabd            v21.16b, v1.16b, v5.16b
    uadalp          v17.8h, v21.16b
    uabd            v22.16b, v2.16b, v6.16b
    uadalp          v18.8h, v22.16b
    uabd            v23.16b, v3.16b, v7.16b
    uadalp          v19.8h, v23.16b
    uabd            v20.16b, v24.16b, v28.16b
    uadalp          v16.8h, v20.16b
    uabd            v21.16b, v25.16b, v29.16b
    uadalp          v17.8h, v21.16b
    uabd            v22.16b, v26.16b, v30.16b
    uadalp          v18.8h, v22.16b
    uabd            v23.16b, v27.16b, v31.16b
    uadalp          v19.8h, v23.16b
.endm

.macro SAD_END_64
    uaddlp          v16.4s, v16.8h
    uadalp          v16.4s, v17.8h
    uadalp          v16.4s, v18.8h
    uadalp          v16.4s, v19.8h
    uaddlv          d0, v16.4s
    fmov            x0, d0
    ret
.endm

.macro SAD_START_12
    movrel          x12, sad12_mask
    ld1             {v31.16b}, [x12]
    movi            v16.16b, #0
    movi            v17.16b, #0
.endm

.macro SAD_12
    ld1             {v0.16b}, [x0], x1
    and             v0.16b, v0.16b, v31.16b
    ld1             {v1.16b}, [x2], x3
    and             v1.16b, v1.16b, v31.16b
    ld1             {v2.16b}, [x0], x1
    and             v2.16b, v2.16b, v31.16b
    ld1             {v3.16b}, [x2], x3
    and             v3.16b, v3.16b, v31.16b
    uabd            v20.16b, v0.16b, v1.16b
    uadalp          v16.8h, v20.16b
    uabd            v21.16b, v2.16b, v3.16b
    uadalp          v17.8h, v21.16b
.endm

.macro SAD_END_12
    add             v16.8h, v16.8h, v17.8h
    uaddlv          s0, v16.8h
    fmov            w0, s0
    ret
.endm

.macro SAD_START_24
    movi            v16.16b, #0
    movi            v17.16b, #0
    sub             x1, x1, #16
    sub             x3, x3, #16
.endm

.macro SAD_24
    ld1             {v0.16b}, [x0], #16
    ld1             {v1.8b}, [x0], x1
    ld1             {v2.16b}, [x2], #16
    ld1             {v3.8b}, [x2], x3
    ld1             {v4.16b}, [x0], #16
    ld1             {v5.8b}, [x0], x1
    ld1             {v6.16b}, [x2], #16
    ld1             {v7.8b}, [x2], x3
    uabd            v20.16b, v0.16b, v2.16b
    uadalp          v16.8h, v20.16b
    uabal           v17.8h, v1.8b, v3.8b
    uabd            v20.16b, v4.16b, v6.16b
    uadalp          v16.8h, v20.16b
    uabal           v17.8h, v5.8b, v7.8b
.endm

.macro SAD_END_24
    add             v16.8h, v16.8h, v17.8h
    uaddlv          s0, v16.8h
    fmov            w0, s0
    ret
.endm

.macro SAD_START_48
    movi            v16.16b, #0
    movi            v17.16b, #0
    movi            v18.16b, #0
.endm

.macro SAD_48
    ld1             {v0.16b-v2.16b}, [x0], x1
    ld1             {v4.16b-v6.16b}, [x2], x3
    ld1             {v24.16b-v26.16b}, [x0], x1
    ld1             {v28.16b-v30.16b}, [x2], x3
    uabd            v20.16b, v0.16b, v4.16b
    uadalp          v16.8h, v20.16b
    uabd            v21.16b, v1.16b, v5.16b
    uadalp          v17.8h, v21.16b
    uabd            v22.16b, v2.16b, v6.16b
    uadalp          v18.8h, v22.16b
    uabd            v20.16b, v24.16b, v28.16b
    uadalp          v16.8h, v20.16b
    uabd            v21.16b, v25.16b, v29.16b
    uadalp          v17.8h, v21.16b
    uabd            v22.16b, v26.16b, v30.16b
    uadalp          v18.8h, v22.16b
.endm

.macro SAD_END_48
    uaddlp          v16.4s, v16.8h
    uadalp          v16.4s, v17.8h
    uadalp          v16.4s, v18.8h
    uaddlv          d0, v16.4s
    fmov            x0, d0
    ret
.endm

// Fully unrolled.
.macro SAD_FUNC w, h
function PFX(pixel_sad_\w\()x\h\()_neon)
    SAD_START_\w uabdl
    SAD_\w \h
.if \w > 8
    add             v16.8h, v16.8h, v17.8h
.endif
    uaddlv          s0, v16.8h
    fmov            w0, s0
    ret
endfunc
.endm

// Loop unrolled to process 4 rows per iteration.
.macro SAD_FUNC_LOOP w, h
function PFX(pixel_sad_\w\()x\h\()_neon)
    SAD_START_\w

    mov             w9, #\h/4
.Loop_\w\()x\h:
    sub             w9, w9, #1
.rept 2
    SAD_\w
.endr
    cbnz            w9, .Loop_\w\()x\h

    SAD_END_\w
endfunc
.endm

SAD_FUNC  4,  4
SAD_FUNC  4,  8
SAD_FUNC  4,  16
SAD_FUNC  8,  4
SAD_FUNC  8,  8
SAD_FUNC  8,  16
SAD_FUNC  8,  32

SAD_FUNC_LOOP  16, 4
SAD_FUNC_LOOP  16, 8
SAD_FUNC_LOOP  16, 12
SAD_FUNC_LOOP  16, 16
SAD_FUNC_LOOP  16, 32
SAD_FUNC_LOOP  16, 64
SAD_FUNC_LOOP  32, 8
SAD_FUNC_LOOP  32, 16
SAD_FUNC_LOOP  32, 24
SAD_FUNC_LOOP  32, 32
SAD_FUNC_LOOP  32, 64
SAD_FUNC_LOOP  64, 16
SAD_FUNC_LOOP  64, 32
SAD_FUNC_LOOP  64, 48
SAD_FUNC_LOOP  64, 64
SAD_FUNC_LOOP  12, 16
SAD_FUNC_LOOP  24, 32
SAD_FUNC_LOOP  48, 64

// SAD_X3 and SAD_X4 code start

.macro SAD_X_START_4 h, x, f
    ldr             s0, [x0]
    ldr             s1, [x1]
    ldr             s2, [x2]
    ldr             s3, [x3]
    add             x0, x0, x9
    add             x1, x1, x5
    add             x2, x2, x5
    add             x3, x3, x5
    ld1             {v0.s}[1], [x0], x9
    ld1             {v1.s}[1], [x1], x5
    ld1             {v2.s}[1], [x2], x5
    ld1             {v3.s}[1], [x3], x5
    \f              v16.8h, v0.8b, v1.8b
    \f              v17.8h, v0.8b, v2.8b
    \f              v18.8h, v0.8b, v3.8b
.if \x == 4
    ldr             s4, [x4]
    add             x4, x4, x5
    ld1             {v4.s}[1], [x4], x5
    \f              v19.8h, v0.8b, v4.8b
.endif
.endm

.macro SAD_X_4 h, x
.rept \h/2 - 1
    SAD_X_START_4 \h, \x, uabal
.endr
.endm

.macro SAD_X_END_4 x
    uaddlv          s0, v16.8h
    uaddlv          s1, v17.8h
    uaddlv          s2, v18.8h
    stp             s0, s1, [x6]
.if \x == 3
    str             s2, [x6, #8]
.elseif \x == 4
    uaddlv          s3, v19.8h
    stp             s2, s3, [x6, #8]
.endif
    ret
.endm

.macro SAD_X_START_8 h, x, f
    ld1             {v0.8b}, [x0], x9
    ld1             {v1.8b}, [x1], x5
    ld1             {v2.8b}, [x2], x5
    ld1             {v3.8b}, [x3], x5
    \f              v16.8h, v0.8b, v1.8b
    \f              v17.8h, v0.8b, v2.8b
    \f              v18.8h, v0.8b, v3.8b
.if \x == 4
    ld1             {v4.8b}, [x4], x5
    \f              v19.8h, v0.8b, v4.8b
.endif
.endm

.macro SAD_X_8 h x
.rept \h - 1
    SAD_X_START_8 \h, \x, uabal
.endr
.endm

.macro SAD_X_END_8 x
    SAD_X_END_4 \x
.endm

.macro SAD_X_START_12 x
    SAD_X_START_16 \x
.endm

.macro SAD_X_12 base v1 v2
    // v2: unused
    // v31: bitmask for 12xh blocks
    ld1             {v0.16b}, [ \base ], x5
    and             v0.16b, v0.16b, v31.16b

    uabd            v24.16b, v0.16b, v6.16b
    uadalp          \v1\().8h, v24.16b
.endm

.macro SAD_X_END_12 x
    SAD_X_END_4 \x
.endm

.macro SAD_X_START_16 x
    movi v16.16b, #0
    movi v17.16b, #0
    movi v18.16b, #0
.if \x == 4
    movi v19.16b, #0
.endif
.endm

.macro SAD_X_16 base v1 v2
    // v2: unused
    ld1             {v0.16b}, [ \base ], x5
    uabd            v24.16b, v0.16b, v6.16b
    uadalp          \v1\().8h, v24.16b
.endm

.macro SAD_X_END_16 x
    SAD_X_END_4 \x
.endm

.macro SAD_X_START_LARGE x
    movi v16.16b, #0
    movi v17.16b, #0
    movi v18.16b, #0
    movi v20.16b, #0
    movi v21.16b, #0
    movi v22.16b, #0
.if \x == 4
    movi v19.16b, #0
    movi v23.16b, #0
.endif
.endm

.macro SAD_X_END_LARGE x
    uaddlp          v16.4s, v16.8h
    uadalp          v16.4s, v20.8h
    uaddlp          v17.4s, v17.8h
    uadalp          v17.4s, v21.8h
    uaddlp          v18.4s, v18.8h
    uadalp          v18.4s, v22.8h
.if \x == 3
    addv            s0, v16.4s
    addv            s1, v17.4s
    addv            s2, v18.4s
    stp             s0, s1, [x6], #8
    str             s2, [x6]
.elseif \x == 4
    uaddlp          v19.4s, v19.8h
    uadalp          v19.4s, v23.8h
    addp            v16.4s, v16.4s, v17.4s
    addp            v18.4s, v18.4s, v19.4s
    addp            v16.4s, v16.4s, v18.4s
    str             q16, [x6]
.endif
    ret
.endm

.macro SAD_X_START_24 x
    SAD_X_START_LARGE \x
    sub             x5, x5, #16
    sub             x9, x9, #16
.endm

.macro SAD_X_24 base v1 v2
    ld1             {v0.16b}, [ \base ], #16
    ld1             {v1.8b}, [ \base ], x5
    uabd            v24.16b, v0.16b, v6.16b
    uadalp          \v1\().8h, v24.16b
    uabal           \v2\().8h, v1.8b, v7.8b
.endm

.macro SAD_X_END_24 x
    SAD_X_END_LARGE \x
.endm

.macro SAD_X_START_32 x
    SAD_X_START_LARGE \x
.endm

.macro SAD_X_32 base v1 v2
    ld1             {v0.16b-v1.16b}, [ \base ], x5
    uabd            v24.16b, v0.16b, v6.16b
    uadalp          \v1\().8h, v24.16b
    uabd            v25.16b, v1.16b, v7.16b
    uadalp          \v2\().8h, v25.16b
.endm

.macro SAD_X_END_32 x
    SAD_X_END_LARGE \x
.endm

.macro SAD_X_START_48 x
    SAD_X_START_LARGE \x
.endm

.macro SAD_X_48 base v1 v2
    ld1             {v0.16b-v2.16b}, [ \base ], x5
    uabd            v24.16b, v0.16b, v4.16b
    uadalp          \v1\().8h, v24.16b
    uabd            v25.16b, v1.16b, v5.16b
    uadalp          \v2\().8h, v25.16b
    uabd            v26.16b, v2.16b, v6.16b
    uadalp          \v1\().8h, v26.16b
.endm

.macro SAD_X_END_48 x
    SAD_X_END_LARGE \x
.endm

.macro SAD_X_START_64 x
    SAD_X_START_LARGE \x
.endm

.macro SAD_X_64 base v1 v2
    ld1             {v0.16b-v3.16b}, [ \base ], x5
    uabd            v24.16b, v0.16b, v4.16b
    uadalp          \v1\().8h, v24.16b
    uabd            v25.16b, v1.16b, v5.16b
    uadalp          \v2\().8h, v25.16b
    uabd            v26.16b, v2.16b, v6.16b
    uadalp          \v1\().8h, v26.16b
    uabd            v27.16b, v3.16b, v7.16b
    uadalp          \v2\().8h, v27.16b
.endm

.macro SAD_X_END_64 x
    SAD_X_END_LARGE \x
.endm

// static void x264_pixel_sad_x3_##size(pixel *fenc, pixel *pix0, pixel *pix1, pixel *pix2, intptr_t i_stride, int scores[3])
// static void x264_pixel_sad_x4_##size(pixel *fenc, pixel *pix0, pixel *pix1,pixel *pix2, pixel *pix3, intptr_t i_stride, int scores[4])
.macro SAD_X_FUNC x, w, h
function PFX(sad_x\x\()_\w\()x\h\()_neon)
    mov             x9, #FENC_STRIDE

// Make function arguments for x == 3 look like x == 4.
.if \x == 3
    mov             x6, x5
    mov             x5, x4
.endif

    SAD_X_START_\w \h, \x, uabdl
    SAD_X_\w \h, \x
    SAD_X_END_\w \x
endfunc
.endm

.macro SAD_X_LOOP x, w, h
function PFX(sad_x\x\()_\w\()x\h\()_neon)
    mov             x9, #FENC_STRIDE

// Make function arguments for x == 3 look like x == 4.
.if \x == 3
    mov             x6, x5
    mov             x5, x4
.endif

.if \w == 12
    movrel          x12, sad12_mask
    ld1             {v31.16b}, [x12]
.endif

    SAD_X_START_\w \x
    mov             w12, #\h/4
.Loop_sad_x\x\()_\w\()x\h:
    sub             w12, w12, #1
 .rept 4
  .if \w == 12
    ld1             {v6.16b}, [x0], x9
    and             v6.16b, v6.16b, v31.16b
  .elseif \w == 16
    ld1             {v6.16b}, [x0], x9
  .elseif \w == 24
    ld1             {v6.16b}, [x0], #16
    ld1             {v7.8b}, [x0], x9
  .elseif \w == 32
    ld1             {v6.16b-v7.16b}, [x0], x9
  .elseif \w == 48
    ld1             {v4.16b-v6.16b}, [x0], x9
  .elseif \w == 64
    ld1             {v4.16b-v7.16b}, [x0], x9
  .endif
    SAD_X_\w x1, v16, v20
    SAD_X_\w x2, v17, v21
    SAD_X_\w x3, v18, v22
  .if \x == 4
    SAD_X_\w x4, v19, v23
  .endif
 .endr
    cbnz            w12, .Loop_sad_x\x\()_\w\()x\h
    SAD_X_END_\w \x
endfunc
.endm


SAD_X_FUNC  3, 4,  4
SAD_X_FUNC  3, 4,  8
SAD_X_FUNC  3, 4,  16
SAD_X_FUNC  3, 8,  4
SAD_X_FUNC  3, 8,  8
SAD_X_FUNC  3, 8,  16
SAD_X_FUNC  3, 8,  32
SAD_X_LOOP  3, 12, 16
SAD_X_LOOP  3, 16, 4
SAD_X_LOOP  3, 16, 8
SAD_X_LOOP  3, 16, 12
SAD_X_LOOP  3, 16, 16
SAD_X_LOOP  3, 16, 32
SAD_X_LOOP  3, 16, 64
SAD_X_LOOP  3, 24, 32
SAD_X_LOOP  3, 32, 8
SAD_X_LOOP  3, 32, 16
SAD_X_LOOP  3, 32, 24
SAD_X_LOOP  3, 32, 32
SAD_X_LOOP  3, 32, 64
SAD_X_LOOP  3, 48, 64
SAD_X_LOOP  3, 64, 16
SAD_X_LOOP  3, 64, 32
SAD_X_LOOP  3, 64, 48
SAD_X_LOOP  3, 64, 64

SAD_X_FUNC  4, 4,  4
SAD_X_FUNC  4, 4,  8
SAD_X_FUNC  4, 4,  16
SAD_X_FUNC  4, 8,  4
SAD_X_FUNC  4, 8,  8
SAD_X_FUNC  4, 8,  16
SAD_X_FUNC  4, 8,  32
SAD_X_LOOP  4, 12, 16
SAD_X_LOOP  4, 16, 4
SAD_X_LOOP  4, 16, 8
SAD_X_LOOP  4, 16, 12
SAD_X_LOOP  4, 16, 16
SAD_X_LOOP  4, 16, 32
SAD_X_LOOP  4, 16, 64
SAD_X_LOOP  4, 24, 32
SAD_X_LOOP  4, 32, 8
SAD_X_LOOP  4, 32, 16
SAD_X_LOOP  4, 32, 24
SAD_X_LOOP  4, 32, 32
SAD_X_LOOP  4, 32, 64
SAD_X_LOOP  4, 48, 64
SAD_X_LOOP  4, 64, 16
SAD_X_LOOP  4, 64, 32
SAD_X_LOOP  4, 64, 48
SAD_X_LOOP  4, 64, 64

const sad12_mask, align=8
.byte 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0
endconst

#else // HIGH_BIT_DEPTH

// int sad(const int16_t* pix1, intptr_t stride_pix1, const int16_t* pix2, intptr_t stride_pix2)
.macro SAD_4 f
    ld1             {v0.4h}, [x0], x1
    ld1             {v1.4h}, [x2], x3
    \f              v16.4s, v0.4h, v1.4h
.endm

.macro SAD_4xH h
    SAD_4 uabdl
.rept \h - 1
    SAD_4 uabal
.endr
    addv            s0, v16.4s
.endm

.macro SAD_8x2 f
    ld1             {v0.8h}, [x0], x1
    ld1             {v1.8h}, [x2], x3
    \f              v16.8h, v0.8h, v1.8h

    ld1             {v0.8h}, [x0], x1
    ld1             {v1.8h}, [x2], x3
    \f              v17.8h, v0.8h, v1.8h
.endm

.macro SAD_8xH h
    SAD_8x2 uabd
.rept \h / 2 - 1
    SAD_8x2 uaba
.endr
    uaddlp          v16.4s, v16.8h
    uadalp          v16.4s, v17.8h
    addv            s0, v16.4s
.endm

.macro SAD_FUNC w, h
function PFX(pixel_sad_\w\()x\h\()_neon)
    // Stride is given in terms of pixel channel size, so double to get number of bytes.
    add             x1, x1, x1
    add             x3, x3, x3

    SAD_\w\()xH \h

    fmov            w0, s0
    ret
endfunc
.endm

.macro SAD_12 f
    ldr             q0, [x0]
    ldr             q1, [x2]
    ldr             d2, [x0, #16]
    ldr             d3, [x2, #16]
    \f              v16.8h, v0.8h, v1.8h
    \f              v17.4h, v2.4h, v3.4h
    add             x0, x0, x1
    add             x2, x2, x3
.endm

.macro SAD_16 f
    ld1             {v0.8h-v1.8h}, [x0], x1
    ld1             {v2.8h-v3.8h}, [x2], x3
    \f              v16.8h, v0.8h, v2.8h
    \f              v17.8h, v1.8h, v3.8h
.endm

.macro SAD_32 f
    ld1             {v0.8h-v3.8h}, [x0], x1
    ld1             {v4.8h-v7.8h}, [x2], x3
    \f              v16.8h, v0.8h, v4.8h
    \f              v17.8h, v1.8h, v5.8h
    \f              v18.8h, v2.8h, v6.8h
    \f              v19.8h, v3.8h, v7.8h
.endm

.macro SAD_END_2_ACCUM
    add             v16.8h, v16.8h, v17.8h
    uaddlv          s0, v16.8h
.endm

.macro SAD_END_2_ACCUM_WIDEN
    uaddlp          v16.4s, v16.8h
    uadalp          v16.4s, v17.8h
    addv            s0, v16.4s
.endm

.macro SAD_END_4_ACCUM_WIDEN
    add             v16.8h, v16.8h, v17.8h
    add             v18.8h, v18.8h, v19.8h
    uaddlp          v16.4s, v16.8h
    uadalp          v16.4s, v18.8h
    addv            s0, v16.4s
.endm

.macro SAD_FUNC_LOOP w, h end_type
function PFX(pixel_sad_\w\()x\h\()_neon)
    // Stride is given in terms of pixel channel size, so double to get number of bytes.
    add             x1, x1, x1
    add             x3, x3, x3

    SAD_\w uabd
    SAD_\w uaba

    mov             w9, #(\h - 2)/2

.Loop_\w\()x\h:
    sub             w9, w9, #1
.rept 2
    SAD_\w uaba
.endr
    cbnz            w9, .Loop_\w\()x\h

    SAD_\end_type

    fmov            w0, s0
    ret
endfunc
.endm

// SAD_<w>_WIDEN kernels widen into 32-bit accumulators.
.macro SAD_16_WIDEN f
    ld1             {v0.8h-v1.8h}, [x0], x1
    ld1             {v2.8h-v3.8h}, [x2], x3
    uabd            v18.8h, v0.8h, v2.8h
    \f              v16.4s, v18.8h
    uabd            v19.8h, v1.8h, v3.8h
    \f              v17.4s, v19.8h
.endm

.macro SAD_24_WIDEN f
    ld1             {v0.8h-v2.8h}, [x0], x1
    ld1             {v3.8h-v5.8h}, [x2], x3
    uabd            v19.8h, v0.8h, v3.8h
    \f              v16.4s, v19.8h
    uabd            v20.8h, v1.8h, v4.8h
    \f              v17.4s, v20.8h
    uabd            v21.8h, v2.8h, v5.8h
    \f              v18.4s, v21.8h
.endm

.macro SAD_32_WIDEN f
    ld1             {v0.8h-v3.8h}, [x0], x1
    ld1             {v4.8h-v7.8h}, [x2], x3
    uabd            v20.8h, v0.8h, v4.8h
    \f              v16.4s, v20.8h
    uabd            v21.8h, v1.8h, v5.8h
    \f              v17.4s, v21.8h
    uabd            v22.8h, v2.8h, v6.8h
    \f              v18.4s, v22.8h
    uabd            v23.8h, v3.8h, v7.8h
    \f              v19.4s, v23.8h
.endm

.macro SAD_48_WIDEN f
    ld1             {v0.8h-v3.8h}, [x0]
    ld1             {v4.8h-v7.8h}, [x2]
    uabd            v20.8h, v0.8h, v4.8h
    \f              v16.4s, v20.8h
    uabd            v21.8h, v1.8h, v5.8h
    \f              v17.4s, v21.8h
    uabd            v22.8h, v2.8h, v6.8h
    \f              v18.4s, v22.8h
    uabd            v23.8h, v3.8h, v7.8h
    \f              v19.4s, v23.8h

    ldp             q0, q1, [x0, #64]
    ldp             q4, q5, [x2, #64]
    uabd            v20.8h, v0.8h, v4.8h
    uadalp          v16.4s, v20.8h
    uabd            v21.8h, v1.8h, v5.8h
    uadalp          v17.4s, v21.8h

    add             x0, x0, x1
    add             x2, x2, x3
.endm

.macro SAD_64_WIDEN f
    ld1             {v0.8h-v3.8h}, [x0]
    ld1             {v4.8h-v7.8h}, [x2]
    uabd            v20.8h, v0.8h, v4.8h
    \f              v16.4s, v20.8h
    uabd            v21.8h, v1.8h, v5.8h
    \f              v17.4s, v21.8h
    uabd            v22.8h, v2.8h, v6.8h
    \f              v18.4s, v22.8h
    uabd            v23.8h, v3.8h, v7.8h
    \f              v19.4s, v23.8h

    ldp             q0, q1, [x0, #64]
    ldp             q2, q3, [x0, #96]
    ldp             q4, q5, [x2, #64]
    ldp             q6, q7, [x2, #96]
    uabd            v20.8h, v0.8h, v4.8h
    uadalp          v16.4s, v20.8h
    uabd            v21.8h, v1.8h, v5.8h
    uadalp          v17.4s, v21.8h
    uabd            v22.8h, v2.8h, v6.8h
    uadalp          v18.4s, v22.8h
    uabd            v23.8h, v3.8h, v7.8h
    uadalp          v19.4s, v23.8h

    add             x0, x0, x1
    add             x2, x2, x3
.endm


.macro SAD_FUNC_LOOP_LARGE w, h
function PFX(pixel_sad_\w\()x\h\()_neon)
    // Stride is given in terms of pixel channel size, so double to get number of bytes.
    add             x1, x1, x1
    add             x3, x3, x3

    SAD_\w\()_WIDEN uaddlp
    SAD_\w\()_WIDEN uadalp

    mov             w9, #(\h - 2)/2
.Loop_\w\()x\h:
    sub             w9, w9, #1
.rept 2
    SAD_\w\()_WIDEN uadalp
.endr
    cbnz            w9, .Loop_\w\()x\h

    add             v16.4s, v16.4s, v17.4s
.if \w != 16
.if \w != 24
    add             v18.4s, v18.4s, v19.4s
.endif
    add             v16.4s, v16.4s, v18.4s
.endif
    addv            s0, v16.4s

    fmov            w0, s0
    ret
endfunc
.endm

SAD_FUNC  4, 4
SAD_FUNC  4, 8
SAD_FUNC  4, 16
SAD_FUNC  8, 4
SAD_FUNC  8, 8
SAD_FUNC  8, 16
SAD_FUNC  8, 32
SAD_FUNC_LOOP  12, 16, END_2_ACCUM_WIDEN
SAD_FUNC_LOOP  16, 4, END_2_ACCUM
SAD_FUNC_LOOP  16, 8, END_2_ACCUM
SAD_FUNC_LOOP  16, 12, END_2_ACCUM_WIDEN
SAD_FUNC_LOOP  16, 16, END_2_ACCUM_WIDEN
SAD_FUNC_LOOP  32, 8, END_4_ACCUM_WIDEN
SAD_FUNC_LOOP_LARGE  16, 32
SAD_FUNC_LOOP_LARGE  16, 64
SAD_FUNC_LOOP_LARGE  24, 32
SAD_FUNC_LOOP_LARGE  32, 16
SAD_FUNC_LOOP_LARGE  32, 24
SAD_FUNC_LOOP_LARGE  32, 32
SAD_FUNC_LOOP_LARGE  32, 64
SAD_FUNC_LOOP_LARGE  48, 64
SAD_FUNC_LOOP_LARGE  64, 16
SAD_FUNC_LOOP_LARGE  64, 32
SAD_FUNC_LOOP_LARGE  64, 48
SAD_FUNC_LOOP_LARGE  64, 64

//void sad_x3(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, intptr_t frefstride, int32_t* res)
//void sad_x4(const pixel* pix1, const pixel* pix2, const pixel* pix3, const pixel* pix4, const pixel* pix5, intptr_t frefstride, int32_t* res)
.macro SAD_xN_4 n, f
    ld1             {v0.4h}, [x0], x7
    ld1             {v1.4h}, [x1], x5
    ld1             {v2.4h}, [x2], x5
    ld1             {v3.4h}, [x3], x5
    \f              v16.4s, v0.4h, v1.4h
    \f              v17.4s, v0.4h, v2.4h
    \f              v18.4s, v0.4h, v3.4h
.if \n == 4
    ld1             {v4.4h}, [x4], x5
    \f              v19.4s, v0.4h, v4.4h
.endif
.endm

.macro SAD_xN_4xH n, h
    SAD_xN_4 \n, uabdl
.rept \h - 1
    SAD_xN_4 \n, uabal
.endr
.endm

.macro SAD_xN_8x2 n, f
    ld1             {v0.8h}, [x0], x7
    ld1             {v1.8h}, [x1], x5
    ld1             {v2.8h}, [x2], x5
    ld1             {v3.8h}, [x3], x5
    \f              v16.8h, v0.8h, v1.8h
    \f              v17.8h, v0.8h, v2.8h
    \f              v18.8h, v0.8h, v3.8h
.if \n == 4
    ld1             {v4.8h}, [x4], x5
    \f              v22.8h, v0.8h, v4.8h
.endif

    ld1             {v0.8h}, [x0], x7
    ld1             {v1.8h}, [x1], x5
    ld1             {v2.8h}, [x2], x5
    ld1             {v3.8h}, [x3], x5
    \f              v19.8h, v0.8h, v1.8h
    \f              v20.8h, v0.8h, v2.8h
    \f              v21.8h, v0.8h, v3.8h
.if \n == 4
    ld1             {v4.8h}, [x4], x5
    \f              v23.8h, v0.8h, v4.8h
.endif
.endm

.macro SAD_xN_8xH n, h
    SAD_xN_8x2 \n, uabd
.rept \h /2 - 1
    SAD_xN_8x2 \n, uaba
.endr
    uaddlp          v16.4s, v16.8h
    uadalp          v16.4s, v19.8h
    uaddlp          v17.4s, v17.8h
    uadalp          v17.4s, v20.8h
    uaddlp          v18.4s, v18.8h
    uadalp          v18.4s, v21.8h
.if \n == 4
    uaddlp          v19.4s, v22.8h
    uadalp          v19.4s, v23.8h
.endif
.endm

.macro SAD_xN_FUNC n, w, h
function PFX(sad_x\n\()_\w\()x\h\()_neon)
    // Make function arguments for n == 3 look like n == 4.
.if \n == 3
    mov             x6, x5
    mov             x5, x4
.endif

    // Stride is given in terms of pixel channel size, so double to get number of bytes.
    add             x5, x5, x5
    mov             x7, #(FENC_STRIDE << 1)

    SAD_xN_\w\()xH \n, \h

.if \n == 3
    addp            v0.4s, v16.4s, v17.4s
    addp            v1.4s, v18.4s, v18.4s
    addp            v0.4s, v0.4s, v1.4s
    str             d0, [x6]
    add             x6, x6, #8
    st1             {v0.s}[2], [x6]
.else
    addp            v16.4s, v16.4s, v17.4s
    addp            v18.4s, v18.4s, v19.4s
    addp            v16.4s, v16.4s, v18.4s
    str             q16, [x6]
.endif

    ret
endfunc
.endm

.macro SAD_xN_12 n, f
    ldr             q0, [x0]
    ldr             q1, [x1]
    ldr             q2, [x2]
    ldr             q3, [x3]
    ldr             d4, [x0, #16]
    ldr             d5, [x1, #16]
    ldr             d6, [x2, #16]
    ldr             d7, [x3, #16]
    \f              v16.8h, v0.8h, v1.8h
    \f              v18.8h, v0.8h, v2.8h
    \f              v20.8h, v0.8h, v3.8h
    \f              v17.8h, v4.8h, v5.8h
    \f              v19.8h, v4.8h, v6.8h
    \f              v21.8h, v4.8h, v7.8h
    add             x0, x0, x7
    add             x1, x1, x5
    add             x2, x2, x5
    add             x3, x3, x5
.if \n == 4
    ldr             q3, [x4]
    ldr             d7, [x4, #16]
    \f              v22.8h, v0.8h, v3.8h
    \f              v23.8h, v4.8h, v7.8h
    add             x4, x4, x5
.endif
.endm

.macro SAD_xN_16 n f
    ld1             {v0.8h-v1.8h}, [x0], x7
    ld1             {v2.8h-v3.8h}, [x1], x5
    \f              v16.8h, v0.8h, v2.8h
    \f              v17.8h, v1.8h, v3.8h
    ld1             {v4.8h-v5.8h}, [x2], x5
    \f              v18.8h, v0.8h, v4.8h
    \f              v19.8h, v1.8h, v5.8h
    ld1             {v6.8h-v7.8h}, [x3], x5
    \f              v20.8h, v0.8h, v6.8h
    \f              v21.8h, v1.8h, v7.8h
.if \n == 4
    ld1             {v6.8h-v7.8h}, [x4], x5
    \f              v22.8h, v0.8h, v6.8h
    \f              v23.8h, v1.8h, v7.8h
.endif
.endm

.macro SAD_xN_32 n f
    ld1             {v0.8h-v3.8h}, [x0], x7
    ld1             {v4.8h-v7.8h}, [x1], x5
    \f              v16.8h, v0.8h, v4.8h
    uaba            v16.8h, v1.8h, v5.8h
    \f              v17.8h, v2.8h, v6.8h
    uaba            v17.8h, v3.8h, v7.8h
    ld1             {v4.8h-v7.8h}, [x2], x5
    \f              v18.8h, v0.8h, v4.8h
    uaba            v18.8h, v1.8h, v5.8h
    \f              v19.8h, v2.8h, v6.8h
    uaba            v19.8h, v3.8h, v7.8h
    ld1             {v4.8h-v7.8h}, [x3], x5
    \f              v20.8h, v0.8h, v4.8h
    uaba            v20.8h, v1.8h, v5.8h
    \f              v21.8h, v2.8h, v6.8h
    uaba            v21.8h, v3.8h, v7.8h
.if \n == 4
    ld1             {v4.8h-v7.8h}, [x4], x5
    \f              v22.8h, v0.8h, v4.8h
    uaba            v22.8h, v1.8h, v5.8h
    \f              v23.8h, v2.8h, v6.8h
    uaba            v23.8h, v3.8h, v7.8h
.endif
.endm

.macro SAD_xN_FUNC_LOOP n, w, h end_type
function PFX(sad_x\n\()_\w\()x\h\()_neon)
    // Make function arguments for n == 3 look like n == 4.
.if \n == 3
    mov             x6, x5
    mov             x5, x4
.endif

    // Stride is given in terms of pixel channel size, so double to get number of bytes.
    add             x5, x5, x5
    mov             x7, #(FENC_STRIDE << 1)

    SAD_xN_\w \n, uabd

    mov             w8, #\h - 1
.Loop_x\n\()_\w\()x\h:
    sub             w8, w8, #1
    SAD_xN_\w \n, uaba
    cbnz            w8, .Loop_x\n\()_\w\()x\h

    uaddlp          v16.4s, v16.8h
    uadalp          v16.4s, v17.8h
    uaddlp          v18.4s, v18.8h
    uadalp          v18.4s, v19.8h
    uaddlp          v20.4s, v20.8h
    uadalp          v20.4s, v21.8h

.if \n == 3
    addp            v0.4s, v16.4s, v18.4s
    addp            v1.4s, v20.4s, v20.4s
    addp            v0.4s, v0.4s, v1.4s
    str             d0, [x6]
    add             x6, x6, #8
    st1             {v0.s}[2], [x6]
.else
    uaddlp          v22.4s, v22.8h
    uadalp          v22.4s, v23.8h
    addp            v16.4s, v16.4s, v18.4s
    addp            v20.4s, v20.4s, v22.4s
    addp            v16.4s, v16.4s, v20.4s
    str             q16, [x6]
.endif

    ret
endfunc
.endm

.macro SAD_xN_16_WIDEN n f
    ld1             {v0.8h-v1.8h}, [x0], x7
    ld1             {v2.8h-v3.8h}, [x1], x5
    uabd            v22.8h, v0.8h, v2.8h
    \f              v16.4s, v22.8h
    uabd            v23.8h, v1.8h, v3.8h
    \f              v17.4s, v23.8h
    ld1             {v4.8h-v5.8h}, [x2], x5
    uabd            v24.8h, v0.8h, v4.8h
    \f              v18.4s, v24.8h
    uabd            v25.8h, v1.8h, v5.8h
    \f              v19.4s, v25.8h
    ld1             {v6.8h-v7.8h}, [x3], x5
    uabd            v26.8h, v0.8h, v6.8h
    \f              v20.4s, v26.8h
    uabd            v27.8h, v1.8h, v7.8h
    \f              v21.4s, v27.8h
.if \n == 4
    ld1             {v2.8h-v3.8h}, [x4], x5
    uabd            v28.8h, v0.8h, v2.8h
    \f              v30.4s, v28.8h
    uabd            v29.8h, v1.8h, v3.8h
    \f              v31.4s, v29.8h
.endif
.endm

.macro SAD_xN_24_WIDEN n f
    ld1             {v0.8h-v2.8h}, [x0], x7
    ld1             {v3.8h-v5.8h}, [x1], x5
    uabd            v6.8h, v0.8h, v3.8h
    uaba            v6.8h, v1.8h, v4.8h
    \f              v16.4s, v6.8h
    uabd            v7.8h, v2.8h, v5.8h
    \f              v17.4s, v7.8h
    ld1             {v27.8h-v29.8h}, [x2], x5
    uabd            v22.8h, v0.8h, v27.8h
    uaba            v22.8h, v1.8h, v28.8h
    \f              v18.4s, v22.8h
    uabd            v23.8h, v2.8h, v29.8h
    \f              v19.4s, v23.8h
    ld1             {v3.8h-v5.8h}, [x3], x5
    uabd            v24.8h, v0.8h, v3.8h
    uaba            v24.8h, v1.8h, v4.8h
    \f              v20.4s, v24.8h
    uabd            v25.8h, v2.8h, v5.8h
    \f              v21.4s, v25.8h
.if \n == 4
    ld1             {v27.8h-v29.8h}, [x4], x5
    uabd            v22.8h, v0.8h, v27.8h
    uaba            v22.8h, v1.8h, v28.8h
    \f              v30.4s, v22.8h
    uabd            v23.8h, v2.8h, v29.8h
    \f              v31.4s, v23.8h
.endif
.endm

.macro SAD_xN_32_WIDEN n f
    ld1             {v0.8h-v3.8h}, [x0], x7
    ld1             {v4.8h-v7.8h}, [x1], x5
    uabd            v22.8h, v0.8h, v4.8h
    uaba            v22.8h, v1.8h, v5.8h
    \f              v16.4s, v22.8h
    uabd            v23.8h, v2.8h, v6.8h
    uaba            v23.8h, v3.8h, v7.8h
    \f              v17.4s, v23.8h

    ld1             {v4.8h-v7.8h}, [x2], x5
    uabd            v24.8h, v0.8h, v4.8h
    uaba            v24.8h, v1.8h, v5.8h
    \f              v18.4s, v24.8h
    uabd            v25.8h, v2.8h, v6.8h
    uaba            v25.8h, v3.8h, v7.8h
    \f              v19.4s, v25.8h

    ld1             {v4.8h-v7.8h}, [x3], x5
    uabd            v26.8h, v0.8h, v4.8h
    uaba            v26.8h, v1.8h, v5.8h
    \f              v20.4s, v26.8h
    uabd            v27.8h, v2.8h, v6.8h
    uaba            v27.8h, v3.8h, v7.8h
    \f              v21.4s, v27.8h

.if \n == 4
    ld1             {v4.8h-v7.8h}, [x4], x5
    uabd            v22.8h, v0.8h, v4.8h
    uaba            v22.8h, v1.8h, v5.8h
    \f              v30.4s, v22.8h
    uabd            v23.8h, v2.8h, v6.8h
    uaba            v23.8h, v3.8h, v7.8h
    \f              v31.4s, v23.8h
.endif
.endm

.macro SAD_xN_48_WIDEN n f
    ld1             {v0.8h-v3.8h}, [x0]
    ld1             {v26.8h-v29.8h}, [x1]
    uabd            v6.8h, v0.8h, v26.8h
    uaba            v6.8h, v1.8h, v27.8h
    \f              v16.4s, v6.8h
    uabd            v7.8h, v2.8h, v28.8h
    uaba            v7.8h, v3.8h, v29.8h
    \f              v17.4s, v7.8h
    ldp             q4, q5, [x0, #64]
    ldp             q26, q27, [x1, #64]
    uabd            v22.8h, v4.8h, v26.8h
    uaba            v22.8h, v5.8h, v27.8h
    uadalp          v16.4s, v22.8h

    ld1             {v26.8h-v29.8h}, [x2]
    uabd            v23.8h, v0.8h, v26.8h
    uaba            v23.8h, v1.8h, v27.8h
    \f              v18.4s, v23.8h
    uabd            v24.8h, v2.8h, v28.8h
    uaba            v24.8h, v3.8h, v29.8h
    \f              v19.4s, v24.8h
    ldp             q26, q27, [x2, #64]
    uabd            v25.8h, v4.8h, v26.8h
    uaba            v25.8h, v5.8h, v27.8h
    uadalp          v18.4s, v25.8h

    ld1             {v26.8h-v29.8h}, [x3]
    uabd            v6.8h, v0.8h, v26.8h
    uaba            v6.8h, v1.8h, v27.8h
    \f              v20.4s, v6.8h
    uabd            v7.8h, v2.8h, v28.8h
    uaba            v7.8h, v3.8h, v29.8h
    \f              v21.4s, v7.8h
    ldp             q26, q27, [x3, #64]
    uabd            v22.8h, v4.8h, v26.8h
    uaba            v22.8h, v5.8h, v27.8h
    uadalp          v20.4s, v22.8h

    add             x0, x0, x7
    add             x1, x1, x5
    add             x2, x2, x5
    add             x3, x3, x5

.if \n == 4
    ld1             {v26.8h-v29.8h}, [x4]
    uabd            v6.8h, v0.8h, v26.8h
    uaba            v6.8h, v1.8h, v27.8h
    \f              v30.4s, v6.8h
    uabd            v7.8h, v2.8h, v28.8h
    uaba            v7.8h, v3.8h, v29.8h
    \f              v31.4s, v7.8h
    ldp             q26, q27, [x4, #64]
    uabd            v22.8h, v4.8h, v26.8h
    uaba            v22.8h, v5.8h, v27.8h
    uadalp          v30.4s, v22.8h
    add             x4, x4, x5
.endif
.endm

.macro SAD_xN_64_WIDEN n f
    ld1             {v0.8h-v3.8h}, [x0]
    ld1             {v26.8h-v29.8h}, [x1]
    uabd            v22.8h, v0.8h, v26.8h
    uaba            v22.8h, v1.8h, v27.8h
    \f              v16.4s, v22.8h
    uabd            v23.8h, v2.8h, v28.8h
    uaba            v23.8h, v3.8h, v29.8h
    \f              v17.4s, v23.8h
    ldp             q4, q5, [x0, #64]
    ldp             q6, q7, [x0, #96]
    ldp             q26, q27, [x1, #64]
    ldp             q28, q29, [x1, #96]
    uabd            v24.8h, v4.8h, v26.8h
    uaba            v24.8h, v5.8h, v27.8h
    uadalp          v16.4s, v24.8h
    uabd            v25.8h, v6.8h, v28.8h
    uaba            v25.8h, v7.8h, v29.8h
    uadalp          v17.4s, v25.8h

    ld1             {v26.8h-v29.8h}, [x2]
    uabd            v22.8h, v0.8h, v26.8h
    uaba            v22.8h, v1.8h, v27.8h
    \f              v18.4s, v22.8h
    uabd            v23.8h, v2.8h, v28.8h
    uaba            v23.8h, v3.8h, v29.8h
    \f              v19.4s, v23.8h
    ldp             q26, q27, [x2, #64]
    ldp             q28, q29, [x2, #96]
    uabd            v24.8h, v4.8h, v26.8h
    uaba            v24.8h, v5.8h, v27.8h
    uadalp          v18.4s, v24.8h
    uabd            v25.8h, v6.8h, v28.8h
    uaba            v25.8h, v7.8h, v29.8h
    uadalp          v19.4s, v25.8h

    ld1             {v26.8h-v29.8h}, [x3]
    uabd            v22.8h, v0.8h, v26.8h
    uaba            v22.8h, v1.8h, v27.8h
    \f              v20.4s, v22.8h
    uabd            v23.8h, v2.8h, v28.8h
    uaba            v23.8h, v3.8h, v29.8h
    \f              v21.4s, v23.8h
    ldp             q26, q27, [x3, #64]
    ldp             q28, q29, [x3, #96]
    uabd            v24.8h, v4.8h, v26.8h
    uaba            v24.8h, v5.8h, v27.8h
    uadalp          v20.4s, v24.8h
    uabd            v25.8h, v6.8h, v28.8h
    uaba            v25.8h, v7.8h, v29.8h
    uadalp          v21.4s, v25.8h

    add             x0, x0, x7
    add             x1, x1, x5
    add             x2, x2, x5
    add             x3, x3, x5

.if \n == 4
    ld1             {v26.8h-v29.8h}, [x4]
    uabd            v22.8h, v0.8h, v26.8h
    uaba            v22.8h, v1.8h, v27.8h
    \f              v30.4s, v22.8h
    uabd            v23.8h, v2.8h, v28.8h
    uaba            v23.8h, v3.8h, v29.8h
    \f              v31.4s, v23.8h
    ldp             q26, q27, [x4, #64]
    ldp             q28, q29, [x4, #96]
    uabd            v24.8h, v4.8h, v26.8h
    uaba            v24.8h, v5.8h, v27.8h
    uadalp          v30.4s, v24.8h
    uabd            v25.8h, v6.8h, v28.8h
    uaba            v25.8h, v7.8h, v29.8h
    uadalp          v31.4s, v25.8h
    add             x4, x4, x5
.endif
.endm

.macro SAD_xN_FUNC_LOOP_LARGE n, w, h
function PFX(sad_x\n\()_\w\()x\h\()_neon)
    // Make function arguments for n == 3 look like n == 4.
.if \n == 3
    mov             x6, x5
    mov             x5, x4
.endif

    // Stride is given in terms of pixel channel size, so double to get number of bytes.
    add             x5, x5, x5
    mov             x7, #(FENC_STRIDE << 1)

    SAD_xN_\w\()_WIDEN \n, uaddlp
    SAD_xN_\w\()_WIDEN \n, uadalp

    mov             w8, #(\h - 2)/2
.Loop_x\n\()_\w\()x\h:
    sub             w8, w8, #1
.rept 2
    SAD_xN_\w\()_WIDEN \n, uadalp
.endr
    cbnz            w8, .Loop_x\n\()_\w\()x\h

    add             v16.4s, v16.4s, v17.4s
    add             v17.4s, v18.4s, v19.4s
    add             v18.4s, v20.4s, v21.4s
.if \n == 3
    addp            v0.4s, v16.4s, v17.4s
    addp            v1.4s, v18.4s, v18.4s
    addp            v0.4s, v0.4s, v1.4s
    str             d0, [x6]
    add             x6, x6, #8
    st1             {v0.s}[2], [x6]
.else
    add             v19.4s, v30.4s, v31.4s
    addp            v16.4s, v16.4s, v17.4s
    addp            v18.4s, v18.4s, v19.4s
    addp            v16.4s, v16.4s, v18.4s
    str             q16, [x6]
.endif

    ret
endfunc
.endm

SAD_xN_FUNC  3, 4, 4
SAD_xN_FUNC  3, 4, 8
SAD_xN_FUNC  3, 4, 16
SAD_xN_FUNC  3, 8, 4
SAD_xN_FUNC  3, 8, 8
SAD_xN_FUNC  3, 8, 16
SAD_xN_FUNC  3, 8, 32
SAD_xN_FUNC_LOOP  3, 12, 16
SAD_xN_FUNC_LOOP  3, 16, 4
SAD_xN_FUNC_LOOP  3, 16, 8
SAD_xN_FUNC_LOOP  3, 16, 12
SAD_xN_FUNC_LOOP  3, 16, 16
SAD_xN_FUNC_LOOP  3, 32, 8
SAD_xN_FUNC_LOOP_LARGE  3, 16, 32
SAD_xN_FUNC_LOOP_LARGE  3, 16, 64
SAD_xN_FUNC_LOOP_LARGE  3, 24, 32
SAD_xN_FUNC_LOOP_LARGE  3, 32, 16
SAD_xN_FUNC_LOOP_LARGE  3, 32, 24
SAD_xN_FUNC_LOOP_LARGE  3, 32, 32
SAD_xN_FUNC_LOOP_LARGE  3, 32, 64
SAD_xN_FUNC_LOOP_LARGE  3, 48, 64
SAD_xN_FUNC_LOOP_LARGE  3, 64, 16
SAD_xN_FUNC_LOOP_LARGE  3, 64, 32
SAD_xN_FUNC_LOOP_LARGE  3, 64, 48
SAD_xN_FUNC_LOOP_LARGE  3, 64, 64

SAD_xN_FUNC  4, 4, 4
SAD_xN_FUNC  4, 4, 8
SAD_xN_FUNC  4, 4, 16
SAD_xN_FUNC  4, 8, 4
SAD_xN_FUNC  4, 8, 8
SAD_xN_FUNC  4, 8, 16
SAD_xN_FUNC  4, 8, 32
SAD_xN_FUNC_LOOP  4, 12, 16
SAD_xN_FUNC_LOOP  4, 16, 4
SAD_xN_FUNC_LOOP  4, 16, 8
SAD_xN_FUNC_LOOP  4, 16, 12
SAD_xN_FUNC_LOOP  4, 16, 16
SAD_xN_FUNC_LOOP  4, 32, 8
SAD_xN_FUNC_LOOP_LARGE  4, 16, 32
SAD_xN_FUNC_LOOP_LARGE  4, 16, 64
SAD_xN_FUNC_LOOP_LARGE  4, 24, 32
SAD_xN_FUNC_LOOP_LARGE  4, 32, 16
SAD_xN_FUNC_LOOP_LARGE  4, 32, 24
SAD_xN_FUNC_LOOP_LARGE  4, 32, 32
SAD_xN_FUNC_LOOP_LARGE  4, 32, 64
SAD_xN_FUNC_LOOP_LARGE  4, 48, 64
SAD_xN_FUNC_LOOP_LARGE  4, 64, 16
SAD_xN_FUNC_LOOP_LARGE  4, 64, 32
SAD_xN_FUNC_LOOP_LARGE  4, 64, 48
SAD_xN_FUNC_LOOP_LARGE  4, 64, 64

#endif // !HIGH_BIT_DEPTH
