// Avisynth C Interface Version 0.20
// Copyright 2003 Kevin Atkinson

// Copyright 2020 AviSynth+ project
// Actual C Interface version follows the global Avisynth+ IF version numbers.

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
// MA 02110-1301 USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// As a special exception, I give you permission to link to the
// Avisynth C interface with independent modules that communicate with
// the Avisynth C interface solely through the interfaces defined in
// avisynth_c.h, regardless of the license terms of these independent
// modules, and to copy and distribute the resulting combined work
// under terms of your choice, provided that every copy of the
// combined work is accompanied by a complete copy of the source code
// of the Avisynth C interface and Avisynth itself (with the version
// used to produce the combined work), being distributed under the
// terms of the GNU General Public License plus this exception.  An
// independent module is a module which is not derived from or based
// on Avisynth C Interface, such as 3rd-party filters, import and
// export plugins, or graphical user interfaces.

// NOTE: this is a partial update of the Avisynth C interface to recognize
// new color spaces and interface elements added in Avisynth 2.60 and AviSynth+.
// This interface is not 100% Avisynth+ CPP interface equivalent.

// 170103: added new CPU constants (FMA4, AVX512xx)
// 171102: define SIZETMOD. do not use yet, experimental. Offsets are size_t instead of int. Affects x64.
// 171106: avs_get_row_size calls into avs_get_row_size_p, instead of direct field access
// 171106: avs_get_height calls into avs_get_row_size_p, instead of direct field access
// 180524: AVSC_EXPORT to dllexport in capi.h for avisynth_c_plugin_init
// 180524: avs_is_same_colorspace VideoInfo parameters to const
// 181230: Readability: functions regrouped to mix less AVSC_API and AVSC_INLINE, put together Avisynth+ specific stuff
// 181230: use #ifndef AVSC_NO_DECLSPEC for AVSC_INLINE functions which are calling API functions
// 181230: comments on avs_load_library (helper for loading API entries dynamically into a struct using AVSC_NO_DECLSPEC define)
// 181230: define alias AVS_FRAME_ALIGN as FRAME_ALIGN
// 181230: remove unused form of avs_get_rowsize and avs_get_height (kept earlier for reference)
// 190104: avs_load_library: smart fallback mechanism for Avisynth+ specific functions:
//         if they are not loadable, they will work in a classic Avisynth compatible mode
//         Example#1: e.g. avs_is_444 will call the existing avs_is_yv24 instead
//         Example#2: avs_bits_per_component will return 8 for all colorspaces (Classic Avisynth supports only 8 bits/pixel)
//         Thus the Avisynth+ specific API functions are safely callable even when connected to classic Avisynth DLL
// 2002xx  non-Windows friendly additions
// 200305  avs_vsprintf parameter type change: (void *) to va_list
// 200330: (remove test SIZETMOD define for clarity)
// 200513: user must use explicite #define AVS26_FALLBACK_SIMULATION for having fallback helpers in dynamic loaded library section
// 200513: Follow AviSynth+ V8 interface additions
//         AVS_VideoFrame struct extended with placeholder for frame property pointer
//         avs_subframe_planar_a
//         avs_copy_frame_props
//         avs_get_frame_props_ro, avs_get_frame_props_rw
//         avs_prop_num_keys, avs_prop_get_key, avs_prop_num_elements, avs_prop_get_type, avs_prop_get_data_size
//         avs_prop_get_int, avs_prop_get_float, avs_prop_get_data, avs_prop_get_clip, avs_prop_get_frame, avs_prop_get_int_array, avs_prop_get_float_array
//         avs_prop_set_int, avs_prop_set_float, avs_prop_set_data, avs_prop_set_clip, avs_prop_set_frame, avs_prop_set_int_array, avs_prop_set_float_array
//         avs_prop_delete_key, avs_clear_map
//         avs_new_video_frame_p, avs_new_video_frame_p_a
//         avs_get_env_property (internal system properties), AVS_AEP_xxx (AvsEnvProperty) enums
//         avs_get_var_try, avs_get_var_bool, avs_get_var_int, avs_get_var_double, avs_get_var_string, avs_get_var_long
//         avs_pool_allocate, avs_pool_free
// 2021:   Follow AviSynth+ V9 interface additions
//         avs_is_property_writable, avs_make_property_writable
//         Add enum AVISYNTHPLUS_INTERFACE_BUGFIX_VERSION (AVISYNTH_INTERFACE_VERSION still exists)
//         Add enum AVS_AEP_HOST_SYSTEM_ENDIANNESS to system property request types (avs_get_env_property)
//         Add enums AVS_AEP_INTERFACE_VERSION and AVS_AEP_INTERFACE_BUGFIX for direct interface version system property request types (avs_get_env_property)
//         Bugfix 9.1: fix avs_prop_get_data
// 2023:   Follow AviSynth+ V10 interface additions
//         Add enum AVS_DEFAULT_PLANE (as 0) to plane constants
//         prop_src argument now const in avs_new_video_frame_p and avs_new_video_frame_p_a (no change in use)
//         Add pixel_type to struct AVS_VideoFrame
//         Add avs_video_frame_get_pixel_type and avs_video_frame_amend_pixel_type for getting and setting AVS_VideoFrame pixel_type
//         Additional AviSynth+ V10 interface additions:
//         Add enum AVS_SPEAKER_xxx, AVS_IT_SPEAKER_xxx
//         audio channel mask support avs_is_channel_mask_known, avs_set_channel_mask, avs_get_channel_mask

#ifndef __AVISYNTH_C__
#define __AVISYNTH_C__

#include "avs/config.h"
#include "avs/capi.h"
#include "avs/types.h"

#define AVS_FRAME_ALIGN FRAME_ALIGN
/////////////////////////////////////////////////////////////////////
//
// Constants
//

#ifndef __AVISYNTH_10_H__
enum {
  AVISYNTH_INTERFACE_CLASSIC_VERSION = 6,
  AVISYNTH_INTERFACE_VERSION = 10,
  AVISYNTHPLUS_INTERFACE_BUGFIX_VERSION = 0 // reset to zero whenever the normal interface version bumps
};
#endif

enum {
  AVS_SAMPLE_INT8  = 1 << 0,
  AVS_SAMPLE_INT16 = 1 << 1,
  AVS_SAMPLE_INT24 = 1 << 2,
  AVS_SAMPLE_INT32 = 1 << 3,
  AVS_SAMPLE_FLOAT = 1 << 4
};

enum {
  AVS_DEFAULT_PLANE = 0,
  AVS_PLANAR_Y = 1 << 0,
  AVS_PLANAR_U = 1 << 1,
  AVS_PLANAR_V = 1 << 2,
  AVS_PLANAR_ALIGNED = 1 << 3,
  AVS_PLANAR_Y_ALIGNED = AVS_PLANAR_Y | AVS_PLANAR_ALIGNED,
  AVS_PLANAR_U_ALIGNED = AVS_PLANAR_U | AVS_PLANAR_ALIGNED,
  AVS_PLANAR_V_ALIGNED = AVS_PLANAR_V | AVS_PLANAR_ALIGNED,
  AVS_PLANAR_A = 1 << 4,
  AVS_PLANAR_R = 1 << 5,
  AVS_PLANAR_G = 1 << 6,
  AVS_PLANAR_B = 1 << 7,
  AVS_PLANAR_A_ALIGNED = AVS_PLANAR_A | AVS_PLANAR_ALIGNED,
  AVS_PLANAR_R_ALIGNED = AVS_PLANAR_R | AVS_PLANAR_ALIGNED,
  AVS_PLANAR_G_ALIGNED = AVS_PLANAR_G | AVS_PLANAR_ALIGNED,
  AVS_PLANAR_B_ALIGNED = AVS_PLANAR_B | AVS_PLANAR_ALIGNED
};

// Colorspace properties.
enum {
  AVS_CS_YUVA = 1 << 27,
  AVS_CS_BGR = 1 << 28,
  AVS_CS_YUV = 1 << 29,
  AVS_CS_INTERLEAVED = 1 << 30,
  AVS_CS_PLANAR = 1 << 31,

  AVS_CS_SHIFT_SUB_WIDTH = 0,
  AVS_CS_SHIFT_SUB_HEIGHT = 8,
  AVS_CS_SHIFT_SAMPLE_BITS = 16,

  AVS_CS_SUB_WIDTH_MASK = 7 << AVS_CS_SHIFT_SUB_WIDTH,
  AVS_CS_SUB_WIDTH_1 = 3 << AVS_CS_SHIFT_SUB_WIDTH, // YV24
  AVS_CS_SUB_WIDTH_2 = 0 << AVS_CS_SHIFT_SUB_WIDTH, // YV12, I420, YV16
  AVS_CS_SUB_WIDTH_4 = 1 << AVS_CS_SHIFT_SUB_WIDTH, // YUV9, YV411

  AVS_CS_VPLANEFIRST = 1 << 3, // YV12, YV16, YV24, YV411, YUV9
  AVS_CS_UPLANEFIRST = 1 << 4, // I420

  AVS_CS_SUB_HEIGHT_MASK = 7 << AVS_CS_SHIFT_SUB_HEIGHT,
  AVS_CS_SUB_HEIGHT_1 = 3 << AVS_CS_SHIFT_SUB_HEIGHT, // YV16, YV24, YV411
  AVS_CS_SUB_HEIGHT_2 = 0 << AVS_CS_SHIFT_SUB_HEIGHT, // YV12, I420
  AVS_CS_SUB_HEIGHT_4 = 1 << AVS_CS_SHIFT_SUB_HEIGHT, // YUV9

  AVS_CS_SAMPLE_BITS_MASK = 7 << AVS_CS_SHIFT_SAMPLE_BITS,
  AVS_CS_SAMPLE_BITS_8 = 0 << AVS_CS_SHIFT_SAMPLE_BITS,
  AVS_CS_SAMPLE_BITS_10 = 5 << AVS_CS_SHIFT_SAMPLE_BITS,
  AVS_CS_SAMPLE_BITS_12 = 6 << AVS_CS_SHIFT_SAMPLE_BITS,
  AVS_CS_SAMPLE_BITS_14 = 7 << AVS_CS_SHIFT_SAMPLE_BITS,
  AVS_CS_SAMPLE_BITS_16 = 1 << AVS_CS_SHIFT_SAMPLE_BITS,
  AVS_CS_SAMPLE_BITS_32 = 2 << AVS_CS_SHIFT_SAMPLE_BITS,

  AVS_CS_PLANAR_MASK = AVS_CS_PLANAR | AVS_CS_INTERLEAVED | AVS_CS_YUV | AVS_CS_BGR | AVS_CS_YUVA
                       | AVS_CS_SAMPLE_BITS_MASK | AVS_CS_SUB_WIDTH_MASK | AVS_CS_SUB_HEIGHT_MASK,
  AVS_CS_PLANAR_FILTER = ~(AVS_CS_VPLANEFIRST | AVS_CS_UPLANEFIRST),

  AVS_CS_RGB_TYPE  = 1 << 0,
  AVS_CS_RGBA_TYPE = 1 << 1,

  AVS_CS_GENERIC_YUV444  = AVS_CS_PLANAR | AVS_CS_YUV | AVS_CS_VPLANEFIRST | AVS_CS_SUB_WIDTH_1 | AVS_CS_SUB_HEIGHT_1,  // 4:4:4 planar
  AVS_CS_GENERIC_YUV422  = AVS_CS_PLANAR | AVS_CS_YUV | AVS_CS_VPLANEFIRST | AVS_CS_SUB_WIDTH_2 | AVS_CS_SUB_HEIGHT_1,  // 4:2:2 planar
  AVS_CS_GENERIC_YUV420  = AVS_CS_PLANAR | AVS_CS_YUV | AVS_CS_VPLANEFIRST | AVS_CS_SUB_WIDTH_2 | AVS_CS_SUB_HEIGHT_2,  // 4:2:0 planar
  AVS_CS_GENERIC_Y       = AVS_CS_PLANAR | AVS_CS_INTERLEAVED | AVS_CS_YUV,                                             // Y only (4:0:0)
  AVS_CS_GENERIC_RGBP    = AVS_CS_PLANAR | AVS_CS_BGR | AVS_CS_RGB_TYPE,                                                // planar RGB
  AVS_CS_GENERIC_RGBAP   = AVS_CS_PLANAR | AVS_CS_BGR | AVS_CS_RGBA_TYPE,                                               // planar RGBA
  AVS_CS_GENERIC_YUVA444 = AVS_CS_PLANAR | AVS_CS_YUVA | AVS_CS_VPLANEFIRST | AVS_CS_SUB_WIDTH_1 | AVS_CS_SUB_HEIGHT_1, // 4:4:4:A planar
  AVS_CS_GENERIC_YUVA422 = AVS_CS_PLANAR | AVS_CS_YUVA | AVS_CS_VPLANEFIRST | AVS_CS_SUB_WIDTH_2 | AVS_CS_SUB_HEIGHT_1, // 4:2:2:A planar
  AVS_CS_GENERIC_YUVA420 = AVS_CS_PLANAR | AVS_CS_YUVA | AVS_CS_VPLANEFIRST | AVS_CS_SUB_WIDTH_2 | AVS_CS_SUB_HEIGHT_2  // 4:2:0:A planar
};

// Specific color formats
enum {
  AVS_CS_UNKNOWN = 0,
  AVS_CS_BGR24 = AVS_CS_RGB_TYPE  | AVS_CS_BGR | AVS_CS_INTERLEAVED,
  AVS_CS_BGR32 = AVS_CS_RGBA_TYPE | AVS_CS_BGR | AVS_CS_INTERLEAVED,
  AVS_CS_YUY2 = 1 << 2 | AVS_CS_YUV | AVS_CS_INTERLEAVED,
  // AVS_CS_YV12 = 1 << 3   Reserved
  // AVS_CS_I420 = 1 << 4   Reserved
  AVS_CS_RAW32 = 1 << 5 | AVS_CS_INTERLEAVED,

  AVS_CS_YV24  = AVS_CS_GENERIC_YUV444 | AVS_CS_SAMPLE_BITS_8,  // YUV 4:4:4 planar
  AVS_CS_YV16  = AVS_CS_GENERIC_YUV422 | AVS_CS_SAMPLE_BITS_8,  // YUV 4:2:2 planar
  AVS_CS_YV12  = AVS_CS_GENERIC_YUV420 | AVS_CS_SAMPLE_BITS_8,  // YUV 4:2:0 planar
  AVS_CS_I420  = AVS_CS_PLANAR | AVS_CS_YUV | AVS_CS_SAMPLE_BITS_8 | AVS_CS_UPLANEFIRST | AVS_CS_SUB_WIDTH_2 | AVS_CS_SUB_HEIGHT_2,  // YUV 4:2:0 planar
  AVS_CS_IYUV  = AVS_CS_I420,
  AVS_CS_YV411 = AVS_CS_PLANAR | AVS_CS_YUV | AVS_CS_SAMPLE_BITS_8 | AVS_CS_VPLANEFIRST | AVS_CS_SUB_WIDTH_4 | AVS_CS_SUB_HEIGHT_1,  // YUV 4:1:1 planar
  AVS_CS_YUV9  = AVS_CS_PLANAR | AVS_CS_YUV | AVS_CS_SAMPLE_BITS_8 | AVS_CS_VPLANEFIRST | AVS_CS_SUB_WIDTH_4 | AVS_CS_SUB_HEIGHT_4,  // YUV 4:1:0 planar
  AVS_CS_Y8    = AVS_CS_GENERIC_Y | AVS_CS_SAMPLE_BITS_8,       // Y   4:0:0 planar

  //-------------------------
  // AVS16: new planar constants go live! Experimental PF 160613
  // 10-12-14-16 bit + planar RGB + BGR48/64 160725
  AVS_CS_YUV444P10 = AVS_CS_GENERIC_YUV444 | AVS_CS_SAMPLE_BITS_10, // YUV 4:4:4 10bit samples
  AVS_CS_YUV422P10 = AVS_CS_GENERIC_YUV422 | AVS_CS_SAMPLE_BITS_10, // YUV 4:2:2 10bit samples
  AVS_CS_YUV420P10 = AVS_CS_GENERIC_YUV420 | AVS_CS_SAMPLE_BITS_10, // YUV 4:2:0 10bit samples
  AVS_CS_Y10       = AVS_CS_GENERIC_Y | AVS_CS_SAMPLE_BITS_10,      // Y   4:0:0 10bit samples

  AVS_CS_YUV444P12 = AVS_CS_GENERIC_YUV444 | AVS_CS_SAMPLE_BITS_12, // YUV 4:4:4 12bit samples
  AVS_CS_YUV422P12 = AVS_CS_GENERIC_YUV422 | AVS_CS_SAMPLE_BITS_12, // YUV 4:2:2 12bit samples
  AVS_CS_YUV420P12 = AVS_CS_GENERIC_YUV420 | AVS_CS_SAMPLE_BITS_12, // YUV 4:2:0 12bit samples
  AVS_CS_Y12       = AVS_CS_GENERIC_Y | AVS_CS_SAMPLE_BITS_12,      // Y   4:0:0 12bit samples

  AVS_CS_YUV444P14 = AVS_CS_GENERIC_YUV444 | AVS_CS_SAMPLE_BITS_14, // YUV 4:4:4 14bit samples
  AVS_CS_YUV422P14 = AVS_CS_GENERIC_YUV422 | AVS_CS_SAMPLE_BITS_14, // YUV 4:2:2 14bit samples
  AVS_CS_YUV420P14 = AVS_CS_GENERIC_YUV420 | AVS_CS_SAMPLE_BITS_14, // YUV 4:2:0 14bit samples
  AVS_CS_Y14       = AVS_CS_GENERIC_Y | AVS_CS_SAMPLE_BITS_14,      // Y   4:0:0 14bit samples

  AVS_CS_YUV444P16 = AVS_CS_GENERIC_YUV444 | AVS_CS_SAMPLE_BITS_16, // YUV 4:4:4 16bit samples
  AVS_CS_YUV422P16 = AVS_CS_GENERIC_YUV422 | AVS_CS_SAMPLE_BITS_16, // YUV 4:2:2 16bit samples
  AVS_CS_YUV420P16 = AVS_CS_GENERIC_YUV420 | AVS_CS_SAMPLE_BITS_16, // YUV 4:2:0 16bit samples
  AVS_CS_Y16       = AVS_CS_GENERIC_Y | AVS_CS_SAMPLE_BITS_16,      // Y   4:0:0 16bit samples

  // 32 bit samples (float)
  AVS_CS_YUV444PS = AVS_CS_GENERIC_YUV444 | AVS_CS_SAMPLE_BITS_32,  // YUV 4:4:4 32bit samples
  AVS_CS_YUV422PS = AVS_CS_GENERIC_YUV422 | AVS_CS_SAMPLE_BITS_32,  // YUV 4:2:2 32bit samples
  AVS_CS_YUV420PS = AVS_CS_GENERIC_YUV420 | AVS_CS_SAMPLE_BITS_32,  // YUV 4:2:0 32bit samples
  AVS_CS_Y32      = AVS_CS_GENERIC_Y | AVS_CS_SAMPLE_BITS_32,       // Y   4:0:0 32bit samples

  // RGB packed
  AVS_CS_BGR48 = AVS_CS_RGB_TYPE | AVS_CS_BGR | AVS_CS_INTERLEAVED | AVS_CS_SAMPLE_BITS_16,    // BGR 3x16 bit
  AVS_CS_BGR64 = AVS_CS_RGBA_TYPE | AVS_CS_BGR | AVS_CS_INTERLEAVED | AVS_CS_SAMPLE_BITS_16,    // BGR 4x16 bit
  // no packed 32 bit (float) support for these legacy types

  // RGB planar
  AVS_CS_RGBP   = AVS_CS_GENERIC_RGBP | AVS_CS_SAMPLE_BITS_8,  // Planar RGB 8 bit samples
  AVS_CS_RGBP10 = AVS_CS_GENERIC_RGBP | AVS_CS_SAMPLE_BITS_10, // Planar RGB 10bit samples
  AVS_CS_RGBP12 = AVS_CS_GENERIC_RGBP | AVS_CS_SAMPLE_BITS_12, // Planar RGB 12bit samples
  AVS_CS_RGBP14 = AVS_CS_GENERIC_RGBP | AVS_CS_SAMPLE_BITS_14, // Planar RGB 14bit samples
  AVS_CS_RGBP16 = AVS_CS_GENERIC_RGBP | AVS_CS_SAMPLE_BITS_16, // Planar RGB 16bit samples
  AVS_CS_RGBPS  = AVS_CS_GENERIC_RGBP | AVS_CS_SAMPLE_BITS_32, // Planar RGB 32bit samples

  // RGBA planar
  AVS_CS_RGBAP   = AVS_CS_GENERIC_RGBAP | AVS_CS_SAMPLE_BITS_8,  // Planar RGBA 8 bit samples
  AVS_CS_RGBAP10 = AVS_CS_GENERIC_RGBAP | AVS_CS_SAMPLE_BITS_10, // Planar RGBA 10bit samples
  AVS_CS_RGBAP12 = AVS_CS_GENERIC_RGBAP | AVS_CS_SAMPLE_BITS_12, // Planar RGBA 12bit samples
  AVS_CS_RGBAP14 = AVS_CS_GENERIC_RGBAP | AVS_CS_SAMPLE_BITS_14, // Planar RGBA 14bit samples
  AVS_CS_RGBAP16 = AVS_CS_GENERIC_RGBAP | AVS_CS_SAMPLE_BITS_16, // Planar RGBA 16bit samples
  AVS_CS_RGBAPS  = AVS_CS_GENERIC_RGBAP | AVS_CS_SAMPLE_BITS_32, // Planar RGBA 32bit samples

  // Planar YUVA
  AVS_CS_YUVA444    = AVS_CS_GENERIC_YUVA444 | AVS_CS_SAMPLE_BITS_8,  // YUVA 4:4:4 8bit samples
  AVS_CS_YUVA422    = AVS_CS_GENERIC_YUVA422 | AVS_CS_SAMPLE_BITS_8,  // YUVA 4:2:2 8bit samples
  AVS_CS_YUVA420    = AVS_CS_GENERIC_YUVA420 | AVS_CS_SAMPLE_BITS_8,  // YUVA 4:2:0 8bit samples

  AVS_CS_YUVA444P10 = AVS_CS_GENERIC_YUVA444 | AVS_CS_SAMPLE_BITS_10, // YUVA 4:4:4 10bit samples
  AVS_CS_YUVA422P10 = AVS_CS_GENERIC_YUVA422 | AVS_CS_SAMPLE_BITS_10, // YUVA 4:2:2 10bit samples
  AVS_CS_YUVA420P10 = AVS_CS_GENERIC_YUVA420 | AVS_CS_SAMPLE_BITS_10, // YUVA 4:2:0 10bit samples

  AVS_CS_YUVA444P12 = AVS_CS_GENERIC_YUVA444 | AVS_CS_SAMPLE_BITS_12, // YUVA 4:4:4 12bit samples
  AVS_CS_YUVA422P12 = AVS_CS_GENERIC_YUVA422 | AVS_CS_SAMPLE_BITS_12, // YUVA 4:2:2 12bit samples
  AVS_CS_YUVA420P12 = AVS_CS_GENERIC_YUVA420 | AVS_CS_SAMPLE_BITS_12, // YUVA 4:2:0 12bit samples

  AVS_CS_YUVA444P14 = AVS_CS_GENERIC_YUVA444 | AVS_CS_SAMPLE_BITS_14, // YUVA 4:4:4 14bit samples
  AVS_CS_YUVA422P14 = AVS_CS_GENERIC_YUVA422 | AVS_CS_SAMPLE_BITS_14, // YUVA 4:2:2 14bit samples
  AVS_CS_YUVA420P14 = AVS_CS_GENERIC_YUVA420 | AVS_CS_SAMPLE_BITS_14, // YUVA 4:2:0 14bit samples

  AVS_CS_YUVA444P16 = AVS_CS_GENERIC_YUVA444 | AVS_CS_SAMPLE_BITS_16, // YUVA 4:4:4 16bit samples
  AVS_CS_YUVA422P16 = AVS_CS_GENERIC_YUVA422 | AVS_CS_SAMPLE_BITS_16, // YUVA 4:2:2 16bit samples
  AVS_CS_YUVA420P16 = AVS_CS_GENERIC_YUVA420 | AVS_CS_SAMPLE_BITS_16, // YUVA 4:2:0 16bit samples

  AVS_CS_YUVA444PS  = AVS_CS_GENERIC_YUVA444 | AVS_CS_SAMPLE_BITS_32,  // YUVA 4:4:4 32bit samples
  AVS_CS_YUVA422PS  = AVS_CS_GENERIC_YUVA422 | AVS_CS_SAMPLE_BITS_32,  // YUVA 4:2:2 32bit samples
  AVS_CS_YUVA420PS  = AVS_CS_GENERIC_YUVA420 | AVS_CS_SAMPLE_BITS_32,  // YUVA 4:2:0 32bit samples
};

// AvsChannelMask enum: Unshifted channel mask constants like in WAVEFORMATEXTENSIBLE
// in AvsImageTypeFlags they are shifted by 4 bits
enum {
  AVS_MASK_SPEAKER_FRONT_LEFT = 0x1,
  AVS_MASK_SPEAKER_FRONT_RIGHT = 0x2,
  AVS_MASK_SPEAKER_FRONT_CENTER = 0x4,
  AVS_MASK_SPEAKER_LOW_FREQUENCY = 0x8,
  AVS_MASK_SPEAKER_BACK_LEFT = 0x10,
  AVS_MASK_SPEAKER_BACK_RIGHT = 0x20,
  AVS_MASK_SPEAKER_FRONT_LEFT_OF_CENTER = 0x40,
  AVS_MASK_SPEAKER_FRONT_RIGHT_OF_CENTER = 0x80,
  AVS_MASK_SPEAKER_BACK_CENTER = 0x100,
  AVS_MASK_SPEAKER_SIDE_LEFT = 0x200,
  AVS_MASK_SPEAKER_SIDE_RIGHT = 0x400,
  AVS_MASK_SPEAKER_TOP_CENTER = 0x800,
  AVS_MASK_SPEAKER_TOP_FRONT_LEFT = 0x1000,
  AVS_MASK_SPEAKER_TOP_FRONT_CENTER = 0x2000,
  AVS_MASK_SPEAKER_TOP_FRONT_RIGHT = 0x4000,
  AVS_MASK_SPEAKER_TOP_BACK_LEFT = 0x8000,
  AVS_MASK_SPEAKER_TOP_BACK_CENTER = 0x10000,
  AVS_MASK_SPEAKER_TOP_BACK_RIGHT = 0x20000,
  // Bit mask locations used up for the above positions
  AVS_MASK_SPEAKER_DEFINED = 0x0003FFFF,
  // Bit mask locations reserved for future use
  AVS_MASK_SPEAKER_RESERVED = 0x7FFC0000,
  // Used to specify that any possible permutation of speaker configurations
  // Due to lack of available bits this one is put differently into image_type
  AVS_MASK_SPEAKER_ALL = 0x80000000
};

// AvsImageTypeFlags
enum {
  AVS_IT_BFF = 1 << 0,
  AVS_IT_TFF = 1 << 1,
  AVS_IT_FIELDBASED = 1 << 2,

  // Audio channel mask support
  AVS_IT_HAS_CHANNELMASK = 1 << 3,
  // shifted by 4 bits compared to WAVEFORMATEXTENSIBLE dwChannelMask
  // otherwise same as AvsChannelMask
  AVS_IT_SPEAKER_FRONT_LEFT = 0x1 << 4,
  AVS_IT_SPEAKER_FRONT_RIGHT = 0x2 << 4,
  AVS_IT_SPEAKER_FRONT_CENTER = 0x4 << 4,
  AVS_IT_SPEAKER_LOW_FREQUENCY = 0x8 << 4,
  AVS_IT_SPEAKER_BACK_LEFT = 0x10 << 4,
  AVS_IT_SPEAKER_BACK_RIGHT = 0x20 << 4,
  AVS_IT_SPEAKER_FRONT_LEFT_OF_CENTER = 0x40 << 4,
  AVS_IT_SPEAKER_FRONT_RIGHT_OF_CENTER = 0x80 << 4,
  AVS_IT_SPEAKER_BACK_CENTER = 0x100 << 4,
  AVS_IT_SPEAKER_SIDE_LEFT = 0x200 << 4,
  AVS_IT_SPEAKER_SIDE_RIGHT = 0x400 << 4,
  AVS_IT_SPEAKER_TOP_CENTER = 0x800 << 4,
  AVS_IT_SPEAKER_TOP_FRONT_LEFT = 0x1000 << 4,
  AVS_IT_SPEAKER_TOP_FRONT_CENTER = 0x2000 << 4,
  AVS_IT_SPEAKER_TOP_FRONT_RIGHT = 0x4000 << 4,
  AVS_IT_SPEAKER_TOP_BACK_LEFT = 0x8000 << 4,
  AVS_IT_SPEAKER_TOP_BACK_CENTER = 0x10000 << 4,
  AVS_IT_SPEAKER_TOP_BACK_RIGHT = 0x20000 << 4,
  // End of officially defined speaker bits
  // The next one is special, since cannot shift SPEAKER_ALL 0x80000000 further.
  // Set mask and get mask handles it.
  AVS_IT_SPEAKER_ALL = 0x40000 << 4,
  // Mask for the defined 18 bits + SPEAKER_ALL
  AVS_IT_SPEAKER_BITS_MASK = (AVS_MASK_SPEAKER_DEFINED << 4) | AVS_IT_SPEAKER_ALL,
  AVS_IT_NEXT_AVAILABLE = 1 << 23
};

enum {
  AVS_FILTER_TYPE = 1,
  AVS_FILTER_INPUT_COLORSPACE = 2,
  AVS_FILTER_OUTPUT_TYPE = 9,
  AVS_FILTER_NAME = 4,
  AVS_FILTER_AUTHOR = 5,
  AVS_FILTER_VERSION = 6,
  AVS_FILTER_ARGS = 7,
  AVS_FILTER_ARGS_INFO = 8,
  AVS_FILTER_ARGS_DESCRIPTION = 10,
  AVS_FILTER_DESCRIPTION = 11
};

enum {  // SUBTYPES
  AVS_FILTER_TYPE_AUDIO = 1,
  AVS_FILTER_TYPE_VIDEO = 2,
  AVS_FILTER_OUTPUT_TYPE_SAME = 3,
  AVS_FILTER_OUTPUT_TYPE_DIFFERENT = 4
};

enum {
  // New 2.6 explicitly defined cache hints.
  AVS_CACHE_NOTHING = 10, // Do not cache video.
  AVS_CACHE_WINDOW = 11, // Hard protect up to X frames within a range of X from the current frame N.
  AVS_CACHE_GENERIC = 12, // LRU cache up to X frames.
  AVS_CACHE_FORCE_GENERIC = 13, // LRU cache up to X frames, override any previous CACHE_WINDOW.

  AVS_CACHE_GET_POLICY = 30, // Get the current policy.
  AVS_CACHE_GET_WINDOW = 31, // Get the current window h_span.
  AVS_CACHE_GET_RANGE = 32, // Get the current generic frame range.

  // Set Audio cache mode and answers to CACHE_GETCHILD_AUDIO_MODE
  AVS_CACHE_AUDIO = 50, // Explicitly do cache audio, X byte cache.
  AVS_CACHE_AUDIO_NOTHING = 51, // Explicitly do not cache audio.
  AVS_CACHE_AUDIO_NONE = 52, // Audio cache off (auto mode), X byte initial cache.
  AVS_CACHE_AUDIO_AUTO_START_OFF = 52, // synonym
  AVS_CACHE_AUDIO_AUTO = 53, // Audio cache on (auto mode), X byte initial cache.
  AVS_CACHE_AUDIO_AUTO_START_ON = 53, // synonym

  // These just returns actual value if clip is cached
  AVS_CACHE_GET_AUDIO_POLICY = 70, // Get the current audio policy.
  AVS_CACHE_GET_AUDIO_SIZE = 71, // Get the current audio cache size.

  AVS_CACHE_PREFETCH_FRAME = 100, // n/a Queue request to prefetch frame N.
  AVS_CACHE_PREFETCH_GO = 101, // n/a Action video prefetches.

  AVS_CACHE_PREFETCH_AUDIO_BEGIN = 120, // n/a Begin queue request transaction to prefetch audio (take critical section).
  AVS_CACHE_PREFETCH_AUDIO_STARTLO = 121, // n/a Set low 32 bits of start.
  AVS_CACHE_PREFETCH_AUDIO_STARTHI = 122, // n/a Set high 32 bits of start.
  AVS_CACHE_PREFETCH_AUDIO_COUNT = 123, // n/a Set low 32 bits of length.
  AVS_CACHE_PREFETCH_AUDIO_COMMIT = 124, // n/a Enqueue request transaction to prefetch audio (release critical section).
  AVS_CACHE_PREFETCH_AUDIO_GO = 125, // n/a Action audio prefetches.

  AVS_CACHE_GETCHILD_CACHE_MODE = 200, // n/a Cache ask Child for desired video cache mode.
  AVS_CACHE_GETCHILD_CACHE_SIZE = 201, // n/a Cache ask Child for desired video cache size.

  // Filters are queried about their desired audio cache mode.
  // Child can answer them with CACHE_AUDIO_xxx
  AVS_CACHE_GETCHILD_AUDIO_MODE = 202, // Cache ask Child for desired audio cache mode.
  AVS_CACHE_GETCHILD_AUDIO_SIZE = 203, // Cache ask Child for desired audio cache size.

  AVS_CACHE_GETCHILD_COST = 220, // n/a Cache ask Child for estimated processing cost.
  AVS_CACHE_COST_ZERO = 221, // n/a Child response of zero cost (ptr arithmetic only).
  AVS_CACHE_COST_UNIT = 222, // n/a Child response of unit cost (less than or equal 1 full frame blit).
  AVS_CACHE_COST_LOW = 223, // n/a Child response of light cost. (Fast)
  AVS_CACHE_COST_MED = 224, // n/a Child response of medium cost. (Real time)
  AVS_CACHE_COST_HI = 225, // n/a Child response of heavy cost. (Slow)

  AVS_CACHE_GETCHILD_THREAD_MODE = 240, // n/a Cache ask Child for thread safety.
  AVS_CACHE_THREAD_UNSAFE = 241, // n/a Only 1 thread allowed for all instances. 2.5 filters default!
  AVS_CACHE_THREAD_CLASS = 242, // n/a Only 1 thread allowed for each instance. 2.6 filters default!
  AVS_CACHE_THREAD_SAFE = 243, // n/a Allow all threads in any instance.
  AVS_CACHE_THREAD_OWN = 244, // n/a Safe but limit to 1 thread, internally threaded.

  AVS_CACHE_GETCHILD_ACCESS_COST = 260, // Cache ask Child for preferred access pattern.
  AVS_CACHE_ACCESS_RAND = 261, // Filter is access order agnostic.
  AVS_CACHE_ACCESS_SEQ0 = 262, // Filter prefers sequential access (low cost)
  AVS_CACHE_ACCESS_SEQ1 = 263, // Filter needs sequential access (high cost)

  AVS_CACHE_AVSPLUS_CONSTANTS = 500,    // Smaller values are reserved for classic Avisynth

  AVS_CACHE_DONT_CACHE_ME = 501,              // Filters that don't need caching (eg. trim, cache etc.) should return 1 to this request
  AVS_CACHE_SET_MIN_CAPACITY = 502,
  AVS_CACHE_SET_MAX_CAPACITY = 503,
  AVS_CACHE_GET_MIN_CAPACITY = 504,
  AVS_CACHE_GET_MAX_CAPACITY = 505,
  AVS_CACHE_GET_SIZE = 506,
  AVS_CACHE_GET_REQUESTED_CAP = 507,
  AVS_CACHE_GET_CAPACITY = 508,
  AVS_CACHE_GET_MTMODE = 509,                 // Filters specify their desired MT mode, see enum MtMode

  // By returning IS_CACHE_ANS to IS_CACHE_REQ, we tell the caller we are a cache
  AVS_CACHE_IS_CACHE_REQ = 510,
  AVS_CACHE_IS_CACHE_ANS = 511,
  // By returning IS_MTGUARD_ANS to IS_MTGUARD_REQ, we tell the caller we are an mt guard
  AVS_CACHE_IS_MTGUARD_REQ = 512,
  AVS_CACHE_IS_MTGUARD_ANS = 513,

  AVS_CACHE_AVSPLUS_CUDA_CONSTANTS = 600,

  AVS_CACHE_GET_DEV_TYPE = 601,          // Device types a filter can return
  AVS_CACHE_GET_CHILD_DEV_TYPE = 602,    // Device types a fitler can receive

  AVS_CACHE_USER_CONSTANTS = 1000       // Smaller values are reserved for the core
};



// enums for frame property functions
// AVSPropTypes
enum {
  AVS_PROPTYPE_UNSET = 'u',
  AVS_PROPTYPE_INT = 'i',
  AVS_PROPTYPE_FLOAT = 'f',
  AVS_PROPTYPE_DATA = 's',
  AVS_PROPTYPE_CLIP = 'c',
  AVS_PROPTYPE_FRAME = 'v'
};

// AVSGetPropErrors for avs_prop_get_...
enum {
  AVS_GETPROPERROR_UNSET = 1,
  AVS_GETPROPERROR_TYPE = 2,
  AVS_GETPROPERROR_INDEX = 4
};

// AVSPropAppendMode for avs_prop_set_...
enum {
  AVS_PROPAPPENDMODE_REPLACE = 0,
  AVS_PROPAPPENDMODE_APPEND = 1,
  AVS_PROPAPPENDMODE_TOUCH = 2
};

// AvsEnvProperty for avs_get_env_property
enum {
  AVS_AEP_PHYSICAL_CPUS = 1,
  AVS_AEP_LOGICAL_CPUS = 2,
  AVS_AEP_THREADPOOL_THREADS = 3,
  AVS_AEP_FILTERCHAIN_THREADS = 4,
  AVS_AEP_THREAD_ID = 5,
  AVS_AEP_VERSION = 6,
  AVS_AEP_HOST_SYSTEM_ENDIANNESS = 7,
  AVS_AEP_INTERFACE_VERSION = 8,
  AVS_AEP_INTERFACE_BUGFIX = 9,

  // Neo additionals
  AVS_AEP_NUM_DEVICES = 901,
  AVS_AEP_FRAME_ALIGN = 902,
  AVS_AEP_PLANE_ALIGN = 903,

  AVS_AEP_SUPPRESS_THREAD = 921,
  AVS_AEP_GETFRAME_RECURSIVE = 922
};

// enum AvsAllocType for avs_allocate
enum {
  AVS_ALLOCTYPE_NORMAL_ALLOC = 1,
  AVS_ALLOCTYPE_POOLED_ALLOC = 2
};

#ifdef BUILDING_AVSCORE
AVSValue create_c_video_filter(AVSValue args, void * user_data, IScriptEnvironment * e0);

struct AVS_ScriptEnvironment {
        IScriptEnvironment * env;
        const char * error;
        AVS_ScriptEnvironment(IScriptEnvironment * e = 0)
                : env(e), error(0) {}
};
#endif

typedef struct AVS_Clip AVS_Clip;
typedef struct AVS_ScriptEnvironment AVS_ScriptEnvironment;

/////////////////////////////////////////////////////////////////////
//
// AVS_VideoInfo
//

// AVS_VideoInfo is laid out identically to VideoInfo
typedef struct AVS_VideoInfo {
  int width, height;    // width=0 means no video
  unsigned fps_numerator, fps_denominator;
  int num_frames;

  int pixel_type;

  int audio_samples_per_second;   // 0 means no audio
  int sample_type;
  int64_t num_audio_samples;
  int nchannels;

  // Image type properties
  // BFF, TFF, FIELDBASED. Also used for storing Channel Mask
  // Manipulate it through the channelmask interface calls
  int image_type;
} AVS_VideoInfo;

// useful functions of the above
AVSC_INLINE int avs_has_video(const AVS_VideoInfo * p)
        { return (p->width!=0); }

AVSC_INLINE int avs_has_audio(const AVS_VideoInfo * p)
        { return (p->audio_samples_per_second!=0); }

AVSC_INLINE int avs_is_rgb(const AVS_VideoInfo * p)
        { return !!(p->pixel_type&AVS_CS_BGR); }

AVSC_INLINE int avs_is_rgb24(const AVS_VideoInfo * p)
        { return ((p->pixel_type&AVS_CS_BGR24)==AVS_CS_BGR24) && ((p->pixel_type & AVS_CS_SAMPLE_BITS_MASK) == AVS_CS_SAMPLE_BITS_8); }

AVSC_INLINE int avs_is_rgb32(const AVS_VideoInfo * p)
       { return ((p->pixel_type&AVS_CS_BGR32)==AVS_CS_BGR32) && ((p->pixel_type & AVS_CS_SAMPLE_BITS_MASK) == AVS_CS_SAMPLE_BITS_8); }

AVSC_INLINE int avs_is_yuv(const AVS_VideoInfo * p)
        { return !!(p->pixel_type&AVS_CS_YUV ); }

AVSC_INLINE int avs_is_yuy2(const AVS_VideoInfo * p)
        { return (p->pixel_type & AVS_CS_YUY2) == AVS_CS_YUY2; }

AVSC_API(int, avs_is_yv24)(const AVS_VideoInfo * p); // avs+: for generic 444 check, use avs_is_yuv444

AVSC_API(int, avs_is_yv16)(const AVS_VideoInfo * p); // avs+: for generic 422 check, use avs_is_yuv422

AVSC_API(int, avs_is_yv12)(const AVS_VideoInfo * p) ; // avs+: for generic 420 check, use avs_is_yuv420

AVSC_API(int, avs_is_yv411)(const AVS_VideoInfo * p);

AVSC_API(int, avs_is_y8)(const AVS_VideoInfo * p); // avs+: for generic grayscale, use avs_is_y

AVSC_API(int, avs_get_plane_width_subsampling)(const AVS_VideoInfo * p, int plane);

AVSC_API(int, avs_get_plane_height_subsampling)(const AVS_VideoInfo * p, int plane);

AVSC_API(int, avs_bits_per_pixel)(const AVS_VideoInfo * p);

AVSC_API(int, avs_bytes_from_pixels)(const AVS_VideoInfo * p, int pixels);

AVSC_API(int, avs_row_size)(const AVS_VideoInfo * p, int plane);

AVSC_API(int, avs_bmp_size)(const AVS_VideoInfo * vi);

AVSC_API(int, avs_is_color_space)(const AVS_VideoInfo * p, int c_space);

// no API for these, inline helper functions
AVSC_INLINE int avs_is_property(const AVS_VideoInfo * p, int property)
{
  return ((p->image_type & property) == property);
}

AVSC_INLINE int avs_is_planar(const AVS_VideoInfo * p)
{
  return !!(p->pixel_type & AVS_CS_PLANAR);
}

AVSC_INLINE int avs_is_field_based(const AVS_VideoInfo * p)
{
  return !!(p->image_type & AVS_IT_FIELDBASED);
}

AVSC_INLINE int avs_is_parity_known(const AVS_VideoInfo * p)
{
  return ((p->image_type & AVS_IT_FIELDBASED) && (p->image_type & (AVS_IT_BFF | AVS_IT_TFF)));
}

AVSC_INLINE int avs_is_bff(const AVS_VideoInfo * p)
{
  return !!(p->image_type & AVS_IT_BFF);
}

AVSC_INLINE int avs_is_tff(const AVS_VideoInfo * p)
{
  return !!(p->image_type & AVS_IT_TFF);
}

AVSC_INLINE int avs_samples_per_second(const AVS_VideoInfo * p)
        { return p->audio_samples_per_second; }

AVSC_INLINE int avs_bytes_per_channel_sample(const AVS_VideoInfo * p)
{
    switch (p->sample_type) {
      case AVS_SAMPLE_INT8:  return sizeof(signed char);
      case AVS_SAMPLE_INT16: return sizeof(signed short);
      case AVS_SAMPLE_INT24: return 3;
      case AVS_SAMPLE_INT32: return sizeof(signed int);
      case AVS_SAMPLE_FLOAT: return sizeof(float);
      default: return 0;
    }
}

AVSC_INLINE int avs_bytes_per_audio_sample(const AVS_VideoInfo * p)
        { return p->nchannels*avs_bytes_per_channel_sample(p);}

AVSC_INLINE int64_t avs_audio_samples_from_frames(const AVS_VideoInfo * p, int64_t frames)
        { return ((int64_t)(frames) * p->audio_samples_per_second * p->fps_denominator / p->fps_numerator); }

AVSC_INLINE int avs_frames_from_audio_samples(const AVS_VideoInfo * p, int64_t samples)
        { return (int)(samples * (int64_t)p->fps_numerator / (int64_t)p->fps_denominator / (int64_t)p->audio_samples_per_second); }

AVSC_INLINE int64_t avs_audio_samples_from_bytes(const AVS_VideoInfo * p, int64_t bytes)
        { return bytes / avs_bytes_per_audio_sample(p); }

AVSC_INLINE int64_t avs_bytes_from_audio_samples(const AVS_VideoInfo * p, int64_t samples)
        { return samples * avs_bytes_per_audio_sample(p); }

AVSC_INLINE int avs_audio_channels(const AVS_VideoInfo * p)
        { return p->nchannels; }

AVSC_INLINE int avs_sample_type(const AVS_VideoInfo * p)
        { return p->sample_type;}

// useful mutator
// Note: these are video format properties, neither frame properties, nor system properties
AVSC_INLINE void avs_set_property(AVS_VideoInfo * p, int property)
        { p->image_type|=property; }

AVSC_INLINE void avs_clear_property(AVS_VideoInfo * p, int property)
        { p->image_type&=~property; }

AVSC_INLINE void avs_set_field_based(AVS_VideoInfo * p, int isfieldbased)
        { if (isfieldbased) p->image_type|=AVS_IT_FIELDBASED; else p->image_type&=~AVS_IT_FIELDBASED; }

AVSC_INLINE void avs_set_fps(AVS_VideoInfo * p, unsigned numerator, unsigned denominator)
{
    unsigned x=numerator, y=denominator;
    while (y) {   // find gcd
      unsigned t = x%y; x = y; y = t;
    }
    p->fps_numerator = numerator/x;
    p->fps_denominator = denominator/x;
}

#ifndef AVSC_NO_DECLSPEC
// this inline function is calling an API function
AVSC_INLINE int avs_is_same_colorspace(const AVS_VideoInfo * x, const AVS_VideoInfo * y)
{
        return (x->pixel_type == y->pixel_type)
                || (avs_is_yv12(x) && avs_is_yv12(y));
}
#endif

// AviSynth+ extensions
AVSC_API(int, avs_is_rgb48)(const AVS_VideoInfo * p);

AVSC_API(int, avs_is_rgb64)(const AVS_VideoInfo * p);

AVSC_API(int, avs_is_yuv444p16)(const AVS_VideoInfo * p); // deprecated, use avs_is_yuv444
AVSC_API(int, avs_is_yuv422p16)(const AVS_VideoInfo * p); // deprecated, use avs_is_yuv422
AVSC_API(int, avs_is_yuv420p16)(const AVS_VideoInfo * p); // deprecated, use avs_is_yuv420
AVSC_API(int, avs_is_y16)(const AVS_VideoInfo * p); // deprecated, use avs_is_y
AVSC_API(int, avs_is_yuv444ps)(const AVS_VideoInfo * p); // deprecated, use avs_is_yuv444
AVSC_API(int, avs_is_yuv422ps)(const AVS_VideoInfo * p); // deprecated, use avs_is_yuv422
AVSC_API(int, avs_is_yuv420ps)(const AVS_VideoInfo * p); // deprecated, use avs_is_yuv420
AVSC_API(int, avs_is_y32)(const AVS_VideoInfo * p); // deprecated, use avs_is_y

AVSC_API(int, avs_is_444)(const AVS_VideoInfo * p);

AVSC_API(int, avs_is_422)(const AVS_VideoInfo * p);

AVSC_API(int, avs_is_420)(const AVS_VideoInfo * p);

AVSC_API(int, avs_is_y)(const AVS_VideoInfo * p);

AVSC_API(int, avs_is_yuva)(const AVS_VideoInfo * p);

AVSC_API(int, avs_is_planar_rgb)(const AVS_VideoInfo * p);

AVSC_API(int, avs_is_planar_rgba)(const AVS_VideoInfo * p);

AVSC_API(int, avs_num_components)(const AVS_VideoInfo * p);

AVSC_API(int, avs_component_size)(const AVS_VideoInfo * p);

AVSC_API(int, avs_bits_per_component)(const AVS_VideoInfo * p);

// V10
AVSC_API(bool, avs_is_channel_mask_known)(const AVS_VideoInfo* p);

AVSC_API(void, avs_set_channel_mask)(const AVS_VideoInfo* p, bool isChannelMaskKnown, unsigned int dwChannelMask);

AVSC_API(unsigned int, avs_get_channel_mask)(const AVS_VideoInfo* p);

// end of Avisynth+ specific

/////////////////////////////////////////////////////////////////////
//
// AVS_VideoFrame
//

// VideoFrameBuffer holds information about a memory block which is used
// for video data.  For efficiency, instances of this class are not deleted
// when the refcount reaches zero; instead they're stored in a linked list
// to be reused.  The instances are deleted when the corresponding AVS
// file is closed.

// AVS_VideoFrameBuffer is laid out identically to VideoFrameBuffer
// DO NOT USE THIS STRUCTURE DIRECTLY
typedef struct AVS_VideoFrameBuffer {
  BYTE * data;
  int data_size;
  // sequence_number is incremented every time the buffer is changed, so
  // that stale views can tell they're no longer valid.
  volatile long sequence_number;

  volatile long refcount;

  void* device; // avs+
} AVS_VideoFrameBuffer;

// VideoFrame holds a "window" into a VideoFrameBuffer.

// AVS_VideoFrame is laid out identically to VideoFrame
// DO NOT USE THIS STRUCTURE DIRECTLY
typedef struct AVS_VideoFrame {
  volatile long refcount;
  AVS_VideoFrameBuffer * vfb;
  int offset;
  // DO NOT USE THEM DIRECTLY
  // Use avs_get_pitch_p, avs_get_row_size_p, avs_get_height_p
  int pitch, row_size, height;
  int offsetU, offsetV;
  int pitchUV;  // U&V offsets are from top of picture.
  int row_sizeUV, heightUV; // for Planar RGB offsetU, offsetV is for the 2nd and 3rd Plane.
                            // for Planar RGB pitchUV and row_sizeUV = 0, because when no VideoInfo (MakeWriteable)
                            // the decision on existence of UV is checked by zero pitch
  // AVS+ extension, avisynth.h: class does not break plugins if appended here
  int offsetA;
  int pitchA, row_sizeA; // 4th alpha plane support, pitch and row_size is 0 is none
  void* properties; // interface V8: frame properties
  // DO NOT USE DIRECTLY
  // Use avs_video_frame_get_pixel_type (and avs_video_frame_amend_pixel_type in special cases)
  int pixel_type; // Interface V10: an automatically maintained copy from AVS_VideoInfo
} AVS_VideoFrame;

// Access functions for AVS_VideoFrame
AVSC_API(int, avs_get_pitch_p)(const AVS_VideoFrame * p, int plane);

AVSC_API(int, avs_get_row_size_p)(const AVS_VideoFrame * p, int plane);

AVSC_API(int, avs_get_height_p)(const AVS_VideoFrame * p, int plane);

AVSC_API(const BYTE *, avs_get_read_ptr_p)(const AVS_VideoFrame * p, int plane);

AVSC_API(int, avs_is_writable)(const AVS_VideoFrame * p);

// V9
AVSC_API(int, avs_is_property_writable)(const AVS_VideoFrame* p);

AVSC_API(BYTE *, avs_get_write_ptr_p)(const AVS_VideoFrame * p, int plane);

AVSC_API(void, avs_release_video_frame)(AVS_VideoFrame *);
// makes a shallow copy of a video frame
AVSC_API(AVS_VideoFrame *, avs_copy_video_frame)(AVS_VideoFrame *);

// V10
AVSC_API(int, avs_video_frame_get_pixel_type)(const AVS_VideoFrame* p);
// V10
AVSC_API(void, avs_video_frame_amend_pixel_type)(AVS_VideoFrame* p, int new_pixel_type);

// no API for these, inline helper functions
#ifndef AVSC_NO_DECLSPEC
// this inline function is calling an API function
AVSC_INLINE int avs_get_pitch(const AVS_VideoFrame * p) {
  return avs_get_pitch_p(p, 0);
}
#endif

#ifndef AVSC_NO_DECLSPEC
// this inline function is calling an API function
AVSC_INLINE int avs_get_row_size(const AVS_VideoFrame * p) {
        return avs_get_row_size_p(p, 0); }
#endif


#ifndef AVSC_NO_DECLSPEC
// this inline function is calling an API function
AVSC_INLINE int avs_get_height(const AVS_VideoFrame * p) {
  return avs_get_height_p(p, 0);
}
#endif

#ifndef AVSC_NO_DECLSPEC
// this inline function is calling an API function
AVSC_INLINE const BYTE* avs_get_read_ptr(const AVS_VideoFrame * p) {
        return avs_get_read_ptr_p(p, 0);}
#endif

#ifndef AVSC_NO_DECLSPEC
// this inline function is calling an API function
AVSC_INLINE BYTE* avs_get_write_ptr(const AVS_VideoFrame * p) {
        return avs_get_write_ptr_p(p, 0);}
#endif

#ifndef AVSC_NO_DECLSPEC
// this inline function is calling an API function
AVSC_INLINE void avs_release_frame(AVS_VideoFrame * f)
  {avs_release_video_frame(f);}
#endif

#ifndef AVSC_NO_DECLSPEC
// this inline function is calling an API function
AVSC_INLINE AVS_VideoFrame * avs_copy_frame(AVS_VideoFrame * f)
  {return avs_copy_video_frame(f);}
#endif

// Interface V8: frame properties
// AVS_Map is just a placeholder for AVSMap
typedef struct AVS_Map {
  void* data;
} AVS_Map;


/////////////////////////////////////////////////////////////////////
//
// AVS_Value
//

// Treat AVS_Value as a fat pointer.  That is use avs_copy_value
// and avs_release_value appropriately as you would if AVS_Value was
// a pointer.

// To maintain source code compatibility with future versions of the
// avisynth_c API don't use the AVS_Value directly.  Use the helper
// functions below.

// AVS_Value is laid out identically to AVSValue
typedef struct AVS_Value AVS_Value;
struct AVS_Value {
  short type;  // 'a'rray, 'c'lip, 'b'ool, 'i'nt, 'f'loat, 's'tring, 'v'oid, or 'l'ong, or fu'n'ction
               // for some function e'rror
  short array_size;
  union {
    void * clip; // do not use directly, use avs_take_clip
    char boolean;
    int integer;
    float floating_pt;
    const char * string;
    const AVS_Value * array;
    void * function; // not supported on C interface
#ifdef X86_64
    // if ever, only x64 will support. It breaks struct size on 32 bit
    int64_t longlong; // 8 bytes
    double double_pt; // 8 bytes
#endif
  } d;
};

// AVS_Value should be initialized with avs_void.
// Should also set to avs_void after the value is released
// with avs_copy_value.  Consider it the equivalent of setting
// a pointer to NULL
static const AVS_Value avs_void = {'v'};

AVSC_API(void, avs_copy_value)(AVS_Value * dest, AVS_Value src);
AVSC_API(void, avs_release_value)(AVS_Value);
AVSC_API(AVS_Clip *, avs_take_clip)(AVS_Value, AVS_ScriptEnvironment *);
AVSC_API(void, avs_set_to_clip)(AVS_Value *, AVS_Clip *);


// no API for these, inline helper functions
AVSC_INLINE int avs_defined(AVS_Value v) { return v.type != 'v'; }
AVSC_INLINE int avs_is_clip(AVS_Value v) { return v.type == 'c'; }
AVSC_INLINE int avs_is_bool(AVS_Value v) { return v.type == 'b'; }
AVSC_INLINE int avs_is_int(AVS_Value v) { return v.type == 'i'; }
AVSC_INLINE int avs_is_float(AVS_Value v) { return v.type == 'f' || v.type == 'i'; }
AVSC_INLINE int avs_is_string(AVS_Value v) { return v.type == 's'; }
AVSC_INLINE int avs_is_array(AVS_Value v) { return v.type == 'a'; }
AVSC_INLINE int avs_is_error(AVS_Value v) { return v.type == 'e'; }

AVSC_INLINE int avs_as_bool(AVS_Value v)
        { return v.d.boolean; }
AVSC_INLINE int avs_as_int(AVS_Value v)
        { return v.d.integer; }
AVSC_INLINE const char * avs_as_string(AVS_Value v)
        { return avs_is_error(v) || avs_is_string(v) ? v.d.string : 0; }
AVSC_INLINE double avs_as_float(AVS_Value v)
        { return avs_is_int(v) ? v.d.integer : v.d.floating_pt; }
AVSC_INLINE const char * avs_as_error(AVS_Value v)
        { return avs_is_error(v) ? v.d.string : 0; }
AVSC_INLINE const AVS_Value * avs_as_array(AVS_Value v)
        { return v.d.array; }
AVSC_INLINE int avs_array_size(AVS_Value v)
        { return avs_is_array(v) ? v.array_size : 1; }
AVSC_INLINE AVS_Value avs_array_elt(AVS_Value v, int index)
        { return avs_is_array(v) ? v.d.array[index] : v; }

// only use these functions on an AVS_Value that does not already have
// an active value.  Remember, treat AVS_Value as a fat pointer.
AVSC_INLINE AVS_Value avs_new_value_bool(int v0)
        { AVS_Value v; v.type = 'b'; v.d.boolean = v0 == 0 ? 0 : 1; return v; }
AVSC_INLINE AVS_Value avs_new_value_int(int v0)
        { AVS_Value v; v.type = 'i'; v.d.integer = v0; return v; }
AVSC_INLINE AVS_Value avs_new_value_string(const char * v0)
        { AVS_Value v; v.type = 's'; v.d.string = v0; return v; }
AVSC_INLINE AVS_Value avs_new_value_float(float v0)
        { AVS_Value v; v.type = 'f'; v.d.floating_pt = v0; return v;}
AVSC_INLINE AVS_Value avs_new_value_error(const char * v0)
        { AVS_Value v; v.type = 'e'; v.d.string = v0; return v; }
#ifndef AVSC_NO_DECLSPEC
// this inline function is calling an API function
AVSC_INLINE AVS_Value avs_new_value_clip(AVS_Clip * v0)
        { AVS_Value v; avs_set_to_clip(&v, v0); return v; }
#endif
AVSC_INLINE AVS_Value avs_new_value_array(AVS_Value * v0, int size)
        { AVS_Value v; v.type = 'a'; v.d.array = v0; v.array_size = (short)size; return v; }
// end of inline helper functions

/////////////////////////////////////////////////////////////////////
//
// AVS_Clip
//

AVSC_API(void, avs_release_clip)(AVS_Clip *);
AVSC_API(AVS_Clip *, avs_copy_clip)(AVS_Clip *);

AVSC_API(const char *, avs_clip_get_error)(AVS_Clip *); // return 0 if no error

AVSC_API(const AVS_VideoInfo *, avs_get_video_info)(AVS_Clip *);

AVSC_API(int, avs_get_version)(AVS_Clip *);

AVSC_API(AVS_VideoFrame *, avs_get_frame)(AVS_Clip *, int n);
// The returned video frame must be released with avs_release_video_frame

AVSC_API(int, avs_get_parity)(AVS_Clip *, int n);
// return field parity if field_based, else parity of first field in frame

AVSC_API(int, avs_get_audio)(AVS_Clip *, void * buf,
                             int64_t start, int64_t count);
// start and count are in samples

AVSC_API(int, avs_set_cache_hints)(AVS_Clip *,
                                   int cachehints, int frame_range);

// This is the callback type used by avs_add_function
typedef AVS_Value (AVSC_CC * AVS_ApplyFunc)
                        (AVS_ScriptEnvironment *, AVS_Value args, void * user_data);

typedef struct AVS_FilterInfo AVS_FilterInfo;
struct AVS_FilterInfo
{
  // these members should not be modified outside of the AVS_ApplyFunc callback
  AVS_Clip * child;
  AVS_VideoInfo vi;
  AVS_ScriptEnvironment * env;
  AVS_VideoFrame * (AVSC_CC * get_frame)(AVS_FilterInfo *, int n);
  int (AVSC_CC * get_parity)(AVS_FilterInfo *, int n);
  int (AVSC_CC * get_audio)(AVS_FilterInfo *, void * buf,
                                  int64_t start, int64_t count);
  int (AVSC_CC * set_cache_hints)(AVS_FilterInfo *, int cachehints,
                                        int frame_range);
  void (AVSC_CC * free_filter)(AVS_FilterInfo *);

  // Should be set when ever there is an error to report.
  // It is cleared before any of the above methods are called
  const char * error;
  // this is to store whatever and may be modified at will
  void * user_data;
};

// Create a new filter
// fi is set to point to the AVS_FilterInfo so that you can
//   modify it once it is initialized.
// store_child should generally be set to true.  If it is not
//    set than ALL methods (the function pointers) must be defined
// If it is set than you do not need to worry about freeing the child
//    clip.
AVSC_API(AVS_Clip *, avs_new_c_filter)(AVS_ScriptEnvironment * e,
                                       AVS_FilterInfo * * fi,
                                       AVS_Value child, int store_child);

/////////////////////////////////////////////////////////////////////
//
// AVS_ScriptEnvironment
//

// For GetCPUFlags.  These are backwards-compatible with those in VirtualDub.
enum {
                                /* slowest CPU to support extension */
  AVS_CPU_FORCE        = 0x01,   // N/A
  AVS_CPU_FPU          = 0x02,   // 386/486DX
  AVS_CPU_MMX          = 0x04,   // P55C, K6, PII
  AVS_CPU_INTEGER_SSE  = 0x08,   // PIII, Athlon
  AVS_CPU_SSE          = 0x10,   // PIII, Athlon XP/MP
  AVS_CPU_SSE2         = 0x20,   // PIV, Hammer
  AVS_CPU_3DNOW        = 0x40,   // K6-2
  AVS_CPU_3DNOW_EXT    = 0x80,   // Athlon
  AVS_CPU_X86_64       = 0xA0,   // Hammer (note: equiv. to 3DNow + SSE2,
                                 // which only Hammer will have anyway)
  AVS_CPUF_SSE3       = 0x100,   //  PIV+, K8 Venice
  AVS_CPUF_SSSE3      = 0x200,   //  Core 2
  AVS_CPUF_SSE4       = 0x400,   //  Penryn, Wolfdale, Yorkfield
  AVS_CPUF_SSE4_1     = 0x400,
  AVS_CPUF_AVX        = 0x800,   //  Sandy Bridge, Bulldozer
  AVS_CPUF_SSE4_2    = 0x1000,   //  Nehalem
  // AVS+
  AVS_CPUF_AVX2      = 0x2000,   //  Haswell
  AVS_CPUF_FMA3      = 0x4000,
  AVS_CPUF_F16C      = 0x8000,
  AVS_CPUF_MOVBE     = 0x10000,   // Big Endian Move
  AVS_CPUF_POPCNT    = 0x20000,
  AVS_CPUF_AES       = 0x40000,
  AVS_CPUF_FMA4      = 0x80000,

  AVS_CPUF_AVX512F    = 0x100000,  // AVX-512 Foundation.
  AVS_CPUF_AVX512DQ   = 0x200000,  // AVX-512 DQ (Double/Quad granular) Instructions
  AVS_CPUF_AVX512PF   = 0x400000,  // AVX-512 Prefetch
  AVS_CPUF_AVX512ER   = 0x800000,  // AVX-512 Exponential and Reciprocal
  AVS_CPUF_AVX512CD   = 0x1000000, // AVX-512 Conflict Detection
  AVS_CPUF_AVX512BW   = 0x2000000, // AVX-512 BW (Byte/Word granular) Instructions
  AVS_CPUF_AVX512VL   = 0x4000000, // AVX-512 VL (128/256 Vector Length) Extensions
  AVS_CPUF_AVX512IFMA = 0x8000000, // AVX-512 IFMA integer 52 bit
  AVS_CPUF_AVX512VBMI = 0x10000000 // AVX-512 VBMI
};


AVSC_API(const char *, avs_get_error)(AVS_ScriptEnvironment *); // return 0 if no error

AVSC_API(int, avs_get_cpu_flags)(AVS_ScriptEnvironment *);
AVSC_API(int, avs_check_version)(AVS_ScriptEnvironment *, int version);

AVSC_API(char *, avs_save_string)(AVS_ScriptEnvironment *, const char* s, int length);
AVSC_API(char *, avs_sprintf)(AVS_ScriptEnvironment *, const char * fmt, ...);

AVSC_API(char *, avs_vsprintf)(AVS_ScriptEnvironment *, const char * fmt, va_list val);

AVSC_API(int, avs_add_function)(AVS_ScriptEnvironment *,
                                const char * name, const char * params,
                                AVS_ApplyFunc apply, void * user_data);

AVSC_API(int, avs_function_exists)(AVS_ScriptEnvironment *, const char * name);

AVSC_API(AVS_Value, avs_invoke)(AVS_ScriptEnvironment *, const char * name,
                               AVS_Value args, const char** arg_names);
// The returned value must be be released with avs_release_value

AVSC_API(AVS_Value, avs_get_var)(AVS_ScriptEnvironment *, const char* name);
// The returned value must be be released with avs_release_value

AVSC_API(int, avs_set_var)(AVS_ScriptEnvironment *, const char* name, AVS_Value val);

AVSC_API(int, avs_set_global_var)(AVS_ScriptEnvironment *, const char* name, const AVS_Value val);

//void avs_push_context(AVS_ScriptEnvironment *, int level=0);
//void avs_pop_context(AVS_ScriptEnvironment *);

// partially deprecated, from V8 use avs_new_video_frame_p_a (frame property copy)
AVSC_API(AVS_VideoFrame *, avs_new_video_frame_a)(AVS_ScriptEnvironment *,
                                          const AVS_VideoInfo * vi, int align);
// align should be at least 16 for classic Avisynth
// Avisynth+: any value, Avs+ ensures a minimum alignment if too small align is provided

// no API for these, inline helper functions
#ifndef AVSC_NO_DECLSPEC
// partially deprecated, from V8 use avs_new_video_frame_p (frame property copy)
// this inline function is calling an API function
AVSC_INLINE AVS_VideoFrame * avs_new_video_frame(AVS_ScriptEnvironment * env,
                                     const AVS_VideoInfo * vi)
  {return avs_new_video_frame_a(env,vi,AVS_FRAME_ALIGN);}

// an older compatibility alias
// this inline function is calling an API function
AVSC_INLINE AVS_VideoFrame * avs_new_frame(AVS_ScriptEnvironment * env,
                               const AVS_VideoInfo * vi)
  {return avs_new_video_frame_a(env,vi,AVS_FRAME_ALIGN);}
#endif
// end of inline helper functions

AVSC_API(int, avs_make_writable)(AVS_ScriptEnvironment *, AVS_VideoFrame * * pvf);

// V9
AVSC_API(int, avs_make_property_writable)(AVS_ScriptEnvironment*, AVS_VideoFrame** pvf);

AVSC_API(void, avs_bit_blt)(AVS_ScriptEnvironment *, BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height);

typedef void (AVSC_CC *AVS_ShutdownFunc)(void* user_data, AVS_ScriptEnvironment * env);
AVSC_API(void, avs_at_exit)(AVS_ScriptEnvironment *, AVS_ShutdownFunc function, void * user_data);

AVSC_API(AVS_VideoFrame *, avs_subframe)(AVS_ScriptEnvironment *, AVS_VideoFrame * src, int rel_offset, int new_pitch, int new_row_size, int new_height);
// The returned video frame must be be released
AVSC_API(AVS_VideoFrame*, avs_subframe_planar)(AVS_ScriptEnvironment*, AVS_VideoFrame* src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV);
// The returned video frame must be be released
// see also avs_subframe_planar_a in interface V8

AVSC_API(int, avs_set_memory_max)(AVS_ScriptEnvironment *, int mem);

AVSC_API(int, avs_set_working_dir)(AVS_ScriptEnvironment *, const char * newdir);

// avisynth.dll exports this; it's a way to use it as a library, without
// writing an AVS script or without going through AVIFile.
AVSC_API(AVS_ScriptEnvironment *, avs_create_script_environment)(int version);

// this symbol is the entry point for the plugin and must
// be defined
AVSC_EXPORT
const char * AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env);


AVSC_API(void, avs_delete_script_environment)(AVS_ScriptEnvironment *);

///////////////////////////////////////////////////////////////////////////////
//
// Avisynth+ V8 interface elements
//

AVSC_API(AVS_VideoFrame*, avs_subframe_planar_a)(AVS_ScriptEnvironment*, AVS_VideoFrame* src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA);
// The returned video frame must be be released

AVSC_API(void, avs_copy_frame_props)(AVS_ScriptEnvironment* p, const AVS_VideoFrame* src, AVS_VideoFrame* dst);
AVSC_API(const AVS_Map*, avs_get_frame_props_ro)(AVS_ScriptEnvironment* p, const AVS_VideoFrame* frame);
AVSC_API(AVS_Map*, avs_get_frame_props_rw)(AVS_ScriptEnvironment* p, AVS_VideoFrame* frame);
AVSC_API(int, avs_prop_num_keys)(AVS_ScriptEnvironment* p, const AVS_Map* map);
AVSC_API(const char*, avs_prop_get_key)(AVS_ScriptEnvironment* p, const AVS_Map* map, int index);
AVSC_API(int, avs_prop_num_elements)(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key);

// see AVS_PROPTYPE_... enums
AVSC_API(char, avs_prop_get_type)(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key);

// see AVS_GETPROPERROR_... enums
AVSC_API(int64_t, avs_prop_get_int)(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key, int index, int* error);
AVSC_API(double, avs_prop_get_float)(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key, int index, int* error);
// Note: avs_prop_get_data was fixed in interface V9.1
AVSC_API(const char*, avs_prop_get_data)(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key, int index, int* error);
AVSC_API(int, avs_prop_get_data_size)(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key, int index, int* error);
AVSC_API(AVS_Clip*, avs_prop_get_clip)(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key, int index, int* error);
AVSC_API(const AVS_VideoFrame*, avs_prop_get_frame)(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key, int index, int* error);

AVSC_API(int, avs_prop_delete_key)(AVS_ScriptEnvironment* p, AVS_Map* map, const char* key);

// see AVS_PROPAPPENDMODE_... enums
AVSC_API(int, avs_prop_set_int)(AVS_ScriptEnvironment* p, AVS_Map* map, const char* key, int64_t i, int append);
AVSC_API(int, avs_prop_set_float)(AVS_ScriptEnvironment* p, AVS_Map* map, const char* key, double d, int append);
AVSC_API(int, avs_prop_set_data)(AVS_ScriptEnvironment* p, AVS_Map* map, const char* key, const char* d, int length, int append);
AVSC_API(int, avs_prop_set_clip)(AVS_ScriptEnvironment* p, AVS_Map* map, const char* key, AVS_Clip* clip, int append);
AVSC_API(int, avs_prop_set_frame)(AVS_ScriptEnvironment* p, AVS_Map* map, const char* key, const AVS_VideoFrame* frame, int append);

AVSC_API(const int64_t*, avs_prop_get_int_array)(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key, int* error);
AVSC_API(const double*, avs_prop_get_float_array)(AVS_ScriptEnvironment* p, const AVS_Map* map, const char* key, int* error);
AVSC_API(int, avs_prop_set_int_array)(AVS_ScriptEnvironment* p, AVS_Map* map, const char* key, const int64_t* i, int size);
AVSC_API(int, avs_prop_set_float_array)(AVS_ScriptEnvironment* p, AVS_Map* map, const char* key, const double* d, int size);

AVSC_API(void, avs_clear_map)(AVS_ScriptEnvironment* p, AVS_Map* map);

// with frame property source
AVSC_API(AVS_VideoFrame*, avs_new_video_frame_p)(AVS_ScriptEnvironment*,
  const AVS_VideoInfo* vi, const AVS_VideoFrame* prop_src);

// with frame property source
AVSC_API(AVS_VideoFrame*, avs_new_video_frame_p_a)(AVS_ScriptEnvironment*,
  const AVS_VideoInfo* vi, const AVS_VideoFrame* prop_src, int align);

// Generic query to ask for various system properties, see AVS_AEP_xxx enums
AVSC_API(size_t, avs_get_env_property)(AVS_ScriptEnvironment*, int avs_aep_prop);

// buffer pool, see AVS_ALLOCTYPE enums
AVSC_API(void *, avs_pool_allocate)(AVS_ScriptEnvironment*, size_t nBytes, size_t alignment, int avs_alloc_type);
AVSC_API(void, avs_pool_free)(AVS_ScriptEnvironment*, void *ptr);

// Interface V8
// Returns TRUE (1) and the requested variable. If the method fails, returns 0 (FALSE) and does not touch 'val'.
// The returned AVS_Value *val value must be be released with avs_release_value only on success
// AVS_Value *val is not caller allocated
AVSC_API(int, avs_get_var_try)(AVS_ScriptEnvironment*, const char* name, AVS_Value* val);

// Interface V8
// Return the value of the requested variable.
// If the variable was not found or had the wrong type,
// return the supplied default value.
AVSC_API(int, avs_get_var_bool)(AVS_ScriptEnvironment*, const char* name, int def);
AVSC_API(int, avs_get_var_int)(AVS_ScriptEnvironment*, const char* name, int def);
AVSC_API(double, avs_get_var_double)(AVS_ScriptEnvironment*, const char* name, double def);
AVSC_API(const char*, avs_get_var_string)(AVS_ScriptEnvironment*, const char* name, const char* def);
AVSC_API(int64_t, avs_get_var_long)(AVS_ScriptEnvironment*, const char* name, int64_t def);

#if defined(AVS_WINDOWS)
// The following stuff is only relevant for Windows DLL handling; Linux does it completely differently.
#ifdef AVSC_NO_DECLSPEC
// This part uses LoadLibrary and related functions to dynamically load Avisynth instead of declspec(dllimport)
// When AVSC_NO_DECLSPEC is defined, you can use avs_load_library to populate API functions into a struct
// AVSC_INLINE functions which call onto an API functions should be treated specially (todo)

/*
  The following functions needs to have been declared, probably from windows.h

  void* malloc(size_t)
  void free(void*);

  HMODULE LoadLibraryA(const char*);
  void* GetProcAddress(HMODULE, const char*);
  FreeLibrary(HMODULE);
*/


typedef struct AVS_Library AVS_Library;

#define AVSC_DECLARE_FUNC(name) name##_func name

// AVSC_DECLARE_FUNC helps keeping naming convention: type is xxxxx_func, function name is xxxxx
// e.g. "AVSC_DECLARE_FUNC(avs_add_function);"
// is a shortcut for "avs_add_function_func avs_add_function;"

// Note: AVSC_INLINE functions, which call into API,
// are guarded by #ifndef AVSC_NO_DECLSPEC.
// They should call the appropriate library-> API entry.

struct AVS_Library {
  HMODULE handle;

  AVSC_DECLARE_FUNC(avs_add_function);
  AVSC_DECLARE_FUNC(avs_at_exit);
  AVSC_DECLARE_FUNC(avs_bit_blt);
  AVSC_DECLARE_FUNC(avs_check_version);
  AVSC_DECLARE_FUNC(avs_clip_get_error);
  AVSC_DECLARE_FUNC(avs_copy_clip);
  AVSC_DECLARE_FUNC(avs_copy_value);
  AVSC_DECLARE_FUNC(avs_copy_video_frame);
  AVSC_DECLARE_FUNC(avs_create_script_environment);
  AVSC_DECLARE_FUNC(avs_delete_script_environment);
  AVSC_DECLARE_FUNC(avs_function_exists);
  AVSC_DECLARE_FUNC(avs_get_audio);
  AVSC_DECLARE_FUNC(avs_get_cpu_flags);
  AVSC_DECLARE_FUNC(avs_get_frame);
  AVSC_DECLARE_FUNC(avs_get_parity);
  AVSC_DECLARE_FUNC(avs_get_var);
  AVSC_DECLARE_FUNC(avs_get_version);
  AVSC_DECLARE_FUNC(avs_get_video_info);
  AVSC_DECLARE_FUNC(avs_invoke);
  AVSC_DECLARE_FUNC(avs_make_writable);
  AVSC_DECLARE_FUNC(avs_new_c_filter);
  AVSC_DECLARE_FUNC(avs_new_video_frame_a);
  AVSC_DECLARE_FUNC(avs_release_clip);
  AVSC_DECLARE_FUNC(avs_release_value);
  AVSC_DECLARE_FUNC(avs_release_video_frame);
  AVSC_DECLARE_FUNC(avs_save_string);
  AVSC_DECLARE_FUNC(avs_set_cache_hints);
  AVSC_DECLARE_FUNC(avs_set_global_var);
  AVSC_DECLARE_FUNC(avs_set_memory_max);
  AVSC_DECLARE_FUNC(avs_set_to_clip);
  AVSC_DECLARE_FUNC(avs_set_var);
  AVSC_DECLARE_FUNC(avs_set_working_dir);
  AVSC_DECLARE_FUNC(avs_sprintf);
  AVSC_DECLARE_FUNC(avs_subframe);
  AVSC_DECLARE_FUNC(avs_subframe_planar);
  AVSC_DECLARE_FUNC(avs_take_clip);
  AVSC_DECLARE_FUNC(avs_vsprintf);

  AVSC_DECLARE_FUNC(avs_get_error);
  AVSC_DECLARE_FUNC(avs_is_yv24);
  AVSC_DECLARE_FUNC(avs_is_yv16);
  AVSC_DECLARE_FUNC(avs_is_yv12);
  AVSC_DECLARE_FUNC(avs_is_yv411);
  AVSC_DECLARE_FUNC(avs_is_y8);
  AVSC_DECLARE_FUNC(avs_is_color_space);

  AVSC_DECLARE_FUNC(avs_get_plane_width_subsampling);
  AVSC_DECLARE_FUNC(avs_get_plane_height_subsampling);
  AVSC_DECLARE_FUNC(avs_bits_per_pixel);
  AVSC_DECLARE_FUNC(avs_bytes_from_pixels);
  AVSC_DECLARE_FUNC(avs_row_size);
  AVSC_DECLARE_FUNC(avs_bmp_size);
  AVSC_DECLARE_FUNC(avs_get_pitch_p);
  AVSC_DECLARE_FUNC(avs_get_row_size_p);
  AVSC_DECLARE_FUNC(avs_get_height_p);
  AVSC_DECLARE_FUNC(avs_get_read_ptr_p);
  AVSC_DECLARE_FUNC(avs_is_writable);
  AVSC_DECLARE_FUNC(avs_get_write_ptr_p);

  // Avisynth+ specific
  // Note: these functions are simulated/use fallback to existing functions
  AVSC_DECLARE_FUNC(avs_is_rgb48);
  AVSC_DECLARE_FUNC(avs_is_rgb64);
  AVSC_DECLARE_FUNC(avs_is_yuv444p16);
  AVSC_DECLARE_FUNC(avs_is_yuv422p16);
  AVSC_DECLARE_FUNC(avs_is_yuv420p16);
  AVSC_DECLARE_FUNC(avs_is_y16);
  AVSC_DECLARE_FUNC(avs_is_yuv444ps);
  AVSC_DECLARE_FUNC(avs_is_yuv422ps);
  AVSC_DECLARE_FUNC(avs_is_yuv420ps);
  AVSC_DECLARE_FUNC(avs_is_y32);
  AVSC_DECLARE_FUNC(avs_is_444);
  AVSC_DECLARE_FUNC(avs_is_422);
  AVSC_DECLARE_FUNC(avs_is_420);
  AVSC_DECLARE_FUNC(avs_is_y);
  AVSC_DECLARE_FUNC(avs_is_yuva);
  AVSC_DECLARE_FUNC(avs_is_planar_rgb);
  AVSC_DECLARE_FUNC(avs_is_planar_rgba);
  AVSC_DECLARE_FUNC(avs_num_components);
  AVSC_DECLARE_FUNC(avs_component_size);
  AVSC_DECLARE_FUNC(avs_bits_per_component);

  ///////////////////////////////////////////////////////////////////////////////
  // Avisynth+ new interface elements from interface version 8
  // avs_subframe_planar with alpha support
  AVSC_DECLARE_FUNC(avs_subframe_planar_a);

  // frame properties
  AVSC_DECLARE_FUNC(avs_copy_frame_props);
  AVSC_DECLARE_FUNC(avs_get_frame_props_ro);
  AVSC_DECLARE_FUNC(avs_get_frame_props_rw);
  AVSC_DECLARE_FUNC(avs_prop_num_keys);
  AVSC_DECLARE_FUNC(avs_prop_get_key);
  AVSC_DECLARE_FUNC(avs_prop_num_elements);
  AVSC_DECLARE_FUNC(avs_prop_get_type);
  AVSC_DECLARE_FUNC(avs_prop_get_int);
  AVSC_DECLARE_FUNC(avs_prop_get_float);
  AVSC_DECLARE_FUNC(avs_prop_get_data);
  AVSC_DECLARE_FUNC(avs_prop_get_data_size);
  AVSC_DECLARE_FUNC(avs_prop_get_clip);
  AVSC_DECLARE_FUNC(avs_prop_get_frame);
  AVSC_DECLARE_FUNC(avs_prop_delete_key);
  AVSC_DECLARE_FUNC(avs_prop_set_int);
  AVSC_DECLARE_FUNC(avs_prop_set_float);
  AVSC_DECLARE_FUNC(avs_prop_set_data);
  AVSC_DECLARE_FUNC(avs_prop_set_clip);
  AVSC_DECLARE_FUNC(avs_prop_set_frame);

  AVSC_DECLARE_FUNC(avs_prop_get_int_array);
  AVSC_DECLARE_FUNC(avs_prop_get_float_array);
  AVSC_DECLARE_FUNC(avs_prop_set_int_array);
  AVSC_DECLARE_FUNC(avs_prop_set_float_array);

  AVSC_DECLARE_FUNC(avs_clear_map);

  // NewVideoFrame with frame properties
  AVSC_DECLARE_FUNC(avs_new_video_frame_p);
  AVSC_DECLARE_FUNC(avs_new_video_frame_p_a);

  AVSC_DECLARE_FUNC(avs_get_env_property);

  AVSC_DECLARE_FUNC(avs_get_var_try);
  AVSC_DECLARE_FUNC(avs_get_var_bool);
  AVSC_DECLARE_FUNC(avs_get_var_int);
  AVSC_DECLARE_FUNC(avs_get_var_double);
  AVSC_DECLARE_FUNC(avs_get_var_string);
  AVSC_DECLARE_FUNC(avs_get_var_long);

  AVSC_DECLARE_FUNC(avs_pool_allocate);
  AVSC_DECLARE_FUNC(avs_pool_free);

  // V9
  AVSC_DECLARE_FUNC(avs_is_property_writable);
  AVSC_DECLARE_FUNC(avs_make_property_writable);

  // V10
  AVSC_DECLARE_FUNC(avs_video_frame_get_pixel_type);
  AVSC_DECLARE_FUNC(avs_video_frame_amend_pixel_type);

  // V10
  AVSC_DECLARE_FUNC(avs_is_channel_mask_known);
  AVSC_DECLARE_FUNC(avs_set_channel_mask);
  AVSC_DECLARE_FUNC(avs_get_channel_mask);

};

#undef AVSC_DECLARE_FUNC

#ifdef AVS26_FALLBACK_SIMULATION
// Helper functions for fallback simulation
// Avisynth+ extensions do not exist in classic Avisynth so they are simulated
AVSC_INLINE int avs_is_xx_fallback_return_false(const AVS_VideoInfo * p)
{
  return 0;
}

// Avisynth+ extensions do not exist in classic Avisynth so they are simulated
AVSC_INLINE int avs_num_components_fallback(const AVS_VideoInfo * p)
{
  switch (p->pixel_type) {
  case AVS_CS_UNKNOWN:
    return 0;
  case AVS_CS_RAW32:
  case AVS_CS_Y8:
    return 1;
  case AVS_CS_BGR32:
    return 4; // not planar but return the count
  default:
    return 3;
  }
}

// Avisynth+ extensions do not exist in classic Avisynth so they are simulated
AVSC_INLINE int avs_component_size_fallback(const AVS_VideoInfo * p)
{
  return 1;
}

// Avisynth+ extensions do not exist in classic Avisynth so they are simulated
AVSC_INLINE int avs_bits_per_component_fallback(const AVS_VideoInfo * p)
{
  return 8;
}
// End of helper functions for fallback simulation
#endif // AVS26_FALLBACK_SIMULATION

// avs_load_library() allocates an array for API procedure entries
// reads and fills the entries with live procedure addresses.
// AVSC_INLINE helpers which are calling into API procedures are not treated here (todo)

AVSC_INLINE AVS_Library * avs_load_library() {
  AVS_Library *library = (AVS_Library *)malloc(sizeof(AVS_Library));
  if (library == NULL)
    return NULL;
  library->handle = LoadLibraryA("avisynth");
  if (library->handle == NULL)
    goto fail;

#define __AVSC_STRINGIFY(x) #x
#define AVSC_STRINGIFY(x) __AVSC_STRINGIFY(x)
#define AVSC_LOAD_FUNC(name) {\
  library->name = (name##_func) GetProcAddress(library->handle, AVSC_STRINGIFY(name));\
  if (library->name == NULL)\
    goto fail;\
}

#ifdef AVS26_FALLBACK_SIMULATION
// When an API function is not loadable, let's try a replacement
// Missing Avisynth+ functions will be substituted with classic Avisynth compatible methods
/*
Avisynth+                 When method is missing (classic Avisynth)
avs_is_rgb48              constant false
avs_is_rgb64              constant false
avs_is_444                avs_is_yv24
avs_is_422                avs_is_yv16
avs_is_420                avs_is_yv12
avs_is_y                  avs_is_y8
avs_is_yuva               constant false
avs_is_planar_rgb         constant false
avs_is_planar_rgba        constant false
avs_num_components        special: avs_num_components_fake Y8:1 RGB32:4 else 3
avs_component_size        constant 1 (1 bytes/component)
avs_bits_per_component    constant 8 (8 bits/component)
*/

  // try to load an alternative function
#define AVSC_LOAD_FUNC_FALLBACK(name,name2) {\
  library->name = (name##_func) GetProcAddress(library->handle, AVSC_STRINGIFY(name));\
  if (library->name == NULL)\
    library->name = (name##_func) GetProcAddress(library->handle, AVSC_STRINGIFY(name2));\
  if (library->name == NULL)\
    goto fail;\
}

  // try to assign a replacement function
#define AVSC_LOAD_FUNC_FALLBACK_SIMULATED(name,name2) {\
  library->name = (name##_func) GetProcAddress(library->handle, AVSC_STRINGIFY(name));\
  if (library->name == NULL)\
    library->name = name2;\
  if (library->name == NULL)\
    goto fail;\
}
#endif // AVS26_FALLBACK_SIMULATION

  AVSC_LOAD_FUNC(avs_add_function);
  AVSC_LOAD_FUNC(avs_at_exit);
  AVSC_LOAD_FUNC(avs_bit_blt);
  AVSC_LOAD_FUNC(avs_check_version);
  AVSC_LOAD_FUNC(avs_clip_get_error);
  AVSC_LOAD_FUNC(avs_copy_clip);
  AVSC_LOAD_FUNC(avs_copy_value);
  AVSC_LOAD_FUNC(avs_copy_video_frame);
  AVSC_LOAD_FUNC(avs_create_script_environment);
  AVSC_LOAD_FUNC(avs_delete_script_environment);
  AVSC_LOAD_FUNC(avs_function_exists);
  AVSC_LOAD_FUNC(avs_get_audio);
  AVSC_LOAD_FUNC(avs_get_cpu_flags);
  AVSC_LOAD_FUNC(avs_get_frame);
  AVSC_LOAD_FUNC(avs_get_parity);
  AVSC_LOAD_FUNC(avs_get_var);
  AVSC_LOAD_FUNC(avs_get_version);
  AVSC_LOAD_FUNC(avs_get_video_info);
  AVSC_LOAD_FUNC(avs_invoke);
  AVSC_LOAD_FUNC(avs_make_writable);
  AVSC_LOAD_FUNC(avs_new_c_filter);
  AVSC_LOAD_FUNC(avs_new_video_frame_a);



  AVSC_LOAD_FUNC(avs_release_clip);
  AVSC_LOAD_FUNC(avs_release_value);
  AVSC_LOAD_FUNC(avs_release_video_frame);
  AVSC_LOAD_FUNC(avs_save_string);
  AVSC_LOAD_FUNC(avs_set_cache_hints);
  AVSC_LOAD_FUNC(avs_set_global_var);
  AVSC_LOAD_FUNC(avs_set_memory_max);
  AVSC_LOAD_FUNC(avs_set_to_clip);
  AVSC_LOAD_FUNC(avs_set_var);
  AVSC_LOAD_FUNC(avs_set_working_dir);
  AVSC_LOAD_FUNC(avs_sprintf);
  AVSC_LOAD_FUNC(avs_subframe);
  AVSC_LOAD_FUNC(avs_subframe_planar);
  AVSC_LOAD_FUNC(avs_take_clip);
  AVSC_LOAD_FUNC(avs_vsprintf);

  AVSC_LOAD_FUNC(avs_get_error);
  AVSC_LOAD_FUNC(avs_is_yv24);
  AVSC_LOAD_FUNC(avs_is_yv16);
  AVSC_LOAD_FUNC(avs_is_yv12);
  AVSC_LOAD_FUNC(avs_is_yv411);
  AVSC_LOAD_FUNC(avs_is_y8);
  AVSC_LOAD_FUNC(avs_is_color_space);

  AVSC_LOAD_FUNC(avs_get_plane_width_subsampling);
  AVSC_LOAD_FUNC(avs_get_plane_height_subsampling);
  AVSC_LOAD_FUNC(avs_bits_per_pixel);
  AVSC_LOAD_FUNC(avs_bytes_from_pixels);
  AVSC_LOAD_FUNC(avs_row_size);
  AVSC_LOAD_FUNC(avs_bmp_size);
  AVSC_LOAD_FUNC(avs_get_pitch_p);
  AVSC_LOAD_FUNC(avs_get_row_size_p);
  AVSC_LOAD_FUNC(avs_get_height_p);
  AVSC_LOAD_FUNC(avs_get_read_ptr_p);
  AVSC_LOAD_FUNC(avs_is_writable);
  AVSC_LOAD_FUNC(avs_get_write_ptr_p);

  // Avisynth+ specific
#ifdef AVS26_FALLBACK_SIMULATION
  // replace with fallback fn when does not exist
  AVSC_LOAD_FUNC_FALLBACK_SIMULATED(avs_is_rgb48, avs_is_xx_fallback_return_false);
  AVSC_LOAD_FUNC_FALLBACK_SIMULATED(avs_is_rgb64, avs_is_xx_fallback_return_false);
  AVSC_LOAD_FUNC_FALLBACK(avs_is_444, avs_is_yv24);
  AVSC_LOAD_FUNC_FALLBACK(avs_is_422, avs_is_yv16);
  AVSC_LOAD_FUNC_FALLBACK(avs_is_420, avs_is_yv12);
  AVSC_LOAD_FUNC_FALLBACK(avs_is_y, avs_is_y8);
  AVSC_LOAD_FUNC_FALLBACK_SIMULATED(avs_is_yuva, avs_is_xx_fallback_return_false);
  AVSC_LOAD_FUNC_FALLBACK_SIMULATED(avs_is_planar_rgb, avs_is_xx_fallback_return_false);
  AVSC_LOAD_FUNC_FALLBACK_SIMULATED(avs_is_planar_rgba, avs_is_xx_fallback_return_false);
  AVSC_LOAD_FUNC_FALLBACK_SIMULATED(avs_num_components, avs_num_components_fallback);
  AVSC_LOAD_FUNC_FALLBACK_SIMULATED(avs_component_size, avs_component_size_fallback);
  AVSC_LOAD_FUNC_FALLBACK_SIMULATED(avs_bits_per_component, avs_bits_per_component_fallback);
#else
  // Avisynth+ specific
  AVSC_LOAD_FUNC(avs_is_rgb48);
  AVSC_LOAD_FUNC(avs_is_rgb64);
  AVSC_LOAD_FUNC(avs_is_444);
  AVSC_LOAD_FUNC(avs_is_422);
  AVSC_LOAD_FUNC(avs_is_420);
  AVSC_LOAD_FUNC(avs_is_y);
  AVSC_LOAD_FUNC(avs_is_yuva);
  AVSC_LOAD_FUNC(avs_is_planar_rgb);
  AVSC_LOAD_FUNC(avs_is_planar_rgba);
  AVSC_LOAD_FUNC(avs_num_components);
  AVSC_LOAD_FUNC(avs_component_size);
  AVSC_LOAD_FUNC(avs_bits_per_component);
#endif
  // Avisynth+ interface V8, no backward compatible simulation
  AVSC_LOAD_FUNC(avs_subframe_planar_a);
  // frame properties
  AVSC_LOAD_FUNC(avs_copy_frame_props);
  AVSC_LOAD_FUNC(avs_get_frame_props_ro);
  AVSC_LOAD_FUNC(avs_get_frame_props_rw);
  AVSC_LOAD_FUNC(avs_prop_num_keys);
  AVSC_LOAD_FUNC(avs_prop_get_key);
  AVSC_LOAD_FUNC(avs_prop_num_elements);
  AVSC_LOAD_FUNC(avs_prop_get_type);
  AVSC_LOAD_FUNC(avs_prop_get_int);
  AVSC_LOAD_FUNC(avs_prop_get_float);
  AVSC_LOAD_FUNC(avs_prop_get_data);
  AVSC_LOAD_FUNC(avs_prop_get_data_size);
  AVSC_LOAD_FUNC(avs_prop_get_clip);
  AVSC_LOAD_FUNC(avs_prop_get_frame);
  AVSC_LOAD_FUNC(avs_prop_delete_key);
  AVSC_LOAD_FUNC(avs_prop_set_int);
  AVSC_LOAD_FUNC(avs_prop_set_float);
  AVSC_LOAD_FUNC(avs_prop_set_data);
  AVSC_LOAD_FUNC(avs_prop_set_clip);
  AVSC_LOAD_FUNC(avs_prop_set_frame);

  AVSC_LOAD_FUNC(avs_prop_get_int_array);
  AVSC_LOAD_FUNC(avs_prop_get_float_array);
  AVSC_LOAD_FUNC(avs_prop_set_int_array);
  AVSC_LOAD_FUNC(avs_prop_set_float_array);

  AVSC_LOAD_FUNC(avs_clear_map);

  // NewVideoFrame with frame properties
  AVSC_LOAD_FUNC(avs_new_video_frame_p);
  AVSC_LOAD_FUNC(avs_new_video_frame_p_a);

  AVSC_LOAD_FUNC(avs_get_env_property);

  AVSC_LOAD_FUNC(avs_get_var_try);
  AVSC_LOAD_FUNC(avs_get_var_bool);
  AVSC_LOAD_FUNC(avs_get_var_int);
  AVSC_LOAD_FUNC(avs_get_var_double);
  AVSC_LOAD_FUNC(avs_get_var_string);
  AVSC_LOAD_FUNC(avs_get_var_long);

  AVSC_LOAD_FUNC(avs_pool_allocate);
  AVSC_LOAD_FUNC(avs_pool_free);

  // V9
  AVSC_LOAD_FUNC(avs_make_property_writable);
  AVSC_LOAD_FUNC(avs_is_property_writable);

  // V10
  AVSC_LOAD_FUNC(avs_video_frame_get_pixel_type);
  AVSC_LOAD_FUNC(avs_video_frame_amend_pixel_type);

  // V10
  AVSC_LOAD_FUNC(avs_is_channel_mask_known);
  AVSC_LOAD_FUNC(avs_set_channel_mask);
  AVSC_LOAD_FUNC(avs_get_channel_mask);

#undef __AVSC_STRINGIFY
#undef AVSC_STRINGIFY
#undef AVSC_LOAD_FUNC
#undef AVSC_LOAD_FUNC_FALLBACK
#undef AVSC_LOAD_FUNC_FALLBACK_SIMULATED

  return library;

fail:
  free(library);
  return NULL;
}

AVSC_INLINE void avs_free_library(AVS_Library *library) {
  if (library == NULL)
    return;
  FreeLibrary(library->handle);
  free(library);
}
#endif

#endif // AVS_WINDOWS

#endif
