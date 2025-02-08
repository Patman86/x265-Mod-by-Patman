/*****************************************************************************
 * avs.c: avisynth input
 *****************************************************************************
 * Copyright (C) 2020 Xinyue Lu, 2025 Avisynth developers
 *
 * Authors: Xinyue Lu <i@7086.in>
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
 *****************************************************************************/

#include "avs.h"

#define FAIL_IF_ERROR( cond, ... )\
if( cond )\
{\
    general_log( NULL, "avs+", X265_LOG_ERROR, __VA_ARGS__ );\
    b_fail = true;\
    return;\
}

using namespace X265_NS;

const int AVSInput::avs_planes_packed[1] = { 0 };
const int AVSInput::avs_planes_grey[1] = { AVS_PLANAR_Y };
const int AVSInput::avs_planes_yuv[3] = { AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V };
const int AVSInput::avs_planes_rgb[3] = { AVS_PLANAR_G, AVS_PLANAR_B, AVS_PLANAR_R };
const int AVSInput::avs_planes_yuva[4] = { AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V, AVS_PLANAR_A };
const int AVSInput::avs_planes_rgba[4] = { AVS_PLANAR_G, AVS_PLANAR_B, AVS_PLANAR_R, AVS_PLANAR_A };

lib_path_t AVSInput::convertLibraryPath(std::string path)
{
#if defined(_WIN32)
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &path[0], (int)path.size(), NULL, 0);
    std::wstring wstrTo( size_needed, 0 );
    MultiByteToWideChar(CP_UTF8, 0, &path[0], (int)path.size(), &wstrTo[0], size_needed);
    return wstrTo;
#else
    return path;
#endif
}

void AVSInput::parseAvsOptions(const char* _options)
{
    std::string options {_options}; options += ";";
    std::string optSeparator {";"};
    std::string valSeparator {"="};
    std::map<std::string, int> knownOptions {
        {std::string {"library"}, 1}
    };

    auto start = 0U;
    auto end = options.find(optSeparator);

    while ((end = options.find(optSeparator, start)) != std::string::npos)
    {
        auto option = options.substr(start, end - start);
        auto valuePos = option.find(valSeparator);
        if (valuePos != std::string::npos)
        {
            auto key = option.substr(0U, valuePos);
            auto value = option.substr(valuePos + 1, option.length());
            switch (knownOptions[key])
            {
            case 1:
                avs_library_path = convertLibraryPath(value);
                general_log(nullptr, "avs+", X265_LOG_INFO, "using external Avisynth library from: \"%s\" \n", value.c_str());
                break;
            }
        }
        else if (option.length() > 0)
        {
            general_log(nullptr, "avs+", X265_LOG_ERROR, "invalid option \"%s\" ignored\n", option.c_str());
        }
        start = end + optSeparator.length();
        end = options.find(optSeparator, start);
    }
}

void AVSInput::load_avs()
{
    avs_open();
    if (!h->library)
        return;
    LOAD_AVS_FUNC(avs_clip_get_error);
    LOAD_AVS_FUNC(avs_create_script_environment);
    LOAD_AVS_FUNC(avs_delete_script_environment);
    LOAD_AVS_FUNC(avs_get_frame);
    LOAD_AVS_FUNC(avs_get_version);
    LOAD_AVS_FUNC(avs_get_video_info);
    LOAD_AVS_FUNC(avs_function_exists);
    LOAD_AVS_FUNC(avs_invoke);
    LOAD_AVS_FUNC(avs_release_clip);
    LOAD_AVS_FUNC(avs_release_value);
    LOAD_AVS_FUNC(avs_release_video_frame);
    LOAD_AVS_FUNC(avs_take_clip);
    LOAD_AVS_FUNC(avs_is_color_space);
    LOAD_AVS_FUNC(avs_bit_blt);
    LOAD_AVS_FUNC(avs_is_y);
    LOAD_AVS_FUNC(avs_is_420);
    LOAD_AVS_FUNC(avs_is_422);
    LOAD_AVS_FUNC(avs_is_444);
    LOAD_AVS_FUNC(avs_is_rgb48);
    LOAD_AVS_FUNC(avs_is_rgb64);
    LOAD_AVS_FUNC(avs_bits_per_component);
    LOAD_AVS_FUNC(avs_get_height_p);
    LOAD_AVS_FUNC(avs_get_pitch_p);
    LOAD_AVS_FUNC(avs_get_read_ptr_p);
    LOAD_AVS_FUNC(avs_get_row_size_p);
    h->env = h->func.avs_create_script_environment(AVS_INTERFACE_26);
    return;
fail:
    avs_close();
}

void AVSInput::info_avs()
{
    if (!h->func.avs_function_exists(h->env, "VersionString"))
        return;
    AVS_Value ver = h->func.avs_invoke(h->env, "VersionString", avs_new_value_array(NULL, 0), NULL);
    if (avs_is_error(ver))
        return;
    if (!avs_is_string(ver))
        return;
    const char *version = avs_as_string(ver);
    h->func.avs_release_value(ver);
    general_log(NULL, "avs+", X265_LOG_INFO, "%s\n", version);
}

void AVSInput::openfile(InputFileInfo& info)
{
    if (info.skipFrames)
    {
        h->next_frame = info.skipFrames;
    }
    AVS_Value res = h->func.avs_invoke(h->env, "Import", avs_new_value_string(info.filename), NULL);
    FAIL_IF_ERROR(avs_is_error(res), "Error loading file: %s\n", avs_as_string(res));
    FAIL_IF_ERROR(!avs_is_clip(res), "File didn't return a video clip\n");
    h->clip = h->func.avs_take_clip(res, h->env);
    h->vi = h->func.avs_get_video_info(h->clip);
    info.width = h->vi->width;
    info.height = h->vi->height;
    info.fpsNum = h->vi->fps_numerator;
    info.fpsDenom = h->vi->fps_denominator;
    info.frameCount = h->vi->num_frames;
    info.depth = h->func.avs_bits_per_component(h->vi);

    int planar = 1; // 0: packed, 1: YUV, 2: Y8, 3: Planar RGB, 4: YUVA, 5: Planar RGBA

    h->plane_count = 3;
    if (h->func.avs_is_y(h->vi))
    {
        h->plane_count = 1;
        planar = 2;
        info.csp = X265_CSP_I400;
    }
    else if (h->func.avs_is_420(h->vi))
    {
        planar = 1;
        info.csp = X265_CSP_I420;
    }
    else if (h->func.avs_is_422(h->vi))
    {
        planar = 1;
        info.csp = X265_CSP_I422;
    }
    else if (h->func.avs_is_444(h->vi))
    {
        planar = 1;
        info.csp = X265_CSP_I444;
    }
    /* RGB not supported at the moment by x265 */
    /*
    else if (avs_is_rgb24(h->vi) || h->func.avs_is_rgb48(h->vi))
    {
      planar = 0;
      info.csp = X265_CSP_BGR;
    }
    else if (avs_is_rgb32(h->vi) || h->func.avs_is_rgb64(h->vi))
    {
      planar = 0;
      info.csp = X265_CSP_BGRA;
    }
    */
    else
    {
        FAIL_IF_ERROR(1, "Video colorspace is not supported\n");
    }

    // some of these are not supported, 
    // still, keep their code for RFU
    switch (planar) {
    case 5: // Planar RGB + Alpha
      h->plane_count = 4;
      h->planes = avs_planes_rgba;
      break;
    case 4: // YUV + Alpha
      h->plane_count = 4;
      h->planes = avs_planes_yuva;
      break;
    case 3: // Planar RGB
      h->plane_count = 3;
      h->planes = avs_planes_rgb;
      break;
    case 2: // Y8
      h->plane_count = 1;
      h->planes = avs_planes_grey;
      break;
    case 1: // YUV
      h->plane_count = 3;
      h->planes = avs_planes_yuv;
      break;
    default:
      h->plane_count = 1;
      h->planes = avs_planes_packed;
    }

}

void AVSInput::release()
{
    if (h->clip)
        h->func.avs_release_clip(h->clip);
    if (h->env)
        h->func.avs_delete_script_environment(h->env);
    if (h->library)
        avs_close();
}

bool AVSInput::readPicture(x265_picture& pic)
{
    AVS_VideoFrame *frm = h->func.avs_get_frame(h->clip, h->next_frame);
    const char *err = h->func.avs_clip_get_error(h->clip);
    if (err)
    {
        general_log(NULL, "avs+", X265_LOG_ERROR, "%s occurred while reading frame %d\n", err, h->next_frame);
        b_fail = true;
        return false;
    }
    pic.width = _info.width;
    pic.height = _info.height;
    if (frame_size == 0 || frame_buffer == nullptr)
    {
        frame_size = 0;
        for (int i = 0; i < h->plane_count; i++) {
            const int plane = h->planes[i];
            const int rowsize = h->func.avs_get_row_size_p(frm, plane);
            const int planeheight = h->func.avs_get_height_p(frm, plane);
            const int target_pitch = rowsize;
            // rowsize instead of source pitch:
            // - pitch can be much larger than needed
            // - can vary frame by frame, we should allocate a constant size
            frame_size += target_pitch * planeheight;
        }
        frame_buffer = reinterpret_cast<uint8_t*>(x265_malloc(frame_size));
    }
    pic.framesize = frame_size;
    uint8_t* ptr = frame_buffer;

    for (int i = 0; i < h->plane_count; i++) {
      const int plane = h->planes[i];
      pic.planes[i] = ptr;

      const uint8_t *src_p = h->func.avs_get_read_ptr_p(frm, plane);
      int pitch = h->func.avs_get_pitch_p(frm, plane);
      const int rowsize = h->func.avs_get_row_size_p(frm, plane);
      const int planeheight = h->func.avs_get_height_p(frm, plane);

      // Flip RGB video.
      /* RGB not supported at the moment in x265 */
      if (h->func.avs_is_color_space(h->vi, AVS_CS_BGR) ||
          h->func.avs_is_color_space(h->vi, AVS_CS_BGR48) ||
          h->func.avs_is_color_space(h->vi, AVS_CS_BGR64)) {
          src_p = src_p + (planeheight - 1) * pitch;
          pitch = -pitch;
      }

      const int target_pitch = rowsize; // like above
      pic.stride[i] = target_pitch;

      h->func.avs_bit_blt(h->env, ptr, target_pitch, src_p, pitch,
          rowsize, planeheight);
      ptr += target_pitch * planeheight;

    }
    pic.colorSpace = _info.csp;
    pic.bitDepth = _info.depth;
    h->func.avs_release_video_frame(frm);
    h->next_frame++;
    return true;
}
