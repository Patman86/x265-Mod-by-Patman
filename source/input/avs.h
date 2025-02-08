#ifndef X265_AVS_H
#define X265_AVS_H

#include "input.h"
#include <string>
#include <map>
#include <avisynth/avisynth_c.h>

#if _WIN32
#include <windows.h>
using lib_path_t = std::wstring;
using lib_t = HMODULE;
using func_t = FARPROC;
#else
#include <dlfcn.h>
using lib_path_t = std::string;
using lib_t = void*;
using func_t = void*;
#define __stdcall
#endif

#define AVS_INTERFACE_26 6

#define LOAD_AVS_FUNC(name) \
{\
	h->func.name = reinterpret_cast<decltype(h->func.name)>((void*)avs_address(#name));\
	if (!h->func.name) goto fail;\
}

namespace X265_NS {
// x265 private namespace

typedef struct
{
    AVS_Clip *clip;
    AVS_ScriptEnvironment *env;
    const AVS_VideoInfo* vi;
    lib_t library;
    /* declare function pointers for the utilized functions to be loaded without __declspec,
       as the avisynth header does not compensate for this type of usage */
    struct
    {
        const char *(__stdcall *avs_clip_get_error)( AVS_Clip *clip );
        AVS_ScriptEnvironment *(__stdcall *avs_create_script_environment)( int version );
        void (__stdcall *avs_delete_script_environment)( AVS_ScriptEnvironment *env );
        AVS_VideoFrame *(__stdcall *avs_get_frame)( AVS_Clip *clip, int n );
        int (__stdcall *avs_get_version)( AVS_Clip *clip );
        const AVS_VideoInfo *(__stdcall *avs_get_video_info)( AVS_Clip *clip );
        int (__stdcall *avs_function_exists)( AVS_ScriptEnvironment *env, const char *name );
        AVS_Value (__stdcall *avs_invoke)( AVS_ScriptEnvironment *env, const char *name,
        AVS_Value args, const char **arg_names );
        void (__stdcall *avs_release_clip)( AVS_Clip *clip );
        void (__stdcall *avs_release_value)( AVS_Value value );
        void (__stdcall *avs_release_video_frame)( AVS_VideoFrame *frame );
        AVS_Clip *(__stdcall *avs_take_clip)( AVS_Value, AVS_ScriptEnvironment *env );
        void (__stdcall *avs_bit_blt)(AVS_ScriptEnvironment*, BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height);
        int (__stdcall *avs_is_color_space)(const AVS_VideoInfo* p, int c_space);
        int (__stdcall *avs_is_y)(const AVS_VideoInfo * p);
        int (__stdcall *avs_is_420)(const AVS_VideoInfo * p);
        int (__stdcall *avs_is_422)(const AVS_VideoInfo * p);
        int (__stdcall *avs_is_444)(const AVS_VideoInfo * p);
        int (__stdcall *avs_is_rgb48)(const AVS_VideoInfo* p);
        int (__stdcall *avs_is_rgb64)(const AVS_VideoInfo* p);
        int (__stdcall *avs_is_planar_rgb)(const AVS_VideoInfo* p);
        int (__stdcall *avs_is_planar_rgba)(const AVS_VideoInfo* p);
        int (__stdcall *avs_bits_per_component)(const AVS_VideoInfo * p);
        int (__stdcall *avs_get_pitch_p)(const AVS_VideoFrame* p, int plane);
        int (__stdcall *avs_get_row_size_p)(const AVS_VideoFrame* p, int plane);
        int (__stdcall *avs_get_height_p)(const AVS_VideoFrame* p, int plane);
        const BYTE *(__stdcall *avs_get_read_ptr_p)(const AVS_VideoFrame* p, int plane);
    } func;
    int next_frame;
    int plane_count;
    const int* planes;
} avs_hnd_t;

class AVSInput : public InputFile
{
protected:

    bool b_fail {false};
    bool b_eof {false};
    avs_hnd_t handle;
    avs_hnd_t *h;
    size_t frame_size {0};
    uint8_t *frame_buffer {nullptr};
    InputFileInfo _info;
    void load_avs();
    void info_avs();
    void openfile(InputFileInfo& info);
    #if _WIN32
        lib_path_t avs_library_path {L"avisynth"};
        void avs_open() { h->library = LoadLibraryW(avs_library_path.c_str()); }
        void avs_close() { FreeLibrary(h->library); h->library = nullptr; }
        func_t avs_address(LPCSTR func) { return GetProcAddress(h->library, func); }
    #else
        #ifdef __MACH__
            lib_path_t avs_library_path {"libavisynth.dylib"};
        #else
            lib_path_t avs_library_path {"libavisynth.so"};
        #endif
        void avs_open() { h->library = dlopen(avs_library_path.c_str(), RTLD_NOW); }
        void avs_close() { dlclose(h->library); h->library = nullptr; }
        func_t avs_address(const char * func) { return dlsym(h->library, func); }
    #endif
    lib_path_t convertLibraryPath(std::string);
    void parseAvsOptions(const char* _options);

public:
    static const int avs_planes_packed[1];
    static const int avs_planes_grey[1];
    static const int avs_planes_yuv[3];
    static const int avs_planes_rgb[3];
    static const int avs_planes_yuva[4];
    static const int avs_planes_rgba[4];

    AVSInput(InputFileInfo& info)
    {
        h = &handle;
        memset(h, 0, sizeof(handle));
        if (info.readerOpts)
            parseAvsOptions(info.readerOpts);
        load_avs();
        info_avs();
        if (!h->library)
        {
            b_fail = true;
            return;
        }
        openfile(info);
        _info = info;
    }
    ~AVSInput() {}
    void release();
    bool isEof() const                            { return h->next_frame >= _info.frameCount; }
    bool isFail()                                 { return b_fail; }
    void startReader() {}
    void stopReader() {}
    bool readPicture(x265_picture&);

    const char *getName() const                   { return "avs+"; }

    int getWidth() const                          { return _info.width; }

    int getHeight() const                         { return _info.height; }
};
}

#endif // ifndef X265_AVS_H
