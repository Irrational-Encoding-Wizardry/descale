/* 
 * Copyright © 2022 Frechdachs <frechdachs@rekt.cc>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <avisynth/avisynth_c.h>
#include "descale.h"
#include "plugin.h"


struct AVSDescaleData
{
    bool initialized;
    pthread_mutex_t lock;

    struct DescaleData dd;
};


static AVS_VideoFrame * AVSC_CC avs_descale_get_frame(AVS_FilterInfo *fi, int n)
{
    struct AVSDescaleData *d = (struct AVSDescaleData *)fi->user_data;

    if (!d->initialized) {
        pthread_mutex_lock(&d->lock);
        if (!d->initialized) {
            initialize_descale_data(&d->dd);
            d->initialized = true;
        }
        pthread_mutex_unlock(&d->lock);
    }

    // What the fuck is this shit?! Why not just index the planes with 0, 1, 2?
    int planes_rgb[] = {AVS_PLANAR_R, AVS_PLANAR_G, AVS_PLANAR_B};
    int planes_yuv[] = {AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V};
    int *planes;
    if (avs_is_planar_rgb(&fi->vi))
        planes = planes_rgb;
    else
        planes = planes_yuv;

    AVS_VideoFrame *src = avs_get_frame(fi->child, n);
    // Using avs_new_video_frame_p_a() instead (to copy frame properties) crashes AviSynth ???
    // So we manually copy the frame properties later.
    AVS_VideoFrame *dst = avs_new_video_frame_a(fi->env, &fi->vi, 32);

    for (int i = 0; i < d->dd.num_planes; i++) {
        int plane = planes[i];
        int src_stride = avs_get_pitch_p(src, plane) / sizeof (float);
        int dst_stride = avs_get_pitch_p(dst, plane) / sizeof (float);
        const float *srcp = (const float *)avs_get_read_ptr_p(src, plane);
        float *dstp = (float *)avs_get_write_ptr_p(dst, plane);

        if (d->dd.process_h && d->dd.process_v) {
            int intermediate_stride = avs_get_pitch_p(dst, plane);
            float *intermediatep = avs_pool_allocate(fi->env, intermediate_stride * d->dd.src_height * sizeof (float), 32, AVS_ALLOCTYPE_POOLED_ALLOC);

            d->dd.dsapi.process_vectors(d->dd.dscore_h[i && d->dd.subsampling_h], DESCALE_DIR_HORIZONTAL, d->dd.src_height >> (i ? d->dd.subsampling_v : 0), src_stride, intermediate_stride, srcp, intermediatep);
            d->dd.dsapi.process_vectors(d->dd.dscore_v[i && d->dd.subsampling_v], DESCALE_DIR_VERTICAL, d->dd.dst_width >> (i ? d->dd.subsampling_h : 0), intermediate_stride, dst_stride, intermediatep, dstp);

            avs_pool_free(fi->env, intermediatep);

        } else if (d->dd.process_h) {
            d->dd.dsapi.process_vectors(d->dd.dscore_h[i && d->dd.subsampling_h], DESCALE_DIR_HORIZONTAL, d->dd.src_height >> (i ? d->dd.subsampling_v : 0), src_stride, dst_stride, srcp, dstp);

        } else if (d->dd.process_v) {
            d->dd.dsapi.process_vectors(d->dd.dscore_v[i && d->dd.subsampling_v], DESCALE_DIR_VERTICAL, d->dd.src_width >> (i ? d->dd.subsampling_h : 0), src_stride, dst_stride, srcp, dstp);
        }
    }

    avs_copy_frame_props(fi->env, src, dst);
    avs_release_video_frame(src);

    return dst;
}


static int AVSC_CC avs_descale_set_cache_hints(AVS_FilterInfo *fi, int cachehints, int frame_range)
{
    return cachehints == AVS_CACHE_GET_MTMODE ? 1 /* MT_NICE_FILTER */ : 0;
}


static void AVSC_CC avs_descale_free(AVS_FilterInfo *fi)
{
    struct AVSDescaleData *d = (struct AVSDescaleData *)fi->user_data;

    if (d->initialized) {
        if (d->dd.process_h) {
            d->dd.dsapi.free_core(d->dd.dscore_h[0]);
            if (d->dd.num_planes > 1 && d->dd.subsampling_h > 0)
                d->dd.dsapi.free_core(d->dd.dscore_h[1]);
        }
        if (d->dd.process_v) {
            d->dd.dsapi.free_core(d->dd.dscore_v[0]);
            if (d->dd.num_planes > 1 && d->dd.subsampling_v > 0)
                d->dd.dsapi.free_core(d->dd.dscore_v[1]);
        }
    }

    pthread_mutex_destroy(&d->lock);

    free(d);
}


static AVS_Value AVSC_CC avs_descale_create(AVS_ScriptEnvironment *env, AVS_Value args, void *user_data)
{
    AVS_Value v;
    AVS_Value c1 = avs_array_elt(args, 0);

    AVS_Clip *clip = avs_take_clip(c1, env);
    //avs_release_value(c1);  // This is apparently wrong, leaving it uncommented crashes sometimes
    const AVS_VideoInfo *vi = avs_get_video_info(clip);

    if (!avs_has_video(vi)) {
        v = avs_new_value_error("Descale: Input clip must have video.");
        goto done;
    }

    if (!avs_is_y(vi) && !avs_is_yuv(vi) && !avs_is_planar_rgb(vi)) {
        v = avs_new_value_error("Descale: Input clip must be Y, YUV, or planar RGB.");
        goto done;
    }

    int num_planes = avs_num_components(vi);

    // Apparently is_yuv() returns true for Y input.
    int subsampling_h = avs_is_yuv(vi) && !avs_is_y(vi) ? avs_get_plane_width_subsampling(vi, AVS_PLANAR_U) : 0;
    int subsampling_v = avs_is_yuv(vi) && !avs_is_y(vi) ? avs_get_plane_height_subsampling(vi, AVS_PLANAR_U) : 0;

    int src_width = vi->width;
    int src_height = vi->height;

    int idx = 1;  // This is incredibly stupid, but apparently we can't get filter arguments by name
                  // which can get problematic if argument index is not the same among different filters

    v = avs_array_elt(args, idx++);
    int dst_width = avs_as_int(v);
    v = avs_array_elt(args, idx++);
    int dst_height = avs_as_int(v);

    if (dst_width < 1) {
        v = avs_new_value_error("Descale: width must be greater than 0.");
        goto done;
    }
    if (dst_height < 8) {
        v = avs_new_value_error("Descale: Output height must be greater than or equal to 8.");
        goto done;
    }

    if (dst_width % (1 << subsampling_h) != 0) {
        v = avs_new_value_error("Descale: Output width and output subsampling are not compatible.");
        goto done;
    }
    if (dst_height % (1 << subsampling_v) != 0) {
        v = avs_new_value_error("Descale: Output height and output subsampling are not compatible.");
        goto done;
    }

    if (dst_width > src_width || dst_height > src_height) {
        v = avs_new_value_error("Descale: Output dimension must be less than or equal to input dimension.");
        goto done;
    }

    enum DescaleMode mode;
    const char *kernel = NULL;
    if (user_data == NULL) {
        v = avs_array_elt(args, idx++);
        if (!avs_defined(v)) {
            v = avs_new_value_error("Descale: kernel is a required argument.");
            goto done;
        }
        kernel = avs_as_string(v);
        if (string_is_equal_ignore_case(kernel, "bilinear"))
            mode = DESCALE_MODE_BILINEAR;
        else if (string_is_equal_ignore_case(kernel, "bicubic"))
            mode = DESCALE_MODE_BICUBIC;
        else if (string_is_equal_ignore_case(kernel, "lanczos"))
            mode = DESCALE_MODE_LANCZOS;
        else if (string_is_equal_ignore_case(kernel, "spline16"))
            mode = DESCALE_MODE_SPLINE16;
        else if (string_is_equal_ignore_case(kernel, "spline36"))
            mode = DESCALE_MODE_SPLINE36;
        else if (string_is_equal_ignore_case(kernel, "spline64"))
            mode = DESCALE_MODE_SPLINE64;
        else {
            v = avs_new_value_error("Descale: Invalid kernel specified.");
            goto done;
        }

    } else {
        mode = (enum DescaleMode)user_data;
        if (mode == DESCALE_MODE_BILINEAR)
            kernel = "bilinear";
        else if (mode == DESCALE_MODE_BICUBIC)
            kernel = "bicubic";
        else if (mode == DESCALE_MODE_LANCZOS)
            kernel = "lanczos";
        else if (mode == DESCALE_MODE_SPLINE16)
            kernel = "spline16";
        else if (mode == DESCALE_MODE_SPLINE36)
            kernel = "spline36";
        else if (mode == DESCALE_MODE_SPLINE64)
            kernel = "spline64";
    }

    int taps;
    if (user_data == NULL || mode == DESCALE_MODE_LANCZOS) {
        v = avs_array_elt(args, idx++);
        taps = avs_defined(v) ? avs_as_int(v) : 3;
        if (mode == DESCALE_MODE_LANCZOS && taps < 1) {
            v = avs_new_value_error("Descale: taps must be greater than 0.");
            goto done;
        }
    } else {
        taps = 3;
    }

    double b;
    double c;
    if (user_data == NULL || mode == DESCALE_MODE_BICUBIC) {
        v = avs_array_elt(args, idx++);
        b = avs_defined(v) ? avs_as_float(v) : 0.0;
        v = avs_array_elt(args, idx++);
        c = avs_defined(v) ? avs_as_float(v) : 0.5;
    } else {
        b = 0.0;
        c = 0.5;
    }

    v = avs_array_elt(args, idx++);
    double shift_h = avs_defined(v) ? avs_as_float(v) : 0.0;
    v = avs_array_elt(args, idx++);
    double shift_v = avs_defined(v) ? avs_as_float(v) : 0.0;
    v = avs_array_elt(args, idx++);
    double active_width = avs_defined(v) ? avs_as_float(v) : (double)dst_width;
    v = avs_array_elt(args, idx++);
    double active_height = avs_defined(v) ? avs_as_float(v) : (double)dst_height;

    bool process_h = dst_width != src_width || shift_h != 0.0 || active_width != (double)dst_width;
    bool process_v = dst_height != src_height || shift_v != 0.0 || active_height != (double)dst_height;

    if (!process_h && !process_v) {
        v = avs_new_value_clip(clip);
        goto done;
    }

    v = avs_array_elt(args, idx++);
    int border_handling = avs_defined(v) ? avs_as_int(v) : 0;
    enum DescaleBorder border_handling_enum;
    if (border_handling == 1)
        border_handling_enum = DESCALE_BORDER_ZERO;
    if (border_handling == 2)
        border_handling_enum = DESCALE_BORDER_REPEAT;
    else
        border_handling_enum = DESCALE_BORDER_MIRROR;

    v = avs_array_elt(args, idx++);
    int opt = avs_defined(v) ? avs_as_int(v) : 0;
    enum DescaleOpt opt_enum;
    if (opt == 1)
        opt_enum = DESCALE_OPT_NONE;
    else if (opt == 2)
        opt_enum = DESCALE_OPT_AVX2;
    else
        opt_enum = DESCALE_OPT_AUTO;

    int bits_per_pixel = avs_bits_per_component(vi);
    if (bits_per_pixel != 32) {
        AVS_Value c2, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12;
        c1 = avs_new_value_clip(clip);
        avs_release_clip(clip);
        a1 = avs_new_value_int(32);
        AVS_Value convert_args[] = {c1, a1};
        c2 = avs_invoke(env, "ConvertBits", avs_new_value_array(convert_args, 2), NULL);
        avs_release_value(c1);
        a1 = avs_new_value_int(dst_width);
        a2 = avs_new_value_int(dst_height);
        a3 = avs_new_value_string(kernel);
        a4 = avs_new_value_int(taps);
        a5 = avs_new_value_float(b);
        a6 = avs_new_value_float(c);
        a7 = avs_new_value_float(shift_h);
        a8 = avs_new_value_float(shift_v);
        a9 = avs_new_value_float(active_width);
        a10 = avs_new_value_float(active_height);
        a11 = avs_new_value_int(border_handling);
        a12 = avs_new_value_int(opt);
        AVS_Value descale_args[] = {c2, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12};
        c1 = avs_invoke(env, "Descale", avs_new_value_array(descale_args, 13), NULL);
        avs_release_value(c2);
        a1 = avs_new_value_int(bits_per_pixel);
        AVS_Value convert_args2[] = {c1, a1};
        c2 = avs_invoke(env, "ConvertBits", avs_new_value_array(convert_args2, 2), NULL);
        avs_release_value(c1);
        return c2;
    }

    struct DescaleParams params = {mode, taps, b, c, 0, 0, border_handling_enum};
    struct DescaleData dd = {
        src_width, src_height,
        dst_width, dst_height,
        subsampling_h, subsampling_v,
        num_planes,
        process_h, process_v,
        shift_h, shift_v,
        active_width, active_height,
        get_descale_api(opt_enum),
        params,
        {NULL, NULL},
        {NULL, NULL}
    };

    struct AVSDescaleData *data = calloc(1, sizeof (struct AVSDescaleData));
    data->dd = dd;
    pthread_mutex_init(&data->lock, NULL);

    c1 = avs_new_value_clip(clip);
    avs_release_clip(clip);
    AVS_FilterInfo *fi;
    clip = avs_new_c_filter(env, &fi, c1, true);
    avs_release_value(c1);

    fi->vi.width = dst_width;
    fi->vi.height = dst_height;
    fi->user_data = data;
    fi->get_frame = &avs_descale_get_frame;
    fi->set_cache_hints = &avs_descale_set_cache_hints;
    fi->free_filter = &avs_descale_free;

    v = avs_new_value_clip(clip);

done:
    avs_release_clip(clip);
    return v;
}


const char * AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment *env)
{
    avs_add_function(
        env,
        "Debilinear",
        "c"
        "i"
        "i"
        "[src_left]f"
        "[src_top]f"
        "[src_width]f"
        "[src_height]f"
        "[border_handling]i"
        "[opt]i",
        avs_descale_create,
        (void *)(DESCALE_MODE_BILINEAR)
    );

    avs_add_function(
        env,
        "Debicubic",
        "c"
        "i"
        "i"
        "[b]f"
        "[c]f"
        "[src_left]f"
        "[src_top]f"
        "[src_width]f"
        "[src_height]f"
        "[border_handling]i"
        "[opt]i",
        avs_descale_create,
        (void *)(DESCALE_MODE_BICUBIC)
    );

    avs_add_function(
        env,
        "Delanczos",
        "c"
        "i"
        "i"
        "[taps]i"
        "[src_left]f"
        "[src_top]f"
        "[src_width]f"
        "[src_height]f"
        "[border_handling]i"
        "[opt]i",
        avs_descale_create,
        (void *)(DESCALE_MODE_LANCZOS)
    );

    avs_add_function(
        env,
        "Despline16",
        "c"
        "i"
        "i"
        "[src_left]f"
        "[src_top]f"
        "[src_width]f"
        "[src_height]f"
        "[border_handling]i"
        "[opt]i",
        avs_descale_create,
        (void *)(DESCALE_MODE_SPLINE16)
    );

    avs_add_function(
        env,
        "Despline36",
        "c"
        "i"
        "i"
        "[src_left]f"
        "[src_top]f"
        "[src_width]f"
        "[src_height]f"
        "[border_handling]i"
        "[opt]i",
        avs_descale_create,
        (void *)(DESCALE_MODE_SPLINE36)
    );

    avs_add_function(
        env,
        "Despline64",
        "c"
        "i"
        "i"
        "[src_left]f"
        "[src_top]f"
        "[src_width]f"
        "[src_height]f"
        "[border_handling]i"
        "[opt]i",
        avs_descale_create,
        (void *)(DESCALE_MODE_SPLINE64)
    );

    avs_add_function(
        env,
        "Descale",
        "c"
        "i"
        "i"
        "[kernel]s"
        "[taps]i"
        "[b]f"
        "[c]f"
        "[src_left]f"
        "[src_top]f"
        "[src_width]f"
        "[src_height]f"
        "[border_handling]i"
        "[opt]i",
        avs_descale_create,
        NULL
    );

    return "Descale plugin";
}
