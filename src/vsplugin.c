/* 
 * Copyright © 2017-2022 Frechdachs <frechdachs@rekt.cc>
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vapoursynth/VapourSynth4.h>
#include <vapoursynth/VSHelper4.h>
#include "descale.h"
#include "plugin.h"


struct VSDescaleData
{
    bool initialized;
    pthread_mutex_t lock;

    VSNode *node;
    VSVideoInfo vi;

    struct DescaleData dd;
};


struct VSCustomKernelData
{
    const VSAPI *vsapi;
    VSFunction *custom_kernel;
};


static const VSFrame *VS_CC descale_get_frame(int n, int activation_reason, void *instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi)
{
    struct VSDescaleData *d = (struct VSDescaleData *)instance_data;

    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frame_ctx);

    } else if (activation_reason == arAllFramesReady) {

        if (!d->initialized) {
            pthread_mutex_lock(&d->lock);
            if (!d->initialized) {
                initialize_descale_data(&d->dd);
                d->initialized = true;
            }
            pthread_mutex_unlock(&d->lock);
        }

        const VSVideoFormat fmt = d->vi.format;
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frame_ctx);

        VSFrame *intermediate = vsapi->newVideoFrame(&fmt, d->dd.dst_width, d->dd.src_height, NULL, core);
        VSFrame *dst = vsapi->newVideoFrame(&fmt, d->dd.dst_width, d->dd.dst_height, src, core);

        for (int plane = 0; plane < d->dd.num_planes; plane++) {
            int src_stride = vsapi->getStride(src, plane) / sizeof (float);
            int dst_stride = vsapi->getStride(dst, plane) / sizeof (float);
            const float *srcp = (const float *)vsapi->getReadPtr(src, plane);
            float *dstp = (float *)vsapi->getWritePtr(dst, plane);

            if (d->dd.process_h && d->dd.process_v) {
                int intermediate_stride = vsapi->getStride(intermediate, plane) / sizeof (float);
                float *intermediatep = (float *)vsapi->getWritePtr(intermediate, plane);

                d->dd.dsapi.process_vectors(d->dd.dscore_h[plane && d->dd.subsampling_h], DESCALE_DIR_HORIZONTAL, d->dd.src_height >> (plane ? d->dd.subsampling_v : 0), src_stride, intermediate_stride, srcp, intermediatep);
                d->dd.dsapi.process_vectors(d->dd.dscore_v[plane && d->dd.subsampling_v], DESCALE_DIR_VERTICAL, d->dd.dst_width >> (plane ? d->dd.subsampling_h : 0), intermediate_stride, dst_stride, intermediatep, dstp);

            } else if (d->dd.process_h) {
                d->dd.dsapi.process_vectors(d->dd.dscore_h[plane && d->dd.subsampling_h], DESCALE_DIR_HORIZONTAL, d->dd.src_height >> (plane ? d->dd.subsampling_v : 0), src_stride, dst_stride, srcp, dstp);

            } else if (d->dd.process_v) {
                d->dd.dsapi.process_vectors(d->dd.dscore_v[plane && d->dd.subsampling_v], DESCALE_DIR_VERTICAL, d->dd.src_width >> (plane ? d->dd.subsampling_h : 0), src_stride, dst_stride, srcp, dstp);
            }
        }

        vsapi->freeFrame(intermediate);
        vsapi->freeFrame(src);

        return dst;
    }

    return NULL;
}


static void VS_CC descale_free(void *instance_data, VSCore *core, const VSAPI *vsapi)
{
    struct VSDescaleData *d = (struct VSDescaleData *)instance_data;

    vsapi->freeNode(d->node);

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

    if (d->dd.params.mode == DESCALE_MODE_CUSTOM) {
        struct VSCustomKernelData *kd = (struct VSCustomKernelData *)d->dd.params.custom_kernel.user_data;
        vsapi->freeFunction(kd->custom_kernel);
        free(kd);
    }

    free(d);
}


static double custom_kernel_f(double x, void *user_data)
{
    struct VSCustomKernelData *kd = (struct VSCustomKernelData *)user_data;

    VSMap *in = kd->vsapi->createMap();
    VSMap *out = kd->vsapi->createMap();
    kd->vsapi->mapSetFloat(in, "x", x, maReplace);
    kd->vsapi->callFunction(kd->custom_kernel, in, out);
    if (kd->vsapi->mapGetError(out)) {
        fprintf(stderr, "Descale: custom kernel error: %s.\n", kd->vsapi->mapGetError(out));
        kd->vsapi->freeMap(in);
        kd->vsapi->freeMap(out);
        return 0.0;
    }
    int err;
    x = kd->vsapi->mapGetFloat(out, "val", 0, &err);
    if (err)
        x = (double)kd->vsapi->mapGetInt(out, "val", 0, &err);
    if (err) {
        fprintf(stderr, "Descale: custom kernel: The custom kernel function returned a value that is neither float nor int.");
        x = 0.0;
    }
    kd->vsapi->freeMap(in);
    kd->vsapi->freeMap(out);

    return x;
}


static void VS_CC descale_create(const VSMap *in, VSMap *out, void *user_data, VSCore *core, const VSAPI *vsapi)
{
    struct VSDescaleData d = {0};
    struct DescaleParams params = {0};

    VSFunction *custom_kernel = NULL;
    if (user_data == NULL) {
        int no_kernel;
        int no_custom_kernel;
        const char *kernel = vsapi->mapGetData(in, "kernel", 0, &no_kernel);
        custom_kernel = vsapi->mapGetFunction(in, "custom_kernel", 0, &no_custom_kernel);
        if (!no_kernel && !no_custom_kernel) {
            vsapi->mapSetError(out, "Descale: Specify either kernel or custom_kernel, but not both.");
            vsapi->freeFunction(custom_kernel);
            return;
        }
        if (no_kernel && no_custom_kernel) {
            vsapi->mapSetError(out, "Descale: Either kernel or custom_kernel must be specified.");
            return;
        }
        if (!no_kernel) {
            if (string_is_equal_ignore_case(kernel, "bilinear"))
                params.mode = DESCALE_MODE_BILINEAR;
            else if (string_is_equal_ignore_case(kernel, "bicubic"))
                params.mode = DESCALE_MODE_BICUBIC;
            else if (string_is_equal_ignore_case(kernel, "lanczos"))
                params.mode = DESCALE_MODE_LANCZOS;
            else if (string_is_equal_ignore_case(kernel, "spline16"))
                params.mode = DESCALE_MODE_SPLINE16;
            else if (string_is_equal_ignore_case(kernel, "spline36"))
                params.mode = DESCALE_MODE_SPLINE36;
            else if (string_is_equal_ignore_case(kernel, "spline64"))
                params.mode = DESCALE_MODE_SPLINE64;
            else {
                vsapi->mapSetError(out, "Descale: Invalid kernel specified.");
                return;
            }
        } else {
            params.mode = DESCALE_MODE_CUSTOM;
            params.custom_kernel.f = &custom_kernel_f;
            struct VSCustomKernelData *kd = malloc(sizeof (struct VSCustomKernelData));
            kd->vsapi = vsapi;
            kd->custom_kernel = custom_kernel;
            params.custom_kernel.user_data = kd;
        }
    } else {
        params.mode = (enum DescaleMode)user_data;
    }

    d.node = vsapi->mapGetNode(in, "src", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);

    if (!vsh_isConstantVideoFormat(&d.vi)) {
        vsapi->mapSetError(out, "Descale: Only constant format input is supported.");
        vsapi->freeNode(d.node);
        return;
    }

    d.dd.src_width = d.vi.width;
    d.dd.src_height = d.vi.height;
    d.dd.dst_width = vsapi->mapGetIntSaturated(in, "width", 0, NULL);
    d.dd.dst_height = vsapi->mapGetIntSaturated(in, "height", 0, NULL);
    d.vi.width = d.dd.dst_width;
    d.vi.height = d.dd.dst_height;
    d.dd.subsampling_h = d.vi.format.subSamplingW;
    d.dd.subsampling_v = d.vi.format.subSamplingH;
    d.dd.num_planes = d.vi.format.numPlanes;

    if (d.dd.dst_width % (1 << d.dd.subsampling_h) != 0) {
        vsapi->mapSetError(out, "Descale: Output width and output subsampling are not compatible.");
        vsapi->freeNode(d.node);
        return;
    }
    if (d.dd.dst_height % (1 << d.dd.subsampling_v) != 0) {
        vsapi->mapSetError(out, "Descale: Output height and output subsampling are not compatible.");
        vsapi->freeNode(d.node);
        return;
    }

    int err;

    d.dd.shift_h = vsapi->mapGetFloat(in, "src_left", 0, &err);
    if (err)
        d.dd.shift_h = 0.0;

    d.dd.shift_v = vsapi->mapGetFloat(in, "src_top", 0, &err);
    if (err)
        d.dd.shift_v = 0.0;

    d.dd.active_width = vsapi->mapGetFloat(in, "src_width", 0, &err);
    if (err)
        d.dd.active_width = (double)d.dd.dst_width;

    d.dd.active_height = vsapi->mapGetFloat(in, "src_height", 0, &err);
    if (err)
        d.dd.active_height = (double)d.dd.dst_height;

    int border_handling = vsapi->mapGetIntSaturated(in, "border_handling", 0, &err);
    if (err)
        border_handling = 0;
    if (border_handling == 1)
        params.border_handling = DESCALE_BORDER_ZERO;
    else
        params.border_handling = DESCALE_BORDER_MIRROR;

    enum DescaleOpt opt_enum;
    int opt = vsapi->mapGetIntSaturated(in, "opt", 0, &err);
    if (err)
        opt = 0;
    if (opt == 1)
        opt_enum = DESCALE_OPT_NONE;
    else if (opt == 2)
        opt_enum = DESCALE_OPT_AVX2;
    else
        opt_enum = DESCALE_OPT_AUTO;

    if (d.dd.dst_width < 1) {
        vsapi->mapSetError(out, "Descale: width must be greater than 0.");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.dd.dst_height < 8) {
        vsapi->mapSetError(out, "Descale: Output height must be greater than or equal to 8.");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.dd.dst_width > d.dd.src_width || d.dd.dst_height > d.dd.src_height) {
        vsapi->mapSetError(out, "Descale: Output dimension must be less than or equal to input dimension.");
        vsapi->freeNode(d.node);
        return;
    }

    d.dd.process_h = d.dd.dst_width != d.dd.src_width || d.dd.shift_h != 0.0 || d.dd.active_width != (double)d.dd.dst_width;
    d.dd.process_v = d.dd.dst_height != d.dd.src_height || d.dd.shift_v != 0.0 || d.dd.active_height != (double)d.dd.dst_height;

    char *funcname;

    if (params.mode == DESCALE_MODE_BILINEAR) {
        funcname = "Debilinear";
    
    } else if (params.mode == DESCALE_MODE_BICUBIC) {
        params.param1 = vsapi->mapGetFloat(in, "b", 0, &err);
        if (err)
            params.param1 = 0.0;

        params.param2 = vsapi->mapGetFloat(in, "c", 0, &err);
        if (err)
            params.param2 = 0.5;

        funcname = "Debicubic";

        // If b != 0 Bicubic is not an interpolation filter, so force processing
        /*if (params.param1 != 0) {
            d.dd.process_h = true;
            d.dd.process_v = true;
        }*/
        // Leaving this check in would make it impossible to only descale a single dimension if this precondition is met.
        // If you want to force sampling use the force/force_h/force_v paramenter of the generic Descale filter.

    } else if (params.mode == DESCALE_MODE_LANCZOS || params.mode == DESCALE_MODE_CUSTOM) {
        params.taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);

        if (err && params.mode == DESCALE_MODE_CUSTOM) {
            vsapi->mapSetError(out, "Descale: If custom_kernel is specified, then taps must also be specified.");
            vsapi->freeFunction(custom_kernel);
            free(params.custom_kernel.user_data);
            vsapi->freeNode(d.node);
            return;

        } else if (err) {
            params.taps = 3;
        }

        if (params.taps < 1) {
            vsapi->mapSetError(out, "Descale: taps must be greater than 0.");
            vsapi->freeNode(d.node);
            return;
        }

        funcname = "Delanczos";

    } else if (params.mode == DESCALE_MODE_SPLINE16) {
        funcname = "Despline16";

    } else if (params.mode == DESCALE_MODE_SPLINE36) {
        funcname = "Despline36";

    } else if (params.mode == DESCALE_MODE_SPLINE64) {
        funcname = "Despline64";
    } else {
        funcname = "none";
    }

    int force = vsapi->mapGetIntSaturated(in, "force", 0, &err);
    int force_h = vsapi->mapGetIntSaturated(in, "force_h", 0, &err);
    if (err)
        force_h = force;
    int force_v = vsapi->mapGetIntSaturated(in, "force_v", 0, &err);
    if (err)
        force_v = force;

    d.dd.process_h = d.dd.process_h || force_h;
    d.dd.process_v = d.dd.process_v || force_v;

    // Return the input clip if no processing is necessary
    if (!d.dd.process_h && !d.dd.process_v) {
        vsapi->mapSetNode(out, "clip", d.node, maReplace);
        vsapi->freeNode(d.node);
        return;
    }

    // If necessary, resample to single precision float, call another descale instance,
    // and resample back to the original format
    if (d.vi.format.sampleType != stFloat || d.vi.format.bitsPerSample != 32) {
        VSMap *map1;
        VSMap *map2;
        VSNode *tmp_node;
        const char *err_msg;
        const VSVideoFormat src_fmt = d.vi.format;
        uint32_t src_fmt_id = vsapi->queryVideoFormatID(src_fmt.colorFamily, src_fmt.sampleType, src_fmt.bitsPerSample, src_fmt.subSamplingW, src_fmt.subSamplingH, core);
        uint32_t flt_fmt_id = vsapi->queryVideoFormatID(src_fmt.colorFamily, stFloat, 32, src_fmt.subSamplingW, src_fmt.subSamplingH, core);
        VSPlugin *resize_plugin = vsapi->getPluginByID("com.vapoursynth.resize", core);
        VSPlugin *descale_plugin = vsapi->getPluginByID("tegaf.asi.xe", core);

        // Convert to float
        map1 = vsapi->createMap();
        vsapi->mapSetNode(map1, "clip", d.node, maReplace);
        vsapi->mapSetInt(map1, "format", flt_fmt_id, maReplace);
        vsapi->mapSetData(map1, "dither_type", "none", -1, dtUtf8, maReplace);
        map2 = vsapi->invoke(resize_plugin, "Point", map1);
        vsapi->freeNode(d.node);
        vsapi->freeMap(map1);
        if (vsapi->mapGetError(map2)) {
            vsapi->mapSetError(out, "Descale: Resampling to single precision float failed.");
            vsapi->freeMap(map2);
            return;
        }
        tmp_node = vsapi->mapGetNode(map2, "clip", 0, NULL);
        vsapi->freeMap(map2);

        // Call Descale
        map1 = vsapi->createMap();
        vsapi->mapSetNode(map1, "src", tmp_node, maReplace);
        vsapi->mapSetInt(map1, "width", d.dd.dst_width, maReplace);
        vsapi->mapSetInt(map1, "height", d.dd.dst_height, maReplace);
        if (params.mode == DESCALE_MODE_CUSTOM) {
            vsapi->mapSetFunction(map1, "custom_kernel", custom_kernel, maReplace);
            vsapi->freeFunction(custom_kernel);
            free(params.custom_kernel.user_data);
        } else {
            vsapi->mapSetData(map1, "kernel", funcname + 2, -1, dtUtf8, maReplace);
        }
        vsapi->mapSetInt(map1, "taps", params.taps, maReplace);
        vsapi->mapSetFloat(map1, "b", params.param1, maReplace);
        vsapi->mapSetFloat(map1, "c", params.param2, maReplace);
        vsapi->mapSetFloat(map1, "src_left", d.dd.shift_h, maReplace);
        vsapi->mapSetFloat(map1, "src_top", d.dd.shift_v, maReplace);
        vsapi->mapSetFloat(map1, "src_width", d.dd.active_width, maReplace);
        vsapi->mapSetFloat(map1, "src_height", d.dd.active_height, maReplace);
        vsapi->mapSetInt(map1, "border_handling", (int)params.border_handling, maReplace);
        vsapi->mapSetInt(map1, "force", force, maReplace);
        vsapi->mapSetInt(map1, "force_h", force_h, maReplace);
        vsapi->mapSetInt(map1, "force_v", force_v, maReplace);
        vsapi->mapSetInt(map1, "opt", (int)opt_enum, maReplace);
        map2 = vsapi->invoke(descale_plugin, "Descale", map1);
        vsapi->freeNode(tmp_node);
        vsapi->freeMap(map1);
        if ((err_msg = vsapi->mapGetError(map2))) {
            vsapi->mapSetError(out, err_msg);
            vsapi->freeMap(map2);
            return;
        }
        tmp_node = vsapi->mapGetNode(map2, "clip", 0, NULL);
        vsapi->freeMap(map2);

        // Convert to original format
        map1 = vsapi->createMap();
        vsapi->mapSetNode(map1, "clip", tmp_node, maReplace);
        vsapi->mapSetInt(map1, "format", src_fmt_id, maReplace);
        vsapi->mapSetData(map1, "dither_type", "none", -1, dtUtf8, maReplace);
        map2 = vsapi->invoke(resize_plugin, "Point", map1);
        vsapi->freeNode(tmp_node);
        vsapi->freeMap(map1);
        if (vsapi->mapGetError(map2)) {
            vsapi->mapSetError(out, "Descale: Resampling to original format failed.");
            vsapi->freeMap(map2);
            return;
        }
        tmp_node = vsapi->mapGetNode(map2, "clip", 0, NULL);
        vsapi->freeMap(map2);

        // Return the clip and exit
        vsapi->mapSetNode(out, "clip", tmp_node, maReplace);
        vsapi->freeNode(tmp_node);

        return;
    }


    d.dd.dsapi = get_descale_api(opt_enum);
    d.initialized = false;
    pthread_mutex_init(&d.lock, NULL);

    struct VSDescaleData *data = malloc(sizeof d);
    *data = d;
    data->dd.params = params;
    VSFilterDependency deps[] = {{data->node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, funcname, &data->vi, descale_get_frame, descale_free, fmParallel, deps, 1, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi)
{
    vspapi->configPlugin("tegaf.asi.xe", "descale", "Undo linear interpolation", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    vspapi->registerFunction("Debilinear",
            "src:vnode;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "border_handling:int:opt;"
            "force:int:opt;"
            "force_h:int:opt;"
            "force_v:int:opt;"
            "opt:int:opt;",
            "clip:vnode;",
            descale_create, (void *)(DESCALE_MODE_BILINEAR), plugin);

    vspapi->registerFunction("Debicubic",
            "src:vnode;"
            "width:int;"
            "height:int;"
            "b:float:opt;"
            "c:float:opt;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "border_handling:int:opt;"
            "force:int:opt;"
            "force_h:int:opt;"
            "force_v:int:opt;"
            "opt:int:opt;",
            "clip:vnode;",
            descale_create, (void *)(DESCALE_MODE_BICUBIC), plugin);

    vspapi->registerFunction("Delanczos",
            "src:vnode;"
            "width:int;"
            "height:int;"
            "taps:int:opt;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "border_handling:int:opt;"
            "force:int:opt;"
            "force_h:int:opt;"
            "force_v:int:opt;"
            "opt:int:opt;",
            "clip:vnode;",
            descale_create, (void *)(DESCALE_MODE_LANCZOS), plugin);

    vspapi->registerFunction("Despline16",
            "src:vnode;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "border_handling:int:opt;"
            "force:int:opt;"
            "force_h:int:opt;"
            "force_v:int:opt;"
            "opt:int:opt;",
            "clip:vnode;",
            descale_create, (void *)(DESCALE_MODE_SPLINE16), plugin);

    vspapi->registerFunction("Despline36",
            "src:vnode;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "border_handling:int:opt;"
            "force:int:opt;"
            "force_h:int:opt;"
            "force_v:int:opt;"
            "opt:int:opt;",
            "clip:vnode;",
            descale_create, (void *)(DESCALE_MODE_SPLINE36), plugin);

    vspapi->registerFunction("Despline64",
            "src:vnode;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "border_handling:int:opt;"
            "force:int:opt;"
            "force_h:int:opt;"
            "force_v:int:opt;"
            "opt:int:opt;",
            "clip:vnode;",
            descale_create, (void *)(DESCALE_MODE_SPLINE64), plugin);

    vspapi->registerFunction("Descale",
            "src:vnode;"
            "width:int;"
            "height:int;"
            "kernel:data:opt;"
            "custom_kernel:func:opt;"
            "taps:int:opt;"
            "b:float:opt;"
            "c:float:opt;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "border_handling:int:opt;"
            "force:int:opt;"
            "force_h:int:opt;"
            "force_v:int:opt;"
            "opt:int:opt;",
            "clip:vnode;",
            descale_create, NULL, plugin);
}
