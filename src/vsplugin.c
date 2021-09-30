/* 
 * Copyright © 2017-2021 Frechdachs <frechdachs@rekt.cc>
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
#include <vapoursynth/VapourSynth.h>
#include <vapoursynth/VSHelper.h>
#include "descale.h"
#include "plugin.h"


struct VSDescaleData
{
    bool initialized;
    pthread_mutex_t lock;

    VSNodeRef *node;
    VSVideoInfo vi;

    struct DescaleData dd;
};


static const VSFrameRef *VS_CC descale_get_frame(int n, int activationReason, void **instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi)
{
    struct VSDescaleData *d = (struct VSDescaleData *)(*instance_data);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frame_ctx);

    } else if (activationReason == arAllFramesReady) {

        if (!d->initialized) {
            pthread_mutex_lock(&d->lock);
            if (!d->initialized) {
                initialize_descale_data(&d->dd);
                d->initialized = true;
            }
            pthread_mutex_unlock(&d->lock);
        }

        const VSFormat *fmt = d->vi.format;
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->node, frame_ctx);

        VSFrameRef *intermediate = vsapi->newVideoFrame(fmt, d->dd.dst_width, d->dd.src_height, NULL, core);
        VSFrameRef *dst = vsapi->newVideoFrame(fmt, d->dd.dst_width, d->dd.dst_height, src, core);

        for (int plane = 0; plane < d->dd.num_planes; plane++) {
            int src_stride = vsapi->getStride(src, plane) / sizeof (float);
            const float *srcp = (const float *)vsapi->getReadPtr(src, plane);

            if (d->dd.process_h && d->dd.process_v) {
                int intermediate_stride = vsapi->getStride(intermediate, plane) / sizeof (float);
                float *intermediatep = (float *)vsapi->getWritePtr(intermediate, plane);

                d->dd.dsapi.process_vectors(d->dd.dscore_h[plane && d->dd.subsampling_h], DESCALE_DIR_HORIZONTAL, d->dd.src_height >> (plane ? d->dd.subsampling_v : 0), src_stride, intermediate_stride, srcp, intermediatep);


                int dst_stride = vsapi->getStride(dst, plane) / sizeof (float);
                float *dstp = (float *)vsapi->getWritePtr(dst, plane);

                d->dd.dsapi.process_vectors(d->dd.dscore_v[plane && d->dd.subsampling_v], DESCALE_DIR_VERTICAL, d->dd.dst_width >> (plane ? d->dd.subsampling_h : 0), intermediate_stride, dst_stride, intermediatep, dstp);

            } else if (d->dd.process_h) {
                int dst_stride = vsapi->getStride(dst, plane) / sizeof (float);
                float *dstp = (float *)vsapi->getWritePtr(dst, plane);

                d->dd.dsapi.process_vectors(d->dd.dscore_h[plane && d->dd.subsampling_h], DESCALE_DIR_HORIZONTAL, d->dd.src_height >> (plane ? d->dd.subsampling_v : 0), src_stride, dst_stride, srcp, dstp);

            } else if (d->dd.process_v) {
                int dst_stride = vsapi->getStride(dst, plane) / sizeof (float);
                float *dstp = (float *)vsapi->getWritePtr(dst, plane);

                d->dd.dsapi.process_vectors(d->dd.dscore_v[plane && d->dd.subsampling_v], DESCALE_DIR_VERTICAL, d->dd.src_width >> (plane ? d->dd.subsampling_h : 0), src_stride, dst_stride, srcp, dstp);
            }
        }

        vsapi->freeFrame(intermediate);
        vsapi->freeFrame(src);

        return dst;
    }

    return NULL;
}


static void VS_CC descale_init(VSMap *in, VSMap *out, void **instance_data, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
    struct VSDescaleData *d = (struct VSDescaleData *)(*instance_data);
    vsapi->setVideoInfo(&d->vi, 1, node);
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

    free(d);
}


static void VS_CC descale_create(const VSMap *in, VSMap *out, void *user_data, VSCore *core, const VSAPI *vsapi)
{
    struct VSDescaleData d = {0};
    struct DescaleParams params = {0};

    if (user_data == NULL) {
        const char *kernel = vsapi->propGetData(in, "kernel", 0, NULL);
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
            vsapi->setError(out, "Descale: Invalid kernel specified.");
            return;
        }
    } else {
        params.mode = (enum DescaleMode)user_data;
    }

    d.node = vsapi->propGetNode(in, "src", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);

    if (!isConstantFormat(&d.vi)) {
        vsapi->setError(out, "Descale: Only constant format input is supported.");
        vsapi->freeNode(d.node);
        return;
    }

    d.dd.src_width = d.vi.width;
    d.dd.src_height = d.vi.height;
    d.dd.dst_width = int64ToIntS(vsapi->propGetInt(in, "width", 0, NULL));
    d.dd.dst_height = int64ToIntS(vsapi->propGetInt(in, "height", 0, NULL));
    d.vi.width = d.dd.dst_width;
    d.vi.height = d.dd.dst_height;
    d.dd.subsampling_h = d.vi.format->subSamplingW;
    d.dd.subsampling_v = d.vi.format->subSamplingH;
    d.dd.num_planes = d.vi.format->numPlanes;

    if (d.dd.dst_width % (1 << d.dd.subsampling_h) != 0) {
        vsapi->setError(out, "Descale: Output width and output subsampling are not compatible.");
        vsapi->freeNode(d.node);
        return;
    }
    if (d.dd.dst_height % (1 << d.dd.subsampling_v) != 0) {
        vsapi->setError(out, "Descale: Output height and output subsampling are not compatible.");
        vsapi->freeNode(d.node);
        return;
    }

    int err;

    d.dd.shift_h = vsapi->propGetFloat(in, "src_left", 0, &err);
    if (err)
        d.dd.shift_h = 0.0;

    d.dd.shift_v = vsapi->propGetFloat(in, "src_top", 0, &err);
    if (err)
        d.dd.shift_v = 0.0;

    d.dd.active_width = vsapi->propGetFloat(in, "src_width", 0, &err);
    if (err)
        d.dd.active_width = (double)d.dd.dst_width;

    d.dd.active_height = vsapi->propGetFloat(in, "src_height", 0, &err);
    if (err)
        d.dd.active_height = (double)d.dd.dst_height;

    DescaleOpt opt_enum;
    int opt = int64ToIntS(vsapi->propGetInt(in, "opt", 0, &err));
    if (err)
        opt = 0;
    if (opt == 1)
        opt_enum = DESCALE_OPT_NONE;
    else if (opt == 2)
        opt_enum = DESCALE_OPT_AVX2;
    else
        opt_enum = DESCALE_OPT_AUTO;

    if (d.dd.dst_width < 1) {
        vsapi->setError(out, "Descale: width must be greater than 0.");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.dd.dst_height < 8) {
        vsapi->setError(out, "Descale: Output height must be greater than or equal to 8.");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.dd.dst_width > d.dd.src_width || d.dd.dst_height > d.dd.src_height) {
        vsapi->setError(out, "Descale: Output dimension must be less than or equal to input dimension.");
        vsapi->freeNode(d.node);
        return;
    }

    d.dd.process_h = (d.dd.dst_width == d.dd.src_width && d.dd.shift_h == 0 && d.dd.active_width != (double)d.dd.dst_width) ? false : true;
    d.dd.process_v = (d.dd.dst_height == d.dd.src_height && d.dd.shift_v == 0 && d.dd.active_height != (double)d.dd.dst_height) ? false : true;

    char *funcname;

    if (params.mode == DESCALE_MODE_BILINEAR) {
        funcname = "Debilinear";
    
    } else if (params.mode == DESCALE_MODE_BICUBIC) {
        params.param1 = vsapi->propGetFloat(in, "b", 0, &err);
        if (err)
            params.param1 = 0.0;

        params.param2 = vsapi->propGetFloat(in, "c", 0, &err);
        if (err)
            params.param2 = 0.5;

        funcname = "Debicubic";

        // If b != 0 Bicubic is not an interpolation filter, so force processing
        if (params.param1 != 0) {
            d.dd.process_h = true;
            d.dd.process_v = true;
        }

    } else if (params.mode == DESCALE_MODE_LANCZOS) {
        params.taps = int64ToIntS(vsapi->propGetInt(in, "taps", 0, &err));
        if (err)
            params.taps = 3;

        if (params.taps < 1) {
            vsapi->setError(out, "Descale: taps must be greater than 0.");
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

    // Return the input clip if no processing is necessary
    if (!d.dd.process_h && !d.dd.process_v) {
        vsapi->propSetNode(out, "clip", d.node, paReplace);
        vsapi->freeNode(d.node);
        return;
    }

    // If necessary, resample to single precision float, call another descale instance,
    // and resample back to the original format
    if (d.vi.format->sampleType != stFloat || d.vi.format->bitsPerSample != 32) {
        VSMap *map1;
        VSMap *map2;
        VSNodeRef *tmp_node;
        const char *err_msg;
        const VSFormat *src_fmt = d.vi.format;
        const VSFormat *flt_fmt = vsapi->registerFormat(src_fmt->colorFamily, stFloat, 32, src_fmt->subSamplingW, src_fmt->subSamplingH, core);
        VSPlugin *resize_plugin = vsapi->getPluginById("com.vapoursynth.resize", core);
        VSPlugin *descale_plugin = vsapi->getPluginById("tegaf.asi.xe", core);

        // Convert to float
        map1 = vsapi->createMap();
        vsapi->propSetNode(map1, "clip", d.node, paReplace);
        vsapi->propSetInt(map1, "format", flt_fmt->id, paReplace);
        vsapi->propSetData(map1, "dither_type", "none", -1, paReplace);
        map2 = vsapi->invoke(resize_plugin, "Point", map1);
        vsapi->freeNode(d.node);
        vsapi->freeMap(map1);
        if (vsapi->getError(map2)) {
            vsapi->setError(out, "Descale: Resampling to single precision float failed.");
            vsapi->freeMap(map2);
            return;
        }
        tmp_node = vsapi->propGetNode(map2, "clip", 0, NULL);
        vsapi->freeMap(map2);

        // Call Descale
        map1 = vsapi->createMap();
        vsapi->propSetNode(map1, "src", tmp_node, paReplace);
        vsapi->propSetInt(map1, "width", d.dd.dst_width, paReplace);
        vsapi->propSetInt(map1, "height", d.dd.dst_height, paReplace);
        vsapi->propSetData(map1, "kernel", funcname + 2, -1, paReplace);
        vsapi->propSetInt(map1, "taps", params.taps, paReplace);
        vsapi->propSetFloat(map1, "b", params.param1, paReplace);
        vsapi->propSetFloat(map1, "c", params.param2, paReplace);
        vsapi->propSetFloat(map1, "src_left", d.dd.shift_h, paReplace);
        vsapi->propSetFloat(map1, "src_top", d.dd.shift_v, paReplace);
        vsapi->propSetFloat(map1, "src_width", d.dd.active_width, paReplace);
        vsapi->propSetFloat(map1, "src_height", d.dd.active_height, paReplace);
        vsapi->propSetInt(map1, "opt", (int)opt_enum, paReplace);
        map2 = vsapi->invoke(descale_plugin, "Descale", map1);
        vsapi->freeNode(tmp_node);
        vsapi->freeMap(map1);
        if ((err_msg = vsapi->getError(map2))) {
            vsapi->setError(out, err_msg);
            vsapi->freeMap(map2);
            return;
        }
        tmp_node = vsapi->propGetNode(map2, "clip", 0, NULL);
        vsapi->freeMap(map2);

        // Convert to original format
        map1 = vsapi->createMap();
        vsapi->propSetNode(map1, "clip", tmp_node, paReplace);
        vsapi->propSetInt(map1, "format", src_fmt->id, paReplace);
        vsapi->propSetData(map1, "dither_type", "none", -1, paReplace);
        map2 = vsapi->invoke(resize_plugin, "Point", map1);
        vsapi->freeNode(tmp_node);
        vsapi->freeMap(map1);
        if (vsapi->getError(map2)) {
            vsapi->setError(out, "Descale: Resampling to original format failed.");
            vsapi->freeMap(map2);
            return;
        }
        tmp_node = vsapi->propGetNode(map2, "clip", 0, NULL);
        vsapi->freeMap(map2);

        // Return the clip and exit
        vsapi->propSetNode(out, "clip", tmp_node, paReplace);
        vsapi->freeNode(tmp_node);

        return;
    }


    d.dd.dsapi = get_descale_api(opt_enum);
    d.initialized = false;
    pthread_mutex_init(&d.lock, NULL);

    struct VSDescaleData *data = malloc(sizeof d);
    *data = d;
    data->dd.params = params;
    vsapi->createFilter(in, out, funcname, descale_init, descale_get_frame, descale_free, fmParallel, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin config_func, VSRegisterFunction register_func, VSPlugin *plugin)
{
    config_func("tegaf.asi.xe", "descale", "Undo linear interpolation", VAPOURSYNTH_API_VERSION, 1, plugin);

    register_func("Debilinear",
            "src:clip;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "opt:int:opt",
            descale_create, (void *)(DESCALE_MODE_BILINEAR), plugin);

    register_func("Debicubic",
            "src:clip;"
            "width:int;"
            "height:int;"
            "b:float:opt;"
            "c:float:opt;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "opt:int:opt",
            descale_create, (void *)(DESCALE_MODE_BICUBIC), plugin);

    register_func("Delanczos",
            "src:clip;"
            "width:int;"
            "height:int;"
            "taps:int:opt;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "opt:int:opt",
            descale_create, (void *)(DESCALE_MODE_LANCZOS), plugin);

    register_func("Despline16",
            "src:clip;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "opt:int:opt",
            descale_create, (void *)(DESCALE_MODE_SPLINE16), plugin);

    register_func("Despline36",
            "src:clip;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "opt:int:opt",
            descale_create, (void *)(DESCALE_MODE_SPLINE36), plugin);

    register_func("Despline64",
            "src:clip;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "opt:int:opt",
            descale_create, (void *)(DESCALE_MODE_SPLINE64), plugin);

    register_func("Descale",
            "src:clip;"
            "width:int;"
            "height:int;"
            "kernel:data;"
            "taps:int:opt;"
            "b:float:opt;"
            "c:float:opt;"
            "src_left:float:opt;"
            "src_top:float:opt;"
            "src_width:float:opt;"
            "src_height:float:opt;"
            "opt:int:opt",
            descale_create, NULL, plugin);
}
