/* 
 * Copyright © 2021-2022 Frechdachs <frechdachs@rekt.cc>
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


#include <ctype.h>
#include <stdbool.h>
#include "descale.h"


struct DescaleData
{
    int src_width, src_height;
    int dst_width, dst_height;
    int subsampling_h, subsampling_v;
    int num_planes;
    bool process_h, process_v;
    double shift_h, shift_v;
    double active_width, active_height;

    struct DescaleAPI dsapi;
    struct DescaleParams params;
    struct DescaleCore *dscore_h[2];
    struct DescaleCore *dscore_v[2];
};


static bool string_is_equal_ignore_case(const char *s1, const char *s2)
{
    int i;
    for (i = 0; s1[i] != '\0'; i++) {
        if (tolower(s1[i]) != tolower(s2[i]))
            return false;
    }
    return s1[i] == s2[i];
}


static void initialize_descale_data(struct DescaleData *dd)
{
    if (dd->process_h) {
        dd->params.shift = dd->shift_h;
        dd->params.active_dim = dd->active_width;
        dd->dscore_h[0] = dd->dsapi.create_core(dd->src_width, dd->dst_width, &dd->params);
        if (dd->num_planes > 1 && dd->subsampling_h > 0) {
            dd->params.shift = 0.25 - 0.25 * (double)dd->dst_width / (double)dd->src_width;  // For now always assume left-aligned chroma
            dd->params.shift += dd->shift_h * (double)(dd->src_width >> dd->subsampling_h) / (double)dd->src_width;
            dd->params.active_dim = dd->active_width * (double)(dd->src_width >> dd->subsampling_h) / (double)dd->src_width;
            dd->dscore_h[1] = dd->dsapi.create_core(dd->src_width >> dd->subsampling_h, dd->dst_width >> dd->subsampling_h, &dd->params);
        }
    }
    if (dd->process_v) {
        dd->params.shift = dd->shift_v;
        dd->params.active_dim = dd->active_height;
        dd->dscore_v[0] = dd->dsapi.create_core(dd->src_height, dd->dst_height, &dd->params);
        if (dd->num_planes > 1 && dd->subsampling_v > 0) {
            dd->params.shift = dd->shift_v * (double)(dd->src_height >> dd->subsampling_v) / (double)dd->src_height;
            dd->params.active_dim = dd->active_height * (double)(dd->src_height >> dd->subsampling_v) / (double)dd->src_height;
            dd->dscore_v[1] = dd->dsapi.create_core(dd->src_height >> dd->subsampling_v, dd->dst_height >> dd->subsampling_v, &dd->params);
        }
    }
}
