/* 
 * Copyright © 2020 Frechdachs <frechdachs@rekt.cc>
 * This program is free software. It comes without any warranty, to
 * the extent permitted by applicable law. You can redistribute it
 * and/or modify it under the terms of the Do What The Fuck You Want
 * To Public License, Version 2, as published by Sam Hocevar.
 * See the COPYING file for more details.
 */


#ifdef DESCALE_X86

#ifndef DESCALE_AVX2_H
#define DESCALE_AVX2_H


#include <vapoursynth/VSHelper.h>


void process_plane_h_b3_avx2(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                             int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                             float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp);


void process_plane_h_b7_avx2(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                             int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                             float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp);


void process_plane_h_avx2(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                          int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                          float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp);


void process_plane_v_b3_avx2(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                             int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                             float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp);


void process_plane_v_b7_avx2(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                             int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                             float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp);


void process_plane_v_avx2(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                          int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                          float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp);


#endif  // DESCALE_AVX2_H
#endif  // DESCALE_X86
