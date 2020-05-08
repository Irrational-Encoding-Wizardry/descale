/* 
 * Copyright © 2020 Frechdachs <frechdachs@rekt.cc>
 * This program is free software. It comes without any warranty, to
 * the extent permitted by applicable law. You can redistribute it
 * and/or modify it under the terms of the Do What The Fuck You Want
 * To Public License, Version 2, as published by Sam Hocevar.
 * See the COPYING file for more details.
 */


#ifdef DESCALE_X86


#include <stdlib.h>
#include <immintrin.h>
#include <vapoursynth/VSHelper.h>
#include "common.h"


// Taken from zimg https://github.com/sekrit-twc/zimg
static inline __attribute__((always_inline)) void mm256_transpose8_ps(__m256 *row0, __m256 *row1, __m256 *row2, __m256 *row3, __m256 *row4, __m256 *row5, __m256 *row6, __m256 *row7)
{
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;
    __m256 tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

    t0 = _mm256_unpacklo_ps(*row0, *row1);
    t1 = _mm256_unpackhi_ps(*row0, *row1);
    t2 = _mm256_unpacklo_ps(*row2, *row3);
    t3 = _mm256_unpackhi_ps(*row2, *row3);
    t4 = _mm256_unpacklo_ps(*row4, *row5);
    t5 = _mm256_unpackhi_ps(*row4, *row5);
    t6 = _mm256_unpacklo_ps(*row6, *row7);
    t7 = _mm256_unpackhi_ps(*row6, *row7);

    tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    *row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
    *row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
    *row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
    *row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
    *row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
    *row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
    *row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
    *row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
}


// Taken from zimg https://github.com/sekrit-twc/zimg
static inline __attribute__((always_inline)) void transpose_line_8x8_ps(float * VS_RESTRICT dst, const float * VS_RESTRICT src, int src_stride, int left, int right)
{
    for (int j = left; j < right; j += 8) {
        __m256 x0, x1, x2, x3, x4, x5, x6, x7;

        x0 = _mm256_load_ps(src + j);
        x1 = _mm256_load_ps(src + src_stride + j);
        x2 = _mm256_load_ps(src + 2 * src_stride + j);
        x3 = _mm256_load_ps(src + 3 * src_stride + j);
        x4 = _mm256_load_ps(src + 4 * src_stride + j);
        x5 = _mm256_load_ps(src + 5 * src_stride + j);
        x6 = _mm256_load_ps(src + 6 * src_stride + j);
        x7 = _mm256_load_ps(src + 7 * src_stride + j);

        mm256_transpose8_ps(&x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7);

        _mm256_store_ps(dst, x0);
        _mm256_store_ps(dst + 8, x1);
        _mm256_store_ps(dst + 16, x2);
        _mm256_store_ps(dst + 24, x3);
        _mm256_store_ps(dst + 32, x4);
        _mm256_store_ps(dst + 40, x5);
        _mm256_store_ps(dst + 48, x6);
        _mm256_store_ps(dst + 56, x7);

        dst += 64;
    }
}


/*
 * Horizontal solver that is specialized for systems with bandwidth 3.
 * It is faster than the generalized version, because it uses much
 * less load/store instructions.
 */
static void process_line8_h_b3_avx2(int width, int current_height, int *current_width, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                    int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT lower, float * VS_RESTRICT upper, float * VS_RESTRICT diagonal,
                                    const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp, float * VS_RESTRICT temp)
{
    transpose_line_8x8_ps(temp, srcp, src_stride, 0, ceil_n(*current_width, 8));
    __m256 x0, x1, x2, x3, x4, x5, x6, x7;
    __m256 a0, a1, lo, up, di, x_last;
    x_last = _mm256_setzero_ps();
    for (int j = 0; j < width; j += 8) {
        x0 = _mm256_setzero_ps();
        x1 = x0;
        x2 = x0;
        x3 = x0;
        x4 = x0;
        x5 = x0;
        x6 = x0;
        x7 = x0;

#define MATMULT(x, a0, a1, wl_idx, wr_idx, w_col, weights, temp, j, m)\
        for (int k = wl_idx[j + m]; k < wr_idx[j + m]; k++) {\
            a0 = _mm256_set1_ps(weights[(j + m) * w_col + k - wl_idx[j + m]]);\
            a1 = _mm256_load_ps(temp + k * 8);\
            x = _mm256_fmadd_ps(a0, a1, x);\
        }

        // A' b
        MATMULT(x0, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 0);
        MATMULT(x1, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 1);
        MATMULT(x2, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 2);
        MATMULT(x3, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 3);
        MATMULT(x4, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 4);
        MATMULT(x5, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 5);
        MATMULT(x6, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 6);
        MATMULT(x7, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 7);

#undef MATMULT

#define SOLVEF(x, lo, di, x_last, j, m)\
        lo = _mm256_set1_ps(lower[j + m]);\
        x = _mm256_fnmadd_ps(lo, x_last, x);\
        di = _mm256_set1_ps(diagonal[j + m]);\
        x = _mm256_mul_ps(x, di);

        // Solve LD y = A' b
        SOLVEF(x0, lo, di, x_last, j, 0);
        SOLVEF(x1, lo, di, x0, j, 1);
        SOLVEF(x2, lo, di, x1, j, 2);
        SOLVEF(x3, lo, di, x2, j, 3);
        SOLVEF(x4, lo, di, x3, j, 4);
        SOLVEF(x5, lo, di, x4, j, 5);
        SOLVEF(x6, lo, di, x5, j, 6);
        SOLVEF(x7, lo, di, x6, j, 7);

#undef SOLVEF

        x_last = x7;

        _mm256_store_ps(dstp + j, x0);
        _mm256_store_ps(dstp + 1 * dst_stride + j, x1);
        _mm256_store_ps(dstp + 2 * dst_stride + j, x2);
        _mm256_store_ps(dstp + 3 * dst_stride + j, x3);
        _mm256_store_ps(dstp + 4 * dst_stride + j, x4);
        _mm256_store_ps(dstp + 5 * dst_stride + j, x5);
        _mm256_store_ps(dstp + 6 * dst_stride + j, x6);
        _mm256_store_ps(dstp + 7 * dst_stride + j, x7);
    }

    // Solve L' x = y
    for (int j = ceil_n(width, 8) - 8; j >= 0; j -= 8) {

        x0 = _mm256_load_ps(dstp + j);
        x1 = _mm256_load_ps(dstp + 1 * dst_stride + j);
        x2 = _mm256_load_ps(dstp + 2 * dst_stride + j);
        x3 = _mm256_load_ps(dstp + 3 * dst_stride + j);
        x4 = _mm256_load_ps(dstp + 4 * dst_stride + j);
        x5 = _mm256_load_ps(dstp + 5 * dst_stride + j);
        x6 = _mm256_load_ps(dstp + 6 * dst_stride + j);
        x7 = _mm256_load_ps(dstp + 7 * dst_stride + j);

#define SOLVEB(x, up, x_last, j, m)\
        up = _mm256_set1_ps(upper[j + m]);\
        x = _mm256_fnmadd_ps(up, x_last, x);

        SOLVEB(x7, up, x_last, j, 7);
        SOLVEB(x6, up, x7, j, 6);
        SOLVEB(x5, up, x6, j, 5);
        SOLVEB(x4, up, x5, j, 4);
        SOLVEB(x3, up, x4, j, 3);
        SOLVEB(x2, up, x3, j, 2);
        SOLVEB(x1, up, x2, j, 1);
        SOLVEB(x0, up, x1, j, 0);

#undef SOLVEB

        x_last = x0;

        mm256_transpose8_ps(&x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7);

        _mm256_store_ps(dstp + j, x0);
        _mm256_store_ps(dstp + 1 * dst_stride + j, x1);
        _mm256_store_ps(dstp + 2 * dst_stride + j, x2);
        _mm256_store_ps(dstp + 3 * dst_stride + j, x3);
        _mm256_store_ps(dstp + 4 * dst_stride + j, x4);
        _mm256_store_ps(dstp + 5 * dst_stride + j, x5);
        _mm256_store_ps(dstp + 6 * dst_stride + j, x6);
        _mm256_store_ps(dstp + 7 * dst_stride + j, x7);
    }
}


/*
 * Horizontal solver that is specialized for systems with bandwidth 7.
 * It is faster than the generalized version, because it uses much
 * less load/store instructions.
 */
static void process_line8_h_b7_avx2(int width, int current_height, int *current_width, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                    int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                                    float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp, float * VS_RESTRICT temp)
{
    transpose_line_8x8_ps(temp, srcp, src_stride, 0, ceil_n(*current_width, 8));
    __m256 x0, x1, x2, x3, x4, x5, x6, x7;
    __m256 a0, a1, lo, up, di, x_last0, x_last1, x_last2;
    x_last0 = _mm256_setzero_ps();
    for (int j = 0; j < width; j += 8) {
        x0 = _mm256_setzero_ps();
        x1 = x0;
        x2 = x0;
        x3 = x0;
        x4 = x0;
        x5 = x0;
        x6 = x0;
        x7 = x0;

#define MATMULT(x, a0, a1, wl_idx, wr_idx, w_col, weights, temp, j, m)\
        for (int k = wl_idx[j + m]; k < wr_idx[j + m]; k++) {\
            a0 = _mm256_set1_ps(weights[(j + m) * w_col + k - wl_idx[j + m]]);\
            a1 = _mm256_load_ps(temp + k * 8);\
            x = _mm256_fmadd_ps(a0, a1, x);\
        }

        // A' b
        MATMULT(x0, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 0);
        MATMULT(x1, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 1);
        MATMULT(x2, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 2);
        MATMULT(x3, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 3);
        MATMULT(x4, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 4);
        MATMULT(x5, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 5);
        MATMULT(x6, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 6);
        MATMULT(x7, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 7);

#undef MATMULT

#define SOLVEF(x, lo, di, x_last0, x_last1, x_last2, j, m)\
        if (j + m > 2) {\
            lo = _mm256_set1_ps(lower[0][j + m]);\
            x = _mm256_fnmadd_ps(lo, x_last2, x);\
            lo = _mm256_set1_ps(lower[1][j + m]);\
            x = _mm256_fnmadd_ps(lo, x_last1, x);\
            lo = _mm256_set1_ps(lower[2][j + m]);\
            x = _mm256_fnmadd_ps(lo, x_last0, x);\
        } else if (j + m > 1) {\
            lo = _mm256_set1_ps(lower[0][j + m]);\
            x = _mm256_fnmadd_ps(lo, x_last1, x);\
            lo = _mm256_set1_ps(lower[1][j + m]);\
            x = _mm256_fnmadd_ps(lo, x_last0, x);\
        } else if (j + m > 0) {\
            lo = _mm256_set1_ps(lower[0][j + m]);\
            x = _mm256_fnmadd_ps(lo, x_last0, x);\
        }\
        di = _mm256_set1_ps(diagonal[j + m]);\
        x = _mm256_mul_ps(x, di);

        // Solve LD y = A' b
        SOLVEF(x0, lo, di, x_last0, x_last1, x_last2, j, 0);
        SOLVEF(x1, lo, di, x0, x_last0, x_last1, j, 1);
        SOLVEF(x2, lo, di, x1, x0, x_last0, j, 2);
        SOLVEF(x3, lo, di, x2, x1, x0, j, 3);
        SOLVEF(x4, lo, di, x3, x2, x1, j, 4);
        SOLVEF(x5, lo, di, x4, x3, x2, j, 5);
        SOLVEF(x6, lo, di, x5, x4, x3, j, 6);
        SOLVEF(x7, lo, di, x6, x5, x4, j, 7);

#undef SOLVEF

        x_last0 = x7;
        x_last1 = x6;
        x_last2 = x5;

        _mm256_store_ps(dstp + j, x0);
        _mm256_store_ps(dstp + 1 * dst_stride + j, x1);
        _mm256_store_ps(dstp + 2 * dst_stride + j, x2);
        _mm256_store_ps(dstp + 3 * dst_stride + j, x3);
        _mm256_store_ps(dstp + 4 * dst_stride + j, x4);
        _mm256_store_ps(dstp + 5 * dst_stride + j, x5);
        _mm256_store_ps(dstp + 6 * dst_stride + j, x6);
        _mm256_store_ps(dstp + 7 * dst_stride + j, x7);
    }

    // Solve L' x = y
    for (int j = ceil_n(width, 8) - 8; j >= 0; j -= 8) {

        x0 = _mm256_load_ps(dstp + j);
        x1 = _mm256_load_ps(dstp + 1 * dst_stride + j);
        x2 = _mm256_load_ps(dstp + 2 * dst_stride + j);
        x3 = _mm256_load_ps(dstp + 3 * dst_stride + j);
        x4 = _mm256_load_ps(dstp + 4 * dst_stride + j);
        x5 = _mm256_load_ps(dstp + 5 * dst_stride + j);
        x6 = _mm256_load_ps(dstp + 6 * dst_stride + j);
        x7 = _mm256_load_ps(dstp + 7 * dst_stride + j);

#define SOLVEB(x, up, x_last0, x_last1, x_last2, width, j, m)\
        if (j + m < width - 3) {\
            up = _mm256_set1_ps(upper[0][j + m]);\
            x = _mm256_fnmadd_ps(up, x_last0, x);\
            up = _mm256_set1_ps(upper[1][j + m]);\
            x = _mm256_fnmadd_ps(up, x_last1, x);\
            up = _mm256_set1_ps(upper[2][j + m]);\
            x = _mm256_fnmadd_ps(up, x_last2, x);\
        } else if (j + m < width - 2) {\
            up = _mm256_set1_ps(upper[1][j + m]);\
            x = _mm256_fnmadd_ps(up, x_last0, x);\
            up = _mm256_set1_ps(upper[2][j + m]);\
            x = _mm256_fnmadd_ps(up, x_last1, x);\
        } else if (j + m < width - 1) {\
            up = _mm256_set1_ps(upper[2][j + m]);\
            x = _mm256_fnmadd_ps(up, x_last0, x);\
        }

        SOLVEB(x7, up, x_last0, x_last1, x_last2, width, j, 7);
        SOLVEB(x6, up, x7, x_last0, x_last1, width, j, 6);
        SOLVEB(x5, up, x6, x7, x_last0, width, j, 5);
        SOLVEB(x4, up, x5, x6, x7, width, j, 4);
        SOLVEB(x3, up, x4, x5, x6, width, j, 3);
        SOLVEB(x2, up, x3, x4, x5, width, j, 2);
        SOLVEB(x1, up, x2, x3, x4, width, j, 1);
        SOLVEB(x0, up, x1, x2, x3, width, j, 0);

#undef SOLVEB

        x_last0 = x0;
        x_last1 = x1;
        x_last2 = x2;

        mm256_transpose8_ps(&x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7);

        _mm256_store_ps(dstp + j, x0);
        _mm256_store_ps(dstp + 1 * dst_stride + j, x1);
        _mm256_store_ps(dstp + 2 * dst_stride + j, x2);
        _mm256_store_ps(dstp + 3 * dst_stride + j, x3);
        _mm256_store_ps(dstp + 4 * dst_stride + j, x4);
        _mm256_store_ps(dstp + 5 * dst_stride + j, x5);
        _mm256_store_ps(dstp + 6 * dst_stride + j, x6);
        _mm256_store_ps(dstp + 7 * dst_stride + j, x7);
    }
}


/*
 * This is a more general solver that has much more load/store
 * instructions than the specialized solvers for bandwidths 3 and 7.
 * The bandwidth can be arbitrarily high, meaning the solver
 * could need arbitarily many past already computed values,
 * so this general implementation just stores values immediately
 * and loads them again when needed.
*/
static void process_line8_h_avx2(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                 int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                                 float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp, float * VS_RESTRICT temp)
{
    __m256 x0, x1, x2, x3, x4, x5, x6, x7;
    __m256 a0, a1, lo, up, di, x_last;
    int start;
    int c = (bandwidth + 1) / 2;
    x_last = _mm256_setzero_ps();
    transpose_line_8x8_ps(temp, srcp, src_stride, 0, ceil_n(*current_width, 8));

    for (int j = 0; j < width; j += 8) {
        x0 = _mm256_setzero_ps();
        x1 = x0;
        x2 = x0;
        x3 = x0;
        x4 = x0;
        x5 = x0;
        x6 = x0;
        x7 = x0;

#define MATMULT(x, a0, a1, wl_idx, wr_idx, w_col, weights, temp, j, m)\
        for (int k = wl_idx[j + m]; k < wr_idx[j + m]; k++) {\
            a0 = _mm256_set1_ps(weights[(j + m) * w_col + k - wl_idx[j + m]]);\
            a1 = _mm256_load_ps(temp + k * 8);\
            x = _mm256_fmadd_ps(a0, a1, x);\
        }

        // A' b
        MATMULT(x0, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 0);
        MATMULT(x1, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 1);
        MATMULT(x2, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 2);
        MATMULT(x3, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 3);
        MATMULT(x4, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 4);
        MATMULT(x5, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 5);
        MATMULT(x6, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 6);
        MATMULT(x7, a0, a1, weights_left_idx, weights_right_idx, weights_columns, weights, temp, j, 7);

#undef MATMULT

#define SOLVESTOREF(x, lo, di, c, start, j, m)\
        start = VSMAX(0, j + m - c + 1);\
        for (int k = start; k < (j + m); k++) {\
            lo = _mm256_set1_ps(lower[k - start][j + m]);\
            x_last = _mm256_load_ps(dstp + (k % 8) * dst_stride + j - 8 * ((j + m) / 8 - k / 8));\
            x = _mm256_fnmadd_ps(lo, x_last, x);\
        }\
        di = _mm256_set1_ps(diagonal[j + m]);\
        x = _mm256_mul_ps(x, di);\
        _mm256_store_ps(dstp + m * dst_stride + j, x);

        SOLVESTOREF(x0, lo, di, c, start, j, 0);
        SOLVESTOREF(x1, lo, di, c, start, j, 1);
        SOLVESTOREF(x2, lo, di, c, start, j, 2);
        SOLVESTOREF(x3, lo, di, c, start, j, 3);
        SOLVESTOREF(x4, lo, di, c, start, j, 4);
        SOLVESTOREF(x5, lo, di, c, start, j, 5);
        SOLVESTOREF(x6, lo, di, c, start, j, 6);
        SOLVESTOREF(x7, lo, di, c, start, j, 7);

#undef SOLVESTOREF
    }

    // Solve L' x = y
    for (int j = ceil_n(width, 8) - 8; j >= 0; j -= 8) {

#define SOLVESTOREB(x, up, c, start, j, m)\
        x = _mm256_load_ps(dstp + m * dst_stride + j);\
        start = VSMIN(width - 1, j + m + c - 1);\
        for (int k = start; k > (j + m); k--) {\
            up = _mm256_set1_ps(upper[k - start + c - 2][j + m]);\
            x_last = _mm256_load_ps(dstp + (k % 8) * dst_stride + j + 8 * (k / 8 - (j + m) / 8));\
            x = _mm256_fnmadd_ps(up, x_last, x);\
        }\
        _mm256_store_ps(dstp + m * dst_stride + j, x);

        SOLVESTOREB(x0, up, c, start, j, 7);
        SOLVESTOREB(x0, up, c, start, j, 6);
        SOLVESTOREB(x0, up, c, start, j, 5);
        SOLVESTOREB(x0, up, c, start, j, 4);
        SOLVESTOREB(x0, up, c, start, j, 3);
        SOLVESTOREB(x0, up, c, start, j, 2);
        SOLVESTOREB(x0, up, c, start, j, 1);
        SOLVESTOREB(x0, up, c, start, j, 0);

#undef SOLVESTOREB
    }

    for (int j = 0; j < width; j += 8) {
        x0 = _mm256_load_ps(dstp + j);
        x1 = _mm256_load_ps(dstp + 1 * dst_stride + j);
        x2 = _mm256_load_ps(dstp + 2 * dst_stride + j);
        x3 = _mm256_load_ps(dstp + 3 * dst_stride + j);
        x4 = _mm256_load_ps(dstp + 4 * dst_stride + j);
        x5 = _mm256_load_ps(dstp + 5 * dst_stride + j);
        x6 = _mm256_load_ps(dstp + 6 * dst_stride + j);
        x7 = _mm256_load_ps(dstp + 7 * dst_stride + j);

        mm256_transpose8_ps(&x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7);

        _mm256_store_ps(dstp + j, x0);
        _mm256_store_ps(dstp + 1 * dst_stride + j, x1);
        _mm256_store_ps(dstp + 2 * dst_stride + j, x2);
        _mm256_store_ps(dstp + 3 * dst_stride + j, x3);
        _mm256_store_ps(dstp + 4 * dst_stride + j, x4);
        _mm256_store_ps(dstp + 5 * dst_stride + j, x5);
        _mm256_store_ps(dstp + 6 * dst_stride + j, x6);
        _mm256_store_ps(dstp + 7 * dst_stride + j, x7);
    }
}


void process_plane_h_b3_avx2(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                    int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                                    float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    float * VS_RESTRICT temp;
    VS_ALIGNED_MALLOC(&temp, ceil_n(*current_width, 8) * 8 * sizeof (float), 32);

    for (int i = 0; i < floor_n(current_height, 8); i += 8) {

        process_line8_h_b3_avx2(width, current_height, current_width, weights_left_idx, weights_right_idx, weights_columns, weights,
                                lower[0], upper[0], diagonal, src_stride, dst_stride, srcp, dstp, temp);

        srcp += src_stride * 8;
        dstp += dst_stride * 8;
    }

    if (floor_n(current_height, 8) != current_height) {

        srcp -= src_stride * (8 - (current_height - floor_n(current_height, 8)));
        dstp -= dst_stride * (8 - (current_height - floor_n(current_height, 8)));

        process_line8_h_b3_avx2(width, current_height, current_width, weights_left_idx, weights_right_idx, weights_columns, weights,
                                lower[0], upper[0], diagonal, src_stride, dst_stride, srcp, dstp, temp);
    }

    VS_ALIGNED_FREE(temp);
    *current_width = width;
}


void process_plane_h_b7_avx2(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                    int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                                    float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    float * VS_RESTRICT temp;
    VS_ALIGNED_MALLOC(&temp, ceil_n(*current_width, 8) * 8 * sizeof (float), 32);

    for (int i = 0; i < floor_n(current_height, 8); i += 8) {

        process_line8_h_b7_avx2(width, current_height, current_width, weights_left_idx, weights_right_idx, weights_columns, weights,
                                lower, upper, diagonal, src_stride, dst_stride, srcp, dstp, temp);

        srcp += src_stride * 8;
        dstp += dst_stride * 8;
    }

    if (floor_n(current_height, 8) != current_height) {

        srcp -= src_stride * (8 - (current_height - floor_n(current_height, 8)));
        dstp -= dst_stride * (8 - (current_height - floor_n(current_height, 8)));

        process_line8_h_b7_avx2(width, current_height, current_width, weights_left_idx, weights_right_idx, weights_columns, weights,
                                lower, upper, diagonal, src_stride, dst_stride, srcp, dstp, temp);
    }

    VS_ALIGNED_FREE(temp);
    *current_width = width;
}


void process_plane_h_avx2(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                 int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                                 float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    float * VS_RESTRICT temp;
    VS_ALIGNED_MALLOC(&temp, ceil_n(*current_width, 8) * 8 * sizeof (float), 32);

    for (int i = 0; i < floor_n(current_height, 8); i += 8) {

        process_line8_h_avx2(width, current_height, current_width, bandwidth, weights_left_idx, weights_right_idx, weights_columns, weights,
                                lower, upper, diagonal, src_stride, dst_stride, srcp, dstp, temp);

        srcp += src_stride * 8;
        dstp += dst_stride * 8;
    }

    if (floor_n(current_height, 8) != current_height) {

        srcp -= src_stride * (8 - (current_height - floor_n(current_height, 8)));
        dstp -= dst_stride * (8 - (current_height - floor_n(current_height, 8)));

        process_line8_h_avx2(width, current_height, current_width, bandwidth, weights_left_idx, weights_right_idx, weights_columns, weights,
                                lower, upper, diagonal, src_stride, dst_stride, srcp, dstp, temp);
    }

    VS_ALIGNED_FREE(temp);
    *current_width = width;
}


/*
 * Unlike the horizontal specialized solver, this vertical one
 * is just slightly faster than the generalized version.
 * To keep past values in the registers, we would have to do the vertical
 * pass actually vertically instead of horizontally, this would lead to
 * a worse memory acess pattern, and is actually slower than using
 * additional load/store instructions.
 */
void process_plane_v_b3_avx2(int height, int current_width, int *current_height, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                    int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower2, float * VS_RESTRICT * VS_RESTRICT upper2,
float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    float * VS_RESTRICT lower = lower2[0];
    float * VS_RESTRICT upper = upper2[0];
    __m256 x, a0, a1, lo, up, di, x_last;
    for (int i = 0; i < height; i++) {

        for (int j = 0; j < current_width; j += 8) {
            x = _mm256_setzero_ps();

            // A' b
            for (int k = weights_left_idx[i]; k < weights_right_idx[i]; k++) {
                a0 = _mm256_set1_ps(weights[i * weights_columns + k - weights_left_idx[i]]);
                a1 = _mm256_load_ps(srcp + k * src_stride + j);
                x = _mm256_fmadd_ps(a0, a1, x);
            }

            // Solve LD y = A' b
            if (i != 0) {
                lo = _mm256_set1_ps(lower[i]);
                x_last = _mm256_load_ps(dstp + (i - 1) * dst_stride + j);
                x = _mm256_fnmadd_ps(lo, x_last, x);
            }
            di = _mm256_set1_ps(diagonal[i]);
            x = _mm256_mul_ps(x, di);
            _mm256_store_ps(dstp + i * dst_stride + j, x);
        
            
        }
    }

    // Solve L' x = y
    for (int i = height - 2; i >= 0; i--) {
        for (int j = 0; j < current_width; j += 8) {
            x = _mm256_load_ps(&dstp[i * dst_stride + j]);
            x_last = _mm256_load_ps(dstp + (i + 1) * dst_stride + j);
            up = _mm256_set1_ps(upper[i]);
            x = _mm256_fnmadd_ps(up, x_last, x);
            _mm256_store_ps(dstp + i * dst_stride + j, x);
        }
    }

    *current_height = height;
}


/*
 * Unlike the horizontal specialized solver, this vertical one
 * is just slightly faster than the generalized version.
 * To keep past values in the registers, we would have to do the vertical
 * pass actually vertically instead of horizontally, this would lead to
 * a worse memory acess pattern, and is actually slower than using
 * additional load/store instructions.
 */
void process_plane_v_b7_avx2(int height, int current_width, int *current_height, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                    int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                                    float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    __m256 x, a0, a1, lo, up, di, x_last;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < current_width; j += 8) {
            x = _mm256_setzero_ps();

            // A' b
            for (int k = weights_left_idx[i]; k < weights_right_idx[i]; k++) {
                a0 = _mm256_set1_ps(weights[i * weights_columns + k - weights_left_idx[i]]);
                a1 = _mm256_load_ps(srcp + k * src_stride + j);
                x = _mm256_fmadd_ps(a0, a1, x);
            }

            // Solve LD y = A' b
            if (i > 2) {
                lo = _mm256_set1_ps(lower[0][i]);
                x_last = _mm256_load_ps(dstp + (i - 3) * dst_stride + j);
                x = _mm256_fnmadd_ps(lo, x_last, x);
                lo = _mm256_set1_ps(lower[1][i]);
                x_last = _mm256_load_ps(dstp + (i - 2) * dst_stride + j);
                x = _mm256_fnmadd_ps(lo, x_last, x);
                lo = _mm256_set1_ps(lower[2][i]);
                x_last = _mm256_load_ps(dstp + (i - 1) * dst_stride + j);
                x = _mm256_fnmadd_ps(lo, x_last, x);
            } else if (i > 1) {
                lo = _mm256_set1_ps(lower[0][i]);
                x_last = _mm256_load_ps(dstp + (i - 2) * dst_stride + j);
                x = _mm256_fnmadd_ps(lo, x_last, x);
                lo = _mm256_set1_ps(lower[1][i]);
                x_last = _mm256_load_ps(dstp + (i - 1) * dst_stride + j);
                x = _mm256_fnmadd_ps(lo, x_last, x);
            } else if (i > 0) {
                lo = _mm256_set1_ps(lower[0][i]);
                x_last = _mm256_load_ps(dstp + (i - 1) * dst_stride + j);
                x = _mm256_fnmadd_ps(lo, x_last, x);
            }
            di = _mm256_set1_ps(diagonal[i]);
            x = _mm256_mul_ps(x, di);
            _mm256_store_ps(dstp + i * dst_stride + j, x);
        }
    }

    // Solve L' x = y
    for (int i = height - 2; i >= 0; i--) {
        for (int j = 0; j < current_width; j += 8) {

            x = _mm256_load_ps(dstp + i * dst_stride + j);

            if (i < height - 3) {
                up = _mm256_set1_ps(upper[0][i]);
                x_last = _mm256_load_ps(dstp + (i + 1) * dst_stride + j);
                x = _mm256_fnmadd_ps(up, x_last, x);
                up = _mm256_set1_ps(upper[1][i]);
                x_last = _mm256_load_ps(dstp + (i + 2) * dst_stride + j);
                x = _mm256_fnmadd_ps(up, x_last, x);
                up = _mm256_set1_ps(upper[2][i]);
                x_last = _mm256_load_ps(dstp + (i + 3) * dst_stride + j);
                x = _mm256_fnmadd_ps(up, x_last, x);
            } else if (i < height - 2) {
                up = _mm256_set1_ps(upper[1][i]);
                x_last = _mm256_load_ps(dstp + (i + 1) * dst_stride + j);
                x = _mm256_fnmadd_ps(up, x_last, x);
                up = _mm256_set1_ps(upper[2][i]);
                x_last = _mm256_load_ps(dstp + (i + 2) * dst_stride + j);
                x = _mm256_fnmadd_ps(up, x_last, x);
            } else if (i < height - 1) {
                up = _mm256_set1_ps(upper[2][i]);
                x_last = _mm256_load_ps(dstp + (i + 1) * dst_stride + j);
                x = _mm256_fnmadd_ps(up, x_last, x);
            }
            _mm256_store_ps(dstp + i * dst_stride + j, x);
        }
    }

    *current_height = height;
}


/*
 * General version of the vertical solver.
 */
void process_plane_v_avx2(int height, int current_width, int *current_height, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                 int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                                 float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    __m256 x, a0, a1, lo, up, di, x_last;
    int start;
    int c = (bandwidth + 1) / 2;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < current_width; j += 8) {
            x = _mm256_setzero_ps();

            // A' b
            for (int k = weights_left_idx[i]; k < weights_right_idx[i]; k++) {
                a0 = _mm256_set1_ps(weights[i * weights_columns + k - weights_left_idx[i]]);
                a1 = _mm256_load_ps(srcp + k * src_stride + j);
                x = _mm256_fmadd_ps(a0, a1, x);
            }

            // Solve LD y = A' b
            start = VSMAX(0, i - c + 1);
            for (int k = start; k < i; k++) {
                lo = _mm256_set1_ps(lower[k - start][i]);
                x_last = _mm256_load_ps(dstp + k * dst_stride + j);
                x = _mm256_fnmadd_ps(lo, x_last, x);
            }
            di = _mm256_set1_ps(diagonal[i]);
            x = _mm256_mul_ps(x, di);
            _mm256_store_ps(dstp + i * dst_stride + j, x);            
        }
    }

    // Solve L' x = y
    for (int i = height - 2; i >= 0; i--) {
        for (int j = 0; j < current_width; j += 8) {

            x = _mm256_load_ps(dstp + i * dst_stride + j);
            start = VSMIN(height - 1, i + c - 1);
            for (int k = start; k > i; k--) {
                up = _mm256_set1_ps(upper[k - start + c - 2][i]);
                x_last = _mm256_load_ps(dstp + k * dst_stride + j);
                x = _mm256_fnmadd_ps(up, x_last, x);
            }
            _mm256_store_ps(dstp + i * dst_stride + j, x);
        }
    }

    *current_height = height;
}
#endif  // DESCALE_X86
