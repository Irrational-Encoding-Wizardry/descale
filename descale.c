/* 
 * Copyright © 2020 Frechdachs <frechdachs@rekt.cc>
 * This program is free software. It comes without any warranty, to
 * the extent permitted by applicable law. You can redistribute it
 * and/or modify it under the terms of the Do What The Fuck You Want
 * To Public License, Version 2, as published by Sam Hocevar.
 * See the COPYING file for more details.
 */


#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <vapoursynth/VapourSynth.h>
#include <vapoursynth/VSHelper.h>


#ifdef __AVX2__
    #include <immintrin.h>
    #define process_plane_h_b3 process_plane_h_b3_avx2
    #define process_plane_v_b3 process_plane_v_b3_avx2
    #define process_plane_h_b7 process_plane_h_b7_avx2
    #define process_plane_v_b7 process_plane_v_b7_avx2
    #define process_plane_h process_plane_h_avx2
    #define process_plane_v process_plane_v_avx2
#else
    #define process_plane_h_b3 process_plane_h_b3_c
    #define process_plane_v_b3 process_plane_v_b3_c
    #define process_plane_h_b7 process_plane_h_b7_c
    #define process_plane_v_b7 process_plane_v_b7_c
    #define process_plane_h process_plane_h_c
    #define process_plane_v process_plane_v_c
#endif


struct DescaleData
{
    VSNodeRef *node;
    VSVideoInfo vi_src;
    VSVideoInfo vi_dst;
    int bandwidth;
    int taps;
    double b, c;
    float shift_h;
    bool process_h;
    float **upper_h;
    float **lower_h;
    float *diagonal_h;
    float *weights_h;
    int *weights_h_left_idx;
    int *weights_h_right_idx;
    int weights_h_columns;
    float shift_v;
    bool process_v;
    float **upper_v;
    float **lower_v;
    float *diagonal_v;
    float *weights_v;
    int *weights_v_left_idx;
    int *weights_v_right_idx;
    int weights_v_columns;
};


enum DescaleMode
{
    bilinear = 0,
    bicubic  = 1,
    lanczos  = 2,
    spline16 = 3,
    spline36 = 4,
    spline64 = 5
};


static int ceil_n(int x, int n)
{
    return (x + (n - 1)) & ~(n - 1);
}


static int floor_n(int x, int n)
{
    return x & ~(n - 1);
}


static void multiply_banded_matrix_with_diagonal(int rows, int bandwidth, double *matrix)
{
    int c = (bandwidth + 1) / 2;

    for (int i = 1; i < rows; i++) {
        int start = VSMAX(i - (c - 1), 0);
        for (int j = start; j < i; j++) {
            matrix[i * rows + j] *= matrix[j * rows + j];
        }
    }
}


/*
 * LDLT decomposition (variant of Cholesky decomposition)
 * Input is a banded symmetrical matrix, the lower part is ignored.
 * The upper part of the input matrix is modified in-place and
 * contains L' and D after decomposition. The main diagonal of
 * ones of L' is not saved.
*/
static void banded_ldlt_decomposition(int rows, int bandwidth, double *matrix)
{
    int c = (bandwidth + 1) / 2;
    // Division by 0 can happen if shift is used
    double eps = DBL_EPSILON;

    for (int i = 0; i < rows; i++) {
        int end = VSMIN(c, rows - i);

        for (int j = 1; j < end; j++) {
            double d = matrix[i * rows + i + j] / (matrix[i * rows + i] + eps);

            for (int k = 0; k < end - j; k++) {
                matrix[(i + j) * rows + i + j + k] -= d * matrix[i * rows + i + j + k];
            }
        }

        double e = 1.0 / (matrix[i * rows + i] + eps);
        for (int j = 1; j < end; j++) {
                matrix[i * rows + i + j] *= e;
        }
    }
}


static void multiply_sparse_matrices(int rows, int columns, const int *lidx, const int *ridx, const double *lm, const double *rm, double **multiplied)
{
    *multiplied = calloc(rows * columns, sizeof (double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            double sum = 0;

            for (int k = lidx[i]; k < ridx[i]; k++) {
                sum += lm[i * columns + k] * rm[k * rows + j];
            }

            (*multiplied)[i * rows + j] = sum;
        }
    }
}


static void transpose_matrix(int rows, int columns, const double *matrix, double **transposed_matrix)
{
    *transposed_matrix = calloc(columns * rows, sizeof (double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            (*transposed_matrix)[i + rows * j] = matrix[i * columns + j];
        }
    }
}


static void extract_compressed_lower_upper_diagonal(int rows, int bandwidth, const double *lower, const double *upper, float ***compressed_lower, float ***compressed_upper, float **diagonal)
{
    *compressed_lower = calloc(bandwidth / 2, sizeof (float *));
    *compressed_upper = calloc(bandwidth / 2, sizeof (float *));
    *diagonal = calloc(ceil_n(rows, 8), sizeof (float));
    int c = (bandwidth + 1) / 2;
    // Division by 0 can happen if shift is used
    double eps = DBL_EPSILON;

    for (int i = 0; i < c - 1; i++) {
        (*compressed_lower)[i] = calloc(ceil_n(rows, 8), sizeof (float));
        (*compressed_upper)[i] = calloc(ceil_n(rows, 8), sizeof (float));
    }

    for (int i = 0; i < rows; i++) {
        int start = VSMAX(i - c + 1, 0);
        for (int j = start; j < start + c - 1; j++) {
            (*compressed_lower)[j - start][i] = (float)lower[i * rows + j];
        }
    }

    for (int i = 0; i < rows; i++) {
        int start = VSMIN(i + c - 1, rows - 1);
        for (int j = start; j > i; j--) {
            (*compressed_upper)[c - 2 + j - start][i] = (float)upper[i * rows + j];
        }
    }

    for (int i = 0; i < rows; i++) {
        (*diagonal)[i] = (float)(1.0 / (lower[i * rows + i] + eps));
    }

}


#define PI 3.14159265358979323846


static double sinc(double x)
{
    return x == 0.0 ? 1.0 : sin(x * PI) / (x * PI);
}


static double square(double x)
{
    return x * x;
}


static double cube(double x)
{
    return x * x * x;
}


static double calculate_weight(enum DescaleMode mode, int support, double distance, double b, double c)
{
    distance = fabs(distance);

    if (mode == bilinear) {
        return VSMAX(1.0 - distance, 0.0);

    } else if (mode == bicubic) {
        if (distance < 1)
            return ((12 - 9 * b - 6 * c) * cube(distance)
                        + (-18 + 12 * b + 6 * c) * square(distance) + (6 - 2 * b)) / 6.0;
        else if (distance < 2) 
            return ((-b - 6 * c) * cube(distance) + (6 * b + 30 * c) * square(distance)
                        + (-12 * b - 48 * c) * distance + (8 * b + 24 * c)) / 6.0;
        else
            return 0.0;

    } else if (mode == lanczos) {
        return distance < support ? sinc(distance) * sinc(distance / support) : 0.0;

    } else if (mode == spline16) {
        if (distance < 1.0) {
            return 1.0 - (1.0 / 5.0 * distance) - (9.0 / 5.0 * square(distance)) + cube(distance);
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-7.0 / 15.0 * distance) + (4.0 / 5.0 * square(distance)) - (1.0 / 3.0 * cube(distance));
        } else {
            return 0.0;
        }

    } else if (mode == spline36) {
        if (distance < 1.0) {
            return 1.0 - (3.0 / 209.0 * distance) - (453.0 / 209.0 * square(distance)) + (13.0 / 11.0 * cube(distance));
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-156.0 / 209.0 * distance) + (270.0 / 209.0 * square(distance)) - (6.0 / 11.0 * cube(distance));
        } else if (distance < 3.0) {
            distance -= 2.0;
            return (26.0 / 209.0 * distance) - (45.0 / 209.0 * square(distance)) + (1.0 / 11.0 * cube(distance));
        } else {
            return 0.0;
        }

    } else if (mode == spline64) {
        if (distance < 1.0) {
            return 1.0 - (3.0 / 2911.0 * distance) - (6387.0 / 2911.0 * square(distance)) + (49.0 / 41.0 * cube(distance));
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-2328.0 / 2911.0 * distance) + (4032.0 / 2911.0 * square(distance)) - (24.0 / 41.0 * cube(distance));
        } else if (distance < 3.0) {
            distance -= 2.0;
            return (582.0 / 2911.0 * distance) - (1008.0 / 2911.0 * square(distance)) + (6.0 / 41.0 * cube(distance));
        } else if (distance < 4.0) {
            distance -= 3.0;
            return (-97.0 / 2911.0 * distance) + (168.0 / 2911.0 * square(distance)) - (1.0 / 41.0 * cube(distance));
        } else {
            return 0.0;
        }
    }
}


// Taken from zimg https://github.com/sekrit-twc/zimg
static double round_halfup(double x)
{
    /* When rounding on the pixel grid, the invariant
     *   round(x - 1) == round(x) - 1
     * must be preserved. This precludes the use of modes such as
     * half-to-even and half-away-from-zero.
     */
    bool sign = (x < 0);

    x = round(fabs(x));
    return sign ? -x : x;
}


// Most of this is taken from zimg 
// https://github.com/sekrit-twc/zimg/blob/ce27c27f2147fbb28e417fbf19a95d3cf5d68f4f/src/zimg/resize/filter.cpp#L227
static void scaling_weights(enum DescaleMode mode, int support, int src_dim, int dst_dim, double b, double c, double shift, double **weights)
{
    *weights = calloc(src_dim * dst_dim, sizeof (double));
    double ratio = (double)dst_dim / src_dim;

    for (int i = 0; i < dst_dim; i++) {

        double total = 0.0;
        double pos = (i + 0.5) / ratio + shift;
        double begin_pos = round_halfup(pos - support) + 0.5;
        for (int j = 0; j < 2 * support; j++) {
            double xpos = begin_pos + j;
            total += calculate_weight(mode, support, xpos - pos, b, c);
        }
        for (int j = 0; j < 2 * support; j++) {
            double xpos = begin_pos + j;
            double real_pos;

            // Mirror the position if it goes beyond image bounds.
            if (xpos < 0.0)
                real_pos = -xpos;
            else if (xpos >= src_dim)
                real_pos = VSMIN(2.0 * src_dim - xpos, src_dim - 0.5);
            else
                real_pos = xpos;

            int idx = (int)floor(real_pos);
            (*weights)[i * src_dim + idx] += calculate_weight(mode, support, xpos - pos, b, c) / total;
        }
    }
}


#ifdef __AVX2__
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

#define SOLVEF(x, a0, lo, di, x_last, j, m)\
        lo = _mm256_set1_ps(lower[j + m]);\
        a0 = _mm256_mul_ps(lo, x_last);\
        x = _mm256_sub_ps(x, a0);\
        di = _mm256_set1_ps(diagonal[j + m]);\
        x = _mm256_mul_ps(x, di);

        // Solve LD y = A' b
        SOLVEF(x0, a0, lo, di, x_last, j, 0);
        SOLVEF(x1, a0, lo, di, x0, j, 1);
        SOLVEF(x2, a0, lo, di, x1, j, 2);
        SOLVEF(x3, a0, lo, di, x2, j, 3);
        SOLVEF(x4, a0, lo, di, x3, j, 4);
        SOLVEF(x5, a0, lo, di, x4, j, 5);
        SOLVEF(x6, a0, lo, di, x5, j, 6);
        SOLVEF(x7, a0, lo, di, x6, j, 7);

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

#define SOLVEB(x, a0, up, x_last, j, m)\
        up = _mm256_set1_ps(upper[j + m]);\
        a0 = _mm256_mul_ps(up, x_last);\
        x = _mm256_sub_ps(x, a0);\

        SOLVEB(x7, a0, up, x_last, j, 7);
        SOLVEB(x6, a0, up, x7, j, 6);
        SOLVEB(x5, a0, up, x6, j, 5);
        SOLVEB(x4, a0, up, x5, j, 4);
        SOLVEB(x3, a0, up, x4, j, 3);
        SOLVEB(x2, a0, up, x3, j, 2);
        SOLVEB(x1, a0, up, x2, j, 1);
        SOLVEB(x0, a0, up, x1, j, 0);

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

#define SOLVEF(x, a0, lo, di, x_last0, x_last1, x_last2, j, m)\
        a0 = _mm256_setzero_ps();\
        if (j + m > 2) {\
            lo = _mm256_set1_ps(lower[0][j + m]);\
            a0 = _mm256_fmadd_ps(lo, x_last2, a0);\
            lo = _mm256_set1_ps(lower[1][j + m]);\
            a0 = _mm256_fmadd_ps(lo, x_last1, a0);\
            lo = _mm256_set1_ps(lower[2][j + m]);\
            a0 = _mm256_fmadd_ps(lo, x_last0, a0);\
        } else if (j + m > 1) {\
            lo = _mm256_set1_ps(lower[0][j + m]);\
            a0 = _mm256_fmadd_ps(lo, x_last1, a0);\
            lo = _mm256_set1_ps(lower[1][j + m]);\
            a0 = _mm256_fmadd_ps(lo, x_last0, a0);\
        } else if (j + m > 0) {\
            lo = _mm256_set1_ps(lower[0][j + m]);\
            a0 = _mm256_fmadd_ps(lo, x_last0, a0);\
        }\
        x = _mm256_sub_ps(x, a0);\
        di = _mm256_set1_ps(diagonal[j + m]);\
        x = _mm256_mul_ps(x, di);

        // Solve LD y = A' b
        SOLVEF(x0, a0, lo, di, x_last0, x_last1, x_last2, j, 0);
        SOLVEF(x1, a0, lo, di, x0, x_last0, x_last1, j, 1);
        SOLVEF(x2, a0, lo, di, x1, x0, x_last0, j, 2);
        SOLVEF(x3, a0, lo, di, x2, x1, x0, j, 3);
        SOLVEF(x4, a0, lo, di, x3, x2, x1, j, 4);
        SOLVEF(x5, a0, lo, di, x4, x3, x2, j, 5);
        SOLVEF(x6, a0, lo, di, x5, x4, x3, j, 6);
        SOLVEF(x7, a0, lo, di, x6, x5, x4, j, 7);


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

#define SOLVEB(x, a0, up, x_last0, x_last1, x_last2, width, j, m)\
        a0 = _mm256_setzero_ps();\
        if (j + m < width - 3) {\
            up = _mm256_set1_ps(upper[0][j + m]);\
            a0 = _mm256_fmadd_ps(up, x_last0, a0);\
            up = _mm256_set1_ps(upper[1][j + m]);\
            a0 = _mm256_fmadd_ps(up, x_last1, a0);\
            up = _mm256_set1_ps(upper[2][j + m]);\
            a0 = _mm256_fmadd_ps(up, x_last2, a0);\
        } else if (j + m < width - 2) {\
            up = _mm256_set1_ps(upper[1][j + m]);\
            a0 = _mm256_fmadd_ps(up, x_last0, a0);\
            up = _mm256_set1_ps(upper[2][j + m]);\
            a0 = _mm256_fmadd_ps(up, x_last1, a0);\
        } else if (j + m < width - 1) {\
            up = _mm256_set1_ps(upper[2][j + m]);\
            a0 = _mm256_fmadd_ps(up, x_last0, a0);\
        }\
        x = _mm256_sub_ps(x, a0);\

        SOLVEB(x7, a0, up, x_last0, x_last1, x_last2, width, j, 7);
        SOLVEB(x6, a0, up, x7, x_last0, x_last1, width, j, 6);
        SOLVEB(x5, a0, up, x6, x7, x_last0, width, j, 5);
        SOLVEB(x4, a0, up, x5, x6, x7, width, j, 4);
        SOLVEB(x3, a0, up, x4, x5, x6, width, j, 3);
        SOLVEB(x2, a0, up, x3, x4, x5, width, j, 2);
        SOLVEB(x1, a0, up, x2, x3, x4, width, j, 1);
        SOLVEB(x0, a0, up, x1, x2, x3, width, j, 0);

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

#define SOLVESTOREF(x, a0, lo, di, c, start, j, m)\
        a0 = _mm256_setzero_ps();\
        start = VSMAX(0, j + m - c + 1);\
        for (int k = start; k < (j + m); k++) {\
            lo = _mm256_set1_ps(lower[k - start][j + m]);\
            x_last = _mm256_load_ps(dstp + (k % 8) * dst_stride + j - 8 * ((j + m) / 8 - k / 8));\
            a0 = _mm256_fmadd_ps(lo, x_last, a0);\
        }\
        x = _mm256_sub_ps(x, a0);\
        di = _mm256_set1_ps(diagonal[j + m]);\
        x = _mm256_mul_ps(x, di);\
        _mm256_store_ps(dstp + m * dst_stride + j, x);

        SOLVESTOREF(x0, a0, lo, di, c, start, j, 0);
        SOLVESTOREF(x1, a0, lo, di, c, start, j, 1);
        SOLVESTOREF(x2, a0, lo, di, c, start, j, 2);
        SOLVESTOREF(x3, a0, lo, di, c, start, j, 3);
        SOLVESTOREF(x4, a0, lo, di, c, start, j, 4);
        SOLVESTOREF(x5, a0, lo, di, c, start, j, 5);
        SOLVESTOREF(x6, a0, lo, di, c, start, j, 6);
        SOLVESTOREF(x7, a0, lo, di, c, start, j, 7);

#undef SOLVESTOREF
    }

    // Solve L' x = y
    for (int j = ceil_n(width, 8) - 8; j >= 0; j -= 8) {

#define SOLVESTOREB(x, a0, up, c, start, j, m)\
        x = _mm256_load_ps(dstp + m * dst_stride + j);\
        a0 = _mm256_setzero_ps();\
        start = VSMIN(width - 1, j + m + c - 1);\
        for (int k = start; k > (j + m); k--) {\
            up = _mm256_set1_ps(upper[k - start + c - 2][j + m]);\
            x_last = _mm256_load_ps(dstp + (k % 8) * dst_stride + j + 8 * (k / 8 - (j + m) / 8));\
            a0 = _mm256_fmadd_ps(up, x_last, a0);\
        }\
        x = _mm256_sub_ps(x, a0);\
        _mm256_store_ps(dstp + m * dst_stride + j, x);

        SOLVESTOREB(x0, a0, up, c, start, j, 7);
        SOLVESTOREB(x0, a0, up, c, start, j, 6);
        SOLVESTOREB(x0, a0, up, c, start, j, 5);
        SOLVESTOREB(x0, a0, up, c, start, j, 4);
        SOLVESTOREB(x0, a0, up, c, start, j, 3);
        SOLVESTOREB(x0, a0, up, c, start, j, 2);
        SOLVESTOREB(x0, a0, up, c, start, j, 1);
        SOLVESTOREB(x0, a0, up, c, start, j, 0);

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


static void process_plane_h_b3_avx2(int width, int current_height, int *current_width, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                    int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT lower, float * VS_RESTRICT upper, float * VS_RESTRICT diagonal,
                                    const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    float * VS_RESTRICT temp;
    VS_ALIGNED_MALLOC(&temp, ceil_n(*current_width, 8) * 8 * sizeof (float), 32);

    for (int i = 0; i < floor_n(current_height, 8); i += 8) {

        process_line8_h_b3_avx2(width, current_height, current_width, weights_left_idx, weights_right_idx, weights_columns, weights,
                                lower, upper, diagonal, src_stride, dst_stride, srcp, dstp, temp);

        srcp += src_stride * 8;
        dstp += dst_stride * 8;
    }

    if (floor_n(current_height, 8) != current_height) {

        srcp -= src_stride * (8 - (current_height - floor_n(current_height, 8)));
        dstp -= dst_stride * (8 - (current_height - floor_n(current_height, 8)));

        process_line8_h_b3_avx2(width, current_height, current_width, weights_left_idx, weights_right_idx, weights_columns, weights,
                                lower, upper, diagonal, src_stride, dst_stride, srcp, dstp, temp);
    }

    VS_ALIGNED_FREE(temp);
    *current_width = width;
}


static void process_plane_h_b7_avx2(int width, int current_height, int *current_width, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
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


static void process_plane_h_avx2(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
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
static void process_plane_v_b3_avx2(int height, int current_width, int *current_height, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                    int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT lower, float * VS_RESTRICT upper, float * VS_RESTRICT diagonal,
                                    const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
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
            if (i != 0) {
                lo = _mm256_set1_ps(lower[i]);
                x_last = _mm256_load_ps(dstp + (i - 1) * dst_stride + j);
                a0 = _mm256_mul_ps(lo, x_last);
                x = _mm256_sub_ps(x, a0);
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
            a0 = _mm256_mul_ps(up, x_last);
            x = _mm256_sub_ps(x, a0);
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
static void process_plane_v_b7_avx2(int height, int current_width, int *current_height, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                    int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                                    float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    __m256 x, a0, a1, lo, up, di, x_last;
    int start;

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
            a0 = _mm256_setzero_ps();
            if (i > 2) {
                lo = _mm256_set1_ps(lower[0][i]);
                x_last = _mm256_load_ps(dstp + (i - 3) * dst_stride + j);
                a0 = _mm256_fmadd_ps(lo, x_last, a0);
                lo = _mm256_set1_ps(lower[1][i]);
                x_last = _mm256_load_ps(dstp + (i - 2) * dst_stride + j);
                a0 = _mm256_fmadd_ps(lo, x_last, a0);
                lo = _mm256_set1_ps(lower[2][i]);
                x_last = _mm256_load_ps(dstp + (i - 1) * dst_stride + j);
                a0 = _mm256_fmadd_ps(lo, x_last, a0);
            } else if (i > 1) {
                lo = _mm256_set1_ps(lower[0][i]);
                x_last = _mm256_load_ps(dstp + (i - 2) * dst_stride + j);
                a0 = _mm256_fmadd_ps(lo, x_last, a0);
                lo = _mm256_set1_ps(lower[1][i]);
                x_last = _mm256_load_ps(dstp + (i - 1) * dst_stride + j);
                a0 = _mm256_fmadd_ps(lo, x_last, a0);
            } else if (i > 0) {
                lo = _mm256_set1_ps(lower[0][i]);
                x_last = _mm256_load_ps(dstp + (i - 1) * dst_stride + j);
                a0 = _mm256_fmadd_ps(lo, x_last, a0);
            }
            x = _mm256_sub_ps(x, a0);
            di = _mm256_set1_ps(diagonal[i]);
            x = _mm256_mul_ps(x, di);
            _mm256_store_ps(dstp + i * dst_stride + j, x);
        }
    }

    // Solve L' x = y
    for (int i = height - 2; i >= 0; i--) {
        for (int j = 0; j < current_width; j += 8) {

            x = _mm256_load_ps(dstp + i * dst_stride + j);
            a0 = _mm256_setzero_ps();

            if (i < height - 3) {
                up = _mm256_set1_ps(upper[0][i]);
                x_last = _mm256_load_ps(dstp + (i + 1) * dst_stride + j);
                a0 = _mm256_fmadd_ps(up, x_last, a0);
                up = _mm256_set1_ps(upper[1][i]);
                x_last = _mm256_load_ps(dstp + (i + 2) * dst_stride + j);
                a0 = _mm256_fmadd_ps(up, x_last, a0);
                up = _mm256_set1_ps(upper[2][i]);
                x_last = _mm256_load_ps(dstp + (i + 3) * dst_stride + j);
                a0 = _mm256_fmadd_ps(up, x_last, a0);
            } else if (i < height - 2) {
                up = _mm256_set1_ps(upper[1][i]);
                x_last = _mm256_load_ps(dstp + (i + 1) * dst_stride + j);
                a0 = _mm256_fmadd_ps(up, x_last, a0);
                up = _mm256_set1_ps(upper[2][i]);
                x_last = _mm256_load_ps(dstp + (i + 2) * dst_stride + j);
                a0 = _mm256_fmadd_ps(up, x_last, a0);
            } else if (i < height - 1) {
                up = _mm256_set1_ps(upper[2][i]);
                x_last = _mm256_load_ps(dstp + (i + 1) * dst_stride + j);
                a0 = _mm256_fmadd_ps(up, x_last, a0);
            }
            x = _mm256_sub_ps(x, a0);
            _mm256_store_ps(dstp + i * dst_stride + j, x);
        }
    }

    *current_height = height;
}


/*
 * General version of the vertical solver.
 */
static void process_plane_v_avx2(int height, int current_width, int *current_height, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
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
            a0 = _mm256_setzero_ps();
            start = VSMAX(0, i - c + 1);
            for (int k = start; k < i; k++) {
                lo = _mm256_set1_ps(lower[k - start][i]);
                x_last = _mm256_load_ps(dstp + k * dst_stride + j);
                a0 = _mm256_fmadd_ps(lo, x_last, a0);
            }
            x = _mm256_sub_ps(x, a0);
            di = _mm256_set1_ps(diagonal[i]);
            x = _mm256_mul_ps(x, di);
            _mm256_store_ps(dstp + i * dst_stride + j, x);            
        }
    }

    // Solve L' x = y
    for (int i = height - 2; i >= 0; i--) {
        for (int j = 0; j < current_width; j += 8) {

            x = _mm256_load_ps(dstp + i * dst_stride + j);
            a0 = _mm256_setzero_ps();
            start = VSMIN(height - 1, i + c - 1);
            for (int k = start; k > i; k--) {
                up = _mm256_set1_ps(upper[k - start + c - 2][i]);
                x_last = _mm256_load_ps(dstp + k * dst_stride + j);
                a0 = _mm256_fmadd_ps(up, x_last, a0);
            }
            x = _mm256_sub_ps(x, a0);
            _mm256_store_ps(dstp + i * dst_stride + j, x);
        }
    }

    *current_height = height;
}
#endif  // __AVX2__


static void process_plane_h_b3_c(int width, int current_height, int *current_width, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                 int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT lower, float * VS_RESTRICT upper, float * VS_RESTRICT diagonal,
                                 const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    for (int i = 0; i < current_height; i++) {

        for (int j = 0; j < width; j++) {

            // A' b
            float sum = 0.0f;
            for (int k = weights_left_idx[j]; k < weights_right_idx[j]; k++) {
                sum += weights[j * weights_columns + k - weights_left_idx[j]] * srcp[k];
            }

            // Solve LD y = A' b
            if (j != 0)
                sum -= lower[j] * dstp[j - 1];

            dstp[j] = sum * diagonal[j];
        }

        // Solve L' x = y
        for (int j = width - 2; j >= 0; j--) {
            dstp[j] -= upper[j] * dstp[j + 1];
        }

        srcp += src_stride;
        dstp += dst_stride;
    }

    *current_width = width;
}


static void process_plane_h_b7_c(int width, int current_height, int *current_width, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                 int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                                 float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    for (int i = 0; i < current_height; i++) {
        for (int j = 0; j < width; j++) {

            // A' b
            float sum = 0.0f;
            for (int k = weights_left_idx[j]; k < weights_right_idx[j]; k++)
                sum += weights[j * weights_columns + k - weights_left_idx[j]] * srcp[k];

            // Solve LD y = A' b
            if (j > 2) {
                sum -= lower[0][j] * dstp[j - 3];
                sum -= lower[1][j] * dstp[j - 2];
                sum -= lower[2][j] * dstp[j - 1];
            } else if (j > 1) {
                sum -= lower[0][j] * dstp[j - 2];
                sum -= lower[1][j] * dstp[j - 1];
            } else if (j > 0) {
                sum -= lower[0][j] * dstp[j - 1];
            }

            dstp[j] = sum * diagonal[j];
        }

        // Solve L' x = y
        for (int j = width - 2; j >= 0; j--) {
            float sum = 0.0f;
            if (j < width - 3) {
                sum += upper[0][j] * dstp[j + 1];
                sum += upper[1][j] * dstp[j + 2];
                sum += upper[2][j] * dstp[j + 3];
            }
            else if (j < width - 2) {
                sum += upper[1][j] * dstp[j + 1];
                sum += upper[2][j] * dstp[j + 2];}
            else if (j < width - 1) {
                sum += upper[2][j] * dstp[j + 1];
            }

            dstp[j] -= sum;
        }

        srcp += src_stride;
        dstp += dst_stride;
    }

    *current_width = width;
}


static void process_plane_h_c(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                              int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                              float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    int c = (bandwidth + 1) / 2;

    for (int i = 0; i < current_height; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            int start = VSMAX(0, j - c + 1);

            // A' b
            for (int k = weights_left_idx[j]; k < weights_right_idx[j]; k++)
                sum += weights[j * weights_columns + k - weights_left_idx[j]] * srcp[k];

            // Solve LD y = A' b
            for (int k = start; k < j; k++) {
                sum -= lower[k - start][j] * dstp[k];
            }

            dstp[j] = sum * diagonal[j];
        }

        // Solve L' x = y
        for (int j = width - 2; j >= 0; j--) {
            float sum = 0.0f;
            int start = VSMIN(width - 1, j + c - 1);

            for (int k = start; k > j; k--) {
                sum += upper[k - start + c - 2][j] * dstp[k];
            }

            dstp[j] -= sum;
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
    *current_width = width;
}


static void process_plane_v_b3_c(int height, int current_width, int *current_height, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                 int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT lower, float * VS_RESTRICT upper, float * VS_RESTRICT diagonal,
                                 const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < current_width; j++) {
            float sum = 0.0f;

            // A' b
            for (int k = weights_left_idx[i]; k < weights_right_idx[i]; k++) {
                sum += weights[i * weights_columns + k - weights_left_idx[i]] * srcp[k * src_stride + j];
            }

            // Solve LD y = A' b
            if (i != 0)
                sum -= lower[i] * dstp[(i - 1) * dst_stride + j];

            dstp[i * dst_stride + j] = sum * diagonal[i];
        }
    }

    // Solve L' x = y
    for (int i = height-2; i >= 0; i--) {
        for (int j = 0; j < current_width; j++) {
            dstp[i * dst_stride + j] -= upper[i] * dstp[(i + 1) * dst_stride + j];
        }
    }

    *current_height = height;
}


static void process_plane_v_b7_c(int height, int current_width, int *current_height, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                 int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                                 float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < current_width; j++) {

            // A' b
            float sum = 0.0f;
            for (int k = weights_left_idx[i]; k < weights_right_idx[i]; k++)
                sum += weights[i * weights_columns + k - weights_left_idx[i]] * srcp[k * src_stride + j];

            // Solve LD y = A' b
            if (i > 2) {
                sum -= lower[0][i] * dstp[(i - 3) * dst_stride + j];
                sum -= lower[1][i] * dstp[(i - 2) * dst_stride + j];
                sum -= lower[2][i] * dstp[(i - 1) * dst_stride + j];
            } else if (i > 1) {
                sum -= lower[0][i] * dstp[(i - 2) * dst_stride + j];
                sum -= lower[1][i] * dstp[(i - 1) * dst_stride + j];
            } else if (i > 0) {
                sum -= lower[0][i] * dstp[(i - 1) * dst_stride + j];
            }

            dstp[i * dst_stride +j] = sum * diagonal[i];
        }
    }

    // Solve L' x = y
    for (int i = height - 2; i >= 0; i--) {
        for (int j = current_width - 1; j >= 0; j--) {
            float sum = 0.0f;
            if (i < height - 3) {
                sum += upper[0][i] * dstp[(i + 1) * dst_stride + j];
                sum += upper[1][i] * dstp[(i + 2) * dst_stride + j];
                sum += upper[2][i] * dstp[(i + 3) * dst_stride + j];
            }
            else if (i < height - 2) {
                sum += upper[1][i] * dstp[(i + 1) * dst_stride + j];
                sum += upper[2][i] * dstp[(i + 2) * dst_stride + j];}
            else if (i < height - 1) {
                sum += upper[2][i] * dstp[(i + 1) * dst_stride + j];
            }

            dstp[i * dst_stride + j] -= sum;
        }
    }

    *current_height = height;
}


static void process_plane_v_c(int height, int current_width, int *current_height, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                              int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower, float * VS_RESTRICT * VS_RESTRICT upper,
                              float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    int c = (bandwidth + 1) / 2;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < current_width; i++) {
            float sum = 0.0f;
            int start = VSMAX(0, j - c + 1);

            // A' b
            for (int k = weights_left_idx[j]; k < weights_right_idx[j]; ++k)
                sum += weights[j * weights_columns + k - weights_left_idx[j]] * srcp[k * src_stride + i];

            // Solve LD y = A' b
            for (int k = start; k < j; k++) {
                sum -= lower[k - start][j] * dstp[k * dst_stride + i];
            }

            dstp[j * dst_stride + i] = sum * diagonal[j];
        }

    }

    // Solve L' x = y
    for (int j = height - 2; j >= 0; j--) {
        for (int i = 0; i < current_width; i++) {
            float sum = 0.0f;
            int start = VSMIN(height - 1, j + c - 1);

            for (int k = start; k > j; k--) {
                sum += upper[k - start + c - 2][j] * dstp[k * dst_stride + i];
            }

            dstp[j * dst_stride + i] -= sum;
        }
    }

    *current_height = height;
}


static const VSFrameRef *VS_CC descale_get_frame(int n, int activationReason, void **instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi)
{
    struct DescaleData *d = (struct DescaleData *)(*instance_data);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frame_ctx);

    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->node, frame_ctx);
        const VSFormat *fi = d->vi_src.format;

        VSFrameRef *intermediate = vsapi->newVideoFrame(fi, d->vi_dst.width, d->vi_src.height, NULL, core);
        VSFrameRef *dst = vsapi->newVideoFrame(fi, d->vi_dst.width, d->vi_dst.height, src, core);

        for (int plane = 0; plane < d->vi_src.format->numPlanes; ++plane) {
            int current_width = vsapi->getFrameWidth(src, plane);
            int current_height = vsapi->getFrameHeight(src, plane);

            const int src_stride = vsapi->getStride(src, plane) / sizeof (float);
            const float *srcp = (const float *)(vsapi->getReadPtr(src, plane));

            if (d->process_h && d->process_v) {
                const int intermediate_stride = vsapi->getStride(intermediate, plane) / sizeof (float);
                float * VS_RESTRICT intermediatep = (float *)vsapi->getWritePtr(intermediate, plane);

                if (d->bandwidth == 3) {
                    process_plane_h_b3(d->vi_dst.width, current_height, &current_width, d->weights_h_left_idx, d->weights_h_right_idx, d->weights_h_columns, d->weights_h,
                                       d->lower_h[0], d->upper_h[0], d->diagonal_h, src_stride, intermediate_stride, srcp, intermediatep);
                } else if (d->bandwidth == 7) {
                    process_plane_h_b7(d->vi_dst.width, current_height, &current_width, d->weights_h_left_idx, d->weights_h_right_idx, d->weights_h_columns, d->weights_h,
                                       d->lower_h, d->upper_h, d->diagonal_h, src_stride, intermediate_stride, srcp, intermediatep);
                } else {
                    process_plane_h(d->vi_dst.width, current_height, &current_width, d->bandwidth, d->weights_h_left_idx, d->weights_h_right_idx, d->weights_h_columns, d->weights_h,
                                    d->lower_h, d->upper_h, d->diagonal_h, src_stride, intermediate_stride, srcp, intermediatep);
                }


                const int dst_stride = vsapi->getStride(dst, plane) / sizeof (float);
                float * VS_RESTRICT dstp = (float *)vsapi->getWritePtr(dst, plane);

                if (d->bandwidth == 3) {
                    process_plane_v_b3(d->vi_dst.height, current_width, &current_height, d->weights_v_left_idx, d->weights_v_right_idx, d->weights_v_columns, d->weights_v,
                                       d->lower_v[0], d->upper_v[0], d->diagonal_v, intermediate_stride, dst_stride, intermediatep, dstp);
                } else if (d->bandwidth == 7) {
                    process_plane_v_b7(d->vi_dst.height, current_width, &current_height, d->weights_v_left_idx, d->weights_v_right_idx, d->weights_v_columns, d->weights_v,
                                       d->lower_v, d->upper_v, d->diagonal_v, intermediate_stride, dst_stride, intermediatep, dstp);
                } else {
                    process_plane_v(d->vi_dst.height, current_width, &current_height, d->bandwidth, d->weights_v_left_idx, d->weights_v_right_idx, d->weights_v_columns, d->weights_v,
                                    d->lower_v, d->upper_v, d->diagonal_v, intermediate_stride, dst_stride, intermediatep, dstp);
                }

            } else if (d->process_h) {
                const int dst_stride = vsapi->getStride(dst, plane) / sizeof (float);
                float * VS_RESTRICT dstp = (float *)(vsapi->getWritePtr(dst, plane));

                if (d->bandwidth == 3) {
                    process_plane_h_b3(d->vi_dst.width, current_height, &current_width, d->weights_h_left_idx, d->weights_h_right_idx, d->weights_h_columns, d->weights_h,
                                       d->lower_h[0], d->upper_h[0], d->diagonal_h, src_stride, dst_stride, srcp, dstp);
                } else if (d->bandwidth == 7) {
                    process_plane_h_b7(d->vi_dst.width, current_height, &current_width, d->weights_h_left_idx, d->weights_h_right_idx, d->weights_h_columns, d->weights_h,
                                       d->lower_h, d->upper_h, d->diagonal_h, src_stride, dst_stride, srcp, dstp);
                } else {
                    process_plane_h(d->vi_dst.width, current_height, &current_width, d->bandwidth, d->weights_h_left_idx, d->weights_h_right_idx, d->weights_h_columns, d->weights_h,
                                    d->lower_h, d->upper_h, d->diagonal_h, src_stride, dst_stride, srcp, dstp);
                }

            } else if (d->process_v) {
                const int dst_stride = vsapi->getStride(dst, plane) / sizeof (float);
                float * VS_RESTRICT dstp = (float *)vsapi->getWritePtr(dst, plane);

                if (d->bandwidth == 3) {
                    process_plane_v_b3(d->vi_dst.height, current_width, &current_height, d->weights_v_left_idx, d->weights_v_right_idx, d->weights_v_columns, d->weights_v,
                                       d->lower_v[0], d->upper_v[0], d->diagonal_v, src_stride, dst_stride, srcp, dstp);
                } else if (d->bandwidth == 7) {
                    process_plane_v_b7(d->vi_dst.height, current_width, &current_height, d->weights_v_left_idx, d->weights_v_right_idx, d->weights_v_columns, d->weights_v,
                                       d->lower_v, d->upper_v, d->diagonal_v, src_stride, dst_stride, srcp, dstp);
                } else {
                    process_plane_v(d->vi_dst.height, current_width, &current_height, d->bandwidth, d->weights_v_left_idx, d->weights_v_right_idx, d->weights_v_columns, d->weights_v,
                                    d->lower_v, d->upper_v, d->diagonal_v, src_stride, dst_stride, srcp, dstp);
                }
            }
        }

        vsapi->freeFrame(intermediate);

        if (d->process_h || d->process_v) {
            vsapi->freeFrame(src);

            return dst;

        } else {
            vsapi->freeFrame(dst);

            return src;
        }
    }

    return NULL;
}


static void VS_CC descale_init(VSMap *in, VSMap *out, void **instance_data, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
    struct DescaleData *d = (struct DescaleData *)(*instance_data);
    vsapi->setVideoInfo(&d->vi_dst, 1, node);
}


static void VS_CC descale_free(void *instance_data, VSCore *core, const VSAPI *vsapi)
{
    struct DescaleData *d = (struct DescaleData *)instance_data;

    vsapi->freeNode(d->node);
    if (d->process_h) {
        free(d->weights_h);
        free(d->weights_h_left_idx);
        free(d->weights_h_right_idx);
        free(d->diagonal_h);
        for (int i = 0; i < d->bandwidth / 2; i++) {
            free(d->lower_h[i]);
            free(d->upper_h[i]);
        }
        free(d->lower_h);
        free(d->upper_h);
    }
    if (d->process_v) {
        free(d->weights_v);
        free(d->weights_v_left_idx);
        free(d->weights_v_right_idx);
        free(d->diagonal_v);
        for (int i = 0; i < d->bandwidth / 2; i++) {
            free(d->lower_v[i]);
            free(d->upper_v[i]);
        }
        free(d->lower_v);
        free(d->upper_v);
    }

    free(d);
}


static void VS_CC descale_create(const VSMap *in, VSMap *out, void *user_data, VSCore *core, const VSAPI *vsapi)
{
    enum DescaleMode mode = (enum DescaleMode)user_data;

    struct DescaleData d = {0};

    d.node = vsapi->propGetNode(in, "src", 0, NULL);
    d.vi_src = *vsapi->getVideoInfo(d.node);
    d.vi_dst = *vsapi->getVideoInfo(d.node);
    int err;

    if (!isConstantFormat(&d.vi_src) || (d.vi_src.format->id != pfGrayS && d.vi_src.format->id != pfRGBS && d.vi_src.format->id != pfYUV444PS)) {
        vsapi->setError(out, "Descale: Constant format GrayS, RGBS, and YUV444PS are the only supported input formats.");
        vsapi->freeNode(d.node);
        return;
    }

    d.vi_dst.width = int64ToIntS(vsapi->propGetInt(in, "width", 0, NULL));

    d.vi_dst.height = int64ToIntS(vsapi->propGetInt(in, "height", 0, NULL));

    d.shift_h = vsapi->propGetFloat(in, "src_left", 0, &err);
    if (err)
        d.shift_h = 0;

    d.shift_v = vsapi->propGetFloat(in, "src_top", 0, &err);
    if (err)
        d.shift_v = 0;

    if (d.vi_dst.width < 1 || d.vi_dst.height < 1) {
        vsapi->setError(out, "Descale: width and height must be bigger than 0.");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.vi_dst.width > d.vi_src.width || d.vi_dst.height > d.vi_src.height) {
        vsapi->setError(out, "Descale: Output dimension has to be smaller than or equal to input dimension.");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.vi_dst.height < 8) {
        vsapi->setError(out, "Descale: Output height has to be greater or equal to 8");
        vsapi->freeNode(d.node);
        return;
    }

    d.process_h = (d.vi_dst.width == d.vi_src.width && d.shift_h == 0) ? false : true;
    d.process_v = (d.vi_dst.height == d.vi_src.height && d.shift_v == 0) ? false : true;

    int support;
    char *funcname;

    if (mode == bilinear) {
        support = 1;
        funcname = "Debilinear";
    
    } else if (mode == bicubic) {
        d.b = vsapi->propGetFloat(in, "b", 0, &err);
        if (err)
            d.b = 0.0;

        d.c = vsapi->propGetFloat(in, "c", 0, &err);
        if (err)
            d.c = 0.5;

        support = 2;
        funcname = "Debicubic";

        // If b != 0 Bicubic is not an interpolation filter, so force processing
        if (d.b != 0) {
            d.process_h = true;
            d.process_v = true;
        }

    } else if (mode == lanczos) {
        d.taps = int64ToIntS(vsapi->propGetInt(in, "taps", 0, &err));
        if (err)
            d.taps = 3;

        if (d.taps < 1) {
            vsapi->setError(out, "Descale: taps must be bigger than 0.");
            vsapi->freeNode(d.node);
            return;
        }

        support = d.taps;
        funcname = "Delanczos";

    } else if (mode == spline16) {
        support = 2;
        funcname = "Despline16";

    } else if (mode == spline36) {
        support = 3;
        funcname = "Despline36";

    } else if (mode == spline64) {
        support = 4;
        funcname = "Despline64";
    }

    d.bandwidth = support * 4 - 1;

    if (d.process_h) {
        double *weights;
        double *transposed_weights;
        double *multiplied_weights;
        double *lower;

        scaling_weights(mode, support, d.vi_dst.width, d.vi_src.width, d.b, d.c, d.shift_h, &weights);
        transpose_matrix(d.vi_src.width, d.vi_dst.width, weights, &transposed_weights);

        d.weights_h_left_idx = calloc(ceil_n(d.vi_dst.width, 8), sizeof (int));
        d.weights_h_right_idx = calloc(ceil_n(d.vi_dst.width, 8), sizeof (int));
        for (int i = 0; i < d.vi_dst.width; i++) {
            for (int j = 0; j < d.vi_src.width; j++) {
                if (transposed_weights[i * d.vi_src.width + j] != 0.0) {
                    d.weights_h_left_idx[i] = j;
                    break;
                }
            }
            for (int j = d.vi_src.width - 1; j >= 0; j--) {
                if (transposed_weights[i * d.vi_src.width + j] != 0.0) {
                    d.weights_h_right_idx[i] = j + 1;
                    break;
                }
            }
        }

        multiply_sparse_matrices(d.vi_dst.width, d.vi_src.width, d.weights_h_left_idx, d.weights_h_right_idx, transposed_weights, weights, &multiplied_weights);
        banded_ldlt_decomposition(d.vi_dst.width, d.bandwidth, multiplied_weights);
        transpose_matrix(d.vi_dst.width, d.vi_dst.width, multiplied_weights, &lower);
        multiply_banded_matrix_with_diagonal(d.vi_dst.width, d.bandwidth, lower);

        int max = 0;
        for (int i = 0; i < d.vi_dst.width; i++) {
            int diff = d.weights_h_right_idx[i] - d.weights_h_left_idx[i];
            if (diff > max)
                max = diff;
        }
        d.weights_h_columns = max;
        d.weights_h = calloc(ceil_n(d.vi_dst.width, 8) * max, sizeof (float));
        for (int i = 0; i < d.vi_dst.width; i++) {
            for (int j = 0; j < d.weights_h_right_idx[i] - d.weights_h_left_idx[i]; j++) {
                d.weights_h[i * max + j] = (float)transposed_weights[i * d.vi_src.width + d.weights_h_left_idx[i] + j];
            }
        }

        extract_compressed_lower_upper_diagonal(d.vi_dst.width, d.bandwidth, lower, multiplied_weights, &d.lower_h, &d.upper_h, &d.diagonal_h);

        free(weights);
        free(transposed_weights);
        free(multiplied_weights);
        free(lower);
    }

    if (d.process_v) {
        double *weights;
        double *transposed_weights;
        double *multiplied_weights;
        double *lower;

        scaling_weights(mode, support, d.vi_dst.height, d.vi_src.height, d.b, d.c, d.shift_v, &weights);
        transpose_matrix(d.vi_src.height, d.vi_dst.height, weights, &transposed_weights);

        d.weights_v_left_idx = calloc(ceil_n(d.vi_dst.height, 8), sizeof (int));
        d.weights_v_right_idx = calloc(ceil_n(d.vi_dst.height, 8), sizeof (int));
        for (int i = 0; i < d.vi_dst.height; i++) {
            for (int j = 0; j < d.vi_src.height; j++) {
                if (transposed_weights[i * d.vi_src.height + j] != 0.0) {
                    d.weights_v_left_idx[i] = j;
                    break;
                }
            }
            for (int j = d.vi_src.height - 1; j >= 0; j--) {
                if (transposed_weights[i * d.vi_src.height + j] != 0.0) {
                    d.weights_v_right_idx[i] = j + 1;
                    break;
                }
            }
        }

        multiply_sparse_matrices(d.vi_dst.height, d.vi_src.height, d.weights_v_left_idx, d.weights_v_right_idx, transposed_weights, weights, &multiplied_weights);
        banded_ldlt_decomposition(d.vi_dst.height, d.bandwidth, multiplied_weights);
        transpose_matrix(d.vi_dst.height, d.vi_dst.height, multiplied_weights, &lower);
        multiply_banded_matrix_with_diagonal(d.vi_dst.height, d.bandwidth, lower);

        int max = 0;
        for (int i = 0; i < d.vi_dst.height; i++) {
            int diff = d.weights_v_right_idx[i] - d.weights_v_left_idx[i];
            if (diff > max)
                max = diff;
        }
        d.weights_v_columns = max;
        d.weights_v = calloc(ceil_n(d.vi_dst.height, 8) * max, sizeof (float));
        for (int i = 0; i < d.vi_dst.height; i++) {
            for (int j = 0; j < d.weights_v_right_idx[i] - d.weights_v_left_idx[i]; j++) {
                d.weights_v[i * max + j] = (float)transposed_weights[i * d.vi_src.height + d.weights_v_left_idx[i] + j];
            }
        }

        extract_compressed_lower_upper_diagonal(d.vi_dst.height, d.bandwidth, lower, multiplied_weights, &d.lower_v, &d.upper_v, &d.diagonal_v);

        free(weights);
        free(transposed_weights);
        free(multiplied_weights);
        free(lower);
    }

    struct DescaleData *data = malloc(sizeof d);
    *data = d;
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
            "src_top:float:opt",
            descale_create, (void *)(bilinear), plugin);

    register_func("Debicubic",
            "src:clip;"
            "width:int;"
            "height:int;"
            "b:float:opt;"
            "c:float:opt;"
            "src_left:float:opt;"
            "src_top:float:opt",
            descale_create, (void *)(bicubic), plugin);

    register_func("Delanczos",
            "src:clip;"
            "width:int;"
            "height:int;"
            "taps:int:opt;"
            "src_left:float:opt;"
            "src_top:float:opt",
            descale_create, (void *)(lanczos), plugin);

    register_func("Despline16",
            "src:clip;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt",
            descale_create, (void *)(spline16), plugin);

    register_func("Despline36",
            "src:clip;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt",
            descale_create, (void *)(spline36), plugin);

    register_func("Despline64",
            "src:clip;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt",
            descale_create, (void *)(spline64), plugin);
}
