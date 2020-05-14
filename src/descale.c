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
#include <pthread.h>
#include <vapoursynth/VapourSynth.h>
#include <vapoursynth/VSHelper.h>
#include "common.h"

#ifdef DESCALE_X86
    #include "x86/cpuinfo_x86.h"
    #include "x86/descale_avx2.h"
#endif


enum DescaleMode
{
    bilinear = 0,
    bicubic  = 1,
    lanczos  = 2,
    spline16 = 3,
    spline36 = 4,
    spline64 = 5
};


struct DescaleData
{
    bool initialized;
    pthread_mutex_t lock;

    VSNodeRef *node;
    VSVideoInfo vi_src;
    VSVideoInfo vi_dst;

    enum DescaleMode mode;
    int support;
    double param1;
    double param2;
    int bandwidth;
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
    void (*process_plane_h)(int, int, int *, int, int * VS_RESTRICT, int * VS_RESTRICT, int, float * VS_RESTRICT,
                            float * VS_RESTRICT * VS_RESTRICT, float * VS_RESTRICT * VS_RESTRICT, float * VS_RESTRICT,
                            const int, const int, const float * VS_RESTRICT, float * VS_RESTRICT);
    void (*process_plane_v)(int, int, int *, int, int * VS_RESTRICT, int * VS_RESTRICT, int, float * VS_RESTRICT,
                            float * VS_RESTRICT * VS_RESTRICT, float * VS_RESTRICT * VS_RESTRICT, float * VS_RESTRICT,
                            const int, const int, const float * VS_RESTRICT, float * VS_RESTRICT);
};


static void multiply_banded_matrix_with_diagonal(int rows, int bandwidth, double *matrix)
{
    int c = bandwidth / 2;

    for (int i = 1; i < rows; i++) {
        int start = VSMAX(i - c, 0);
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
static void banded_ldlt_decomposition(int n, int bandwidth, double *matrix)
{
    int c = bandwidth / 2;
    // Division by 0 can happen if shift is used
    double eps = DBL_EPSILON;

    for (int i = 0; i < n; i++) {
        int end = VSMIN(c + 1, n - i);

        for (int j = 1; j < end; j++) {
            double d = matrix[i * n + i + j] / (matrix[i * n + i] + eps);

            for (int k = 0; k < end - j; k++) {
                matrix[(i + j) * n + i + j + k] -= d * matrix[i * n + i + j + k];
            }
        }

        double e = 1.0 / (matrix[i * n + i] + eps);
        for (int j = 1; j < end; j++) {
                matrix[i * n + i + j] *= e;
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


static void extract_compressed_lower_upper_diagonal(int n, int bandwidth, const double *lower, const double *upper, float ***compressed_lower, float ***compressed_upper, float **diagonal)
{
    int c = bandwidth / 2;
    // Division by 0 can happen if shift is used
    double eps = DBL_EPSILON;
    *compressed_lower = calloc(c, sizeof (float *));
    *compressed_upper = calloc(c, sizeof (float *));
    *diagonal = calloc(ceil_n(n, 8), sizeof (float));

    for (int i = 0; i < c; i++) {
        (*compressed_lower)[i] = calloc(ceil_n(n, 8), sizeof (float));
        (*compressed_upper)[i] = calloc(ceil_n(n, 8), sizeof (float));
    }

    for (int i = 0; i < n; i++) {
        int start = VSMAX(i - c, 0);
        for (int j = start; j < i; j++) {
            (*compressed_lower)[j - i + c][i] = (float)lower[i * n + j];
        }
    }

    for (int i = 0; i < n; i++) {
        int start = VSMIN(i + c, n - 1);
        for (int j = start; j > i; j--) {
            (*compressed_upper)[j - i - 1][i] = (float)upper[i * n + j];
        }
    }

    for (int i = 0; i < n; i++) {
        (*diagonal)[i] = (float)(1.0 / (lower[i * n + i] + eps));
    }

}


#define PI 3.14159265358979323846


static inline double sinc(double x)
{
    return x == 0.0 ? 1.0 : sin(x * PI) / (x * PI);
}


static inline double square(double x)
{
    return x * x;
}


static inline double cube(double x)
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

    return 0.0;
}


// Taken from zimg
// https://github.com/sekrit-twc/zimg/blob/09802b8751c18165519d32407c498f0e3024f1f1/src/zimg/resize/filter.cpp#L33
static double round_halfup(double x)
{
    /* When rounding on the pixel grid, the invariant
     *   round(x - 1) == round(x) - 1
     * must be preserved. This precludes the use of modes such as
     * half-to-even and half-away-from-zero.
     */
    return x < 0 ? floor(x + 0.5) : floor(x + 0.49999999999999994);
}


// Most of this is taken from zimg 
// https://github.com/sekrit-twc/zimg/blob/ce27c27f2147fbb28e417fbf19a95d3cf5d68f4f/src/zimg/resize/filter.cpp#L227
static void scaling_weights(enum DescaleMode mode, int support, int src_dim, int dst_dim, double param1, double param2, double shift, double **weights)
{
    *weights = calloc(src_dim * dst_dim, sizeof (double));
    double ratio = (double)dst_dim / src_dim;

    for (int i = 0; i < dst_dim; i++) {

        double total = 0.0;
        double pos = (i + 0.5) / ratio + shift;
        double begin_pos = round_halfup(pos - support) + 0.5;
        for (int j = 0; j < 2 * support; j++) {
            double xpos = begin_pos + j;
            total += calculate_weight(mode, support, xpos - pos, param1, param2);
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
            (*weights)[i * src_dim + idx] += calculate_weight(mode, support, xpos - pos, param1, param2) / total;
        }
    }
}


static void process_plane_h_b3_c(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                 int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower2, float * VS_RESTRICT * VS_RESTRICT upper2,
                                 float * VS_RESTRICT diagonal, const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    float * VS_RESTRICT lower = lower2[0];
    float * VS_RESTRICT upper = upper2[0];

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


static void process_plane_h_b7_c(int width, int current_height, int *current_width, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
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
                sum -= lower[1][j] * dstp[j - 2];
                sum -= lower[2][j] * dstp[j - 1];
            } else if (j > 0) {
                sum -= lower[2][j] * dstp[j - 1];
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
                sum += upper[0][j] * dstp[j + 1];
                sum += upper[1][j] * dstp[j + 2];}
            else if (j < width - 1) {
                sum += upper[0][j] * dstp[j + 1];
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
    int c = bandwidth / 2;

    for (int i = 0; i < current_height; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            int start = VSMAX(0, j - c);

            // A' b
            for (int k = weights_left_idx[j]; k < weights_right_idx[j]; k++)
                sum += weights[j * weights_columns + k - weights_left_idx[j]] * srcp[k];

            // Solve LD y = A' b
            for (int k = start; k < j; k++) {
                sum -= lower[k - j + c][j] * dstp[k];
            }

            dstp[j] = sum * diagonal[j];
        }

        // Solve L' x = y
        for (int j = width - 2; j >= 0; j--) {
            float sum = 0.0f;
            int start = VSMIN(width - 1, j + c);

            for (int k = start; k > j; k--) {
                sum += upper[k - j - 1][j] * dstp[k];
            }

            dstp[j] -= sum;
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
    *current_width = width;
}


static void process_plane_v_b3_c(int height, int current_width, int *current_height, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
                                 int weights_columns, float * VS_RESTRICT weights, float * VS_RESTRICT * VS_RESTRICT lower2, float * VS_RESTRICT * VS_RESTRICT upper2, float * VS_RESTRICT diagonal,
                                 const int src_stride, const int dst_stride, const float * VS_RESTRICT srcp, float * VS_RESTRICT dstp)
{
    float * VS_RESTRICT lower = lower2[0];
    float * VS_RESTRICT upper = upper2[0];

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


static void process_plane_v_b7_c(int height, int current_width, int *current_height, int bandwidth, int * VS_RESTRICT weights_left_idx, int * VS_RESTRICT weights_right_idx,
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
                sum -= lower[1][i] * dstp[(i - 2) * dst_stride + j];
                sum -= lower[2][i] * dstp[(i - 1) * dst_stride + j];
            } else if (i > 0) {
                sum -= lower[2][i] * dstp[(i - 1) * dst_stride + j];
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
            } else if (i < height - 2) {
                sum += upper[0][i] * dstp[(i + 1) * dst_stride + j];
                sum += upper[1][i] * dstp[(i + 2) * dst_stride + j];
            } else if (i < height - 1) {
                sum += upper[0][i] * dstp[(i + 1) * dst_stride + j];
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
    int c = bandwidth / 2;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < current_width; i++) {
            float sum = 0.0f;
            int start = VSMAX(0, j - c);

            // A' b
            for (int k = weights_left_idx[j]; k < weights_right_idx[j]; ++k)
                sum += weights[j * weights_columns + k - weights_left_idx[j]] * srcp[k * src_stride + i];

            // Solve LD y = A' b
            for (int k = start; k < j; k++) {
                sum -= lower[k - j + c][j] * dstp[k * dst_stride + i];
            }

            dstp[j * dst_stride + i] = sum * diagonal[j];
        }

    }

    // Solve L' x = y
    for (int j = height - 2; j >= 0; j--) {
        for (int i = 0; i < current_width; i++) {
            float sum = 0.0f;
            int start = VSMIN(height - 1, j + c);

            for (int k = start; k > j; k--) {
                sum += upper[k - j - 1][j] * dstp[k * dst_stride + i];
            }

            dstp[j * dst_stride + i] -= sum;
        }
    }

    *current_height = height;
}


static void initialize_descale_data(struct DescaleData *d)
{
    if (d->process_h) {
        double *weights;
        double *transposed_weights;
        double *multiplied_weights;
        double *lower;

        scaling_weights(d->mode, d->support, d->vi_dst.width, d->vi_src.width, d->param1, d->param2, d->shift_h, &weights);
        transpose_matrix(d->vi_src.width, d->vi_dst.width, weights, &transposed_weights);

        d->weights_h_left_idx = calloc(ceil_n(d->vi_dst.width, 8), sizeof (int));
        d->weights_h_right_idx = calloc(ceil_n(d->vi_dst.width, 8), sizeof (int));
        for (int i = 0; i < d->vi_dst.width; i++) {
            for (int j = 0; j < d->vi_src.width; j++) {
                if (transposed_weights[i * d->vi_src.width + j] != 0.0) {
                    d->weights_h_left_idx[i] = j;
                    break;
                }
            }
            for (int j = d->vi_src.width - 1; j >= 0; j--) {
                if (transposed_weights[i * d->vi_src.width + j] != 0.0) {
                    d->weights_h_right_idx[i] = j + 1;
                    break;
                }
            }
        }

        multiply_sparse_matrices(d->vi_dst.width, d->vi_src.width, d->weights_h_left_idx, d->weights_h_right_idx, transposed_weights, weights, &multiplied_weights);
        banded_ldlt_decomposition(d->vi_dst.width, d->bandwidth, multiplied_weights);
        transpose_matrix(d->vi_dst.width, d->vi_dst.width, multiplied_weights, &lower);
        multiply_banded_matrix_with_diagonal(d->vi_dst.width, d->bandwidth, lower);

        int max = 0;
        for (int i = 0; i < d->vi_dst.width; i++) {
            int diff = d->weights_h_right_idx[i] - d->weights_h_left_idx[i];
            if (diff > max)
                max = diff;
        }
        d->weights_h_columns = max;
        d->weights_h = calloc(ceil_n(d->vi_dst.width, 8) * max, sizeof (float));
        for (int i = 0; i < d->vi_dst.width; i++) {
            for (int j = 0; j < d->weights_h_right_idx[i] - d->weights_h_left_idx[i]; j++) {
                d->weights_h[i * max + j] = (float)transposed_weights[i * d->vi_src.width + d->weights_h_left_idx[i] + j];
            }
        }

        extract_compressed_lower_upper_diagonal(d->vi_dst.width, d->bandwidth, lower, multiplied_weights, &d->lower_h, &d->upper_h, &d->diagonal_h);

        free(weights);
        free(transposed_weights);
        free(multiplied_weights);
        free(lower);
    }

    if (d->process_v) {
        double *weights;
        double *transposed_weights;
        double *multiplied_weights;
        double *lower;

        scaling_weights(d->mode, d->support, d->vi_dst.height, d->vi_src.height, d->param1, d->param2, d->shift_v, &weights);
        transpose_matrix(d->vi_src.height, d->vi_dst.height, weights, &transposed_weights);

        d->weights_v_left_idx = calloc(ceil_n(d->vi_dst.height, 8), sizeof (int));
        d->weights_v_right_idx = calloc(ceil_n(d->vi_dst.height, 8), sizeof (int));
        for (int i = 0; i < d->vi_dst.height; i++) {
            for (int j = 0; j < d->vi_src.height; j++) {
                if (transposed_weights[i * d->vi_src.height + j] != 0.0) {
                    d->weights_v_left_idx[i] = j;
                    break;
                }
            }
            for (int j = d->vi_src.height - 1; j >= 0; j--) {
                if (transposed_weights[i * d->vi_src.height + j] != 0.0) {
                    d->weights_v_right_idx[i] = j + 1;
                    break;
                }
            }
        }

        multiply_sparse_matrices(d->vi_dst.height, d->vi_src.height, d->weights_v_left_idx, d->weights_v_right_idx, transposed_weights, weights, &multiplied_weights);
        banded_ldlt_decomposition(d->vi_dst.height, d->bandwidth, multiplied_weights);
        transpose_matrix(d->vi_dst.height, d->vi_dst.height, multiplied_weights, &lower);
        multiply_banded_matrix_with_diagonal(d->vi_dst.height, d->bandwidth, lower);

        int max = 0;
        for (int i = 0; i < d->vi_dst.height; i++) {
            int diff = d->weights_v_right_idx[i] - d->weights_v_left_idx[i];
            if (diff > max)
                max = diff;
        }
        d->weights_v_columns = max;
        d->weights_v = calloc(ceil_n(d->vi_dst.height, 8) * max, sizeof (float));
        for (int i = 0; i < d->vi_dst.height; i++) {
            for (int j = 0; j < d->weights_v_right_idx[i] - d->weights_v_left_idx[i]; j++) {
                d->weights_v[i * max + j] = (float)transposed_weights[i * d->vi_src.height + d->weights_v_left_idx[i] + j];
            }
        }

        extract_compressed_lower_upper_diagonal(d->vi_dst.height, d->bandwidth, lower, multiplied_weights, &d->lower_v, &d->upper_v, &d->diagonal_v);

        free(weights);
        free(transposed_weights);
        free(multiplied_weights);
        free(lower);
    }
}


static const VSFrameRef *VS_CC descale_get_frame(int n, int activationReason, void **instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi)
{
    struct DescaleData *d = (struct DescaleData *)(*instance_data);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frame_ctx);

    } else if (activationReason == arAllFramesReady) {
        if (!d->initialized) {
            pthread_mutex_lock(&d->lock);
            if (!d->initialized) {
                initialize_descale_data(d);
                d->initialized = true;
            }
            pthread_mutex_unlock(&d->lock);
        }

        const VSFrameRef *src = vsapi->getFrameFilter(n, d->node, frame_ctx);
        const VSFormat *fi = d->vi_src.format;

        VSFrameRef *intermediate = vsapi->newVideoFrame(fi, d->vi_dst.width, d->vi_src.height, NULL, core);
        VSFrameRef *dst = vsapi->newVideoFrame(fi, d->vi_dst.width, d->vi_dst.height, src, core);

        for (int plane = 0; plane < d->vi_src.format->numPlanes; plane++) {
            int current_width = vsapi->getFrameWidth(src, plane);
            int current_height = vsapi->getFrameHeight(src, plane);

            const int src_stride = vsapi->getStride(src, plane) / sizeof (float);
            const float *srcp = (const float *)(vsapi->getReadPtr(src, plane));

            if (d->process_h && d->process_v) {
                const int intermediate_stride = vsapi->getStride(intermediate, plane) / sizeof (float);
                float * VS_RESTRICT intermediatep = (float *)vsapi->getWritePtr(intermediate, plane);

                d->process_plane_h(d->vi_dst.width, current_height, &current_width, d->bandwidth, d->weights_h_left_idx, d->weights_h_right_idx, d->weights_h_columns, d->weights_h,
                                   d->lower_h, d->upper_h, d->diagonal_h, src_stride, intermediate_stride, srcp, intermediatep);


                const int dst_stride = vsapi->getStride(dst, plane) / sizeof (float);
                float * VS_RESTRICT dstp = (float *)vsapi->getWritePtr(dst, plane);

                d->process_plane_v(d->vi_dst.height, current_width, &current_height, d->bandwidth, d->weights_v_left_idx, d->weights_v_right_idx, d->weights_v_columns, d->weights_v,
                                   d->lower_v, d->upper_v, d->diagonal_v, intermediate_stride, dst_stride, intermediatep, dstp);

            } else if (d->process_h) {
                const int dst_stride = vsapi->getStride(dst, plane) / sizeof (float);
                float * VS_RESTRICT dstp = (float *)(vsapi->getWritePtr(dst, plane));

                d->process_plane_h(d->vi_dst.width, current_height, &current_width, d->bandwidth, d->weights_h_left_idx, d->weights_h_right_idx, d->weights_h_columns, d->weights_h,
                                   d->lower_h, d->upper_h, d->diagonal_h, src_stride, dst_stride, srcp, dstp);

            } else if (d->process_v) {
                const int dst_stride = vsapi->getStride(dst, plane) / sizeof (float);
                float * VS_RESTRICT dstp = (float *)vsapi->getWritePtr(dst, plane);

                d->process_plane_v(d->vi_dst.height, current_width, &current_height, d->bandwidth, d->weights_v_left_idx, d->weights_v_right_idx, d->weights_v_columns, d->weights_v,
                                   d->lower_v, d->upper_v, d->diagonal_v, src_stride, dst_stride, srcp, dstp);
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

    if (d->initialized) {
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
    }

    pthread_mutex_destroy(&d->lock);

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

    if (d.vi_dst.width < 1) {
        vsapi->setError(out, "Descale: width must be greater than 0.");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.vi_dst.height < 8) {
        vsapi->setError(out, "Descale: Output height must be greater than or equal to 8.");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.vi_dst.width > d.vi_src.width || d.vi_dst.height > d.vi_src.height) {
        vsapi->setError(out, "Descale: Output dimension must be less than or equal to input dimension.");
        vsapi->freeNode(d.node);
        return;
    }

    d.process_h = (d.vi_dst.width == d.vi_src.width && d.shift_h == 0) ? false : true;
    d.process_v = (d.vi_dst.height == d.vi_src.height && d.shift_v == 0) ? false : true;

    char *funcname;

    if (mode == bilinear) {
        d.support = 1;
        funcname = "Debilinear";
    
    } else if (mode == bicubic) {
        d.param1 = vsapi->propGetFloat(in, "b", 0, &err);
        if (err)
            d.param1 = 0.0;

        d.param2 = vsapi->propGetFloat(in, "c", 0, &err);
        if (err)
            d.param2 = 0.5;

        d.support = 2;
        funcname = "Debicubic";

        // If b != 0 Bicubic is not an interpolation filter, so force processing
        if (d.param1 != 0) {
            d.process_h = true;
            d.process_v = true;
        }

    } else if (mode == lanczos) {
        d.support = int64ToIntS(vsapi->propGetInt(in, "taps", 0, &err));
        if (err)
            d.support = 3;

        if (d.support < 1) {
            vsapi->setError(out, "Descale: taps must be greater than 0.");
            vsapi->freeNode(d.node);
            return;
        }

        funcname = "Delanczos";

    } else if (mode == spline16) {
        d.support = 2;
        funcname = "Despline16";

    } else if (mode == spline36) {
        d.support = 3;
        funcname = "Despline36";

    } else if (mode == spline64) {
        d.support = 4;
        funcname = "Despline64";
    } else {
        d.support = 0;
        funcname = "none";
    }

    d.mode = mode;
    d.bandwidth = d.support * 4 - 1;

#ifdef DESCALE_X86
    if (query_x86_capabilities().avx2) {
        if (d.bandwidth == 3) {
            d.process_plane_h = process_plane_h_b3_avx2;
            d.process_plane_v = process_plane_v_b3_avx2;
        } else if (d.bandwidth == 7) {
            d.process_plane_h = process_plane_h_b7_avx2;
            d.process_plane_v = process_plane_v_b7_avx2;
        } else {
            d.process_plane_h = process_plane_h_avx2;
            d.process_plane_v = process_plane_v_avx2;
        }
    } else {
#endif
        if (d.bandwidth == 3) {
            d.process_plane_h = process_plane_h_b3_c;
            d.process_plane_v = process_plane_v_b3_c;
        } else if (d.bandwidth == 7) {
            d.process_plane_h = process_plane_h_b7_c;
            d.process_plane_v = process_plane_v_b7_c;
        } else {
            d.process_plane_h = process_plane_h_c;
            d.process_plane_v = process_plane_v_c;
        }
#ifdef DESCALE_X86
    }
#endif

    d.initialized = false;
    pthread_mutex_init(&d.lock, NULL);

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
