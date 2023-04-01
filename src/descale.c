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


#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include "common.h"
#include "descale.h"

#ifdef DESCALE_X86
    #include "x86/cpuinfo_x86.h"
    #include "x86/descale_avx2.h"
#endif


static void multiply_banded_matrix_with_diagonal(int rows, int bandwidth, double *matrix)
{
    int c = bandwidth / 2;

    for (int i = 1; i < rows; i++) {
        int start = DSMAX(i - c, 0);
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
        int end = DSMIN(c + 1, n - i);

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
        int start = DSMAX(i - c, 0);
        for (int j = start; j < i; j++) {
            (*compressed_lower)[j - i + c][i] = (float)lower[i * n + j];
        }
    }

    for (int i = 0; i < n; i++) {
        int start = DSMIN(i + c, n - 1);
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


static double calculate_weight(enum DescaleMode mode, int support, double distance, double b, double c, struct DescaleCustomKernel *ck)
{
    distance = fabs(distance);

    if (mode == DESCALE_MODE_BILINEAR) {
        return DSMAX(1.0 - distance, 0.0);

    } else if (mode == DESCALE_MODE_BICUBIC) {
        if (distance < 1)
            return ((12 - 9 * b - 6 * c) * cube(distance)
                        + (-18 + 12 * b + 6 * c) * square(distance) + (6 - 2 * b)) / 6.0;
        else if (distance < 2) 
            return ((-b - 6 * c) * cube(distance) + (6 * b + 30 * c) * square(distance)
                        + (-12 * b - 48 * c) * distance + (8 * b + 24 * c)) / 6.0;
        else
            return 0.0;

    } else if (mode == DESCALE_MODE_LANCZOS) {
        return distance < support ? sinc(distance) * sinc(distance / support) : 0.0;

    } else if (mode == DESCALE_MODE_SPLINE16) {
        if (distance < 1.0) {
            return 1.0 - (1.0 / 5.0 * distance) - (9.0 / 5.0 * square(distance)) + cube(distance);
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-7.0 / 15.0 * distance) + (4.0 / 5.0 * square(distance)) - (1.0 / 3.0 * cube(distance));
        } else {
            return 0.0;
        }

    } else if (mode == DESCALE_MODE_SPLINE36) {
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

    } else if (mode == DESCALE_MODE_SPLINE64) {
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
    } else if (mode == DESCALE_MODE_CUSTOM) {
        return ck->f(distance, ck->user_data);
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
static void scaling_weights(enum DescaleMode mode, int support, int src_dim, int dst_dim, double param1, double param2, double shift, double active_dim, enum DescaleBorder border_handling, struct DescaleCustomKernel *ck, double **weights)
{
    *weights = calloc(src_dim * dst_dim, sizeof (double));
    double ratio = (double)dst_dim / active_dim;

    for (int i = 0; i < dst_dim; i++) {

        double total = 0.0;
        double pos = (i + 0.5) / ratio + shift;
        double begin_pos = round_halfup(pos - support) + 0.5;
        for (int j = 0; j < 2 * support; j++) {
            double xpos = begin_pos + j;
            total += calculate_weight(mode, support, xpos - pos, param1, param2, ck);
        }
        for (int j = 0; j < 2 * support; j++) {
            double xpos = begin_pos + j;
            double real_pos = xpos;

            if (xpos < 0.0 || xpos > src_dim) {
                if (border_handling == DESCALE_BORDER_ZERO) {
                    continue;
                } else if (border_handling == DESCALE_BORDER_REPEAT) {
                    if (xpos < 0.0)
                        real_pos = 0.0;
                    else if (xpos >= src_dim)
                        real_pos = src_dim - 0.5;
                } else {    // Mirror
                    if (xpos < 0.0)
                        real_pos = -xpos;
                    else if (xpos >= src_dim)
                        real_pos = DSMIN(2.0 * src_dim - xpos, src_dim - 0.5);
                }
            }

            int idx = (int)floor(real_pos);
            (*weights)[i * src_dim + idx] += calculate_weight(mode, support, xpos - pos, param1, param2, ck) / total;
        }
    }
}


static void process_plane_h_b3_c(int width, int current_width, int current_height, int bandwidth, int * restrict weights_left_idx, int * restrict weights_right_idx,
                                 int weights_columns, float * restrict weights, float * restrict * restrict lower2, float * restrict * restrict upper2,
                                 float * restrict diagonal, int src_stride, int dst_stride, const float * restrict srcp, float * restrict dstp)
{
    float * restrict lower = lower2[0];
    float * restrict upper = upper2[0];

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
}


static void process_plane_h_b7_c(int width, int current_width, int current_height, int bandwidth, int * restrict weights_left_idx, int * restrict weights_right_idx,
                                 int weights_columns, float * restrict weights, float * restrict * restrict lower, float * restrict * restrict upper,
                                 float * restrict diagonal, int src_stride, int dst_stride, const float * restrict srcp, float * restrict dstp)
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
            } else if (j < width - 2) {
                sum += upper[0][j] * dstp[j + 1];
                sum += upper[1][j] * dstp[j + 2];
            } else if (j < width - 1) {
                sum += upper[0][j] * dstp[j + 1];
            }

            dstp[j] -= sum;
        }

        srcp += src_stride;
        dstp += dst_stride;
    }
}


static void process_plane_h_c(int width, int current_width, int current_height, int bandwidth, int * restrict weights_left_idx, int * restrict weights_right_idx,
                              int weights_columns, float * restrict weights, float * restrict * restrict lower, float * restrict * restrict upper,
                              float * restrict diagonal, int src_stride, int dst_stride, const float * restrict srcp, float * restrict dstp)
{
    int c = bandwidth / 2;

    for (int i = 0; i < current_height; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            int start = DSMAX(0, j - c);

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
            int start = DSMIN(width - 1, j + c);

            for (int k = start; k > j; k--) {
                sum += upper[k - j - 1][j] * dstp[k];
            }

            dstp[j] -= sum;
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
}


static void process_plane_v_b3_c(int height, int current_height, int current_width, int bandwidth, int * restrict weights_left_idx, int * restrict weights_right_idx,
                                 int weights_columns, float * restrict weights, float * restrict * restrict lower2, float * restrict * restrict upper2, float * restrict diagonal,
                                 int src_stride, int dst_stride, const float * restrict srcp, float * restrict dstp)
{
    float * restrict lower = lower2[0];
    float * restrict upper = upper2[0];

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
    for (int i = height - 2; i >= 0; i--) {
        for (int j = 0; j < current_width; j++) {
            dstp[i * dst_stride + j] -= upper[i] * dstp[(i + 1) * dst_stride + j];
        }
    }
}


static void process_plane_v_b7_c(int height, int current_height, int current_width, int bandwidth, int * restrict weights_left_idx, int * restrict weights_right_idx,
                                 int weights_columns, float * restrict weights, float * restrict * restrict lower, float * restrict * restrict upper,
                                 float * restrict diagonal, int src_stride, int dst_stride, const float * restrict srcp, float * restrict dstp)
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
}


static void process_plane_v_c(int height, int current_height, int current_width, int bandwidth, int * restrict weights_left_idx, int * restrict weights_right_idx,
                              int weights_columns, float * restrict weights, float * restrict * restrict lower, float * restrict * restrict upper,
                              float * restrict diagonal, int src_stride, int dst_stride, const float * restrict srcp, float * restrict dstp)
{
    int c = bandwidth / 2;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < current_width; i++) {
            float sum = 0.0f;
            int start = DSMAX(0, j - c);

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
            int start = DSMIN(height - 1, j + c);

            for (int k = start; k > j; k--) {
                sum += upper[k - j - 1][j] * dstp[k * dst_stride + i];
            }

            dstp[j * dst_stride + i] -= sum;
        }
    }
}


static void descale_process_vectors_c(struct DescaleCore *core, enum DescaleDir dir, int vector_count,
                                      int src_stride, int dst_stride, const float *srcp, float *dstp)
{
    if (dir == DESCALE_DIR_HORIZONTAL) {
        if (core->bandwidth == 3)
            process_plane_h_b3_c(core->dst_dim, core->src_dim, vector_count, core->bandwidth, core->weights_left_idx, core->weights_right_idx,
                                 core->weights_columns, core->weights, core->lower, core->upper, core->diagonal, src_stride, dst_stride, srcp, dstp);
        else if (core->bandwidth == 7)
            process_plane_h_b7_c(core->dst_dim, core->src_dim, vector_count, core->bandwidth, core->weights_left_idx, core->weights_right_idx,
                                 core->weights_columns, core->weights, core->lower, core->upper, core->diagonal, src_stride, dst_stride, srcp, dstp);
        else
            process_plane_h_c(core->dst_dim, core->src_dim, vector_count, core->bandwidth, core->weights_left_idx, core->weights_right_idx,
                              core->weights_columns, core->weights, core->lower, core->upper, core->diagonal, src_stride, dst_stride, srcp, dstp);
    } else {
        if (core->bandwidth == 3)
            process_plane_v_b3_c(core->dst_dim, core->src_dim, vector_count, core->bandwidth, core->weights_left_idx, core->weights_right_idx,
                                 core->weights_columns, core->weights, core->lower, core->upper, core->diagonal, src_stride, dst_stride, srcp, dstp);
        else if (core->bandwidth == 7)
            process_plane_v_b7_c(core->dst_dim, core->src_dim, vector_count, core->bandwidth, core->weights_left_idx, core->weights_right_idx,
                                 core->weights_columns, core->weights, core->lower, core->upper, core->diagonal, src_stride, dst_stride, srcp, dstp);
        else
            process_plane_v_c(core->dst_dim, core->src_dim, vector_count, core->bandwidth, core->weights_left_idx, core->weights_right_idx,
                              core->weights_columns, core->weights, core->lower, core->upper, core->diagonal, src_stride, dst_stride, srcp, dstp);
    }
}


static struct DescaleCore *create_core(int src_dim, int dst_dim, struct DescaleParams *params)
{
    int support;
    struct DescaleCore core = {0};

    if (params->mode == DESCALE_MODE_BILINEAR) {
        support = 1;
    } else if (params->mode == DESCALE_MODE_BICUBIC) {
        support = 2;
    } else if (params->mode == DESCALE_MODE_LANCZOS) {
        support = params->taps;
    } else if (params->mode == DESCALE_MODE_SPLINE16) {
        support = 2;
    } else if (params->mode == DESCALE_MODE_SPLINE36) {
        support = 3;
    } else if (params->mode == DESCALE_MODE_SPLINE64) {
        support = 4;
    } else if (params->mode == DESCALE_MODE_CUSTOM) {
        support = params->taps;
    } else {
        return NULL;
    }

    if (support == 0)
        return NULL;

    core.src_dim = src_dim;
    core.dst_dim = dst_dim;
    core.bandwidth = support * 4 - 1;

    double *weights;
    double *transposed_weights;
    double *multiplied_weights;
    double *lower;

    scaling_weights(params->mode, support, dst_dim, src_dim, params->param1, params->param2, params->shift, params->active_dim, params->border_handling, &params->custom_kernel, &weights);
    transpose_matrix(src_dim, dst_dim, weights, &transposed_weights);

    core.weights_left_idx = calloc(ceil_n(dst_dim, 8), sizeof (int));
    core.weights_right_idx = calloc(ceil_n(dst_dim, 8), sizeof (int));
    for (int i = 0; i < dst_dim; i++) {
        for (int j = 0; j < src_dim; j++) {
            if (transposed_weights[i * src_dim + j] != 0.0) {
                core.weights_left_idx[i] = j;
                break;
            }
        }
        for (int j = src_dim - 1; j >= 0; j--) {
            if (transposed_weights[i * src_dim + j] != 0.0) {
                core.weights_right_idx[i] = j + 1;
                break;
            }
        }
    }

    multiply_sparse_matrices(dst_dim, src_dim, core.weights_left_idx, core.weights_right_idx, transposed_weights, weights, &multiplied_weights);
    banded_ldlt_decomposition(dst_dim, core.bandwidth, multiplied_weights);
    transpose_matrix(dst_dim, dst_dim, multiplied_weights, &lower);
    multiply_banded_matrix_with_diagonal(dst_dim, core.bandwidth, lower);

    int max = 0;
    for (int i = 0; i < dst_dim; i++) {
        int diff = core.weights_right_idx[i] - core.weights_left_idx[i];
        if (diff > max)
            max = diff;
    }
    core.weights_columns = max;
    core.weights = calloc(ceil_n(dst_dim, 8) * max, sizeof (float));
    for (int i = 0; i < dst_dim; i++) {
        for (int j = 0; j < core.weights_right_idx[i] - core.weights_left_idx[i]; j++) {
            core.weights[i * max + j] = (float)transposed_weights[i * src_dim + core.weights_left_idx[i] + j];
        }
    }

    extract_compressed_lower_upper_diagonal(dst_dim, core.bandwidth, lower, multiplied_weights, &core.lower, &core.upper, &core.diagonal);

    free(weights);
    free(transposed_weights);
    free(multiplied_weights);
    free(lower);

    struct DescaleCore *corep = malloc(sizeof core);
    *corep = core;

    return corep;
}


static void free_core(struct DescaleCore *core)
{
    free(core->weights);
    free(core->weights_left_idx);
    free(core->weights_right_idx);
    free(core->diagonal);
    for (int i = 0; i < core->bandwidth / 2; i++) {
        free(core->lower[i]);
        free(core->upper[i]);
    }
    free(core->lower);
    free(core->upper);
    free(core);
}


struct DescaleAPI get_descale_api(enum DescaleOpt opt)
{
    struct DescaleAPI dsapi = {
        &create_core,
        &free_core,
        NULL
    };
#ifdef DESCALE_X86
    struct X86Capabilities caps = {0};
    if (opt == DESCALE_OPT_AUTO)
        caps = query_x86_capabilities();
    if ((opt == DESCALE_OPT_AUTO && caps.avx2 && caps.fma) || opt == DESCALE_OPT_AVX2) {
        dsapi.process_vectors = &descale_process_vectors_avx2;
    } else {
#endif
        dsapi.process_vectors = &descale_process_vectors_c;
#ifdef DESCALE_X86
    }
#endif
    return dsapi;
}
