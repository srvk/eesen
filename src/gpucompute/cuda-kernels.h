// gpucompute/cuda-kernels.h

// Copyright 2009-2012  Karel Vesely
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2015  Yajie Miao

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef EESEN_GPUCOMPUTE_CUDA_KERNELS_H_
#define EESEN_GPUCOMPUTE_CUDA_KERNELS_H_

#include "gpucompute/cuda-matrixdim.h"

#if HAVE_CUDA == 1
extern "C" {

/*********************************************************
 * int32 CUDA kernel calls (no template wrapper)
 */
void cudaI32_set_const(dim3 Gr, dim3 Bl, int32_cuda *mat, int32_cuda value, MatrixDim d);



/*********************************************************
 * float CUDA kernel calls
 */

/*
 * CuMatrix 
 */
void cudaF_apply_exp(dim3 Gr, dim3 Bl, float* mat, MatrixDim d);
void cudaD_apply_exp(dim3 Gr, dim3 Bl, double* mat, MatrixDim d);

void cudaF_apply_pow(dim3 Gr, dim3 Bl, float* mat, float power, MatrixDim d);
void cudaD_apply_pow(dim3 Gr, dim3 Bl, double* mat, double power, MatrixDim d);

void cudaF_apply_pow(dim3 Gr, dim3 Bl, float* mat, float epsilon, MatrixDim d);
void cudaD_apply_pow(dim3 Gr, dim3 Bl, double* mat, double epsilon, MatrixDim d);

void cudaF_sqrt_elements(dim3 Gr, dim3 Bl, float* mat, float epsilon, MatrixDim d);
void cudaD_sqrt_elements(dim3 Gr, dim3 Bl, double* mat, double epsilon, MatrixDim d);

void cudaF_invert_elements(dim3 Gr, dim3 Bl, float* mat, MatrixDim d);
void cudaD_invert_elements(dim3 Gr, dim3 Bl, double* mat, MatrixDim d);

void cudaF_apply_floor(dim3 Gr, dim3 Bl, float* mat, float floor_val, MatrixDim d);
void cudaD_apply_floor(dim3 Gr, dim3 Bl, double* mat, double floor_val, MatrixDim d);

void cudaF_apply_ceiling(dim3 Gr, dim3 Bl, float* mat, float ceiling_val, MatrixDim d);
void cudaD_apply_ceiling(dim3 Gr, dim3 Bl, double* mat, double ceiling_val, MatrixDim d);

void cudaF_apply_heaviside(dim3 Gr, dim3 Bl, float* mat, MatrixDim d);
void cudaD_apply_heaviside(dim3 Gr, dim3 Bl, double* mat, MatrixDim d);

void cudaF_set_const(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d);
void cudaD_set_const(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d);

void cudaF_add(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d);
void cudaD_add(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d);

void cudaF_scale(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d);
void cudaD_scale(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d);

void cudaF_apply_log(dim3 Gr, dim3 Bl, float *mat, MatrixDim d);
void cudaD_apply_log(dim3 Gr, dim3 Bl, double *mat, MatrixDim d);

void cudaF_mul_elements(dim3 Gr, dim3 Bl, float *mat, const float *A, MatrixDim dst_d, int src_stride);
void cudaD_mul_elements(dim3 Gr, dim3 Bl, double *mat, const double *A, MatrixDim dst_d, int src_stride);

void cudaF_mul_rows_vec(dim3 Gr, dim3 Bl, float *mat, const float *scale, MatrixDim d);
void cudaD_mul_rows_vec(dim3 Gr, dim3 Bl, double *mat, const double *scale, MatrixDim d);

void cudaF_add_mat(dim3 Gr, dim3 Bl, float alpha, const float *src, float *dst, MatrixDim d, int src_stride, int A_trans);
void cudaD_add_mat(dim3 Gr, dim3 Bl, double alpha, const double *src, double *dst, MatrixDim d, int src_stride, int A_trans);

void cudaF_add_vec_to_rows(dim3 Gr, dim3 Bl, float alpha, const float *row, float beta, float *dst, MatrixDim d);
void cudaD_add_vec_to_rows(dim3 Gr, dim3 Bl, double alpha, const double *row, double beta, double *dst, MatrixDim d); 

void cudaD_add_mat_mat_elements(dim3 Gr, dim3 Bl, double *data, const double *srcA_data, const double *srcB_data, MatrixDim dim, int srcA_stride, int srcB_stride, 
                                double alpha, double beta); 
void cudaF_add_mat_mat_elements(dim3 Gr, dim3 Bl, float *data, const float *srcA_data, const float *srcB_data, MatrixDim dim, int srcA_stride, int srcB_stride,
                                float alpha, float beta);

/*
 * CuVector
 */
void cudaF_copy_from_vec_df(int Gr, int Bl, double* v_out, const float* v_in, int dim);
void cudaD_copy_from_vec_df(int Gr, int Bl, double* v_out, const double* v_in, int dim);

void cudaF_copy_from_vec_fd(int Gr, int Bl, float* v_out, const float* v_in, int dim);
void cudaD_copy_from_vec_fd(int Gr, int Bl, float* v_out, const double* v_in, int dim);

void cudaF_vec_mul_elements(int Gr, int Bl, float* v, const float* a, int dim);
void cudaD_vec_mul_elements(int Gr, int Bl, double* v, const double* a, int dim);

void cudaF_vec_min(const float* v, float* value, int dim);
void cudaD_vec_min(const double* v, double* value, int dim);

void cudaF_vec_max(const float* v, float* value, int dim);
void cudaD_vec_max(const double* v, double* value, int dim);

void cudaF_add_diag_mat_mat(int Gr, int Bl, float alpha, float* v, int v_dim, const float* M, 
                            int M_cols, int M_row_stride, int M_col_stride, const float *N, int N_row_stride, 
                            int N_col_stride, int threads_per_element, float beta); 
void cudaD_add_diag_mat_mat(int Gr, int Bl, double alpha, double* v, int v_dim, const double* M,
                            int M_cols, int M_row_stride, int M_col_stride, const double *N, int N_row_stride,
                            int N_col_stride, int threads_per_element, double beta);

void cudaF_add_vec_vec(int Gr, int Bl, float alpha, float* v, const float* x, const float* y, float beta, int dim);
void cudaD_add_vec_vec(int Gr, int Bl, double alpha, double* v, const double* x, const double* y, double beta, int dim);

void cudaF_vec_sum(int Gr, int Bl, float* v, float* value, int dim, int inc);
void cudaD_vec_sum(int Gr, int Bl, double* v, double* value, int dim, int inc);

void cudaF_pvec_sum(int Gr, int Bl, float* vec, float* pvec_sum, int dim, int size);
void cudaD_pvec_sum(int Gr, int Bl, double* vec, double* pvec_sum, int dim, int size);

void cudaF_vec_apply_floor(int Gr, int Bl, float* v, float floor_val, float* num, int dim);
void cudaD_vec_apply_floor(int Gr, int Bl, double* v, double floor_val, float* num, int dim);

void cudaF_vec_apply_exp(int Gr, int Bl, float* v, int dim);
void cudaD_vec_apply_exp(int Gr, int Bl, double* v, int dim);

void cudaF_vec_apply_log(int Gr, int Bl, float* v, float* flag, int dim);
void cudaD_vec_apply_log(int Gr, int Bl, double* v, double* flag, int dim);

void cudaF_add_row_sum_mat(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d);
void cudaD_add_row_sum_mat(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d);

void cudaF_add_col_sum_mat(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d);
void cudaD_add_col_sum_mat(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d);

/*
 * cu::
 */
void cudaF_softmax_reduce(size_t Gr, size_t Bl, float *y, const float *x, MatrixDim d, int src_stride);
void cudaD_softmax_reduce(size_t Gr, size_t Bl, double *y, const double *x, MatrixDim d, int src_stride);

void cudaF_sigmoid(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d, int src_stride);
void cudaD_sigmoid(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d, int src_stride);

void cudaF_diff_sigmoid(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d, int e_stride, int y_stride);
void cudaD_diff_sigmoid(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d, int e_stride, int y_stride);

void cudaF_tanh(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d, int src_stride);
void cudaD_tanh(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d, int src_stride);

void cudaF_diff_tanh(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d, int e_stride, int y_stride);
void cudaD_diff_tanh(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d, int e_stride, int y_stride);

void cudaF_regularize_l1(dim3 Gr, dim3 Bl, float *wei, float *grad, float l1, float lr, MatrixDim d, int stride_grad);
void cudaD_regularize_l1(dim3 Gr, dim3 Bl, double *wei, double *grad, double l1, double lr, MatrixDim d, int stride_grad);

void cudaF_find_row_max_id(dim3 Gr, dim3 Bl, const float *mat, float *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d);
void cudaD_find_row_max_id(dim3 Gr, dim3 Bl, const double *mat, double *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d);

void cudaF_randomize(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in);
void cudaD_randomize(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in);

void cudaF_splice(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in);
void cudaD_splice(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in);

void cudaF_copy(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in);
void cudaD_copy(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in);


// some mostly mixed-type kernels.
void cuda_copy_from_mat_df(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_ff(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_fd(dim3 Gr, dim3 Bl, float *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_dd(dim3 Gr, dim3 Bl, double *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_df_trans(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_ff_trans(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_fd_trans(dim3 Gr, dim3 Bl, float *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in);
void cuda_copy_from_mat_dd_trans(dim3 Gr, dim3 Bl, double *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in);

/*
 * lstm::
 */
void cudaF_add_mat_diag_vec(dim3 Gr, dim3 Bl, float alpha, float *mat, MatrixDim mat_dim,
                            const float *mat2, int mat2_row_stride, int mat2_col_stride,
                            const float *vec, float beta);
void cudaD_add_mat_diag_vec(dim3 Gr, dim3 Bl, double alpha, double *mat, MatrixDim mat_dim,
                            const double *mat2, int mat2_row_stride, int mat2_col_stride,
                            const double *vec, double beta);

void cudaF_add_mat_dot_mat(dim3 Gr, dim3 Bl, float *data, const float *srcA_data, const float *srcB_data, int transA,
        int transB, MatrixDim dim, int srcA_stride, int srcB_stride, float alpha, float beta);

void cudaD_add_mat_dot_mat(dim3 Gr, dim3 Bl, double *data, const double *srcA_data, const double *srcB_data, int transA,
        int transB, MatrixDim dim, int srcA_stride, int srcB_stride, double alpha, double beta);

/*
 * ctc::
 */
void cudaF_compute_ctc_alpha(dim3 Gr, dim3 Bl, float *alpha, int row_idx, MatrixDim dim_alpha, const float *prob, MatrixDim dim_prob, const int *labels);
void cudaD_compute_ctc_alpha(dim3 Gr, dim3 Bl, double *alpha, int row_idx, MatrixDim dim_alpha, const double *prob, MatrixDim dim_prob, const int *labels);

void cudaF_compute_ctc_beta(dim3 Gr, dim3 Bl, float *beta, int row_idx, MatrixDim dim_beta, const float *prob, MatrixDim dim_prob, const int *labels);
void cudaD_compute_ctc_beta(dim3 Gr, dim3 Bl, double *beta, int row_idx, MatrixDim dim_beta, const double *prob, MatrixDim dim_prob, const int *labels);

void cudaF_compute_ctc_error(dim3 Gr, dim3 Bl, float *error, MatrixDim dim_error, const float *alpha, const float *beta, MatrixDim dim_alpha, const float *prob, const int *labels, float pzx);
void cudaD_compute_ctc_error(dim3 Gr, dim3 Bl, double *error, MatrixDim dim_error, const double *alpha, const double *beta, MatrixDim dim_alpha, const double *prob, const int *labels, double pzx);

void cudaF_compute_ctc_alpha_rescale(dim3 Gr, dim3 Bl, float *alpha, int row_idx, MatrixDim dim_alpha, const float *prob, MatrixDim dim_prob, const int *labels);
void cudaD_compute_ctc_alpha_rescale(dim3 Gr, dim3 Bl, double *alpha, int row_idx, MatrixDim dim_alpha, const double *prob, MatrixDim dim_prob, const int *labels);

void cudaF_compute_ctc_beta_rescale(dim3 Gr, dim3 Bl, float *beta, int row_idx, MatrixDim dim_beta, const float *prob, MatrixDim dim_prob, const int *labels);
void cudaD_compute_ctc_beta_rescale(dim3 Gr, dim3 Bl, double *beta, int row_idx, MatrixDim dim_beta, const double *prob, MatrixDim dim_prob, const int *labels);

void cudaF_compute_ctc_error_rescale(dim3 Gr, dim3 Bl, float *error, MatrixDim dim_error, const float *alpha, const float *beta, MatrixDim dim_alpha, const float *prob, const int *labels, const float *zt);
void cudaD_compute_ctc_error_rescale(dim3 Gr, dim3 Bl, double *error, MatrixDim dim_error, const double *alpha, const double *beta, MatrixDim dim_alpha, const double *prob, const int *labels, const double *zt);

void cudaF_distribute_prob_by_label(dim3 Gr, dim3 Bl, float *prob_dist, MatrixDim dim_prob_dist, const float *prob, MatrixDim dim_prob, const int *labels);
void cudaD_distribute_prob_by_label(dim3 Gr, dim3 Bl, double *prob_dist, MatrixDim dim_prob_dist, const double *prob, MatrixDim dim_prob, const int *labels);

void cudaF_compute_ctc_alpha_multiple_sequence(dim3 Gr, dim3 Bl, float *alpha, int seq_num, int row_idx, MatrixDim dim_alpha, const float *prob, MatrixDim dim_prob, const int *labels, int dim_label_stride, const int *seq_lengths);
void cudaD_compute_ctc_alpha_multiple_sequence(dim3 Gr, dim3 Bl, double *alpha, int seq_num, int row_idx, MatrixDim dim_alpha, const double *prob, MatrixDim dim_prob, const int *labels, int dim_label_stride, const int *seq_lengths);

void cudaF_compute_ctc_beta_multiple_sequence(dim3 Gr, dim3 Bl, float *beta, int seq_num, int row_idx, MatrixDim dim_beta, const float *prob, MatrixDim dim_prob, const int *labels, int dim_label_stride, const int *seq_lengths, const int *label_lengths);
void cudaD_compute_ctc_beta_multiple_sequence(dim3 Gr, dim3 Bl, double *beta, int seq_num, int row_idx, MatrixDim dim_beta, const double *prob, MatrixDim dim_prob, const int *labels, int dim_label_stride, const int *seq_lengths, const int *label_lengths);

void cudaF_compute_ctc_error_multiple_sequence(dim3 Gr, dim3 Bl, float *error, int seq_num, MatrixDim dim_error, const float *alpha, const float *beta, MatrixDim dim_alpha, const float *prob, const int *labels, int dim_label_stride, const int *seq_lengths, const float *pzx);
void cudaD_compute_ctc_error_multiple_sequence(dim3 Gr, dim3 Bl, double *error, int seq_num, MatrixDim dim_error, const double *alpha, const double *beta, MatrixDim dim_alpha, const double *prob, const int *labels, int dim_label_stride, const int *seq_lengths, const double *pzx);

} // extern "C" 

#endif // HAVE_CUDA


#endif
