// gpucompute/cuda-kernels-wrappers.h

// Copyright 2009-2012  Karel Vesely
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


#ifndef EESEN_GPUCOMPUTE_CUDA_KERNELS_WRAPPERS_H_
#define EESEN_GPUCOMPUTE_CUDA_KERNELS_WRAPPERS_H_

#if HAVE_CUDA == 1

//#include "base/kaldi-error.h"
#include "gpucompute/cuda-kernels.h"
namespace eesen {

/*
 * CUDA functions for Matrix 
 */
inline void cuda_copy_from_mat(dim3 Gr, dim3 Bl, float* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_fd(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_ff(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat(dim3 Gr, dim3 Bl, double* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_dd(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_df(Gr, Bl, mat_out, mat_in, d_out, d_in);
}

inline void cuda_copy_from_mat_trans(dim3 Gr, dim3 Bl, float* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_fd_trans(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat_trans(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_ff_trans(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat_trans(dim3 Gr, dim3 Bl, double* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_dd_trans(Gr, Bl, mat_out, mat_in, d_out, d_in);
}
inline void cuda_copy_from_mat_trans(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  cuda_copy_from_mat_df_trans(Gr, Bl, mat_out, mat_in, d_out, d_in);
}

inline void cuda_apply_exp(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) { cudaF_apply_exp(Gr,Bl,mat,d); }
inline void cuda_apply_exp(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) { cudaD_apply_exp(Gr,Bl,mat,d); }

inline void cuda_apply_pow(dim3 Gr, dim3 Bl, float* mat, float power, MatrixDim dim) { cudaF_apply_pow(Gr,Bl,mat,power,dim); }
inline void cuda_apply_pow(dim3 Gr, dim3 Bl, double* mat, double power, MatrixDim dim) { cudaD_apply_pow(Gr,Bl,mat,power,dim); }

inline void cuda_apply_floor(dim3 Gr, dim3 Bl, float* mat, float floor_val, MatrixDim dim) { cudaF_apply_floor(Gr,Bl,mat,floor_val,dim); }
inline void cuda_apply_floor(dim3 Gr, dim3 Bl, double* mat, double floor_val, MatrixDim dim) { cudaD_apply_floor(Gr,Bl,mat,floor_val,dim); }
inline void cuda_apply_ceiling(dim3 Gr, dim3 Bl, float* mat, float ceiling_val, MatrixDim dim) { cudaF_apply_ceiling(Gr,Bl,mat,ceiling_val,dim); }
inline void cuda_apply_ceiling(dim3 Gr, dim3 Bl, double* mat, double ceiling_val, MatrixDim dim) { cudaD_apply_ceiling(Gr,Bl,mat,ceiling_val,dim); }

inline void cuda_apply_heaviside(dim3 Gr, dim3 Bl, float* mat, MatrixDim dim) { cudaF_apply_heaviside(Gr,Bl,mat,dim); }
inline void cuda_apply_heaviside(dim3 Gr, dim3 Bl, double* mat, MatrixDim dim) { cudaD_apply_heaviside(Gr,Bl,mat,dim); }

inline void cuda_set_const(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d) { cudaF_set_const(Gr,Bl,mat,value,d); }
inline void cuda_set_const(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d) { cudaD_set_const(Gr,Bl,mat,value,d); }

inline void cuda_add(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d) { cudaF_add(Gr,Bl,mat,value,d); }
inline void cuda_add(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d) { cudaD_add(Gr,Bl,mat,value,d); }

inline void cuda_scale(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d) { cudaF_scale(Gr,Bl,mat,value,d); }
inline void cuda_scale(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d) { cudaD_scale(Gr,Bl,mat,value,d); }

inline void cuda_apply_log(dim3 Gr, dim3 Bl, float *mat, MatrixDim d) { cudaF_apply_log(Gr,Bl,mat,d); }
inline void cuda_apply_log(dim3 Gr, dim3 Bl, double *mat, MatrixDim d) { cudaD_apply_log(Gr,Bl,mat,d); }

inline void cuda_mul_elements(dim3 Gr, dim3 Bl, float *mat, const float *A, MatrixDim dst_d, int src_stride) {
  cudaF_mul_elements(Gr,Bl,mat,A,dst_d,src_stride);
}
inline void cuda_mul_elements(dim3 Gr, dim3 Bl, double *mat, const double *A, MatrixDim dst_d, int src_stride) {
  cudaD_mul_elements(Gr,Bl,mat,A,dst_d,src_stride);
}

inline void cuda_mul_rows_vec(dim3 Gr, dim3 Bl, float *mat, const float *scale, MatrixDim d) { cudaF_mul_rows_vec(Gr,Bl,mat,scale,d); }
inline void cuda_mul_rows_vec(dim3 Gr, dim3 Bl, double *mat, const double *scale, MatrixDim d) { cudaD_mul_rows_vec(Gr,Bl,mat,scale,d); }

inline void cuda_add_vec_to_rows(dim3 Gr, dim3 Bl, float alpha, const float *row, float beta, float *dst, MatrixDim d) { cudaF_add_vec_to_rows(Gr,Bl,alpha,row,beta,dst,d); }
inline void cuda_add_vec_to_rows(dim3 Gr, dim3 Bl, double alpha, const double *row, double beta, double *dst, MatrixDim d) { cudaD_add_vec_to_rows(Gr,Bl,alpha,row,beta,dst,d); }

inline void cuda_add_mat_mat_elements(dim3 Gr, dim3 Bl, float *data, const float *srcA_data,const float *srcB_data, MatrixDim dim,
                                int srcA_stride, int srcB_stride, float alpha, float beta)
                                {cudaF_add_mat_mat_elements(Gr,Bl,data,srcA_data,srcB_data,dim,srcA_stride,srcB_stride,alpha,beta);}

inline void cuda_add_mat_mat_elements(dim3 Gr, dim3 Bl, double *data, const double *srcA_data,const double *srcB_data, MatrixDim dim,
                                int srcA_stride, int srcB_stride, double alpha, double beta)
                                {cudaD_add_mat_mat_elements(Gr,Bl,data,srcA_data,srcB_data,dim,srcA_stride,srcB_stride,alpha,beta);}

/*
 * CuVector
 */
inline void cuda_copy_from_vec_df(int Gr, int Bl, double* v_out, const float* v_in, int dim) { cudaF_copy_from_vec_df(Gr,Bl,v_out,v_in,dim); }
inline void cuda_copy_from_vec_df(int Gr, int Bl, double* v_out, const double* v_in, int dim) { cudaD_copy_from_vec_df(Gr,Bl,v_out,v_in,dim); }
inline void cuda_copy_from_vec_fd(int Gr, int Bl, float* v_out, const float* v_in, int dim) { cudaF_copy_from_vec_fd(Gr,Bl,v_out,v_in,dim); }
inline void cuda_copy_from_vec_fd(int Gr, int Bl, float* v_out, const double* v_in, int dim) { cudaD_copy_from_vec_fd(Gr,Bl,v_out,v_in,dim); }

inline void cuda_vec_mul_elements(int Gr, int Bl, float* v, const float* a, int dim) { cudaF_vec_mul_elements(Gr,Bl,v,a,dim); }
inline void cuda_vec_mul_elements(int Gr, int Bl, double* v, const double* a, int dim) { cudaD_vec_mul_elements(Gr,Bl,v,a,dim); }

inline void cuda_vec_min(const float* v, float* value, int dim) { cudaF_vec_min(v,value,dim); }
inline void cuda_vec_min(const double* v, double* value, int dim) { cudaD_vec_min(v,value,dim); }

inline void cuda_vec_max(const float* v, float* value, int dim) { cudaF_vec_max(v,value,dim); }
inline void cuda_vec_max(const double* v, double* value, int dim) { cudaD_vec_max(v,value,dim); }

inline void cuda_add_diag_mat_mat(int Gr, int Bl, float alpha, float* v, int v_dim, const float* M, 
                                  int M_cols, int M_row_stride, int M_col_stride, const float *N, int N_row_stride, 
                                  int N_col_stride, int threads_per_element, float beta) {
    cudaF_add_diag_mat_mat(Gr, Bl, alpha, v, v_dim, M, M_cols, M_row_stride, M_col_stride, N, N_row_stride,
                             N_col_stride, threads_per_element, beta);
}

inline void cuda_add_diag_mat_mat(int Gr, int Bl, double alpha, double* v, int v_dim, const double* M,
                                  int M_cols, int M_row_stride, int M_col_stride, const double *N, int N_row_stride,
                                  int N_col_stride, int threads_per_element, double beta) {
    cudaD_add_diag_mat_mat(Gr, Bl, alpha, v, v_dim, M, M_cols, M_row_stride, M_col_stride, N, N_row_stride,
                             N_col_stride, threads_per_element, beta);
}

inline void cuda_add_vec_vec(int Gr, int Bl, float alpha, float* v, const float* x, const float* y, float beta, int dim) { cudaF_add_vec_vec(Gr,Bl,alpha,v,x,y,beta,dim); }
inline void cuda_add_vec_vec(int Gr, int Bl, double alpha, double* v, const double* x, const double* y, double beta, int dim) { cudaD_add_vec_vec(Gr,Bl,alpha,v,x,y,beta,dim); }

inline void cuda_vec_sum(int Gr, int Bl, float* v, float* value, int dim, int inc) { cudaF_vec_sum(Gr,Bl,v,value,dim,inc); }
inline void cuda_vec_sum(int Gr, int Bl, double* v, double* value, int dim, int inc) { cudaD_vec_sum(Gr,Bl,v,value,dim,inc); }

inline void cuda_pvec_sum(int Gr, int Bl, float* vec, float* pvec_sum, int dim, int size) { cudaF_pvec_sum(Gr, Bl, vec, pvec_sum, dim, size); }
inline void cuda_pvec_sum(int Gr, int Bl, double* vec, double* pvec_sum, int dim, int size) { cudaD_pvec_sum(Gr,Bl,vec,pvec_sum,dim,size); }

inline void cuda_vec_apply_floor(int Gr, int Bl, float* v, float floor_val, float* num, int dim) { cudaF_vec_apply_floor(Gr,Bl,v,floor_val,num,dim); }
inline void cuda_vec_apply_floor(int Gr, int Bl, double* v, double floor_val, float* num, int dim) { cudaD_vec_apply_floor(Gr,Bl,v,floor_val,num,dim); }

inline void cuda_vec_apply_exp(int Gr, int Bl, float* v, int dim) { cudaF_vec_apply_exp(Gr,Bl,v,dim); }
inline void cuda_vec_apply_exp(int Gr, int Bl, double* v, int dim) { cudaD_vec_apply_exp(Gr,Bl,v,dim); }

inline void cuda_vec_apply_log(int Gr, int Bl, float* v, float* flag, int dim) { cudaF_vec_apply_log(Gr,Bl,v,flag,dim); }
inline void cuda_vec_apply_log(int Gr, int Bl, double* v, double* flag, int dim) { cudaD_vec_apply_log(Gr,Bl,v,flag,dim); }

inline void cuda_sqrt_elements(dim3 Gr, dim3 Bl, float* data, float epsilon, MatrixDim d) { cudaF_sqrt_elements(Gr, Bl, data, epsilon, d); }
inline void cuda_sqrt_elements(dim3 Gr, dim3 Bl, double* data, double epsilon, MatrixDim d) { cudaD_sqrt_elements(Gr, Bl, data, epsilon, d); }

inline void cuda_invert_elements(dim3 Gr, dim3 Bl, float *data, MatrixDim d) { cudaF_invert_elements(Gr, Bl, data, d); }
inline void cuda_invert_elements(dim3 Gr, dim3 Bl, double *data, MatrixDim d) { cudaD_invert_elements(Gr, Bl, data, d); }

inline void cuda_add_row_sum_mat(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d) { cudaF_add_row_sum_mat(Gr,Bl,mat,vec_sum,d); }
inline void cuda_add_row_sum_mat(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d) { cudaD_add_row_sum_mat(Gr,Bl,mat,vec_sum,d); }

inline void cuda_add_col_sum_mat(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d) { cudaF_add_col_sum_mat(Gr,Bl,mat,vec_sum,d); }
inline void cuda_add_col_sum_mat(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d) { cudaD_add_col_sum_mat(Gr,Bl,mat,vec_sum,d); }

/*
 * cu::
 */
inline void cuda_sigmoid(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d, int src_stride) { cudaF_sigmoid(Gr,Bl,y,x,d,src_stride); }
inline void cuda_sigmoid(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d, int src_stride) { cudaD_sigmoid(Gr,Bl,y,x,d,src_stride); }
inline void cuda_diff_sigmoid(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d, int e_stride, int y_stride) { cudaF_diff_sigmoid(Gr,Bl,eout,e,y,d,e_stride,y_stride); }
inline void cuda_diff_sigmoid(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d, int e_stride, int y_stride) { cudaD_diff_sigmoid(Gr,Bl,eout,e,y,d,e_stride,y_stride); }

inline void cuda_tanh(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d, int src_stride) { cudaF_tanh(Gr,Bl,y,x,d,src_stride); }
inline void cuda_tanh(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d, int src_stride) { cudaD_tanh(Gr,Bl,y,x,d,src_stride); }
inline void cuda_diff_tanh(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d, int e_stride, int y_stride) { cudaF_diff_tanh(Gr,Bl,eout,e,y,d,e_stride,y_stride); }
inline void cuda_diff_tanh(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d, int e_stride, int y_stride) { cudaD_diff_tanh(Gr,Bl,eout,e,y,d,e_stride,y_stride); }

inline void cuda_softmax_reduce(size_t Gr, size_t Bl, float *y, const float *x, MatrixDim d, int src_stride) { cudaF_softmax_reduce(Gr,Bl,y,x,d,src_stride); }
inline void cuda_softmax_reduce(size_t Gr, size_t Bl, double *y, const double *x, MatrixDim d, int src_stride) { cudaD_softmax_reduce(Gr,Bl,y,x,d,src_stride); }

inline void cuda_regularize_l1(dim3 Gr, dim3 Bl, float *wei, float *grad, float l1, float lr, MatrixDim d, int stride_grad) { cudaF_regularize_l1(Gr,Bl,wei,grad,l1,lr,d,stride_grad); }
inline void cuda_regularize_l1(dim3 Gr, dim3 Bl, double *wei, double *grad, double l1, double lr, MatrixDim d, int stride_grad) { cudaD_regularize_l1(Gr,Bl,wei,grad,l1,lr,d,stride_grad); }

inline void cuda_find_row_max_id(dim3 Gr, dim3 Bl, const float *mat, float *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d) { cudaF_find_row_max_id(Gr,Bl,mat,vec_val,vec_id,voff,d); }
inline void cuda_find_row_max_id(dim3 Gr, dim3 Bl, const double *mat, double *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d) { cudaD_find_row_max_id(Gr,Bl,mat,vec_val,vec_id,voff,d); }

inline void cuda_randomize(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaF_randomize(Gr,Bl,y,x,copy_from,d_out,d_in); }
inline void cuda_randomize(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaD_randomize(Gr,Bl,y,x,copy_from,d_out,d_in); }

inline void cuda_splice(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in) { cudaF_splice(Gr,Bl,y,x,off,d_out,d_in); }
inline void cuda_splice(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in) { cudaD_splice(Gr,Bl,y,x,off,d_out,d_in); }

inline void cuda_copy(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaF_copy(Gr,Bl,y,x,copy_from,d_out,d_in); }
inline void cuda_copy(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaD_copy(Gr,Bl,y,x,copy_from,d_out,d_in); }

inline void cuda_add_mat(dim3 Gr, dim3 Bl, float alpha, const float *src, float *dst, MatrixDim d, int src_stride, int A_trans) { cudaF_add_mat(Gr,Bl,alpha,src,dst,d,src_stride, A_trans); }
inline void cuda_add_mat(dim3 Gr, dim3 Bl, double alpha, const double *src, double *dst, MatrixDim d, int src_stride, int A_trans) { cudaD_add_mat(Gr,Bl,alpha,src,dst,d,src_stride, A_trans); }

inline void cuda_add_mat_diag_vec(dim3 Gr, dim3 Bl, float alpha, float *mat, MatrixDim mat_dim,
                                  const float *mat2, int mat2_row_stride, int mat2_col_stride,
                                  const float *vec,  float beta) {
    cudaF_add_mat_diag_vec(Gr, Bl, alpha, mat, mat_dim, mat2,
                           mat2_row_stride, mat2_col_stride, vec, beta);
}
inline void cuda_add_mat_diag_vec(dim3 Gr, dim3 Bl, double alpha, double *mat, MatrixDim mat_dim,
                                  const double *mat2, int mat2_row_stride, int mat2_col_stride,
                                  const double *vec,  double beta) {
    cudaD_add_mat_diag_vec(Gr, Bl, alpha, mat, mat_dim, mat2,
                           mat2_row_stride, mat2_col_stride, vec, beta);
}

inline void cuda_add_mat_dot_mat(dim3 Gr, dim3 Bl, float *data, const float *srcA_data, const float *srcB_data, int transA,
        int transB, MatrixDim dim, int srcA_stride, int srcB_stride, float alpha, float beta) {
    cudaF_add_mat_dot_mat(Gr, Bl, data, srcA_data, srcB_data, transA, transB, dim, srcA_stride, srcB_stride, alpha, beta);
}
inline void cuda_add_mat_dot_mat(dim3 Gr, dim3 Bl, double *data, const double *srcA_data, const double *srcB_data, int transA,
        int transB, MatrixDim dim, int srcA_stride, int srcB_stride, double alpha, double beta) {
    cudaD_add_mat_dot_mat(Gr, Bl, data, srcA_data, srcB_data, transA, transB, dim, srcA_stride, srcB_stride, alpha, beta);
}

/// CTC Training
inline void cuda_compute_ctc_alpha(dim3 Gr, dim3 Bl, float *alpha, int row_idx, MatrixDim dim_alpha, const float *prob, MatrixDim dim_prob, const int *labels) {
  cudaF_compute_ctc_alpha(Gr, Bl, alpha, row_idx, dim_alpha, prob, dim_prob, labels);
}
inline void cuda_compute_ctc_alpha(dim3 Gr, dim3 Bl, double *alpha, int row_idx, MatrixDim dim_alpha, const double *prob, MatrixDim dim_prob, const int *labels) {
  cudaD_compute_ctc_alpha(Gr, Bl, alpha, row_idx, dim_alpha, prob, dim_prob, labels);
}

inline void cuda_compute_ctc_beta(dim3 Gr, dim3 Bl, float *beta, int row_idx, MatrixDim dim_beta, const float *prob, MatrixDim dim_prob, const int *labels) {
  cudaF_compute_ctc_beta(Gr, Bl, beta, row_idx, dim_beta, prob, dim_prob, labels);
}
inline void cuda_compute_ctc_beta(dim3 Gr, dim3 Bl, double *beta, int row_idx, MatrixDim dim_beta, const double *prob, MatrixDim dim_prob, const int *labels) {
  cudaD_compute_ctc_beta(Gr, Bl, beta, row_idx, dim_beta, prob, dim_prob, labels);
}

inline void cuda_compute_ctc_error(dim3 Gr, dim3 Bl, float *error, MatrixDim dim_error, const float *alpha, const float *beta, MatrixDim dim_alpha, const float *prob, const int *labels, float pzx) {
  cudaF_compute_ctc_error(Gr, Bl, error, dim_error, alpha, beta, dim_alpha, prob, labels, pzx);
}
inline void cuda_compute_ctc_error(dim3 Gr, dim3 Bl, double *error, MatrixDim dim_error, const double *alpha, const double *beta, MatrixDim dim_alpha, const double *prob, const int *labels, double pzx) {
  cudaD_compute_ctc_error(Gr, Bl, error, dim_error, alpha, beta, dim_alpha, prob, labels, pzx);
}

inline void cuda_compute_ctc_alpha_rescale(dim3 Gr, dim3 Bl, float *alpha, int row_idx, MatrixDim dim_alpha, const float *prob, MatrixDim dim_prob, const int *labels) {
  cudaF_compute_ctc_alpha_rescale(Gr, Bl, alpha, row_idx, dim_alpha, prob, dim_prob, labels);
}
inline void cuda_compute_ctc_alpha_rescale(dim3 Gr, dim3 Bl, double *alpha, int row_idx, MatrixDim dim_alpha, const double *prob, MatrixDim dim_prob, const int *labels) {
  cudaD_compute_ctc_alpha_rescale(Gr, Bl, alpha, row_idx, dim_alpha, prob, dim_prob, labels);
}

inline void cuda_compute_ctc_beta_rescale(dim3 Gr, dim3 Bl, float *beta, int row_idx, MatrixDim dim_beta, const float *prob, MatrixDim dim_prob, const int *labels) {
  cudaF_compute_ctc_beta_rescale(Gr, Bl, beta, row_idx, dim_beta, prob, dim_prob, labels);
}
inline void cuda_compute_ctc_beta_rescale(dim3 Gr, dim3 Bl, double *beta, int row_idx, MatrixDim dim_beta, const double *prob, MatrixDim dim_prob, const int *labels) {
  cudaD_compute_ctc_beta_rescale(Gr, Bl, beta, row_idx, dim_beta, prob, dim_prob, labels);
}

inline void cuda_compute_ctc_error_rescale(dim3 Gr, dim3 Bl, float *error, MatrixDim dim_error, const float *alpha, const float *beta, MatrixDim dim_alpha, const float *prob, const int *labels, float *zt) {
  cudaF_compute_ctc_error_rescale(Gr, Bl, error, dim_error, alpha, beta, dim_alpha, prob, labels, zt);
}
inline void cuda_compute_ctc_error_rescale(dim3 Gr, dim3 Bl, double *error, MatrixDim dim_error, const double *alpha, const double *beta, MatrixDim dim_alpha, const double *prob, const int *labels, double *zt) {
  cudaD_compute_ctc_error_rescale(Gr, Bl, error, dim_error, alpha, beta, dim_alpha, prob, labels, zt);
}

inline void cuda_distribute_prob_by_label(dim3 Gr, dim3 Bl, float *prob_dist, MatrixDim dim_prob_dist, const float *prob, MatrixDim dim_prob, const int *labels) {
  cudaF_distribute_prob_by_label(Gr, Bl, prob_dist, dim_prob_dist, prob, dim_prob, labels);
}
inline void cuda_distribute_prob_by_label(dim3 Gr, dim3 Bl, double *prob_dist, MatrixDim dim_prob_dist, const double *prob, MatrixDim dim_prob, const int *labels) {
  cudaD_distribute_prob_by_label(Gr, Bl, prob_dist, dim_prob_dist, prob, dim_prob, labels);
}

inline void cuda_compute_ctc_alpha_multiple_sequence(dim3 Gr, dim3 Bl, float *alpha, int seq_num, int row_idx, MatrixDim dim_alpha, const float *prob, MatrixDim dim_prob, const int *labels, int dim_label_stride, const int *seq_lengths) {
  cudaF_compute_ctc_alpha_multiple_sequence(Gr, Bl, alpha, seq_num, row_idx, dim_alpha, prob, dim_prob, labels, dim_label_stride, seq_lengths);
}
inline void cuda_compute_ctc_alpha_multiple_sequence(dim3 Gr, dim3 Bl, double *alpha, int seq_num, int row_idx, MatrixDim dim_alpha, const double *prob, MatrixDim dim_prob, const int *labels, int dim_label_stride, const int *seq_lengths) {
  cudaD_compute_ctc_alpha_multiple_sequence(Gr, Bl, alpha, seq_num, row_idx, dim_alpha, prob, dim_prob, labels, dim_label_stride, seq_lengths);
}

inline void cuda_compute_ctc_beta_multiple_sequence(dim3 Gr, dim3 Bl, float *beta, int seq_num, int row_idx, MatrixDim dim_beta, const float *prob, MatrixDim dim_prob, const int *labels, int dim_label_stride, const int *seq_lengths, const int *label_lengths) {
  cudaF_compute_ctc_beta_multiple_sequence(Gr, Bl, beta, seq_num, row_idx, dim_beta, prob, dim_prob, labels, dim_label_stride, seq_lengths, label_lengths);
}
inline void cuda_compute_ctc_beta_multiple_sequence(dim3 Gr, dim3 Bl, double *beta, int seq_num, int row_idx, MatrixDim dim_beta, const double *prob, MatrixDim dim_prob, const int *labels, int dim_label_stride, const int *seq_lengths, const int *label_lengths) {
  cudaD_compute_ctc_beta_multiple_sequence(Gr, Bl, beta, seq_num, row_idx, dim_beta, prob, dim_prob, labels, dim_label_stride, seq_lengths, label_lengths);
}

inline void cuda_compute_ctc_error_multiple_sequence(dim3 Gr, dim3 Bl, float *error, int seq_num, MatrixDim dim_error, const float *alpha, const float *beta, MatrixDim dim_alpha, const float *prob, const int *labels, int dim_label_stride, const int *seq_lengths, const float *pzx) {
  cudaF_compute_ctc_error_multiple_sequence(Gr, Bl, error, seq_num, dim_error, alpha, beta, dim_alpha, prob, labels, dim_label_stride, seq_lengths, pzx);
}
inline void cuda_compute_ctc_error_multiple_sequence(dim3 Gr, dim3 Bl, double *error, int seq_num, MatrixDim dim_error, const double *alpha, const double *beta, MatrixDim dim_alpha, const double *prob, const int *labels, int dim_label_stride, const int *seq_lengths, const double *pzx) {
  cudaD_compute_ctc_error_multiple_sequence(Gr, Bl, error, seq_num, dim_error, alpha, beta, dim_alpha, prob, labels, dim_label_stride, seq_lengths, pzx);
}

} // namespace eesen



#endif // HAVE_CUDA

#endif
