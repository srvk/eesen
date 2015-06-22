// cudamatrix/bd-cu-kernels.cu

// In this file is the CUDA code of the CUDA kernels, plus the ANSI-C wrappers

#include <cfloat>
#include "bd-cu-kernels-ansi.h"

/***********************************************************************
 * CUDA kernels
 * the functions are templated to have the float/double operations
 */

/*
 * CuMatrix
 */

template<typename Real>
__global__
static void _add_mat_diag_vec(Real alpha, Real *mat, MatrixDim mat_dim,
                              const Real *mat2, int mat2_row_stride, int mat2_col_stride, 
                              const Real *vec, Real beta) {
  // Note from Dan: in this kernel, we make the x dimension correspond to the
  // row index and y to the column index.  That was not always the case for
  // earlier kernels written by others.
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row index
  int j = blockIdx.y * blockDim.y + threadIdx.y; // column index
  
  int index = i * mat_dim.stride + j,
      index2 = i * mat2_row_stride + j * mat2_col_stride;
  
  if (i < mat_dim.rows && j < mat_dim.cols) {
    mat[index] = alpha * mat2[index2] * vec[j] + beta * mat[index];
  }
}

template<typename Real>
__global__
static void _add_mat_dot_mat(Real *data, const Real *srcA_data, const Real *srcB_data, int trasA, int transB, MatrixDim dim, int srcA_stride, int srcB_stride, Real alpha, Real beta) {
    // 1 represents kTrans, 0 represents kNoTrans
    // but for now, only kNoTrans is availiable
    int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
    int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
    int32_cuda tgt_index = i + j*dim.stride;
    int32_cuda srcA_index = i + j*srcA_stride;
    int32_cuda srcB_index = i + j*srcB_stride;
    if (i < dim.cols && j < dim.rows) {
        data[tgt_index] = alpha*srcA_data[srcA_index]*srcB_data[srcB_index] + beta * data[tgt_index] ;
    }
}

/***********************************************************************
 * ANSI-C wrappers of CUDA kernels
 */

void cudaF_add_mat_diag_vec(dim3 Gr, dim3 Bl, float alpha, float *mat, MatrixDim mat_dim,
                            const float *mat2, int mat2_row_stride, int mat2_col_stride, 
                            const float *vec,  float beta) {
  _add_mat_diag_vec<<<Gr,Bl>>>(alpha, mat, mat_dim, mat2, mat2_row_stride,
                               mat2_col_stride, vec, beta);
}
void cudaD_add_mat_diag_vec(dim3 Gr, dim3 Bl, double alpha, double *mat, MatrixDim mat_dim,
                            const double *mat2, int mat2_row_stride, int mat2_col_stride, 
                            const double *vec,  double beta) {
  _add_mat_diag_vec<<<Gr,Bl>>>(alpha, mat, mat_dim, mat2, mat2_row_stride,
                               mat2_col_stride, vec, beta);
}

void cudaF_add_mat_dot_mat(dim3 Gr, dim3 Bl, float *data, const float *srcA_data, const float *srcB_data, int transA, int transB, MatrixDim dim, int srcA_stride, int srcB_stride, float alpha, float beta) {
    _add_mat_dot_mat<<<Gr, Bl>>>(data, srcA_data, srcB_data, transA, transB, dim, srcA_stride, srcB_stride, alpha, beta);
}
void cudaD_add_mat_dot_mat(dim3 Gr, dim3 Bl, double *data, const double *srcA_data, const double *srcB_data, int transA, int transB, MatrixDim dim, int srcA_stride, int srcB_stride, double alpha, double beta) {
    _add_mat_dot_mat<<<Gr, Bl>>>(data, srcA_data, srcB_data, transA, transB, dim, srcA_stride, srcB_stride, alpha, beta);
}

