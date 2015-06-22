// cudamatrix/bd-cu-kernels.h

#ifndef KALDI_BD_CUDAMATRIX_CU_KERNELS_H_
#define KALDI_BD_CUDAMATRIX_CU_KERNELS_H_

#if HAVE_CUDA == 1

#include "base/kaldi-error.h"
#include "cudamatrix/bd-cu-kernels-ansi.h"

/*
 * In this file are C++ templated wrappers 
 * of the ANSI-C CUDA kernels
 */

namespace kaldi {

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

} // namespace kaldi

#endif // HAVE_CUDA

#endif
