// cudamatrix/bd-cu-kernels-ansi.h

#ifndef KALDI_BD_CUDAMATRIX_CU_KERNELS_ANSI_H_
#define KALDI_BD_CUDAMATRIX_CU_KERNELS_ANSI_H_

#include "cudamatrix/cu-matrixdim.h"

#if HAVE_CUDA == 1
extern "C" {
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

} // extern "C" 

#endif // HAVE_CUDA

#endif
