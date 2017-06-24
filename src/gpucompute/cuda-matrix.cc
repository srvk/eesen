// gpucompute/cuda-matrix.cc

// Copyright 2009-2012  Karel Vesely, Lucas Ondel
//                2013  Ehsan Variani
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2013  Hainan Xu
//                2013  Xiaohui Zhang
//                2013  Johns Hopkins University (author: Guoguo Chen)
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


#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include "base/timer.h"
#include "gpucompute/cuda-common.h"
#include "gpucompute/cuda-vector.h"
#include "gpucompute/cuda-device.h"
#include "gpucompute/cuda-kernels-wrappers.h"
#include "gpucompute/cuda-randkernels-wrappers.h"
#include "gpucompute/cuda-array.h"
#include "gpucompute/cuda-math.h"
#include "gpucompute/cuda-matrix.h"
#include "gpucompute/cublas-wrappers.h"

namespace eesen {

template<typename Real>
void CuMatrix<Real>::Resize(MatrixIndexT rows, MatrixIndexT cols,
                            MatrixResizeType resize_type) {
  // This code does not currently support the other resize_type options.
  KALDI_ASSERT(resize_type == kSetZero || resize_type == kUndefined);
  if (rows * cols == 0) KALDI_ASSERT(rows == 0 && cols == 0);
  if (this->num_rows_ == rows && this->num_cols_ == cols) {
    if (resize_type == kSetZero) this->SetZero();
    return;
  }

  if (this->num_rows_ != 0)
    this->Destroy();
  if (rows == 0) return;  
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    MatrixIndexT row_bytes = cols * sizeof(Real);
    size_t pitch;
    this->data_ = static_cast<Real*>(CuDevice::Instantiate().MallocPitch(
        row_bytes, rows, &pitch));
    this->num_rows_ = rows;
    this->num_cols_ = cols; 
    this->stride_ = pitch / sizeof(Real);
    if (resize_type == kSetZero) this->SetZero();
    CuDevice::Instantiate().AccuProfile("CuMatrix::Resize", tim.Elapsed());    
  } else
#endif
  { // Let the initializer of Matrix<Real> handle the allocation,
    // and then just do Swap which will switch the pointers.
    // This wastes a few instructions but is simple to code.
    Matrix<Real> mat(rows, cols, resize_type);
    this->Swap(&mat);
  }
}

template<typename Real>
void CuMatrix<Real>::Destroy() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (this->data_ != NULL) {
      Timer tim;
      CuDevice::Instantiate().Free(this->data_);
      CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());    
    }
  } else
#endif
  {
    if (this->data_ != NULL) KALDI_MEMALIGN_FREE(this->data_);
  }
  this->data_ = NULL;
  this->num_rows_ = 0;
  this->num_cols_ = 0;
  this->stride_ = 0;
}

template<typename Real>
void CuMatrix<Real>::Swap(CuMatrix<Real> *mat) {
  std::swap(mat->data_, this->data_);
  std::swap(mat->num_cols_, this->num_cols_);
  std::swap(mat->num_rows_, this->num_rows_);
  std::swap(mat->stride_, this->stride_);
}


template<typename Real>
void CuMatrix<Real>::Swap(Matrix<Real> *mat) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (this->num_rows_ == 0) {
      if (mat->num_rows_ != 0) {
        // *this is empty, but mat is nonempty.
        this->Resize(mat->num_rows_, mat->num_cols_, kUndefined);
        this->CopyFromMat(*mat);
        mat->Resize(0, 0);
      }
      // else both are empty.
    } else { // *this is nonempty.
      if (mat->num_rows_ != 0) {
        // Both *this and *mat are nonempty.  Recurse to simpler cases.
        // this could be done more efficiently in the case where
        // the size does not change.
        Matrix<Real> temp;
        this->Swap(&temp); // now temp is full, *this is empty.
        mat->Swap(&temp); // now mat has data from *this, temp has
        // data from mat.
        this->Swap(&temp); // copy data in mat to *this, which is now empty.
      } else { // *this is full but *mat is empty.
        mat->Resize(this->num_rows_, this->num_cols_, kUndefined);
        this->CopyToMat(mat);
        this->Destroy();
      }
    }
  } else
#endif
  {
    std::swap(mat->data_, this->data_);
    std::swap(mat->num_cols_, this->num_cols_);
    std::swap(mat->num_rows_, this->num_rows_);
    std::swap(mat->stride_, this->stride_);
  }
}


template<class Real>
template<class OtherReal>
void CuMatrixBase<Real>::CopyFromMat(const CuMatrixBase<OtherReal> &M,
                                     MatrixTransposeType Trans) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (Trans == kNoTrans) {
      KALDI_ASSERT(M.NumRows() == num_rows_ && M.NumCols() == num_cols_);
    } else {
      KALDI_ASSERT(M.NumCols() == num_rows_ && M.NumRows() == num_cols_);
    }    
    if (M.num_rows_ == 0) return; // Nothing to do.
    Timer tim;
    if (sizeof(Real) == sizeof(OtherReal) && Trans == kNoTrans ) {
      MatrixIndexT dst_pitch = stride_ * sizeof(Real);
      MatrixIndexT src_pitch = M.Stride() * sizeof(Real);
      MatrixIndexT width = M.NumCols() * sizeof(Real);
      CU_SAFE_CALL(cudaMemcpy2D(data_, dst_pitch, M.data_, src_pitch,
                                width, M.num_rows_, cudaMemcpyDeviceToDevice));
    } else {
      dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
      dim3 dimGrid(n_blocks(num_rows_, CU2DBLOCK), n_blocks(num_cols_, CU2DBLOCK));
      if (Trans == kNoTrans) {
        cuda_copy_from_mat(dimGrid, dimBlock, data_, M.data_, Dim(), M.Dim());
      } else {
        cuda_copy_from_mat_trans(dimGrid, dimBlock, data_, M.data_, Dim(), M.Dim());
      }
    }
    CuDevice::Instantiate().AccuProfile("CuMatrixBase::CopyFromMat(from other CuMatrixBase)", tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), Trans);
  }
}

template
void CuMatrixBase<float>::CopyFromMat<float>(const CuMatrixBase<float> &M,
                                             MatrixTransposeType Trans);
template
void CuMatrixBase<float>::CopyFromMat<double>(const CuMatrixBase<double> &M,
                                              MatrixTransposeType Trans);
template
void CuMatrixBase<double>::CopyFromMat<float>(const CuMatrixBase<float> &M,
                                              MatrixTransposeType Trans);
template
void CuMatrixBase<double>::CopyFromMat<double>(const CuMatrixBase<double> &M,
                                               MatrixTransposeType Trans);

template<typename Real>
void CuMatrixBase<Real>::CopyFromMat(const MatrixBase<Real> &src,
                                     MatrixTransposeType trans) {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    if (trans == kNoTrans) {
      KALDI_ASSERT(src.NumRows() == num_rows_ && src.NumCols() == num_cols_);      
      Timer tim;

      MatrixIndexT dst_pitch = stride_*sizeof(Real);
      MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
      MatrixIndexT width = src.NumCols()*sizeof(Real);
      CU_SAFE_CALL(cudaMemcpy2D(data_, dst_pitch, src.Data(), src_pitch,
                                width, src.NumRows(), cudaMemcpyHostToDevice));
      
      CuDevice::Instantiate().AccuProfile("CuMatrixBase::CopyFromMat(from CPU)",tim.Elapsed());
    } else {
      CuMatrix<Real> trans_mat(src); // Do the transpose on the GPU board.
      this->CopyFromMat(trans_mat, kTrans);
    }
  } else
#endif
  {
    Mat().CopyFromMat(src, trans);
  }
}


template<typename Real>
template<typename OtherReal>
void CuMatrixBase<Real>::CopyFromMat(const MatrixBase<OtherReal> &src,
                                     MatrixTransposeType trans) {
  CuMatrix<OtherReal> temp(src);
  this->CopyFromMat(temp, trans);
}


// instantiate the template above.
template
void CuMatrixBase<float>::CopyFromMat(const MatrixBase<double> &src,
                                      MatrixTransposeType trans);
template
void CuMatrixBase<double>::CopyFromMat(const MatrixBase<float> &src,
                                      MatrixTransposeType trans);

template<typename Real>
CuMatrix<Real>::CuMatrix(const CuMatrix<Real> &other, MatrixTransposeType trans) {
  if (trans == kNoTrans)
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
  else
    this->Resize(other.NumCols(), other.NumRows(), kUndefined);
  this->CopyFromMat(other, trans);
}

template<typename Real>
CuMatrix<Real>::CuMatrix(const CuMatrixBase<Real> &other, MatrixTransposeType trans) {
  if (trans == kNoTrans)
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
  else
    this->Resize(other.NumCols(), other.NumRows(), kUndefined);
  this->CopyFromMat(other, trans);
}


template<typename Real>
template<typename OtherReal>
CuMatrix<Real>::CuMatrix(const MatrixBase<OtherReal> &other, MatrixTransposeType trans) {
  if (trans == kNoTrans)
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
  else
    this->Resize(other.NumCols(), other.NumRows(), kUndefined);
  this->CopyFromMat(other, trans);
}

template
CuMatrix<float>::CuMatrix(const MatrixBase<float> &other, MatrixTransposeType trans);
template
CuMatrix<double>::CuMatrix(const MatrixBase<float> &other, MatrixTransposeType trans);
template
CuMatrix<float>::CuMatrix(const MatrixBase<double> &other, MatrixTransposeType trans);
template
CuMatrix<double>::CuMatrix(const MatrixBase<double> &other, MatrixTransposeType trans);


template<typename Real>
template<typename OtherReal>
void CuMatrixBase<Real>::CopyToMat(MatrixBase<OtherReal> *dst,
                                   MatrixTransposeType trans) const {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {    
    if (trans == kTrans || sizeof(OtherReal) != sizeof(Real)) {
      CuMatrix<OtherReal> this_trans(*this, trans);
      this_trans.CopyToMat(dst, kNoTrans);
    } else {
      KALDI_ASSERT(dst->NumRows() == NumRows() && dst->NumCols() == NumCols());
      if (num_rows_ == 0) return;
      Timer tim;
   
      MatrixIndexT src_pitch = stride_*sizeof(Real);
      MatrixIndexT dst_pitch = dst->Stride()*sizeof(Real);
      MatrixIndexT width = NumCols()*sizeof(Real);
      CU_SAFE_CALL(cudaMemcpy2D(dst->Data(), dst_pitch, this->data_, src_pitch,
                                width, this->num_rows_, cudaMemcpyDeviceToHost));

      CuDevice::Instantiate().AccuProfile("CuMatrix::CopyToMatD2H",tim.Elapsed());
    }
  } else
  #endif
  {
    dst->CopyFromMat(Mat(), trans);
  }
}

template
void CuMatrixBase<float>::CopyToMat(MatrixBase<float> *dst,
                                    MatrixTransposeType trans) const;
template
void CuMatrixBase<double>::CopyToMat(MatrixBase<float> *dst,
                                     MatrixTransposeType trans) const;
template
void CuMatrixBase<float>::CopyToMat(MatrixBase<double> *dst,
                                    MatrixTransposeType trans) const;
template
void CuMatrixBase<double>::CopyToMat(MatrixBase<double> *dst,
                                     MatrixTransposeType trans) const;


template<typename Real>
void CuMatrix<Real>::Read(std::istream &is, bool binary) {
  Matrix<Real> temp;
  temp.Read(is, binary);
  Destroy();
  Swap(&temp);
}

template<typename Real>
void CuMatrixBase<Real>::Write(std::ostream &os, bool binary) const {
  Matrix<Real> temp(this->num_rows_, this->num_cols_, kUndefined);
  this->CopyToMat(&temp);
  temp.Write(os, binary);
}

template<typename Real>
void CuMatrixBase<Real>::SetZero() {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    CU_SAFE_CALL(cudaMemset2D(data_, stride_ * sizeof(Real), 0, 
                              num_cols_ * sizeof(Real), num_rows_ ));
    CuDevice::Instantiate().AccuProfile("CuMatrix::SetZero", tim.Elapsed());
  } else
#endif
  {
    Mat().SetZero();
  }
}


template<typename Real> 
void CuMatrixBase<Real>::Add(Real value) { 
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    if (num_rows_ == 0) return;
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_add(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Add(value);
  }
}

/*
 * Methods wrapping the ANSI-C CUDA kernels
 */
template<typename Real> 
void CuMatrixBase<Real>::Set(Real value) {
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    if (num_rows_ == 0) return;
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_set_const(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Set(value);
  }
}


template<typename Real> 
void CuMatrixBase<Real>::Scale(Real value) { 
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    if (num_rows_ == 0) return;
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_scale(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().Scale(value);
  }
}

template<typename Real> 
void CuMatrixBase<Real>::ApplyLog() { 
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    if (num_rows_ == 0) return;
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_apply_log(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().ApplyLog();
  }
}

template<typename Real>
Real CuMatrixBase<Real>::Sum() const {
  CuVector<Real> row_sum(NumCols());
  row_sum.AddRowSumMat(1.0, *this, 0.0);
  return row_sum.Sum();
}

template<typename Real>
void CuMatrixBase<Real>::MulElements(const CuMatrixBase<Real>& A) {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(num_cols_ == A.NumCols());
    KALDI_ASSERT(num_rows_ == A.NumRows());
    
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_mul_elements(dimGrid, dimBlock, data_, A.data_, Dim(), A.Stride());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().MulElements(A.Mat());
  }
}


template<typename Real>
void CuMatrixBase<Real>::MulRowsVec(const CuVectorBase<Real> &scale) {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(scale.Dim() == NumRows());

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_mul_rows_vec(dimGrid, dimBlock, data_, scale.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else 
  #endif
  {
    Mat().MulRowsVec(scale.Vec());
  }
}


template<typename Real>
void CuMatrixBase<Real>::AddMat(Real alpha, const CuMatrixBase<Real>& A, 
                                MatrixTransposeType transA) {

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (transA == kNoTrans) {
      KALDI_ASSERT(A.NumRows() == num_rows_ && A.NumCols() == num_cols_);
    } else {
      KALDI_ASSERT(A.NumCols() == num_rows_ && A.NumRows() == num_cols_);
    }
    if (num_rows_ == 0) return;
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));
    cuda_add_mat(dimGrid, dimBlock, alpha, A.data_, data_, Dim(), A.Stride(),
                 (transA == kTrans ? 1 : 0)); 
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().AddMat(alpha, A.Mat(), transA);
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddVecToRows(Real alpha,
                                      const CuVectorBase<Real> &row,
                                      Real beta) { 
  if (row.Dim() != NumCols()) {
    KALDI_ERR << "Non matching dimensions: Cols:" << NumCols() << " VectorDim:" << row.Dim();
  }
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
   
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_add_vec_to_rows(dimGrid, dimBlock, alpha, row.data_, beta, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    if (beta != 1.0) Mat().Scale(beta);
    Mat().AddVecToRows(alpha, row.Vec());
  }
}

//cudaF_sqrt_elements(int Gr, int Bl, float* data,
//MatrixDim d);

template<typename Real>
void CuMatrixBase<Real>::ApplySqrt(Real epsilon) {
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    if (num_rows_ == 0) return;
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_sqrt_elements(dimGrid, dimBlock, data_, epsilon, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    printf("ApplySqrt is currently not available on CPU.");
    exit(-101);
    //Mat().ApplySqrt(epsilon);
  }
}

template<typename Real>
void CuMatrixBase<Real>::InvertElements() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_invert_elements(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    printf("InvertElements is currently not available on CPU.");
    exit(-101);
    //Mat().InvertElements();
  }
}

/*
 * Method wrapping the CUBLAS function GEMM
 */
template<typename Real>
void CuMatrixBase<Real>::AddMatMat(
    Real alpha, const CuMatrixBase<Real> &A, MatrixTransposeType transA,
    const CuMatrixBase<Real> &B, MatrixTransposeType transB, Real beta) {


    // CUBLAS is col-major, cudamatrix is row-major, how to do the mapping?
    // keep trans..., just swap A&B matrices: A->B B->A
    MatrixIndexT m = ((transB==kTrans)? B.NumRows() : B.NumCols()); 
    MatrixIndexT n = ((transA==kTrans)? A.NumCols() : A.NumRows());
    MatrixIndexT k = ((transB==kTrans)? B.NumCols() : B.NumRows());
    MatrixIndexT k1 = ((transA==kTrans)? A.NumRows() : A.NumCols());

    KALDI_ASSERT(m == NumCols());
    KALDI_ASSERT(n == NumRows());
    KALDI_ASSERT(k == k1);

    if (m == 0) return;
    
    
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    cublas_gemm((transB==kTrans?'T':'N'), (transA==kTrans?'T':'N'), m, n, k, 
                alpha, B.data_, B.Stride(), A.data_, A.Stride(), 
                beta, data_, Stride());

    CU_SAFE_CALL(cublasGetError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().AddMatMat(alpha, A.Mat(), transA, B.Mat(), transB, beta);
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddMatMatElements(Real alpha,
    const CuMatrixBase<Real> &A, const CuMatrixBase<Real> &B, Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(SameDim(*this, A) && SameDim(A, B));
    Timer tim;
    
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));
    
    cuda_add_mat_mat_elements(dimGrid, dimBlock, this->data_, A.Data(),
                              B.Data(), Dim(), A.Stride(), B.Stride(), alpha, beta);
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    printf("ApplySqrt is currently not available on CPU.");
    exit(-101);
    //Mat().AddMatMatElements(alpha, A.Mat(), B.Mat(), beta);
  }
}

// <jiayu>
template<typename Real>
void CuMatrixBase<Real>::AddMatDiagVec(const Real alpha, 
                                       const CuMatrixBase<Real> &M, MatrixTransposeType transM,
                                       CuVectorBase<Real> &v,
                                       Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (transM == kNoTrans) {
      KALDI_ASSERT(SameDim(*this, M));
    } else {
      KALDI_ASSERT(M.NumRows() == NumCols() && M.NumCols() == NumRows());
    }
    KALDI_ASSERT(v.Dim() == this->NumCols());

    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    // Caution, this dimGrid is not the same way around as much of the other
    // code: going forward, I want to use the (rows, cols) order.
    dim3 dimGrid(n_blocks(num_rows_, CU2DBLOCK), n_blocks(num_cols_, CU2DBLOCK));

    MatrixIndexT M_row_stride = M.Stride(), M_col_stride = 1;
    if (transM == kTrans) std::swap(M_row_stride, M_col_stride);

    cuda_add_mat_diag_vec(dimGrid, dimBlock, alpha, data_, Dim(),
                          M.Data(), M_row_stride, M_col_stride, v.Data(),  beta);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().AddMatDiagVec(alpha, M.Mat(), transM, v.Vec(), beta);
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddMatDotMat(Real alpha, 
                                      const CuMatrixBase<Real> &A, MatrixTransposeType transA,
                                      const CuMatrixBase<Real> &B, MatrixTransposeType transB, 
                                      Real beta) {
    // for now kTrans is not supported
    KALDI_ASSERT(transA != kTrans);
    KALDI_ASSERT(transB != kTrans);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));
    cuda_add_mat_dot_mat(dimGrid, dimBlock, this->data_, A.Data(), B.Data(), 0, 0, Dim(), A.Stride(), B.Stride(), alpha, beta);
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().AddMatDotMat(alpha, A.Mat(), transA, B.Mat(), transB, beta);
  }
}

template<typename Real> // Y->this, X->src
void CuMatrixBase<Real>::ApplySoftMaxPerRow(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    size_t dimBlock = src.num_cols_ > CU1DBLOCK ? CU1DBLOCK : src.num_cols_;
    size_t dimGrid = src.num_rows_;
    cuda_softmax_reduce(dimGrid, dimBlock, data_, src.data_, Dim(), src.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<Real> &mat(this->Mat());
    mat.CopyFromMat(src.Mat());
    for(MatrixIndexT r = 0; r < mat.NumRows(); r++) {
      mat.Row(r).ApplySoftMax();
    }
  }
}

template<typename Real>
void CuMatrixBase<Real>::Sigmoid(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(src.NumCols(), CU2DBLOCK), n_blocks(src.NumRows(), CU2DBLOCK));
    
    cuda_sigmoid(dimGrid, dimBlock, this->data_, src.data_, this->Dim(), src.Stride());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Sigmoid(src.Mat());
  }
}

// DiffSigmoid(Ein, Y, Eout) -> Eout.DiffSigmoid(Y, Ein).
template<typename Real> // Eout -> *this, Ein -> diff, Y -> value
void CuMatrixBase<Real>::DiffSigmoid(const CuMatrixBase<Real> &value,
                                     const CuMatrixBase<Real> &diff) {
  KALDI_ASSERT(SameDim(*this, value) && SameDim(*this, diff));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CU2DBLOCK), n_blocks(num_rows_, CU2DBLOCK));

    cuda_diff_sigmoid(dimGrid, dimBlock, data_, diff.data_, value.data_, Dim(), diff.Stride(), value.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().DiffSigmoid(value.Mat(), diff.Mat());
  }
}

  
template<typename Real>
void CuMatrixBase<Real>::Tanh(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(src.NumCols(), CU2DBLOCK), n_blocks(src.NumRows(), CU2DBLOCK));

    cuda_tanh(dimGrid, dimBlock, this->data_, src.data_, this->Dim(), src.Stride());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().Tanh(src.Mat());
  }
}



template<typename Real> // Ein -> diff, Y -> value
void CuMatrixBase<Real>::DiffTanh(const CuMatrixBase<Real> &value,
                                  const CuMatrixBase<Real> &diff) {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CU2DBLOCK), n_blocks(num_rows_, CU2DBLOCK));

    cuda_diff_tanh(dimGrid, dimBlock, data_, diff.data_, value.data_, Dim(), diff.Stride(), value.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().DiffTanh(value.Mat(), diff.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::ComputeCtcAlpha(const CuMatrixBase<Real> &prob,
                                         int32 row_idx,
                                         const std::vector<MatrixIndexT> &labels,
                                         bool rescale) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(prob.NumRows() == NumRows());
    KALDI_ASSERT(static_cast<MatrixIndexT>(labels.size()) == NumCols());
#ifdef KALDI_PARANOID
    MatrixIndexT prob_cols = prob.NumCols();
    for (size_t i = 0; i < labels.size(); i++)
      KALDI_ASSERT(labels[i] >= 0 && labels[i] < prob_cols);
#endif
    CuArray<MatrixIndexT> cuda_labels(labels);

    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(num_cols_,CU1DBLOCK));
    if (rescale) {
      cuda_compute_ctc_alpha_rescale(dimGrid, dimBlock, data_, row_idx, Dim(), prob.data_, prob.Dim(), cuda_labels.Data());
    } else {
      cuda_compute_ctc_alpha(dimGrid, dimBlock, data_, row_idx, Dim(), prob.data_, prob.Dim(), cuda_labels.Data());
    }
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
 {
    // not implemented for CPU yet
 }
}

template<typename Real>
void CuMatrixBase<Real>::ComputeCtcAlphaMSeq(const CuMatrixBase<Real> &prob,
                                         int32 row_idx,
                                         const std::vector<MatrixIndexT> &labels,
                                         const std::vector<int32> &frame_num_utt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(prob.NumRows() == NumRows());
    KALDI_ASSERT(static_cast<MatrixIndexT>(labels.size()) % NumCols() == 0);
//    KALDI_ASSERT(frame_num_utt.size() == (labels.size() / exp_len_labels));
#ifdef KALDI_PARANOID
    MatrixIndexT prob_cols = prob.NumCols();
    for (size_t i = 0; i < labels.size(); i++)
      KALDI_ASSERT(labels[i] >= -1 && labels[i] < prob_cols);
#endif
    CuArray<MatrixIndexT> cuda_labels(labels);
    CuArray<int32> cuda_frame_nums(frame_num_utt);
    int32 seq_num = frame_num_utt.size();

    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(seq_num, CU2DBLOCK), n_blocks(NumCols(), CU2DBLOCK));
    cuda_compute_ctc_alpha_multiple_sequence(dimGrid, dimBlock, data_, seq_num, row_idx, Dim(), prob.data_, prob.Dim(), cuda_labels.Data(), NumCols(), cuda_frame_nums.Data());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
 {
    // not implemented for CPU yet
 }
}

template<typename Real>
void CuMatrixBase<Real>::ComputeCtcBeta(const CuMatrixBase<Real> &prob,
                                         int32 row_idx,
                                         const std::vector<MatrixIndexT> &labels,
                                         bool rescale) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(prob.NumRows() == NumRows());
    KALDI_ASSERT(static_cast<MatrixIndexT>(labels.size()) == NumCols());
#ifdef KALDI_PARANOID
    MatrixIndexT prob_cols = prob.NumCols();
    for (size_t i = 0; i < labels.size(); i++)
      KALDI_ASSERT(labels[i] >= 0 && labels[i] < prob_cols);
#endif
    CuArray<MatrixIndexT> cuda_labels(labels);
    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(num_cols_,CU1DBLOCK));
    if (rescale) {
       cuda_compute_ctc_beta_rescale(dimGrid, dimBlock, data_, row_idx, Dim(), prob.data_, prob.Dim(), cuda_labels.Data());
    } else {
       cuda_compute_ctc_beta(dimGrid, dimBlock, data_, row_idx, Dim(), prob.data_, prob.Dim(), cuda_labels.Data());
    }
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
 {
    // not implemented for CPU yet
 }
}

template<typename Real>
void CuMatrixBase<Real>::ComputeCtcBetaMSeq(const CuMatrixBase<Real> &prob,
                                         int32 row_idx,
                                         const std::vector<MatrixIndexT> &labels,
                                         const std::vector<int32> &frame_num_utt,
                                         const std::vector<int32> &label_lengths_utt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(prob.NumRows() == NumRows());
    KALDI_ASSERT(static_cast<MatrixIndexT>(labels.size()) % NumCols() == 0);
#ifdef KALDI_PARANOID
    MatrixIndexT prob_cols = prob.NumCols();
    for (size_t i = 0; i < labels.size(); i++)
      KALDI_ASSERT(labels[i] >= -1 && labels[i] < prob_cols);
#endif
    CuArray<MatrixIndexT> cuda_labels(labels);
    CuArray<int32> cuda_frame_nums(frame_num_utt);
    CuArray<int32> cuda_label_lengths(label_lengths_utt);
    int32 seq_num = frame_num_utt.size();

    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(seq_num, CU2DBLOCK), n_blocks(NumCols(), CU2DBLOCK));
    cuda_compute_ctc_beta_multiple_sequence(dimGrid, dimBlock, data_, seq_num, row_idx, Dim(), prob.data_, prob.Dim(), cuda_labels.Data(), NumCols(), cuda_frame_nums.Data(), cuda_label_lengths.Data());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
 {
    // not implemented for CPU yet
 }
}

template<typename Real>
void CuMatrixBase<Real>::ComputeCtcError(const CuMatrixBase<Real> &alpha,
                                         const CuMatrixBase<Real> &beta,
                                         const CuMatrixBase<Real> &prob,
                                         const std::vector<int32> &labels,
                                         Real pzx) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(alpha.NumRows() == NumRows() && beta.NumRows() == NumRows() && prob.NumRows() == NumRows());
    KALDI_ASSERT(alpha.NumCols() == beta.NumCols());
    KALDI_ASSERT(prob.NumCols() == NumCols());
    KALDI_ASSERT(static_cast<MatrixIndexT>(labels.size()) == alpha.NumCols());
#ifdef KALDI_PARANOID
    MatrixIndexT prob_cols = prob.NumCols();
    for (size_t i = 0; i < labels.size(); i++)
      KALDI_ASSERT(labels[i] >= 0 && labels[i] < prob_cols);
#endif
    CuArray<MatrixIndexT> cuda_labels(labels);

    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumRows(), CU2DBLOCK), n_blocks(NumCols(), CU2DBLOCK));
    cuda_compute_ctc_error(dimGrid, dimBlock, data_, Dim(), alpha.data_, beta.data_, alpha.Dim(), prob.data_, cuda_labels.Data(), pzx);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
 {
    // not implemented for CPU yet
 }
}

template<typename Real>
void CuMatrixBase<Real>::ComputeCtcErrorMSeq(const CuMatrixBase<Real> &alpha,
                                         const CuMatrixBase<Real> &beta,
                                         const CuMatrixBase<Real> &prob,
                                         const std::vector<int32> &labels,
                                         const std::vector<int32> &frame_num_utt,
                                         const CuVector<Real> pzx) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(alpha.NumRows() == NumRows() && beta.NumRows() == NumRows() && prob.NumRows() == NumRows());
    KALDI_ASSERT(alpha.NumCols() == beta.NumCols());
    KALDI_ASSERT(prob.NumCols() == NumCols());
    KALDI_ASSERT(static_cast<MatrixIndexT>(labels.size()) % alpha.NumCols() == 0);
#ifdef KALDI_PARANOID
    MatrixIndexT prob_cols = prob.NumCols();
    for (size_t i = 0; i < labels.size(); i++)
      KALDI_ASSERT(labels[i] >= -1 && labels[i] < prob_cols);
#endif
    CuArray<MatrixIndexT> cuda_labels(labels);
    CuArray<MatrixIndexT> cuda_frame_nums(frame_num_utt);
    int32 seq_num = frame_num_utt.size();

    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumRows(), CU2DBLOCK), n_blocks(NumCols(), CU2DBLOCK));
    cuda_compute_ctc_error_multiple_sequence(dimGrid, dimBlock, data_, seq_num, Dim(), alpha.data_, beta.data_, alpha.Dim(), prob.data_, cuda_labels.Data(), alpha.NumCols(), cuda_frame_nums.Data(), pzx.Data());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
 {
    // not implemented for CPU yet
 }
}


template<typename Real>
void CuMatrixBase<Real>::FindRowMaxId(CuArray<int32> *id) const {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
     
    // initialize the vectors
    CuVector<Real> max(num_rows_);
    max.Set(-1e21);
    id->Resize(num_rows_);
    id->Set(-1);

    MatrixDim d=Dim(); // only stride will be used!
   
    // process per 256 column blocks 
    for (int32 block = 0; (block+1)*256 <= num_cols_; block++) {
      dim3 dimBlock(CU1DBLOCK, 1);
      dim3 dimGrid(1, num_rows_);
      int32 offset = block*CU1DBLOCK;

      cuda_find_row_max_id(dimGrid, dimBlock, data_ + offset,
                           max.data_, id->Data(), offset, d);
    }
    
    // process the remainder
    int32 div = num_cols_ / 256;
    int32 mod = num_cols_ % 256;
    if (mod != 0) {
      dim3 dimBlock(mod, 1);
      dim3 dimGrid(1, num_rows_);
      int32 offset=div*256;
      
      cuda_find_row_max_id(dimGrid, dimBlock, data_ + offset,
                           max.data_, id->Data(), offset, d);
    }
    // now we have the indices!
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    // allocate index buffer
    id->Resize(num_rows_);
    id->Set(-1);
    // find maxima
    MatrixIndexT num_rows = num_rows_, num_cols = num_cols_;
    for(MatrixIndexT r = 0; r < num_rows; r++) {
      Real max = -1e21;
      int32 max_id = -1;
      const Real *row_data = Mat().RowData(r);
      for(MatrixIndexT c = 0; c < num_cols; c++) {
        if (max < row_data[c]) {
          max = row_data[c];
          max_id = c;
        }
      }
      id->Data()[r] = max_id;
    }
  }
}


template<typename Real>
void CuMatrixBase<Real>::ApplyFloor(Real floor_val) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_apply_floor(dimGrid, dimBlock, data_, floor_val, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().ApplyFloor(floor_val);
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyCeiling(Real ceiling_val) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_apply_ceiling(dimGrid, dimBlock, data_, ceiling_val, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().ApplyCeiling(ceiling_val);
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyPow(Real power) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_apply_pow(dimGrid, dimBlock, data_, power, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().ApplyPow(power);
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyHeaviside() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumRows(), CU2DBLOCK),
                 n_blocks(NumCols(), CU2DBLOCK));

    cuda_apply_heaviside(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().ApplyHeaviside();
  }
}

template<typename Real>
void VectorBase<Real>::CopyRowsFromMat(const CuMatrixBase<Real> &mat) {
  KALDI_ASSERT(dim_ == mat.NumCols() * mat.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    if (mat.Stride() == mat.NumCols()) {
      cudaMemcpy(data_, mat.Data(), sizeof(Real)*dim_, cudaMemcpyDeviceToHost);
    } else {
      Real* vec_data = data_;
      for (MatrixIndexT r = 0; r < mat.NumRows(); r++) {
        cudaMemcpy(vec_data, mat.RowData(r), sizeof(Real) * mat.NumCols(),
                   cudaMemcpyDeviceToHost);
        vec_data += mat.NumCols();
      }
    }
    CuDevice::Instantiate().AccuProfile("CuVectorBase::CopyRowsFromMat", tim.Elapsed());
  } else
#endif
  {
    CopyRowsFromMat(mat.Mat());
  }
}

// Instantiate the template above.
template
void VectorBase<float>::CopyRowsFromMat(const CuMatrixBase<float> &mat);
template
void VectorBase<double>::CopyRowsFromMat(const CuMatrixBase<double> &mat);

template<typename Real>
void CuMatrixBase<Real>::SetRandn() {
  if (num_rows_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuRand<Real> tmp;
    tmp.RandGaussian(this);
  } else 
#endif
  {
    Mat().SetRandn();
  }
}

template<typename Real>
void CuMatrixBase<Real>::SetRandUniform() {
  if (num_rows_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuRand<Real> tmp;
    tmp.RandUniform(this);
  } else
#endif
  {
    Mat().SetRandUniform();
  }
}

template<typename Real>
void CuMatrixBase<Real>::InitRandUniform(Real range) {
  if (num_rows_ == 0) return;
  this->SetRandUniform();   // randomly between [0, 1]
//  this->Scale(2 * range);   // then between [0, 2*range]
//  this->Add(-range);        // between [-range, range]
  this->Add(-0.5);
  this->Scale(2 * range);
}

template<typename Real>
void Matrix<Real>::Swap(CuMatrix<Real> *mat) { mat->Swap(this); }
// instantiate the template above.
template void Matrix<float>::Swap(CuMatrix<float> *mat);
template void Matrix<double>::Swap(CuMatrix<double> *mat);

/// Copy constructor from another type.
template<typename Real>
template<typename OtherReal>
CuMatrix<Real>::CuMatrix(const CuMatrixBase<OtherReal> & M,
                         MatrixTransposeType trans) : CuMatrixBase<Real>() {

  if (trans == kNoTrans) {
    Resize(M.NumRows(), M.NumCols());
    this->CopyFromMat(M);
  } else {
    Resize(M.NumCols(), M.NumRows());
    this->CopyFromMat(M, kTrans);
  }

}

// Instantiate this constructor for float->double and double->float.
template
CuMatrix<float>::CuMatrix(const CuMatrixBase<double> & M,
                          MatrixTransposeType trans);
template
CuMatrix<double>::CuMatrix(const CuMatrixBase<float> & M,
                           MatrixTransposeType trans);

/*
template<typename Real>
void CuMatrix<Real>::Transpose() {
  if (this->num_rows_ == 0)
    return;
#if HAVE_CUDA == 1
  if (this->num_rows_ == this->num_cols_ && CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    // (x,y) indices will be (row of *this, col of *this)
    dim3 dimGrid(n_blocks(this->num_rows_, CU2DBLOCK),
                 n_blocks(this->num_cols_, CU2DBLOCK));
    cuda_transpose_matrix(dimGrid, dimBlock, this->data_, this->Dim());
    CU_SAFE_CALL(cudaGetLastError());    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    CuMatrix<Real> tmp(*this, kTrans);
    *this = tmp;
  }
}
*/

/**
 * Print the matrix to stream
 */
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuMatrixBase<Real> &mat) {
  Matrix<Real> temp(mat.NumRows(), mat.NumCols());
  mat.CopyToMat(&temp);
  out << temp;
  return out;
}
// instantiate the template
template
std::ostream &operator << (std::ostream &out, const CuMatrixBase<float> &mat);
template 
std::ostream &operator << (std::ostream &out, const CuMatrixBase<double> &mat);


// Instantiate classes CuMatrix and CuMatrixBase for float and double.
template class CuMatrix<float>;
template class CuMatrix<double>;
template class CuMatrixBase<float>;
template class CuMatrixBase<double>;


} // namespace eesen
