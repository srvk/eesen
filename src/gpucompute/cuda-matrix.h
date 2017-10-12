// gpucompute/cuda-matrix.h

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


#ifndef EESEN_GPUCOMPUTE_CUDA_MATRIX_H_
#define EESEN_GPUCOMPUTE_CUDA_MATRIX_H_

#include <sstream>

#include "gpucompute/cuda-matrixdim.h"
#include "gpucompute/cuda-common.h"
#include "gpucompute/cuda-value.h"
#include "cpucompute/matrix-common.h"
#include "cpucompute/matrix.h"
#include "gpucompute/cuda-array.h"
#include "gpucompute/cuda-math.h"
#include "gpucompute/cuda-rand.h"

namespace eesen {

template<typename Real>
class CuMatrixBase {
 public:
  friend class CuMatrixBase<float>;
  friend class CuMatrixBase<double>;
  friend class CuVectorBase<float>;
  friend class CuVectorBase<double>;
  friend class VectorBase<Real>;
  friend class CuVectorBase<Real>;
  friend class CuSubMatrix<Real>;
  friend class CuRand<Real>;
  friend class CuSubVector<Real>;
  friend void cu::RegularizeL1<Real>(CuMatrixBase<Real> *weight,
                                     CuMatrixBase<Real> *grad, Real l1, Real lr);
  friend void cu::Splice<Real>(const CuMatrixBase<Real> &src,
                               const CuArray<int32> &frame_offsets,
                               CuMatrixBase<Real> *tgt);
  friend void cu::Copy<Real>(const CuMatrixBase<Real> &src,
                             const CuArray<int32> &copy_from_indices,
                             CuMatrixBase<Real> *tgt);
  friend void cu::Randomize<Real>(const CuMatrixBase<Real> &src,
                                  const CuArray<int32> &copy_from_idx,
                                  CuMatrixBase<Real> *tgt);

  /////////////////////////////////////////////////////
  ///  Dimensions
  ////////////////////////////////////////////////////
  MatrixIndexT NumRows() const { return num_rows_;  }
  MatrixIndexT NumCols() const { return num_cols_;  }
  MatrixIndexT Stride() const { return stride_; }

  // MatrixDim is a struct containing "rows", "cols" and "stride",
  // that is an argument of most CUDA kernels.
  ::MatrixDim Dim() const {
    ::MatrixDim d = { num_rows_, num_cols_, stride_ };
    return d;
  }

  /////////////////////////////////////////////////////
  /// Various copy functions
  /////////////////////////////////////////////////////
  template<typename OtherReal>
  void CopyFromMat(const MatrixBase<OtherReal> &src,
                   MatrixTransposeType trans = kNoTrans);

  void CopyFromMat(const MatrixBase<Real> &src,
                   MatrixTransposeType trans = kNoTrans);

  template<typename OtherReal>
  void CopyFromMat(const CuMatrixBase<OtherReal> &M,
                   MatrixTransposeType trans = kNoTrans);

  template<typename OtherReal>
  void CopyToMat(MatrixBase<OtherReal> *dst,
                 MatrixTransposeType trans = kNoTrans) const;


  /////////////////////////////////////////////////////
  ///////  Basic operations
  /////////////////////////////////////////////////////
  /// Set all the elements to 0
  void SetZero();
  /// Set all the elements to value
  void Set(Real value);
  /// Add value to every element
  void Add(Real value);
  /// Scale every element by value
  void Scale(Real value);
  /// Apply log()
  void ApplyLog();
  /// Apply pow()
  void ApplyPow(Real power);
  /// Apply sqrt(x+epsilon)
  void ApplySqrt(Real epsilon);
  /// Sum of the matrix
  Real Sum() const;
  /// If the elements < floor_val, set them to floor_val
  void ApplyFloor(Real floor_val);
  /// If the elements > ceiling_val, set them to ceiling_val
  void ApplyCeiling(Real ceiling_val);
  /// Element-wise x > 0 ? 1.0 : 0.0
  void ApplyHeaviside();
  /// Find the id of the maximal element for each row
  void FindRowMaxId(CuArray<int32> *id) const;
  /// Set to random values drawn from Gaussian distribution
  void SetRandn();
  /// Set to random values drawn from a uniform distribution [0, 1]
  void SetRandUniform();
  /// Set to random values drawn from a uniform distribution [-range, range]
  void InitRandUniform(Real range);
  // Invert elements
  void InvertElements();

  /////////////////////////////////////////////////////
  /////  Activation
  /////////////////////////////////////////////////////

  /// Apply softmax to each row
  void ApplySoftMaxPerRow(const CuMatrixBase<Real> &src);

  /// Apply the sigmoid function to each element: x = 1 / (1 + exp(-x))
  void Sigmoid(const CuMatrixBase<Real> &src);

  /// Apply the tanh function to each element x = (1 - exp(-2x)) / (1 + exp(-2x))
  void Tanh(const CuMatrixBase<Real> &src);

  /// Back-propagation through the sigmoid function.  Here, "value" is the
  /// sigmoid output. *this = diff * value * (1 - value).
  void DiffSigmoid(const CuMatrixBase<Real> &value,
                   const CuMatrixBase<Real> &diff);

  /// Back-propagation through the tanh function.  Here, "value" is the
  /// tanh output.  *this = diff * (1 - value^2).
  void DiffTanh(const CuMatrixBase<Real> &value,
                const CuMatrixBase<Real> &diff);


  /////////////////////////////////////////////////////
  /////  CTC Training
  /////////////////////////////////////////////////////

  /// Perform a CTC foward pass over a single sequence, computing the alpha values. Here, "rescale"
  /// is a boolean value indicating whether the scaling version is used.
  void ComputeCtcAlpha(const CuMatrixBase<Real> &prob,
                       int32 row_idx,
                       const std::vector<int32> &labels,
                       bool rescale);

  /// Computing alpha values by processing multiple sequences at one time.
  void ComputeCtcAlphaMSeq(const CuMatrixBase<Real> &prob,
                       int32 row_idx,
                       const std::vector<int32> &labels,
                       const std::vector<int32> &frame_num_utt);

  /// Perform a CTC backward pass over a single sequence, computing the beta values. Here, "rescale"
  /// is a boolean value indicating whether the scaling version is used.
  void ComputeCtcBeta(const CuMatrixBase<Real> &prob,
                      int32 row_idx,
                      const std::vector<int32> &labels,
                      bool rescale);

  /// Computing beta values by processing multiple sequences at one time.
  void ComputeCtcBetaMSeq(const CuMatrixBase<Real> &prob,
                       int32 row_idx,
                       const std::vector<int32> &labels,
                       const std::vector<int32> &frame_num_utt,
                       const std::vector<int32> &label_lengths_utt);

  /// Evaluate the errors from the CTC objective over a single sequence.
  void ComputeCtcError(const CuMatrixBase<Real> &alpha,
                       const CuMatrixBase<Real> &beta,
                       const CuMatrixBase<Real> &prob,
                       const std::vector<int32> &labels,
                       Real pzx);

 /// Evaluate the errors from the CTC objective over multiple sequences.
 void  ComputeCtcErrorMSeq(const CuMatrixBase<Real> &alpha,
                           const CuMatrixBase<Real> &beta,
                           const CuMatrixBase<Real> &prob,
                           const std::vector<int32> &labels,
                           const std::vector<int32> &frame_num_utt,
                           const CuVector<Real> pzx);


  /////////////////////////////////////////////////////
  ///// Misc for Matrix and Vector Computation
  /////////////////////////////////////////////////////

  /// Elementwise multiplication of *this matrix and A
  void MulElements(const CuMatrixBase<Real> &A);

  /// Scale i'th row of *this matrix by scale[i]
  void MulRowsVec(const CuVectorBase<Real> &scale);

  /// Add vector to each row of *this matrix. For each row r of *this, r = alpha * row + beta * r
  void AddVecToRows(Real alpha, const CuVectorBase<Real> &row, Real beta = 1.0);

  /// *this = alpha * A
  void AddMat(Real alpha, const CuMatrixBase<Real> &A, MatrixTransposeType transA = kNoTrans);

  /// Inner product of two matrices *this = alpha * A(^T)*B(^T) + beta * *this
  void AddMatMat(Real alpha, const CuMatrixBase<Real> &A, MatrixTransposeType transA,
                 const CuMatrixBase<Real> &B, MatrixTransposeType transB, Real beta);

  /// Same as adding M, but scaling the i-th column of M by v(i)
  /// *this = beta * *this + alpha * M  * diag(v).
  void AddMatDiagVec(const Real alpha,
                     const CuMatrixBase<Real> &M, MatrixTransposeType transM,
                     CuVectorBase<Real> &v,
                     Real beta = 1.0);

  // Dot product *this = alpha * a * b + beta * *this;
  void AddMatDotMat(const Real alpha,
                    const CuMatrixBase<Real>& A, MatrixTransposeType transA,
                    const CuMatrixBase<Real>& B, MatrixTransposeType transB,
                    const Real beta);


  void AddMatMatElements(Real alpha,
                         const CuMatrixBase<Real> &A, 
                         const CuMatrixBase<Real> &B, 
                         Real beta);

  /////////////////////////////////////////////////////
  ///// SubMatrix and SubVector
  /////////////////////////////////////////////////////
  inline CuSubMatrix<Real> Range(const MatrixIndexT row_offset,
                                 const MatrixIndexT num_rows,
                                 const MatrixIndexT col_offset,
                                 const MatrixIndexT num_cols) const {
    return CuSubMatrix<Real>(*this, row_offset, num_rows,
                             col_offset, num_cols);
  }
  inline CuSubMatrix<Real> RowRange(const MatrixIndexT row_offset,
                                    const MatrixIndexT num_rows) const {
    return CuSubMatrix<Real>(*this, row_offset, num_rows,
                             0, num_cols_);
  }
  inline CuSubMatrix<Real> ColRange(const MatrixIndexT col_offset,
                                    const MatrixIndexT num_cols) const {
    return CuSubMatrix<Real>(*this, 0, num_rows_, col_offset, num_cols);
  }

  inline const CuSubVector<Real> Row(MatrixIndexT i) const {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
    return CuSubVector<Real>(data_ + (i * stride_), NumCols());
  }

  inline CuSubVector<Real> Row(MatrixIndexT i) {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
    return CuSubVector<Real>(data_ + (i * stride_), NumCols());
  }

  /////////////////////////////////////////////////////
  ///// Specific Element
  /////////////////////////////////////////////////////
  inline CuValue<Real> operator() (MatrixIndexT r, MatrixIndexT c) {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                          static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                          static_cast<UnsignedMatrixIndexT>(c) <
                          static_cast<UnsignedMatrixIndexT>(num_cols_));
    return CuValue<Real>(data_ + r * stride_ + c);
  }

  inline Real operator() (MatrixIndexT r, MatrixIndexT c) const {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                          static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                          static_cast<UnsignedMatrixIndexT>(c) <
                          static_cast<UnsignedMatrixIndexT>(num_cols_));
    return CuValue<Real>(data_ + r * stride_ + c);  // will be casted to Real.
  }

  void Write(std::ostream &os, bool binary) const;


 protected:
  // The following two functions should only be called if we did not compile with CUDA
  // or could not get a CUDA card; in that case the contents are interpreted the
  // same as a regular matrix.
  inline const MatrixBase<Real> &Mat() const {
    return *(reinterpret_cast<const MatrixBase<Real>* >(this));
  }
  inline MatrixBase<Real> &Mat() {
    return *(reinterpret_cast<MatrixBase<Real>* >(this));
  }

  /// Get raw row pointer
  inline const Real* RowData(MatrixIndexT r) const { return data_ + r * stride_; }
  inline Real* RowData(MatrixIndexT r) { return data_ + r * stride_; }
  inline const Real *Data() const { return data_; }
  inline Real *Data() { return data_; }



  // The constructors are protected to prevent the user creating an instance of
  // this class.

  /// Default constructor
  CuMatrixBase<Real>(): data_(NULL), num_cols_(0), num_rows_(0), stride_(0) { }

  /// This constructor takes the #rows, #cols and stride; it's called from
  /// the constructor of CuSubMatrix.
  CuMatrixBase<Real>(Real *data,
                     MatrixIndexT num_rows,
                     MatrixIndexT num_cols,
                     MatrixIndexT stride):
  data_(data), num_cols_(num_cols), num_rows_(num_rows), stride_(stride) { }

  Real *data_;       ///GPU data pointer (or regular matrix data pointer,
  MatrixIndexT num_cols_;
  MatrixIndexT num_rows_;
  MatrixIndexT stride_;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(CuMatrixBase);
}; // class CuMatrixBase

/// This class represents a matrix that's stored on the GPU if we have one,
/// and in memory if not.
template<typename Real>
class CuMatrix: public CuMatrixBase<Real> {
 public:

  CuMatrix() { }

  /// Constructor with memory initialisation
  CuMatrix(MatrixIndexT rows, MatrixIndexT cols,
           MatrixResizeType resize_type = kSetZero) {
    Resize(rows, cols, resize_type);
  }

  // Note: we had to remove the "explicit" keyword due
  // to problems with STL vectors of CuMatrixBase.
  CuMatrix(const CuMatrix<Real> &other,
           MatrixTransposeType trans = kNoTrans);

  explicit CuMatrix(const CuMatrixBase<Real> &other,
                    MatrixTransposeType trans = kNoTrans);

  template<typename OtherReal>
  explicit CuMatrix(const MatrixBase<OtherReal> &other,
                    MatrixTransposeType trans = kNoTrans);

  template<typename OtherReal>
  explicit CuMatrix(const CuMatrixBase<OtherReal> &M,
                    MatrixTransposeType trans = kNoTrans);

  CuMatrix<Real> &operator = (const CuMatrixBase<Real> &other) {
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
    this->CopyFromMat(other);
    return *this;
  }

  CuMatrix<Real> &operator = (const CuMatrix<Real> &other) {
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
    this->CopyFromMat(other);
    return *this;
  }

  CuMatrix<Real> &operator = (const MatrixBase<Real> &other) {
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
    this->CopyFromMat(other);
    return *this;
  }

  /// Allocate the memory
  void Resize(MatrixIndexT rows, MatrixIndexT cols,
              MatrixResizeType resize_type = kSetZero);

  void Swap(Matrix<Real> *mat);
  void Swap(CuMatrix<Real> *mat);

  /// I/O functions
  void Read(std::istream &is, bool binary);

  /// Destructor
  ~CuMatrix() { Destroy(); }

  inline const Matrix<Real> &Mat() const {
    return *(reinterpret_cast<const Matrix<Real>* >(this));
  }
  inline Matrix<Real> &Mat() {
    return *(reinterpret_cast<Matrix<Real>* >(this));
  }

 private:
  void Destroy();
};


/// This class is used for a piece of a CuMatrix.
template<typename Real>
class CuSubMatrix: public CuMatrixBase<Real> {
 public:
  CuSubMatrix() { }

  inline CuSubMatrix(const CuMatrixBase<Real> &mat,
                     const MatrixIndexT row_offset,
                     const MatrixIndexT num_rows,
                     const MatrixIndexT col_offset,
                     const MatrixIndexT num_cols);

  /// This type of constructor is needed for Range() to work [in CuMatrix base
  /// class]. Cannot make it explicit or that breaks.
  inline CuSubMatrix<Real> (const CuSubMatrix &other):
  CuMatrixBase<Real> (other.data_, other.num_cols_, other.num_rows_,
                      other.stride_) {}

  CuSubMatrix<Real> &operator = (const CuMatrix<Real> &other) {
    this->data_ = other.data_;
    this->num_cols_ = other.num_cols_;
    this->num_rows_ = other.num_rows_;
    this->stride_ = other.stride_;
    return *this;
  }

  CuSubMatrix<Real> &operator = (const CuSubMatrix<Real> &other) {
    this->data_ = other.data_;
    this->num_cols_ = other.num_cols_;
    this->num_rows_ = other.num_rows_;
    this->stride_ = other.stride_;
    return *this;
  }

 private:
  /// Disallow assignment.
  //CuSubMatrix<Real> &operator = (const CuSubMatrix<Real> &other);
};

template<typename Real>
bool SameDim(const CuMatrixBase<Real> &M, const CuMatrixBase<Real> &N) {
  return (M.NumRows() == N.NumRows() && M.NumCols() == N.NumCols());
}
/// I/O
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuMatrixBase<Real> &mat);


template<typename Real>
template<typename OtherReal>
Matrix<Real>::Matrix(const CuMatrixBase<OtherReal> &M,
                     MatrixTransposeType trans) {
  if (trans == kNoTrans) Init(M.NumRows(), M.NumCols());
  else Init(M.NumCols(), M.NumRows());
  M.CopyToMat(this, trans);
}

template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::CopyFromMat(const CuMatrixBase<OtherReal> &cu,
                                   MatrixTransposeType trans) {
  cu.CopyToMat(this, trans);
}

}  // namespace


#include "gpucompute/cuda-matrix-inl.h"

#endif
