// net/utils-functions.h

// Copyright 2012-2014  Brno University of Technology (author: Karel Vesely)
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


#ifndef EESEN_NET_UTILS_FUNCTIONS_H_
#define EESEN_NET_UTILS_FUNCTIONS_H_

#include "net/layer.h"
#include "gpucompute/cuda-math.h"
#include "util/text-utils.h"

#include <algorithm>
#include <sstream>

namespace eesen {

/**
 * Convert basic type to string (try not to overuse as ostringstream creation is slow)
 */
template <typename T> 
std::string ToString(const T& t) { 
  std::ostringstream os; 
  os << t; 
  return os.str(); 
}


/**
 * Get a string with statistics of the data in a vector,
 * so we can print them easily.
 */
template <typename Real>
std::string MomentStatistics(const VectorBase<Real> &vec) {
  // we use an auxiliary vector for the higher order powers
  Vector<Real> vec_aux(vec);
  Vector<Real> vec_no_mean(vec); // vec with mean subtracted
  // mean
  Real mean = vec.Sum() / vec.Dim();
  // variance
  vec_aux.Add(-mean); vec_no_mean = vec_aux;
  vec_aux.MulElements(vec_no_mean); // (vec-mean)^2
  Real variance = vec_aux.Sum() / vec.Dim();
  // skewness 
  // - negative : left tail is longer, 
  // - positive : right tail is longer, 
  // - zero : symmetric
  vec_aux.MulElements(vec_no_mean); // (vec-mean)^3
  Real skewness = vec_aux.Sum() / pow(variance, 3.0/2.0) / vec.Dim();
  // kurtosis (tailedness)
  // - makes sense for symmetric distributions (skewness is zero)
  // - negative : 'lighter tails' than Normal distribution
  // - positive : 'heavier tails' than Normal distribution
  // - zero : same as the Normal distribution
  vec_aux.MulElements(vec_no_mean); // (vec-mean)^4
  Real kurtosis = vec_aux.Sum() / (variance * variance) / vec.Dim() - 3.0;
  // send the statistics to stream,
  std::ostringstream ostr;
  ostr << " ( min " << vec.Min() << ", max " << vec.Max()
       << ", mean " << mean 
       << ", variance " << variance 
       << ", skewness " << skewness
       << ", kurtosis " << kurtosis
       << " ) ";
  return ostr.str();
}

/**
 * Overload MomentStatistics to MatrixBase<Real>
 */
template <typename Real>
std::string MomentStatistics(const MatrixBase<Real> &mat) {
  Vector<Real> vec(mat.NumRows()*mat.NumCols());
  vec.CopyRowsFromMat(mat);
  return MomentStatistics(vec);
}

/**
 * Overload MomentStatistics to CuVectorBase<Real>
 */
template <typename Real>
std::string MomentStatistics(const CuVectorBase<Real> &vec) {
  Vector<Real> vec_host(vec.Dim());
  vec.CopyToVec(&vec_host);
  return MomentStatistics(vec_host);
}

/**
 * Overload MomentStatistics to CuMatrix<Real>
 */
template <typename Real>
std::string MomentStatistics(const CuMatrixBase<Real> &mat) {
  Matrix<Real> mat_host(mat.NumRows(),mat.NumCols());
  mat.CopyToMat(&mat_host);
  return MomentStatistics(mat_host);
}

/**
 * Check that matrix contains no nan or inf
 */
template <typename Real>
void CheckNanInf(const CuMatrixBase<Real> &mat, const char *msg = "") {
  Real sum = mat.Sum();
  if(KALDI_ISINF(sum)) { KALDI_ERR << "'inf' in " << msg; }
  if(KALDI_ISNAN(sum)) { KALDI_ERR << "'nan' in " << msg; }
}

/**
 * Get the standard deviation of values in the matrix
 */
template <typename Real>
Real ComputeStdDev(const CuMatrixBase<Real> &mat) {
  int32 N = mat.NumRows() * mat.NumCols();
  Real mean = mat.Sum() / N;
  CuMatrix<Real> pow_2(mat);
  pow_2.MulElements(mat);
  Real var = pow_2.Sum() / N - mean * mean;
  if (var < 0.0) {
    KALDI_WARN << "Forcing the variance to be non-negative! " << var << "->0.0";
    var = 0.0;
  }
  return sqrt(var);
}

} // namespace eesen

#endif // EESEN_NET_UTILS_FUNCTIONS_H_
