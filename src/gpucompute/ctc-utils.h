// gpucompute/ctc-utils.h

// Copyright 2015  Yajie Miao

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



#ifndef EESEN_GPUCOMPUTE_CTC_UTILS_H_
#define EESEN_GPUCOMPUTE_CTC_UTILS_H_

/*
 * Some numeric limits and operations. These limits and operations
 * are used in CTC computation/evaluation.
 */
template <typename T>
struct NumericLimits;

template <>
struct NumericLimits<float>
{
  static constexpr float log_zero_ = -1e30f;
  static constexpr float exp_limit_ = 88.722839f;
  static constexpr float log_inf_ = 1e30f;
  static constexpr float max_ = 3.4028235e+038f;
};

template <>
struct NumericLimits<double>
{
  static constexpr double log_zero_ = -1e100;
  static constexpr double exp_limit_ = 709.78271289338397;
  static constexpr double log_inf_ = 1e100;
  static constexpr double max_ = 1.7976931348623157e+308;
};

#if HAVE_CUDA == 1

// a + b, where a and b are assumed to be in the log scale 
template <typename T>
static inline __host__ __device__ T AddAB(T a, T b)
{
  if (a == NumericLimits<T>::log_zero_ || b == NumericLimits<T>::log_zero_)
    return NumericLimits<T>::log_zero_;
  else
    return a + b;
}

// a - b, where a and b are assumed to be in the log scale
template <typename T>
static inline __host__ __device__ T SubAB(T a, T b)
{
  if (a == NumericLimits<T>::log_zero_)
    return NumericLimits<T>::log_zero_;
  else if (b == NumericLimits<T>::log_zero_)
    return NumericLimits<T>::log_inf_;
  else
    return a - b;
}

// exp(a)
template <typename T>
static inline __host__ __device__ T ExpA(T a)
{
  if (a <= NumericLimits<T>::log_zero_)
    return 0;
  else if (a >= NumericLimits<T>::exp_limit_)
    return NumericLimits<T>::max_;
  else
    return exp(a);
}

// Approximation of  log(a + b) = log(a) + log(1 + b/a), if b < a
//                              = log(b) + log(1 + a/b), if a < b
template <typename T>
static inline __host__ __device__ T LogAPlusB(T a, T b) // x and y are in log scale and so is the result
  {
    if (b < a)
      return AddAB(a, log(1 + ExpA(SubAB(b, a))));
    else
      return AddAB(b, log(1 + ExpA(SubAB(a, b))));
  }

#else

// exp(a)
template <typename T>
static inline T ExpA(T a)
{
  return exp(a);
}

#endif // HAVE_CUDA

#endif
