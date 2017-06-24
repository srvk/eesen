// gpucompute/cuda-kernels.cu

// Copyright 2009-2012  Karel Vesely
//                2013  Ehsan Variani
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2013  Hainan Xu
//                2013  Xiaohui Zhang
//                2013  Johns Hopkins University (author: Guoguo Chen)
//                2015  Yajie Miao

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

// In this file is the CUDA code of the CUDA kernels, plus the ANSI-C wrappers

#include <cfloat>
#include "cuda-kernels.h"
#include "cuPrintf.cuh"
#include "cuPrintf.cu"
#include "ctc-utils.h"
#include "stdio.h"

/***********************************************************************
 * Generic __device__ functions
 */
template<typename Real>
__device__
static Real _sum_reduce(Real buffer[]) {
  // Total number of active threads
  int32_cuda nTotalThreads = blockDim.x;	
  __syncthreads();
  // perform tree-based reduction (sum)
  while(nTotalThreads > 1) {
    int32_cuda halfPoint = ((1+nTotalThreads) >> 1);	// divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x >= halfPoint)  { // was <
      // Get the shared value stored by another thread
      Real temp = 0.0;
      if(threadIdx.x < nTotalThreads) { // was +halfPoint
        temp = buffer[threadIdx.x]; // was +halfPoint
      }
      buffer[threadIdx.x - halfPoint] += temp;
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);	// divide by two.
  }
  // the result
  return buffer[0];
}


template<typename Real>
__device__
static Real _min_reduce(Real buffer[]) {
  // Total number of active threads
  int32_cuda nTotalThreads = blockDim.x;
  __syncthreads();
  // perform tree-based reduction (min)
  while(nTotalThreads > 1) {
    int32_cuda halfPoint = ((1+nTotalThreads) >> 1); // divide by two
    // only the first half of the threads will be active
    if (threadIdx.x < halfPoint) {
      if (threadIdx.x + halfPoint < nTotalThreads) {
        Real temp = buffer[threadIdx.x + halfPoint];
        if (temp < buffer[threadIdx.x]) 
           buffer[threadIdx.x] = temp;
      }
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1); // divide by two
  }
  // the result
  return buffer[0];
}


template<typename Real>
__device__
static Real _max_reduce(Real buffer[]) {
  // Total number of active threads
  int32_cuda nTotalThreads = blockDim.x;	
  __syncthreads();
  // perform tree-based reduction (max)
  while(nTotalThreads > 1) {
    int32_cuda halfPoint = ((1+nTotalThreads) >> 1);	// divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      if(threadIdx.x+halfPoint < nTotalThreads) {
        Real temp = buffer[threadIdx.x + halfPoint];
        if (temp > buffer[threadIdx.x]) 
          buffer[threadIdx.x] = temp;
      }
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);	// divide by two.
  }
  // the result
  return buffer[0];
}



template<typename Real>
__device__
static int32_cuda _max_id_reduce(Real val[], int32_cuda idx[]) {
  // Total number of active threads
  int32_cuda nTotalThreads = blockDim.x;	
  __syncthreads();
  // perform tree-based reduction (get index of maximum)
  while(nTotalThreads > 1) {
    int32_cuda halfPoint = ((1+nTotalThreads) >> 1);	// divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      Real temp = -1e20;
      if(threadIdx.x+halfPoint < nTotalThreads) {
        temp = val[idx[threadIdx.x + halfPoint]];
      }
      if (temp > val[idx[threadIdx.x]]) idx[threadIdx.x]=idx[threadIdx.x + halfPoint];
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);	// divide by two.
  }
  // the result
  return idx[0];
}




/***********************************************************************
 * CUDA kernels
 * the functions are templated to have the float/double operations
 */


// for this kernel, following the newer pattern, the x-dim is the row-index, the
// y-dim is the col-index.
template<typename Real, typename OtherReal>
__global__
static void _copy_from_mat(Real* mat_out, const OtherReal* mat_in, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; // row-index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; // col-index.
  int32_cuda index_out = j + i * d_out.stride;
  int32_cuda index_in = j + i * d_in.stride;
  if (i < d_out.rows && j < d_out.cols)
    mat_out[index_out] = static_cast<Real>(mat_in[index_in]);
}



// for this kernel, the x-dim is the row-index at the output, the y-dim is the
// col-index at the output
template<typename Real, typename OtherReal>
__global__
static void _copy_from_mat_trans(Real* mat_out, const OtherReal* mat_in, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; // row-index out
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; // col-index out
  int32_cuda index_out = j + i * d_out.stride;
  int32_cuda index_in = i + j * d_in.stride;
  if (i < d_out.rows && j < d_out.cols)
    mat_out[index_out] = static_cast<Real>(mat_in[index_in]);
}

template<typename Real>
__global__
static void _apply_exp(Real* mat, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if ( i < d.cols && j < d.rows ) {
    mat[index] = exp(mat[index]);
  }
}

template<typename Real>
__global__
static void _set_const(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = value;
}

template<typename Real>
__global__
static void _add(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = mat[index] + value;
}


template<typename Real>
__global__
static void _scale(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = mat[index] * value;
}


template<typename Real>
__global__
static void _apply_log(Real* mat, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = log(mat[index]);
}

template<typename Real>
__global__
static void _mul_elements(Real* mat, const Real* A, MatrixDim dst_d, int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda dst_index = i + j*dst_d.stride, src_index = i + j*src_stride;
  if (i < dst_d.cols  &&  j < dst_d.rows)
    mat[dst_index] = mat[dst_index] * A[src_index];
}

template<typename Real>
__global__
static void _vec_mul_elements(Real* v, const Real* a, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim)
    v[i] = v[i] * a[i];
}

template<typename Real>
__global__
static void _mul_rows_vec(Real* mat, const Real* scale, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] *= scale[j];
}

template<typename Real>
__global__
static void _add_mat(Real alpha, const Real* src, Real* dst, MatrixDim d, int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  int32_cuda index_src = i + j*src_stride;
  if (i < d.cols && j < d.rows)
    dst[index] = alpha*src[index_src] + dst[index];
}

template<typename Real>
__global__
static void _add_mat_trans(Real alpha, const Real* src, Real* dst, MatrixDim d, int src_stride) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j *d.stride;
  int32_cuda index_src = j + i*src_stride;
  if (i < d.cols && j < d.rows)
    dst[index] = alpha*src[index_src] + dst[index];
}

template<typename Real>
__global__
static void _add_vec_to_rows(Real alpha, const Real* row, Real beta, Real* dst, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride;
  if (i < d.cols && j < d.rows)
    dst[index] = alpha*row[i] + beta*dst[index];
}

/*
 * CuVector
 */
template<typename Real>
__global__
static void _copy_from_vec_df(double* v_out, const Real* v_in, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  //  if (blockIdx.y > 0) return;
  
  if (i < dim) {
    v_out[i] = (double) v_in[i];
  }
}

template<typename Real>
__global__
static void _copy_from_vec_fd(float* v_out, const Real* v_in, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  //  if (blockIdx.y > 0) return;
  
  if ( i < dim) {
    v_out[i] = (float) v_in[i];
  }
}


template<typename Real>
__global__
static void _vec_min(const Real* v, Real* value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= CU1DBLOCK) return;
  
  __shared__ Real row_data[CU1DBLOCK];

  int block_size = (dim + CU1DBLOCK - 1) / CU1DBLOCK;

  Real min = 1.0 / 0.0; // infinity.

  for (int j = i * block_size; j < (i+1) * block_size && j < dim; j++) {
     Real v_j = v[j];
     if (v_j < min) min = v_j;
  }

  row_data[i] = min;

  __syncthreads();

  //get the sum
  *value = _min_reduce(row_data);
}


template<typename Real>
__global__
static void _vec_max(const Real* v, Real* value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  if(blockIdx.y > 0) return;

  __shared__ Real row_data[CU1DBLOCK];

  if(i >= CU1DBLOCK) return;

  int block_size = (dim + CU1DBLOCK - 1) / CU1DBLOCK;

  Real max = -1.0 / 0.0; // -infinity.

  for (int j = i * block_size; j < (i+1) * block_size && j < dim; j++) {
     Real v_j = v[j];
     if (v_j > max) max = v_j;
  }

  row_data[i] = max;

  __syncthreads();

  //get the sum
  *value = _max_reduce(row_data);
}

// Adds diag(M N) to v, where M and N are matrices.  We supply row_stride and
// col_stride arguments for M and N, and swapping them allows us to transpose
// those matrices.  Note: we imagine row-major indexing here, just like Kaldi 
// and CBLAS (but unlike CUBLAS).
// This kernel expects the blockDim to be (CU1DBLOCK, 1) and the
// gridDim times CU1DBLOCK to be at least num-rows-of-v * threads_per_element.
// threads_per_element should be a power of 2.
template<typename Real>
__global__
static void _add_diag_mat_mat(
       Real alpha, Real* v, int v_dim, const Real* M, int M_cols, int M_row_stride,
       int M_col_stride, const Real *N, int N_row_stride, int N_col_stride,
       int threads_per_element, Real beta) {
  
  // we actually assume blockDim.x == CU1DBLOCK here.
  // Each diagonal element of v is processed by "threads_per_element" threads.
  __shared__ Real temp_data[CU1DBLOCK];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int v_idx = i / threads_per_element,   // v_idx is the index into v that we are supposed to
      sub_idx = i % threads_per_element; // add to; 0 <= sub_idx < threads_per_element tells 
                                         // us which block of elements we sum up.
  if (v_idx >= v_dim) return;
      
  Real sum = 0.0;
  for (int j = sub_idx; j < M_cols; j += threads_per_element) {
    int M_index = v_idx * M_row_stride + j * M_col_stride,
        N_index = j * N_row_stride + v_idx * N_col_stride;
    sum += M[M_index] * N[N_index];
  }
  temp_data[threadIdx.x] = sum;

  // start_idx = threadIdx.x - sub_idx; // start of the position in temp_data
                                     // that we want to sum up.
  // The following is a tree-based reduction of the elements of temp_data from
  // start_idx to start_idx + threads_per_element - 1; our own index is "sub_idx".
  __syncthreads();
  int num_total_threads = threads_per_element;
  while (num_total_threads > 1) {
    int half_point = ((1 + num_total_threads) >> 1);
    if (sub_idx < half_point) {
      Real temp = 0.0;
      if (sub_idx + half_point < num_total_threads) {
        temp = temp_data[threadIdx.x + half_point];
      }
      temp_data[threadIdx.x] += temp;
    }
    __syncthreads();
    num_total_threads = half_point;
  }
  if (sub_idx == 0) {
    v[v_idx] = beta * v[v_idx] + alpha * temp_data[threadIdx.x];
  }
}


template<typename Real>
__global__
static void _add_vec_vec(Real alpha, Real* v, const Real* x, const Real* y, Real beta, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  // if (blockIdx.y > 0) return;

  if (i < dim)
    v[i] = alpha * x[i] * y[i] + beta * v[i];
}

template<typename Real>
__global__
static void _vec_apply_exp(Real* v, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  // if (blockIdx.y > 0) return;
  
  if (i < dim) {
    v[i] = exp(v[i]);
  } 
}


template<typename Real>
__global__
static void _vec_apply_log(Real* v, Real* flag, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  //  if (blockIdx.y > 0) return;
  
  if (i < dim) {
    if (v[i] < 0) {
      *flag = 1;
      return;
    }
    v[i] = log(v[i]);
  }
}


template<typename Real>
__global__
static void _vec_sum(Real *v, Real *sum, int dim, int inc) {
  int i = threadIdx.x;
  __shared__ Real row_data[CU1DBLOCK];  

  if (i >= CU1DBLOCK) return;
  
  Real tmp_sum = 0;
  int size = dim / CU1DBLOCK; //the least size in a loop (later part)
  int threshold = dim - size * CU1DBLOCK; //any loop below this number would + 1

  int loop_start;
  int loop_end;
  if(i < threshold) {
    loop_start = i * (size + 1);
    loop_end = (i+1) * (size + 1);
  }
  else {
    loop_start = threshold + i * size;
    loop_end = threshold + (i+1) * size;
  }
  for(int j = loop_start; j< loop_end; j++) {
    tmp_sum += v[j * inc];
  }
 
  row_data[threadIdx.x] = tmp_sum;
  __syncthreads();
  *sum = _sum_reduce(row_data);
}


template<typename Real>
__global__
static void _pvec_sum(Real* v, Real* g, int dim, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int start = size * i;
  if (start >= dim) return;
  int end = start + size;
  if (end > dim) end = dim;
  __shared__ Real row_data[CU1DBLOCK];
  Real sum = 0;
  for (int j = start; j < end; j++)
    sum += v[j];
  row_data[threadIdx.x] = sum;
  __syncthreads();
  g[blockIdx.x] = _sum_reduce(row_data);
}



template<typename Real>
__global__
static void _vec_apply_floor(Real *v, Real floor_val, float *count, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if ( i < dim) {
    if ( v[i] < floor_val) {
      v[i] = floor_val;
      count[i] = 1;
    } else {
      count[i] = 0;
    }
  }
}


// Caution, here i/block{idx,dim}.x is the row index and j/block{idx,dim}.y is the col index.
// this is for no reason, really, I just happened to prefer this
// at the time. [dan]
template<typename Real>
__global__
static void _apply_heaviside(Real* mat, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i * d.stride + j;

  if (i < d.rows && j < d.cols) {
    mat[index] = (mat[index] > 0.0 ? 1.0 : 0.0);
  }
}


// Caution, here i/block{idx,dim}.x is the row index and j/block{idx,dim}.y is the col index.
// this is for no reason, really, I just happened to prefer this
// at the time. [dan]
template<typename Real>
__global__
static void _apply_pow(Real* mat, Real power, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i * d.stride + j;

  if (i < d.rows && j < d.cols) {
    if (power == 1.0)
      return;
    if (power == 2.0) {
      mat[index] = mat[index] * mat[index];
    } else if (power == 0.5) {
      if (!(mat[index] >= 0.0))
        return;
      mat[index] = sqrt(mat[index]);
    } else {
      mat[index] = pow(mat[index], power);
    }
  }
}

template<typename Real>
__global__
static void _apply_floor(Real* mat, Real floor_val, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j * d.stride;

  if (i < d.cols && j < d.rows) {
    if (mat[index] < floor_val)
      mat[index] = floor_val;
  }
}

template<typename Real>
__global__
static void _apply_ceiling(Real* mat, Real ceiling_val, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j * d.stride;

  if (i < d.cols && j < d.rows ) {
    if (mat[index] > ceiling_val)
      mat[index] = ceiling_val;
  }
}

template<typename Real>
__global__
static void _invert_elements(Real* data, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    data[index] = 1.0 / data[index];
}

template<typename Real>
__global__
static void _sqrt_elements(Real* data, Real epsilon, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    data[index] = sqrt(data[index]+epsilon);
}

template<typename Real>
__global__
static void _add_row_sum_mat(const Real* mat, Real* vec_sum, MatrixDim d) {
  int i = blockIdx.y * blockDim.y + threadIdx.y; //col
  int j = blockIdx.x * blockDim.x + threadIdx.x; //row

  if(blockIdx.x > 0) return;
  if(blockDim.y != 1) return;

  __shared__ Real row_data[CU1DBLOCK];

  //copy the input to row_data
  row_data[j] = mat[i+j*d.stride];
  __syncthreads();

  //get the sum
  Real sum = _sum_reduce(row_data);
  __syncthreads();
  
  //add to previously accumulated sum
  if(threadIdx.x == 0)
    vec_sum[i] += sum;
}


template<typename Real>
__global__
static void _add_col_sum_mat(const Real* mat, Real* vec_sum, MatrixDim d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; //row
  int j = blockIdx.y * blockDim.y + threadIdx.y; //col

  if(blockIdx.x > 0) return;
  if(blockDim.y != 1) return;

  __shared__ Real row_data[CU1DBLOCK];

  //copy the input to row_data
  row_data[i] = mat[i+j*d.stride];
  __syncthreads();

  //get the sum
  Real sum = _sum_reduce(row_data);
  __syncthreads();
  
  //add to previously accumulated sum
  if(threadIdx.x == 0) 
    vec_sum[j] += sum;
}

template<typename Real>
__global__
static void _add_mat_mat_elements(Real *data, const Real *srcA_data,
                                  const Real *srcB_data, MatrixDim dim,
                                  int srcA_stride, int srcB_stride, Real alpha,
                                  Real beta) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda tgt_index = i + j * dim.stride;
  int32_cuda srcA_index = i + j * srcA_stride;
  int32_cuda srcB_index = i + j * srcB_stride;
  if (i < dim.cols && j < dim.rows) {
    data[tgt_index] = alpha * srcA_data[srcA_index] * srcB_data[srcB_index]
        + beta * data[tgt_index];
  }
}

/*
 * cu::
 */
template<typename Real>
__global__
static void _sigmoid(Real*y, const Real*x, MatrixDim d, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j*d.stride, src_index = i + j*src_stride;
  if(i < d.cols && j < d.rows) {
    Real res = 1.0 / (1.0 + exp(-x[src_index]));
    y[dst_index] = res;
  }
}

template<typename Real>
__global__
static void _diff_sigmoid(Real*eout, const Real*e, const Real*y, MatrixDim d, int e_stride, int y_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j*d.stride;
  int e_index = i + j*e_stride;
  int y_index = i + j*y_stride;
  if (i < d.cols  && j < d.rows ) 
    eout[dst_index] = y[y_index]*(1.0-y[y_index]) * e[e_index];
}


template<typename Real>
__global__
static void _tanh(Real*y, const Real*x, MatrixDim d, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j*d.stride, src_index = i + j * src_stride;
  if(i < d.cols && j < d.rows) {
    Real exp_2x = exp(2.0*x[src_index]);
    Real res;
    if(isinf(exp_2x)) {
      res = 1.0;
    } else {
      res = (exp_2x - 1.0) / (exp_2x + 1.0);
    }
    y[dst_index] = res;
  }
}


template<typename Real>
__global__
static void _diff_tanh(Real*eout, const Real*e, const Real*y, MatrixDim d, int e_stride, int y_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j*d.stride; 
  int e_index   = i + j*e_stride; 
  int y_index   = i + j*y_stride;
  if (i < d.cols  && j < d.rows ) 
    eout[dst_index] = (1.0 - y[y_index]*y[y_index]) * e[e_index];
}

template<typename Real>
__global__
static void _softmax_reduce(Real*y, const Real*x, MatrixDim d, int src_stride) {
  int j = blockIdx.x;
  int THREADS = blockDim.x;
  if (j >= d.rows) return;

  __shared__ Real aux[CU1DBLOCK];
  int steps = (d.cols - 1) / THREADS + 1;

  //copy input to aux
  aux[threadIdx.x] = x[threadIdx.x+j*d.stride];
  for(int i=1; i<steps; ++i) {
    if(threadIdx.x+i*THREADS < d.cols && aux[threadIdx.x] < x[threadIdx.x+i*THREADS+j*d.stride])
	aux[threadIdx.x] = x[threadIdx.x+i*THREADS+j*d.stride];
  }

  //get the maximum value
  int nTotalThreads = THREADS;
  __syncthreads();
  while(nTotalThreads > 1) {
    int halfPoint = ((1+nTotalThreads) >> 1);   // divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      if(threadIdx.x+halfPoint < nTotalThreads && aux[threadIdx.x] < aux[threadIdx.x+halfPoint])
        aux[threadIdx.x] = aux[threadIdx.x + halfPoint];
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);   // divide by two.
  }
  Real max = aux[0];
  __syncthreads();
  
   // subtract max, apply exp, sum up...
  y[threadIdx.x+j*d.stride] = exp(x[threadIdx.x+j*d.stride] - max);
  aux[threadIdx.x] = y[threadIdx.x+j*d.stride];
  for(int i=1; i<steps; i++) {
    if(threadIdx.x+i*THREADS < d.cols) {
      y[threadIdx.x+i*THREADS+j*d.stride] = exp(x[threadIdx.x+i*THREADS+j*d.stride] - max);
      aux[threadIdx.x] += y[threadIdx.x+i*THREADS+j*d.stride];
    }
  }
  nTotalThreads = THREADS;
  __syncthreads();
  while(nTotalThreads > 1) {
    int halfPoint = ((1+nTotalThreads) >> 1);   // divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x < halfPoint)  {
      // Get the shared value stored by another thread
      if(threadIdx.x+halfPoint < nTotalThreads)
        aux[threadIdx.x] += aux[threadIdx.x + halfPoint];
    }
    __syncthreads();
    nTotalThreads = ((1+nTotalThreads) >> 1);   // divide by two.
  }
  Real sum = aux[0];
  __syncthreads();

  //normalize by sum...
  for(int i=0; i<steps; i++) {
    if(threadIdx.x+i*THREADS < d.cols) {
      y[threadIdx.x+i*THREADS+j*d.stride] = y[threadIdx.x+i*THREADS+j*d.stride] / sum;
    }
  }

}


template<typename Real>
__global__
static void _splice(Real* y, const Real* x, const int32_cuda* off, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d_out.stride;
  if (i < d_out.cols  && j < d_out.rows ) {
    int32_cuda src_col = i % d_in.cols;
    int32_cuda src_row = j + off[i / d_in.cols];
    if(src_row < 0) src_row = 0;
    if(src_row >= d_in.rows) src_row = d_in.rows-1;
    y[index] = x[src_col + src_row*d_in.stride];
  }
}

template<typename Real>
__global__
static void _copy(Real* y, const Real* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d_out.stride;
  if (i < d_out.cols  && j < d_out.rows ) {
    int32_cuda src_col = copy_from[i];
    if(src_col >= 0 && src_col < d_in.cols) {
      y[index] = x[src_col + j*d_in.stride];
    } else {
      y[index] = 1.0/0.0;
    }
  }
}

template<typename Real>
__global__
static void _randomize(Real* y, const Real* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d_out.stride;
  if (i < d_out.cols  && j < d_out.rows ) {
    int32_cuda src_row = copy_from[j];
    y[index] = x[i + src_row*d_in.stride];
  }
}


template<typename Real>
__global__
static void _regularize_l1(Real* wei, Real* grad, Real l1, Real lr, MatrixDim d, int stride_grad) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j*d.stride,
             grad_index = i + j*stride_grad;
  if (i < d.cols && j < d.rows) {

    if(wei[index]==0.0) return; //skip L1 if zero weight!
    
    Real l1_signed = l1;
    if(wei[index] < 0.0) //flip sign
      l1_signed = -l1;

    Real before = wei[index];
    Real after = wei[index] -lr*grad[grad_index] -l1_signed;//simulate update
    if((after > 0.0) ^ (before > 0.0)) { //sign changed?
      wei[index] = 0.0;
      grad[grad_index] = 0.0;
    } else {
      wei[index] -= l1_signed;
    }
  }
}

template<typename Real>
__global__
static void _find_row_max_id(const Real* mat, Real* vec_val, int32_cuda* vec_id, int32_cuda voff, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;

  if(blockIdx.x > 0) return;
  if(blockDim.y != 1) return;

  __shared__ Real value[CU1DBLOCK];
  __shared__ int32_cuda index[CU1DBLOCK];

  //copy to shared memory
  value[threadIdx.x] = mat[i+j*d.stride];
  index[threadIdx.x] = threadIdx.x;
  __syncthreads();
  
  //get the id of the max value
  int32_cuda out_max = _max_id_reduce(value, index);
  __syncthreads();

  //see if it's bigger value
  if(threadIdx.x == 0) {
    if(vec_val[j] <= mat[out_max+j*d.stride]) {
      vec_val[j] = mat[out_max+j*d.stride];
      vec_id[j]  = voff+out_max;
    }
  }
}



/***********************************************************************
 * ANSI-C wrappers of CUDA kernels
 */

/*
 * "int32" 
 */
void cudaI32_set_const(dim3 Gr, dim3 Bl, int32_cuda* mat, int32_cuda value, MatrixDim d) {
  _set_const<<<Gr,Bl>>>(mat,value,d); 
}


/*
 * CuMatrix
 */
void cudaF_apply_exp(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _apply_exp<<<Gr,Bl>>>(mat,d);
}
void cudaD_apply_exp(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _apply_exp<<<Gr,Bl>>>(mat,d);
}

void cudaF_apply_pow(dim3 Gr, dim3 Bl, float* mat, float power, MatrixDim d) {
  _apply_pow<<<Gr,Bl>>>(mat, power, d);
}
void cudaD_apply_pow(dim3 Gr, dim3 Bl, double* mat, double power, MatrixDim d) {
  _apply_pow<<<Gr,Bl>>>(mat, power, d);
}

void cudaF_apply_floor(dim3 Gr, dim3 Bl, float* mat, float floor_val, MatrixDim d) {
  _apply_floor<<<Gr,Bl>>>(mat, floor_val, d);
}
void cudaD_apply_floor(dim3 Gr, dim3 Bl, double* mat, double floor_val, MatrixDim d) {
  _apply_floor<<<Gr,Bl>>>(mat, floor_val, d);
}

void cudaF_apply_heaviside(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _apply_heaviside<<<Gr,Bl>>>(mat, d);

}
void cudaD_apply_heaviside(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _apply_heaviside<<<Gr,Bl>>>(mat, d);
}

void cudaF_apply_ceiling(dim3 Gr, dim3 Bl, float* mat, float ceiling_val, MatrixDim d) {
  _apply_ceiling<<<Gr,Bl>>>(mat, ceiling_val, d);
}
void cudaD_apply_ceiling(dim3 Gr, dim3 Bl, double* mat, double ceiling_val, MatrixDim d) {
  _apply_ceiling<<<Gr,Bl>>>(mat, ceiling_val, d);
}


void cudaF_set_const(dim3 Gr, dim3 Bl, float* mat, float value, MatrixDim d) {
  _set_const<<<Gr,Bl>>>(mat,value,d); 
}
void cudaD_set_const(dim3 Gr, dim3 Bl, double* mat, double value, MatrixDim d) {
  _set_const<<<Gr,Bl>>>(mat,value,d);
}

void cudaF_add(dim3 Gr, dim3 Bl, float* mat, float value, MatrixDim d) {
  _add<<<Gr,Bl>>>(mat,value,d); 
}
void cudaD_add(dim3 Gr, dim3 Bl, double* mat, double value, MatrixDim d) {
  _add<<<Gr,Bl>>>(mat,value,d);
}

void cudaF_scale(dim3 Gr, dim3 Bl, float* mat, float value, MatrixDim d) {
  _scale<<<Gr,Bl>>>(mat,value,d); 
}
void cudaD_scale(dim3 Gr, dim3 Bl, double* mat, double value, MatrixDim d) {
  _scale<<<Gr,Bl>>>(mat,value,d);
}

void cudaF_apply_log(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) {
  _apply_log<<<Gr,Bl>>>(mat,d); 
}
void cudaD_apply_log(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) {
  _apply_log<<<Gr,Bl>>>(mat,d);
}

void cudaF_mul_elements(dim3 Gr, dim3 Bl, float* mat, const float* A, MatrixDim dst_d, int src_stride) {
  _mul_elements<<<Gr,Bl>>>(mat,A,dst_d,src_stride); 
}
void cudaD_mul_elements(dim3 Gr, dim3 Bl, double* mat, const double* A, MatrixDim dst_d, int src_stride) {
  _mul_elements<<<Gr,Bl>>>(mat,A,dst_d,src_stride);
}

void cudaF_mul_rows_vec(dim3 Gr, dim3 Bl, float* mat, const float* scale, MatrixDim d) {
  _mul_rows_vec<<<Gr,Bl>>>(mat,scale,d);
}
void cudaD_mul_rows_vec(dim3 Gr, dim3 Bl, double* mat, const double* scale, MatrixDim d) {
  _mul_rows_vec<<<Gr,Bl>>>(mat,scale,d);
}

void cudaF_add_mat(dim3 Gr, dim3 Bl, float alpha, const float* src, float* dst, MatrixDim d, int src_stride, int A_trans) {
  if (A_trans) {
    _add_mat_trans<<<Gr,Bl>>>(alpha,src,dst,d,src_stride);  
  } else {
    _add_mat<<<Gr,Bl>>>(alpha,src,dst,d,src_stride);
  }
}
void cudaD_add_mat(dim3 Gr, dim3 Bl, double alpha, const double* src, double* dst, MatrixDim d, int src_stride, int A_trans) {
  if (A_trans) {
    _add_mat_trans<<<Gr,Bl>>>(alpha,src,dst,d,src_stride);
  } else {
    _add_mat<<<Gr,Bl>>>(alpha,src,dst,d,src_stride);
  }
}

void cudaF_add_vec_to_rows(dim3 Gr, dim3 Bl, float alpha, const float* row, float beta, float* dst, MatrixDim d) {
  _add_vec_to_rows<<<Gr,Bl>>>(alpha,row,beta,dst,d); 
}
void cudaD_add_vec_to_rows(dim3 Gr, dim3 Bl, double alpha, const double* row, double beta, double* dst, MatrixDim d) {
  _add_vec_to_rows<<<Gr,Bl>>>(alpha,row,beta,dst,d);
}

void cudaF_add_mat_mat_elements(dim3 Gr, dim3 Bl, float *data, const float *srcA_data, const float *srcB_data,
MatrixDim dim, int srcA_stride, int srcB_stride, float alpha, float beta) {
  _add_mat_mat_elements<<<Gr, Bl>>>(data, srcA_data, srcB_data, dim, srcA_stride, srcB_stride, alpha, beta);
}

void cudaD_add_mat_mat_elements(dim3 Gr, dim3 Bl, double *data, const double *srcA_data, const double *srcB_data,
MatrixDim dim, int srcA_stride, int srcB_stride, double alpha, double beta) {
  _add_mat_mat_elements<<<Gr, Bl>>>(data, srcA_data, srcB_data, dim, srcA_stride, srcB_stride, alpha, beta);
}

/*
 * CuVector
 */

void cudaF_copy_from_vec_df(int Gr, int Bl, double* v_out, const float* v_in, int dim) {
  _copy_from_vec_df<<<Gr,Bl>>>(v_out,v_in,dim);
}
void cudaD_copy_from_vec_df(int Gr, int Bl, double* v_out, const double* v_in, int dim) {
  _copy_from_vec_df<<<Gr,Bl>>>(v_out,v_in,dim);
}

void cudaF_copy_from_vec_fd(int Gr, int Bl, float* v_out, const float* v_in, int dim) {
  _copy_from_vec_fd<<<Gr,Bl>>>(v_out,v_in,dim);
}
void cudaD_copy_from_vec_fd(int Gr, int Bl, float* v_out, const double* v_in, int dim) {
  _copy_from_vec_fd<<<Gr,Bl>>>(v_out,v_in,dim);
}

void cudaF_vec_mul_elements(int Gr, int Bl, float* v, const float* a, int dim) {
  _vec_mul_elements<<<Gr,Bl>>>(v, a, dim);
}
void cudaD_vec_mul_elements(int Gr, int Bl, double* v, const double* a, int dim) {
  _vec_mul_elements<<<Gr,Bl>>>(v, a, dim);
}

void cudaF_vec_min(const float* v, float* value, int dim) {
  _vec_min<<<1,CU1DBLOCK>>>(v, value, dim);
}
void cudaD_vec_min(const double* v, double* value, int dim) {
  _vec_min<<<1,CU1DBLOCK>>>(v, value, dim);
}

void cudaF_vec_max(const float* v, float* value, int dim) {
  _vec_max<<<1,CU1DBLOCK>>>(v, value, dim);
}
void cudaD_vec_max(const double* v, double* value, int dim) {
  _vec_max<<<1,CU1DBLOCK>>>(v, value, dim);
}

void cudaF_add_diag_mat_mat(int Gr, int Bl, float alpha, float* v, int v_dim, const float* M, 
     int M_cols, int M_row_stride, int M_col_stride, const float *N, int N_row_stride, 
                            int N_col_stride, int threads_per_element, float beta) {
   _add_diag_mat_mat<<<Gr,Bl>>>(alpha, v, v_dim, M, M_cols, M_row_stride, M_col_stride,
                                N, N_row_stride, N_col_stride, threads_per_element, beta);
}
void cudaD_add_diag_mat_mat(int Gr, int Bl, double alpha, double* v, int v_dim, const double* M,
     int M_cols, int M_row_stride, int M_col_stride, const double *N, int N_row_stride,
     int N_col_stride, int threads_per_element, double beta) {
   _add_diag_mat_mat<<<Gr,Bl>>>(alpha, v, v_dim, M, M_cols, M_row_stride, M_col_stride,
                                N, N_row_stride, N_col_stride, threads_per_element, beta);
}

void cudaF_add_vec_vec(int Gr, int Bl, float alpha, float* v, const float* x, const float* y, float beta, int dim) {
  _add_vec_vec<<<Gr,Bl>>>(alpha,v,x,y,beta,dim);
}
void cudaD_add_vec_vec(int Gr, int Bl, double alpha, double* v, const double* x, const double* y, double beta, int dim) {
  _add_vec_vec<<<Gr,Bl>>>(alpha,v,x,y,beta,dim);
}

void cudaF_vec_sum(int Gr, int Bl, float* v, float* value, int dim, int inc) {
  _vec_sum<<<Gr,Bl>>>(v, value, dim, inc);
}
void cudaD_vec_sum(int Gr, int Bl, double* v, double* value, int dim, int inc) {
  _vec_sum<<<Gr,Bl>>>(v,value,dim,inc);
}

void cudaF_pvec_sum(int Gr, int Bl, float* v, float* pvec_sum, int dim, int size) {
  _pvec_sum<<<Gr,Bl>>>(v, pvec_sum, dim, size);
}
void cudaD_pvec_sum(int Gr, int Bl, double* v, double* pvec_sum, int dim, int size) {
  _pvec_sum<<<Gr,Bl>>>(v,pvec_sum,dim,size);
}

void cudaF_vec_apply_floor(int Gr, int Bl, float* v, float floor_val, float *count, int dim) {
  _vec_apply_floor<<<Gr,Bl>>>(v,floor_val,count,dim);
}
void cudaD_vec_apply_floor(int Gr, int Bl, double* v, double floor_val, float *count, int dim) {
  _vec_apply_floor<<<Gr,Bl>>>(v,floor_val,count,dim);
}

void cudaF_vec_apply_exp(int Gr, int Bl, float* v, int dim) {
  _vec_apply_exp<<<Gr,Bl>>>(v,dim);
}
void cudaD_vec_apply_exp(int Gr, int Bl, double* v, int dim) {
  _vec_apply_exp<<<Gr,Bl>>>(v,dim);
}

void cudaF_sqrt_elements(dim3 Gr, dim3 Bl, float* data, float epsilon, MatrixDim d) {
    _sqrt_elements<<<Gr,Bl>>>(data, epsilon, d);
}
void cudaD_sqrt_elements(dim3 Gr, dim3 Bl, double* data, double epsilon, MatrixDim d) {
    _sqrt_elements<<<Gr,Bl>>>(data, epsilon, d);
}

void cudaF_invert_elements(dim3 Gr, dim3 Bl, float* data, MatrixDim d) {
    _invert_elements<<<Gr,Bl>>>(data, d);
}
void cudaD_invert_elements(dim3 Gr, dim3 Bl, double* data, MatrixDim d) {
    _invert_elements<<<Gr,Bl>>>(data, d);
}

void cudaF_vec_apply_log(int Gr, int Bl, float* v, float* flag, int dim) {
  _vec_apply_log<<<Gr,Bl>>>(v,flag,dim);
}
void cudaD_vec_apply_log(int Gr, int Bl, double* v, double* flag, int dim) {
  _vec_apply_log<<<Gr,Bl>>>(v,flag,dim);
}

void cudaF_add_row_sum_mat(dim3 Gr, dim3 Bl, const float* mat, float* vec_sum, MatrixDim d) {
  _add_row_sum_mat<<<Gr,Bl>>>(mat,vec_sum,d);
}
void cudaD_add_row_sum_mat(dim3 Gr, dim3 Bl, const double* mat, double* vec_sum, MatrixDim d) {
  _add_row_sum_mat<<<Gr,Bl>>>(mat,vec_sum,d);
}

void cudaF_add_col_sum_mat(dim3 Gr, dim3 Bl, const float* mat, float* vec_sum, MatrixDim d) {
  _add_col_sum_mat<<<Gr,Bl>>>(mat,vec_sum,d);
}
void cudaD_add_col_sum_mat(dim3 Gr, dim3 Bl, const double* mat, double* vec_sum, MatrixDim d) {
  _add_col_sum_mat<<<Gr,Bl>>>(mat,vec_sum,d);
}

/*
 * cu::
 */
void cudaF_sigmoid (dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d, int src_stride) {
  _sigmoid<<<Gr,Bl>>>(y, x, d, src_stride); 
}
void cudaD_sigmoid (dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d, int src_stride) {
  _sigmoid<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_diff_sigmoid (dim3 Gr, dim3 Bl, float* eout, const float* e, const float* y, MatrixDim d, int e_stride, int y_stride) {
  _diff_sigmoid<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride);
}
void cudaD_diff_sigmoid (dim3 Gr, dim3 Bl, double* eout, const double* e, const double* y, MatrixDim d, int e_stride, int y_stride) {
  _diff_sigmoid<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride);
}

void cudaF_tanh (dim3 Gr, dim3 Bl, float* y, const float* x, MatrixDim d, int src_stride) {
  _tanh<<<Gr,Bl>>>(y, x, d, src_stride); 
}
void cudaD_tanh (dim3 Gr, dim3 Bl, double* y, const double* x, MatrixDim d, int src_stride) {
  _tanh<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_diff_tanh (dim3 Gr, dim3 Bl, float* eout, const float* e, const float* y, MatrixDim d, int e_stride, int y_stride) {
  _diff_tanh<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride);
}
void cudaD_diff_tanh (dim3 Gr, dim3 Bl, double* eout, const double* e, const double* y, MatrixDim d, int e_stride, int y_stride) {
  _diff_tanh<<<Gr,Bl>>>(eout, e, y, d, e_stride, y_stride);
}

void cudaF_softmax_reduce (size_t Gr, size_t Bl, float* y, const float* x, MatrixDim d, int src_stride) {
  _softmax_reduce<<<Gr,Bl>>>(y, x, d, src_stride);
}
void cudaD_softmax_reduce (size_t Gr, size_t Bl, double* y, const double* x, MatrixDim d, int src_stride) {
  _softmax_reduce<<<Gr,Bl>>>(y, x, d, src_stride);
}

void cudaF_splice(dim3 Gr, dim3 Bl, float* y, const float* x, const int32_cuda* off, MatrixDim d_out, MatrixDim d_in) {
  _splice<<<Gr,Bl>>>(y,x,off,d_out,d_in); 
}
void cudaD_splice(dim3 Gr, dim3 Bl, double* y, const double* x, const int32_cuda* off, MatrixDim d_out, MatrixDim d_in) {
  _splice<<<Gr,Bl>>>(y,x,off,d_out,d_in);
}

void cudaF_copy(dim3 Gr, dim3 Bl, float* y, const float* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  _copy<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in); 
}
void cudaD_copy(dim3 Gr, dim3 Bl, double* y, const double* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) {
  _copy<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in);
}
 
void cudaF_randomize(dim3 Gr, dim3 Bl, float* y, const float* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) { 
  _randomize<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in); 
}
void cudaD_randomize(dim3 Gr, dim3 Bl, double* y, const double* x, const int32_cuda* copy_from, MatrixDim d_out, MatrixDim d_in) { 
  _randomize<<<Gr,Bl>>>(y,x,copy_from,d_out,d_in);
}

void cudaF_regularize_l1(dim3 Gr, dim3 Bl, float* wei, float* grad, float l1, float lr, MatrixDim d, int stride_grad) {
  _regularize_l1<<<Gr,Bl>>>(wei,grad,l1,lr,d,stride_grad); 
}
void cudaD_regularize_l1(dim3 Gr, dim3 Bl, double* wei, double* grad, double l1, double lr, MatrixDim d,int stride_grad) {
  _regularize_l1<<<Gr,Bl>>>(wei,grad,l1,lr,d,stride_grad);
}

void cudaF_find_row_max_id(dim3 Gr, dim3 Bl, const float* mat, float* vec_val, int32_cuda* vec_id, int32_cuda voff, MatrixDim d) {
  _find_row_max_id<<<Gr,Bl>>>(mat, vec_val, vec_id, voff, d);
}
void cudaD_find_row_max_id(dim3 Gr, dim3 Bl, const double* mat, double* vec_val, int32_cuda* vec_id, int32_cuda voff, MatrixDim d) {
  _find_row_max_id<<<Gr,Bl>>>(mat, vec_val, vec_id, voff, d);
}

/* Some conversion kernels for which it's more convenient to not name them F or D. */

void cuda_copy_from_mat_df(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_ff(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_fd(dim3 Gr, dim3 Bl, float *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_dd(dim3 Gr, dim3 Bl, double *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_df_trans(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_trans<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_ff_trans(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_trans<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_fd_trans(dim3 Gr, dim3 Bl, float *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_trans<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}

void cuda_copy_from_mat_dd_trans(dim3 Gr, dim3 Bl, double *mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) {
  _copy_from_mat_trans<<<Gr,Bl>>>(mat_out,mat_in,d_out,d_in);
}


/*
 * lstm::
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

/*
 * All the following kernels are written by Yajie Miao for CTC training
 */
template<typename Real>
__global__
static void _compute_ctc_alpha_one_sequence(Real* mat_alpha, int row, MatrixDim dim_alpha, const Real* mat_prob, MatrixDim dim_prob, const int32_cuda* labels) {

  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda dim = dim_alpha.cols;

  if (i < dim) {

  int32_cuda index_alpha = i + row * dim_alpha.stride;
  int32_cuda class_idx = labels[i];
  int32_cuda index_prob = class_idx + row * dim_prob.stride;

  int32_cuda index_alpha_rm1_i = i + (row - 1) * dim_alpha.stride;
  int32_cuda index_alpha_rm1_im1 = (i - 1) + (row - 1) * dim_alpha.stride;
  int32_cuda index_alpha_rm1_im2 = (i - 2) + (row - 1) * dim_alpha.stride;

  if (row == 0) {
    if (i < 2) mat_alpha[index_alpha] = mat_prob[index_prob];
    else mat_alpha[index_alpha] = NumericLimits<Real>::log_zero_;
  } else {
    if (i > 1) {
      if (i % 2 == 0 || labels[i-2] == labels[i]) {
        mat_alpha[index_alpha] = AddAB(mat_prob[index_prob], LogAPlusB(mat_alpha[index_alpha_rm1_im1], mat_alpha[index_alpha_rm1_i]));
      } else {
        Real tmp = LogAPlusB(mat_alpha[index_alpha_rm1_im1], mat_alpha[index_alpha_rm1_i]);
        mat_alpha[index_alpha] = AddAB(mat_prob[index_prob], LogAPlusB(mat_alpha[index_alpha_rm1_im2], tmp));
      }
    } else if (i == 1) {
      mat_alpha[index_alpha] = AddAB(mat_prob[index_prob], LogAPlusB(mat_alpha[index_alpha_rm1_im1], mat_alpha[index_alpha_rm1_i]));
    } else {
      mat_alpha[index_alpha] = AddAB(mat_prob[index_prob], mat_alpha[index_alpha_rm1_i]);
    }
  }
 }
}

template<typename Real>
__global__
static void _compute_ctc_alpha_multiple_sequence(Real* mat_alpha, int sequence_num, int row, MatrixDim dim_alpha, const Real* mat_prob, MatrixDim dim_prob, const int32_cuda* labels, int32_cuda dim_label_stride, const int32_cuda* seq_lengths) {

  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;  // row index, that is, the index for sequence
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;  // label index, cannot exceed 2*|l|+1
  int32_cuda dim = dim_alpha.cols;

  if (j >= dim || i >= sequence_num) return;
   
  int32_cuda index_alpha = j + (row * sequence_num + i) * dim_alpha.stride;
  int32_cuda index_label = j + i * dim_label_stride;
  int32_cuda class_idx = labels[index_label];// if -1, this is the padding cell;labels now is a matrix which has the same size as mat_alpha
  if (class_idx == -1 || row >= seq_lengths[i]) {
    mat_alpha[index_alpha] = NumericLimits<Real>::log_zero_;
    return;
  }
  int32_cuda index_label_m2 = (j-2) + i * dim_label_stride;
  int32_cuda index_prob = class_idx + (row * sequence_num + i) * dim_prob.stride;

  int32_cuda index_alpha_rm1_i = j + ((row-1) * sequence_num + i) * dim_alpha.stride;
  int32_cuda index_alpha_rm1_im1 = (j-1) + ((row-1) * sequence_num + i) * dim_alpha.stride;
  int32_cuda index_alpha_rm1_im2 = (j-2) + ((row-1) * sequence_num + i) * dim_alpha.stride;

  if (row == 0) {
    if (j < 2) mat_alpha[index_alpha] = mat_prob[index_prob];
    else mat_alpha[index_alpha] = NumericLimits<Real>::log_zero_;
  } else {
    if (j > 1) {
      if (j % 2 == 0 || labels[index_label_m2] == labels[index_label]) {
        mat_alpha[index_alpha] = AddAB(mat_prob[index_prob], LogAPlusB(mat_alpha[index_alpha_rm1_im1], mat_alpha[index_alpha_rm1_i]));
      } else {
        Real tmp = LogAPlusB(mat_alpha[index_alpha_rm1_im1], mat_alpha[index_alpha_rm1_i]);
        mat_alpha[index_alpha] = AddAB(mat_prob[index_prob], LogAPlusB(mat_alpha[index_alpha_rm1_im2], tmp));
      }
    } else if (j == 1) {
      mat_alpha[index_alpha] = AddAB(mat_prob[index_prob], LogAPlusB(mat_alpha[index_alpha_rm1_im1], mat_alpha[index_alpha_rm1_i]));
    } else {
      mat_alpha[index_alpha] = AddAB(mat_prob[index_prob], mat_alpha[index_alpha_rm1_i]);
    }
  }
}

template<typename Real>
__global__
static void _compute_ctc_alpha_one_sequence_rescale(Real* mat_alpha, int row, MatrixDim dim_alpha, const Real* mat_prob, MatrixDim dim_prob, const int32_cuda* labels) {

  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda dim = dim_alpha.cols;

  if (i < dim) {

  int32_cuda index_alpha = i + row * dim_alpha.stride;
  int32_cuda class_idx = labels[i];
  int32_cuda index_prob = class_idx + row * dim_prob.stride;

  int32_cuda index_alpha_rm1_i = i + (row - 1) * dim_alpha.stride;
  int32_cuda index_alpha_rm1_im1 = (i - 1) + (row - 1) * dim_alpha.stride;
  int32_cuda index_alpha_rm1_im2 = (i - 2) + (row - 1) * dim_alpha.stride;

  if (row == 0) {
    if (i < 2) mat_alpha[index_alpha] = mat_prob[index_prob];
    else mat_alpha[index_alpha] = 0.0;
  } else {
    if (i > 1) {
      if (i % 2 == 0 || labels[i-2] == labels[i]) {
        mat_alpha[index_alpha] = mat_prob[index_prob] * (mat_alpha[index_alpha_rm1_im1] + mat_alpha[index_alpha_rm1_i]);
      } else {
        mat_alpha[index_alpha] = mat_prob[index_prob] * (mat_alpha[index_alpha_rm1_im1] + mat_alpha[index_alpha_rm1_i] + mat_alpha[index_alpha_rm1_im2]);
      }
    } else if (i == 1) {
      mat_alpha[index_alpha] = mat_prob[index_prob] * (mat_alpha[index_alpha_rm1_im1] + mat_alpha[index_alpha_rm1_i]);
    } else {
      mat_alpha[index_alpha] = mat_prob[index_prob] * mat_alpha[index_alpha_rm1_i];
    }
  }
 }
}

template<typename Real>
__global__
static void _compute_ctc_beta_one_sequence(Real* mat_beta, int row, MatrixDim dim_beta, const Real* mat_prob, MatrixDim dim_prob, const int32_cuda* labels) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda dim = dim_beta.cols;
  if (i < dim) {

  int32_cuda index_beta = i + row * dim_beta.stride;
  int32_cuda class_idx = labels[i];
  int32_cuda index_prob = class_idx + row * dim_prob.stride;

  int32_cuda index_beta_rp1_i = i + (row + 1) * dim_beta.stride;
  int32_cuda index_beta_rp1_ip1 = (i + 1) + (row + 1) * dim_beta.stride;
  int32_cuda index_beta_rp1_ip2 = (i + 2) + (row + 1) * dim_beta.stride;

  int32_cuda row_num = dim_beta.rows;
  if (row == row_num - 1) {
    if (i > dim - 3) mat_beta[index_beta] = mat_prob[index_prob];
    else mat_beta[index_beta] = NumericLimits<Real>::log_zero_;
  } else {
   if (i < dim - 2) {
     if (i % 2 == 0 || labels[i+2] == labels[i]) {
       mat_beta[index_beta] = AddAB(mat_prob[index_prob], LogAPlusB(mat_beta[index_beta_rp1_ip1], mat_beta[index_beta_rp1_i]));
     } else {
       Real tmp = LogAPlusB(mat_beta[index_beta_rp1_ip1], mat_beta[index_beta_rp1_i]);
       mat_beta[index_beta] = AddAB(mat_prob[index_prob], LogAPlusB(mat_beta[index_beta_rp1_ip2], tmp));
     }
   } else if (i == dim - 2) {
     mat_beta[index_beta] = AddAB(mat_prob[index_prob], LogAPlusB(mat_beta[index_beta_rp1_ip1], mat_beta[index_beta_rp1_i]));
   } else {
     mat_beta[index_beta] = AddAB(mat_prob[index_prob], mat_beta[index_beta_rp1_i]);
   }
  }
 }
}

template<typename Real>
__global__
static void _compute_ctc_beta_multiple_sequence(Real* mat_beta, int sequence_num, int row, MatrixDim dim_beta, const Real* mat_prob, MatrixDim dim_prob, const int32_cuda* labels, int32_cuda dim_label_stride, const int32_cuda* seq_lengths, const int32_cuda* label_lengths) {

  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;  // row index, that is, the index for sequence
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;  // label index, cannot exceed 2*|l|+1
  int32_cuda dim = dim_beta.cols;

  if (j >= dim || i >= sequence_num) return;

  int32_cuda index_beta = j + (row * sequence_num + i) * dim_beta.stride;
  int32_cuda index_label = j + i * dim_label_stride;
  int32_cuda class_idx = labels[index_label];// if -1, this is the padding cell;labels now is a matrix which has the same size as mat_alpha
  if (class_idx == -1 || row >= seq_lengths[i]) {
    mat_beta[index_beta] = NumericLimits<Real>::log_zero_;
    return;
  }
  int32_cuda index_label_p2 = (j+2) + i * dim_label_stride;
  int32_cuda index_prob = class_idx + (row * sequence_num + i) * dim_prob.stride;

  int32_cuda index_beta_rp1_i = j + ((row+1) * sequence_num + i) * dim_beta.stride;
  int32_cuda index_beta_rp1_ip1 = (j+1) + ((row+1) * sequence_num + i) * dim_beta.stride;
  int32_cuda index_beta_rp1_ip2 = (j+2) + ((row+1) * sequence_num + i) * dim_beta.stride;
  
  int32_cuda row_num = seq_lengths[i];
  int32_cuda label_len = label_lengths[i];

/*  if (row == row_num - 1) {
    if (j > dim - 3) mat_beta[index_beta] = mat_prob[index_prob];
    else mat_beta[index_beta] = NumericLimits<Real>::log_zero_;
  } else {
    if (j < dim - 2) {
      if (j % 2 == 0 || labels[index_label_p2] == labels[index_label]) {
        mat_beta[index_beta] = AddAB(mat_prob[index_prob], LogAPlusB(mat_beta[index_beta_rp1_ip1], mat_beta[index_beta_rp1_i]));
      } else {
        Real tmp = LogAPlusB(mat_beta[index_beta_rp1_ip1], mat_beta[index_beta_rp1_i]);
        mat_beta[index_beta] = AddAB(mat_prob[index_prob], LogAPlusB(mat_beta[index_beta_rp1_ip2], tmp));
      }
    } else if (j == dim - 2) {
      mat_beta[index_beta] = AddAB(mat_prob[index_prob], LogAPlusB(mat_beta[index_beta_rp1_ip1], mat_beta[index_beta_rp1_i]));
    } else {
      mat_beta[index_beta] = AddAB(mat_prob[index_prob], mat_beta[index_beta_rp1_i]);
    }
  }
*/
  if (row == row_num - 1) {
    if (j > label_len - 3) mat_beta[index_beta] = mat_prob[index_prob];
    else mat_beta[index_beta] = NumericLimits<Real>::log_zero_;
  } else {
    if (j < label_len - 2) {
      if (j % 2 == 0 || labels[index_label_p2] == labels[index_label]) {
        mat_beta[index_beta] = AddAB(mat_prob[index_prob], LogAPlusB(mat_beta[index_beta_rp1_ip1], mat_beta[index_beta_rp1_i]));
      } else {
        Real tmp = LogAPlusB(mat_beta[index_beta_rp1_ip1], mat_beta[index_beta_rp1_i]);
        mat_beta[index_beta] = AddAB(mat_prob[index_prob], LogAPlusB(mat_beta[index_beta_rp1_ip2], tmp));
      }
    } else if (j == label_len - 2) {
      mat_beta[index_beta] = AddAB(mat_prob[index_prob], LogAPlusB(mat_beta[index_beta_rp1_ip1], mat_beta[index_beta_rp1_i]));
    } else {
      mat_beta[index_beta] = AddAB(mat_prob[index_prob], mat_beta[index_beta_rp1_i]);
    }
  }
}

template<typename Real>
__global__
static void _compute_ctc_beta_one_sequence_rescale(Real* mat_beta, int row, MatrixDim dim_beta, const Real* mat_prob, MatrixDim dim_prob, const int32_cuda* labels) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda dim = dim_beta.cols;
  if (i < dim) {

  int32_cuda index_beta = i + row * dim_beta.stride;
  int32_cuda class_idx = labels[i];
  int32_cuda index_prob = class_idx + row * dim_prob.stride;

  int32_cuda index_beta_rp1_i = i + (row + 1) * dim_beta.stride;
  int32_cuda index_beta_rp1_ip1 = (i + 1) + (row + 1) * dim_beta.stride;
  int32_cuda index_beta_rp1_ip2 = (i + 2) + (row + 1) * dim_beta.stride;

  int32_cuda row_num = dim_beta.rows;
  if (row == row_num - 1) {
    if (i > dim - 3) mat_beta[index_beta] = mat_prob[index_prob];
    else mat_beta[index_beta] = 0;
  } else {
   if (i < dim - 2) {
     if (i % 2 == 0 || labels[i+2] == labels[i]) {
       mat_beta[index_beta] = mat_prob[index_prob] * (mat_beta[index_beta_rp1_ip1] + mat_beta[index_beta_rp1_i]);
     } else {
       mat_beta[index_beta] = mat_prob[index_prob] * (mat_beta[index_beta_rp1_ip1] + mat_beta[index_beta_rp1_i] + mat_beta[index_beta_rp1_ip2]);
     }
   } else if (i == dim - 2) {
     mat_beta[index_beta] = mat_prob[index_prob] * (mat_beta[index_beta_rp1_ip1] + mat_beta[index_beta_rp1_i]);
   } else {
     mat_beta[index_beta] = mat_prob[index_prob] * mat_beta[index_beta_rp1_i];
   }
  }
 }
}

// mat_prob are in probability scale.
template<typename Real>
__global__
static void _compute_ctc_error_one_sequence(Real* mat_error, MatrixDim dim_error, const Real* mat_alpha, const Real* mat_beta, MatrixDim dim_alpha, const Real* mat_prob, const int32_cuda* labels, Real pzx) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; // row index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; // column index
  if (i < dim_error.rows && j < dim_error.cols) {

    Real err = NumericLimits<Real>::log_zero_;
    int32_cuda index_error = i * dim_error.stride + j;
    for(int s = 0; s < dim_alpha.cols; s++) {
      if (labels[s] == j) {  //
        int32_cuda index_alpha = i * dim_alpha.stride + s;
        err = LogAPlusB(err, AddAB(mat_alpha[index_alpha], mat_beta[index_alpha]));
      }
    }
    Real val = ExpA(SubAB(err, AddAB(pzx, mat_prob[index_error] == 0? NumericLimits<Real>::log_zero_ : 2*log(mat_prob[index_error]))));
    mat_error[index_error] = -1.0 * val;
  }
}

// mat_prob are in probability scale.
template<typename Real>
__global__
static void _compute_ctc_error_multiple_sequence(Real* mat_error, int32_cuda sequence_num, MatrixDim dim_error, const Real* mat_alpha, const Real* mat_beta, MatrixDim dim_alpha, const Real* mat_prob, const int32_cuda* labels, int32_cuda dim_label_stride, const int32_cuda* seq_lengths, const Real* pzx) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; // row index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; // column index
  if (i >= dim_error.rows || j >= dim_error.cols) return;

  int32_cuda seqX = i % sequence_num;
  int32_cuda rowX = i / sequence_num;

  if (rowX >= seq_lengths[seqX]) return;

  Real err = NumericLimits<Real>::log_zero_;
  int32_cuda index_error = i * dim_error.stride + j;
  for(int s = 0; s < dim_alpha.cols; s++) {
    int32_cuda index_label = s + seqX * dim_label_stride;
    if (labels[index_label] == -1) {continue;}
    if (labels[index_label] == j) {  //
      int32_cuda index_alpha = i * dim_alpha.stride + s;
      err = LogAPlusB(err, AddAB(mat_alpha[index_alpha], mat_beta[index_alpha]));
    }
  }
  Real val = ExpA(SubAB(err, AddAB(pzx[seqX], mat_prob[index_error] == 0? NumericLimits<Real>::log_zero_ : 2*log(mat_prob[index_error]))));
  mat_error[index_error] = -1.0 * val;
}

template<typename Real>
__global__
static void _distribute_prob_by_label(Real* mat_prob_dist, MatrixDim dim_prob_dist, const Real* mat_prob, MatrixDim dim_prob, const int32_cuda* labels) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; // row index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; // column index
  if (i < dim_prob_dist.rows && j < dim_prob_dist.cols) {
    int32_cuda index_prob_dist = i * dim_prob_dist.stride + j;
    int32_cuda index_prob = i * dim_prob.stride + labels[j];
    mat_prob_dist[index_prob_dist] = mat_prob[index_prob];
  }
}

// directly get the errors for the prior-softmax values
template<typename Real>
__global__
static void _compute_ctc_error_one_sequence_rescale(Real* mat_error, MatrixDim dim_error, const Real* mat_alpha, const Real* mat_beta, MatrixDim dim_alpha, const Real* mat_prob, const int32_cuda* labels, const Real* zt) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x; // row index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y; // column index
  if (i < dim_error.rows && j < dim_error.cols) {

    Real err = 0;
    int32_cuda index_error = i * dim_error.stride + j;
    for(int s = 0; s < dim_alpha.cols; s++) {
      if (labels[s] == j) {  //
        int32_cuda index_alpha = i * dim_alpha.stride + s;
        err += mat_alpha[index_alpha] * mat_beta[index_alpha];
      }
    }
    if (mat_prob[index_error] == 0 || zt[i] == 0) {
        mat_error[index_error] = 0;
    } else {
       mat_error[index_error] = mat_prob[index_error] - (err / zt[i]) / mat_prob[index_error];
    }
  }
}

void cudaF_compute_ctc_alpha(dim3 Gr, dim3 Bl, float *alpha, int row_idx, MatrixDim dim_alpha, const float *prob, MatrixDim dim_prob, const int *labels) {
  _compute_ctc_alpha_one_sequence<<<Gr, Bl>>>(alpha, row_idx, dim_alpha, prob, dim_prob, labels);
}
void cudaF_compute_ctc_beta(dim3 Gr, dim3 Bl, float *beta, int row_idx, MatrixDim dim_beta, const float *prob, MatrixDim dim_prob, const int *labels) {
  _compute_ctc_beta_one_sequence<<<Gr, Bl>>>(beta, row_idx, dim_beta, prob, dim_prob, labels);
}
void cudaF_compute_ctc_error(dim3 Gr, dim3 Bl, float *error, MatrixDim dim_error, const float *alpha, const float *beta, MatrixDim dim_alpha, const float *prob, const int *labels, float pzx) {
  _compute_ctc_error_one_sequence<<<Gr, Bl>>>(error, dim_error, alpha, beta, dim_alpha, prob, labels, pzx);
}

void cudaF_compute_ctc_alpha_rescale(dim3 Gr, dim3 Bl, float *alpha, int row_idx, MatrixDim dim_alpha, const float *prob, MatrixDim dim_prob, const int *labels) {
  _compute_ctc_alpha_one_sequence_rescale<<<Gr, Bl>>>(alpha, row_idx, dim_alpha, prob, dim_prob, labels);
}
void cudaF_compute_ctc_beta_rescale(dim3 Gr, dim3 Bl, float *beta, int row_idx, MatrixDim dim_beta, const float *prob, MatrixDim dim_prob, const int *labels) {
  _compute_ctc_beta_one_sequence_rescale<<<Gr, Bl>>>(beta, row_idx, dim_beta, prob, dim_prob, labels);
}
void cudaF_compute_ctc_error_rescale(dim3 Gr, dim3 Bl, float *error, MatrixDim dim_error, const float *alpha, const float *beta, MatrixDim dim_alpha, const float *prob, const int *labels, const float *zt) {
  _compute_ctc_error_one_sequence_rescale<<<Gr, Bl>>>(error, dim_error, alpha, beta, dim_alpha, prob, labels, zt);
}
void cudaF_distribute_prob_by_label(dim3 Gr, dim3 Bl, float *prob_dist, MatrixDim dim_prob_dist, const float *prob, MatrixDim dim_prob, const int *labels) {
  _distribute_prob_by_label<<<Gr, Bl>>>(prob_dist, dim_prob_dist, prob, dim_prob, labels);
}
void cudaF_compute_ctc_alpha_multiple_sequence(dim3 Gr, dim3 Bl, float *alpha, int seq_num, int row_idx, MatrixDim dim_alpha, const float *prob, MatrixDim dim_prob, const int *labels, int dim_label_stride, const int *seq_lengths) {
  _compute_ctc_alpha_multiple_sequence<<<Gr, Bl>>>(alpha, seq_num, row_idx, dim_alpha, prob, dim_prob, labels, dim_label_stride, seq_lengths);
}
void cudaF_compute_ctc_beta_multiple_sequence(dim3 Gr, dim3 Bl, float *beta, int seq_num, int row_idx, MatrixDim dim_beta, const float *prob, MatrixDim dim_prob, const int *labels, int dim_label_stride, const int *seq_lengths, const int *label_lengths) {
  _compute_ctc_beta_multiple_sequence<<<Gr, Bl>>>(beta, seq_num, row_idx, dim_beta, prob, dim_prob, labels, dim_label_stride, seq_lengths, label_lengths);
}
void cudaF_compute_ctc_error_multiple_sequence(dim3 Gr, dim3 Bl, float *error, int seq_num, MatrixDim dim_error, const float *alpha, const float *beta, MatrixDim dim_alpha, const float *prob, const int *labels, int dim_label_stride, const int *seq_lengths, const float *pzx) {
  _compute_ctc_error_multiple_sequence<<<Gr, Bl>>>(error, seq_num, dim_error, alpha, beta, dim_alpha, prob, labels, dim_label_stride, seq_lengths, pzx);
}


void cudaD_compute_ctc_alpha(dim3 Gr, dim3 Bl, double *alpha, int row_idx, MatrixDim dim_alpha, const double *prob, MatrixDim dim_prob, const int *labels) {
  _compute_ctc_alpha_one_sequence<<<Gr, Bl>>>(alpha, row_idx, dim_alpha, prob, dim_prob, labels);
}
void cudaD_compute_ctc_beta(dim3 Gr, dim3 Bl, double *beta, int row_idx, MatrixDim dim_beta, const double *prob, MatrixDim dim_prob, const int *labels) {
  _compute_ctc_beta_one_sequence<<<Gr, Bl>>>(beta, row_idx, dim_beta, prob, dim_prob, labels);
}
void cudaD_compute_ctc_error(dim3 Gr, dim3 Bl, double *error, MatrixDim dim_error, const double *alpha, const double *beta, MatrixDim dim_alpha, const double *prob, const int *labels, double pzx) {
  _compute_ctc_error_one_sequence<<<Gr, Bl>>>(error, dim_error, alpha, beta, dim_alpha, prob, labels, pzx);
}
void cudaD_compute_ctc_alpha_rescale(dim3 Gr, dim3 Bl, double *alpha, int row_idx, MatrixDim dim_alpha, const double *prob, MatrixDim dim_prob, const int *labels) {
  _compute_ctc_alpha_one_sequence_rescale<<<Gr, Bl>>>(alpha, row_idx, dim_alpha, prob, dim_prob, labels);
}
void cudaD_compute_ctc_beta_rescale(dim3 Gr, dim3 Bl, double *beta, int row_idx, MatrixDim dim_beta, const double *prob, MatrixDim dim_prob, const int *labels) {
  _compute_ctc_beta_one_sequence_rescale<<<Gr, Bl>>>(beta, row_idx, dim_beta, prob, dim_prob, labels);
}
void cudaD_compute_ctc_error_rescale(dim3 Gr, dim3 Bl, double *error, MatrixDim dim_error, const double *alpha, const double *beta, MatrixDim dim_alpha, const double *prob, const int *labels, const double *zt) {
  _compute_ctc_error_one_sequence_rescale<<<Gr, Bl>>>(error, dim_error, alpha, beta, dim_alpha, prob, labels, zt);
}
void cudaD_distribute_prob_by_label(dim3 Gr, dim3 Bl, double *prob_dist, MatrixDim dim_prob_dist, const double *prob, MatrixDim dim_prob, const int *labels) {
  _distribute_prob_by_label<<<Gr, Bl>>>(prob_dist, dim_prob_dist, prob, dim_prob, labels);
}
void cudaD_compute_ctc_alpha_multiple_sequence(dim3 Gr, dim3 Bl, double *alpha, int seq_num, int row_idx, MatrixDim dim_alpha, const double *prob, MatrixDim dim_prob, const int *labels, int dim_label_stride, const int *seq_lengths) {
  _compute_ctc_alpha_multiple_sequence<<<Gr, Bl>>>(alpha, seq_num, row_idx, dim_alpha, prob, dim_prob, labels, dim_label_stride, seq_lengths);
}
void cudaD_compute_ctc_beta_multiple_sequence(dim3 Gr, dim3 Bl, double *beta, int seq_num, int row_idx, MatrixDim dim_beta, const double *prob, MatrixDim dim_prob, const int *labels, int dim_label_stride, const int *seq_lengths, const int *label_lengths) {
  _compute_ctc_beta_multiple_sequence<<<Gr, Bl>>>(beta, seq_num, row_idx, dim_beta, prob, dim_prob, labels, dim_label_stride, seq_lengths, label_lengths);
}
void cudaD_compute_ctc_error_multiple_sequence(dim3 Gr, dim3 Bl, double *error, int seq_num, MatrixDim dim_error, const double *alpha, const double *beta, MatrixDim dim_alpha, const double *prob, const int *labels, int dim_label_stride, const int *seq_lengths, const double *pzx) {
  _compute_ctc_error_multiple_sequence<<<Gr, Bl>>>(error, seq_num, dim_error, alpha, beta, dim_alpha, prob, labels, dim_label_stride, seq_lengths, pzx);
}

