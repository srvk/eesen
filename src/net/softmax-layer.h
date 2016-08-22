// net/softmax-layer.h

// Copyright 2011-2013  Brno University of Technology (author: Karel Vesely)
//                2015  Yajie Miao
//		  2015  Mohammad Gowayyed

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


#ifndef EESEN_SOFTMAX_LAYER_H_
#define EESEN_SOFTMAX_LAYER_H_

#include "net/layer.h"
#include "net/utils-functions.h"
#include "gpucompute/cuda-math.h"
#include "gpucompute/cuda-rand.h"
#include "util/text-utils.h"

namespace eesen {

class Softmax : public Layer {
 public:
  Softmax(int32 dim_in, int32 dim_out)
    : Layer(dim_in, dim_out)
  { }
  ~Softmax()
  { }

  Layer* Copy() const { return new Softmax(*this); }
  LayerType GetType() const { return l_Softmax; }
  LayerType GetTypeNonParal() const { return l_Softmax; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = e^x_j/sum_j(e^x_j)
    out->ApplySoftMaxPerRow(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // simply copy the error derivative
    // (ie. assume crossentropy error function, 
    // while in_diff contains (net_output-target) :
    // this is already derivative of the error with 
    // respect to activations of last layer neurons)
    in_diff->CopyFromMat(out_diff);
  }
  
};

class BlockSoftmax : public Layer {
 public:
 BlockSoftmax(int32 dim_in, int32 dim_out)
   : Layer(dim_in, dim_out)
    { }
  ~BlockSoftmax()
    { }

  Layer* Copy() const { return new BlockSoftmax(*this); }
  LayerType GetType() const { return l_BlockSoftmax; }
  LayerType GetTypeNonParal() const { return l_BlockSoftmax; }

  void InitData(std::istream &is) {
    // parse config
    std::string token,
      dims_str;
    while (!is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<BlockDims>") is >> dims_str;
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (BlockDims)";
      is >> std::ws; // eat-up whitespace
    }
    // parse dims,
    if (!eesen::SplitStringToIntegers(dims_str, ",:", false, &block_dims))
      KALDI_ERR << "Invalid block-dims " << dims_str;
    // sanity check
    int32 sum = 0;
    for (int32 i=0; i<block_dims.size(); i++) {
      sum += block_dims[i];
    }
    KALDI_ASSERT(sum == OutputDim());
  }

  void ReadData(std::istream &is, bool binary) {
    ReadIntegerVector(is, binary, &block_dims);
    block_offset.resize(block_dims.size()+1, 0);
    for (int32 i = 0; i < block_dims.size(); i++) {
      block_offset[i+1] = block_offset[i] + block_dims[i];
    }
    KALDI_ASSERT(OutputDim() == block_offset[block_offset.size()-1]);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteIntegerVector(os, binary, block_dims);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      CuSubMatrix<BaseFloat> in_bl = in.ColRange(block_offset[bl], block_dims[bl]);
      CuSubMatrix<BaseFloat> out_bl = out->ColRange(block_offset[bl], block_dims[bl]);
      // y = e^x_j/sum_j(e^x_j)
      out_bl.ApplySoftMaxPerRow(in_bl);
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // copy the error derivative:
    in_diff->CopyFromMat(out_diff);
  }

  std::string Info() const {
    std::string res = "\n  softmax-dims ";
    for(int i = 0; i < block_dims.size(); i++)
      res += ToString(block_dims[i]) + ":";
    KALDI_LOG << "WRITING INFO " << res;
    return res;
  }

  std::vector<int32> block_dims;
  std::vector<int32> block_offset;
};

} // namespace eesen

#endif

