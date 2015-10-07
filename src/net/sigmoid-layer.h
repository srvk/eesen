// net/sigmoid-layer.h

// Copyright 2011-2013  Brno University of Technology (author: Karel Vesely)
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


#ifndef EESEN_SIGMOID_LAYER_H_
#define EESEN_SIGMOID_LAYER_H_

#include "net/layer.h"
#include "gpucompute/cuda-math.h"
#include "gpucompute/cuda-rand.h"
#include "util/text-utils.h"

namespace eesen {

class Sigmoid : public Layer {
 public:
  Sigmoid(int32 dim_in, int32 dim_out) 
    : Layer(dim_in, dim_out)
  { }
  ~Sigmoid()
  { }

  Layer* Copy() const { return new Sigmoid(*this); }
  LayerType GetType() const { return l_Sigmoid; }
  LayerType GetTypeNonParal() const { return l_Sigmoid; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    out->Sigmoid(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->DiffSigmoid(out, out_diff);
  }
  
};

} // namespace eesen

#endif

