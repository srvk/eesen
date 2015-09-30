// nnet/nnet-activation.h

// Copyright 2011-2013  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_ACTIVATION_H_
#define KALDI_NNET_NNET_ACTIVATION_H_

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"
#include "util/text-utils.h"

namespace kaldi {
namespace nnet1 {

class Softmax : public Component {
 public:
  Softmax(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Softmax()
  { }

  Component* Copy() const { return new Softmax(*this); }
  ComponentType GetType() const { return kSoftmax; }
  ComponentType GetTypeNonParal() const { return kSoftmax; }

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

class Normalize : public Component {
 public:
  Normalize(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  {kNormFloor = pow(2.0, -66); }
  ~Normalize()
  { }

  Component* Copy() const { return new Normalize(*this); }
  ComponentType GetType() const { return kNormalize; }
  ComponentType GetTypeNonParal() const { return kNormalize; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // This component modifies the vector of activations by scaling it so that the
    // root-mean-square equals 1.0.
    out->CopyFromMat(in);
    CuVector<BaseFloat> in_norm(in.NumRows());
    in_norm.AddDiagMat2(1.0 / in.NumCols(), in, kNoTrans, 0.0);
    in_norm.ApplyFloor(kNormFloor);
    in_norm.ApplyPow(-0.5);
    out->MulRowsVec(in_norm);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // 
    CuVector<BaseFloat> in_norm(in.NumRows());
    in_norm.AddDiagMat2(1.0 / in.NumCols(), in, kNoTrans, 0.0);
    in_norm.ApplyFloor(kNormFloor);
    in_norm.ApplyPow(-0.5);
    in_diff->AddDiagVecMat(1.0, in_norm, out_diff, kNoTrans, 0.0);
    in_norm.ReplaceValue(1.0 / sqrt(kNormFloor), 0.0);
    in_norm.ApplyPow(3.0);
    CuVector<BaseFloat> dot_prod(in_diff->NumRows());
    dot_prod.AddDiagMatMat(1.0, out_diff, kNoTrans, in, kTrans, 0.0);
    dot_prod.MulElements(in_norm);
    in_diff->AddDiagVecMat(-1.0 / in.NumCols(), dot_prod, in, kNoTrans, 1.0);
  }

  BaseFloat kNormFloor;
};

class Sigmoid : public Component {
 public:
  Sigmoid(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Sigmoid()
  { }

  Component* Copy() const { return new Sigmoid(*this); }
  ComponentType GetType() const { return kSigmoid; }
  ComponentType GetTypeNonParal() const { return kSigmoid; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = 1/(1+e^-x)
    out->Sigmoid(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = y(1-y)ex
    in_diff->DiffSigmoid(out, out_diff);
  }
};

 
} // namespace nnet1
} // namespace kaldi

#endif

