// net/trainable-layer.h

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)
//                2015  Yajie Miao, Hang Su

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



#ifndef EESEN_TRAINABLE_LAYER_H_
#define EESEN_TRAINABLE_LAYER_H_


#include "base/kaldi-common.h"
#include "cpucompute/matrix-lib.h"
#include "gpucompute/cuda-matrix.h"
#include "gpucompute/cuda-vector.h"
#include "net/train-opts.h"
#include "net/layer.h"

#include <iostream>

namespace eesen {

enum UpdateRule {invalid_update=0, sgd_update=1, adagrad_update=2, rmsprop_update=3};

/**
 * Class TrainableLayer is a Layer which has trainable parameters,
 * contains SGD training hyper-parameters in NetTrainOptions.
 */
class TrainableLayer : public Layer {
 public: 
  TrainableLayer(int32 input_dim, int32 output_dim)
    : Layer(input_dim, output_dim) { }
  virtual ~TrainableLayer() { }

  /// Check if contains trainable parameters 
  bool IsTrainable() const { 
    return true; 
  }

  /// Number of trainable parameters
  virtual int32 NumParams() const = 0;
  virtual void GetParams(Vector<BaseFloat> *params) const = 0;

  /// Compute gradient and update parameters
  virtual void Update(const CuMatrixBase<BaseFloat> &input,
                      const CuMatrixBase<BaseFloat> &diff, 
                      const UpdateRule rule=sgd_update ) = 0;

  /// Compute accu+grad**2 elementwise for matrices
  inline void AdagradAccuUpdate(CuMatrixBase<BaseFloat> &accu, CuMatrixBase<BaseFloat> &grad, 
                                CuMatrixBase<BaseFloat> &grad_tmp) {
    grad_tmp.CopyFromMat(grad);
    grad_tmp.MulElements(grad);
    accu.AddMat(1.0, grad_tmp);
  }

  /// Compute accu+grad**2 elementwise for vectors
  inline void AdagradAccuUpdate(CuVectorBase<BaseFloat> &accu, CuVectorBase<BaseFloat> &grad, 
                                CuVectorBase<BaseFloat> &grad_tmp) {
    grad_tmp.CopyFromVec(grad);
    grad_tmp.MulElements(grad);
    accu.AddVec(1.0, grad_tmp);
  }

  /// Compute accu_new = rho * accu + (1 - rho) * grad ** 2 elementwise for matrices
  inline void RMSPropAccuUpdate(CuMatrixBase<BaseFloat> &accu, CuMatrixBase<BaseFloat> &grad, 
                                CuMatrixBase<BaseFloat> &grad_tmp) {
    grad_tmp.CopyFromMat(grad);
    grad_tmp.MulElements(grad);
    accu.Scale(opts_.rmsprop_rho);
    accu.AddMat(opts_.rmsprop_one_minus_rho, grad_tmp);
  }

  /// Compute accu_new = rho * accu + (1 - rho) * grad ** 2 elementwise for vectors
  inline void RMSPropAccuUpdate(CuVectorBase<BaseFloat> &accu, CuVectorBase<BaseFloat> &grad, 
                                CuVectorBase<BaseFloat> &grad_tmp) {
    grad_tmp.CopyFromVec(grad);
    grad_tmp.MulElements(grad);
    accu.Scale(opts_.rmsprop_rho);
    accu.AddVec(opts_.rmsprop_one_minus_rho, grad_tmp);
  }

  /// calculate 1.0 / sqrt(accu + epsilon) elementwise for matrices
  inline void AdagradScaleCompute(CuMatrixBase<BaseFloat> &accu_scale, CuMatrixBase<BaseFloat> &accu) {
    accu_scale.CopyFromMat(accu);
    //accu_scale.Add(adagrad_epsilon);
    //accu_scale.ApplyPow(0.5);
    accu_scale.ApplySqrt(opts_.adagrad_epsilon);
    accu_scale.InvertElements();
  }

  /// calculate 1.0 / sqrt(accu + epsilon) elementwise for vectors
  inline void AdagradScaleCompute(CuVectorBase<BaseFloat> &accu_scale, CuVectorBase<BaseFloat> &accu) {
    accu_scale.CopyFromVec(accu);
    //accu_scale.Add(adagrad_epsilon);
    //accu_scale.ApplyPow(0.5);
    accu_scale.ApplySqrt(opts_.adagrad_epsilon);
    accu_scale.InvertElements();
  }

  virtual void Scale(BaseFloat scale) = 0;

  virtual void Add(BaseFloat scale, const TrainableLayer & layer_other) = 0;

  /// Sets the training options to the component
  virtual void SetTrainOptions(const NetTrainOptions &opts) {
    opts_ = opts;
  }
  /// Gets the training options from the component
  const NetTrainOptions& GetTrainOptions() const { 
    return opts_; 
  }

  virtual void InitData(std::istream &is) = 0;

 protected:
  /// Option-class with training hyper-parameters
  NetTrainOptions opts_;
};

} // namespace eesen


#endif
