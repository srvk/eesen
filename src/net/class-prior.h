// net/class-prior.h

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)
//           2015  Yajie Miao

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

#ifndef EESEN_CLASS_PRIOR_H_
#define EESEN_CLASS_PRIOR_H_

#include <cfloat>
#include <string>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cpucompute/matrix-lib.h"
#include "gpucompute/cuda-matrix.h"
#include "gpucompute/cuda-vector.h"

namespace eesen {

struct ClassPriorOptions {
  std::string class_frame_counts;
  BaseFloat prior_scale;
  BaseFloat prior_cutoff;
  BaseFloat blank_scale;

  ClassPriorOptions() : class_frame_counts(""),
                      prior_scale(1.0),
                      prior_cutoff(1e-10),
                      blank_scale(1.0) {}

  void Register(OptionsItf *po) {
    po->Register("class-frame-counts", &class_frame_counts,
                 "Vector with frame-counts of classes to compute log-priors."
                 " (priors are typically subtracted from log-posteriors"
                 " or pre-softmax activations)");
    po->Register("prior-scale", &prior_scale,
                 "Scaling factor to be applied on class-log-priors");
    po->Register("prior-cutoff", &prior_cutoff,
                 "Classes with priors lower than cutoff will have 0 likelihood");
    po->Register("blank-scale", &blank_scale,
                 "Scale probability of class 0 (blank) by this factor");
  }
};

class ClassPrior {
 public:
  /// Initialize class-prior from options
  explicit ClassPrior(const ClassPriorOptions &opts);

  /// Subtract class priors from log-posteriors to get pseudo log-likelihoods
  void SubtractOnLogpost(CuMatrixBase<BaseFloat> *llk);

 private:
  BaseFloat prior_scale_;
  CuVector<BaseFloat> log_priors_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(ClassPrior);
};

}  // namespace eesen

#endif  // EESEN_CLASS_PRIOR_H_
