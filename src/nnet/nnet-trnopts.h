// nnet/nnet-trnopts.h

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)

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

#ifndef KALDI_NNET_NNET_TRNOPTS_H_
#define KALDI_NNET_NNET_TRNOPTS_H_

#include "base/kaldi-common.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace nnet1 {


struct NnetTrainOptions {
  // option declaration
  BaseFloat learn_rate;
  BaseFloat momentum;
  BaseFloat l2_penalty;
  BaseFloat l1_penalty;
  // default values
  NnetTrainOptions() : learn_rate(0.008),
                       momentum(0.0),
                       l2_penalty(0.0),
                       l1_penalty(0.0) 
                       { }
  // register options
  void Register(OptionsItf *po) {
    po->Register("learn-rate", &learn_rate, "Learning rate");
    po->Register("momentum", &momentum, "Momentum");
    po->Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po->Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");
  }
  // print for debug purposes
  friend std::ostream& operator<<(std::ostream& os, const NnetTrainOptions& opts) {
    os << "RbmTrainOptions : "
       << "learn_rate" << opts.learn_rate << ", "
       << "momentum" << opts.momentum << ", "
       << "l2_penalty" << opts.l2_penalty << ", "
       << "l1_penalty" << opts.l1_penalty;
    return os;
  }
};


}//namespace nnet1
}//namespace kaldi

#endif
