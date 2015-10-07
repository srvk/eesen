// net/train-opts.h

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

#ifndef EESEN_TRAIN_OPTS_H_
#define EESEN_TRAIN_OPTS_H_

#include "base/kaldi-common.h"
#include "util/options-itf.h"

namespace eesen {


struct NetTrainOptions {
  // option declaration
  BaseFloat learn_rate;
  BaseFloat momentum;
  // default values
  NetTrainOptions() : learn_rate(0.008),
                      momentum(0.0)
                      {}
  // register options
  void Register(OptionsItf *po) {
    po->Register("learn-rate", &learn_rate, "Learning rate");
    po->Register("momentum", &momentum, "Momentum");
  }
  // print for debug purposes
  friend std::ostream& operator<<(std::ostream& os, const NetTrainOptions& opts) {
    os << "TrainOptions : "
       << "learn_rate" << opts.learn_rate << ", "
       << "momentum" << opts.momentum;
    return os;
  }
};

}//namespace eesen

#endif
