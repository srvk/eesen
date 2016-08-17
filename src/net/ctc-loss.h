// net/ctc-loss.h

// Copyright 2015  Yajie Miao, Hang Su, Mohammad Gowayyed

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

#ifndef EESEN_CTC_LOSS_H_
#define EESEN_CTC_LOSS_H_

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"
#include "gpucompute/cuda-matrix.h"
#include "gpucompute/cuda-vector.h"
#include "gpucompute/cuda-array.h"

namespace eesen {

class Ctc {
 public:
  Ctc() : frames_(0), sequences_num_(0), ref_num_(0), error_num_(0), 
          frames_progress_(0), ref_num_progress_(0), error_num_progress_(0),
          sequences_progress_(0), obj_progress_(0.0), report_step_(100) { }
  ~Ctc() { }

  /// CTC training over a single sequence from the labels. The errors are returned to [diff]
  void Eval(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, CuMatrix<BaseFloat> *diff);

  /// CTC training over multiple sequences. The errors are returned to [diff]
  void EvalParallel(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
                    std::vector< std::vector<int32> > &label, CuMatrixBase<BaseFloat> *diff, const bool block);

  /// Compute token error rate from the softmax-layer activations and the given labels. From the softmax activations,
  /// we get the frame-level labels, by selecting the label with the largest probability at each frame. Then, the frame
  /// -level labels are shrunk by removing the blanks and collasping the repetitions. This gives us the utterance-level
  /// labels, from which we can compute the error rate. The error rate is the Levenshtein distance between the hyp labels
  /// and the given reference label sequence.
  void ErrorRate(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, float* err, std::vector<int32> *hyp);

  /// Compute token error rate over multiple sequences. 
  void ErrorRateMSeq(const std::vector<int> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out, std::vector< std::vector<int> > &label, std::string &out);

  /// Set the step of reporting
  void SetReportStep(int32 report_step) { report_step_ = report_step;  }

  /// Generate string with report
  std::string Report();

  float NumErrorTokens() const { return error_num_;}
  int32 NumRefTokens() const { return ref_num_;}

 private:
  int32 frames_;                    // total frame number
  int32 sequences_num_; 
  int32 ref_num_;                   // total number of tokens in label sequences
  float error_num_;                 // total number of errors (edit distance between hyp and ref)

  int32 frames_progress_;
  int32 ref_num_progress_;
  float error_num_progress_;

  int32 sequences_progress_;         // registry for the number of sequences
  double obj_progress_;              // registry for the optimization objective

  int32 report_step_;                // report obj and accuracy every so many sequences/utterances

  std::vector<int32> label_expand_;  // expanded version of the label sequence
  CuMatrix<BaseFloat> alpha_;        // alpha values
  CuMatrix<BaseFloat> beta_;         // beta values
  CuMatrix<BaseFloat> ctc_err_;      // ctc errors
};

} // namespace eesen

#endif
