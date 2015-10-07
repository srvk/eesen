// net/ce-loss.h

// Copyright 2011  Brno University of Technology (author: Karel Vesely)
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

#ifndef EESEN_CE_LOSS_H_
#define EESEN_CE_LOSS_H_

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"
#include "gpucompute/cuda-matrix.h"
#include "gpucompute/cuda-vector.h"
#include "gpucompute/cuda-array.h"

namespace eesen {

class CE {
 public:
  CE() : frames_(0), sequences_num_(0), correct_(0), obj_(0.0), 
         frames_progress_(0), sequences_progress_(0), correct_progress_(0.0), obj_progress_(0.0) {}
  ~CE() { }

  /// CE training over a single sequence from the labels. The errors are returned to [diff]
  void Eval(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &target, CuMatrix<BaseFloat> *diff);
 
  /// CE training over multiple sequences. Frame masks need to be used to mask out padding frames.
  void EvalParallel(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &target, 
                    CuMatrix<BaseFloat> *diff, const VectorBase<BaseFloat> &frame_mask_host,
                    int sequence_number_in_batch);
 
  /// Set the step of reporting
  void SetReportStep(int32 report_step) { report_step_ = report_step;  }

  /// Generate string with report
  std::string Report();

 private:
  int32 frames_;
  int32 sequences_num_;
  int32 correct_;
  double obj_;

  int32 frames_progress_;
  int32 sequences_progress_;
  int32 correct_progress_;
  double obj_progress_;

  int32 report_step_;

  CuVector<BaseFloat> frame_mask_device_;

  /// GPU matrix to store a sequence's labels
  CuMatrix<BaseFloat> target_mat_device_;

  /// Class IDs with the max network outputs, in GPU and CPU vectors
  CuArray<int32> max_id_pred_device_;
  std::vector<int32> max_id_pred_host_;        

  /// CPU matrix to store the cross-entropy values
  CuMatrix<BaseFloat> cross_entropy_device_;

};

} // namespace eesen

#endif

