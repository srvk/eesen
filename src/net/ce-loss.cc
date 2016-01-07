// net/ce-loss.cc

// Copyright 2011  Brno University of Technology (author: Karel Vesely)
//           2015  Yajie Miao
//           2015  Guoli Ye

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

#include "net/ce-loss.h"
#include "gpucompute/cuda-math.h"

#include <sstream>
#include <iterator>

namespace eesen {

void CE::Eval(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &target, CuMatrix<BaseFloat> *diff) {
  KALDI_ASSERT(net_out.NumRows() == target.size());
  diff->Resize(net_out.NumRows(), net_out.NumCols());
  int32 num_frames = net_out.NumRows();
  int32 num_classes = net_out.NumCols();

  // Convert labels to a matrix
  Matrix<BaseFloat> target_mat_host(num_frames, num_classes, kSetZero);
  for (int32 t = 0; t < num_frames; t++) {
    int32 classX = target[t];
    if (classX >= num_classes) {
      KALDI_ERR << "Class id out of network output dimension. Net outputs: "
                << num_classes << ", class ID : " << classX;
    }
    target_mat_host(t, classX) = 1.0;
  }
  // Copy the target mat to GPU
  target_mat_device_ = target_mat_host;

  // Compute derivatives with respect to the activation before softmax
  *diff = net_out;
  diff->AddMat(-1.0, target_mat_device_);

  // Evaluate the frame-level classification accuracy
  net_out.FindRowMaxId(&max_id_pred_device_);  // The label with the max posterior at each frame
  max_id_pred_host_.resize(num_frames);
  max_id_pred_device_.CopyToVec(&max_id_pred_host_);
  for(int32 t = 0; t < num_frames; t++) {
    if (target[t] == max_id_pred_host_[t]) {correct_++; correct_progress_++;}
  }

  // Evaluate the cross-entropy objective
  cross_entropy_device_ = net_out;
  cross_entropy_device_.ApplyLog();
  cross_entropy_device_.MulElements(target_mat_device_);
  double cross_entropy = -cross_entropy_device_.Sum();
  
  obj_ += cross_entropy;
  obj_progress_ += cross_entropy;
  sequences_progress_ += 1;
  sequences_num_ += 1;
  frames_progress_ += num_frames;
  frames_ += num_frames;

  // progressive reporting
  {
    if (sequences_progress_ > report_step_) {
      KALDI_LOG << "After " << sequences_num_ << " sequences (" << frames_/(100.0 * 3600) << "Hr): "
                    << "CE-Obj = " << obj_progress_/sequences_progress_
                    << "Frame-level CE-Obj = " << obj_progress_/frames_progress_
                    << "   FrameAcc = " << 100.0*(double(correct_progress_)/frames_progress_) << "%"
		    << " obj_progress_=  " << obj_progress_  
		    << " sequences_progress_=  " << sequences_progress_  
		    << " frames_progress_=  " << frames_progress_  ;
      // reset
      sequences_progress_ = 0;
      frames_progress_ = 0;
      obj_progress_ = 0.0;
      correct_progress_ = 0;
    }
  }

}

void CE::EvalParallel(const CuMatrixBase<BaseFloat> &net_out, 
                      const std::vector<int32> &target, 
                      CuMatrix<BaseFloat> *diff,
                      const VectorBase<BaseFloat> &frame_mask_host,
                      int sequence_number_in_batch) {

  KALDI_ASSERT(net_out.NumRows() == target.size());
  diff->Resize(net_out.NumRows(), net_out.NumCols());
  int32 num_frames = net_out.NumRows();
  int32 num_classes = net_out.NumCols();

  // Convert labels to a matrix
  Matrix<BaseFloat> target_mat_host(num_frames, num_classes, kSetZero);
  for (int32 t = 0; t < num_frames; t++) {
    int32 classX = target[t];
    if (classX >= num_classes) {
      KALDI_ERR << "Class id out of network output dimension. Net outputs: "
                << num_classes << ", class ID : " << classX;
    }
    target_mat_host(t, classX) = 1.0;
  }
  // Copy the target mat to GPU
  target_mat_device_ = target_mat_host;

  // Compute derivatives with respect to the activation before softmax 
  *diff = net_out;
  diff->AddMat(-1.0, target_mat_device_);

  // Frame mask to GPU
  frame_mask_device_.Resize(frame_mask_host.Dim());
  frame_mask_device_.CopyFromVec(frame_mask_host);
  diff->MulRowsVec(frame_mask_device_);

  // Evaluate the frame-level classification accuracy
  net_out.FindRowMaxId(&max_id_pred_device_);  // The label with the max posterior at each frame
  max_id_pred_host_.resize(num_frames);
  max_id_pred_device_.CopyToVec(&max_id_pred_host_);
  for(int32 t = 0; t < num_frames; t++) {
    if (frame_mask_host(t) == 1 && target[t] == max_id_pred_host_[t]) {
      correct_++; correct_progress_++;
    }
  }

  // Evaluate the cross-entropy objective
  cross_entropy_device_ = net_out;
  cross_entropy_device_.ApplyLog();
  cross_entropy_device_.MulElements(target_mat_device_);
  cross_entropy_device_.MulRowsVec(frame_mask_device_);
  double cross_entropy = -cross_entropy_device_.Sum();

  obj_ += cross_entropy;
  obj_progress_ += cross_entropy;
  sequences_progress_ += sequence_number_in_batch;
  sequences_num_ += sequence_number_in_batch;
  frames_progress_ += num_frames;
  frames_ += num_frames;

  // progressive reporting
  {
    if (sequences_progress_ > report_step_) {
      KALDI_LOG << "After " << sequences_num_ << " sequences (" << frames_/(100.0 * 3600) << "Hr): "
                    << "CE-Obj = " << obj_progress_/sequences_progress_
                    << "Frame-level CE-Obj = " << obj_progress_/frames_progress_
                    << "   FrameAcc = " << 100.0*(double(correct_progress_)/frames_progress_) << "%"
		    << " obj_progress_=  " << obj_progress_  
		    << " sequences_progress_=  " << sequences_progress_  
		    << " frames_progress_=  " << frames_progress_  ;
      // reset
      sequences_progress_ = 0;
      frames_progress_ = 0;
      obj_progress_ = 0.0;
      correct_progress_ = 0;
    }
  }

}

std::string CE::Report() {
  std::ostringstream oss;
  oss << "\nFRAME_ACCURACY >> " << 100.0*(correct_/frames_) << "% <<";
  return oss.str();
}

}
