// net/ctc-loss.cc

// Copyright 2015   Yajie Miao

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

#include "net/ctc-loss.h"
#include "gpucompute/cuda-math.h"
#include "gpucompute/ctc-utils.h"
#include "util/edit-distance.h"

#include <sstream>
#include <iterator>

namespace eesen {

void Ctc::Eval(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, CuMatrix<BaseFloat> *diff) {
  diff->Resize(net_out.NumRows(), net_out.NumCols());
  int32 num_frames = net_out.NumRows();
  int32 num_classes = net_out.NumCols();

  // label expansion by inserting blank (indexed by 0) at the beginning and end, 
  // and between every pair of labels
  int32 len_labels = label.size();
  int32 exp_len_labels = 2*len_labels + 1;

  label_expand_.resize(0);
  label_expand_.resize(exp_len_labels, 0);
  for (int l = 0; l < len_labels; l++) {
    label_expand_[2*l+1] = label[l];
  }

  // compute in log scale
  CuMatrix<BaseFloat> log_nnet_out(net_out);
  log_nnet_out.ApplyLog();

  alpha_.Resize(num_frames, exp_len_labels, kSetZero);
  beta_.Resize(num_frames, exp_len_labels, kSetZero);
  for (int t = 0; t < num_frames; t++) {
    alpha_.ComputeCtcAlpha(log_nnet_out, t, label_expand_, false);
  }
  for (int t = (num_frames - 1); t >= 0; t--) {
    beta_.ComputeCtcBeta(log_nnet_out, t, label_expand_, false);
  }

  // compute the log-likelihood of the label sequence given the inputs logP(z|x)
  BaseFloat tmp1 = alpha_(num_frames-1, exp_len_labels-1); 
  BaseFloat tmp2 = alpha_(num_frames-1, exp_len_labels-2);
  BaseFloat pzx = tmp1 + log(1 + ExpA(tmp2 - tmp1));

  // compute the errors
  ctc_err_.Resize(num_frames, num_classes, kSetZero);
  ctc_err_.ComputeCtcError(alpha_, beta_, net_out, label_expand_, pzx);  // here should use the original ??

  // back-propagate the errors through the softmax layer
  ctc_err_.MulElements(net_out);
  CuVector<BaseFloat> row_sum(num_frames, kSetZero);
  row_sum.AddColSumMat(1.0, ctc_err_, 0.0);
  
  CuMatrix<BaseFloat> net_out_tmp(net_out);
  net_out_tmp.MulRowsVec(row_sum);
  diff->CopyFromMat(ctc_err_);

  diff->AddMat(-1.0, net_out_tmp);

  // update registries
  obj_progress_ += pzx;
  sequences_progress_ += 1;
  sequences_num_ += 1;
  frames_progress_ += num_frames;
  frames_ += num_frames;

  // progressive reporting
  {
    if (sequences_progress_ >= report_step_) {
      KALDI_VLOG(1) << "After " << sequences_num_ << " sequences (" << frames_/(100.0 * 3600) << "Hr): "
                    << "Obj(log[Pzx]) = " << obj_progress_/sequences_progress_
                    << "   TokenAcc = " << 100.0*(1.0 - error_num_progress_/ref_num_progress_) << "%";
      // reset
      sequences_progress_ = 0;
      frames_progress_ = 0;
      obj_progress_ = 0.0;
      error_num_progress_ = 0;
      ref_num_progress_ = 0;
    }
  }

}

void Ctc::EvalParallel(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
                       std::vector< std::vector<int32> > &label, CuMatrix<BaseFloat> *diff) {
  diff->Resize(net_out.NumRows(), net_out.NumCols());

  int32 num_sequence = frame_num_utt.size();  // number of sequences
  int32 num_frames = net_out.NumRows();
  KALDI_ASSERT(num_frames % num_sequence == 0);  // after padding, number of frames is a multiple of number of sequences

  int32 num_frames_per_sequence = num_frames / num_sequence;
  int32 num_classes = net_out.NumCols();
  int32 max_label_len = 0;
  for (int32 s = 0; s < num_sequence; s++) {
    if (label[s].size() > max_label_len) max_label_len = label[s].size();
  }

  // label expansion
  std::vector<int32> label_lengths_utt(num_sequence);
  int32 exp_len_labels = 2*max_label_len + 1;
  label_expand_.resize(0);
  label_expand_.resize(num_sequence * exp_len_labels, -1);
  for (int32 s = 0; s < num_sequence; s++) {
    std::vector<int32> label_s = label[s];
    label_lengths_utt[s] = 2 * label_s.size() + 1;
    for (int32 l = 0; l < label_s.size(); l++) {
      label_expand_[s*exp_len_labels + 2*l] = 0;
      label_expand_[s*exp_len_labels + 2*l + 1] = label_s[l];
    }
    label_expand_[s*exp_len_labels + 2*label_s.size()] = 0;
  }

  // convert into the log scale
  CuMatrix<BaseFloat> log_nnet_out(net_out);
  log_nnet_out.ApplyLog();

  // do the forward and backward pass, to compute alpha and beta values
  alpha_.Resize(num_frames, exp_len_labels);
  beta_.Resize(num_frames, exp_len_labels);
  alpha_.Set(NumericLimits<BaseFloat>::log_zero_);
  beta_.Set(NumericLimits<BaseFloat>::log_zero_);
  for (int t = 0; t < num_frames_per_sequence; t++) {
    alpha_.ComputeCtcAlphaMSeq(log_nnet_out, t, label_expand_, frame_num_utt);
  }
  for (int t = (num_frames_per_sequence - 1); t >= 0; t--) {
    beta_.ComputeCtcBetaMSeq(log_nnet_out, t, label_expand_, frame_num_utt, label_lengths_utt);
  }
  CuVector<BaseFloat> pzx(num_sequence, kSetZero);
  for (int s = 0; s < num_sequence; s++) {
    int label_len = 2* label[s].size() + 1;
    int frame_num = frame_num_utt[s];
    BaseFloat tmp1 = alpha_((frame_num-1)*num_sequence + s, label_len - 1);
    BaseFloat tmp2 = alpha_((frame_num-1)*num_sequence + s, label_len-2);
    pzx(s) = tmp1 + log(1 + ExpA(tmp2 - tmp1));
  }

  // gradients from CTC
  ctc_err_.Resize(num_frames, num_classes, kSetZero);
  ctc_err_.ComputeCtcErrorMSeq(alpha_, beta_, net_out, label_expand_, frame_num_utt, pzx);  // here should use the original ??

  // back-propagate the errors through the softmax layer
  ctc_err_.MulElements(net_out);
  CuVector<BaseFloat> row_sum(num_frames, kSetZero);
  row_sum.AddColSumMat(1.0, ctc_err_, 0.0);

  CuMatrix<BaseFloat> net_out_tmp(net_out);
  net_out_tmp.MulRowsVec(row_sum);
  diff->CopyFromMat(ctc_err_);

  diff->AddMat(-1.0, net_out_tmp);

  // update registries
  obj_progress_ += pzx.Sum();
  sequences_progress_ += num_sequence;
  sequences_num_ += num_sequence;
  for (int s = 0; s < num_sequence; s++) {
    frames_progress_ += frame_num_utt[s];
    frames_ += frame_num_utt[s];
  }

  // progressive reporting
  {
    if (sequences_progress_ >= report_step_) {
      KALDI_VLOG(1) << "After " << sequences_num_ << " sequences (" << frames_/(100.0 * 3600) << "Hr): "
                    << "Obj(log[Pzx]) = " << obj_progress_/sequences_progress_
                    << "   TokenAcc = " << 100.0*(1.0 - error_num_progress_/ref_num_progress_) << "%";
      // reset
      sequences_progress_ = 0;
      frames_progress_ = 0;
      obj_progress_ = 0.0;
      error_num_progress_ = 0;
      ref_num_progress_ = 0;
    }
  }

}
  
void Ctc::ErrorRate(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, float* err_rate, std::vector<int32> *hyp) {

  // frame-level labels, by selecting the label with the largest probability at each frame
  CuArray<int32> maxid(net_out.NumRows());
  net_out.FindRowMaxId(&maxid);

  int32 dim = maxid.Dim();
  
  std::vector<int32> data(dim);
  maxid.CopyToVec(&data);

  // remove the repetitions
  int32 i = 1, j = 1;
  while(j < dim) {
    if (data[j] != data[j-1]) {
      data[i] = data[j];
      i++;
    }
    j++;
  }
  // remove the blanks
  std::vector<int32> hyp_seq(0);
  for (int32 n = 0; n < i; n++) {
    if (data[n] != 0) {
      hyp_seq.push_back(data[n]);
    }
  }
  hyp->resize(0);
  *hyp = hyp_seq;

  int32 err, ins, del, sub;
  err =  LevenshteinEditDistance(label, hyp_seq, &ins, &del, &sub);
  *err_rate = (100.0 * err) / label.size();
  error_num_ += err;
  ref_num_ += label.size();
  error_num_progress_ += err;
  ref_num_progress_ += label.size();
}

void Ctc::ErrorRateMSeq(const std::vector<int> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out, std::vector< std::vector<int> > &label) {

  // frame-level labels
  CuArray<int32> maxid(net_out.NumRows());
  net_out.FindRowMaxId(&maxid);

  int32 dim = maxid.Dim();
  std::vector<int32> data(dim);
  maxid.CopyToVec(&data);

  // compute errors sequence by sequence
  int32 num_seq = frame_num_utt.size();
  for (int32 s = 0; s < num_seq; s++) {
    int32 num_frame = frame_num_utt[s];
    std::vector<int32> raw_hyp_seq(num_frame);
    for (int32 f = 0; f < num_frame; f++) {
      raw_hyp_seq[f] = data[f*num_seq + s];
    }    
    int32 i = 1, j = 1;
    while(j < num_frame) {
      if (raw_hyp_seq[j] != raw_hyp_seq[j-1]) {
        raw_hyp_seq[i] = raw_hyp_seq[j];
        i++;
      }
      j++;
    }
    std::vector<int32> hyp_seq(0);
    for (int32 n = 0; n < i; n++) {
      if (raw_hyp_seq[n] != 0) {
        hyp_seq.push_back(raw_hyp_seq[n]);
      }
    }
    int32 err, ins, del, sub;
    err =  LevenshteinEditDistance(label[s], hyp_seq, &ins, &del, &sub);
    error_num_ += err;
    ref_num_ += label[s].size();
    error_num_progress_ += err;
    ref_num_progress_ += label[s].size();
  }
}

std::string Ctc::Report() {
  std::ostringstream oss;
  oss << "\nTOKEN_ACCURACY >> " << 100.0*(1.0 - error_num_/ref_num_) << "% <<";
  return oss.str(); 
}

} // namespace eesen
