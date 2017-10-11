// net/bilstm-layer.h

// Copyright 2015   Yajie Miao, Hang Su

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

#ifndef EESEN_BILSTM_LAYER_H_
#define EESEN_BILSTM_LAYER_H_

#include "net/layer.h"
#include "net/trainable-layer.h"
#include "net/utils-functions.h"
#include "gpucompute/cuda-math.h"

namespace eesen {

class BiLstm : public TrainableLayer {
public:
    BiLstm(int32 input_dim, int32 output_dim) :
        TrainableLayer(input_dim, output_dim),
        cell_dim_(output_dim/2),
        learn_rate_coef_(1.0), max_grad_(0.0),
        forward_dropout(0.0), forward_step_dropout(false), forward_sequence_dropout(false),
        recurrent_step_dropout(false), recurrent_sequence_dropout(false),
        rnndrop(false), no_mem_loss_dropout(false),
        recurrent_dropout(0.0),
        in_train(true),
        drop_factor_(0.0), adaBuffersInitialized(false)
    { }

    ~BiLstm()
    { }

    Layer* Copy() const { return new BiLstm(*this); }
    LayerType GetType() const { return l_BiLstm; }
    LayerType GetTypeNonParal() const { return l_BiLstm; }

    void SetTestMode() {
      in_train = false;
    }

    void SetTrainMode() {
      in_train = true;
    }

    void SetDropFactor(float drop_factor) {
      //Not used
      //drop_factor_ = drop_factor;
    }

    void ChangeDropoutParameters(BaseFloat in_forward_dropout,
                                  bool in_fw_step_dropout,
                                  bool in_fw_sequence_dropout,

                                  bool in_rnndrop ,
                                  bool in_no_mem_loss_dropout,
                                  BaseFloat in_recurrent_dropout,
                                  bool in_recurrent_step_dropout,
                                  bool in_recurrent_sequence_dropout,

                                  bool in_twiddleforward) {

      // Forward Dropout checks and initialization
      if (in_forward_dropout > 0.0 && !(in_fw_sequence_dropout || in_fw_step_dropout))
        KALDI_ERR << "ForwardDropoutFactor > 0 but ForwardTimeStepDropout and ForwardSequenceDropout are both false, One must be true.";

      if (in_fw_sequence_dropout && in_fw_step_dropout)
        KALDI_ERR << "Both ForwardTimeStepDropout and ForwardSequenceDropout are true, Only one can be true.";

      if (in_forward_dropout == 0.0 && (in_fw_sequence_dropout || in_fw_step_dropout))
        KALDI_ERR << "ForwardDropoutFactor = 0 but ForwardTimeStepDropout and/or ForwardSequenceDropout is true, both must be false.";

      forward_dropout          = in_forward_dropout;
      forward_step_dropout     = in_fw_step_dropout;
      forward_sequence_dropout = in_fw_sequence_dropout;
      twiddle_forward          = in_twiddleforward;

      // Report values in logs
      if (forward_dropout >0)
        KALDI_LOG << "Forward drop selected with factor @ " << forward_dropout;
      if (forward_step_dropout)
        KALDI_LOG << "Step DROPOUT selected for forward (non-recurrent) connections";
      if (forward_sequence_dropout)
        KALDI_LOG << "Sequence DROPOUT selected for forward (non-recurrent) connections";
      if (twiddle_forward)
        KALDI_LOG << "TWIDDLE selected for forward (non-recurrent) connections";


      // Recurrent Dropout checks and initialization

      if (in_recurrent_sequence_dropout && in_recurrent_step_dropout)
        KALDI_ERR << "RecurrentSequenceDropout and RecurrentTimeStepDropout cannot be true at the same time. Pick one.";

      if (in_rnndrop == true && in_no_mem_loss_dropout== true)
        KALDI_ERR << "Only one of RNNDrop, NoMemLossDropout can be true. Pick one.";

      if (in_recurrent_dropout == 0.0 && (in_no_mem_loss_dropout || in_rnndrop))
        KALDI_ERR << "RecurrentDropoutFactor must be nonzero if RNNDrop or NoMemLossDropout is true";

      if (!(in_recurrent_step_dropout || in_recurrent_sequence_dropout) && (in_rnndrop || in_no_mem_loss_dropout))
        KALDI_ERR << " Either RecurrentSequenceDropout or RecurrentTimeStepDropout must be true if RNNDrop or NoMemLossDropout is true";

      recurrent_dropout = in_recurrent_dropout;
      recurrent_step_dropout = in_recurrent_step_dropout;
      recurrent_sequence_dropout = in_recurrent_sequence_dropout;

      rnndrop = in_rnndrop;
      no_mem_loss_dropout = in_no_mem_loss_dropout;

      // Report values in logs
      if (recurrent_step_dropout)
        KALDI_LOG << "Step DROPOUT selected for recurrent connections";
      if (recurrent_sequence_dropout)
        KALDI_LOG << "Sequence DROPOUT selected for recurrent connections";

      if (rnndrop)
        KALDI_LOG << "RNNdrop selected with factor @ " << recurrent_dropout ;

      if (no_mem_loss_dropout)
        KALDI_LOG << "NML drop selected with factor @ " << recurrent_dropout ;

    }

    void InitData(std::istream &is) {
      // define options
      float param_range = 0.02, max_grad = 0.0;
      float learn_rate_coef = 1.0;
      float fgate_bias_init = 0.0;   // the initial value for the bias of the forget gates

      // t for temp as in temp variable
      BaseFloat t_forward_dropout = 0.0;
      bool t_fw_step_dropout = false;
      bool t_fw_sequence_dropout = false;

      bool t_recurrent_step_dropout = false;
      bool t_recurrent_sequence_dropout = false;

      bool t_rnndrop = false;
      bool t_no_mem_loss_dropout = false;

      bool t_twiddleforward = false;

      BaseFloat t_recurrent_dropout = 0.0;

      // parse config
      std::string token;
      while (!is.eof()) {
        ReadToken(is, false, &token);
        if (token == "<ParamRange>")  ReadBasicType(is, false, &param_range);
        else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
        else if (token == "<MaxGrad>") ReadBasicType(is, false, &max_grad);
        else if (token == "<FgateBias>") ReadBasicType(is, false, &fgate_bias_init);
        // Forward dropout parameters
        else if (token == "<ForwardDropoutFactor>") ReadBasicType(is, false, &t_forward_dropout);
        else if (token == "<ForwardTimeStepDropout>") ReadBasicType(is, false, &t_fw_step_dropout);
        else if (token == "<ForwardSequenceDropout>") ReadBasicType(is, false, &t_fw_sequence_dropout);

        // Recurrent dropout parameters
        else if (token == "<RecurrentTimeStepDropout>") ReadBasicType(is, false, &t_recurrent_step_dropout);
        else if (token == "<RecurrentSequenceDropout>") ReadBasicType(is, false, &t_recurrent_sequence_dropout);
        else if (token == "<RecurrentDropoutFactor>") ReadBasicType(is, false, &t_recurrent_dropout);

        // Recurrent dropout type
        else if (token == "<RNNDrop>") ReadBasicType(is, false, &t_rnndrop);
        else if (token == "<NoMemLossDropout>") ReadBasicType(is, false, &t_no_mem_loss_dropout);

        // Twiddle
        else if (token == "<TwiddleForward>") ReadBasicType(is, false, &t_twiddleforward);

        else KALDI_ERR << "Unknown token " << token << ", a typo in config? Check code for all options"
                       << " (ParamRange|LearnRateCoef|BiasLearnRateCoef|MaxGrad)";
        is >> std::ws; // eat-up whitespace
      }

      // initialize weights and biases for the forward sub-layer
      wei_gifo_x_fw_.Resize(4 * cell_dim_, input_dim_); wei_gifo_x_fw_.InitRandUniform(param_range);
      // the weights connecting momory cell outputs with the units/gates
      wei_gifo_m_fw_.Resize(4 * cell_dim_, cell_dim_);  wei_gifo_m_fw_.InitRandUniform(param_range);
      // the bias for the units/gates
      bias_fw_.Resize(4 * cell_dim_); bias_fw_.InitRandUniform(param_range);
      if (fgate_bias_init != 0.0) {   // reset the bias of the forget gates
        bias_fw_.Range(2 * cell_dim_, cell_dim_).Set(fgate_bias_init);
      }
      // peephole connections for i, f, and o, with diagonal matrices (vectors)
      phole_i_c_fw_.Resize(cell_dim_); phole_i_c_fw_.InitRandUniform(param_range);
      phole_f_c_fw_.Resize(cell_dim_); phole_f_c_fw_.InitRandUniform(param_range);
      phole_o_c_fw_.Resize(cell_dim_); phole_o_c_fw_.InitRandUniform(param_range);

      // initialize weights and biases for the backward sub-layer
      wei_gifo_x_bw_.Resize(4 * cell_dim_, input_dim_); wei_gifo_x_bw_.InitRandUniform(param_range);
      wei_gifo_m_bw_.Resize(4 * cell_dim_, cell_dim_);  wei_gifo_m_bw_.InitRandUniform(param_range);
      bias_bw_.Resize(4 * cell_dim_); bias_bw_.InitRandUniform(param_range);
      if (fgate_bias_init != 0.0) {   // reset the bias of the forget gates
        bias_bw_.Range(2 * cell_dim_, cell_dim_).Set(fgate_bias_init);
      }

      phole_i_c_bw_.Resize(cell_dim_); phole_i_c_bw_.InitRandUniform(param_range);
      phole_f_c_bw_.Resize(cell_dim_); phole_f_c_bw_.InitRandUniform(param_range);
      phole_o_c_bw_.Resize(cell_dim_); phole_o_c_bw_.InitRandUniform(param_range);

      learn_rate_coef_ = learn_rate_coef;
      max_grad_ = max_grad;

      // Forward Dropout checks and initialization
      if (t_forward_dropout > 0.0 && !(t_fw_sequence_dropout || t_fw_step_dropout))
        KALDI_ERR << "ForwardDropoutFactor > 0 but ForwardTimeStepDropout and ForwardSequenceDropout are both false, One must be true.";

      if (t_fw_sequence_dropout && t_fw_step_dropout)
        KALDI_ERR << "Both ForwardTimeStepDropout and ForwardSequenceDropout are true, Only one can be true.";

      if (t_forward_dropout == 0.0 && (t_fw_sequence_dropout || t_fw_step_dropout))
        KALDI_ERR << "ForwardDropoutFactor = 0 but ForwardTimeStepDropout and/or ForwardSequenceDropout is true, both must be false.";

      forward_dropout          = t_forward_dropout;
      forward_step_dropout     = t_fw_step_dropout;
      forward_sequence_dropout = t_fw_sequence_dropout;
      twiddle_forward          = t_twiddleforward;

      // Report values in logs
      if (forward_dropout >0)
        KALDI_LOG << "Forward drop selected with factor @ " << forward_dropout;
      if (forward_step_dropout)
        KALDI_LOG << "Step DROPOUT selected for forward (non-recurrent) connections";
      if (forward_sequence_dropout)
        KALDI_LOG << "Sequence DROPOUT selected for forward (non-recurrent) connections";
      if (twiddle_forward)
        KALDI_LOG << "TWIDDLE selected for forward/recurrent connections";


      // Recurrent Dropout checks and initialization
      if (t_recurrent_sequence_dropout && t_recurrent_step_dropout )
        KALDI_ERR << "RecurrentSequenceDropout and RecurrentTimeStepDropout cannot be true at the same time. Pick one.";

      if (t_rnndrop == true && t_no_mem_loss_dropout== true)
        KALDI_ERR << "Only one of RNNDrop, NoMemLossDropout can be true. Pick one.";

      if (t_recurrent_dropout == 0.0 && (t_no_mem_loss_dropout || t_rnndrop))
        KALDI_ERR << "RecurrentDropoutFactor must be nonzero if RNNDrop or NoMemLossDropout is true";

      if (!(t_recurrent_step_dropout || t_recurrent_sequence_dropout) && (t_rnndrop  || t_no_mem_loss_dropout))
        KALDI_ERR << " Either RecurrentSequenceDropout or RecurrentTimeStepDropout must be true if RNNDrop or NoMemLossDropout is true";

      recurrent_dropout = t_recurrent_dropout;
      recurrent_step_dropout = t_recurrent_step_dropout;
      recurrent_sequence_dropout = t_recurrent_sequence_dropout;

      rnndrop = t_rnndrop;
      no_mem_loss_dropout = t_no_mem_loss_dropout;

      // Report values in logs
      if (recurrent_step_dropout)
        KALDI_LOG << "Step DROPOUT selected for recurrent connections";
      if (recurrent_sequence_dropout)
        KALDI_LOG << "Sequence DROPOUT selected for recurrent connections";

      if (rnndrop)
        KALDI_LOG << "RNNdrop selected with factor @ " << recurrent_dropout ;

      if (no_mem_loss_dropout)
        KALDI_LOG << "NML drop selected with factor @ " << recurrent_dropout ;

    }

   void InitAdaBuffers() {
      //fw for Ada:
      wei_gifo_x_fw_corr_accu.Resize(4 * cell_dim_, input_dim_); wei_gifo_x_fw_corr_accu.Set(0.0);
      wei_gifo_x_fw_corr_accu_scale.Resize(4 * cell_dim_, input_dim_);

      wei_gifo_m_fw_corr_accu.Resize(4 * cell_dim_, cell_dim_);  wei_gifo_m_fw_corr_accu.Set(0.0);
      wei_gifo_m_fw_corr_accu_scale.Resize(4 * cell_dim_, cell_dim_);

      bias_fw_corr_accu.Resize(4 * cell_dim_);  bias_fw_corr_accu.Set(0.0);
      bias_fw_corr_accu_scale.Resize(4 * cell_dim_);

      phole_i_c_fw_corr_accu.Resize(cell_dim_); phole_i_c_fw_corr_accu.Set(0.0);
      phole_i_c_fw_corr_accu_scale.Resize(cell_dim_);

      phole_f_c_fw_corr_accu.Resize(cell_dim_); phole_f_c_fw_corr_accu.Set(0.0);
      phole_f_c_fw_corr_accu_scale.Resize(cell_dim_);

      phole_o_c_fw_corr_accu.Resize(cell_dim_); phole_o_c_fw_corr_accu.Set(0.0);
      phole_o_c_fw_corr_accu_scale.Resize(cell_dim_);

      //bw for Ada:
      wei_gifo_x_bw_corr_accu.Resize(4 * cell_dim_, input_dim_); wei_gifo_x_bw_corr_accu.Set(0.0);
      wei_gifo_x_bw_corr_accu_scale.Resize(4 * cell_dim_, input_dim_);

      wei_gifo_m_bw_corr_accu.Resize(4 * cell_dim_, cell_dim_);  wei_gifo_m_bw_corr_accu.Set(0.0);
      wei_gifo_m_bw_corr_accu_scale.Resize(4 * cell_dim_, cell_dim_);

      bias_bw_corr_accu.Resize(4 * cell_dim_);  bias_bw_corr_accu.Set(0.0);
      bias_bw_corr_accu_scale.Resize(4 * cell_dim_);

      phole_i_c_bw_corr_accu.Resize(cell_dim_); phole_i_c_bw_corr_accu.Set(0.0);
      phole_i_c_bw_corr_accu_scale.Resize(cell_dim_);

      phole_f_c_bw_corr_accu.Resize(cell_dim_); phole_f_c_bw_corr_accu.Set(0.0);
      phole_f_c_bw_corr_accu_scale.Resize(cell_dim_);

      phole_o_c_bw_corr_accu.Resize(cell_dim_); phole_o_c_bw_corr_accu.Set(0.0);
      phole_o_c_bw_corr_accu_scale.Resize(cell_dim_);

      adaBuffersInitialized = true;
    }

    void ReadData(std::istream &is, bool binary) {
      //for initAdaBuffers();
      adaBuffersInitialized = false;
      
      // optional learning-rate coefs
      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<LearnRateCoef>");
        ReadBasicType(is, binary, &learn_rate_coef_);
      }
      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<MaxGrad>");
        ReadBasicType(is, binary, &max_grad_);
      }

      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<ForwardDropoutFactor>");
        ReadBasicType(is, binary, &forward_dropout);
      }

      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<ForwardTimeStepDropout>");
        ReadBasicType(is, binary, &forward_step_dropout);
      }

      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<ForwardSequenceDropout>");
        ReadBasicType(is, binary, &forward_sequence_dropout);
      }

      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<RecurrentTimeStepDropout>");
        ReadBasicType(is, binary, &recurrent_step_dropout);
      }

      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<RecurrentSequenceDropout>");
        ReadBasicType(is, binary, &recurrent_sequence_dropout);
      }

      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<RNNDrop>");
        ReadBasicType(is, binary, &rnndrop);
      }
      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<NoMemLossDropout>");
        ReadBasicType(is, binary, &no_mem_loss_dropout);
      }

      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<RecurrentDropoutFactor>");
        ReadBasicType(is, binary, &recurrent_dropout);
      }

      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<TwiddleForward>");
        ReadBasicType(is, binary, &twiddle_forward);
      }

      // optionally read in accumolators for AdaGrad and RMSProp
      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<BiLstmAccus>");

        InitAdaBuffers();
        
        wei_gifo_x_fw_corr_accu.Read(is, binary);
        wei_gifo_m_fw_corr_accu.Read(is, binary);
        bias_fw_corr_accu.Read(is, binary);
        phole_i_c_fw_corr_accu.Read(is, binary);
        phole_f_c_fw_corr_accu.Read(is, binary);
        phole_o_c_fw_corr_accu.Read(is, binary);

        wei_gifo_x_bw_corr_accu.Read(is, binary);
        wei_gifo_m_bw_corr_accu.Read(is, binary);
        bias_bw_corr_accu.Read(is, binary);
        phole_i_c_bw_corr_accu.Read(is, binary);
        phole_f_c_bw_corr_accu.Read(is, binary);
        phole_o_c_bw_corr_accu.Read(is, binary);

      }

      // read parameters of forward layer
      wei_gifo_x_fw_.Read(is, binary);
      wei_gifo_m_fw_.Read(is, binary);
      bias_fw_.Read(is, binary);
      phole_i_c_fw_.Read(is, binary);
      phole_f_c_fw_.Read(is, binary);
      phole_o_c_fw_.Read(is, binary);
      // initialize the buffer for gradients updates
      wei_gifo_x_fw_corr_ = wei_gifo_x_fw_; wei_gifo_x_fw_corr_.SetZero();
      wei_gifo_m_fw_corr_ = wei_gifo_m_fw_; wei_gifo_m_fw_corr_.SetZero();
      bias_fw_corr_ = bias_fw_; bias_fw_corr_.SetZero();
      phole_i_c_fw_corr_ = phole_i_c_fw_; phole_i_c_fw_corr_.SetZero();
      phole_f_c_fw_corr_ = phole_f_c_fw_; phole_f_c_fw_corr_.SetZero();
      phole_o_c_fw_corr_ = phole_o_c_fw_; phole_o_c_fw_corr_.SetZero();

      // read parameters of backward layer
      wei_gifo_x_bw_.Read(is, binary);
      wei_gifo_m_bw_.Read(is, binary);
      bias_bw_.Read(is, binary);
      phole_i_c_bw_.Read(is, binary);
      phole_f_c_bw_.Read(is, binary);
      phole_o_c_bw_.Read(is, binary);
      // initialize the buffer for gradients updates
      wei_gifo_x_bw_corr_ = wei_gifo_x_bw_; wei_gifo_x_bw_corr_.SetZero();
      wei_gifo_m_bw_corr_ = wei_gifo_m_bw_; wei_gifo_m_bw_corr_.SetZero();
      bias_bw_corr_ = bias_bw_; bias_bw_corr_.SetZero();
      phole_i_c_bw_corr_ = phole_i_c_bw_; phole_i_c_bw_corr_.SetZero();
      phole_f_c_bw_corr_ = phole_f_c_bw_; phole_f_c_bw_corr_.SetZero();
      phole_o_c_bw_corr_ = phole_o_c_bw_; phole_o_c_bw_corr_.SetZero();

    }

    void WriteData(std::ostream &os, bool binary) const {
      WriteToken(os, binary, "<LearnRateCoef>");
      WriteBasicType(os, binary, learn_rate_coef_);
      WriteToken(os, binary, "<MaxGrad>");
      WriteBasicType(os, binary, max_grad_);

      WriteToken(os, binary, "<ForwardDropoutFactor>");
      WriteBasicType(os, binary, forward_dropout);
      WriteToken(os, binary, "<ForwardTimeStepDropout>");
      WriteBasicType(os, binary, forward_step_dropout);
      WriteToken(os, binary, "<ForwardSequenceDropout>");
      WriteBasicType(os, binary, forward_sequence_dropout);

      WriteToken(os, binary, "<RecurrentTimeStepDropout>");
      WriteBasicType(os, binary, recurrent_step_dropout);
      WriteToken(os, binary, "<RecurrentSequenceDropout>");
      WriteBasicType(os, binary, recurrent_sequence_dropout);

      WriteToken(os, binary, "<RNNDrop>");
      WriteBasicType(os, binary, rnndrop);
      WriteToken(os, binary, "<NoMemLossDropout>");
      WriteBasicType(os, binary, no_mem_loss_dropout);

      WriteToken(os, binary, "<RecurrentDropoutFactor>");
      WriteBasicType(os, binary, recurrent_dropout);

      WriteToken(os, binary, "<TwiddleForward>");
      WriteBasicType(os, binary, twiddle_forward);

      if(adaBuffersInitialized)
      {
        WriteToken(os, binary, "<BiLstmAccus>");
        
        wei_gifo_x_fw_corr_accu.Write(os, binary);
        wei_gifo_m_fw_corr_accu.Write(os, binary);
        bias_fw_corr_accu.Write(os, binary);
        phole_i_c_fw_corr_accu.Write(os, binary);
        phole_f_c_fw_corr_accu.Write(os, binary);
        phole_o_c_fw_corr_accu.Write(os, binary);

        wei_gifo_x_bw_corr_accu.Write(os, binary);
        wei_gifo_m_bw_corr_accu.Write(os, binary);
        bias_bw_corr_accu.Write(os, binary);
        phole_i_c_bw_corr_accu.Write(os, binary);
        phole_f_c_bw_corr_accu.Write(os, binary);
        phole_o_c_bw_corr_accu.Write(os, binary);
        
      }
      
      // write parameters of the forward layer
      wei_gifo_x_fw_.Write(os, binary);
      wei_gifo_m_fw_.Write(os, binary);
      bias_fw_.Write(os, binary);
      phole_i_c_fw_.Write(os, binary);
      phole_f_c_fw_.Write(os, binary);
      phole_o_c_fw_.Write(os, binary);

      // write parameters of the backward layer
      wei_gifo_x_bw_.Write(os, binary);
      wei_gifo_m_bw_.Write(os, binary);
      bias_bw_.Write(os, binary);
      phole_i_c_bw_.Write(os, binary);
      phole_f_c_bw_.Write(os, binary);
      phole_o_c_bw_.Write(os, binary);
    }

    // print statistics of the parameters
    std::string Info() const {
        return std::string("    ") +
            "\n  wei_gifo_x_fw_  "   + MomentStatistics(wei_gifo_x_fw_) +
            "\n  wei_gifo_m_fw_  "   + MomentStatistics(wei_gifo_m_fw_) +
            "\n  bias_fw_  "         + MomentStatistics(bias_fw_) +
            "\n  phole_i_c_fw_  "      + MomentStatistics(phole_i_c_fw_) +
            "\n  phole_f_c_fw_  "      + MomentStatistics(phole_f_c_fw_) +
            "\n  phole_o_c_fw_  "      + MomentStatistics(phole_o_c_fw_) +
            "\n  wei_gifo_x_bw_  "   + MomentStatistics(wei_gifo_x_bw_) +
            "\n  wei_gifo_m_bw_  "   + MomentStatistics(wei_gifo_m_bw_) +
            "\n  bias_bw_  "         + MomentStatistics(bias_bw_) +
            "\n  phole_i_c_bw_  "      + MomentStatistics(phole_i_c_bw_) +
            "\n  phole_f_c_bw_  "      + MomentStatistics(phole_f_c_bw_) +
            "\n  phole_o_c_bw_  "      + MomentStatistics(phole_o_c_bw_);
    }

    // print statistics of the gradients buffer
    std::string InfoGradient() const {
        std::string extra = std::string("");
        if (adaBuffersInitialized)
        {
            extra += "\n  wei_gifo_x_fw_corr_accu  "   + MomentStatistics(wei_gifo_x_fw_corr_accu) +
            "\n  wei_gifo_m_fw_corr_accu  "   + MomentStatistics(wei_gifo_m_fw_corr_accu) +
            "\n  bias_fw_corr_accu  "         + MomentStatistics(bias_fw_corr_accu) +
            "\n  phole_i_c_fw_corr_accu  "      + MomentStatistics(phole_i_c_fw_corr_accu) +
            "\n  phole_f_c_fw_corr_accu  "      + MomentStatistics(phole_f_c_fw_corr_accu) +
            "\n  phole_o_c_fw_corr_accu  "      + MomentStatistics(phole_o_c_fw_corr_accu) +
            "\n  wei_gifo_x_bw_corr_accu  "   + MomentStatistics(wei_gifo_x_bw_corr_accu) +
            "\n  wei_gifo_m_bw_corr_accu  "   + MomentStatistics(wei_gifo_m_bw_corr_accu) +
            "\n  bias_bw_corr_accu  "         + MomentStatistics(bias_bw_corr_accu) +
            "\n  phole_i_c_bw_corr_accu  "      + MomentStatistics(phole_i_c_bw_corr_accu) +
            "\n  phole_f_c_bw_corr_accu  "      + MomentStatistics(phole_f_c_bw_corr_accu) +
            "\n  phole_o_c_bw_corr_accu  "      + MomentStatistics(phole_o_c_bw_corr_accu);          
        }

        return std::string("    ") +
            "\n  wei_gifo_x_fw_corr_  "   + MomentStatistics(wei_gifo_x_fw_corr_) +
            "\n  wei_gifo_m_fw_corr_  "   + MomentStatistics(wei_gifo_m_fw_corr_) +
            "\n  bias_fw_corr_  "         + MomentStatistics(bias_fw_corr_) +
            "\n  phole_i_c_fw_corr_  "      + MomentStatistics(phole_i_c_fw_corr_) +
            "\n  phole_f_c_fw_corr_  "      + MomentStatistics(phole_f_c_fw_corr_) +
            "\n  phole_o_c_fw_corr_  "      + MomentStatistics(phole_o_c_fw_corr_) +
            "\n  wei_gifo_x_bw_corr_  "   + MomentStatistics(wei_gifo_x_bw_corr_) +
            "\n  wei_gifo_m_bw_corr_  "   + MomentStatistics(wei_gifo_m_bw_corr_) +
            "\n  bias_bw_corr_  "         + MomentStatistics(bias_bw_corr_) +
            "\n  phole_i_c_bw_corr_  "      + MomentStatistics(phole_i_c_bw_corr_) +
            "\n  phole_f_c_bw_corr_  "      + MomentStatistics(phole_f_c_bw_corr_) +
            "\n  phole_o_c_bw_corr_  "      + MomentStatistics(phole_o_c_bw_corr_) + extra;
    }

    // the feedforward pass
    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
        int32 T = in.NumRows();  // total number of frames
        // resize & clear propagation buffers for the forward sub-layer. [0] - the initial states with all the values to be 0
        // [1, T] - correspond to the inputs  [T+1] - not used; for alignment with the backward layer
        propagate_buf_fw_.Resize(T + 2, 7 * cell_dim_, kSetZero);
        // resize & clear propagation buffers for the backward sub-layer
        propagate_buf_bw_.Resize(T + 2, 7 * cell_dim_, kSetZero);

        if (in_train && (forward_dropout > 0.0 || recurrent_dropout > 0.0))
          KALDI_ERR << "Dropout not implemented on BiLstm";

        // the forward layer
        if (1) {
          CuSubMatrix<BaseFloat> YG(propagate_buf_fw_.ColRange(0, cell_dim_));
          CuSubMatrix<BaseFloat> YI(propagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YF(propagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YO(propagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YC(propagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YH(propagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YM(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));

          CuSubMatrix<BaseFloat> YGIFO(propagate_buf_fw_.ColRange(0, 4 * cell_dim_));
          // no recurrence involved in the inputs
          YGIFO.RowRange(1,T).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_fw_, kTrans, 0.0);
          YGIFO.RowRange(1,T).AddVecToRows(1.0, bias_fw_);

          for (int t = 1; t <= T; t++) {
            // variables representing invidivual units/gates. we additionally use the Matrix forms
            // because we want to take advantage of the Mat.Sigmoid and Mat.Tanh function.
            CuSubVector<BaseFloat> y_g(YG.Row(t));  CuSubMatrix<BaseFloat> YG_t(YG.RowRange(t,1));
            CuSubVector<BaseFloat> y_i(YI.Row(t));  CuSubMatrix<BaseFloat> YI_t(YI.RowRange(t,1));
            CuSubVector<BaseFloat> y_f(YF.Row(t));  CuSubMatrix<BaseFloat> YF_t(YF.RowRange(t,1));
            CuSubVector<BaseFloat> y_o(YO.Row(t));  CuSubMatrix<BaseFloat> YO_t(YO.RowRange(t,1));
            CuSubVector<BaseFloat> y_c(YC.Row(t));  CuSubMatrix<BaseFloat> YC_t(YC.RowRange(t,1));
            CuSubVector<BaseFloat> y_h(YH.Row(t));  CuSubMatrix<BaseFloat> YH_t(YH.RowRange(t,1));
            CuSubVector<BaseFloat> y_m(YM.Row(t));  CuSubMatrix<BaseFloat> YM_t(YM.RowRange(t,1));
            CuSubVector<BaseFloat> y_gifo(YGIFO.Row(t));
            // add the recurrence of the previous memory cell to various gates/units
            y_gifo.AddMatVec(1.0, wei_gifo_m_fw_, kNoTrans, YM.Row(t-1), 1.0);
            // input gate
            y_i.AddVecVec(1.0, phole_i_c_fw_, YC.Row(t-1), 1.0);
            // forget gate
            y_f.AddVecVec(1.0, phole_f_c_fw_, YC.Row(t-1), 1.0);
            // apply sigmoid/tanh functionis to squash the outputs
            YI_t.Sigmoid(YI_t);
            YF_t.Sigmoid(YF_t);
            YG_t.Tanh(YG_t);

            // memory cell
            y_c.AddVecVec(1.0, y_i, y_g, 0.0);
            y_c.AddVecVec(1.0, y_f, YC.Row(t-1), 1.0);
            // h - the tanh-squashed version of c
            YH_t.Tanh(YC_t);

            // output gate
            y_o.AddVecVec(1.0, phole_o_c_fw_, y_c, 1.0);
            YO_t.Sigmoid(YO_t);

            // finally the outputs
            y_m.AddVecVec(1.0, y_o, y_h, 0.0);
          }  // end of loop t
        }  // end of the forward layer

        // the backward layer; follows the same procedures, but iterates from t=T to t=1
        if (1) {
          CuSubMatrix<BaseFloat> YG(propagate_buf_bw_.ColRange(0, cell_dim_));
          CuSubMatrix<BaseFloat> YI(propagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YF(propagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YO(propagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YC(propagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YH(propagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YM(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));

          CuSubMatrix<BaseFloat> YGIFO(propagate_buf_bw_.ColRange(0, 4 * cell_dim_));
          YGIFO.RowRange(1,T).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_bw_, kTrans, 0.0);
          YGIFO.RowRange(1,T).AddVecToRows(1.0, bias_bw_);

          for (int t = T; t >= 1; t--) {
            CuSubVector<BaseFloat> y_g(YG.Row(t));  CuSubMatrix<BaseFloat> YG_t(YG.RowRange(t,1));
            CuSubVector<BaseFloat> y_i(YI.Row(t));  CuSubMatrix<BaseFloat> YI_t(YI.RowRange(t,1));
            CuSubVector<BaseFloat> y_f(YF.Row(t));  CuSubMatrix<BaseFloat> YF_t(YF.RowRange(t,1));
            CuSubVector<BaseFloat> y_o(YO.Row(t));  CuSubMatrix<BaseFloat> YO_t(YO.RowRange(t,1));
            CuSubVector<BaseFloat> y_c(YC.Row(t));  CuSubMatrix<BaseFloat> YC_t(YC.RowRange(t,1));
            CuSubVector<BaseFloat> y_h(YH.Row(t));  CuSubMatrix<BaseFloat> YH_t(YH.RowRange(t,1));
            CuSubVector<BaseFloat> y_m(YM.Row(t));  CuSubMatrix<BaseFloat> YM_t(YM.RowRange(t,1));
            CuSubVector<BaseFloat> y_gifo(YGIFO.Row(t));

            y_gifo.AddMatVec(1.0, wei_gifo_m_bw_, kNoTrans, YM.Row(t+1), 1.0);
            // input gate
            y_i.AddVecVec(1.0, phole_i_c_bw_, YC.Row(t+1), 1.0);
            // forget gate
            y_f.AddVecVec(1.0, phole_f_c_bw_, YC.Row(t+1), 1.0);
            // apply sigmoid/tanh function
            YI_t.Sigmoid(YI_t);
            YF_t.Sigmoid(YF_t);
            YG_t.Tanh(YG_t);

            // memory cell
            y_c.AddVecVec(1.0, y_i, y_g, 0.0);
            y_c.AddVecVec(1.0, y_f, YC.Row(t+1), 1.0);
            // h, the tanh-squashed version of c
            YH_t.Tanh(YC_t);

            // output gate
            y_o.AddVecVec(1.0, phole_o_c_bw_, y_c, 1.0);
            YO_t.Sigmoid(YO_t);

            // the final output
            y_m.AddVecVec(1.0, y_o, y_h, 0.0);
          }
        }
        // final outputs now become the concatenation of the foward and backward activations
        CuMatrix<BaseFloat> YM_RB;
        YM_RB.Resize(T+2, 2 * cell_dim_, kSetZero);
        YM_RB.ColRange(0, cell_dim_).CopyFromMat(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));
        YM_RB.ColRange(cell_dim_, cell_dim_).CopyFromMat(propagate_buf_bw_.ColRange(6* cell_dim_, cell_dim_));

        out->CopyFromMat(YM_RB.RowRange(1,T));
    }

    // the back-propagation pass
    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                          const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        int32 T = in.NumRows();
        // initialize the back-propagation buffer
        backpropagate_buf_fw_.Resize(T + 2, 7 * cell_dim_, kSetZero);
        backpropagate_buf_bw_.Resize(T + 2, 7 * cell_dim_, kSetZero);

        if (1) {
          // get the activations of the gates/units from the feedforward buffer; these variabiles will be used
          // in gradients computation
          CuSubMatrix<BaseFloat> YG(propagate_buf_fw_.ColRange(0, cell_dim_));
          CuSubMatrix<BaseFloat> YI(propagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YF(propagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YO(propagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YC(propagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YH(propagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YM(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));

          // errors back-propagated to individual gates/units
          CuSubMatrix<BaseFloat> DG(backpropagate_buf_fw_.ColRange(0, cell_dim_));
          CuSubMatrix<BaseFloat> DI(backpropagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DF(backpropagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DO(backpropagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DC(backpropagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DH(backpropagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DM(backpropagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DGIFO(backpropagate_buf_fw_.ColRange(0, 4 * cell_dim_));

          // assume that the fist half of out_diff is about the forward layer
          DM.RowRange(1,T).CopyFromMat(out_diff.ColRange(0, cell_dim_));

          for (int t = T; t >= 1; t--) {
            // variables representing activations of invidivual units/gates
            CuSubVector<BaseFloat> y_g(YG.Row(t));  CuSubMatrix<BaseFloat> YG_t(YG.RowRange(t,1));
            CuSubVector<BaseFloat> y_i(YI.Row(t));  CuSubMatrix<BaseFloat> YI_t(YI.RowRange(t,1));
            CuSubVector<BaseFloat> y_f(YF.Row(t));  CuSubMatrix<BaseFloat> YF_t(YF.RowRange(t,1));
            CuSubVector<BaseFloat> y_o(YO.Row(t));  CuSubMatrix<BaseFloat> YO_t(YO.RowRange(t,1));
            CuSubVector<BaseFloat> y_c(YC.Row(t));  CuSubMatrix<BaseFloat> YC_t(YC.RowRange(t,1));
            CuSubVector<BaseFloat> y_h(YH.Row(t));  CuSubMatrix<BaseFloat> YH_t(YH.RowRange(t,1));
            CuSubVector<BaseFloat> y_m(YM.Row(t));  CuSubMatrix<BaseFloat> YM_t(YM.RowRange(t,1));
            // variables representing errors of invidivual units/gates
            CuSubVector<BaseFloat> d_g(DG.Row(t));  CuSubMatrix<BaseFloat> DG_t(DG.RowRange(t,1));
            CuSubVector<BaseFloat> d_i(DI.Row(t));  CuSubMatrix<BaseFloat> DI_t(DI.RowRange(t,1));
            CuSubVector<BaseFloat> d_f(DF.Row(t));  CuSubMatrix<BaseFloat> DF_t(DF.RowRange(t,1));
            CuSubVector<BaseFloat> d_o(DO.Row(t));  CuSubMatrix<BaseFloat> DO_t(DO.RowRange(t,1));
            CuSubVector<BaseFloat> d_c(DC.Row(t));  CuSubMatrix<BaseFloat> DC_t(DC.RowRange(t,1));
            CuSubVector<BaseFloat> d_h(DH.Row(t));  CuSubMatrix<BaseFloat> DH_t(DH.RowRange(t,1));
            CuSubVector<BaseFloat> d_m(DM.Row(t));  CuSubMatrix<BaseFloat> DM_t(DM.RowRange(t,1));

            // d_m comes from two parts: errors from the upper layer and errors from the following frame (t+1)
            d_m.AddMatVec(1.0, wei_gifo_m_fw_, kTrans, DGIFO.Row(t+1), 1.0);

            // d_h
            d_h.AddVecVec(1.0, y_o, d_m, 0.0);
            DH_t.DiffTanh(YH_t, DH_t);

            // d_o - output gate
            d_o.AddVecVec(1.0, y_h, d_m, 0.0);
            DO_t.DiffSigmoid(YO_t, DO_t);

            // d_c - memory cell
            d_c.AddVec(1.0, d_h, 0.0);
            d_c.AddVecVec(1.0, phole_o_c_fw_, d_o, 1.0);
            d_c.AddVecVec(1.0, YF.Row(t+1), DC.Row(t+1), 1.0);
            d_c.AddVecVec(1.0, phole_f_c_fw_, DF.Row(t+1), 1.0);
            d_c.AddVecVec(1.0, phole_i_c_fw_, DI.Row(t+1), 1.0);

            // d_f - forge gate
            d_f.AddVecVec(1.0, YC.Row(t-1), d_c, 0.0);
            DF_t.DiffSigmoid(YF_t, DF_t);

            // d_i - input gate
            d_i.AddVecVec(1.0, y_g, d_c, 0.0);
            DI_t.DiffSigmoid(YI_t, DI_t);

            // d_g
            d_g.AddVecVec(1.0, y_i, d_c, 0.0);
            DG_t.DiffTanh(YG_t, DG_t);
          } // end of t

          // errors back-propagated to the inputs
          in_diff->AddMatMat(1.0, DGIFO.RowRange(1,T), kNoTrans, wei_gifo_x_fw_, kNoTrans, 0.0);
          // updates to the model parameters
          const BaseFloat mmt = opts_.momentum;
          wei_gifo_x_fw_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, in, kNoTrans, mmt);
          wei_gifo_m_fw_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, YM.RowRange(0,T), kNoTrans, mmt);
          bias_fw_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1,T), mmt);
          phole_i_c_fw_corr_.AddDiagMatMat(1.0, DI.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);
          phole_f_c_fw_corr_.AddDiagMatMat(1.0, DF.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);
          phole_o_c_fw_corr_.AddDiagMatMat(1.0, DO.RowRange(1,T), kTrans, YC.RowRange(1,T), kNoTrans, mmt);
        } // end of the forward layer

        // back-propagation in the backward layer
        if (1) {
          // get the activations of the gates/units from the feedforward buffer
          CuSubMatrix<BaseFloat> YG(propagate_buf_bw_.ColRange(0, cell_dim_));
          CuSubMatrix<BaseFloat> YI(propagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YF(propagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YO(propagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YC(propagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YH(propagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YM(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));

          // errors back-propagated to individual gates/units
          CuSubMatrix<BaseFloat> DG(backpropagate_buf_bw_.ColRange(0, cell_dim_));
          CuSubMatrix<BaseFloat> DI(backpropagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DF(backpropagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DO(backpropagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DC(backpropagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DH(backpropagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DM(backpropagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DGIFO(backpropagate_buf_bw_.ColRange(0, 4 * cell_dim_));

          // the second half of the error vector corresponds to the backward layer
          DM.RowRange(1,T).CopyFromMat(out_diff.ColRange(cell_dim_, cell_dim_));

          for (int t = 1; t <= T; t++) {
            // variables representing activations of invidivual units/gates
            CuSubVector<BaseFloat> y_g(YG.Row(t));  CuSubMatrix<BaseFloat> YG_t(YG.RowRange(t,1));
            CuSubVector<BaseFloat> y_i(YI.Row(t));  CuSubMatrix<BaseFloat> YI_t(YI.RowRange(t,1));
            CuSubVector<BaseFloat> y_f(YF.Row(t));  CuSubMatrix<BaseFloat> YF_t(YF.RowRange(t,1));
            CuSubVector<BaseFloat> y_o(YO.Row(t));  CuSubMatrix<BaseFloat> YO_t(YO.RowRange(t,1));
            CuSubVector<BaseFloat> y_c(YC.Row(t));  CuSubMatrix<BaseFloat> YC_t(YC.RowRange(t,1));
            CuSubVector<BaseFloat> y_h(YH.Row(t));  CuSubMatrix<BaseFloat> YH_t(YH.RowRange(t,1));
            CuSubVector<BaseFloat> y_m(YM.Row(t));  CuSubMatrix<BaseFloat> YM_t(YM.RowRange(t,1));
            // variables representing errors of invidivual units/gates
            CuSubVector<BaseFloat> d_g(DG.Row(t));  CuSubMatrix<BaseFloat> DG_t(DG.RowRange(t,1));
            CuSubVector<BaseFloat> d_i(DI.Row(t));  CuSubMatrix<BaseFloat> DI_t(DI.RowRange(t,1));
            CuSubVector<BaseFloat> d_f(DF.Row(t));  CuSubMatrix<BaseFloat> DF_t(DF.RowRange(t,1));
            CuSubVector<BaseFloat> d_o(DO.Row(t));  CuSubMatrix<BaseFloat> DO_t(DO.RowRange(t,1));
            CuSubVector<BaseFloat> d_c(DC.Row(t));  CuSubMatrix<BaseFloat> DC_t(DC.RowRange(t,1));
            CuSubVector<BaseFloat> d_h(DH.Row(t));  CuSubMatrix<BaseFloat> DH_t(DH.RowRange(t,1));
            CuSubVector<BaseFloat> d_m(DM.Row(t));  CuSubMatrix<BaseFloat> DM_t(DM.RowRange(t,1));

            // d_m comes from two parts: errors from the upper layer and errors from the previous frame (t-1)
            d_m.AddMatVec(1.0, wei_gifo_m_bw_, kTrans, DGIFO.Row(t-1), 1.0);

            // d_h
            d_h.AddVecVec(1.0, y_o, d_m, 0.0);
            DH_t.DiffTanh(YH_t, DH_t);

            // d_o - output gate
            d_o.AddVecVec(1.0, y_h, d_m, 0.0);
            DO_t.DiffSigmoid(YO_t, DO_t);

            // d_c - memory cell
            d_c.AddVec(1.0, d_h, 0.0);
            d_c.AddVecVec(1.0, phole_o_c_bw_, d_o, 1.0);
            d_c.AddVecVec(1.0, YF.Row(t-1), DC.Row(t-1), 1.0);
            d_c.AddVecVec(1.0, phole_f_c_bw_, DF.Row(t-1), 1.0);
            d_c.AddVecVec(1.0, phole_i_c_bw_, DI.Row(t-1), 1.0);

            // d_f - forget gate
            d_f.AddVecVec(1.0, YC.Row(t+1), d_c, 0.0);
            DF_t.DiffSigmoid(YF_t, DF_t);

            // d_i - input gate
            d_i.AddVecVec(1.0, y_g, d_c, 0.0);
            DI_t.DiffSigmoid(YI_t, DI_t);

            // d_g
            d_g.AddVecVec(1.0, y_i, d_c, 0.0);
            DG_t.DiffTanh(YG_t, DG_t);
          }  // end of t

          // errors back-propagated to the inputs
          in_diff->AddMatMat(1.0, DGIFO.RowRange(1,T), kNoTrans, wei_gifo_x_bw_, kNoTrans, 1.0);
          // updates to the parameters
          const BaseFloat mmt = opts_.momentum;
          wei_gifo_x_bw_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, in, kNoTrans, mmt);
          wei_gifo_m_bw_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, YM.RowRange(0,T), kNoTrans, mmt);
          bias_bw_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1,T), mmt);
          phole_i_c_bw_corr_.AddDiagMatMat(1.0, DI.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);
          phole_f_c_bw_corr_.AddDiagMatMat(1.0, DF.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);
          phole_o_c_bw_corr_.AddDiagMatMat(1.0, DO.RowRange(1,T), kTrans, YC.RowRange(1,T), kNoTrans, mmt);
        } // end of the backward layer
    }

    void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff, 
    const UpdateRule rule=sgd_update) {
      if (max_grad_ > 0) {
        wei_gifo_x_fw_corr_.ApplyFloor(-max_grad_); wei_gifo_x_fw_corr_.ApplyCeiling(max_grad_);
        wei_gifo_m_fw_corr_.ApplyFloor(-max_grad_); wei_gifo_m_fw_corr_.ApplyCeiling(max_grad_);
        bias_fw_corr_.ApplyFloor(-max_grad_); bias_fw_corr_.ApplyCeiling(max_grad_);
        phole_i_c_fw_corr_.ApplyFloor(-max_grad_); phole_i_c_fw_corr_.ApplyCeiling(max_grad_);
        phole_f_c_fw_corr_.ApplyFloor(-max_grad_); phole_f_c_fw_corr_.ApplyCeiling(max_grad_);
        phole_o_c_fw_corr_.ApplyFloor(-max_grad_); phole_o_c_fw_corr_.ApplyCeiling(max_grad_);

        wei_gifo_x_bw_corr_.ApplyFloor(-max_grad_); wei_gifo_x_bw_corr_.ApplyCeiling(max_grad_);
        wei_gifo_m_bw_corr_.ApplyFloor(-max_grad_); wei_gifo_m_bw_corr_.ApplyCeiling(max_grad_);
        bias_bw_corr_.ApplyFloor(-max_grad_); bias_bw_corr_.ApplyCeiling(max_grad_);
        phole_i_c_bw_corr_.ApplyFloor(-max_grad_); phole_i_c_bw_corr_.ApplyCeiling(max_grad_);
        phole_f_c_bw_corr_.ApplyFloor(-max_grad_); phole_f_c_bw_corr_.ApplyCeiling(max_grad_);
        phole_o_c_bw_corr_.ApplyFloor(-max_grad_); phole_o_c_bw_corr_.ApplyCeiling(max_grad_);
      }

      // update parameters
      BaseFloat lr = opts_.learn_rate;

      if (rule==sgd_update) {

        lr *= learn_rate_coef_;

        wei_gifo_x_fw_.AddMat(-lr, wei_gifo_x_fw_corr_);
        wei_gifo_m_fw_.AddMat(-lr, wei_gifo_m_fw_corr_);
        bias_fw_.AddVec(-lr, bias_fw_corr_, 1.0);
        phole_i_c_fw_.AddVec(-lr, phole_i_c_fw_corr_, 1.0);
        phole_f_c_fw_.AddVec(-lr, phole_f_c_fw_corr_, 1.0);
        phole_o_c_fw_.AddVec(-lr, phole_o_c_fw_corr_, 1.0);

        wei_gifo_x_bw_.AddMat(-lr, wei_gifo_x_bw_corr_);
        wei_gifo_m_bw_.AddMat(-lr, wei_gifo_m_bw_corr_);
        bias_bw_.AddVec(-lr, bias_bw_corr_, 1.0);
        phole_i_c_bw_.AddVec(-lr, phole_i_c_bw_corr_, 1.0);
        phole_f_c_bw_.AddVec(-lr, phole_f_c_bw_corr_, 1.0);
        phole_o_c_bw_.AddVec(-lr, phole_o_c_bw_corr_, 1.0);

      } else if (rule==adagrad_update || rule==rmsprop_update) {

        if (!adaBuffersInitialized) {
          InitAdaBuffers();
        }

        if (rule==adagrad_update)
        {
          // update the accumolators for fw
          AdagradAccuUpdate(wei_gifo_x_fw_corr_accu, wei_gifo_x_fw_corr_, wei_gifo_x_fw_corr_accu_scale);
          AdagradAccuUpdate(wei_gifo_m_fw_corr_accu, wei_gifo_m_fw_corr_, wei_gifo_m_fw_corr_accu_scale);
          AdagradAccuUpdate(bias_fw_corr_accu, bias_fw_corr_, bias_fw_corr_accu_scale);
          AdagradAccuUpdate(phole_i_c_fw_corr_accu, phole_i_c_fw_corr_, phole_i_c_fw_corr_accu_scale);
          AdagradAccuUpdate(phole_f_c_fw_corr_accu, phole_f_c_fw_corr_, phole_f_c_fw_corr_accu_scale);
          AdagradAccuUpdate(phole_o_c_fw_corr_accu, phole_o_c_fw_corr_, phole_o_c_fw_corr_accu_scale);
          // update the accumolators for bw
          AdagradAccuUpdate(wei_gifo_x_bw_corr_accu, wei_gifo_x_bw_corr_, wei_gifo_x_bw_corr_accu_scale);
          AdagradAccuUpdate(wei_gifo_m_bw_corr_accu, wei_gifo_m_bw_corr_, wei_gifo_m_bw_corr_accu_scale);
          AdagradAccuUpdate(bias_bw_corr_accu, bias_bw_corr_, bias_bw_corr_accu_scale);
          AdagradAccuUpdate(phole_i_c_bw_corr_accu, phole_i_c_bw_corr_, phole_i_c_bw_corr_accu_scale);
          AdagradAccuUpdate(phole_f_c_bw_corr_accu, phole_f_c_bw_corr_, phole_f_c_bw_corr_accu_scale);
          AdagradAccuUpdate(phole_o_c_bw_corr_accu, phole_o_c_bw_corr_, phole_o_c_bw_corr_accu_scale);
        }else {
          // update the accumolators for fw
          RMSPropAccuUpdate(wei_gifo_x_fw_corr_accu, wei_gifo_x_fw_corr_, wei_gifo_x_fw_corr_accu_scale);
          RMSPropAccuUpdate(wei_gifo_m_fw_corr_accu, wei_gifo_m_fw_corr_, wei_gifo_m_fw_corr_accu_scale);
          RMSPropAccuUpdate(bias_fw_corr_accu, bias_fw_corr_, bias_fw_corr_accu_scale);
          RMSPropAccuUpdate(phole_i_c_fw_corr_accu, phole_i_c_fw_corr_, phole_i_c_fw_corr_accu_scale);
          RMSPropAccuUpdate(phole_f_c_fw_corr_accu, phole_f_c_fw_corr_, phole_f_c_fw_corr_accu_scale);
          RMSPropAccuUpdate(phole_o_c_fw_corr_accu, phole_o_c_fw_corr_, phole_o_c_fw_corr_accu_scale);
          // update the accumolators for bw
          RMSPropAccuUpdate(wei_gifo_x_bw_corr_accu, wei_gifo_x_bw_corr_, wei_gifo_x_bw_corr_accu_scale);
          RMSPropAccuUpdate(wei_gifo_m_bw_corr_accu, wei_gifo_m_bw_corr_, wei_gifo_m_bw_corr_accu_scale);
          RMSPropAccuUpdate(bias_bw_corr_accu, bias_bw_corr_, bias_bw_corr_accu_scale);
          RMSPropAccuUpdate(phole_i_c_bw_corr_accu, phole_i_c_bw_corr_, phole_i_c_bw_corr_accu_scale);
          RMSPropAccuUpdate(phole_f_c_bw_corr_accu, phole_f_c_bw_corr_, phole_f_c_bw_corr_accu_scale);
          RMSPropAccuUpdate(phole_o_c_bw_corr_accu, phole_o_c_bw_corr_, phole_o_c_bw_corr_accu_scale);
        }

        // calculate 1.0 / sqrt(accu + epsilon) for fw
        AdagradScaleCompute(wei_gifo_x_fw_corr_accu_scale,wei_gifo_x_fw_corr_accu);
        AdagradScaleCompute(wei_gifo_m_fw_corr_accu_scale,wei_gifo_m_fw_corr_accu);
        AdagradScaleCompute(bias_fw_corr_accu_scale,bias_fw_corr_accu);
        AdagradScaleCompute(phole_i_c_fw_corr_accu_scale,phole_i_c_fw_corr_accu);
        AdagradScaleCompute(phole_f_c_fw_corr_accu_scale,phole_f_c_fw_corr_accu);
        AdagradScaleCompute(phole_o_c_fw_corr_accu_scale,phole_o_c_fw_corr_accu);

         // calculate 1.0 / sqrt(accu + epsilon) for bw
        AdagradScaleCompute(wei_gifo_x_bw_corr_accu_scale,wei_gifo_x_bw_corr_accu);
        AdagradScaleCompute(wei_gifo_m_bw_corr_accu_scale,wei_gifo_m_bw_corr_accu);
        AdagradScaleCompute(bias_bw_corr_accu_scale,bias_bw_corr_accu);
        AdagradScaleCompute(phole_i_c_bw_corr_accu_scale,phole_i_c_bw_corr_accu);
        AdagradScaleCompute(phole_f_c_bw_corr_accu_scale,phole_f_c_bw_corr_accu);
        AdagradScaleCompute(phole_o_c_bw_corr_accu_scale,phole_o_c_bw_corr_accu);

        // update the parameters for fw
        wei_gifo_x_fw_.AddMatMatElements(-lr, wei_gifo_x_fw_corr_accu_scale, wei_gifo_x_fw_corr_, 1.0);
        wei_gifo_m_fw_.AddMatMatElements(-lr, wei_gifo_m_fw_corr_accu_scale, wei_gifo_m_fw_corr_, 1.0);
        bias_fw_.AddVecVec(-lr, bias_fw_corr_accu_scale, bias_fw_corr_, 1.0);
        phole_i_c_fw_.AddVecVec(-lr, phole_i_c_fw_corr_accu_scale, phole_i_c_fw_corr_, 1.0);
        phole_f_c_fw_.AddVecVec(-lr, phole_f_c_fw_corr_accu_scale, phole_f_c_fw_corr_, 1.0);
        phole_o_c_fw_.AddVecVec(-lr, phole_o_c_fw_corr_accu_scale, phole_o_c_fw_corr_, 1.0);

        // update the parameters for bw
        wei_gifo_x_bw_.AddMatMatElements(-lr, wei_gifo_x_bw_corr_accu_scale, wei_gifo_x_bw_corr_, 1.0);
        wei_gifo_m_bw_.AddMatMatElements(-lr, wei_gifo_m_bw_corr_accu_scale, wei_gifo_m_bw_corr_, 1.0);
        bias_bw_.AddVecVec(-lr, bias_bw_corr_accu_scale, bias_bw_corr_, 1.0);
        phole_i_c_bw_.AddVecVec(-lr, phole_i_c_bw_corr_accu_scale, phole_i_c_bw_corr_, 1.0);
        phole_f_c_bw_.AddVecVec(-lr, phole_f_c_bw_corr_accu_scale, phole_f_c_bw_corr_, 1.0);
        phole_o_c_bw_.AddVecVec(-lr, phole_o_c_bw_corr_accu_scale, phole_o_c_bw_corr_, 1.0);
      }
    }

    void Scale(BaseFloat scale) {
      wei_gifo_x_fw_.Scale(scale);
      wei_gifo_m_fw_.Scale(scale);
      bias_fw_.Scale(scale);
      phole_i_c_fw_.Scale(scale);
      phole_f_c_fw_.Scale(scale);
      phole_o_c_fw_.Scale(scale);

      wei_gifo_x_bw_.Scale(scale);
      wei_gifo_m_bw_.Scale(scale);
      bias_bw_.Scale(scale);
      phole_i_c_bw_.Scale(scale);
      phole_f_c_bw_.Scale(scale);
      phole_o_c_bw_.Scale(scale);
    }

    void Add(BaseFloat scale, const TrainableLayer & layer_other) {
      const BiLstm *other = dynamic_cast<const BiLstm*>(&layer_other);
      wei_gifo_x_fw_.AddMat(scale, other->wei_gifo_x_fw_);
      wei_gifo_m_fw_.AddMat(scale, other->wei_gifo_m_fw_);
      bias_fw_.AddVec(scale, other->bias_fw_);
      phole_i_c_fw_.AddVec(scale, other->phole_i_c_fw_);
      phole_f_c_fw_.AddVec(scale, other->phole_f_c_fw_);
      phole_o_c_fw_.AddVec(scale, other->phole_o_c_fw_);

      wei_gifo_x_bw_.AddMat(scale, other->wei_gifo_x_bw_);
      wei_gifo_m_bw_.AddMat(scale, other->wei_gifo_m_bw_);
      bias_bw_.AddVec(scale, other->bias_bw_);
      phole_i_c_bw_.AddVec(scale, other->phole_i_c_bw_);
      phole_f_c_bw_.AddVec(scale, other->phole_f_c_bw_);
      phole_o_c_bw_.AddVec(scale, other->phole_o_c_bw_);
    }

    int32 NumParams() const {
      return 2 * ( wei_gifo_x_fw_.NumRows() * wei_gifo_x_fw_.NumCols() +
                   wei_gifo_m_fw_.NumRows() * wei_gifo_m_fw_.NumCols() +
                   bias_fw_.Dim() +
                   phole_i_c_fw_.Dim() +
                   phole_f_c_fw_.Dim() +
                   phole_o_c_fw_.Dim() );
    }

    void GetParams(Vector<BaseFloat>* wei_copy) const {
      wei_copy->Resize(NumParams());
      int32 offset = 0, size;
      // copy parameters of the forward sub-layer
      size = wei_gifo_x_fw_.NumRows() * wei_gifo_x_fw_.NumCols();
      wei_copy->Range(offset, size).CopyRowsFromMat(wei_gifo_x_fw_); offset += size;
      size = wei_gifo_m_fw_.NumRows() * wei_gifo_m_fw_.NumCols();
      wei_copy->Range(offset, size).CopyRowsFromMat(wei_gifo_m_fw_); offset += size;
      size = bias_fw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(bias_fw_); offset += size;
      size = phole_i_c_fw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_i_c_fw_); offset += size;
      size = phole_f_c_fw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_f_c_fw_); offset += size;
      size = phole_o_c_fw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_o_c_fw_); offset += size;

      // copy parameters of the backward sub-layer
      size = wei_gifo_x_bw_.NumRows() * wei_gifo_x_bw_.NumCols();
      wei_copy->Range(offset, size).CopyRowsFromMat(wei_gifo_x_bw_); offset += size;
      size = wei_gifo_m_bw_.NumRows() * wei_gifo_m_bw_.NumCols();
      wei_copy->Range(offset, size).CopyRowsFromMat(wei_gifo_m_bw_); offset += size;
      size = bias_bw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(bias_bw_); offset += size;
      size = phole_i_c_bw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_i_c_bw_); offset += size;
      size = phole_f_c_bw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_f_c_bw_); offset += size;
      size = phole_o_c_bw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_o_c_bw_); offset += size;
    }

//private:
protected:
    int32 cell_dim_;
    BaseFloat learn_rate_coef_;
    BaseFloat max_grad_;
    BaseFloat drop_factor_;
    bool adaBuffersInitialized;

    BaseFloat forward_dropout;
    bool forward_step_dropout;
    bool forward_sequence_dropout;

    bool recurrent_step_dropout;
    bool recurrent_sequence_dropout;

    bool rnndrop;
    bool no_mem_loss_dropout;

    BaseFloat recurrent_dropout;

    bool twiddle_forward; // twiddle forward if true

    bool twiddle_apply_forward;

    bool in_train; // are we training or testing model (impacts dropout)

    CuMatrix<BaseFloat> forward_drop_mask_;
    Matrix<BaseFloat>   forward_drop_mask_cpu_;

    CuMatrix<BaseFloat> recurrent_drop_mask_fw_;
    CuMatrix<BaseFloat> recurrent_drop_mask_bw_;
    Matrix<BaseFloat> recurrent_drop_mask_cpu_;

    CuMatrix<BaseFloat> d_h_mask;
    CuMatrix<BaseFloat> d_c_mask;

    // parameters of the forward layer
    CuMatrix<BaseFloat> wei_gifo_x_fw_;
    CuMatrix<BaseFloat> wei_gifo_m_fw_;
    CuVector<BaseFloat> bias_fw_;
    CuVector<BaseFloat> phole_i_c_fw_;
    CuVector<BaseFloat> phole_f_c_fw_;
    CuVector<BaseFloat> phole_o_c_fw_;
    // the corresponding parameter updates
    CuMatrix<BaseFloat> wei_gifo_x_fw_corr_;
    CuMatrix<BaseFloat> wei_gifo_m_fw_corr_;
    CuVector<BaseFloat> bias_fw_corr_;
    CuVector<BaseFloat> phole_i_c_fw_corr_;
    CuVector<BaseFloat> phole_f_c_fw_corr_;
    CuVector<BaseFloat> phole_o_c_fw_corr_;

    // parameters of the backward layer
    CuMatrix<BaseFloat> wei_gifo_x_bw_;
    CuMatrix<BaseFloat> wei_gifo_m_bw_;
    CuVector<BaseFloat> bias_bw_;
    CuVector<BaseFloat> phole_i_c_bw_;
    CuVector<BaseFloat> phole_f_c_bw_;
    CuVector<BaseFloat> phole_o_c_bw_;
    // the corresponding parameter updates
    CuMatrix<BaseFloat> wei_gifo_x_bw_corr_;
    CuMatrix<BaseFloat> wei_gifo_m_bw_corr_;
    CuVector<BaseFloat> bias_bw_corr_;
    CuVector<BaseFloat> phole_i_c_bw_corr_;
    CuVector<BaseFloat> phole_f_c_bw_corr_;
    CuVector<BaseFloat> phole_o_c_bw_corr_;

    // fw accumolators for e.g. AdaGrad
    CuMatrix<BaseFloat> wei_gifo_x_fw_corr_accu;
    CuMatrix<BaseFloat> wei_gifo_m_fw_corr_accu;
    CuVector<BaseFloat> bias_fw_corr_accu;
    CuVector<BaseFloat> phole_i_c_fw_corr_accu;
    CuVector<BaseFloat> phole_f_c_fw_corr_accu;
    CuVector<BaseFloat> phole_o_c_fw_corr_accu;

    // for fw scale computation, e.g. AdaGrad
    CuMatrix<BaseFloat> wei_gifo_x_fw_corr_accu_scale;
    CuMatrix<BaseFloat> wei_gifo_m_fw_corr_accu_scale;
    CuVector<BaseFloat> bias_fw_corr_accu_scale;
    CuVector<BaseFloat> phole_i_c_fw_corr_accu_scale;
    CuVector<BaseFloat> phole_f_c_fw_corr_accu_scale;
    CuVector<BaseFloat> phole_o_c_fw_corr_accu_scale;

    // bw accumolators for e.g. AdaGrad
    CuMatrix<BaseFloat> wei_gifo_x_bw_corr_accu;
    CuMatrix<BaseFloat> wei_gifo_m_bw_corr_accu;
    CuVector<BaseFloat> bias_bw_corr_accu;
    CuVector<BaseFloat> phole_i_c_bw_corr_accu;
    CuVector<BaseFloat> phole_f_c_bw_corr_accu;
    CuVector<BaseFloat> phole_o_c_bw_corr_accu;

    // for bw scale computation, e.g. AdaGrad
    CuMatrix<BaseFloat> wei_gifo_x_bw_corr_accu_scale;
    CuMatrix<BaseFloat> wei_gifo_m_bw_corr_accu_scale;
    CuVector<BaseFloat> bias_bw_corr_accu_scale;
    CuVector<BaseFloat> phole_i_c_bw_corr_accu_scale;
    CuVector<BaseFloat> phole_f_c_bw_corr_accu_scale;
    CuVector<BaseFloat> phole_o_c_bw_corr_accu_scale;

    // propagation buffer
    CuMatrix<BaseFloat> propagate_buf_fw_;
    CuMatrix<BaseFloat> propagate_buf_bw_;

    // back-propagation buffer
    CuMatrix<BaseFloat> backpropagate_buf_fw_;
    CuMatrix<BaseFloat> backpropagate_buf_bw_;

};

} // namespace eesen

#endif
