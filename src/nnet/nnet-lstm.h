// nnet/nnet-lstm.h

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

#ifndef KALDI_NNET_LSTM_H_
#define KALDI_NNET_LSTM_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {
class Lstm : public UpdatableComponent {
public:
    Lstm(int32 input_dim, int32 output_dim) :
        UpdatableComponent(input_dim, output_dim),
        cell_dim_(output_dim),
        learn_rate_coef_(1.0), bias_learn_rate_coef_(1.0), max_grad_(0.0)
    { }

    ~Lstm()
    { }

    Component* Copy() const { return new Lstm(*this); }
    ComponentType GetType() const { return kLstm; }
    ComponentType GetTypeNonParal() const { return kLstm; }   
 
    void InitData(std::istream &is) {
      // define options
      float param_range = 0.02, max_grad = 0.0;
      float learn_rate_coef = 1.0, bias_learn_rate_coef = 1.0;
      // parse config
      std::string token;
      while (!is.eof()) {
        ReadToken(is, false, &token);
        if (token == "<ParamRange>")  ReadBasicType(is, false, &param_range);
        else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
        else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef);
        else if (token == "<MaxGrad>") ReadBasicType(is, false, &max_grad);
        else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                       << " (ParamRange|LearnRateCoef|BiasLearnRateCoef|MaxGrad)";
        is >> std::ws; // eat-up whitespace
      }

      // initialize weights and biases
      wei_gifo_x_.Resize(4 * cell_dim_, input_dim_); wei_gifo_x_.InitRandUniform(param_range);
      // the weights connecting momory cell outputs with the units/gates
      wei_gifo_m_.Resize(4 * cell_dim_, cell_dim_);  wei_gifo_m_.InitRandUniform(param_range);
      // the bias for the units/gates
      bias_.Resize(4 * cell_dim_); bias_.InitRandUniform(param_range);
      // peephole connections for i, f, and o, with diagonal matrices (vectors)
      phole_i_c_.Resize(cell_dim_); phole_i_c_.InitRandUniform(param_range);
      phole_f_c_.Resize(cell_dim_); phole_f_c_.InitRandUniform(param_range);
      phole_o_c_.Resize(cell_dim_); phole_o_c_.InitRandUniform(param_range);

      //
      learn_rate_coef_ = learn_rate_coef; bias_learn_rate_coef_ = bias_learn_rate_coef;
      max_grad_ = max_grad;
    }

    void ReadData(std::istream &is, bool binary) {
      // optional learning-rate coefs
      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<LearnRateCoef>");
        ReadBasicType(is, binary, &learn_rate_coef_);
        ExpectToken(is, binary, "<BiasLearnRateCoef>");
        ReadBasicType(is, binary, &bias_learn_rate_coef_);
      }
      if ('<' == Peek(is, binary)) {
        ExpectToken(is, binary, "<MaxGrad>");
        ReadBasicType(is, binary, &max_grad_);
      }

      // read parameters
      wei_gifo_x_.Read(is, binary);
      wei_gifo_m_.Read(is, binary);
      bias_.Read(is, binary);
      phole_i_c_.Read(is, binary);
      phole_f_c_.Read(is, binary);
      phole_o_c_.Read(is, binary);
      // initialize the buffer for gradients updates
      wei_gifo_x_corr_ = wei_gifo_x_; wei_gifo_x_corr_.SetZero();
      wei_gifo_m_corr_ = wei_gifo_m_; wei_gifo_m_corr_.SetZero();
      bias_corr_ = bias_; bias_corr_.SetZero();
      phole_i_c_corr_ = phole_i_c_; phole_i_c_corr_.SetZero();
      phole_f_c_corr_ = phole_f_c_; phole_f_c_corr_.SetZero();
      phole_o_c_corr_ = phole_o_c_; phole_o_c_corr_.SetZero();
    }

    void WriteData(std::ostream &os, bool binary) const {
      WriteToken(os, binary, "<LearnRateCoef>");
      WriteBasicType(os, binary, learn_rate_coef_);
      WriteToken(os, binary, "<BiasLearnRateCoef>");
      WriteBasicType(os, binary, bias_learn_rate_coef_);
      WriteToken(os, binary, "<MaxGrad>");
      WriteBasicType(os, binary, max_grad_);

      // write parameters of the forward layer
      wei_gifo_x_.Write(os, binary);
      wei_gifo_m_.Write(os, binary);
      bias_.Write(os, binary);
      phole_i_c_.Write(os, binary);
      phole_f_c_.Write(os, binary);
      phole_o_c_.Write(os, binary);
    }

    // print statistics of the parameters
    std::string Info() const {
        return std::string("    ") + 
            "\n  wei_gifo_x_  "   + MomentStatistics(wei_gifo_x_) + 
            "\n  wei_gifo_m_  "   + MomentStatistics(wei_gifo_m_) +
            "\n  bias_  "         + MomentStatistics(bias_) +
            "\n  phole_i_c_  "      + MomentStatistics(phole_i_c_) +
            "\n  phole_f_c_  "      + MomentStatistics(phole_f_c_) +
            "\n  phole_o_c_  "      + MomentStatistics(phole_o_c_);
    }
  
    // print statistics of the gradients buffer
    std::string InfoGradient() const {
        return std::string("    ") +
            "\n  wei_gifo_x_corr_  "   + MomentStatistics(wei_gifo_x_corr_) +
            "\n  wei_gifo_m_corr_  "   + MomentStatistics(wei_gifo_m_corr_) +
            "\n  bias_corr_  "         + MomentStatistics(bias_corr_) +
            "\n  phole_i_c_corr_  "      + MomentStatistics(phole_i_c_corr_) +
            "\n  phole_f_c_corr_  "      + MomentStatistics(phole_f_c_corr_) +
            "\n  phole_o_c_corr_  "      + MomentStatistics(phole_o_c_corr_);
    }

    // the feedforward pass
    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
        int32 T = in.NumRows();  // total number of frames
        // resize & clear propagation buffers. [0] - the initial states with all the values to be 0
        // [1, T] - correspond to the inputs  [T+1] - not used; for alignment with the backward layer 
        propagate_buf_.Resize(T + 2, 7 * cell_dim_, kSetZero);

        CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0, cell_dim_));
        CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6 * cell_dim_, cell_dim_));

        CuSubMatrix<BaseFloat> YGIFO(propagate_buf_.ColRange(0, 4 * cell_dim_));
        // no recurrence involved in the inputs
        YGIFO.RowRange(1,T).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_, kTrans, 0.0);
        YGIFO.RowRange(1,T).AddVecToRows(1.0, bias_);

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
          y_gifo.AddMatVec(1.0, wei_gifo_m_, kNoTrans, YM.Row(t-1), 1.0);
          // input gate
          y_i.AddVecVec(1.0, phole_i_c_, YC.Row(t-1), 1.0);
          // forget gate
          y_f.AddVecVec(1.0, phole_f_c_, YC.Row(t-1), 1.0);
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
          y_o.AddVecVec(1.0, phole_o_c_, y_c, 1.0);
          YO_t.Sigmoid(YO_t);

          // finally the outputs
          y_m.AddVecVec(1.0, y_o, y_h, 0.0);
        }  // end of loop t

        out->CopyFromMat(YM.RowRange(1,T));
    }

    // the back-propagation pass
    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                          const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        int32 T = in.NumRows();
        // initialize the back-propagation buffer
        backpropagate_buf_.Resize(T + 2, 7 * cell_dim_, kSetZero);

        // get the activations of the gates/units from the feedforward buffer; these variabiles will be used
        // in gradients computation
        CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0, cell_dim_));
        CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6 * cell_dim_, cell_dim_));
    
        // errors back-propagated to individual gates/units
        CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(0, cell_dim_));
        CuSubMatrix<BaseFloat> DI(backpropagate_buf_.ColRange(1 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DF(backpropagate_buf_.ColRange(2 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DO(backpropagate_buf_.ColRange(3 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(4 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(5 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(6 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DGIFO(backpropagate_buf_.ColRange(0, 4 * cell_dim_));

        DM.RowRange(1,T).CopyFromMat(out_diff);

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
          d_m.AddMatVec(1.0, wei_gifo_m_, kTrans, DGIFO.Row(t+1), 1.0);
    
          // d_h
          d_h.AddVecVec(1.0, y_o, d_m, 0.0);
          DH_t.DiffTanh(YH_t, DH_t);
    
          // d_o - output gate
          d_o.AddVecVec(1.0, y_h, d_m, 0.0);
          DO_t.DiffSigmoid(YO_t, DO_t);
    
          // d_c - memory cell
          d_c.AddVec(1.0, d_h, 0.0);  
          d_c.AddVecVec(1.0, phole_o_c_, d_o, 1.0);
          d_c.AddVecVec(1.0, YF.Row(t+1), DC.Row(t+1), 1.0);
          d_c.AddVecVec(1.0, phole_f_c_, DF.Row(t+1), 1.0);
          d_c.AddVecVec(1.0, phole_i_c_, DI.Row(t+1), 1.0);
    
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
        in_diff->AddMatMat(1.0, DGIFO.RowRange(1,T), kNoTrans, wei_gifo_x_, kNoTrans, 0.0);
        // updates to the model parameters 
        const BaseFloat mmt = opts_.momentum;
        wei_gifo_x_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, in, kNoTrans, mmt);
        wei_gifo_m_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, YM.RowRange(0,T), kNoTrans, mmt);
        bias_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1,T), mmt);
        phole_i_c_corr_.AddDiagMatMat(1.0, DI.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);
        phole_f_c_corr_.AddDiagMatMat(1.0, DF.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);
        phole_o_c_corr_.AddDiagMatMat(1.0, DO.RowRange(1,T), kTrans, YC.RowRange(1,T), kNoTrans, mmt);
    }

    void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
      // clip gradients 
      if (max_grad_ > 0) {
        wei_gifo_x_corr_.ApplyFloor(-max_grad_); wei_gifo_x_corr_.ApplyCeiling(max_grad_);
        wei_gifo_m_corr_.ApplyFloor(-max_grad_); wei_gifo_m_corr_.ApplyCeiling(max_grad_);
        bias_corr_.ApplyFloor(-max_grad_); bias_corr_.ApplyCeiling(max_grad_);
        phole_i_c_corr_.ApplyFloor(-max_grad_); phole_i_c_corr_.ApplyCeiling(max_grad_);
        phole_f_c_corr_.ApplyFloor(-max_grad_); phole_f_c_corr_.ApplyCeiling(max_grad_);
        phole_o_c_corr_.ApplyFloor(-max_grad_); phole_o_c_corr_.ApplyCeiling(max_grad_);
      }

      // update parameters
      const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
      const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;

      wei_gifo_x_.AddMat(-lr, wei_gifo_x_corr_);
      wei_gifo_m_.AddMat(-lr, wei_gifo_m_corr_);
      bias_.AddVec(-lr_bias, bias_corr_, 1.0);
      phole_i_c_.AddVec(-lr, phole_i_c_corr_, 1.0);
      phole_f_c_.AddVec(-lr, phole_f_c_corr_, 1.0);
      phole_o_c_.AddVec(-lr, phole_o_c_corr_, 1.0);
    }

    int32 NumParams() const {
      return wei_gifo_x_.NumRows() * wei_gifo_x_.NumCols() +
             wei_gifo_m_.NumRows() * wei_gifo_m_.NumCols() +
             bias_.Dim() +
             phole_i_c_.Dim() +
             phole_f_c_.Dim() +
             phole_o_c_.Dim();
    }

    void GetParams(Vector<BaseFloat>* wei_copy) const {
      wei_copy->Resize(NumParams());
      int32 offset = 0, size;
      // copy parameters of the forward sub-layer
      size = wei_gifo_x_.NumRows() * wei_gifo_x_.NumCols();
      wei_copy->Range(offset, size).CopyRowsFromMat(wei_gifo_x_); offset += size;
      size = wei_gifo_m_.NumRows() * wei_gifo_m_.NumCols();
      wei_copy->Range(offset, size).CopyRowsFromMat(wei_gifo_m_); offset += size;
      size = bias_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(bias_); offset += size;
      size = phole_i_c_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_i_c_); offset += size;
      size = phole_f_c_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_f_c_); offset += size;
      size = phole_o_c_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_o_c_); offset += size;
    }

//private:
protected:
    int32 cell_dim_;
    BaseFloat learn_rate_coef_;
    BaseFloat bias_learn_rate_coef_;
    BaseFloat max_grad_;

    // parameters of the forward layer
    CuMatrix<BaseFloat> wei_gifo_x_;
    CuMatrix<BaseFloat> wei_gifo_m_;
    CuVector<BaseFloat> bias_;
    CuVector<BaseFloat> phole_i_c_;
    CuVector<BaseFloat> phole_f_c_;
    CuVector<BaseFloat> phole_o_c_;
    // the corresponding parameter updates
    CuMatrix<BaseFloat> wei_gifo_x_corr_;
    CuMatrix<BaseFloat> wei_gifo_m_corr_;
    CuVector<BaseFloat> bias_corr_;
    CuVector<BaseFloat> phole_i_c_corr_;
    CuVector<BaseFloat> phole_f_c_corr_;
    CuVector<BaseFloat> phole_o_c_corr_;

    // propagation buffer
    CuMatrix<BaseFloat> propagate_buf_;

    // back-propagation buffer
    CuMatrix<BaseFloat> backpropagate_buf_;

};
} // namespace nnet1
} // namespace kaldi

#endif
