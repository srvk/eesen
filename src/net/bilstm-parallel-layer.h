// net/bilstm-parallel-layer.h

// Copyright 2015  Yajie Miao

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

#ifndef EESEN_BILSTM_PARALLEL_LAYER_H_
#define EESEN_BILSTM_PARALLEL_LAYER_H_

#include "net/layer.h"
#include "net/trainable-layer.h"
#include "net/bilstm-layer.h"
#include "net/utils-functions.h"
#include "gpucompute/cuda-math.h"

namespace eesen {

class BiLstmParallel : public BiLstm {
public:
    BiLstmParallel(int32 input_dim, int32 output_dim) : BiLstm(input_dim, output_dim)
    { }
    ~BiLstmParallel()
    { }

    Layer* Copy() const { return new BiLstmParallel(*this); }
    LayerType GetType() const { return l_BiLstm_Parallel; }
    LayerType GetTypeNonParal() const { return l_BiLstm; }

    void SetSeqLengths(std::vector<int> &sequence_lengths) {
        sequence_lengths_ = sequence_lengths;
    }

    void InitializeForwardMask(int32 T, int32 S) {

      if (!in_train) return;

      // create the mask on the CPU and copy it over (GPU version was significantly slower by factor of 4-5)
      forward_drop_mask_cpu_.Resize(T*S, 2 * cell_dim_, kUndefined);
      forward_drop_mask_.Resize(T*S, 2 * cell_dim_, kUndefined);

      if (forward_step_dropout)
        forward_drop_mask_cpu_.SetRandUniform();
      else if (forward_sequence_dropout)
        forward_drop_mask_cpu_.SetRandUniformCol();

      forward_drop_mask_cpu_.Add(-forward_dropout);
      forward_drop_mask_cpu_.ApplyHeaviside();
      forward_drop_mask_cpu_.Scale(1.0/(1.0-forward_dropout)); // scale mask
      forward_drop_mask_.CopyFromMat(forward_drop_mask_cpu_);

    }

    void InitializeRecurrentMasks(int32 T, int32 S) {

      if (!in_train) return;

      // Set the row size based on whether sequence or step dropout selected
      // assumes that if sequence_dropout is true step is false.
      int mask_row_size = recurrent_sequence_dropout? S: (T+2)*S;

      if (recurrent_dropout != 0.0 && (rnndrop || no_mem_loss_dropout)) {
          recurrent_drop_mask_fw_.Resize(mask_row_size, cell_dim_, kUndefined);
          recurrent_drop_mask_bw_.Resize(mask_row_size, cell_dim_, kUndefined);
          recurrent_drop_mask_cpu_.Resize(mask_row_size, 2*cell_dim_, kUndefined);

        if (recurrent_sequence_dropout)
          recurrent_drop_mask_cpu_.SetRandUniformCol();
        else
          recurrent_drop_mask_cpu_.SetRandUniform();

        recurrent_drop_mask_cpu_.Add(-recurrent_dropout);
        recurrent_drop_mask_cpu_.ApplyHeaviside();
        // scale the masks so as to not need it during test.
        recurrent_drop_mask_cpu_.Scale(1.0 /(1.0-recurrent_dropout));
        //forward cells mask
        recurrent_drop_mask_fw_.CopyFromMat(recurrent_drop_mask_cpu_.ColRange(0, cell_dim_));
        //backward cells mask
        recurrent_drop_mask_bw_.CopyFromMat(recurrent_drop_mask_cpu_.ColRange(cell_dim_, cell_dim_));
      }

    }


    void PropagateFncVanillaPassForward(const CuMatrixBase<BaseFloat> &in, int32 T, int32 S) {

      CuSubMatrix<BaseFloat> YG(propagate_buf_fw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> YI(propagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YF(propagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YO(propagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YC(propagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YH(propagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YM(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));

      CuSubMatrix<BaseFloat> YGIFO(propagate_buf_fw_.ColRange(0, 4 * cell_dim_));
      // no temporal recurrence involved in the inputs
      YGIFO.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_fw_, kTrans, 0.0);
      YGIFO.RowRange(1*S,T*S).AddVecToRows(1.0, bias_fw_);

      for (int t = 1; t <= T; t++) {
        // variables representing invidivual units/gates
        CuSubMatrix<BaseFloat> y_all(propagate_buf_fw_.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_GIFO(YGIFO.RowRange(t*S,S));

        // add the recurrence of the previous memory cell to various gates/units
        y_GIFO.AddMatMat(1.0, YM.RowRange((t-1)*S,S), kNoTrans, wei_gifo_m_fw_, kTrans,  1.0);
        // input gate
        y_i.AddMatDiagVec(1.0, YC.RowRange((t-1)*S,S), kNoTrans, phole_i_c_fw_, 1.0);
        // forget gate
        y_f.AddMatDiagVec(1.0, YC.RowRange((t-1)*S,S), kNoTrans, phole_f_c_fw_, 1.0);
        // apply sigmoid/tanh functionis to squash the outputs
        y_i.Sigmoid(y_i);
        y_f.Sigmoid(y_f);
        y_g.Tanh(y_g);

        // memory cell
        y_c.AddMatDotMat(1.0, y_g, kNoTrans, y_i, kNoTrans, 0.0);
        y_c.AddMatDotMat(1.0, YC.RowRange((t-1)*S,S), kNoTrans, y_f, kNoTrans, 1.0);

        // the tanh-squashed version of c
        y_h.Tanh(y_c);

        // output gate
        y_o.AddMatDiagVec(1.0, y_c, kNoTrans, phole_o_c_fw_, 1.0);
        y_o.Sigmoid(y_o);

        // the final output
        y_m.AddMatDotMat(1.0, y_h, kNoTrans, y_o, kNoTrans, 0.0);

      } // end of t
    }  // end of the forward layer

    void PropagateFncVanillaPassBackward(const CuMatrixBase<BaseFloat> &in, int32 T, int32 S) {

      CuSubMatrix<BaseFloat> YG(propagate_buf_bw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> YI(propagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YF(propagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YO(propagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YC(propagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YH(propagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YM(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));

      CuSubMatrix<BaseFloat> YGIFO(propagate_buf_bw_.ColRange(0, 4 * cell_dim_));
      YGIFO.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_bw_, kTrans, 0.0);
      YGIFO.RowRange(1*S,T*S).AddVecToRows(1.0, bias_bw_);

      for (int t = T; t >= 1; t--) {
        CuSubMatrix<BaseFloat> y_all(propagate_buf_bw_.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S, S));

        CuSubMatrix<BaseFloat> y_GIFO(YGIFO.RowRange(t*S,S));
        y_GIFO.AddMatMat(1.0, YM.RowRange((t+1)*S,S), kNoTrans, wei_gifo_m_bw_, kTrans,  1.0);

        // input gate
        y_i.AddMatDiagVec(1.0, YC.RowRange((t+1)*S,S), kNoTrans, phole_i_c_bw_, 1.0);
        // forget gate
        y_f.AddMatDiagVec(1.0, YC.RowRange((t+1)*S,S), kNoTrans, phole_f_c_bw_, 1.0);
        // apply sigmoid/tanh function
        y_i.Sigmoid(y_i);
        y_f.Sigmoid(y_f);
        y_g.Tanh(y_g);

        // memory cell
        y_c.AddMatDotMat(1.0, y_g, kNoTrans, y_i, kNoTrans, 0.0);
        y_c.AddMatDotMat(1.0, YC.RowRange((t+1)*S,S), kNoTrans, y_f, kNoTrans, 1.0);
        // h -- the tanh-squashed version of c
        y_h.Tanh(y_c);

        // output gate
        y_o.AddMatDiagVec(1.0, y_c, kNoTrans, phole_o_c_bw_, 1.0);
        y_o.Sigmoid(y_o);

        // the final output
        y_m.AddMatDotMat(1.0, y_h, kNoTrans, y_o, kNoTrans, 0.0);

        for (int s = 0; s < S; s++) {
          if (t > sequence_lengths_[s])
            y_all.Row(s).SetZero();
        }
      } // end of t
    }  // end of the backward layer


    void PropagateFncRecurrentDropoutPassForward(const CuMatrixBase<BaseFloat> &in, int32 T, int32 S) {

      CuSubMatrix<BaseFloat> YG(propagate_buf_fw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> YI(propagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YF(propagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YO(propagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YC(propagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YH(propagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YM(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));

      CuSubMatrix<BaseFloat> YGIFO(propagate_buf_fw_.ColRange(0, 4 * cell_dim_));
      // no temporal recurrence involved in the inputs
      YGIFO.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_fw_, kTrans, 0.0);
      YGIFO.RowRange(1*S,T*S).AddVecToRows(1.0, bias_fw_);

      if (!in_train) KALDI_ERR << "Recurrent dropout attempted in test mode";

      CuSubMatrix<BaseFloat> r_mask, zc_mask, zc_mask_i, zh_mask, zh_mask_i;

      // point the mask to the correct position
      if (recurrent_sequence_dropout) {

        if (rnndrop || no_mem_loss_dropout) {
          r_mask =  recurrent_drop_mask_fw_;
        }

      }

      for (int t = 1; t <= T; t++) {
        // variables representing invidivual units/gates
        CuSubMatrix<BaseFloat> y_all(propagate_buf_fw_.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_GIFO(YGIFO.RowRange(t*S,S));

        // point the mask to the correct position
        if (recurrent_step_dropout) {

          if (rnndrop || no_mem_loss_dropout) {
            r_mask =  recurrent_drop_mask_fw_.RowRange(t*S,S);
          }

        }
        // add the recurrence of the previous memory cell to various gates/units
        y_GIFO.AddMatMat(1.0, YM.RowRange((t-1)*S,S), kNoTrans, wei_gifo_m_fw_, kTrans,  1.0);
        // input gate
        y_i.AddMatDiagVec(1.0, YC.RowRange((t-1)*S,S), kNoTrans, phole_i_c_fw_, 1.0);
        // forget gate
        y_f.AddMatDiagVec(1.0, YC.RowRange((t-1)*S,S), kNoTrans, phole_f_c_fw_, 1.0);
        // apply sigmoid/tanh functionis to squash the outputs
        y_i.Sigmoid(y_i);
        y_f.Sigmoid(y_f);
        y_g.Tanh(y_g);
        // memory cell
        y_c.AddMatDotMat(1.0, y_g, kNoTrans, y_i, kNoTrans, 0.0);

        if (no_mem_loss_dropout)
          y_c.AddMatDotMat(1.0, r_mask, kNoTrans, y_c, kNoTrans, 0.0);

        y_c.AddMatDotMat(1.0, YC.RowRange((t-1)*S,S), kNoTrans, y_f, kNoTrans, 1.0);

        if (rnndrop)
          y_c.AddMatDotMat(1.0, r_mask, kNoTrans, y_c, kNoTrans, 0.0);

        // the tanh-squashed version of c
        y_h.Tanh(y_c);

        // output gate
        y_o.AddMatDiagVec(1.0, y_c, kNoTrans, phole_o_c_fw_, 1.0);
        y_o.Sigmoid(y_o);

        // the final output
        y_m.AddMatDotMat(1.0, y_h, kNoTrans, y_o, kNoTrans, 0.0);

      } // end of t
    }  // end of the forward layer

    void PropagateFncRecurrentDropoutPassBackward(const CuMatrixBase<BaseFloat> &in, int32 T, int32 S) {

      CuSubMatrix<BaseFloat> YG(propagate_buf_bw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> YI(propagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YF(propagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YO(propagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YC(propagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YH(propagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YM(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));

      CuSubMatrix<BaseFloat> YGIFO(propagate_buf_bw_.ColRange(0, 4 * cell_dim_));
      YGIFO.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_bw_, kTrans, 0.0);
      YGIFO.RowRange(1*S,T*S).AddVecToRows(1.0, bias_bw_);

      if (!in_train) KALDI_ERR << "Recurrent dropout attempted in test mode";

      CuSubMatrix<BaseFloat> r_mask, zc_mask, zc_mask_i, zh_mask, zh_mask_i;

      // point the mask to the correct position
      if (recurrent_sequence_dropout) {

        if (rnndrop || no_mem_loss_dropout) {
          r_mask =  recurrent_drop_mask_bw_;
        }

      }

      for (int t = T; t >= 1; t--) {
        CuSubMatrix<BaseFloat> y_all(propagate_buf_bw_.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S, S));

        // point the mask to the correct position
        if (recurrent_step_dropout) {

          if (rnndrop || no_mem_loss_dropout) {
            r_mask =  recurrent_drop_mask_bw_.RowRange(t*S,S);
          }

        }

        CuSubMatrix<BaseFloat> y_GIFO(YGIFO.RowRange(t*S,S));

        y_GIFO.AddMatMat(1.0, YM.RowRange((t+1)*S,S), kNoTrans, wei_gifo_m_bw_, kTrans,  1.0);

        // input gate
        y_i.AddMatDiagVec(1.0, YC.RowRange((t+1)*S,S), kNoTrans, phole_i_c_bw_, 1.0);
        // forget gate
        y_f.AddMatDiagVec(1.0, YC.RowRange((t+1)*S,S), kNoTrans, phole_f_c_bw_, 1.0);
        // apply sigmoid/tanh function
        y_i.Sigmoid(y_i);
        y_f.Sigmoid(y_f);
        y_g.Tanh(y_g);

        // memory cell
        y_c.AddMatDotMat(1.0, y_g, kNoTrans, y_i, kNoTrans, 0.0);

        if (no_mem_loss_dropout)
            y_c.AddMatDotMat(1.0, r_mask, kNoTrans, y_c, kNoTrans, 0.0);

        y_c.AddMatDotMat(1.0, YC.RowRange((t+1)*S,S), kNoTrans, y_f, kNoTrans, 1.0);

        if (rnndrop)
          y_c.AddMatDotMat(1.0, r_mask, kNoTrans, y_c, kNoTrans, 0.0);

        // h -- the tanh-squashed version of c
        y_h.Tanh(y_c);

        // output gate
        y_o.AddMatDiagVec(1.0, y_c, kNoTrans, phole_o_c_bw_, 1.0);
        y_o.Sigmoid(y_o);

        // the final output
        y_m.AddMatDotMat(1.0, y_h, kNoTrans, y_o, kNoTrans, 0.0);


        for (int s = 0; s < S; s++) {
          if (t > sequence_lengths_[s])
            y_all.Row(s).SetZero();
        }
      } // end of t
    }  // end of the backward layer

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
      int32 nstream_ = sequence_lengths_.size();  // the number of sequences to be processed in parallel
      KALDI_ASSERT(in.NumRows() % nstream_ == 0);
      int32 T = in.NumRows() / nstream_;
      int32 S = nstream_;

      if (twiddle_forward) {
        twiddle_apply_forward = BernoulliDist(0.5);
      }

      bool apply_recurrent_dropout = in_train && (rnndrop || no_mem_loss_dropout) && (!twiddle_forward || (twiddle_forward && !twiddle_apply_forward));
      bool apply_forward_dropout   = in_train && forward_dropout > 0.0                       && (!twiddle_forward || (twiddle_forward &&  twiddle_apply_forward));

      // initialize the propagation buffers
      propagate_buf_fw_.Resize((T+2)*S, 7 * cell_dim_, kSetZero);
      propagate_buf_bw_.Resize((T+2)*S, 7 * cell_dim_, kSetZero);

      // initialize recurrent masks as needed
      InitializeRecurrentMasks(T,S);

      // propagate forward and then backward cells (same procedures, but iterates from t=T to t=1)
      if (apply_recurrent_dropout) {
        PropagateFncRecurrentDropoutPassForward(in, T, S);
        PropagateFncRecurrentDropoutPassBackward(in, T, S);
      } else {
        PropagateFncVanillaPassForward(in, T, S);
        PropagateFncVanillaPassBackward(in, T, S);
      }

      // final outputs now become the concatenation of the forward and backward activations
      CuMatrix<BaseFloat> YR_RB;
      YR_RB.Resize((T+2)*S, 2 * cell_dim_, kSetZero);
      YR_RB.ColRange(0, cell_dim_).CopyFromMat(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));
      YR_RB.ColRange(cell_dim_, cell_dim_).CopyFromMat(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));

      if (apply_forward_dropout) {
        InitializeForwardMask(T,S);
        YR_RB.RowRange(S,T*S).MulElements(forward_drop_mask_);
      }

      out->CopyFromMat(YR_RB.RowRange(S,T*S));
    }

    void BackpropagateFncVanillaPassForward(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out_diff_drop,
                                              CuMatrixBase<BaseFloat> *in_diff, int32 T, int32 S) {

      if (!in_train) KALDI_ERR << "Can't backpropagate in test mode";

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

      //  assume that the fist half of out_diff is about the forward layer
      DM.RowRange(1*S,T*S).CopyFromMat(out_diff_drop.ColRange(0, cell_dim_));

      for (int t = T; t >= 1; t--) {
        // variables representing activations of invidivual units/gates
        CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S, S));
        // variables representing errors of invidivual units/gates
        CuSubMatrix<BaseFloat> d_g(DG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_i(DI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_f(DF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_o(DO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_c(DC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_h(DH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_m(DM.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_all(backpropagate_buf_fw_.RowRange(t*S, S));

        // d_m comes from two parts: errors from the upper layer and errors from the following frame (t+1)
        d_m.AddMatMat(1.0, DGIFO.RowRange((t+1)*S,S), kNoTrans, wei_gifo_m_fw_, kNoTrans, 1.0);

        // d_h
        d_h.AddMatDotMat(1.0, d_m, kNoTrans, y_o, kNoTrans, 0.0);
        d_h.DiffTanh(y_h, d_h);

        // d_o
        d_o.AddMatDotMat(1.0, d_m, kNoTrans, y_h, kNoTrans, 0.0);
        d_o.DiffSigmoid(y_o, d_o);

        // d_c
        d_c.AddMat(1.0, d_h);
        d_c.AddMatDotMat(1.0, DC.RowRange((t+1)*S,S), kNoTrans, YF.RowRange((t+1)*S,S), kNoTrans, 1.0);
        d_c.AddMatDiagVec(1.0, DI.RowRange((t+1)*S,S), kNoTrans, phole_i_c_fw_, 1.0);
        d_c.AddMatDiagVec(1.0, DF.RowRange((t+1)*S,S), kNoTrans, phole_f_c_fw_, 1.0);
        d_c.AddMatDiagVec(1.0, d_o, kNoTrans, phole_o_c_fw_, 1.0);

        // d_f
        d_f.AddMatDotMat(1.0, d_c, kNoTrans, YC.RowRange((t-1)*S,S), kNoTrans, 0.0);
        d_f.DiffSigmoid(y_f, d_f);

        // d_i
        d_i.AddMatDotMat(1.0, d_c, kNoTrans, y_g, kNoTrans, 0.0);
        d_i.DiffSigmoid(y_i, d_i);

        // d_g
        d_g.AddMatDotMat(1.0, d_c, kNoTrans, y_i, kNoTrans, 0.0);
        d_g.DiffTanh(y_g, d_g);

      }  // end of t

      // errors back-propagated to the inputs
      in_diff->AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kNoTrans, wei_gifo_x_fw_, kNoTrans, 0.0);
      //  updates to the model parameters
      const BaseFloat mmt = opts_.momentum;
      wei_gifo_x_fw_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S, T*S), kTrans, in, kNoTrans, mmt);
      wei_gifo_m_fw_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S, T*S), kTrans, YM.RowRange(0*S,T*S), kNoTrans, mmt);
      bias_fw_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1*S, T*S), mmt);
      phole_i_c_fw_corr_.AddDiagMatMat(1.0, DI.RowRange(1*S, T*S), kTrans, YC.RowRange(0*S, T*S), kNoTrans, mmt);
      phole_f_c_fw_corr_.AddDiagMatMat(1.0, DF.RowRange(1*S, T*S), kTrans, YC.RowRange(0*S, T*S), kNoTrans, mmt);
      phole_o_c_fw_corr_.AddDiagMatMat(1.0, DO.RowRange(1*S, T*S), kTrans, YC.RowRange(1*S, T*S), kNoTrans, mmt);

    }

    void BackpropagateFncVanillaPassBackward(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out_diff_drop,
                                              CuMatrixBase<BaseFloat> *in_diff, int32 T, int32 S) {

      if (!in_train) KALDI_ERR << "Can't backpropagate in test mode";

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
      DM.RowRange(1*S, T*S).CopyFromMat(out_diff_drop.ColRange(cell_dim_, cell_dim_));

      for (int t = 1; t <= T; t++) {
        // variables representing activations of invidivual units/gates
        CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S, S));
        // errors back-propagated to individual gates/units
        CuSubMatrix<BaseFloat> d_all(backpropagate_buf_bw_.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_g(DG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_i(DI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_f(DF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_o(DO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_c(DC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_h(DH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_m(DM.RowRange(t*S, S));

        // d_m comes from two parts: errors from the upper layer and errors from the previous frame (t-1)
        d_m.AddMatMat(1.0, DGIFO.RowRange((t-1)*S,S), kNoTrans, wei_gifo_m_bw_, kNoTrans, 1.0);

        // d_h
        d_h.AddMatDotMat(1.0, d_m, kNoTrans, y_o, kNoTrans, 0.0);
        d_h.DiffTanh(y_h, d_h);

        // d_o
        d_o.AddMatDotMat(1.0, d_m, kNoTrans, y_h, kNoTrans, 0.0);
        d_o.DiffSigmoid(y_o, d_o);

        // d_c
        d_c.AddMat(1.0, d_h);
        d_c.AddMatDotMat(1.0, DC.RowRange((t-1)*S,S), kNoTrans, YF.RowRange((t-1)*S,S), kNoTrans, 1.0);
        d_c.AddMatDiagVec(1.0, DI.RowRange((t-1)*S,S), kNoTrans, phole_i_c_bw_, 1.0);
        d_c.AddMatDiagVec(1.0, DF.RowRange((t-1)*S,S), kNoTrans, phole_f_c_bw_, 1.0);
        d_c.AddMatDiagVec(1.0, d_o, kNoTrans, phole_o_c_bw_, 1.0);

        // d_f
        d_f.AddMatDotMat(1.0, d_c, kNoTrans, YC.RowRange((t+1)*S,S), kNoTrans, 0.0);
        d_f.DiffSigmoid(y_f, d_f);

        // d_i
        d_i.AddMatDotMat(1.0, d_c, kNoTrans, y_g, kNoTrans, 0.0);
        d_i.DiffSigmoid(y_i, d_i);

        // d_g
        d_g.AddMatDotMat(1.0, d_c, kNoTrans, y_i, kNoTrans, 0.0);
        d_g.DiffTanh(y_g, d_g);

      }  // end of t

      // errors back-propagated to the inputs
      in_diff->AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kNoTrans, wei_gifo_x_bw_, kNoTrans, 1.0);
      // updates to the parameters
      const BaseFloat mmt = opts_.momentum;
      wei_gifo_x_bw_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kTrans, in, kNoTrans, mmt);
      wei_gifo_m_bw_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kTrans, YM.RowRange(2*S,T*S), kNoTrans, mmt);
      bias_bw_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1*S,T*S), mmt);
      phole_i_c_bw_corr_.AddDiagMatMat(1.0, DI.RowRange(1*S,T*S), kTrans, YC.RowRange(2*S,T*S), kNoTrans, mmt);
      phole_f_c_bw_corr_.AddDiagMatMat(1.0, DF.RowRange(1*S,T*S), kTrans, YC.RowRange(2*S,T*S), kNoTrans, mmt);
      phole_o_c_bw_corr_.AddDiagMatMat(1.0, DO.RowRange(1*S,T*S), kTrans, YC.RowRange(1*S,T*S), kNoTrans, mmt);
    }

    void BackpropagateFncRecurrentDropoutPassForward(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out_diff_drop,
                                              CuMatrixBase<BaseFloat> *in_diff, int32 T, int32 S) {

      if (!in_train) KALDI_ERR << "Can't backpropagate in test mode";

      // get the activations of the gates/units from the feedforward buffer; these variabiles will be used
      // in gradients computation
      CuSubMatrix<BaseFloat> YG(propagate_buf_fw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> YI(propagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YF(propagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YO(propagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YC(propagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YH(propagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YM(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));

      // buffers for intermediate values
      d_h_mask.Resize((T+2)*S, cell_dim_, kSetZero);
      d_c_mask.Resize((T+2)*S, cell_dim_, kSetZero);

      // errors back-propagated to individual gates/units
      CuSubMatrix<BaseFloat> DG(backpropagate_buf_fw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> DI(backpropagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> DF(backpropagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> DO(backpropagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> DC(backpropagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> DH(backpropagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> DM(backpropagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> DGIFO(backpropagate_buf_fw_.ColRange(0, 4 * cell_dim_));
      CuSubMatrix<BaseFloat> DCM(d_c_mask.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> DHM(d_h_mask.ColRange(0, cell_dim_));

      //  assume that the fist half of out_diff is about the forward layer
      DM.RowRange(1*S,T*S).CopyFromMat(out_diff_drop.ColRange(0, cell_dim_));


      CuSubMatrix<BaseFloat> r_mask, zc_mask, zc_mask_i, zh_mask, zh_mask_i;

      // point the mask to the correct position
      if (recurrent_sequence_dropout) {

        if (rnndrop || no_mem_loss_dropout) {

          r_mask =  recurrent_drop_mask_fw_;
        }

      }

      for (int t = T; t >= 1; t--) {
        // variables representing activations of invidivual units/gates
        CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S, S));
        // variables representing errors of invidivual units/gates
        CuSubMatrix<BaseFloat> d_g(DG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_i(DI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_f(DF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_o(DO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_c(DC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_h(DH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_m(DM.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_all(backpropagate_buf_fw_.RowRange(t*S, S));

        CuSubMatrix<BaseFloat> d_c_m(DCM.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_h_m(DHM.RowRange(t*S, S));

        // point the mask to the correct position
        if (recurrent_step_dropout) {

          if (rnndrop || no_mem_loss_dropout) {

            r_mask =  recurrent_drop_mask_fw_.RowRange(t*S,S);
          }

        }

        // d_m comes from two parts: errors from the upper layer and errors from the following frame (t+1)
        d_m.AddMatMat(1.0, DGIFO.RowRange((t+1)*S,S), kNoTrans, wei_gifo_m_fw_, kNoTrans, 1.0);

        {
          // d_h
          d_h.AddMatDotMat(1.0, d_m, kNoTrans, y_o, kNoTrans, 0.0);
          d_h.DiffTanh(y_h, d_h);

          // d_o
          d_o.AddMatDotMat(1.0, d_m, kNoTrans, y_h, kNoTrans, 0.0);
          d_o.DiffSigmoid(y_o, d_o);

        }

        // d_c
        d_c.AddMat(1.0, d_h);
        d_c.AddMatDiagVec(1.0, DI.RowRange((t+1)*S,S), kNoTrans, phole_i_c_fw_, 1.0);
        d_c.AddMatDiagVec(1.0, DF.RowRange((t+1)*S,S), kNoTrans, phole_f_c_fw_, 1.0);
        d_c.AddMatDiagVec(1.0, d_o, kNoTrans, phole_o_c_fw_, 1.0);

        if (rnndrop) {
          d_c.AddMatDotMat(1.0, DCM.RowRange((t+1)*S,S), kNoTrans, YF.RowRange((t+1)*S,S), kNoTrans, 1.0);
          d_c_m.AddMatDotMat(1.0, d_c, kNoTrans, r_mask, kNoTrans, 0.0);
        }

        if (no_mem_loss_dropout) {
          d_c.AddMatDotMat(1.0, DC.RowRange((t+1)*S,S), kNoTrans, YF.RowRange((t+1)*S,S), kNoTrans, 1.0);
          d_c_m.AddMatDotMat(1.0, d_c, kNoTrans, r_mask, kNoTrans, 0.0);
        }


        // d_f
        if (rnndrop ) {
          d_f.AddMatDotMat(1.0, d_c_m, kNoTrans, YC.RowRange((t-1)*S,S), kNoTrans, 0.0);
        } else {
          d_f.AddMatDotMat(1.0, d_c, kNoTrans, YC.RowRange((t-1)*S,S), kNoTrans, 0.0);
        }
        d_f.DiffSigmoid(y_f, d_f);

        // d_i
        d_i.AddMatDotMat(1.0, d_c_m, kNoTrans, y_g, kNoTrans, 0.0);
        d_i.DiffSigmoid(y_i, d_i);

        // d_g
        d_g.AddMatDotMat(1.0, d_c_m, kNoTrans, y_i, kNoTrans, 0.0);
        d_g.DiffTanh(y_g, d_g);

      }  // end of t

      // errors back-propagated to the inputs
      in_diff->AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kNoTrans, wei_gifo_x_fw_, kNoTrans, 0.0);
      //  updates to the model parameters
      const BaseFloat mmt = opts_.momentum;
      wei_gifo_x_fw_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S, T*S), kTrans, in, kNoTrans, mmt);
      wei_gifo_m_fw_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S, T*S), kTrans, YM.RowRange(0*S,T*S), kNoTrans, mmt);
      bias_fw_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1*S, T*S), mmt);
      phole_i_c_fw_corr_.AddDiagMatMat(1.0, DI.RowRange(1*S, T*S), kTrans, YC.RowRange(0*S, T*S), kNoTrans, mmt);
      phole_f_c_fw_corr_.AddDiagMatMat(1.0, DF.RowRange(1*S, T*S), kTrans, YC.RowRange(0*S, T*S), kNoTrans, mmt);
      phole_o_c_fw_corr_.AddDiagMatMat(1.0, DO.RowRange(1*S, T*S), kTrans, YC.RowRange(1*S, T*S), kNoTrans, mmt);

    }

    void BackpropagateFncRecurrentDropoutPassBackward(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out_diff_drop,
                                              CuMatrixBase<BaseFloat> *in_diff, int32 T, int32 S) {

      if (!in_train) KALDI_ERR << "Can't backpropagate in test mode";

     // get the activations of the gates/units from the feedforward buffer
      CuSubMatrix<BaseFloat> YG(propagate_buf_bw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> YI(propagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YF(propagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YO(propagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YC(propagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YH(propagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> YM(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));

      // buffers for intermediate values
      d_h_mask.Resize((T+2)*S, cell_dim_, kSetZero);
      d_c_mask.Resize((T+2)*S, cell_dim_, kSetZero);

      // errors back-propagated to individual gates/units
      CuSubMatrix<BaseFloat> DG(backpropagate_buf_bw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> DI(backpropagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> DF(backpropagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> DO(backpropagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> DC(backpropagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> DH(backpropagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> DM(backpropagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> DGIFO(backpropagate_buf_bw_.ColRange(0, 4 * cell_dim_));
      CuSubMatrix<BaseFloat> DCM(d_c_mask.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> DHM(d_h_mask.ColRange(0, cell_dim_));

      // the second half of the error vector corresponds to the backward layer
      DM.RowRange(1*S, T*S).CopyFromMat(out_diff_drop.ColRange(cell_dim_, cell_dim_));

      CuSubMatrix<BaseFloat> r_mask, zc_mask, zc_mask_i, zh_mask, zh_mask_i;

      // point the mask to the correct position
      if (recurrent_sequence_dropout) {

        if (rnndrop || no_mem_loss_dropout) {
          r_mask =  recurrent_drop_mask_bw_;
        }

      }

      for (int t = 1; t <= T; t++) {
        // variables representing activations of invidivual units/gates
        CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S, S));
        // errors back-propagated to individual gates/units
        CuSubMatrix<BaseFloat> d_all(backpropagate_buf_bw_.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_g(DG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_i(DI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_f(DF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_o(DO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_c(DC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_h(DH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_m(DM.RowRange(t*S, S));


        CuSubMatrix<BaseFloat> d_c_m(DCM.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_h_m(DHM.RowRange(t*S, S));

        // point the mask to the correct position
        if (recurrent_step_dropout) {

          if (rnndrop || no_mem_loss_dropout) {
            r_mask =  recurrent_drop_mask_bw_.RowRange(t*S,S);
          }

        }
        // d_m comes from two parts: errors from the upper layer and errors from the previous frame (t-1)
        d_m.AddMatMat(1.0, DGIFO.RowRange((t-1)*S,S), kNoTrans, wei_gifo_m_bw_, kNoTrans, 1.0);

        {
          // d_h
          d_h.AddMatDotMat(1.0, d_m, kNoTrans, y_o, kNoTrans, 0.0);
          d_h.DiffTanh(y_h, d_h);

          // d_o
          d_o.AddMatDotMat(1.0, d_m, kNoTrans, y_h, kNoTrans, 0.0);  // y_h is based on y_c
          d_o.DiffSigmoid(y_o, d_o);

        }

        // d_c
        d_c.AddMat(1.0, d_h);
        //d_c.AddMatDotMat(1.0, DC.RowRange((t-1)*S,S), kNoTrans, YF.RowRange((t-1)*S,S), kNoTrans, 1.0);
        d_c.AddMatDiagVec(1.0, DI.RowRange((t-1)*S,S), kNoTrans, phole_i_c_bw_, 1.0);
        d_c.AddMatDiagVec(1.0, DF.RowRange((t-1)*S,S), kNoTrans, phole_f_c_bw_, 1.0);
        d_c.AddMatDiagVec(1.0, d_o, kNoTrans, phole_o_c_bw_, 1.0);

        if (rnndrop) {
          d_c.AddMatDotMat(1.0, DCM.RowRange((t-1)*S,S), kNoTrans, YF.RowRange((t-1)*S,S), kNoTrans, 1.0);
          d_c_m.AddMatDotMat(1.0, d_c, kNoTrans, r_mask, kNoTrans, 0.0);
        }

        if (no_mem_loss_dropout) {
          d_c.AddMatDotMat(1.0, DC.RowRange((t-1)*S,S), kNoTrans, YF.RowRange((t-1)*S,S), kNoTrans, 1.0);
          d_c_m.AddMatDotMat(1.0, d_c, kNoTrans, r_mask, kNoTrans, 0.0);
        }

        // d_f
        if (rnndrop ) {
          d_f.AddMatDotMat(1.0, d_c_m, kNoTrans, YC.RowRange((t+1)*S,S), kNoTrans, 0.0);
        } else {
          d_f.AddMatDotMat(1.0, d_c, kNoTrans, YC.RowRange((t+1)*S,S), kNoTrans, 0.0);
        }
        d_f.DiffSigmoid(y_f, d_f);

        // d_i
        d_i.AddMatDotMat(1.0, d_c_m, kNoTrans, y_g, kNoTrans, 0.0);
        d_i.DiffSigmoid(y_i, d_i);

        // d_g
        d_g.AddMatDotMat(1.0, d_c_m, kNoTrans, y_i, kNoTrans, 0.0);
        d_g.DiffTanh(y_g, d_g);

      }  // end of t

      // errors back-propagated to the inputs
      in_diff->AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kNoTrans, wei_gifo_x_bw_, kNoTrans, 1.0);
      // updates to the parameters
      const BaseFloat mmt = opts_.momentum;
      wei_gifo_x_bw_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kTrans, in, kNoTrans, mmt);
      wei_gifo_m_bw_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kTrans, YM.RowRange(2*S,T*S), kNoTrans, mmt);
      bias_bw_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1*S,T*S), mmt);
      phole_i_c_bw_corr_.AddDiagMatMat(1.0, DI.RowRange(1*S,T*S), kTrans, YC.RowRange(2*S,T*S), kNoTrans, mmt);
      phole_f_c_bw_corr_.AddDiagMatMat(1.0, DF.RowRange(1*S,T*S), kTrans, YC.RowRange(2*S,T*S), kNoTrans, mmt);
      phole_o_c_bw_corr_.AddDiagMatMat(1.0, DO.RowRange(1*S,T*S), kTrans, YC.RowRange(1*S,T*S), kNoTrans, mmt);
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
      int32 nstream_ = sequence_lengths_.size();  // the number of sequences to be processed in parallel
      KALDI_ASSERT(in.NumRows() % nstream_ == 0);
      int32 T = in.NumRows() / nstream_;
      int32 S = nstream_;

      bool apply_recurrent_dropout = in_train && (rnndrop || no_mem_loss_dropout) && (!twiddle_forward || (twiddle_forward && !twiddle_apply_forward));
      bool apply_forward_dropout   = in_train && forward_dropout > 0.0                       && (!twiddle_forward || (twiddle_forward &&  twiddle_apply_forward));

      CuMatrix<BaseFloat> out_diff_drop;
      out_diff_drop.Resize(out_diff.NumRows(), out_diff.NumCols());
      out_diff_drop.CopyFromMat(out_diff);
      if (apply_forward_dropout) {
        out_diff_drop.MulElements(forward_drop_mask_);
      }

      // initialize the back-propagation buffer
      backpropagate_buf_fw_.Resize((T+2)*S, 7 * cell_dim_, kSetZero);
      backpropagate_buf_bw_.Resize((T+2)*S, 7 * cell_dim_, kSetZero);

     // back-propagation in the forward then backard cell layer
      if (apply_recurrent_dropout) {
        BackpropagateFncRecurrentDropoutPassForward(in, out_diff_drop, in_diff, T, S);
        //CheckNanInf(backpropagate_buf_fw_," BackpropagateFncRecurrentDropoutPassForward ");
        BackpropagateFncRecurrentDropoutPassBackward(in, out_diff_drop, in_diff, T, S);
        //CheckNanInf(backpropagate_buf_bw_," BackpropagateFncRecurrentDropoutPassBackward ");
      } else {
        BackpropagateFncVanillaPassForward(in, out_diff_drop, in_diff, T, S);
        BackpropagateFncVanillaPassBackward(in, out_diff_drop, in_diff, T, S);
      }

    }

private:

    int32 nstream_;
    std::vector<int> sequence_lengths_;

};
} // namespace eesen

#endif
