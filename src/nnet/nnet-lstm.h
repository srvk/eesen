// nnet/nnet-lstm.h
// nnet/nnet-affine-transform.h

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

#ifndef KALDI_NNET_LSTM_H_
#define KALDI_NNET_LSTM_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"

/*************************************
 * x: input neuron
 * g: squashing neuron near input
 * i: Input gate
 * f: Forget gate
 * o: Output gate
 * c: memory Cell (CEC)
 * h: squashing neuron near output
 * m: output neuron of Memory block
 * r: recurrent projection neuron
 * y: output neuron of LSTMP
 *************************************/

namespace kaldi {
namespace nnet1 {
class Lstm : public UpdatableComponent {
public:
    Lstm(int32 input_dim, int32 output_dim) :
        UpdatableComponent(input_dim, output_dim),
        ncell_(output_dim)
    { }

    ~Lstm()
    { }

    Component* Copy() const { return new Lstm(*this); }
    ComponentType GetType() const { return kLstm; }
    ComponentType GetTypeNonParal() const { return kLstm; }

    static void InitMatParam(CuMatrix<BaseFloat> &m, float scale) {
        m.SetRandUniform();  // uniform in [0, 1]
        m.Add(-0.5);         // uniform in [-0.5, 0.5]
        m.Scale(2 * scale);  // uniform in [-scale, +scale]
    }

    static void InitVecParam(CuVector<BaseFloat> &v, float scale) {
        Vector<BaseFloat> tmp(v.Dim());
        for (int i=0; i < tmp.Dim(); i++) {
            tmp(i) = (RandUniform() - 0.5) * 2 * scale;
        }
        v = tmp;
    }

    void InitData(std::istream &is) {
        // define options
        float param_scale = 0.02;
        // parse config
        std::string token;
        while (!is.eof()) {
            ReadToken(is, false, &token); 
            if (token == "<CellDim>") 
                ReadBasicType(is, false, &ncell_);
            else if (token == "<ParamScale>") 
                ReadBasicType(is, false, &param_scale);
            else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                           << " (CellDim|ParamScale)";
            is >> std::ws; // eat-up whitespace
        }

        // init weight and bias (Uniform)
        w_gifo_x_.Resize(4*ncell_, input_dim_, kUndefined);  InitMatParam(w_gifo_x_, param_scale);
        w_gifo_m_.Resize(4*ncell_, ncell_, kUndefined);  InitMatParam(w_gifo_m_, param_scale);

        bias_.Resize(4*ncell_, kUndefined);        InitVecParam(bias_, param_scale);
        peephole_i_c_.Resize(ncell_, kUndefined);  InitVecParam(peephole_i_c_, param_scale);
        peephole_f_c_.Resize(ncell_, kUndefined);  InitVecParam(peephole_f_c_, param_scale);
        peephole_o_c_.Resize(ncell_, kUndefined);  InitVecParam(peephole_o_c_, param_scale);

        // init delta buffers
        w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero); 
        w_gifo_m_corr_.Resize(4*ncell_, ncell_, kSetZero);    
        bias_corr_.Resize(4*ncell_, kSetZero);     

        peephole_i_c_corr_.Resize(ncell_, kSetZero);
        peephole_f_c_corr_.Resize(ncell_, kSetZero);
        peephole_o_c_corr_.Resize(ncell_, kSetZero);
    }

    void ReadData(std::istream &is, bool binary) {
        ExpectToken(is, binary, "<CellDim>");
        ReadBasicType(is, binary, &ncell_);

        w_gifo_x_.Read(is, binary);
        w_gifo_m_.Read(is, binary);
        bias_.Read(is, binary);

        peephole_i_c_.Read(is, binary);
        peephole_f_c_.Read(is, binary);
        peephole_o_c_.Read(is, binary);

        // init delta buffers
        w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero); 
        w_gifo_m_corr_.Resize(4*ncell_, ncell_, kSetZero);    
        bias_corr_.Resize(4*ncell_, kSetZero);     

        peephole_i_c_corr_.Resize(ncell_, kSetZero);
        peephole_f_c_corr_.Resize(ncell_, kSetZero);
        peephole_o_c_corr_.Resize(ncell_, kSetZero);
    }

    void WriteData(std::ostream &os, bool binary) const {
        WriteToken(os, binary, "<CellDim>");
        WriteBasicType(os, binary, ncell_);

        w_gifo_x_.Write(os, binary);
        w_gifo_m_.Write(os, binary);
        bias_.Write(os, binary);

        peephole_i_c_.Write(os, binary);
        peephole_f_c_.Write(os, binary);
        peephole_o_c_.Write(os, binary);
    }

    int32 NumParams() const { 
        return ( w_gifo_x_.NumRows() * w_gifo_x_.NumCols() +
                 w_gifo_m_.NumRows() * w_gifo_m_.NumCols() +
                 bias_.Dim() +
                 peephole_i_c_.Dim() +
                 peephole_f_c_.Dim() +
                 peephole_o_c_.Dim() );
    }

    void GetParams(Vector<BaseFloat>* wei_copy) const {
        wei_copy->Resize(NumParams());

        int32 offset, len;

        offset = 0;    len = w_gifo_x_.NumRows() * w_gifo_x_.NumCols();
        wei_copy->Range(offset, len).CopyRowsFromMat(w_gifo_x_);

        offset += len; len = w_gifo_m_.NumRows() * w_gifo_m_.NumCols();
        wei_copy->Range(offset, len).CopyRowsFromMat(w_gifo_m_);

        offset += len; len = bias_.Dim();
        wei_copy->Range(offset, len).CopyFromVec(bias_);

        offset += len; len = peephole_i_c_.Dim();
        wei_copy->Range(offset, len).CopyFromVec(peephole_i_c_);

        offset += len; len = peephole_f_c_.Dim();
        wei_copy->Range(offset, len).CopyFromVec(peephole_f_c_);

        offset += len; len = peephole_o_c_.Dim();
        wei_copy->Range(offset, len).CopyFromVec(peephole_o_c_);

        return;
    }

    std::string Info() const {
        return std::string("    ") + 
            "\n  w_gifo_x_  "     + MomentStatistics(w_gifo_x_) + 
            "\n  w_gifo_m_  "     + MomentStatistics(w_gifo_m_) +
            "\n  bias_  "         + MomentStatistics(bias_) +
            "\n  peephole_i_c_  " + MomentStatistics(peephole_i_c_) +
            "\n  peephole_f_c_  " + MomentStatistics(peephole_f_c_) +
            "\n  peephole_o_c_  " + MomentStatistics(peephole_o_c_);
    }
  
    std::string InfoGradient() const {
        return std::string("    ") + 
            "\n  w_gifo_x_corr_  "     + MomentStatistics(w_gifo_x_corr_) + 
            "\n  w_gifo_m_corr_  "     + MomentStatistics(w_gifo_m_corr_) +
            "\n  bias_corr_  "         + MomentStatistics(bias_corr_) +
            "\n  peephole_i_c_corr_  " + MomentStatistics(peephole_i_c_corr_) +
            "\n  peephole_f_c_corr_  " + MomentStatistics(peephole_f_c_corr_) +
            "\n  peephole_o_c_corr_  " + MomentStatistics(peephole_o_c_corr_);
    }

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
        int DEBUG = 0;
        int32 T = in.NumRows();

        // resize & clear propagate buffers
        propagate_buf_.Resize(T+2, 7 * ncell_, kSetZero);  // 0:forward pass history, [1, T]:current sequence, T+1:dummy

        // disassemble entire neuron activation buffer into different neurons
        CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*ncell_, ncell_));

        CuSubMatrix<BaseFloat> YGIFO(propagate_buf_.ColRange(0, 4*ncell_));

        // (x & bias) -> g, i, f, o, not recurrent, do it all in once
        YGIFO.RowRange(1,T).AddMatMat(1.0, in, kNoTrans, w_gifo_x_, kTrans, 0.0);
        YGIFO.RowRange(1,T).AddVecToRows(1.0, bias_);

        for (int t = 1; t <= T; t++) {
            // (vector & matrix) representations of neuron activations at frame t 
            // so we can borrow rich APIs from both CuMatrix and CuVector
            CuSubVector<BaseFloat> y_g(YG.Row(t));  CuSubMatrix<BaseFloat> YG_t(YG.RowRange(t,1));  
            CuSubVector<BaseFloat> y_i(YI.Row(t));  CuSubMatrix<BaseFloat> YI_t(YI.RowRange(t,1));  
            CuSubVector<BaseFloat> y_f(YF.Row(t));  CuSubMatrix<BaseFloat> YF_t(YF.RowRange(t,1));  
            CuSubVector<BaseFloat> y_o(YO.Row(t));  CuSubMatrix<BaseFloat> YO_t(YO.RowRange(t,1));  
            CuSubVector<BaseFloat> y_c(YC.Row(t));  CuSubMatrix<BaseFloat> YC_t(YC.RowRange(t,1));  
            CuSubVector<BaseFloat> y_h(YH.Row(t));  CuSubMatrix<BaseFloat> YH_t(YH.RowRange(t,1));  
            CuSubVector<BaseFloat> y_m(YM.Row(t));  CuSubMatrix<BaseFloat> YM_t(YM.RowRange(t,1));  

            CuSubVector<BaseFloat> y_gifo(YGIFO.Row(t));
    
            // recursion r(t-1) -> g, i, f, o
            y_gifo.AddMatVec(1.0, w_gifo_m_, kNoTrans, YM.Row(t-1), 1.0);
            // peephole c(t-1) -> i(t)
            y_i.AddVecVec(1.0, peephole_i_c_, YC.Row(t-1), 1.0);
            // peephole c(t-1) -> f(t)
            y_f.AddVecVec(1.0, peephole_f_c_, YC.Row(t-1), 1.0);

            // i, f sigmoid squashing
            YI_t.Sigmoid(YI_t);
            YF_t.Sigmoid(YF_t);
    
            // g tanh squashing
            YG_t.Tanh(YG_t);
    
            // c memory cell
            y_c.AddVecVec(1.0, y_i, y_g, 0.0);
            // CEC connection via forget gate: c(t-1) -> c(t)
            y_c.AddVecVec(1.0, y_f, YC.Row(t-1), 1.0);

//            YC_t.ApplyFloor(-50);   // optional clipping of cell activation
//            YC_t.ApplyCeiling(50);  // google paper Interspeech2014: LSTM for LVCSR
    
            // h tanh squashing
            YH_t.Tanh(YC_t);
    
            // o output gate
            y_o.AddVecVec(1.0, peephole_o_c_, y_c, 1.0);  // notice: output gate peephole is not recurrent
            YO_t.Sigmoid(YO_t);
    
            // m
            y_m.AddVecVec(1.0, y_o, y_h, 0.0);
            
            if (DEBUG) {
                std::cerr << "forward-pass frame " << t << "\n";
                std::cerr << "activation of g: " << y_g;
                std::cerr << "activation of i: " << y_i;
                std::cerr << "activation of f: " << y_f;
                std::cerr << "activation of o: " << y_o;
                std::cerr << "activation of c: " << y_c;
                std::cerr << "activation of h: " << y_h;
                std::cerr << "activation of m: " << y_m;
            }
        }

        // recurrent projection layer is also feed-forward as LSTM output
        out->CopyFromMat(YM.RowRange(1,T));
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        int DEBUG = 0;
        int32 T = in.NumRows();
        // disassemble propagated buffer into neurons
        CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*ncell_, ncell_));
    
        // 0-init backpropagate buffer
        backpropagate_buf_.Resize(T+2, 7 * ncell_, kSetZero);  // 0:dummy, [1,T] frames, T+1 backward pass history

        // disassemble backpropagate buffer into neurons
        CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(0*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DI(backpropagate_buf_.ColRange(1*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DF(backpropagate_buf_.ColRange(2*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DO(backpropagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(4*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(5*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(6*ncell_, ncell_));

        CuSubMatrix<BaseFloat> DGIFO(backpropagate_buf_.ColRange(0, 4*ncell_));

        // projection layer to LSTM output is not recurrent, so backprop it all in once
        DM.RowRange(1,T).CopyFromMat(out_diff);

        for (int t = T; t >= 1; t--) {
            // vector representation                  // matrix representation
            CuSubVector<BaseFloat> y_g(YG.Row(t));    CuSubMatrix<BaseFloat> YG_t(YG.RowRange(t,1));  
            CuSubVector<BaseFloat> y_i(YI.Row(t));    CuSubMatrix<BaseFloat> YI_t(YI.RowRange(t,1));  
            CuSubVector<BaseFloat> y_f(YF.Row(t));    CuSubMatrix<BaseFloat> YF_t(YF.RowRange(t,1));  
            CuSubVector<BaseFloat> y_o(YO.Row(t));    CuSubMatrix<BaseFloat> YO_t(YO.RowRange(t,1));  
            CuSubVector<BaseFloat> y_c(YC.Row(t));    CuSubMatrix<BaseFloat> YC_t(YC.RowRange(t,1));  
            CuSubVector<BaseFloat> y_h(YH.Row(t));    CuSubMatrix<BaseFloat> YH_t(YH.RowRange(t,1));  
            CuSubVector<BaseFloat> y_m(YM.Row(t));    CuSubMatrix<BaseFloat> YM_t(YM.RowRange(t,1));  
    
            CuSubVector<BaseFloat> d_g(DG.Row(t));    CuSubMatrix<BaseFloat> DG_t(DG.RowRange(t,1));
            CuSubVector<BaseFloat> d_i(DI.Row(t));    CuSubMatrix<BaseFloat> DI_t(DI.RowRange(t,1));
            CuSubVector<BaseFloat> d_f(DF.Row(t));    CuSubMatrix<BaseFloat> DF_t(DF.RowRange(t,1));
            CuSubVector<BaseFloat> d_o(DO.Row(t));    CuSubMatrix<BaseFloat> DO_t(DO.RowRange(t,1));
            CuSubVector<BaseFloat> d_c(DC.Row(t));    CuSubMatrix<BaseFloat> DC_t(DC.RowRange(t,1));
            CuSubVector<BaseFloat> d_h(DH.Row(t));    CuSubMatrix<BaseFloat> DH_t(DH.RowRange(t,1));
            CuSubVector<BaseFloat> d_m(DM.Row(t));    CuSubMatrix<BaseFloat> DM_t(DM.RowRange(t,1));
    
            // r
            //   Version 1 (precise gradients): 
            //   backprop error from g(t+1), i(t+1), f(t+1), o(t+1) to r(t)
            d_m.AddMatVec(1.0, w_gifo_m_, kTrans, DGIFO.Row(t+1), 1.0);

            
            //   Version 2 (Alex Graves' PhD dissertation): 
            //   only backprop g(t+1) to r(t) 
//            CuSubMatrix<BaseFloat> w_g_m_(w_gifo_m_.RowRange(0, ncell_));
//            d_m.AddMatVec(1.0, w_g_m_, kTrans, DG.Row(t+1), 1.0);
            

            /*
            //   Version 3 (Felix Gers' PhD dissertation): 
            //   truncate gradients of g(t+1), i(t+1), f(t+1), o(t+1) once they leak out memory block
            //   CEC(with forget connection) is the only "error-bridge" through time
            ;
            */
    
            // h
            d_h.AddVecVec(1.0, y_o, d_m, 0.0);
            DH_t.DiffTanh(YH_t, DH_t);
    
            // o
            d_o.AddVecVec(1.0, y_h, d_m, 0.0);
            DO_t.DiffSigmoid(YO_t, DO_t);
    
            // c
            //   1. diff from h(t)
            //   2. diff from o(t) (via peephole)
            //   3. diff from c(t+1) (via forget-gate between CEC)
            //   4. diff from f(t+1) (via peephole)
            //   5. diff from i(t+1) (via peephole)
            d_c.AddVec(1.0, d_h, 0.0);  
            d_c.AddVecVec(1.0, peephole_o_c_,  d_o, 1.0);
            d_c.AddVecVec(1.0, YF.Row(t+1),    DC.Row(t+1), 1.0);
            d_c.AddVecVec(1.0, peephole_f_c_ , DF.Row(t+1), 1.0);
            d_c.AddVecVec(1.0, peephole_i_c_,  DI.Row(t+1), 1.0);
    
            // f
            d_f.AddVecVec(1.0, YC.Row(t-1), d_c, 0.0);
            DF_t.DiffSigmoid(YF_t, DF_t);
    
            // i
            d_i.AddVecVec(1.0, y_g, d_c, 0.0);
            DI_t.DiffSigmoid(YI_t, DI_t);
    
            // g
            d_g.AddVecVec(1.0, y_i, d_c, 0.0);
            DG_t.DiffTanh(YG_t, DG_t);
    
            // debug info
            if (DEBUG) {
                std::cerr << "backward-pass frame " << t << "\n";
                std::cerr << "derivative wrt input m " << d_m;
                std::cerr << "derivative wrt input h " << d_h;
                std::cerr << "derivative wrt input o " << d_o;
                std::cerr << "derivative wrt input c " << d_c;
                std::cerr << "derivative wrt input f " << d_f;
                std::cerr << "derivative wrt input i " << d_i;
                std::cerr << "derivative wrt input g " << d_g;
            }
        }

        // backprop derivatives to input x, do it all in once
        in_diff->AddMatMat(1.0, DGIFO.RowRange(1,T), kNoTrans, w_gifo_x_, kNoTrans, 0.0);
    
        // calculate delta
        const BaseFloat mmt = opts_.momentum;
    
        w_gifo_x_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, in              , kNoTrans, mmt);
        w_gifo_m_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, YM.RowRange(0,T), kNoTrans, mmt);  // recurrent r -> g

        bias_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1,T), mmt);
    
        peephole_i_c_corr_.AddDiagMatMat(1.0, DI.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);  // recurrent c -> i
        peephole_f_c_corr_.AddDiagMatMat(1.0, DF.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);  // recurrent c -> f
        peephole_o_c_corr_.AddDiagMatMat(1.0, DO.RowRange(1,T), kTrans, YC.RowRange(1,T), kNoTrans, mmt);
    
        if (DEBUG) {
            std::cerr << "gradients(with optional momentum): \n";
            std::cerr << "w_gifo_x_corr_ " << w_gifo_x_corr_;
            std::cerr << "w_gifo_m_corr_ " << w_gifo_m_corr_;
            std::cerr << "bias_corr_ " << bias_corr_;
            std::cerr << "peephole_i_c_corr_ " << peephole_i_c_corr_;
            std::cerr << "peephole_f_c_corr_ " << peephole_f_c_corr_;
            std::cerr << "peephole_o_c_corr_ " << peephole_o_c_corr_;
        }
    }

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out, const CuMatrixBase<BaseFloat> &targets) {
     PropagateFnc(in, out);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff,
                        const CuMatrixBase<BaseFloat> &targets) {
    BackpropagateFnc(in, out, out_diff, in_diff);
  }

    void ClipGradMat(CuMatrixBase<BaseFloat> &M, BaseFloat thres) {
        M.ApplyFloor(-thres);
        M.ApplyCeiling(thres);
    }   

    void ClipGradVec(CuVectorBase<BaseFloat> &V, BaseFloat thres) {
        Vector<BaseFloat> tmp(V);
        tmp.ApplyFloor(-thres);
        tmp.ApplyCeiling(thres);
        V.CopyFromVec(tmp);
    }

    void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
        // gradient clipping (element-wise)
        BaseFloat max_grad = 50.0;

//        ClipGradMat(w_gifo_x_corr_, max_grad);
//        ClipGradMat(w_gifo_m_corr_, max_grad);

//        ClipGradVec(bias_corr_, max_grad);

//        ClipGradVec(peephole_i_c_corr_, max_grad);
//        ClipGradVec(peephole_f_c_corr_, max_grad);
//        ClipGradVec(peephole_o_c_corr_, max_grad);

	// update
        const BaseFloat lr  = opts_.learn_rate;
        w_gifo_x_.AddMat(-lr, w_gifo_x_corr_);
        w_gifo_m_.AddMat(-lr, w_gifo_m_corr_);
        bias_.AddVec(-lr, bias_corr_, 1.0);
    
        peephole_i_c_.AddVec(-lr, peephole_i_c_corr_, 1.0);
        peephole_f_c_.AddVec(-lr, peephole_f_c_corr_, 1.0);
        peephole_o_c_.AddVec(-lr, peephole_o_c_corr_, 1.0);
    

//        /* 
//          Here we deal with the famous "vanishing & exploding difficulties" in RNN learning.
//
//          *For gradients vanishing*
//            LSTM architecture introduces linear CEC as the "error bridge" across long time distance
//            solving vanishing problem.
//
//          *For gradients exploding*
//            LSTM is still vulnerable to gradients explosing in BPTT(with large weight & deep time expension).
//            To prevent this, we tried L2 regularization, which didn't work well
//
//          Our approach is a *modified* version of Max Norm Regularization:
//          For each nonlinear neuron, 
//            1. fan-in weights & bias model a seperation hyper-plane: W x + b = 0
//            2. squashing function models a differentiable nonlinear slope around this hyper-plane.
//
//          Conventional max norm regularization scale W to keep its L2 norm bounded,
//          As a modification, we scale down large (W & b) *simultaneously*, this:
//            1. keeps all fan-in weights small, prevents gradients from exploding during backward-pass.
//            2. keeps the location of the hyper-plane unchanged, so we don't wipe out already learned knowledge.
//            3. shrinks the "normal" of the hyper-plane, smooths the nonlinear slope, improves generalization.
//            4. makes the network *well-conditioned* (weights are constrained in a reasonible range).
//
//          We've observed faster convergence and performance gain by doing this.
//        */
//
//        if (DEBUG) {
//            if (shrink.Min() < 0.95) {   // we dont want too many trivial logs here
//                std::cerr << "gifo shrinking coefs: " << shrink;
//            }
//        }
//        
    }

private:
    // dims
    int32 ncell_;

    // feed-forward connections: from x to [g, i, f, o]
    CuMatrix<BaseFloat> w_gifo_x_;
    CuMatrix<BaseFloat> w_gifo_x_corr_;

    // recurrent projection connections: from r to [g, i, f, o]
    CuMatrix<BaseFloat> w_gifo_m_;
    CuMatrix<BaseFloat> w_gifo_m_corr_;

    // biases of [g, i, f, o]
    CuVector<BaseFloat> bias_;
    CuVector<BaseFloat> bias_corr_;

    // peephole from c to i, f, g 
    // peephole connections are block-internal, so we use vector form
    CuVector<BaseFloat> peephole_i_c_;
    CuVector<BaseFloat> peephole_f_c_;
    CuVector<BaseFloat> peephole_o_c_;

    CuVector<BaseFloat> peephole_i_c_corr_;
    CuVector<BaseFloat> peephole_f_c_corr_;
    CuVector<BaseFloat> peephole_o_c_corr_;

    // propagate buffer: output of [g, i, f, o, c, h, m, r]
    CuMatrix<BaseFloat> propagate_buf_;

    // back-propagate buffer: diff-input of [g, i, f, o, c, h, m, r]
    CuMatrix<BaseFloat> backpropagate_buf_;

};
} // namespace nnet1
} // namespace kaldi

#endif
