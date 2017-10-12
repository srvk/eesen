// net/layer.h

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)
//                2015  Yajie Miao, Hang Su

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



#ifndef EESEN_NNET_LAYER_H_
#define EESEN_NNET_LAYER_H_


#include "base/kaldi-common.h"
#include "cpucompute/matrix-lib.h"
#include "gpucompute/cuda-matrix.h"
#include "gpucompute/cuda-vector.h"
#include "net/train-opts.h"

#include <iostream>

namespace eesen {

/**
 * Abstract class, building block of the network.
 * It is able to propagate (PropagateFnc: compute the output based on its input)
 * and backpropagate (BackpropagateFnc: i.e. transform loss derivative w.r.t. output to derivative w.r.t. the input)
 * the formulas are implemented in descendant classes (AffineTransform,Sigmoid,Softmax,...).
 */
class Layer {

 /// Layer type identification mechanism
 public:
  /// Types of Layers
  typedef enum {
    l_Unknown = 0x0,

    l_Trainable = 0x0100,
    l_Affine_Transform,
    l_BiLstm,
    l_BiLstm_Parallel,
    l_Lstm,
    l_Lstm_Parallel,

    l_Activation = 0x0200,
    l_Softmax,
    l_Sigmoid,
    l_Tanh,
  } LayerType;
  /// A pair of type and marker
  struct key_value {
    const Layer::LayerType key;
    const char *value;
  };
  /// Mapping of types and markers (the table is defined in nnet-component.cc)
  static const struct key_value kMarkerMap[];
  /// Convert component type to marker
  static const char* TypeToMarker(LayerType t);
  /// Convert marker to component type (case insensitive)
  static LayerType MarkerToType(const std::string &s);

 /// General interface of a component
 public:
  Layer(int32 input_dim, int32 output_dim)
      : input_dim_(input_dim), output_dim_(output_dim) { }
  virtual ~Layer() { }

  /// Copy component (deep copy).
  virtual Layer* Copy() const = 0;

  /// Get Type Identification of the component
  virtual LayerType GetType() const = 0;
  /// Get Type Identification of the non-parallel version of the component
  virtual LayerType GetTypeNonParal() const = 0;
  /// Check if contains trainable parameters
  virtual bool IsTrainable() const {
    return false;
  }

  virtual void SetTrainMode() {};
  virtual void SetTestMode() {};
  virtual void ChangeDropoutParameters(BaseFloat forward_dropout,
                                  bool fw_step_dropout,
                                  bool fw_sequence_dropout,

                                  bool rnndrop ,
                                  bool no_mem_loss_dropout,
                                  BaseFloat recurrent_dropout,
                                  bool recurrent_step_dropout,
                                  bool recurrent_sequence_dropout,

                                  bool twiddleforward) {};
  bool IsBiLstm(LayerType t) {
    return ( t == l_BiLstm);
  }

  /// Get size of input vectors
  int32 InputDim() const {
    return input_dim_;
  }
  /// Get size of output vectors
  int32 OutputDim() const {
    return output_dim_;
  }

  /// Perform forward pass propagation Input->Output
  void Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out);
  /// Perform backward pass propagation, out_diff -> in_diff
  void Backpropagate(const CuMatrixBase<BaseFloat> &in,
                     const CuMatrixBase<BaseFloat> &out,
                     const CuMatrixBase<BaseFloat> &out_diff,
                     CuMatrix<BaseFloat> *in_diff);

  /// Initialize component from a line in config file
  static Layer* Init(const std::string &conf_line);
  /// Read component from stream
  static Layer* Read(std::istream &is, bool binary);
  static Layer* Read(std::istream &is, bool binary, bool convertparal);
  /// Read component from stream (with layer initialized before)
  void ReRead(std::istream &is, bool binary);
  /// Write component to stream
  void Write(std::ostream &os, bool binary) const;
  void WriteNonParal(std::ostream &os, bool binary) const;

  /// Optionally print some additional info
  virtual std::string Info() const { return ""; }
  virtual std::string InfoGradient() const { return ""; }

  /// Set the lengths of sequences that are processed in parallel
  /// during training of LSTM models.
  virtual void SetSeqLengths(std::vector<int> &sequence_lengths) { }

 /// Abstract interface for propagation/backpropagation
 protected:
  /// Forward pass transformation (to be implemented by descending class...)
  virtual void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                            CuMatrixBase<BaseFloat> *out) = 0;
  /// Backward pass transformation (to be implemented by descending class...)
  virtual void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                                const CuMatrixBase<BaseFloat> &out,
                                const CuMatrixBase<BaseFloat> &out_diff,
                                CuMatrixBase<BaseFloat> *in_diff) = 0;

  /// Initialize internal data of a component
  virtual void InitData(std::istream &is) { }

  /// Reads the component content
  virtual void ReadData(std::istream &is, bool binary) { }

  /// Writes the component content
  virtual void WriteData(std::ostream &os, bool binary) const { }

 /// Data members
 protected:
  int32 input_dim_;  ///< Size of input vectors
  int32 output_dim_; ///< Size of output vectors

 private:
  /// Create new intance of layer
  static Layer* NewLayerOfType(LayerType t, int32 input_dim, int32 output_dim);

};

inline bool IsLstmType(std::string layer_type_string) {
  if (layer_type_string == "<BiLstm>" || layer_type_string == "<Lstm>" || layer_type_string == "<BiLstmParallel>" || layer_type_string == "<LstmParallel>" ) {
    return true;
  }
  return false;
}

inline void Layer::Propagate(const CuMatrixBase<BaseFloat> &in,
                                   CuMatrix<BaseFloat> *out) {
  // Check the dims
  if (input_dim_ != in.NumCols()) {
    KALDI_ERR << "Non-matching dims! " << TypeToMarker(GetType())
              << " input-dim : " << input_dim_ << " data : " << in.NumCols();
  }
  // Allocate target buffer
  out->Resize(in.NumRows(), output_dim_, kSetZero); // reset
  // Call the propagation implementation of the component
  PropagateFnc(in, out);
}

inline void Layer::Backpropagate(const CuMatrixBase<BaseFloat> &in,
                                 const CuMatrixBase<BaseFloat> &out,
                                 const CuMatrixBase<BaseFloat> &out_diff,
                                 CuMatrix<BaseFloat> *in_diff) {
  // Check the dims
  if (output_dim_ != out_diff.NumCols()) {
    KALDI_ERR << "Non-matching output dims, component:" << output_dim_
              << " data:" << out_diff.NumCols();
  }

  // Allocate target buffer
  in_diff->Resize(out_diff.NumRows(), input_dim_, kSetZero); // reset
  // Asserts on the dims
  KALDI_ASSERT((in.NumRows() == out.NumRows()) &&
               (in.NumRows() == out_diff.NumRows()) &&
               (in.NumRows() == in_diff->NumRows()));
  KALDI_ASSERT(in.NumCols() == in_diff->NumCols());
  KALDI_ASSERT(out.NumCols() == out_diff.NumCols());
    // Call the backprop implementation of the component
  BackpropagateFnc(in, out, out_diff, in_diff);
}

} // namespace eesen


#endif
