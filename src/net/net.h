// net/net.h

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

#ifndef EESEN_NET_H_
#define EESEN_NET_H_

#include <iostream>
#include <sstream>
#include <vector>

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "cpucompute/matrix-lib.h"
#include "net/train-opts.h"
#include "net/layer.h"
#include "net/trainable-layer.h"

namespace eesen {

class Net {
 public:
  Net() {}
  Net(const Net& other); // Copy constructor.
  Net &operator = (const Net& other); // Assignment operator.

  ~Net(); 

 public:
  /// Perform forward pass through the network
  void Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out); 
  /// Perform backward pass through the network
  void Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff);
  /// Perform forward pass through the network, don't keep buffers (use it when not training)
  void Feedforward(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out); 

  /// Dimensionality on network input (input feature dim.)
  int32 InputDim() const; 
  /// Dimensionality of network outputs (posteriors | bn-features | etc.)
  int32 OutputDim() const; 

  /// Returns number of layers
  int32 NumLayers() const { return layers_.size(); }

  const Layer& GetLayer(int32 c) const;
  Layer& GetLayer(int32 c);

  /// Sets the c'th layer to "layer", taking ownership of the pointer
  /// and deleting the corresponding one that we own.
  void SetLayer(int32 c, Layer *layer);
 
  /// Remove component
  void RemoveLayer(int32 c);
  void RemoveLastLayer() { RemoveLayer(NumLayers()-1); }

  /// Access to forward pass buffers
  const std::vector<CuMatrix<BaseFloat> >& PropagateBuffer() const { 
    return propagate_buf_; 
  }
  /// Access to backward pass buffers
  const std::vector<CuMatrix<BaseFloat> >& BackpropagateBuffer() const { 
    return backpropagate_buf_; 
  }

  /// Get the number of parameters in the network
  int32 NumParams() const;
  /// Get the network weights in a supervector
  void GetParams(Vector<BaseFloat>* wei_copy) const;
  /// Appends this layer to the layers already in the neural net.
  void AppendLayer(Layer *dynamically_allocated_layer);

  /// Initialize MLP from config
  void Init(const std::string &config_file);
  /// Read the MLP from file (can add layers to exisiting instance of Net)
  void Read(const std::string &file);  
  /// Read the MLP from stream (can add layers to exisiting instance of Net)
  void Read(std::istream &in, bool binary);  
  /// Re-read a MLP from file (of the same structure of current Net)
  void ReRead(const std::string &file);
  /// Re-read a MLP from stream (of the same structure of current Net)
  void ReRead(std::istream &in, bool binary);

  /// Write MLP to file
  void Write(const std::string &file, bool binary) const;
  /// Write MLP to stream 
  void Write(std::ostream &out, bool binary) const;   
 
  /// Write MLP to file. The parallel version of a layer (BiLstmParallel) is saved
  /// into a non-parallel version (BiLstm).
  void WriteNonParal(const std::string &file, bool binary) const;
  /// Write MLP to stream. The parallel version of a layer (BiLstmParallel) is saved
  /// into a non-parallel version (BiLstm).
  void WriteNonParal(std::ostream &out, bool binary) const;
 
  /// Create string with human readable description of the nnet
  std::string Info() const;
  /// Create string with per-layer gradient statistics
  std::string InfoGradient() const;

  /// Scale the weights
  void Scale(BaseFloat scale);
  /// Add another net to current net
  void AddNet(BaseFloat scale, Net &net_other);

  /// Consistency check.
  void Check() const;
  /// Relese the memory
  void Destroy();

  /// Set training hyper-parameters to the network and its TrainableLayer(s)
  void SetTrainOptions(const NetTrainOptions& opts);
  /// Get training hyper-parameters from the network
  const NetTrainOptions& GetTrainOptions() const {
    return opts_;
  }

  // Set lengths of utterances for LSTM parallel training
  void SetSeqLengths(std::vector<int> &sequence_lengths) { 
    for(int32 i=0; i < (int32)layers_.size(); i++) {
        layers_[i]->SetSeqLengths(sequence_lengths);
    }
  }

  std::vector<int> GetBlockSoftmaxDims();

 private:
  /// Vector which contains all the layers composing the neural network,
  /// the layers are for example: AffineTransform, Sigmoid, Softmax
  std::vector<Layer*> layers_; 

  std::vector<CuMatrix<BaseFloat> > propagate_buf_; ///< buffers for forward pass
  std::vector<CuMatrix<BaseFloat> > backpropagate_buf_; ///< buffers for backward pass

  /// Option class with hyper-parameters passed to TrainableLayer(s)
  NetTrainOptions opts_;
};
  

} // namespace eesen

#endif  // EESEN_NNET_H_

