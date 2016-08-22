// net/net.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)
//                2015  Yajie Miao, Hang Su
//
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

#include "net/net.h"
#include "net/layer.h"
#include "net/trainable-layer.h"
//#include "net/sigmoid-layer.h"
//#include "net/tanh-layer.h"
#include "net/softmax-layer.h"
#include "net/affine-trans-layer.h"
#include "net/utils-functions.h"


namespace eesen {

Net::Net(const Net& other) {
  // copy the layers
  for(int32 i=0; i<other.NumLayers(); i++) {
    layers_.push_back(other.GetLayer(i).Copy());
  }
  // create empty buffers
  propagate_buf_.resize(NumLayers()+1);
  backpropagate_buf_.resize(NumLayers()+1);
  // copy train opts
  SetTrainOptions(other.opts_);
  Check(); 
}

Net & Net::operator = (const Net& other) {
  Destroy();
  // copy the components
  for(int32 i=0; i<other.NumLayers(); i++) {
    layers_.push_back(other.GetLayer(i).Copy());
  }
  // create empty buffers
  propagate_buf_.resize(NumLayers()+1);
  backpropagate_buf_.resize(NumLayers()+1);
  // copy train opts
  SetTrainOptions(other.opts_); 
  Check();
  return *this;
}


Net::~Net() {
  Destroy();
}


void Net::Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
  KALDI_ASSERT(NULL != out);

  if (NumLayers() == 0) {
    (*out) = in; // copy 
    return; 
  }

  // we need at least L+1 input buffers
  KALDI_ASSERT((int32)propagate_buf_.size() >= NumLayers()+1);
  
  propagate_buf_[0].Resize(in.NumRows(), in.NumCols());
  propagate_buf_[0].CopyFromMat(in);

  for(int32 i=0; i<(int32)layers_.size(); i++) {
    layers_[i]->Propagate(propagate_buf_[i], &propagate_buf_[i+1]);
  }
  
  (*out) = propagate_buf_[layers_.size()];
}

void Net::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {

  if (NumLayers() == 0) { (*in_diff) = out_diff; return; }

  KALDI_ASSERT((int32)propagate_buf_.size() == NumLayers()+1);
  KALDI_ASSERT((int32)backpropagate_buf_.size() == NumLayers()+1);

  // copy out_diff to last buffer
  backpropagate_buf_[NumLayers()] = out_diff;
  // backpropagate using buffers
  for (int32 i = NumLayers()-1; i >= 0; i--) {
    layers_[i]->Backpropagate(propagate_buf_[i], propagate_buf_[i+1],
                              backpropagate_buf_[i+1], &backpropagate_buf_[i]);
    if (layers_[i]->IsTrainable()) {
      TrainableLayer *tl = dynamic_cast<TrainableLayer*>(layers_[i]);
      tl->Update(propagate_buf_[i], backpropagate_buf_[i+1]);
    }
  }
  // eventually export the derivative
  if (NULL != in_diff) (*in_diff) = backpropagate_buf_[0];
}

void Net::Feedforward(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
  KALDI_ASSERT(NULL != out);

  if (NumLayers() == 0) { 
    out->Resize(in.NumRows(), in.NumCols());
    out->CopyFromMat(in); 
    return; 
  }

  if (NumLayers() == 1) {
    layers_[0]->Propagate(in, out);
    return;
  }

  // we need at least 2 input buffers
  KALDI_ASSERT(propagate_buf_.size() >= 2);

  // propagate by using exactly 2 auxiliary buffers
  int32 L = 0;
  layers_[L]->Propagate(in, &propagate_buf_[L%2]);
  for(L++; L<=NumLayers()-2; L++) {
    layers_[L]->Propagate(propagate_buf_[(L-1)%2], &propagate_buf_[L%2]);
  }
  layers_[L]->Propagate(propagate_buf_[(L-1)%2], out);
  // release the buffers we don't need anymore
  propagate_buf_[0].Resize(0,0);
  propagate_buf_[1].Resize(0,0);
}


int32 Net::OutputDim() const {
  KALDI_ASSERT(!layers_.empty());
  return layers_.back()->OutputDim();
}

int32 Net::InputDim() const {
  KALDI_ASSERT(!layers_.empty());
  return layers_.front()->InputDim();
}

const Layer& Net::GetLayer(int32 layer) const {
  KALDI_ASSERT(static_cast<size_t>(layer) < layers_.size());
  return *(layers_[layer]);
}

Layer& Net::GetLayer(int32 layer) {
  KALDI_ASSERT(static_cast<size_t>(layer) < layers_.size());
  return *(layers_[layer]);
}

void Net::SetLayer(int32 c, Layer *layer) {
  KALDI_ASSERT(static_cast<size_t>(c) < layers_.size());
  delete layers_[c];
  layers_[c] = layer;
  Check(); // Check that all the dimensions still match up.
}

void Net::RemoveLayer(int32 layer) {
  KALDI_ASSERT(layer < NumLayers());
  // remove,
  Layer* ptr = layers_[layer];
  layers_.erase(layers_.begin()+layer);
  delete ptr;
  // create training buffers,
  propagate_buf_.resize(NumLayers()+1);
  backpropagate_buf_.resize(NumLayers()+1);
  // 
  Check();
}


void Net::GetParams(Vector<BaseFloat>* wei_copy) const {
  wei_copy->Resize(NumParams());
  int32 pos = 0;
  // copy the params
  for(int32 i=0; i<layers_.size(); i++) {
    if(layers_[i]->IsTrainable()) {
      TrainableLayer& tl = dynamic_cast<TrainableLayer&>(*layers_[i]);
      Vector<BaseFloat> c_params; 
      tl.GetParams(&c_params);
      wei_copy->Range(pos,c_params.Dim()).CopyFromVec(c_params);
      pos += c_params.Dim();
    }
  }
  KALDI_ASSERT(pos == NumParams());
}

std::vector<int> Net::GetBlockSoftmaxDims() {
  KALDI_ASSERT(layers_[layers_.size()-1]->GetType() == Layer::l_BlockSoftmax);
  return dynamic_cast<const BlockSoftmax*>(layers_[layers_.size()-1])->block_dims;
}

void Net::AppendLayer(Layer* dynamically_allocated_layer) {
  // append,
  layers_.push_back(dynamically_allocated_layer);
  // create training buffers,
  propagate_buf_.resize(NumLayers()+1);
  backpropagate_buf_.resize(NumLayers()+1);
  //
  Check();
}

int32 Net::NumParams() const {
  int32 n_params = 0;
  for(int32 n=0; n<layers_.size(); n++) {
    if(layers_[n]->IsTrainable()) {
      n_params += dynamic_cast<TrainableLayer*>(layers_[n])->NumParams();
    }
  }
  return n_params;
}

void Net::Init(const std::string &file) {
  Input in(file);
  std::istream &is = in.Stream();
  // do the initialization with config lines,
  std::string conf_line, token;
  while (!is.eof()) {
    KALDI_ASSERT(is.good());
    std::getline(is, conf_line); // get a line from config file,
    if (conf_line == "") continue;
    KALDI_VLOG(1) << conf_line; 
    std::istringstream(conf_line) >> std::ws >> token; // get 1st token,
    if (token == "<Nnet>" || token == "</Nnet>") continue; // ignored tokens,
    AppendLayer(Layer::Init(conf_line+"\n"));
    is >> std::ws;
  }
  // cleanup
  in.Close();
  Check();
}

void Net::Read(const std::string &file) {
  bool binary;
  Input in(file, &binary);
  Read(in.Stream(), binary);
  in.Close();
  // Warn if the NN is empty
  if(NumLayers() == 0) {
    KALDI_WARN << "The network '" << file << "' is empty.";
  }
}


void Net::Read(std::istream &is, bool binary) {
  // get the network layers from a factory
  Layer *layer;
  while (NULL != (layer = Layer::Read(is, binary))) {
    if (NumLayers() > 0 && layers_.back()->OutputDim() != layer->InputDim()) {
      KALDI_ERR << "Dimensionality mismatch!"
                << " Previous layer output:" << layers_.back()->OutputDim()
                << " Current layer input:" << layer->InputDim();
    }
    layers_.push_back(layer);
  }
  // create empty buffers
  propagate_buf_.resize(NumLayers()+1);
  backpropagate_buf_.resize(NumLayers()+1);
  // reset learn rate
  opts_.learn_rate = 0.0;
  
  Check(); //check consistency (dims...)
}

void Net::ReRead(const std::string &file) {
  bool binary;
  Input in(file, &binary);
  ReRead(in.Stream(), binary);
  in.Close();
  // Warn if the NN is empty
  if(NumLayers() == 0) {
    KALDI_WARN << "The network '" << file << "' is empty.";
  }
}


void Net::ReRead(std::istream &is, bool binary) {
  // get the network layers from a factory
  for(int32 i=0; i<NumLayers(); i++) {
    layers_[i]->ReRead(is, binary);
  }
}

void Net::Write(const std::string &file, bool binary) const {
  Output out(file, binary, true);
  Write(out.Stream(), binary);
  out.Close();
}


void Net::Write(std::ostream &os, bool binary) const {
  Check();
  WriteToken(os, binary, "<Nnet>");
  if(binary == false) os << std::endl;
  for(int32 i=0; i<NumLayers(); i++) {
    layers_[i]->Write(os, binary);
  }
  WriteToken(os, binary, "</Nnet>");  
  if(binary == false) os << std::endl;
}


void Net::WriteNonParal(const std::string &file, bool binary) const {
  Output out(file, binary, true);
  WriteNonParal(out.Stream(), binary);
  out.Close();
}


void Net::WriteNonParal(std::ostream &os, bool binary) const {
  Check();
  WriteToken(os, binary, "<Nnet>");
  if(binary == false) os << std::endl;
  for(int32 i=0; i<NumLayers(); i++) {
    layers_[i]->WriteNonParal(os, binary);
  }
  WriteToken(os, binary, "</Nnet>");
  if(binary == false) os << std::endl;
}


std::string Net::Info() const {
  // global info
  std::ostringstream ostr;
  ostr << "num-layers " << NumLayers() << std::endl;
  ostr << "input-dim " << InputDim() << std::endl;
  ostr << "output-dim " << OutputDim() << std::endl;
  ostr << "number-of-parameters " << static_cast<float>(NumParams())/1e6 
       << " millions" << std::endl;
  // topology & weight stats
  for (int32 i = 0; i < NumLayers(); i++) {
    ostr << "layer " << i+1 << " : " 
         << Layer::TypeToMarker(layers_[i]->GetType()) 
         << ", input-dim " << layers_[i]->InputDim()
         << ", output-dim " << layers_[i]->OutputDim()
         << ", " << layers_[i]->Info() << std::endl;
  }
  return ostr.str();
}

std::string Net::InfoGradient() const {
  std::ostringstream ostr;
  // gradient stats
  ostr << "### Gradient stats :\n";
  for (int32 i = 0; i < NumLayers(); i++) {
    ostr << "Layer " << i+1 << " : " 
         << Layer::TypeToMarker(layers_[i]->GetType()) 
         << ", " << layers_[i]->InfoGradient() << std::endl;
  }
  return ostr.str();
}

void Net::Scale(BaseFloat scale) {
  for(int32 i=0; i < (int32)layers_.size(); i++) {
    if (layers_[i]->IsTrainable()) {
      TrainableLayer *tl = dynamic_cast<TrainableLayer*>(layers_[i]);
      tl->Scale(scale);
    }
  }
}

void Net::AddNet(BaseFloat scale, Net &net_other) {
  KALDI_ASSERT(net_other.NumLayers() == NumLayers());

  for(int32 i=0; i < (int32)layers_.size(); i++) {
    if (layers_[i]->IsTrainable()) {
      TrainableLayer *tl = dynamic_cast<TrainableLayer*>(layers_[i]);
      TrainableLayer *tl_other = dynamic_cast<TrainableLayer*>(&(net_other.GetLayer(i)));
      tl->Add(scale, *tl_other);
    }
  }
}

void Net::Check() const {
  // check we have correct number of buffers,
  KALDI_ASSERT(propagate_buf_.size() == NumLayers()+1)
  KALDI_ASSERT(backpropagate_buf_.size() == NumLayers()+1)
  // check dims,
  for (size_t i = 0; i + 1 < layers_.size(); i++) {
    KALDI_ASSERT(layers_[i] != NULL);
    int32 output_dim = layers_[i]->OutputDim(),
      next_input_dim = layers_[i+1]->InputDim();
    KALDI_ASSERT(output_dim == next_input_dim);
  }
  // check for nan/inf in network weights,
  Vector<BaseFloat> weights;
  GetParams(&weights);
  BaseFloat sum = weights.Sum();
  if(KALDI_ISINF(sum)) {
    KALDI_ERR << "'inf' in network parameters";
  }
  if(KALDI_ISNAN(sum)) {
    KALDI_ERR << "'nan' in network parameters";
  }
}


void Net::Destroy() {
  for(int32 i=0; i<NumLayers(); i++) {
    delete layers_[i];
  }
  layers_.resize(0);
  propagate_buf_.resize(0);
  backpropagate_buf_.resize(0);
}


void Net::SetTrainOptions(const NetTrainOptions& opts) {
  opts_ = opts;
  //set values to individual components
  for (int32 l=0; l<NumLayers(); l++) {
    if(GetLayer(l).IsTrainable()) {
      dynamic_cast<TrainableLayer&>(GetLayer(l)).SetTrainOptions(opts_);
    }
  }
}

} // namespace eesen
