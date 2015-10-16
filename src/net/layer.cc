// net/layer.cc

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

#include "net/layer.h"
#include "net/net.h"
#include "net/softmax-layer.h"
#include "net/sigmoid-layer.h"
#include "net/tanh-layer.h"
#include "net/affine-trans-layer.h"
#include "net/utils-functions.h"
#include "net/bilstm-layer.h"
#include "net/bilstm-parallel-layer.h"
#include "net/lstm-layer.h"
#include "net/lstm-parallel-layer.h"

#include <sstream>

namespace eesen {

const struct Layer::key_value Layer::kMarkerMap[] = {
  { Layer::l_Affine_Transform,"<AffineTransform>" },
  { Layer::l_BiLstm,"<BiLstm>"},
  { Layer::l_BiLstm_Parallel,"<BiLstmParallel>"},
  { Layer::l_Lstm,"<Lstm>"},
  { Layer::l_Lstm_Parallel,"<LstmParallel>"},
  { Layer::l_Softmax,"<Softmax>" },
  { Layer::l_Sigmoid,"<Sigmoid>" },
  { Layer::l_Tanh,"<Tanh>" },
};


const char* Layer::TypeToMarker(LayerType t) {
  int32 N=sizeof(kMarkerMap)/sizeof(kMarkerMap[0]);
  for(int i=0; i<N; i++) {
    if (kMarkerMap[i].key == t) return kMarkerMap[i].value;
  }
  KALDI_ERR << "Unknown type" << t;
  return NULL;
}

Layer::LayerType Layer::MarkerToType(const std::string &s) {
  std::string s_lowercase(s);
  std::transform(s.begin(), s.end(), s_lowercase.begin(), ::tolower); // lc
  int32 N=sizeof(kMarkerMap)/sizeof(kMarkerMap[0]);
  for(int i=0; i<N; i++) {
    std::string m(kMarkerMap[i].value);
    std::string m_lowercase(m);
    std::transform(m.begin(), m.end(), m_lowercase.begin(), ::tolower);
    if (s_lowercase == m_lowercase) return kMarkerMap[i].key;
  }
  KALDI_ERR << "Unknown marker : '" << s << "'";
  return l_Unknown;
}


Layer* Layer::NewLayerOfType(LayerType layer_type,
                             int32 input_dim, int32 output_dim) {
  Layer *layer = NULL;
  switch (layer_type) {
    case Layer::l_Affine_Transform :
      layer = new AffineTransform(input_dim, output_dim); 
      break;
    case Layer::l_BiLstm :
      layer = new BiLstm(input_dim, output_dim);
      break;
    case Layer::l_BiLstm_Parallel :
      layer = new BiLstmParallel(input_dim, output_dim);
      break;
    case Layer::l_Lstm :
      layer = new Lstm(input_dim, output_dim);
      break;
    case Layer::l_Lstm_Parallel :
      layer = new LstmParallel(input_dim, output_dim);
      break;
    case Layer::l_Softmax :
      layer = new Softmax(input_dim, output_dim);
      break;
    case Layer::l_Sigmoid :
      layer = new Sigmoid(input_dim, output_dim);
      break;
    case Layer::l_Tanh :
      layer = new Tanh(input_dim, output_dim);
      break;
    case Layer::l_Unknown :
    default :
      KALDI_ERR << "Missing type: " << TypeToMarker(layer_type);
  }
  return layer;
}

Layer* Layer::Init(const std::string &conf_line) {
  std::istringstream is(conf_line);
  std::string layer_type_string;
  int32 input_dim, output_dim;
 
  // initialize layer
  ReadToken(is, false, &layer_type_string);
  LayerType layer_type = MarkerToType(layer_type_string);
  ExpectToken(is, false, "<InputDim>");
  ReadBasicType(is, false, &input_dim);

  if (IsLstmType(layer_type_string)) {
    ExpectToken(is, false, "<CellDim>");
  } else {
    ExpectToken(is, false, "<OutputDim>");
  }
  ReadBasicType(is, false, &output_dim);
  Layer *layer = NewLayerOfType(layer_type, input_dim, output_dim);

  // initialize internal data with the remaining part of config line
  layer->InitData(is);

  return layer;
}


Layer* Layer::Read(std::istream &is, bool binary) {
  int32 dim_out, dim_in;
  std::string token;

  int first_char = Peek(is, binary);
  if (first_char == EOF) return NULL;

  ReadToken(is, binary, &token);
  // Finish reading when optional terminal token appears
  if(token == "</Nnet>") {
    return NULL;
  }
  // Skip optional initial token
  if(token == "<Nnet>") {
    ReadToken(is, binary, &token);
  }

  ExpectToken(is, binary, "<InputDim>");
  ReadBasicType(is, binary, &dim_in);
  if (IsLstmType(token)) {
    ExpectToken(is, binary, "<CellDim>");
  } else {
    ExpectToken(is, binary, "<OutputDim>");
  }
  ReadBasicType(is, binary, &dim_out);
  
  Layer *layer = NewLayerOfType(MarkerToType(token), dim_in, dim_out);
  layer->ReadData(is, binary);
  return layer;
}

void Layer::ReRead(std::istream &is, bool binary) {
  int32 dim_out, dim_in;
  std::string token;

  int first_char = Peek(is, binary);
  if (first_char == EOF) return;

  ReadToken(is, binary, &token);
  // Finish reading when optional terminal token appears
  if(token == "</Nnet>") return;
  
  // Skip optional initial token
  if(token == "<Nnet>") {
    ReadToken(is, binary, &token);
  }

  ExpectToken(is, binary, "<InputDim>");
  ReadBasicType(is, binary, &dim_in);
  if (IsLstmType(token)) {
    ExpectToken(is, binary, "<CellDim>");
  } else {
    ExpectToken(is, binary, "<OutputDim>");
  }
  ReadBasicType(is, binary, &dim_out);

  KALDI_ASSERT(dim_in == input_dim_);
  KALDI_ASSERT(dim_out == output_dim_);
  
  this->ReadData(is, binary);
}

void Layer::Write(std::ostream &os, bool binary) const {
  std::string layer_type_string = Layer::TypeToMarker(GetType());
  WriteToken(os, binary, layer_type_string);
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, InputDim());
  if (IsLstmType(layer_type_string)) {
    WriteToken(os, binary, "<CellDim>");
  } else {
    WriteToken(os, binary, "<OutputDim>");
  }
  WriteBasicType(os, binary, OutputDim());
  if(!binary) os << "\n";
  this->WriteData(os, binary);
}

void Layer::WriteNonParal(std::ostream &os, bool binary) const {
  std::string layer_type_string = Layer::TypeToMarker(GetTypeNonParal());
  WriteToken(os, binary, layer_type_string);
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, InputDim());
  if (IsLstmType(layer_type_string)) {
    WriteToken(os, binary, "<CellDim>");
  } else {
    WriteToken(os, binary, "<OutputDim>");
  }
  WriteBasicType(os, binary, OutputDim());
  if(!binary) os << "\n";
  this->WriteData(os, binary);
}

} // namespace eesen
