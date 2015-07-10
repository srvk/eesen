// nnet/nnet-component.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

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

#include "nnet/nnet-component.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-various.h"
#include "nnet/nnet-bilstm.h"
#include "nnet/nnet-bilstm-parallel.h"

#include <sstream>

namespace kaldi {
namespace nnet1 {

const struct Component::key_value Component::kMarkerMap[] = {
  { Component::kAffineTransform,"<AffineTransform>" },
  { Component::kBiLstm,"<BiLstm>"},
  { Component::kBiLstmParallel,"<BiLstmParallel>"},
  { Component::kLstm,"<Lstm>"},
  { Component::kLstmParallel,"<LstmParallel>"},
  { Component::kSoftmax,"<Softmax>" },
};


const char* Component::TypeToMarker(ComponentType t) {
  int32 N=sizeof(kMarkerMap)/sizeof(kMarkerMap[0]);
  for(int i=0; i<N; i++) {
    if (kMarkerMap[i].key == t) return kMarkerMap[i].value;
  }
  KALDI_ERR << "Unknown type" << t;
  return NULL;
}

Component::ComponentType Component::MarkerToType(const std::string &s) {
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
  return kUnknown;
}


Component* Component::NewComponentOfType(ComponentType comp_type,
                      int32 input_dim, int32 output_dim) {
  Component *ans = NULL;
  switch (comp_type) {
    case Component::kAffineTransform :
      ans = new AffineTransform(input_dim, output_dim); 
      break;
    case Component::kBiLstm :
      ans = new BiLstm(input_dim, output_dim);
      break;
    case Component::kBiLstmParallel :
      ans = new BiLstmParallel(input_dim, output_dim);
      break;
    case Component::kLstm :
      ans = new Lstm(input_dim, output_dim);
      break;
    case Component::kLstmParallel :
      ans = new LstmParallel(input_dim, output_dim);
      break;
    case Component::kSoftmax :
      ans = new Softmax(input_dim, output_dim);
      break;
    case Component::kUnknown :
    default :
      KALDI_ERR << "Missing type: " << TypeToMarker(comp_type);
  }
  return ans;
}


Component* Component::Init(const std::string &conf_line) {
  std::istringstream is(conf_line);
  std::string component_type_string;
  int32 input_dim, output_dim;

  // initialize component w/o internal data
  ReadToken(is, false, &component_type_string);
  ComponentType component_type = MarkerToType(component_type_string);
  ExpectToken(is, false, "<InputDim>");
  ReadBasicType(is, false, &input_dim); 
  ExpectToken(is, false, "<OutputDim>");
  ReadBasicType(is, false, &output_dim);
  Component *ans = NewComponentOfType(component_type, input_dim, output_dim);

  // initialize internal data with the remaining part of config line
  ans->InitData(is);

  return ans;
}


Component* Component::Read(std::istream &is, bool binary) {
  int32 dim_out, dim_in;
  std::string token;

  int first_char = Peek(is, binary);
  if (first_char == EOF) return NULL;

  ReadToken(is, binary, &token);
  // Skip optional initial token
  if(token == "<Nnet>") {
    ReadToken(is, binary, &token); // Next token is a Component
  }
  // Finish reading when optional terminal token appears
  if(token == "</Nnet>") {
    return NULL;
  }

  ReadBasicType(is, binary, &dim_out); 
  ReadBasicType(is, binary, &dim_in);

  Component *ans = NewComponentOfType(MarkerToType(token), dim_in, dim_out);
  ans->ReadData(is, binary);
  return ans;
}


void Component::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, Component::TypeToMarker(GetType()));
  WriteBasicType(os, binary, OutputDim());
  WriteBasicType(os, binary, InputDim());
  if(!binary) os << "\n";
  this->WriteData(os, binary);
}

void Component::WriteNonParal(std::ostream &os, bool binary) const {
  WriteToken(os, binary, Component::TypeToMarker(GetTypeNonParal()));
  WriteBasicType(os, binary, OutputDim());
  WriteBasicType(os, binary, InputDim());
  if(!binary) os << "\n";
  this->WriteData(os, binary);
}

} // namespace nnet1
} // namespace kaldi
