// decoder/decodable-matrix.h

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)
// Copyright      2015  Yajie Miao  Carnegie Mellon University
//                      (Modified by Yajie for CTC decoding)      

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

#ifndef KALDI_DECODER_DECODABLE_MATRIX_H_
#define KALDI_DECODER_DECODABLE_MATRIX_H_

#include <vector>
#include "base/kaldi-common.h"
#include "itf/decodable-itf.h"

namespace kaldi {

// Yajie deleted the DecodableMatrixScaledMapped class simply because we don't need it 
// for CTC decoding.
class DecodableMatrixScaled: public DecodableInterface {
 public:
  DecodableMatrixScaled(const Matrix<BaseFloat> &likes,
                        BaseFloat scale): likes_(likes),
                                          scale_(scale) { }
  
  virtual int32 NumFramesReady() const { return likes_.NumRows(); }
  
  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }
  
  // Note, frames are numbered from zero. Here "tid" means token id, the indexes of the
  // CTC label tokens. When we compile the search graph, the tokens are indexed from 1
  // because 0 is always occupied by <eps>. However, in the softmax layer of the RNN
  // model, CTC tokens are indexed from 0. Thus, we simply shift "tid" by 1, to solve
  // the mismatch. 
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_ * likes_(frame, tid-1);
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return likes_.NumCols(); }

 private:
  const Matrix<BaseFloat> &likes_;
  BaseFloat scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixScaled);
};


}  // namespace kaldi

#endif  // KALDI_DECODER_DECODABLE_MATRIX_H_
