// decoder/decoder-wrappers.h

// Copyright   2014  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_DECODER_DECODER_WRAPPERS_H_
#define KALDI_DECODER_DECODER_WRAPPERS_H_

#include "util/options-itf.h"
#include "decoder/lattice-faster-decoder.h"

// This header contains declarations from various convenience functions that are called
// from binary-level programs such as gmm-decode-faster.cc, gmm-align-compiled.cc, and
// so on.

namespace eesen {

bool DecodeUtteranceLatticeFaster(
    LatticeFasterDecoder &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignments_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr);

} // end namespace eesen.


#endif
