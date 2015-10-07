// lat/lattice-functions.h

// Copyright 2009-2012   Saarland University (author: Arnab Ghoshal)
//           2012-2013   Johns Hopkins University (Author: Daniel Povey);
//                       Bagher BabaAli
//                2014   Guoguo Chen

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


#ifndef KALDI_LAT_LATTICE_FUNCTIONS_H_
#define KALDI_LAT_LATTICE_FUNCTIONS_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "decoder/decodable-itf.h"

namespace eesen {

/// This function iterates over the states of a topologically sorted lattice and
/// counts the time instance corresponding to each state. The times are returned
/// in a vector of integers 'times' which is resized to have a size equal to the
/// number of states in the lattice. The function also returns the maximum time
/// in the lattice (this will equal the number of frames in the file).
int32 LatticeStateTimes(const Lattice &lat, std::vector<int32> *times);

/// As LatticeStateTimes, but in the CompactLattice format.  Note: must
/// be topologically sorted.  Returns length of the utterance in frames, which
/// may not be the same as the maximum time in the lattice, due to frames
/// in the final-prob.
int32 CompactLatticeStateTimes(const CompactLattice &clat,
                               std::vector<int32> *times);

/// Topologically sort the compact lattice if not already topologically sorted.
/// Will crash if the lattice cannot be topologically sorted.
void TopSortCompactLatticeIfNeeded(CompactLattice *clat);


/// Topologically sort the lattice if not already topologically sorted.
/// Will crash if lattice cannot be topologically sorted.
void TopSortLatticeIfNeeded(Lattice *clat);

/// Returns the depth of the lattice, defined as the average number of arcs (or
/// final-prob strings) crossing any given frame.  Returns 1 for empty lattices.
/// Requires that clat is topologically sorted!
BaseFloat CompactLatticeDepth(const CompactLattice &clat,
                              int32 *num_frames = NULL);

/// This function returns, for each frame, the number of arcs crossing that
/// frame.
void CompactLatticeDepthPerFrame(const CompactLattice &clat,
                                 std::vector<int32> *depth_per_frame);


/// This function limits the depth of the lattice, per frame: that means, it
/// does not allow more than a specified number of arcs active on any given
/// frame.  This can be used to reduce the size of the "very deep" portions of
/// the lattice.
void CompactLatticeLimitDepth(int32 max_arcs_per_frame,
                              CompactLattice *clat);


/// Prunes a lattice or compact lattice.  Returns true on success, false if
/// there was some kind of failure.
template<class LatticeType>
bool PruneLattice(BaseFloat beam, LatticeType *lat);


/// This function takes a CompactLattice that should only contain a single
/// linear sequence (e.g. derived from lattice-1best), and that should have been
/// processed so that the arcs in the CompactLattice align correctly with the
/// word boundaries (e.g. by lattice-align-words).  It outputs 3 vectors of the
/// same size, which give, for each word in the lattice (in sequence), the word
/// label and the begin time and length in frames.  This is done even for zero
/// (epsilon) words, generally corresponding to optional silence-- if you don't
/// want them, just ignore them in the output.
/// This function will print a warning and return false, if the lattice
/// did not have the correct format (e.g. if it is empty or it is not
/// linear).
bool CompactLatticeToWordAlignment(const CompactLattice &clat,
                                   std::vector<int32> *words,
                                   std::vector<int32> *begin_times,
                                   std::vector<int32> *lengths);

/// A form of the shortest-path/best-path algorithm that's specially coded for
/// CompactLattice.  Requires that clat be acyclic.
void CompactLatticeShortestPath(const CompactLattice &clat,
                                CompactLattice *shortest_path);

/// This function add the word insertion penalty to graph score of each word
/// in the compact lattice
void AddWordInsPenToCompactLattice(BaseFloat word_ins_penalty,
                                   CompactLattice *clat);

/// This function *adds* the negated scores obtained from the Decodable object,
/// to the acoustic scores on the arcs.  If you want to replace them, you should
/// use ScaleCompactLattice to first set the acoustic scores to zero.  Returns
/// true on success, false on error (typically some kind of mismatched inputs).
//bool RescoreCompactLattice(DecodableInterface *decodable,
//                           CompactLattice *clat);


/// This function returns the number of words in the longest sentence in a
/// CompactLattice (i.e. the the maximum of any path, of the count of
/// olabels on that path).
int32 LongestSentenceLength(const Lattice &lat);

/// This function returns the number of words in the longest sentence in a
/// CompactLattice, i.e. the the maximum of any path, of the count of
/// labels on that path... note, in CompactLattice, the ilabels and olabels
/// are identical because it is an acceptor.
int32 LongestSentenceLength(const CompactLattice &lat);


/// This function *adds* the negated scores obtained from the Decodable object,
/// to the acoustic scores on the arcs.  If you want to replace them, you should
/// use ScaleCompactLattice to first set the acoustic scores to zero.  Returns
/// true on success, false on error (e.g. some kind of mismatched inputs).
/// The input labels, if nonzero, are interpreted as transition-ids or whatever
/// other index the Decodable object expects.
bool RescoreLattice(DecodableInterface *decodable,
                    Lattice *lat);

/// This function Composes a CompactLattice format lattice with a
/// DeterministicOnDemandFst<fst::StdFst> format fst, and outputs another
/// CompactLattice format lattice. The first element (the one that corresponds
/// to LM weight) in CompactLatticeWeight is used for composition.
///
/// Note that the DeterministicOnDemandFst interface is not "const", therefore
/// we cannot use "const" for <det_fst>.
void ComposeCompactLatticeDeterministic(
    const CompactLattice& clat,
    fst::DeterministicOnDemandFst<fst::StdArc>* det_fst,
    CompactLattice* composed_clat);

}  // namespace eesen

#endif  // KALDI_LAT_LATTICE_FUNCTIONS_H_
