#!/bin/bash

# Copyright 2015  Yajie Miao   (Carnegie Mellon University)
#           2017  Jayadev Billa (librispeech adaptations)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Creates a lexicon in which each word is represented by the sequence of its characters (spelling). In theory, we can build such
# a lexicon from the words in the training transcripts.  However, for  consistent comparison to the phoneme-based system, we use
# the word list from the CMU dictionary. No pronunciations are employed.

# run this from ../
phndir=data/local/dict_phn
dir=data/local/dict_char
#mkdir -p $dir

. utils/parse_options.sh || exit 1;
[ -f path.sh ] && . ./path.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <lm-dir> <dict-dir> <dict-name>"
  echo "e.g.: data/lm data/dict librispeech-lexicon.txt"
  exit 1
fi

lm_dir=$1
dst_dir=$2
dict_name=$3
mkdir -p $lm_dir $dst_dir


if [[ ! -s "$lm_dir/$dict_name" ]]; then
  echo $lm_dir/$dict_name "not found"  || exit 1
fi


# For phoneme based speech recognition we provide the pronunciation so no need to expand to char
# We keep only one pronunciation for each word. Other alternative pronunciations are discarded.
cat $lm_dir/$dict_name | \
  perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; print $_;}}' \
  > $dst_dir/lexicon2_raw_nosil.txt || exit 1;

#  Get the set of lexicon units without noises
perl -nae 'shift @F; foreach my $w (@F) {print $w . "\n"}' $dst_dir/lexicon2_raw_nosil.txt | sort -u > $dst_dir/units_nosil.txt

# Add special noises words & characters into the lexicon.  To be consistent with the blank <blk>,
# we add "< >" to the noises characters
(echo '<SPOKEN_NOISE> <SPOKEN_NOISE>'; echo '<UNK> <UNK>'; echo '<NOISE> <NOISE>'; echo '<SPACE> <SPACE>';) | \
 cat - $dst_dir/lexicon2_raw_nosil.txt | sort | uniq > $dst_dir/lexicon.txt || exit 1;

#  The complete set of lexicon units, indexed by numbers starting from 1
(echo '<NOISE>'; echo '<SPOKEN_NOISE>'; echo '<SPACE>'; echo '<UNK>'; ) | cat - $dst_dir/units_nosil.txt | awk '{print $1 " " NR}' > $dst_dir/units.txt

# Convert character sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dst_dir/units.txt < $dst_dir/lexicon.txt > $dst_dir/lexicon_numbers.txt

echo "Phoneme-based dictionary preparation succeeded"
