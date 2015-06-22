#!/bin/bash

# Copyright 2015  Yajie Miao   (Carnegie Mellon University)

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
mkdir -p $dir

[ -f path.sh ] && . ./path.sh

# Use the word list of the phoneme-based lexicon. Create the lexicon using characters.
cat $phndir/lexicon2_raw_nosil.txt | awk '{print $1}' | \
  perl -e 'while(<>){ chop; $str="$_"; foreach $p (split("", $_)) {$str="$str $p"}; print "$str\n";}' \
  > $dir/lexicon2_raw_nosil.txt

#  Get the set of lexicon units without noises
cut -d' ' -f2- $dir/lexicon2_raw_nosil.txt | tr ' ' '\n' | sort -u > $dir/units_nosil.txt

# Add special noises words & characters into the lexicon.  To be consistent with the blank <blk>,
# we add "< >" to the noises characters
(echo '<SPOKEN_NOISE> <SPOKEN_NOISE>'; echo '<UNK> <UNK>'; echo '<NOISE> <NOISE>'; echo '<SPACE> <SPACE>';) | \
 cat - $dir/lexicon2_raw_nosil.txt | sort | uniq > $dir/lexicon.txt || exit 1;

#  The complete set of lexicon units, indexed by numbers starting from 1
(echo '<NOISE>'; echo '<SPOKEN_NOISE>'; echo '<SPACE>'; echo '<UNK>'; ) | cat - $dir/units_nosil.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert character sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Character-based dictionary (word spelling) preparation succeeded"
