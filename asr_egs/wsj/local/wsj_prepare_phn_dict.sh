#!/bin/bash

# Copyright 2015  Yajie Miao  (Carnegie Mellon University) 

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

# This script prepares the phoneme-based lexicon from the CMU dictionary. It also generates
# the list of lexicon units and represents the lexicon using the indices of the units. 

dir=data/local/dict_phn
mkdir -p $dir

[ -f path.sh ] && . ./path.sh

# Get the CMU dictionary. We specify the version 11745 because this is something we used. You can 
# of course remove "-r 11745" to get the latest version.
svn co -r 11745 https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict $dir/cmudict || exit 1;

# Raw dictionary preparation.
grep -v ';;;' $dir/cmudict/cmudict.0.7a | \
 perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; print; }' | \
 perl -e 'while(<>){ chop; $_=~ s/ +/ /; print "$_\n";}' \
  > $dir/lexicon1_raw_nosil.txt || exit 1;

# We keep only one pronunciation for each word. Other alternative pronunciations are discarded.
cat $dir/lexicon1_raw_nosil.txt | \
  perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; print $_;}}' \
  > $dir/lexicon2_raw_nosil.txt || exit 1;

# Get the set of lexicon units without noises
cut -d' ' -f2- $dir/lexicon2_raw_nosil.txt | tr ' ' '\n' | sort -u > $dir/units_nosil.txt

# Add the noises etc. to the lexicon. No silence is added.  To be consistent with the blank <blk>,
# we add "< >" to the 2 noises phonemes
(echo '<SPOKEN_NOISE> <SPN>'; echo '<UNK> <SPN>'; echo '<NOISE> <NSN>'; ) | \
 cat - $dir/lexicon2_raw_nosil.txt | sort | uniq > $dir/lexicon.txt || exit 1;

# The complete set of lexicon units, indexed by numbers starting from 1
(echo '<NSN>'; echo '<SPN>';) | cat - $dir/units_nosil.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Phoneme-based dictionary preparation succeeded"

