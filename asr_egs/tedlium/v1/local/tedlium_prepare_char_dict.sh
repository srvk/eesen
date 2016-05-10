#!/bin/bash

# Copyright 2015 Yajie Miao
# Apache 2.0

# Creates a lexicon in which each word is represented by the sequence of its characters (spelling). 

phndir=data/local/dict_phn
dir=data/local/dict_char
mkdir -p $dir

[ -f path.sh ] && . ./path.sh

# Use the word list of the phoneme-based lexicon. Create the lexicon using characters.
cat $phndir/lexicon_words.txt | awk '{print $1}' | \
  perl -e 'while(<>){ chop; $str="$_"; foreach $p (split("", $_)) {$str="$str $p"}; print "$str\n";}' \
  > $dir/lexicon_words.txt

#  Get the set of lexicon units without noises
cut -d' ' -f2- $dir/lexicon_words.txt | tr ' ' '\n' | sort -u > $dir/units_nosil.txt

# Add special noises words & characters into the lexicon.
( echo '[BREATH] [BREATH]'; echo '[NOISE] [NOISE]'; echo '[COUGH] [COUGH]';
  echo '[SMACK] [SMACK]'; echo '[UM] [UM]'; echo '[UH] [UH]'; echo '<UNK> <UNK>'; echo '<SPACE> <SPACE>';) | \
  cat - $dir/lexicon_words.txt | sort | uniq > $dir/lexicon.txt || exit 1;

# The complete set of lexicon units, indexed by numbers starting from 1
(echo '[BREATH]'; echo '[NOISE]'; echo '[COUGH]'; echo '[SMACK]'; 
 echo '[UM]'; echo '[UH]'; echo '<SPACE>'; echo '<UNK>'; ) | cat - $dir/units_nosil.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Character-based dictionary (word spelling) preparation succeeded"
