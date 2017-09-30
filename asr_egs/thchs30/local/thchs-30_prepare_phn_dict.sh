#!/bin/bash

# This script prepares the phoneme-based lexicon. It also generates the list of lexicon units
# and represents the lexicon using the indices of the units. 

dir=data/dict_phn
mkdir -p $dir
srcdict=data/dict/lexicon.txt

[ -f path.sh ] && . ./path.sh

[ ! -f "$srcdict" ] && echo "No such file $srcdict" && exit 1;

echo ==============================================
echo "    Phoneme-based Dictionary Preparation    "
echo ==============================================

# Raw dictionary preparation
cat $srcdict | grep -v "!SIL" | \
  perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; print $_;}}' \
  > $dir/lexicon.txt || exit 1;

# Get the set of lexicon units without noises
cut -d' ' -f2- $dir/lexicon.txt | tr ' ' '\n' | sort -u  | awk '{print $1 " " NR}' > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Phoneme-based dictionary preparation succeeded"
echo -e "\n"
