#!/bin/bash

# Creates a lexicon in which each word is represented by the sequence of its characters (spelling). 

phndir=data/local/dict_phn
dir=data/local/dict_char
mkdir -p $dir

[ -f path.sh ] && . ./path.sh

# Use the word list of the phoneme-based lexicon. Create the lexicon using characters.
local/swbd1_map_words.pl -f 1 $phndir/lexicon1.txt | awk '{print $1}' | \
  perl -e 'while(<>){ chop; $str="$_"; foreach $p (split("", $_)) {$str="$str $p"}; print "$str\n";}' \
  > $dir/lexicon1.txt

#  Get the set of lexicon units without noises
cut -d' ' -f2- $dir/lexicon1.txt | tr ' ' '\n' | sort -u > $dir/units_nosil.txt

# Add special noises words & characters into the lexicon.
(echo '[vocalized-noise] [vocalized-noise]'; echo '[noise] [noise]'; echo '[laughter] [laughter]'; echo '<unk> <unk>'; echo '<space> <space>';) | \
  cat - $dir/lexicon1.txt | sort | uniq > $dir/lexicon2.txt || exit 1;

cat $dir/lexicon2.txt | sort -u > $dir/lexicon.txt || exit 1;

# The complete set of lexicon units, indexed by numbers starting from 1
(echo '[vocalized-noise]'; echo '[noise]'; echo '[laughter]'; echo '<unk>'; echo '<space>';) | cat - $dir/units_nosil.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Character-based dictionary (word spelling) preparation succeeded"
