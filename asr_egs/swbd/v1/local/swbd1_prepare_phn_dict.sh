#!/bin/bash

# This script prepares the phoneme-based lexicon. It also generates the list of lexicon units
# and represents the lexicon using the indices of the units. 

srcdir=data/local/train
dir=data/local/dict_phn
mkdir -p $dir
srcdict=$srcdir/swb_ms98_transcriptions/sw-ms98-dict.text

[ -f path.sh ] && . ./path.sh

. utils/parse_options.sh

[ ! -f "$srcdict" ] && echo "No such file $srcdict" && exit 1;

# Raw dictionary preparation (lower-case, remove comments)
awk 'BEGIN{getline}($0 !~ /^#/) {$0=tolower($0); print}' \
  $srcdict | sort | awk '($0 !~ /^[[:space:]]*$/) {print}' | \
  perl -e 'while(<>){ chop; $_=~ s/ +/ /; $_=~ s/\s*$//; print "$_\n";}' \
   > $dir/lexicon1.txt || exit 1;

# Get the set of lexicon units without noises
cut -d' ' -f2- $dir/lexicon1.txt | tr ' ' '\n' | sort -u > $dir/units_nosil.txt

# Add the noises etc. to the lexicon. No silence is added.
(echo '[vocalized-noise] spn'; echo '[noise] nsn'; echo '[laughter] lau'; echo '<unk> spn'; ) | \
 cat - $dir/lexicon1.txt | sort | uniq > $dir/lexicon2.txt || exit 1;

local/swbd1_map_words.pl -f 1 $dir/lexicon2.txt | sort -u > $dir/lexicon.txt || exit 1;

# The complete set of lexicon units, indexed by numbers starting from 1
(echo 'spn'; echo 'nsn'; echo 'lau';) | cat - $dir/units_nosil.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Phoneme-based dictionary preparation succeeded"
