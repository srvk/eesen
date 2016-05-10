#!/bin/bash
#
# Copyright  2015   Yajie Miao 
# Apache 2.0
#

dir=data/local/dict_phn
mkdir -p $dir

srcdict=db/cantab-TEDLIUM/cantab-TEDLIUM.dct

. parse_options.sh

[ ! -r $srcdict ] && echo "Missing $srcdict" && exit 1

# Join dicts and fix some troubles
cat $srcdict | grep -v "<s>" | grep -v "</s>" | LANG= LC_ALL= sort | sed 's:([0-9])::g' > $dir/lexicon_words.txt 

# Get the set of lexicon units without noises
cut -d' ' -f2- $dir/lexicon_words.txt | tr ' ' '\n' | sort -u > $dir/units_nosil.txt

# Add to the lexicon the silences, noises etc.
(echo '[BREATH] BRH'; echo '[NOISE] NSN'; echo '[COUGH] CGH';
 echo '[SMACK] SMK'; echo '[UM] UM'; echo '[UH] UHH'
 echo '<UNK> NSN' ) | \
 cat - $dir/lexicon_words.txt | sort | uniq > $dir/lexicon.txt

# The complete set of lexicon units, indexed by numbers starting from 1
( echo BRH; echo CGH; echo NSN ; echo SMK; echo UM; echo UHH ) | cat - $dir/units_nosil.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

