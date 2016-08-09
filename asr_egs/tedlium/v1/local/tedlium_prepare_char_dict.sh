#!/bin/bash
#
# Copyright 2015 Yajie Miao
# Copyright 2016 Carnegie Mellon University (Florian Metze)
# Apache 2.0

# Creates a lexicon in which each word is represented by the sequence of its characters (spelling). 

phndir=""
#data/local/dict_phn
dir=data/local/dict_char
txt=data/train/text
lm=""
mkdir -p $dir

. parse_options.sh

[ -f path.sh ] && . ./path.sh

if [ -n "$phndir" -a -f "$phndir/lexicon_words.txt" ]; then
    # Use the word list of the phoneme-based lexicon. Create the lexicon using characters.
    cat $phndir/lexicon_words.txt | awk '{print $1}' | \
	perl -e 'while(<>){ chop; $str="$_"; foreach $p (split("", $_)) {$str="$str $p"}; print "$str\n";}' \
	     > $dir/lexicon_words.txt
elif [ -n "$lm" -a -f "$lm" ]; then
    # Use wordlist from language model
    echo no phndir given, creating dictionary from vocab in $lm
    awk '{$1=""; sub(" ",""); print tolower($0)}' $txt|tr ' ' '\n'|sort|uniq -c|sort -rgk 1 > $dir/lexicon_freq.txt
    awk '! /-$/ {if ($1>1) {print $2}}' $dir/lexicon_freq.txt|sort > $dir/lexicon_train.txt
    gzip -cd $lm | awk '/1-grams/,/2-grams/ {if (NF>1) {print $2}}' | grep -v "<" > $dir/lexicon_test.txt
    cat $dir/lexicon_test.txt $dir/lexicon_train.txt | sort -u | awk '/^\[|^\+|^%/ {print $1,$1} ! /^\[|^\+|^%/ { a=$1; gsub(""," ",a); gsub ("^ | $","",a); print $1,a}' > $dir/lexicon_words.txt
else
    # No word list given, make it up ourselves
    echo no phndir given, creating dictionary from $txt
    awk '{$1=""; sub(" ",""); print tolower($0)}' $txt|tr ' ' '\n'|sort|uniq -c|sort -rgk 1 > $dir/lexicon_freq.txt
    awk '! /-$/ {if ($1>1) {print $2}}' $dir/lexicon_freq.txt|sort > $dir/lexicon_train.txt
    cat $dir/lexicon_train.txt | sort -u | awk '/^\[|^\+|^%/ {print $1,$1} ! /^\[|^\+|^%/ { a=$1; gsub(""," ",a); gsub ("^ | $","",a); print $1,a}' > $dir/lexicon_words.txt
fi

#  Get the set of lexicon units without noises
cut -d' ' -f2- $dir/lexicon_words.txt | tr ' ' '\n' | sort -u > $dir/units_nosil.txt

# Add special noises words & characters into the lexicon.
( echo '<UNK> NSN';) | \
  cat - $dir/lexicon_words.txt | sort | uniq > $dir/lexicon.txt || exit 1;

# The complete set of lexicon units, indexed by numbers starting from 1
(echo 'NSN'; ) | cat - $dir/units_nosil.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Character-based dictionary (word spelling) preparation succeeded"
