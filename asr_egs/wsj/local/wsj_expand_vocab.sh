#!/bin/bash

# This script expands the word list by including OOVs from the training transcripts.
# New entries are added to the character-based  lexicon simply by  getting the words
# and their spellings. Since no phonemes are involved, we need no G2P models/rules.
# The expanded lexicon is saved in a separate directory.

if [ $# -ne 3 ]; then
  echo "usage: local/wsj_expand_vocab.sh <wsj-corpus-dir> <dict-src-dir> <dict-larger-dir>"
  echo "e.g.: local/wsj_expand_vocab.sh /foo/bar/WSJ/13-32.1/ data/local/lang_char data/local/lang_char_larger"
  exit 1;
fi

if [ "`basename $1`" != 13-32.1 ]; then
  echo "Expecting the first argument to this script to end in 13-32.1"
  exit 1
fi

corpusdir=$1
srcdir=$2
dir=$3

mincount=2 # Minimum count of an OOV we include into the lexicon.

mkdir -p $dir
cp $srcdir/lexicon.txt $dir/lexicon.ori.txt
cp $srcdir/units.txt $dir

# the original wordlist
cat $dir/lexicon.ori.txt | awk '{print $1}' | sort | uniq > $dir/wordlist.ori

# Get the training transcripts
echo "Getting the training transcripts, may take some time ..."

touch $dir/cleaned.gz
if [ `du -m $dir/cleaned.gz | cut -f 1` -eq 73 ]; then
  echo "Not getting cleaned data in $dir/cleaned.gz again [already exists]";
else
 gunzip -c $corpusdir/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
  | awk '/^</{next}{print toupper($0)}' | perl -e '
   open(F, "<$ARGV[0]")||die;
   while(<F>){ chop; $isword{$_} = 1; }
   while(<STDIN>) { 
    @A = split(" ", $_); 
    for ($n = 0; $n < @A; $n++) {
      $a = $A[$n];
      if (! $isword{$a} && $a =~ s/^([^\.]+)\.$/$1/) { # nonwords that end in "."
         # and have no other "." in them: treat as period.
         print "$a";
         if ($n+1 < @A) { print "\n"; }
      } else { print "$a "; }
    }
    print "\n";
  }
 ' $dir/wordlist.ori | gzip -c > $dir/cleaned.gz
fi
  
# Get unigram counts and the counts of the oov words
echo "Getting unigram counts"
gunzip -c $dir/cleaned.gz | tr -s ' ' '\n' | \
  awk '{count[$1]++} END{for (w in count) { print count[w], w; }}' | sort -nr > $dir/unigrams

cat $dir/unigrams | awk -v dict=$dir/wordlist.ori \
  'BEGIN{while(getline<dict) seen[$1]=1;} {if(!seen[$2]){print;}}' \
   > $dir/oov.counts

echo "Most frequent unseen unigrams are: "
head $dir/oov.counts

# Select the OOVs whose counts > $mincount. Include these OOVs into the lexicon. 
cat $dir/oov.counts | awk -v thresh=$mincount '{if ($1 >= thresh) { print $2; }}' > $dir/oovlist
cat $dir/oovlist | perl -e 'while(<>){ chop; $str="$_"; foreach $p (split("", $_)) {$str="$str $p"}; print "$str\n";}' > $dir/lexicon.oov.txt

# filter out oov words that have characters not in units.txt
cat $dir/lexicon.oov.txt | awk -v dict=$dir/units.txt \
 'BEGIN{while(getline<dict) seen[$1]=1;} {for(i=2;i<=NF;i++) {if(!seen[$i]){break;}}; if (i==(NF+1)){print;}}' > $dir/lexicon.oov.filt.txt

# THe final expanded lexicon
cat $dir/lexicon.ori.txt $dir/lexicon.oov.filt.txt > $dir/lexicon.txt

# Convert character sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Number of OOVs we handled is `cat $dir/lexicon.oov.filt.txt | wc -l`"
echo "Created the larger lexicon $dir/lexicon.txt"
