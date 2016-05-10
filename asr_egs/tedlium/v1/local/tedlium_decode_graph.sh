#!/bin/bash

# This script compiles the ARPA-formatted language models into FSTs. Finally it composes the LM, lexicon
# and token FSTs together into the decoding graph.

. ./path.sh || exit 1;

arpa_lm=db/cantab-TEDLIUM/cantab-TEDLIUM-pruned.lm3.gz
oov_list=/dev/null

. parse_options.sh || exit 1;

langdir=$1

[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

outlangdir=${langdir}_test

rm -rf $outlangdir
cp -r $langdir $outlangdir

gunzip -c "$arpa_lm" | \
   grep -v '<s> <s>' | \
   grep -v '</s> <s>' | \
   grep -v '</s> </s>' | \
   arpa2fst - | fstprint | \
   utils/remove_oovs.pl $oov_list | \
   utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$outlangdir/words.txt \
     --osymbols=$outlangdir/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon | fstarcsort --sort_type=ilabel > $outlangdir/G.fst

# Compose the final decoding graph. The composition of L.fst and G.fst is determinized and
# minimized.
fsttablecompose $outlangdir/L.fst $outlangdir/G.fst | fstdeterminizestar --use-log=true | \
  fstminimizeencoded | fstarcsort --sort_type=ilabel > $outlangdir/LG.fst || exit 1;
fsttablecompose $outlangdir/T.fst $outlangdir/LG.fst > $outlangdir/TLG.fst || exit 1;
rm -rf $outlangdir/LG.fst

echo "Composing decoding graph TLG.fst succeeded"
