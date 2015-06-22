#!/bin/bash

# Copyright 2015       Yajie Miao    (Carnegie Mellon University)

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

# This script compiles the ARPA-formatted language models into FSTs. Finally it composes the LM, lexicon
# and token FSTs together into the decoding graph. We only showcase the trigram LMs. 

. ./path.sh

langdir=$1

lm_srcdir_3g=data/local/local_lm/3gram-mincount
tmpdir=data/local/graph_tmp

mkdir -p $tmpdir

[ ! -d "$lm_srcdir_3g" ] && echo "No such dir $lm_srcdir_3g" && exit 1;

for d in ${langdir}_test_{tg,tgpr}; do
  rm -r $d 2>/dev/null
  mkdir -p $d; cp $langdir/words.txt $d
done

# The pruned LM
gunzip -c $lm_srcdir_3g/lm_pr6.0.gz | \
  grep -v '<s> <s>' | \
  grep -v '</s> <s>' | \
  grep -v '</s> </s>' | \
  arpa2fst - | fstprint | \
    utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$langdir/words.txt \
      --osymbols=$langdir/words.txt  --keep_isymbols=false --keep_osymbols=false | \
     fstrmepsilon | fstarcsort --sort_type=ilabel > ${langdir}_test_tgpr/G.fst || exit 1;
  fstisstochastic ${langdir}_test_tgpr/G.fst

# compose into decoding graph
fsttablecompose ${langdir}/L.fst ${langdir}_test_tgpr/G.fst | fstdeterminizestar --use-log=true | \
  fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst || exit 1;
fsttablecompose ${langdir}/T.fst $tmpdir/LG.fst > ${langdir}_test_tgpr/TLG.fst || exit 1;

rm -rf $tmpdir/*

# The unpruned LM
gunzip -c $lm_srcdir_3g/lm_unpruned.gz | \
  grep -v '<s> <s>' | \
  grep -v '</s> <s>' | \
  grep -v '</s> </s>' | \
  arpa2fst - | fstprint | \
    utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$langdir/words.txt \
      --osymbols=$langdir/words.txt  --keep_isymbols=false --keep_osymbols=false | \
     fstrmepsilon | fstarcsort --sort_type=ilabel > ${langdir}_test_tg/G.fst || exit 1;
  fstisstochastic ${langdir}_test_tg/G.fst

# compose into decoding graph
fsttablecompose ${langdir}/L.fst ${langdir}_test_tg/G.fst | fstdeterminizestar --use-log=true | \
  fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst || exit 1;
fsttablecompose ${langdir}/T.fst $tmpdir/LG.fst > ${langdir}_test_tg/TLG.fst || exit 1;

rm -rf $tmpdir 

exit 0;
