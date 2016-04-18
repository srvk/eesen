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
# and token FSTs together into the decoding graph. 

# define so it can be used as an arg to parse_options.sh
# overridden if arg passed in

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
   echo "Usage: $0 [options] <data-dir>";
   echo "e.g.: $0 data/lang"
   exit 1;
fi

langdir=$1

lm=lm_build/local_lm/3gram-mincount/lm_pr6.0.gz
tmpdir=data/local/graph_tmp
mkdir -p $tmpdir

# These language models have been obtained when you run local/wsj_data_prep.sh
echo "Preparing language models for testing, may take some time ... "

suffix=newlm
for lm_suffix in ${suffix}; do
    test=${langdir}_test_${lm_suffix}
    mkdir -p $test
    cp ${langdir}/words.txt $test || exit 1;

    gunzip -c $lm | \
     utils/find_arpa_oovs.pl $test/words.txt  > $tmpdir/oovs_${lm_suffix}.txt

    # grep -v '<s> <s>' because the LM seems to have some strange and useless
    # stuff in it with multiple <s>'s in the history.
    gunzip -c $lm | \
      grep -v '<s> <s>' | \
      grep -v '</s> <s>' | \
      grep -v '</s> </s>' | \
      grep -v '<UNK>' | \
       arpa2fst - | fstprint | \
      utils/remove_oovs.pl $tmpdir/oovs_${lm_suffix}.txt | \
      utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$test/words.txt \
	--osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
      fstrmepsilon | fstarcsort --sort_type=ilabel > $test/G.fst
    
    # Compose the final decoding graph. The composition of L.fst and G.fst is determinized and
    # minimized.
    fsttablecompose ${langdir}/L.fst $test/G.fst | fstdeterminizestar --use-log=true | \
      fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst || exit 1;
    fsttablecompose ${langdir}/T.fst $tmpdir/LG.fst > $test/TLG.fst || exit 1;
done

echo "Composing decoding graph TLG.fst succeeded"
#rm -r $tmpdir
