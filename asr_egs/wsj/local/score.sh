#!/bin/bash
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
stage=0
min_acwt=5
max_acwt=10
acwt_factor=0.1   # the scaling factor for the acoustic scale. The scaling factor for acoustic likelihoods
                 # needs to be 0.5 ~1.0. However, the job submission script can only take integers as the
                 # job marker. That's why we set the acwt to be integers (5 ~ 10), but scale them with 0.1
                 # when they are actually used.
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --min_acwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_acwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done

mkdir -p $dir/scoring/log

cat $data/text | sed 's:<UNK>::g' | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' > $dir/scoring/test_filt.txt

$cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/best_path.ACWT.log \
  lattice-scale --acoustic-scale=ACWT --ascale-factor=$acwt_factor  "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
  lattice-best-path --word-symbol-table=$symtab ark:- ark,t:$dir/scoring/ACWT.tra || exit 1;

cat $data/text | sed 's:<UNK>::g' | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' > $dir/scoring/text_filt

for acwt in `seq $min_acwt $max_acwt`; do
  cat $dir/scoring/${acwt}.tra | utils/int2sym.pl -f 2- $symtab | \
    sed 's:<UNK>::g' | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' | \
    compute-wer --text --mode=present ark:$dir/scoring/text_filt  ark,p:-  >& $dir/wer_$acwt || exit 1;
done

exit 0;
