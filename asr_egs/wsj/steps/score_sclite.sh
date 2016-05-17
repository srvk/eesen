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

hubscr=$EESEN_ROOT/tools/sctk/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $data/stm $data/glm $symtab $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval2000

mkdir -p $dir/scoring/log

# We are not using lattice-align-words, which may result in minor degradation 
if [ $stage -le 0 ]; then
  $cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/get_ctm.ACWT.log \
    mkdir -p $dir/score_ACWT/ '&&' \
    lattice-1best --acoustic-scale=ACWT --ascale-factor=$acwt_factor "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
    nbest-to-ctm ark:- - \| \
    utils/int2sym.pl -f 5 $symtab  \| \
    utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
    '>' $dir/score_ACWT/$name.ctm || exit 1;
fi

if [ $stage -le 1 ]; then
  # Remove some stuff we don't want to score, from the ctm.
  for x in $dir/score_*/$name.ctm; do
    cp $x $dir/tmpf;
    cat $dir/tmpf | grep -i -v -E '\[BREATH|NOISE|COUGH|SMACK|UM|UH\]' | \
      grep -i -v -E '<UNK>' > $x;
  done
fi

# Score the set...
if [ $stage -le 2 ]; then
  $cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/score.ACWT.log \
    cp $data/stm $dir/score_ACWT/ '&&' \
    $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/score_ACWT/stm $dir/score_ACWT/${name}.ctm || exit 1;
fi

exit 0;
