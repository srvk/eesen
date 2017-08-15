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
word_ins_penalty=0.0,0.5,1.0,1.5,2.0
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
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    $cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/get_ctm.p${wip}.ACWT.log \
      mkdir -p $dir/score_p${wip}_ACWT/ '&&' \
      lattice-scale --acoustic-scale=ACWT --ascale-factor=$acwt_factor "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
      lattice-to-ctm-conf --decode-mbr=true ark:- - \| \
      utils/int2sym.pl -f 5 $symtab  \| \
      utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
      '>' $dir/score_p${wip}_ACWT/$name.ctm || exit 1;
  done
fi

if [ $stage -le 1 ]; then
  # Remove some stuff we don't want to score, from the ctm.
  for x in $dir/score_*/$name.ctm; do
    cp $x $dir/tmpf;
    cat $dir/tmpf | grep -i -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
      grep -i -v -E '<UNK>' > $x;
#      grep -i -v -E '<UNK>|%HESITATION' > $x;  # hesitation is scored
  done
fi

# Score the set...
if [ $stage -le 2 ]; then
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    $cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/score.p${wip}.ACWT.log \
      cp $data/stm $dir/score_p${wip}_ACWT/ '&&' \
      $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/score_p${wip}_ACWT/stm $dir/score_p${wip}_ACWT/${name}.ctm || exit 1;
  done
fi

# For eval2000 score the subsets
case "$name" in eval2000* )
  # Score only the, swbd part...
  if [ $stage -le 3 ]; then
    for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
      $cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/score.swbd.p${wip}.ACWT.log \
        grep -v '^en_' $data/stm '>' $dir/score_p${wip}_ACWT/stm.swbd '&&' \
        grep -v '^en_' $dir/score_p${wip}_ACWT/${name}.ctm '>' $dir/score_p${wip}_ACWT/${name}.ctm.swbd '&&' \
        $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/score_p${wip}_ACWT/stm.swbd $dir/score_p${wip}_ACWT/${name}.ctm.swbd || exit 1;
    done
  fi
  # Score only the, callhome part...
  if [ $stage -le 3 ]; then
    for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
      $cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/score.callhm.p${wip}.ACWT.log \
        grep -v '^sw_' $data/stm '>' $dir/score_p${wip}_ACWT/stm.callhm '&&' \
        grep -v '^sw_' $dir/score_p${wip}_ACWT/${name}.ctm '>' $dir/score_p${wip}_ACWT/${name}.ctm.callhm '&&' \
        $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/score_p${wip}_ACWT/stm.callhm $dir/score_p${wip}_ACWT/${name}.ctm.callhm || exit 1;
    done
  fi
 ;;
esac

exit 0;
