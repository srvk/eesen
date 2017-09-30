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

function filter_text {
  perl -e 'foreach $w (@ARGV) { $bad{$w} = 1; } 
   while(<STDIN>) { @A  = split(" ", $_); $id = shift @A; print "$id ";
     foreach $a (@A) { if (!defined $bad{$a}) { print "$a "; }} print "\n"; }' \
   '[NOISE]' '[LAUGHTER]' '[VOCALIZED-NOISE]' '<UNK>' '%HESITATION'
}
filter_text <$data/text >$dir/scoring/text.filt

$cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/best_path.ACWT.log \
  lattice-scale --acoustic-scale=ACWT --ascale-factor=$acwt_factor  "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
  lattice-best-path --word-symbol-table=$symtab ark:- ark,t:$dir/scoring/ACWT.tra || exit 1;

for acwt in `seq $min_acwt $max_acwt`; do
  cat $dir/scoring/${acwt}.tra | utils/int2sym.pl -f 2- $symtab | \
    filter_text > $dir/scoring/$acwt.txt || exit 1;
done

unset LC_ALL
#for character error rate
cat $dir/scoring/text.filt | awk '{ print $1}' > $dir/scoring/utt_id
cat $dir/scoring/text.filt | awk '{{for (i = 2; i <= NF; i++) printf(" %s", $i);} printf("\n"); }' | sed -e 's/\(\S\)/\1 /g' > $dir/scoring/utt_tra
paste $dir/scoring/utt_id $dir/scoring/utt_tra  > $dir/scoring/char.filt

for acwt in `seq $min_acwt $max_acwt`; do
  cat $dir/scoring/$acwt.txt | awk '{ print $1}' > $dir/scoring/utt_id
  cat $dir/scoring/$acwt.txt | awk '{{for (i = 2; i <= NF; i++) printf(" %s", $i);} printf("\n"); }' | sed -e 's/\(\S\)/\1 /g' > $dir/scoring/utt_tra
  paste $dir/scoring/utt_id $dir/scoring/utt_tra  > $dir/scoring/${acwt}.char
done

rm $dir/scoring/utt_tra $dir/scoring/utt_id

export LC_ALL=C

$cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/score.ACWT.log \
  compute-wer --text --mode=present \
   ark:$dir/scoring/text.filt ark:$dir/scoring/ACWT.txt ">&" $dir/wer_ACWT || exit 1;

$cmd ACWT=$min_acwt:$max_acwt $dir/scoring/log/score.ACWT.cer.log \
  compute-wer --text --mode=present \
   ark:$dir/scoring/char.filt ark:$dir/scoring/ACWT.char ">&" $dir/cer_ACWT || exit 1;

exit 0;
