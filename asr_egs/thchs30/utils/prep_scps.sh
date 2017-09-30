#!/bin/bash
{
# Copyright  2015  Hang Su
# Apache 2.0

# This script prepares feature scp file for CTC training

set -e
set -o pipefail

## Begin configuration section
clean_up=true
seed=
cmd=
nj=1
# End of configuration

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh;

. utils/parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: "
  echo " e.g.: "
  exit 1
fi

feat_tr=$1
feat_cv=$2
num_sequence=$3
frame_num_limit=$4
tmpdir=$5
dir=$6

for part in tr cv; do
  feat=$(eval echo "\$feat_${part}")

  feat-to-len scp:$feat ark,t:- | sort -k2 -n | \
    awk -v num_sequence=$num_sequence -v frame_num_limit=$frame_num_limit '
      BEGIN {max_frame_num = 0; num_utts = 0;}
      { 
        printf("%s ",$1);
        num_utts++;
        if (max_frame_num < $2) {
          max_frame_num = $2;
        }
        if (num_utts >= num_sequence || num_utts * max_frame_num > frame_num_limit) {
          printf("\n");
          num_utts = 0;
          max_frame_num = 0;
        }
      }' | utils/shuffle_list.pl --srand ${seed:-777} > $dir/batch.$part.list

  split_batches=""
  for n in $(seq $nj); do
    split_batches="$split_batches $tmpdir/batch.$part.$n.list"
  done
  utils/split_scp.pl $dir/batch.$part.list $split_batches

  for n in $(seq $nj); do
    awk '
      NR==FNR {a[$1]=$2;next}
      {
        for (i=1; i<=NF; i++) {
          printf("%s %s\n", $i, a[$i]);
        }
      }' $feat $tmpdir/batch.$part.$n.list > $tmpdir/batch.$part.$n.scp
  done
  if [ $nj -ne 1 ]; then
    $cmd JOB=1:$nj $dir/log/prepare_feats_$part.JOB.log \
      copy-feats scp:$tmpdir/batch.$part.JOB.scp ark,scp:$tmpdir/feats_$part.JOB.ark,$dir/feats_$part.JOB.scp
  else
    copy-feats scp:$tmpdir/batch.$part.1.scp ark,scp:$tmpdir/feats_$part.1.ark,$dir/feats_$part.1.scp
  fi

done

}
