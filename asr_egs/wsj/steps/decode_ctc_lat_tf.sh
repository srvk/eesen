#!/bin/bash

# Apache 2.0

# Decode the CTC-trained model by generating lattices.


## Begin configuration section
nj=16
cmd=run.pl

acwt=0.9
min_active=200
max_active=7000 # max-active
beam=15.0       # beam used
lattice_beam=8.0
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes
model=final.nnet
scoredir=
label_counts=
label_scales=
blank_scale=
noise_scale=
temperature=
subsampled_utt=1  # set to 0 to average over all subsamples

online_storage=false
use_priors=true

skip_scoring=false # whether to skip WER scoring
scoring_opts="--min-acwt 5 --max-acwt 15 --acwt-factor 0.1"

# feature configurations; will be read from the training dir if not provided
norm_vars=true
## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: steps/decode_ctc_lat_tf.sh [options] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_ctc_lat_tf.sh data/lang data/test exp/train_l4_c320/decode"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt                                   # default 0.9, the acoustic scale to be used"
   exit 1;
fi

# process options
graphdir=$1
data=$2
dir=`echo $3 | sed 's:/$::g'` # remove any trailing slash.

srcdir=`dirname $dir`; # assume model directory one level up from decoding directory.
sdata=$data/split$nj;

[ -z "$label_counts" ]   && label_counts=${srcdir}/label.counts
# set optional arguments to pass later if needed
[ -n "$label_scales" ]   && label_scales_option="--label-scales ${label_scales}"
[ -n "$blank_scale" ]    && blank_scale_option="--blank-scale ${blank_scale}"
[ -n "$noise_scale" ]    && noise_scale_option="--noise-scale ${blank_scale}"
[ -n "$temperature" ]    && temperature_option="--temperature ${temperature}"
[ -n "$subsampled_utt" ] && subsampled_utt_option="--subsampled_utt $subsampled_utt"
norm_vars_option=${norm_vars}
[ -z "$norm_vars" ] && norm_vars=`cat $srcdir/norm_vars 2>/dev/null`
online_storage_option=""
if $online_storage ; then online_storage_option="--online_storage" ; fi

# make log directory, split data into number of jobs
mkdir -p $dir/log
split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Check if necessary files exist.
for f in $graphdir/TLG.fst $label_counts $data/feats.scp $model.*; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

## Set up the features
echo "$0: feature: norm_vars(${norm_vars})"
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:$data/feats.scp ark:- |"

# This script used to do splicing, subsampling, and adding deltas.  Now handled internally in tf test module through config.
# Note that you probably don't want to do averaging over subsampling, so subsampling_utt=1 will fix that


# maybe we want to keep this
if [ -z "$scoredir" ]; then
    tmpdir=`mktemp -d`
    trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT ERR
else
    mkdir -p $scoredir
    tmpdir=$scoredir
fi

if [ ! -f $tmpdir/labels.cv ]; then
    # EFL: not clear if we need references any more 
    #
    # we need (fake) references
    #if [ -f $data/textn ]; then
    #	local/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt  $data/text "<unk>" > $tmpdir/labels.cv
    #else
    #	cat $data/feats.scp | awk ' { print $1,1 } ' > $tmpdir/labels.cv
    #fi
    #cp $tmpdir/labels.cv $tmpdir/labels.tr

    # copy features
    copy-feats "${feats}" ark,scp:$tmpdir/f.ark,$tmpdir/test_local.scp

    # let's call tensorflow, output will be in tmpdir
    set -x
    python -m test \
	--data_dir $tmpdir \
	--results_dir $tmpdir \
	--trained_weights $model \
	--train_config `dirname $model`/config.pkl \
	--temperature 1 \
	$subsampled_utt_option $online_storage_option
    set +x
else
    echo Assuming data is already in $tmpdir
fi

# Decode for each of the acoustic scales
$cmd JOB=1:$nj $dir/log/decode.JOB.log \
  utils/filter_scp.pl \
    $sdata/JOB/feats.scp \
    $tmpdir/logit_no_target_name.scp \| \
  sort -k 1 \| \
  python utils/nnet_notf.py \
    --label-counts $label_counts \
    $label_scales_option \
    $temperature_option \
    $blank_scale_option \
    $noise_scale_option \| \
  latgen-faster \
    --max-active=$max_active \
    --max-mem=$max_mem \
    --beam=$beam \
    --lattice-beam=$lattice_beam \
    --acoustic-scale=$acwt \
    --allow-partial=true \
    --word-symbol-table=$graphdir/words.txt \
    $graphdir/TLG.fst \
    ark:- "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

# Scoring
if ! $skip_scoring ; then
  if [ -f $data/stm ]; then # use sclite scoring.
    [ ! -x local/score_sclite.sh ] && echo "Not scoring because local/score_sclite.sh does not exist or not executable." && exit 1;
    local/score_sclite.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
  else
    [ ! -x local/score.sh ] && echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
  fi
fi

exit 0;
