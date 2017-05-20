#!/bin/bash

# Copyright 2015  Yajie Miao    (Carnegie Mellon University)
# Copyright 2016  Florian Metze (Carnegie Mellon University)
# Apache 2.0

# This script trains acoustic models based on CTC and using SGD.

## Begin configuration section
train_tool="python /data/ASR5/fmetze/eesen-tf/tf/ctc-train/main.py" # the command for training; by default, we use the
                # parallel version which processes multiple utterances at the same time
train_opts="--store_model --lstm_type=cudnn"

# configs for multiple sequences
num_sequence=60          # during training, how many utterances to be processed in parallel
valid_num_sequence=60    # number of parallel sequences in validation
frame_num_limit=20000    # the number of frames to be processed at a time in training; this config acts to
         # to prevent running out of GPU memory if #num_sequence very long sequences are processed; the max
         # number of training examples is decided by if num_sequence or frame_num_limit is reached first.

# learning rate
learn_rate=4e-5          # learning rate
final_learn_rate=1e-6    # final learning rate
momentum=0.9             # momentum

# learning rate schedule
max_iters=25             # max number of iterations
min_iters=               # min number of iterations
start_epoch_num=1        # start from which epoch, used for resuming training from a break point

start_halving_inc=0.1    # start halving learning rates when the accuracy improvement falls below this amount
end_training_inc=0.01    # terminate training when the accuracy improvement falls below this amount
halving_factor=0.8       # learning rate decay factor
halving_after_epoch=10   # halving becomes enabled after this many epochs
force_halving_epoch=     # force halving after this epoch

# logging
report_step=1000         # during training, the step (number of utterances) of reporting objective and accuracy
verbose=1

# feature configs
sort_by_len=true         # whether to sort the utterances by their lengths
seed=777                 # random seed
block_softmax=false      # multi-lingual training
shuffle=true             # shuffle feature order after first iteration

feats_std=1.0            # scale features
splice_feats=false       # whether to splice neighboring frams
subsample_feats=false    # whether to subsample features
norm_vars=true           # whether to apply variance normalization when we do cmn
add_deltas=true          # whether to add deltas
copy_feats=true          # whether to copy features into a local dir (on the GPU machine)
context_window=1         # how many frames to stack
augment_dirs=            # directories from which to read the augmented features

# status of learning rate schedule; useful when training is resumed from a break point
cvacc=0
pvacc=0
halving=false

## End configuration section

function shuffle_data() {
    local iter=$1
    local num_sequence=$2
    local frame_num_limit=$3
    local dir=$4
    local tmpdir=$5
    #
    mkdir -p $tmpdir/shuffle
    [ -f $tmpdir/train_local.org ] || cp $tmpdir/train_local.scp $tmpdir/train_local.org
    [ -f $tmpdir/cv_local.org    ] || cp $tmpdir/cv_local.scp    $tmpdir/cv_local.org
    utils/prep_scps.sh --cp false --nj 1 --cmd "run.pl" --seed $iter \
        $tmpdir/train_local.org $tmpdir/cv_local.org $num_sequence $frame_num_limit $tmpdir/shuffle $tmpdir
    if false; then
        mv $tmpdir/feats_tr.1.scp $tmpdir/train_local.scp
	mv $tmpdir/feats_cv.1.scp $tmpdir/cv_local.scp
	rm $tmpdir/batch.tr.list  $tmpdir/batch.cv.list
    fi
}

function prepare_features() {
    # this uses a lot of global variables
    # in particular it sets feats_tr and feats_cv (amongst others)
    local trfile=$1
    local cvfile=$2
    local m=$3
    local targetdir=$4
    local tmpdir=$5
    local sources=( $@ ) && sources="${sources[@]:5}"

    # do we do data augmentation?
    if [ -d $targetdir ]; then
	: # no need to do anything
    elif [ `echo "$sources"|wc -w` -gt 1 ]; then
        utils/mix_data_dirs.sh $m $data_tr $targetdir $sources >& $dir/log/mix.iter${m}.log || exit 1;
        local data_tr=$targetdir
    else
        local data_tr=$sources
    fi

    if $sort_by_len; then
	gzip -cd $dir/labels.tr.gz | join <(feat-to-len scp:$data_tr/feats.scp ark,t:- | paste -d " " - $data_tr/feats.scp) - | sort -gk 2 | \
	    awk -v c=$context_window '{out=""; for (i=5;i<=NF;i++) {out=out" "$i}; if (!(out in done) && $2 > (2*c+1)*NF) {done[out]=1; print $3 " " $4}}' > $dir/train.scp
	feat-to-len scp:$data_cv/feats.scp ark,t:- | awk '{print $2}' | \
	    paste -d " " $data_cv/feats.scp - | sort -k3 -n - | awk '{print $1 " " $2}' > $dir/cv.scp || exit 1;
    else
	cat $data_tr/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
	cat $data_cv/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/cv.scp
    fi

    feats_tr="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$dir/train.scp ark:- |"
    feats_cv="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$dir/cv.scp ark:- |"

    if [ 1 == $(bc <<< "$feats_std != 1.0") ]; then
	compute-cmvn-stats "$feats_tr" $dir/global_cmvn_stats
	echo $feats_std > $dir/feats_std
	feats_tr="$feats_tr apply-cmvn --norm-means=true --norm-vars=true $dir/global_cmvn_stats ark:- ark:- | copy-matrix --scale=$feats_std ark:- ark:- |"
	feats_cv="$feats_cv apply-cmvn --norm-means=true --norm-vars=true $dir/global_cmvn_stats ark:- ark:- | copy-matrix --scale=$feats_std ark:- ark:- |"
	# at present, copy-feats is a Kaldi program (not yet in Eesen)
    fi

    if $splice_feats; then
	feats_tr="$feats_tr splice-feats --left-context=$context_window --right-context=$context_window ark:- ark:- |"
	feats_cv="$feats_cv splice-feats --left-context=$context_window --right-context=$context_window ark:- ark:- |"
    fi

    mkdir -p $tmpdir
    if $subsample_feats; then
	#tmpdir=$(mktemp -d)
    
	copy-feats "$feats_tr" "ark:-" | tee \
            >(subsample-feats --n=3 --offset=2 ark:- ark,scp:$tmpdir/train2.ark,$tmpdir/train2local.scp) \
	    >(subsample-feats --n=3 --offset=1 ark:- ark,scp:$tmpdir/train1.ark,$tmpdir/train1local.scp) | \
	      subsample-feats --n=3 --offset=0 ark:- ark,scp:$tmpdir/train0.ark,$tmpdir/train0local.scp || exit 1;

	copy-feats "$feats_cv" "ark:-" | tee \
	    >(subsample-feats --n=3 --offset=2 ark:- ark,scp:$tmpdir/cv2.ark,$tmpdir/cv2local.scp) \
	    >(subsample-feats --n=3 --offset=1 ark:- ark,scp:$tmpdir/cv1.ark,$tmpdir/cv1local.scp) | \
	      subsample-feats --n=3 --offset=0 ark:- ark,scp:$tmpdir/cv0.ark,$tmpdir/cv0local.scp || exit 1;

	sed 's/^/0x/' $tmpdir/train0local.scp        > $tmpdir/train_local.scp
	sed 's/^/1x/' $tmpdir/train1local.scp | tac >> $tmpdir/train_local.scp
	sed 's/^/2x/' $tmpdir/train2local.scp       >> $tmpdir/train_local.scp
	sed 's/^/0x/' $tmpdir/cv0local.scp  > $tmpdir/cv_local.scp
	sed 's/^/1x/' $tmpdir/cv1local.scp >> $tmpdir/cv_local.scp
	sed 's/^/2x/' $tmpdir/cv2local.scp >> $tmpdir/cv_local.scp
    
	feats_tr="ark,s,cs:copy-feats scp:$tmpdir/shuffle/batch.tr.1.scp ark:- |"
	feats_cv="ark,s,cs:copy-feats scp:$tmpdir/shuffle/batch.cv.1.scp ark:- |"
    
	gzip -cd $dir/labels.tr.gz | sed 's/^/0x/'  > $tmpdir/labels.tr
	gzip -cd $dir/labels.cv.gz | sed 's/^/0x/'  > $tmpdir/labels.cv
	gzip -cd $dir/labels.tr.gz | sed 's/^/1x/' >> $tmpdir/labels.tr
	gzip -cd $dir/labels.cv.gz | sed 's/^/1x/' >> $tmpdir/labels.cv
	gzip -cd $dir/labels.tr.gz | sed 's/^/2x/' >> $tmpdir/labels.tr
	gzip -cd $dir/labels.cv.gz | sed 's/^/2x/' >> $tmpdir/labels.cv
	
	#labels_tr="ark:cat $tmpdir/labels.tr|"
	#labels_cv="ark:cat $tmpdir/labels.cv|"
    
	trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT
    
    elif $copy_feats; then
	# Save the features to a local dir on the GPU machine. On Linux, this usually points to /tmp
	#tmpdir=$(mktemp -d)
	copy-feats "$feats_tr" ark,scp:$tmpdir/train.ark,$tmpdir/train_local.scp || exit 1;
	copy-feats "$feats_cv" ark,scp:$tmpdir/cv.ark,$tmpdir/cv_local.scp || exit 1;
	feats_tr="ark,s,cs:copy-feats scp:$tmpdir/feats_tr.scp ark:- |"
	feats_cv="ark,s,cs:copy-feats scp:$tmpdir/feats_cv.scp ark:- |"

	gzip -cd $dir/labels.tr.gz > $tmpdir/labels.tr
	gzip -cd $dir/labels.cv.gz > $tmpdir/labels.cv
	
	trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT
    fi

    if $add_deltas; then
	feats_tr="$feats_tr add-deltas ark:- ark:- |"
	feats_cv="$feats_cv add-deltas ark:- ark:- |"
    fi

    # shuffle and partition the (current) data
    #shuffle_data $m $num_sequence $frame_num_limit $dir $tmpdir

    # let's return the value of feats_tr and feats_cv
    echo "$feats_tr" > $trfile
    echo "$feats_cv" > $cvfile
}
## End function section

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;

. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 <data-tr> <data-cv> <exp-dir>"
   echo " e.g.: $0 data/train_tr data/train_cv exp/train_phn"
   exit 1;
fi

data_tr=$1
data_cv=$2
dir=$3

mkdir -p $dir/log $dir/nnet

if [ -z "$augment_dirs" ]; then
    augment__dirs=( $data_tr )
    augment_dirs=${augment__dirs[0]}
fi

for f in $data_tr/feats.scp $data_cv/feats.scp $dir/labels.tr.gz $dir/labels.cv.gz $dir/nnet.proto; do
  [ ! -f $f ] && echo `basename "$0"`": no such file $f" && exit 1;
done

## Read the training status for resuming
[ -f $dir/.epoch ]   && start_epoch_num=`cat $dir/.epoch 2>/dev/null`
[ -f $dir/.cvacc ]   && cvacc=`cat $dir/.cvacc 2>/dev/null`
[ -f $dir/.pvacc ]   && pvacc=`cat $dir/.pvacc 2>/dev/null`
[ -f $dir/.halving ] && halving=`cat $dir/.halving 2>/dev/null`
[ -f $dir/.lrate ]   && learn_rate=`cat $dir/.lrate 2>/dev/null`

# Compute the occurrence counts of labels in the label sequences. These counts will be used to
# derive prior probabilities of the labels.
gunzip -c $dir/labels.tr.gz | awk '{line=$0; gsub(" "," 0 ",line); print line " 0";}' | \
  analyze-counts --verbose=1 --binary=false ark:- $dir/label.counts >& $dir/log/compute_label_counts.log || exit 1

## Set up labels
labels_tr="ark:gunzip -c $dir/labels.tr.gz|"
labels_cv="ark:gunzip -c $dir/labels.cv.gz|"

## Setup features
# output feature configs which will be used in decoding
echo $norm_vars > $dir/norm_vars
echo $add_deltas > $dir/add_deltas
echo $splice_feats > $dir/splice_feats
echo $subsample_feats > $dir/subsample_feats
echo $context_window > $dir/context_window
#
tmpdir=`mktemp -d`
trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT
#
# this will now be done on the fly ...
splice_feats=false
subsample_feats=false
add_deltas=false
#
if [ $start_epoch_num -eq 1 ]; then
    prepare_features $tmpdir/.tr $tmpdir/.cv 1 $tmpdir $tmpdir/T${start_epoch_num} $data_tr || exit 1;
else
    prepare_features $tmpdir/.tr $tmpdir/.cv $start_epoch_num \
		     $tmpdir/X$start_epoch_num $tmpdir/T$start_epoch_num $augment_dirs || exit 1;
fi
feats_tr=`cat $tmpdir/.tr 2>/dev/null`
feats_cv=`cat $tmpdir/.cv 2>/dev/null`
## End of feature setup


## Read parameters from description file
if [ `grep AffineTransform $dir/nnet.proto|wc -l` -gt 1 ]; then
    nproj=`awk '/AffineTransform/ {printf("--nproj %d", $5); exit}' $dir/nnet.proto`
else
    nproj=""
fi
nlayer=`grep BiLstmParallel $dir/nnet.proto|wc -l`
nhidden=`awk '/BiLstmParallel/ {print $5/2; exit}' $dir/nnet.proto`
if [ $start_epoch_num -gt 1 ]; then
    ckpt=" --continue_ckpt dbr-run*/model/`printf 'epoch%02d.ckpt' $[start_epoch_num-1]`"
fi


## Main loop
cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING STARTS [$cur_time]"

# still have to deal with augment dirs and all the parameters
# - block softmax
# - temperature
# - continuation of training
# - data mixing in multiple directories
# 
$train_tool $train_opts \
    --nhidden $nhidden --nlayer $nlayer $nproj \
    --train_dir $dir --data_dir ${tmpdir}/T$start_epoch_num || exit 1;
#    >& $dir/log/tf.log || exit 1;

lrate=$(grep -a "learning rate" $dir/log/tf.log | sort -rgk 11 | tail -n 1 | awk '{ print $11 }')
tracc=$(grep -a "Train cost" $dir/log/tf.log    | sort -rgk 5  | tail -n 1 | awk '{ acc=$5; gsub(",","",acc); print (1-acc)*100 }')
cvacc=$(grep -a "Validate cost" $dir/log/tf.log | sort -rgk 5  | tail -n 1 | awk '{ acc=$5; gsub(",","",acc); print (1-acc)*100 }')

echo -n "lrate $(printf "%.6g" $lrate), TRAIN ACCURACY $(printf "%.2f" $tracc)%, VALID ACCURACY $(printf "%.2f" $cvacc)%"


## Done
cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING ENDS [$cur_time]"

exit
