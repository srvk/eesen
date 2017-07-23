#!/bin/bash

# Copyright 2015  Yajie Miao    (Carnegie Mellon University)
# Copyright 2016  Florian Metze (Carnegie Mellon University)
# Apache 2.0

# This script trains acoustic models based on CTC and using SGD.

## Begin configuration section
train_tool="python -m main" # the command for training; by default, we use the
                # parallel version which processes multiple utterances at the same time
train_opts="--store_model --lstm_type=cudnn --augment"

# network size
nlayer=5                 # number of layers
nhidden=320              # cells in each layer and direction
nproj=0                  # projection layers, 0 if none
feat_proj=0              # project features, 0 if none

# configs for multiple sequences
num_sequence=16          # during training, how many utterances to be processed in parallel

# learning rate
learn_rate=0.02          # learning rate
final_learn_rate=1e-6    # final learning rate
l2=0.001                 # l2 regularization

# learning rate schedule
max_iters=30             # max number of iterations
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
add_deltas=false         # whether to add deltas
copy_feats=true          # whether to copy features into a local dir (on the GPU machine)
context_window=1         # how many frames to stack
mix_dirs=                # directories from which to read the features for mixing

# status of learning rate schedule; useful when training is resumed from a break point
cvacc=0
pvacc=0
halving=false

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
## End configuration section


function prepare_features() {
    # this uses a lot of global variables
    local m=$1
    local tmpdir=$2
    local sources=( $@ ) && sources="${sources[@]:2}"

    # This can be 1 or 3 and controls parallelization
    local par=3

    # do we do data augmentation?
    if [ -d $tmpdir ]; then
	: # no need to do anything
    elif [ `echo "$sources"|wc -w` -gt 1 ]; then
        utils/mix_data_dirs.sh $m $data_tr $tmpdir $sources >& $dir/log/mix.iter${m}.log || exit 1;
        local data_tr=$tmpdir
    else
        local data_tr=$sources
	mkdir $tmpdir
    fi

    if [ $par -gt 1 ]; then
	feats_tr="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$tmpdir/xxxaa ark:- |"
    else
	feats_tr="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$data_tr/feats.scp ark:- |"
    fi
    feats_cv="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$data_cv/feats.scp ark:- |"

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

    # Save the features to a local dir on the GPU machine. On Linux, this usually points to /tmp
    if [ $par -gt 1 ]; then
        split -l `wc -l $data_tr/feats.scp | awk -v c=$par '{print int((\$1+c)/c)}'` $data_tr/feats.scp $tmpdir/xxx
        f2=`echo $feats_tr | sed 's/xxxaa/xxxab/'`
        f3=`echo $feats_tr | sed 's/xxxaa/xxxac/'`
        copy-feats "$feats_tr" ark,scp:$tmpdir/train1.ark,$tmpdir/train1local.scp &
        copy-feats "$f2"       ark,scp:$tmpdir/train2.ark,$tmpdir/train2local.scp &
        copy-feats "$f3"       ark,scp:$tmpdir/train3.ark,$tmpdir/train3local.scp &
        wait
        cat $tmpdir/train?local.scp > $tmpdir/train_local.scp || exit 1;
    else
	copy-feats "$feats_tr" ark,scp:$tmpdir/train.ark,$tmpdir/train_local.scp || exit 1;
    fi
    copy-feats "$feats_cv" ark,scp:$tmpdir/cv.ark,$tmpdir/cv_local.scp || exit 1;
}
## End function section


mkdir -p $dir/log
for f in $data_tr/feats.scp $data_cv/feats.scp $dir/labels.tr.gz $dir/labels.cv.gz; do
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
    analyze-counts --verbose=1 --binary=false ark:- $dir/label.counts >& \
    $dir/log/compute_label_counts.log || exit 1


## Set up labels
tmpdir=`mktemp -d`
labels_tr="ark:gunzip -c $dir/labels.tr.gz|"
labels_cv="ark:gunzip -c $dir/labels.cv.gz|"
if [ -f $dir/labels.tr.gz ]; then	
    gzip -cd $dir/labels.tr.gz > $tmpdir/labels.tr
    gzip -cd $dir/labels.cv.gz > $tmpdir/labels.cv
fi


## Setup features
# output feature configs which will be used in decoding
echo $norm_vars > $dir/norm_vars
echo $add_deltas > $dir/add_deltas
echo $splice_feats > $dir/splice_feats
echo $subsample_feats > $dir/subsample_feats
echo $context_window > $dir/context_window
#
if [ -n "$mix_dirs" ]; then
    train_opts="$train_opts --mix"
    {
    i=0 && d=
    # create per-set folders
    for j in $data_tr $mix_dirs; do
	echo preparing $tmpdir/A$i from $j ...
	prepare_features 0 $tmpdir/A$i $j || exit 1;
	mv $tmpdir/A$i/train_local.scp $tmpdir/A$i/feats.scp
	cp $j/utt2spk $j/spk2utt $j/segments $tmpdir/A$i
	d="$d $tmpdir/A$i"
	wc $tmpdir/A$i/*
	((i++))
    done
    #
    # create the per-iteration folders
    mkdir $tmpdir/X0 && ln -s $tmpdir/A0/* $tmpdir/X0
    mv $tmpdir/X0/feats.scp $tmpdir/X0/train_local.scp
    ln -s $tmpdir/X0/cv_local.scp $tmpdir
    ln -s $tmpdir/labels.* $tmpdir/X0
    echo mixing $d $max_iters times ...
    for i in `seq 1 $max_iters`; do
	./utils/mix_data_dirs.sh $i $data_tr $tmpdir/X$i $d
	mv $tmpdir/X$i/feats.scp $tmpdir/X$i/train_local.scp
       	utils/filter_scp.pl $tmpdir/X$i/utt2spk $tmpdir/X0/labels.tr > $tmpdir/X$i/labels.tr
	ln -s $tmpdir/A0/*cv* $tmpdir/labels.cv $tmpdir/X$i
    done
    } >& $dir/log/mix.log
else
    echo preparing $dir to $tmpdir
    prepare_features 0 $tmpdir $data_tr
fi

# it is safe to do this now (hopefully)
trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT ERR


## Adjust parameter variables
if [ $start_epoch_num -gt 1 ]; then
    ckpt="$dir/dbr-run*/model/`printf 'epoch%02d.ckpt.index' $[start_epoch_num-1]`"
    ckpt="--continue_ckpt `echo $ckpt|awk '{print $NF}'|sed 's/.index//'`"
else
    ckpt=""
fi
if [ $nproj -gt 0 ]; then
    nproj="--nproj $nproj"
else
    nproj=""
fi
if [ $feat_proj -gt 0 ]; then
    feat_proj="--feat_proj $feat_proj"
else
    feat_proj=""
fi
if [ -n "$max_iters" ]; then
    max_iters="--nepoch $max_iters"
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
$train_tool $train_opts --lr_rate $learn_rate --l2 $l2 --batch_size $num_sequence \
    --nhidden $nhidden --nlayer $nlayer $nproj $feat_proj $ckpt $max_iters \
    --train_dir $dir --data_dir $tmpdir || exit 1;

## Done
cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING ENDS [$cur_time]"

exit
