#!/bin/bash

# Copyright 2015  Yajie Miao    (Carnegie Mellon University)
# Copyright 2016  Florian Metze (Carnegie Mellon University)
# Apache 2.0

# This script trains acoustic models using tensorflow

## Begin configuration section

#main calls and arguments
train_tool="python -m train"
train_opts="--store_model --lstm_type=cudnn --augment"

#network architecture
model="deepbilstm"
nlayer=5
nhidden=320
nproj=0
nfinalproj=0
norm=false

#speaker adaptation configuration
sat_type=""
sat_stage=""
sat_path=""
sat_nlayer=2
continue_ckpt_sat=false

#training configuration
batch_size=16
learn_rate=0.02
l2=0.0001
max_iters=25
half_after=6
debug=false

#continue training
continue_ckpt=""
diff_num_target_ckpt=false
force_lr_epoch_ckpt=false

#augmentation argument
window=3

## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;

. utils/parse_options.sh || exit 1;

#checking number of arguments
if [ $# != 3 ]; then
   echo "Usage: $0 <data-tr> <data-cv> <exp-dir>"
   echo " e.g.: $0 data/train_tr data/train_cv exp/train_phn"
   exit 1;
fi

#getting main arguments
data_tr=$1
data_cv=$2
dir=$3

#creating tmp directory (concrete tmp path is defined in path.sh)
tmpdir=`mktemp -d`
trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT ERR

#checking folders
for f in $data_tr/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo `basename "$0"`": no such file $f" && exit 1;
done

#multitarget training check

train_labels=$(ls $dir | grep labels | grep tr)

if [ -z "$train_labels" ]; then
    echo "no training labels found in: $dir"
    echo "training dir should have \"labels*.tr\""
    exit
fi

#multitarget training checking

cv_labels=$(ls $dir | grep labels | grep cv)

if [ -z "$train_labels" ]; then
    echo "no training labels found in: $dir"
    echo "training dir should have \"labels*.cv\""
    exit
fi


## Adjust parameter variables

if $force_lr_epoch_ckpt; then
    force_lr_epoch_ckpt="--force_lr_epoch_ckpt"
else
    force_lr_epoch_ckpt=""
fi

if $debug; then
    debug="--debug"
else
    debug=""
fi

if $norm ; then

      norm="--batch_norm"
else
      norm=
fi

if $diff_num_target_ckpt; then
    diff_num_target_ckpt="--diff_num_target_ckpt"
else
    diff_num_target_ckpt=""
fi

if [[ $continue_ckpt != "" ]]; then
    continue_ckpt="--continue_ckpt $continue_ckpt"
else
    continue_ckpt=""
fi

if [ $nfinalproj -gt 0 ]; then
    nfinalproj="--nfinalproj $nfinalproj"
else
    nfinalproj=""
fi

if [ $nproj -gt 0 ]; then
    nproj="--nproj $nproj"
else
    nproj=""
fi

if [ -n "$max_iters" ]; then
    max_iters="--nepoch $max_iters"
fi


#SPEAKER ADAPTATION

if [[ "$sat_type" != "" ]]; then
    cat $sat_path | copy-feats ark,t:- ark,scp:$tmpdir/sat_local.ark,$tmpdir/sat_local.scp

    sat_type="--sat_type $sat_type"
else
    sat_type=""
fi

if [[ "$sat_stage" != "" ]]; then
    sat_stage="--sat_stage $sat_stage"
else
    sat_stage=""
fi

if $continue_ckpt_sat; then
    continue_ckpt_sat="--continue_ckpt_sat"
else
    continue_ckpt_sat=""
fi

sat_nlayer="--sat_nlayer $sat_nlayer"

echo ""
echo copying cv features ...
echo ""

data_tr=$1
data_cv=$2

feats_cv="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$data_cv/feats.scp ark:- |"
copy-feats "$feats_cv" ark,scp:$tmpdir/cv.ark,$tmpdir/cv_tmp.scp || exit 1;

echo ""
echo copying training features ...
echo ""

feats_tr="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$data_tr/feats.scp ark:- |"
copy-feats "$feats_tr" ark,scp:$tmpdir/train.ark,$tmpdir/train_tmp.scp || exit 1;

echo ""
echo copying labels ...
echo ""

if [ -f $dir/labels.tr.gz ]; then
    gzip -cd $dir/labels.tr.gz > $tmpdir/labels.tr || exit 1
    gzip -cd $dir/labels.cv.gz > $tmpdir/labels.cv || exit 1
else
    cp $dir/labels*.tr $tmpdir/ || exit 1
    cp $dir/labels*.cv $tmpdir/ || exit 1
fi

echo ""
echo cleaning train set ...
echo ""

for f in $tmpdir/*.tr; do

	echo ""
	echo cleaning train set $(basename $f)...
	echo ""

	python ./utils/clean_length.py --scp_in  $tmpdir/train_tmp.scp --labels $f --subsampling 3 --scp_out $tmpdir/train_local.scp
done

for f in $tmpdir/*.cv; do

    echo ""
    echo cleaning cv set $(basename $f)...
    echo ""

    python ./utils/clean_length.py --scp_in  $tmpdir/cv_tmp.scp --labels $f --subsampling 3 --scp_out $tmpdir/cv_local.scp

done


cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING STARTS [$cur_time]"


$train_tool $train_opts --lr_rate $learn_rate --batch_size $batch_size --l2 $l2 \
    --nhidden $nhidden --nlayer $nlayer $nproj $nfinalproj $ckpt $max_iters \
    --train_dir $dir --data_dir $tmpdir --half_after $half_after $sat_stage $sat_type $sat_nlayer $debug --model $model --window $window $norm $continue_ckpt $continue_ckpt_sat $diff_num_target_ckpt $force_lr_epoch_ckpt  || exit 1;

cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING ENDS [$cur_time]"

exit
