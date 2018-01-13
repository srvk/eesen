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

nproj=""
nfinalproj=
ninitproj=""

#speaker adaptation configuration
sat_type=""
sat_stage=""
sat_path=""
sat_nlayer=2
continue_ckpt_sat=false

#training configuration
nepoch=""
lr_rate=""
debug=false

#continue training
continue_ckpt=""
diff_num_target_ckpt=false
force_lr_epoch_ckpt=false

#training options
deduplicate=true
subsampling_default=3

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

trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir &" ERR EXIT

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

if $debug; then
    debug="--debug"
else
    debug=""
fi

if $force_lr_epoch_ckpt; then
    force_lr_epoch_ckpt="--force_lr_epoch_ckpt"
else
    force_lr_epoch_ckpt=""
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

if [ -n "$ninitproj" ]; then
    ninitproj="--ninitproj $ninitproj"
fi

if [ -n "$nfinalproj" ]; then
    nfinalproj="--nfinalproj $nfinalproj"
fi

if [ -n "$nproj" ]; then
    nproj="--nproj $nproj"
fi

if [ -n "$nepoch" ]; then
    nepoch="--nepoch $nepoch"
fi

if [ -n "$lr_rate" ]; then
    lr_rate="--lr_rate $lr_rate"
fi

if [ -n "$deduplicate" ]; then
    deduplicate="--deduplicate"
else
    deduplicate=""
fi

subsampling=`echo $train_opts | sed 's/.*--subsampling \([0-9]*\).*/\1/'`

if [[ "$subsampling" == [0-9]* ]]; then
    #this is needed for the filtering - let's hope this value is correct
    :
else
    subsampling=3
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

# Compute the occurrence counts of labels in the label sequences.
# These counts will be used to derive prior probabilities of the labels.
awk '{line=$0; gsub(" "," 0 ",line); print line " 0";}' $tmpdir/labels.tr | \
  analyze-counts --verbose=1 --binary=false ark:- $dir/label.counts || exit 1

echo ""
echo cleaning train set ...
echo ""

for f in $tmpdir/*.tr; do

	echo ""
	echo cleaning train set $(basename $f)...
	echo ""

	python ./utils/clean_length.py --scp_in  $tmpdir/train_tmp.scp --labels $f \
	       --subsampling $subsampling --scp_out $tmpdir/train_local.scp $deduplicate
done

for f in $tmpdir/*.cv; do

    echo ""
    echo cleaning cv set $(basename $f)...
    echo ""

    python ./utils/clean_length.py --scp_in  $tmpdir/cv_tmp.scp --labels $f \
	   --subsampling $subsampling --scp_out $tmpdir/cv_local.scp

done

#path were cache cuda binaries will be compiled and stored
export CUDA_CACHE_PATH=$tmpdir


cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING STARTS [$cur_time]"

$train_tool $train_opts \
    --model $model --nlayer $nlayer --nhidden $nhidden $ninitproj $nproj $nfinalproj $nepoch $lr_rate \
    --train_dir $dir --data_dir $tmpdir $sat_stage $sat_type $sat_nlayer $debug $continue_ckpt $continue_ckpt_sat $diff_num_target_ckpt $force_lr_epoch_ckpt  || exit 1;

cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING ENDS [$cur_time]"

exit
