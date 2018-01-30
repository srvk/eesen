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
nfinalproj=0
norm=false

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
batch_size=16
learn_rate=0.02
l2=0.0001
max_iters=25
half_after=6
debug=false

#continue training
continue_ckpt=""
import_config=""
diff_num_target_ckpt=false
force_lr_epoch_ckpt=false

#augmentation argument
window=3

## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;

. utils/parse_options.sh || exit 1;

#getting last argument (dir)
dir="${!#}"

#getting all languages
all_lan=( "$@" )
unset "all_lan[${#all_lan[@]}-1]"

#creating tmp directory (concrete tmp path is defined in path.sh)
tmpdir=`mktemp -d`
trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir &" ERR EXIT


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

if [ -n "$ninitproj" ]; then
    ninitproj="--ninitproj $ninitproj"
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

if [[ $import_config != "" ]]; then
    import_config="--import_config $import_config"
else
    import_config=""
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

if [ $window -gt 0 ]; then
    window="--window $window"
else
    window=""
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




for language_dir in "${all_lan[@]}"; do

    language_name=$(basename $language_dir)

    mkdir $tmpdir/$language_name

    echo ""
    echo START COPYING LANGUAGE: $language_name

    echo copying training features ...

    cv_folder=$(ls $language_dir | grep \_cv)

    if [ -z "$cv_folder" ]; then
	echo "no training folder found for language: $language_name ($language_dir)"
	echo "training dir should have \"_cv\""
	exit
    fi

    echo copying cv features ...

    data_cv=$language_dir/$cv_folder

    feats_cv="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$data_cv/feats.scp ark:- |"
    copy-feats "$feats_cv" ark,scp:$tmpdir/$language_name/cv.ark,$tmpdir/$language_name/cv_local.scp || exit 1;


    tr_folder=$(ls $language_dir | grep \_tr)

    if [ -z "$tr_folder" ]; then
	echo "no training folder found for language: $language_name ($language_dir)"
	echo "training dir should have \"_tr\""
	exit
    fi

    echo copying train features ...

    data_tr=$language_dir/$tr_folder

    feats_tr="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$data_tr/feats.scp ark:- |"

    copy-feats "$feats_tr" ark,scp:$tmpdir/$language_name/train.ark,$tmpdir/$language_name/train_local.scp || exit 1;


    labels_tr=$(ls $language_dir | grep labels | grep \.tr)

    if [ -z "$labels_tr" ]; then
	echo "no training labels found: $language_name ($language_dir)"
	echo "training dir should have \".tr\""
	exit
    fi

    labels_cv=$(ls $language_dir | grep labels | grep \.cv)

    if [ -z "$labels_tr" ]; then
	echo "no training labels found: $language_name ($language_dir)"
	echo "training dir should have \".cv\""
	exit
    fi

    echo copying labels ...

    cp $language_dir/labels.tr $tmpdir/$language_name/ || exit 1
    cp $language_dir/labels.cv $tmpdir/$language_name/ || exit 1

    echo cleaning train set ...

    python ./utils/clean_length.py --scp_in  $tmpdir/$language_name/train_local.scp --labels $tmpdir/$language_name/labels.tr --subsampling 3 --scp_out $tmpdir/$language_name/train_local.scp

    echo cleaning cv set ...

    python ./utils/clean_length.py --scp_in  $tmpdir/$language_name/cv_local.scp --labels $tmpdir/$language_name/labels.cv --subsampling 3 --scp_out $tmpdir/$language_name/cv_local.scp

    echo ""

done


echo ""
echo final distribution of data_dir:
echo ""

#path were cache cuda binaries will be compiled and stored
export CUDA_CACHE_PATH=$tmpdir

find $tmpdir

cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING STARTS [$cur_time]"

$train_tool $train_opts --lr_rate $learn_rate --batch_size $batch_size --l2 $l2 \
    --nhidden $nhidden $ninitproj $nproj $nfinalproj --nlayer $nlayer $nproj  $nfinalproj $ckpt $max_iters \
    --train_dir $dir --data_dir $tmpdir --half_after $half_after $sat_stage $sat_type $sat_nlayer $debug --model $model $window $norm $continue_ckpt $continue_ckpt_sat $diff_num_target_ckpt $force_lr_epoch_ckpt $import_config  || exit 1;

cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING ENDS [$cur_time]"

exit
