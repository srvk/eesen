#!/bin/bash

# Copyright 2015  Yajie Miao    (Carnegie Mellon University)
#           2016  Florian Metze (Carnegie Mellon University)
# Apache 2.0

# This script trains acoustic models based on CTC and using SGD. 

## Begin configuration section
train_tool=train-ctc-parallel  # the command for training; by default, we use the
         # parallel version which processes multiple utterances at the same time 

# configs for multiple sequences
num_sequence=20          # during training, how many utterances to be processed in parallel
valid_num_sequence=60    # number of parallel sequences in validation
frame_num_limit=10000    # the number of frames to be processed at a time in training; this config acts to
         # to prevent running out of GPU memory if #num_sequence very long sequences are processed;the max
         # number of training examples is decided by if num_sequence or frame_num_limit is reached first. 

# learning rate
learn_rate=4e-5          # learning rate
final_learn_rate=0.0     # final learning rate
momentum=0.9             # momentum

# learning rate schedule
max_iters=25             # max number of iterations
min_iters=               # min number of iterations
start_epoch_num=1        # start from which epoch, used for resuming training from a break point

start_halving_inc=0.5    # start halving learning rates when the accuracy improvement falls below this amount
end_halving_inc=0.1      # terminate training when the accuracy improvement falls below this amount
halving_factor=0.5       # learning rate decay factor
halving_after_epoch=1    # halving becomes enabled after this many epochs
force_halving_epoch=     # force halving after this epoch

# logging
report_step=1000         # during training, the step (number of utterances) of reporting objective and accuracy
safe_mode=false          # seems to be needed on bridges
verbose=1

# feature configs
sort_by_len=true         # whether to sort the utterances by their lengths
min_ratio=6              # minimal ratio of transcription to consider
min_len=10               # minimal length of utterance
keep_duplicates=false    # keep duplicate utterances 
seed=777                 # random seed

splice_feats=false       # whether to splice neighboring frams
subsample_feats=false    # whether to subsample features
norm_vars=true           # whether to apply variance normalization when we do cmn
add_deltas=true          # whether to add deltas
copy_feats=true          # whether to copy features into a local dir (on the GPU machine)
feats_tmpdir=""          # the tmp dir to save the copied features, when copy_feats=true
cache_dir=""             # get the data from this directory (rather than re-generate it)

# status of learning rate schedule; useful when training is resumed from a break point
cvacc=0
halving=0

## End configuration section

echo "$0 $@"  # Print the command line for logging
cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "SETUP STARTS [$cur_time]"

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

for f in $data_tr/feats.scp $data_cv/feats.scp $dir/labels.tr.gz $dir/labels.cv.gz $dir/nnet.proto; do
  [ ! -f $f ] && echo `basename "$0"`": no such file $f" && exit 1;
done

## Read the training status for resuming
[ -f $dir/.epoch ] && start_epoch_num=`cat $dir/.epoch 2>/dev/null`
[ -f $dir/.cvacc ] && cvacc=`cat $dir/.cvacc 2>/dev/null`
[ -f $dir/.halving ] && halving=`cat $dir/.halving 2>/dev/null`
[ -f $dir/.lrate ] && learn_rate=`cat $dir/.lrate 2>/dev/null`

## Set up labels  
labels_tr="ark:gunzip -c $dir/labels.tr.gz|"
labels_cv="ark:gunzip -c $dir/labels.cv.gz|"
# Compute the occurrence counts of labels in the label sequences. These counts will be used to
# derive prior probabilities of the labels.
gunzip -c $dir/labels.tr.gz | awk '{line=$0; gsub(" "," 0 ",line); print line " 0";}' | \
  analyze-counts --verbose=1 --binary=false ark:- $dir/label.counts >& $dir/log/compute_label_counts.log || exit 1
##

## Setup up features
# output feature configs which will be used in decoding
echo $norm_vars > $dir/norm_vars
echo $add_deltas > $dir/add_deltas
echo $splice_feats > $dir/splice_feats
echo $subsample_feats > $dir/subsample_feats

if $sort_by_len; then
  feat-to-len scp:$data_tr/feats.scp ark,t:- | awk '{print $2}' | \
    paste -d " " $data_tr/feats.scp - <(gzip -cd $dir/labels.tr.gz) | sort -k3 -n | \
    awk -v m=$min_ratio -v n=$min_len -v d=$keep_duplicates \
	'{out=$0; sub(/^[ ]*([^ ]+ +){4}/, "", out); if (d=="false" && !(out in done) && $3 > m*NF && $3 > n) {done[out]=1; print $1 " " $2}}' > $dir/train.scp || exit 1;
  feat-to-len scp:$data_cv/feats.scp ark,t:- | awk '{print $2}' | \
    paste -d " " $data_cv/feats.scp - | sort -k3 -n | awk '{print $1 " " $2}' > $dir/cv.scp || exit 1;
else
  cat $data_tr/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
  cat $data_cv/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/cv.scp
fi

feats_tr="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$dir/train.scp ark:- |"
feats_cv="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$dir/cv.scp ark:- |"

if $splice_feats; then
  feats_tr="$feats_tr splice-feats --left-context=1 --right-context=1 ark:- ark:- |"
  feats_cv="$feats_cv splice-feats --left-context=1 --right-context=1 ark:- ark:- |"
fi

if [ ! -z "$cache_dir" ]; then
  # Get the features from a previously saved directory
  if $copy_feats; then
    tmpdir=$(mktemp -dp "$feats_tmpdir")
    trap "echo \"Trying to move tmpdir $(hostname):$tmpdir to `pwd`\"; mv $tmpdir ." EXIT
    cp $cache_dir/labels.* $tmpdir

    copy-feats scp:$cache_dir/train_local.scp ark,scp:$tmpdir/train.ark,$tmpdir/train_local.scp || exit 1;
    copy-feats scp:$cache_dir/cv_local.scp ark,scp:$tmpdir/cv.ark,$tmpdir/cv_local.scp || exit 1;

    feats_tr="ark,s,cs:copy-feats scp:$tmpdir/train_local.scp ark:- |"
    feats_cv="ark,s,cs:copy-feats scp:$tmpdir/cv_local.scp ark:- |"
    labels_tr="ark:cat $tmpdir/labels.tr|"
    labels_cv="ark:cat $tmpdir/labels.cv|"
  else
    feats_tr="ark,s,cs:copy-feats scp:$cache_dir/train_local.scp ark:- |"
    feats_cv="ark,s,cs:copy-feats scp:$cache_dir/cv_local.scp ark:- |"
    labels_tr="ark:cat $cache_dir/labels.tr|"
    labels_cv="ark:cat $cache_dir/labels.cv|"	
  fi
    
elif $subsample_feats; then
  #Sub-sample the features 2 or 3 frames
  tmpdir=$(mktemp -dp "$feats_tmpdir")
  trap "echo \"Trying to move tmpdir $(hostname):$tmpdir to `pwd`\"; mv $tmpdir ." EXIT
  if [ ! $copy_feats ]; then
    echo "WARNING: subsample_feats implies copy_feats - if features have already been sub-sampled, set to 'false'"
  fi
  
  #copy-feats "$feats_tr" ark:$tmpdir/tr.tmp.ark || exit 1;
  subsample-feats --n=3 --offset=0 "$feats_tr" ark,scp:$tmpdir/train0.ark,$tmpdir/train0local.scp || exit 1;
  subsample-feats --n=3 --offset=1 "$feats_tr" ark,scp:$tmpdir/train1.ark,$tmpdir/train1local.scp || exit 1;
  subsample-feats --n=3 --offset=2 "$feats_tr" ark,scp:$tmpdir/train2.ark,$tmpdir/train2local.scp || exit 1;  

  subsample-feats --n=3 --offset=0 "$feats_cv" ark,scp:$tmpdir/cv0.ark,$tmpdir/cv0local.scp || exit 1;
  subsample-feats --n=3 --offset=1 "$feats_cv" ark,scp:$tmpdir/cv1.ark,$tmpdir/cv1local.scp || exit 1;
  subsample-feats --n=3 --offset=2 "$feats_cv" ark,scp:$tmpdir/cv2.ark,$tmpdir/cv2local.scp || exit 1;
  rm -f $tmpdir/tr.tmp.ark $tmpdir/cv.tmp.ark

  sed 's/^/0x/' $tmpdir/train0local.scp        > $tmpdir/train_local.scp
  sed 's/^/1x/' $tmpdir/train1local.scp | tac >> $tmpdir/train_local.scp
  sed 's/^/2x/' $tmpdir/train2local.scp | tee >(awk '!(NR%2)'       >> $tmpdir/train_local.scp) | \
                                                awk  '(NR%2)' | tac >> $tmpdir/train_local.scp
#  paste -d '\n' <(sed 's/^/0x/' $tmpdir/train0local.scp) \
#	<(sed 's/^/1x/' $tmpdir/train1local.scp) \
#	<(sed 's/^/2x/' $tmpdir/train2local.scp) | \
#      tee >(awk '!(NR%2)' >> $tmpdir/train_local.scp) | awk '(NR%2)' | tac >> $tmpdir/train_local.scp
  sed 's/^/0x/' $tmpdir/cv0local.scp        > $tmpdir/cv_local.scp
  sed 's/^/1x/' $tmpdir/cv1local.scp | tac >> $tmpdir/cv_local.scp
  sed 's/^/2x/' $tmpdir/cv2local.scp       >> $tmpdir/cv_local.scp

  feats_tr="ark,s,cs:copy-feats scp:$tmpdir/train_local.scp ark:- |"
  feats_cv="ark,s,cs:copy-feats scp:$tmpdir/cv_local.scp ark:- |"

  gzip -cd $dir/labels.tr.gz | sed 's/^/0x/'  > $tmpdir/labels.tr
  gzip -cd $dir/labels.tr.gz | sed 's/^/1x/' >> $tmpdir/labels.tr
  gzip -cd $dir/labels.tr.gz | sed 's/^/2x/' >> $tmpdir/labels.tr
  gzip -cd $dir/labels.cv.gz | sed 's/^/0x/'  > $tmpdir/labels.cv
  gzip -cd $dir/labels.cv.gz | sed 's/^/1x/' >> $tmpdir/labels.cv
  gzip -cd $dir/labels.cv.gz | sed 's/^/2x/' >> $tmpdir/labels.cv

  labels_tr="ark:cat $tmpdir/labels.tr|"
  labels_cv="ark:cat $tmpdir/labels.cv|"

else
  # Save the features to a local dir on the GPU machine. On Linux, this usually points to /tmp
  if $copy_feats; then
    tmpdir=$(mktemp -dp "$feats_tmpdir")
    trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir" EXIT

    copy-feats "$feats_tr" ark,scp:$tmpdir/train.ark,$tmpdir/train_local.scp || exit 1;
    copy-feats "$feats_cv" ark,scp:$tmpdir/cv.ark,$tmpdir/cv_local.scp || exit 1;
    feats_tr="ark,s,cs:copy-feats scp:$tmpdir/train_local.scp ark:- |"
    feats_cv="ark,s,cs:copy-feats scp:$tmpdir/cv_local.scp ark:- |"
  fi
fi

if $add_deltas; then
  feats_tr="$feats_tr add-deltas ark:- ark:- |"
  feats_cv="$feats_cv add-deltas ark:- ark:- |"
fi
## End of feature setup

# Initialize model parameters
if [ ! -f $dir/nnet/nnet.iter0 ]; then
  echo "Initializing model as $dir/nnet/nnet.iter0"
  net-initialize --binary=true $dir/nnet.proto $dir/nnet/nnet.iter0 >& $dir/log/initialize_model.log || exit 1;
fi

cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING STARTS [$cur_time]"
echo "[NOTE] TOKEN_ACCURACY refers to token accuracy, i.e., (1.0 - token_error_rate)."
for iter in $(seq $start_epoch_num $max_iters); do
    cvacc_prev=$cvacc
    echo -n "EPOCH $iter RUNNING ... "

    # train
    $train_tool --report-step=$report_step --num-sequence=$num_sequence --frame-limit=$frame_num_limit \
        --learn-rate=$learn_rate --momentum=$momentum \
        --verbose=$verbose \
        "$feats_tr" "$labels_tr" $dir/nnet/nnet.iter$[iter-1] $dir/nnet/nnet.iter${iter} \
        >& $dir/log/tr.iter$iter.log || exit 1;

    end_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
    echo -n "ENDS [$end_time]: "

    tracc=$(cat $dir/log/tr.iter${iter}.log | grep -a "TOKEN_ACCURACY" | tail -n 1 | awk '{ acc=$3; gsub("%","",acc); print acc; }')
    echo -n "lrate $(printf "%.6g" $learn_rate), TRAIN ACCURACY $(printf "%.4f" $tracc)%, "

    # validation
    $train_tool --report-step=$report_step --num-sequence=$valid_num_sequence --frame-limit=$frame_num_limit \
        --learn-rate=$learn_rate --momentum=$momentum \
        --verbose=$verbose --cross-validate=true \
        "$feats_cv" "$labels_cv" $dir/nnet/nnet.iter${iter} \
        >& $dir/log/cv.iter$iter.log || exit 1;

    cvacc=$(cat $dir/log/cv.iter${iter}.log | grep -a "TOKEN_ACCURACY" | tail -n 1 | awk '{ acc=$3; gsub("%","",acc); print acc; }')
    echo "VALID ACCURACY $(printf "%.4f" $cvacc)%"

    # stopping criterion
    intre='^[0-9]+$'
    rel_impr=$(bc <<< "($cvacc-$cvacc_prev)")
    if [ 1 == $halving -a 1 == $(bc <<< "$rel_impr < $end_halving_inc") ]; then
      if [[ ( "$min_iters" =~ $intre ) && ( $min_iters -gt $iter ) ]]; then
        echo we were supposed to finish, but we continue as min_iters : $min_iters
      else
        echo finished, too small rel. improvement $rel_impr
        break
      fi
    fi

    # start annealing when improvement is low
    if [ 1 == $(bc <<< "$rel_impr < $start_halving_inc") -a $iter -ge $halving_after_epoch ]; then
      halving=1
    fi
    if [[ ( "$force_halving_epoch" =~ $intre ) && ( $iter -ge $force_halving_epoch ) ]]; then
      halving=1
    fi
    
    # do annealing
    if [ 1 == $halving ]; then
      learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
      learn_rate=$(awk "BEGIN{if ($learn_rate<$final_learn_rate) {print $final_learn_rate} else {print $learn_rate}}")
    fi
    # save the status 
    echo $[$iter+1] > $dir/.epoch    # +1 because we save the epoch to start from
    echo $cvacc > $dir/.cvacc
    echo $halving > $dir/.halving
    echo $learn_rate > $dir/.lrate
done

# Convert the model marker from "<BiLstmParallel>" to "<BiLstm>"
format-to-nonparallel $dir/nnet/nnet.iter${iter} $dir/final.nnet >& $dir/log/model_to_nonparal.log || exit 1;

echo "Training succeeded. The final model $dir/final.nnet"
trap - EXIT
