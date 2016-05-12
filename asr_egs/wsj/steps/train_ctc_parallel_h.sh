#!/bin/bash
{
# Copyright 2015  Yajie Miao    (Carnegie Mellon University)
#           2015  Hang Su
# Apache 2.0

# This script trains acoustic models based on CTC and using SGD. 

## Begin configuration section
train_tool=train-ctc-parallel  # the command for training; by default, we use the
                # parallel version which processes multiple utterances at the same time 

# configs for multiple sequences
num_sequence=5           # during training, how many utterances to be processed in parallel
valid_num_sequence=10    # number of parallel sequences in validation
frame_num_limit=1000000  # the number of frames to be processed at a time in training; this config acts to
         # to prevent running out of GPU memory if #num_sequence very long sequences are processed;the max
         # number of training examples is decided by if num_sequence or frame_num_limit is reached first. 

# learning rate
learn_rate=0.0001        # learning rate
momentum=0.9             # momentum

# learning rate schedule
max_iters=25             # max number of iterations
min_iters=               # min number of iterations
start_epoch_num=1        # start from which epoch, used for resuming training from a break point

start_halving_inc=0.5    # start halving learning rates when the accuracy improvement falls below this amount
end_halving_inc=0.1      # terminate training when the accuracy improvement falls below this amount
halving_factor=0.5       # learning rate decay factor
halving_after_epoch=1    # halving bcomes enabled after this many epochs

# logging
report_step=100          # during training, the step (number of utterances) of reporting objective and accuracy
verbose=1

# feature configs
sort_by_len=true         # whether to sort the utterances by their lengths

norm_vars=true           # whether to apply variance normalization when we do cmn
add_deltas=true          # whether to add deltas

# status of learning rate schedule; useful when training is resumed from a break point
cvacc=-1
halving=0

# Multi-GPU training
nj=1
utts_per_avg=700

clean_up=true

## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; 
[ -f ./cmd.sh ] && . ./cmd.sh; 

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
  [ ! -f $f ] && echo "train_ctc_parallel.sh: no such file $f" && exit 1;
done

## Read the training status for resuming
[ -f $dir/.epoch ] && start_epoch_num=`cat $dir/.epoch 2>/dev/null`
[ -f $dir/.cvacc ] && cvacc=`cat $dir/.cvacc 2>/dev/null`
[ -f $dir/.halving ] && halving=`cat $dir/.halving 2>/dev/null`
[ -f $dir/.lrate ] && learn_rate=`cat $dir/.lrate 2>/dev/null`

## Setup up features
echo $norm_vars > $dir/norm_vars  # output feature configs which will be used in decoding
echo $add_deltas > $dir/add_deltas

echo "Preparing train and cv features"
tmpdir=$dir/feats; 
[ -d $tmpdir ] || mkdir -p $tmpdir
[ $clean_up == true ] && trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir" EXIT
utils/prep_scps.sh --nj $nj --cmd "$train_cmd" ${seed:+ --seed=$seed} --clean-up $clean_up \
  $data_tr/feats.scp $data_cv/feats.scp $num_sequence $frame_num_limit $tmpdir $dir || exit 1;

feats_tr="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$dir/feats_tr.JOB.scp ark:- |"
feats_cv="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$dir/feats_cv.JOB.scp ark:- |"

if [ $nj -eq 1 ]; then
  feats_tr=$(echo $feats_tr | sed 's#JOB#1#')
  feats_cv=$(echo $feats_cv | sed 's#JOB#1#')
fi

if $add_deltas; then
  feats_tr="$feats_tr add-deltas ark:- ark:- |"
  feats_cv="$feats_cv add-deltas ark:- ark:- |"
fi
## End of feature setup

## Set up labels  
labels_tr="ark:gunzip -c $dir/labels.tr.gz|"
labels_cv="ark:gunzip -c $dir/labels.cv.gz|"
# Compute the occurrence counts of labels in the label sequences. These counts will be used to derive prior probabilities of
# the labels.
gunzip -c $dir/labels.tr.gz | awk '{line=$0; gsub(" "," 0 ",line); print line " 0";}' | \
  analyze-counts --verbose=1 --binary=false ark:- $dir/label.counts >& $dir/log/compute_label_counts.log || exit 1
##

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
    if [ -z "$nj" ]; then
      $train_tool --report-step=$report_step --num-sequence=$num_sequence --frame-limit=$frame_num_limit \
        --learn-rate=$learn_rate --momentum=$momentum \
        --verbose=$verbose \
        "$feats_tr" "$labels_tr" $dir/nnet/nnet.iter$[iter-1] $dir/nnet/nnet.iter${iter} \
        >& $dir/log/tr.iter$iter.log || exit 1;
      tracc=$(cat $dir/log/tr.iter${iter}.log | grep "TOKEN_ACCURACY" | tail -n 1 | awk '{ acc=$3; gsub("%","",acc); print acc; }')
    else
      $cuda_cmd JOB=1:$nj $dir/log/tr.iter$iter.JOB.log \
        $train_tool --report-step=$report_step --num-sequence=$num_sequence --frame-limit=$frame_num_limit \
        --learn-rate=$learn_rate --momentum=$momentum --num-jobs=$nj --job-id=JOB \
        --verbose=$verbose \
        ${utts_per_avg:+ --utts-per-avg=$utts_per_avg} \
        "$feats_tr" "$labels_tr" $dir/nnet/nnet.iter$[iter-1] $dir/nnet/nnet.iter${iter} >& $dir/log/tr.iter$iter.log || exit 1
      tracc=$(cat $dir/log/tr.iter${iter}.1.log | grep "TOTAL TOKEN_ACCURACY" | tail -n 1 | awk '{ acc=$(NF-1); gsub("%","",acc); print acc; }')
    fi


    echo -n "lrate $(printf "%.6g" $learn_rate), TRAIN ACCURACY $(printf "%.4f" $tracc)%, "
    end_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
    echo -n "ENDS [$end_time]: "

    # validation
    if [ -z "$nj" ]; then
      $train_tool --report-step=$report_step --num-sequence=$valid_num_sequence --frame-limit=$frame_num_limit \
        --cross-validate=true \
        --learn-rate=$learn_rate \
        --verbose=$verbose \
        "$feats_cv" "$labels_cv" $dir/nnet/nnet.iter${iter} \
        >& $dir/log/cv.iter$iter.log || exit 1;
      cvacc=$(cat $dir/log/cv.iter${iter}.log | grep "TOKEN_ACCURACY" | tail -n 1 | awk '{ acc=$3; gsub("%","",acc); print acc; }')
    else
      $cuda_cmd JOB=1:$nj $dir/log/cv.iter$iter.JOB.log \
        $train_tool --report-step=$report_step --num-sequence=$valid_num_sequence --frame-limit=$frame_num_limit \
        --cross-validate=true --num-jobs=$nj --job-id=JOB \
        --learn-rate=$learn_rate \
        --verbose=$verbose \
        "$feats_cv" "$labels_cv" $dir/nnet/nnet.iter${iter} >& $dir/log/cv.iter$iter.log || exit 1;
      cvacc=$(cat $dir/log/cv.iter${iter}.1.log | grep "TOTAL TOKEN_ACCURACY" | tail -n 1 | awk '{ acc=$(NF-1); gsub("%","",acc); print acc; }')
    fi

    echo "VALID ACCURACY $(printf "%.4f" $cvacc)%"

    # stopping criterion
    rel_impr=$(bc <<< "($cvacc-$cvacc_prev)")
    if [[ 1 == "$halving" && 1 == $(bc <<< "$rel_impr < $end_halving_inc") ]]; then
      if [[ "$min_iters" != "" ]]; then
        if [ $min_iters -gt $iter ]; then
          echo we were supposed to finish, but we continue as min_iters : $min_iters
          continue
        fi
      fi
      echo finished, too small rel. improvement $rel_impr
      break
    fi

    # start annealing when improvement is low
    if [ 1 == $(bc <<< "$rel_impr < $start_halving_inc") ]; then
      if [ $iter -gt $halving_after_epoch ]; then
        halving=1
      fi
    fi

    # do annealing
    if [ 1 == $halving ]; then
      learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
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
}
