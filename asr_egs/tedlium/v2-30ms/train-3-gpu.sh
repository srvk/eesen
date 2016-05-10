#!/bin/bash

#PBS -q gpu
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -V
#PBS -l walltime=48:00:00

uname -a
date

. ./cmd.sh
. ./path.sh

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh

# Specify network structure and generate the network topology
input_feat_dim=`feat-to-dim scp:data/train/feats.scp ark,t:- 2>/dev/null|head -n 1|awk '{print 3*$2}'`
# dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
lstm_layer_num=5     # number of LSTM layers
lstm_cell_dim=320    # number of memory cells in every LSTM layer

dir=exp/train_phn_l${lstm_layer_num}_c${lstm_cell_dim}_w3
mkdir -p $dir
    
# Output the network topology
[ ! -f $dir/nnet.proto ] && utils/model_topo.py --input-feat-dim ${input_feat_dim} --lstm-layer-num ${lstm_layer_num} --fgate-bias-init 1.0 \
    --lstm-cell-dim ${lstm_cell_dim} --target-num `cat data/lang_phn/units.txt | awk 'END {print 1+$2}'` > $dir/nnet.proto
utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_tr95/text "<UNK>" | gzip -c - > $dir/labels.tr.gz
utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_cv05/text "<UNK>" | gzip -c - > $dir/labels.cv.gz

# Train the network with CTC. Refer to the script for details about the arguments
steps/train_ctc_parallel_x3.sh --end-halving-inc 0.001 --halving-factor 0.7 \
    --add-deltas false --num-sequence 20 --frame-num-limit 20000 \
    --learn-rate 4e-5 --report-step 1000 --halving-after-epoch 12 --min-iters 28 \
    --max-iters 32 --splice-feats true --subsample-feats true --min-len 20 \
    data/train_tr95 data/train_cv05 $dir
#  --final-learn-rate 2e-6 --min-iters 28
