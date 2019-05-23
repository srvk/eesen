#!/bin/bash
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

. parse_options.sh


echo =====================================================================
echo "                          Model Training                           "
echo =====================================================================
# Specify network structure and generate the network topology
input_feat_dim=120   # dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
lstm_layer_num=4     # number of LSTM layers
lstm_cell_dim=320    # number of memory cells in every LSTM layer

dir=exp/model_l${lstm_layer_num}_c${lstm_cell_dim}
mkdir -p $dir

target_num=`cat data/lang/units.txt | wc -l`; target_num=$[$target_num+1]; #  #targets = #labels + 1 (the blank)

# Output the network topology
utils/model_topo.py --input-feat-dim $input_feat_dim --lstm-layer-num $lstm_layer_num --lstm-cell-dim $lstm_cell_dim --target-num $target_num --fgate-bias-init 1.0 > $dir/nnet.proto || exit 1;

# Label sequences; simply convert words into their label indices
utils/prep_ctc_trans.py data/lang/lexicon_numbers.txt data/train/text "<UNK>" | gzip -c - > $dir/labels.tr.gz

utils/prep_ctc_trans.py data/lang/lexicon_numbers.txt data/dev/text "<UNK>" | gzip -c - > $dir/labels.cv.gz

# Train the network with CTC. Refer to the script for details about the arguments
steps/train_ctc_parallel.sh  --add-deltas true --num-sequence 10 --learn-rate 0.00004 --report-step 100 --halving-after-epoch 12 --feats-tmpdir $dir/XXXXX data/train data/dev $dir || exit 1;
echo -e "\n"
