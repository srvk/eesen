#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh


echo =====================================================================
echo "             Data Preparation and FST Construction                 "
echo =====================================================================
# Use the same datap prepatation script from Kaldi
local/hkust_data_prep.sh /path/to/LDC2005S15/ /path/to/LDC2005T32/  || exit 1;

# Run the original script for dict preparation
local/hkust_prepare_dict.sh || exit 1;

# Construct the character-based dict. 
# We get 3667 tokens, inclduing Madarin characters, English characters,
# various noise marks, and the space.
local/hkust_prepare_char_dict.sh || exit 1;

# Compile the lexicon and token FSTs
utils/ctc_compile_dict_token.sh --dict-type "char" --space-char "<space>" \
  data/local/dict_char data/local/lang_char_tmp data/lang_char || exit 1;

# Train and compile LMs. 
local/hkust_train_lms.sh data/local/train/text data/local/dict_char/lexicon.txt data/local/lm_char || exit 1;

# Compile the language-model FST and the final decoding graph TLG.fst
local/hkust_decode_graph.sh data/local/lm_char data/lang_char data/lang_char_test || exit 1;

echo =====================================================================
echo "                    FBank Feature Generation                       "
echo =====================================================================
fbankdir=fbank

# Generate the fbank features; by default 40-dimensional fbanks on each frame
steps/make_fbank.sh --cmd "$train_cmd" --nj 32 data/train exp/make_fbank/train $fbankdir || exit 1;
utils/fix_data_dir.sh data/train || exit;
steps/compute_cmvn_stats.sh data/train exp/make_fbank/train $fbankdir || exit 1;

steps/make_fbank.sh --cmd "$train_cmd" --nj 10 data/dev exp/make_fbank/dev $fbankdir || exit 1;
utils/fix_data_dir.sh data/dev || exit;
steps/compute_cmvn_stats.sh data/dev exp/make_fbank/dev $fbankdir || exit 1;

# Use the first 4k sentences as dev set, around 5 hours
utils/subset_data_dir.sh --first data/train 4000 data/train_dev
n=$[`cat data/train/segments | wc -l` - 4000]
utils/subset_data_dir.sh --last data/train $n data/train_nodev

echo =====================================================================
echo "                          Model Training                           "
echo =====================================================================
# Specify network structure and generate the network topology
input_feat_dim=120   # dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
lstm_layer_num=5     # number of LSTM layers
lstm_cell_dim=320    # number of memory cells in every LSTM layer

dir=exp/train_char_l${lstm_layer_num}_c${lstm_cell_dim}
mkdir -p $dir

target_num=`cat data/lang_char/units.txt | wc -l`; target_num=$[$target_num+1]; #  #targets = #labels + 1 (the blank)

# Output the network topology
utils/model_topo.py --input-feat-dim $input_feat_dim --lstm-layer-num $lstm_layer_num \
  --lstm-cell-dim $lstm_cell_dim --target-num $target_num \
  --fgate-bias-init 1.0 > $dir/nnet.proto || exit 1;

# Label sequences; simply convert words into their label indices
utils/prep_ctc_trans.py data/lang_char/lexicon_numbers.txt data/train_nodev/text \
  "<UNK>" "<space>" | gzip -c - > $dir/labels.tr.gz
utils/prep_ctc_trans.py data/lang_char/lexicon_numbers.txt data/train_dev/text \
  "<UNK>" "<space>" | gzip -c - > $dir/labels.cv.gz

# Train the network with CTC. Refer to the script for details about the arguments
steps/train_ctc_parallel.sh --add-deltas true --num-sequence 10 \
    --learn-rate 0.00004 --report-step 1000 --halving-after-epoch 12 \
    --feats-tmpdir $dir/XXXXX \
    data/train_nodev data/train_dev $dir || exit 1;

echo =====================================================================
echo "                             Decoding                              "
echo =====================================================================
# decoding
steps/decode_ctc_lat.sh --cmd "$decode_cmd" --nj 20 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.9 \
    data/lang_char_test data/dev $dir/decode_dev || exit 1;

echo =====================================================================
echo "                    Pitch Feature Generation                       "
echo =====================================================================
fbankdir=pitch

for set in train dev; do
  cp -r data/$set data/${set}_pitch
  ( cd data/${set}_pitch; rm -rf feats.scp cmvn.scp split*; )
done

# Generate the fbank+pitch features, 43 dimensions on each frame
steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 data/train_pitch exp/make_pitch/train $fbankdir || exit 1;
utils/fix_data_dir.sh data/train_pitch || exit;
steps/compute_cmvn_stats.sh data/train_pitch exp/make_pitch/train $fbankdir || exit 1;

steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 data/dev_pitch exp/make_pitch/dev $fbankdir || exit 1;
utils/fix_data_dir.sh data/dev_pitch || exit;
steps/compute_cmvn_stats.sh data/dev_pitch exp/make_pitch/dev $fbankdir || exit 1;

# Use the first 4k sentences as dev set, around 5 hours
utils/subset_data_dir.sh --first data/train_pitch 4000 data/train_dev_pitch
n=$[`cat data/train/segments | wc -l` - 4000]
utils/subset_data_dir.sh --last data/train_pitch $n data/train_nodev_pitch

echo =====================================================================
echo "                Model Training with FBank+Pitch Features           "
echo =====================================================================
# Specify network structure and generate the network topology
input_feat_dim=129   # dimension of the input features; we will use 43-dimensional fbanks+pitch with deltas and double deltas
lstm_layer_num=5     # number of LSTM layers
lstm_cell_dim=320    # number of memory cells in every LSTM layer

dir=exp/train_char_l${lstm_layer_num}_c${lstm_cell_dim}_pitch
mkdir -p $dir

target_num=`cat data/lang_char/units.txt | wc -l`; target_num=$[$target_num+1]; #  #targets = #labels + 1 (the blank)

# Output the network topology
utils/model_topo.py --input-feat-dim $input_feat_dim --lstm-layer-num $lstm_layer_num \
  --lstm-cell-dim $lstm_cell_dim --target-num $target_num \
  --fgate-bias-init 1.0 > $dir/nnet.proto || exit 1;

# Label sequences; simply convert words into their label indices
utils/prep_ctc_trans.py data/lang_char/lexicon_numbers.txt data/train_nodev/text \
  "<UNK>" "<space>" | gzip -c - > $dir/labels.tr.gz
utils/prep_ctc_trans.py data/lang_char/lexicon_numbers.txt data/train_dev/text \
  "<UNK>" "<space>" | gzip -c - > $dir/labels.cv.gz

# Train the network with CTC. Refer to the script for details about the arguments
steps/train_ctc_parallel.sh --add-deltas true --num-sequence 10 \
    --learn-rate 0.00004 --report-step 1000 --halving-after-epoch 12 \
    --feats-tmpdir $dir/XXXXX \
    data/train_nodev_pitch data/train_dev_pitch $dir || exit 1;

echo =====================================================================
echo "                             Decoding                              "
echo =====================================================================
# decoding
steps/decode_ctc_lat.sh --cmd "$decode_cmd" --nj 20 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.9 \
    data/lang_char_test data/dev_pitch $dir/decode_dev || exit 1;
