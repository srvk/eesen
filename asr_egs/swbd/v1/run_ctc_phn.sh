#!/bin/bash

### for XSede comet cluster ###
### submit sbatch ---ignore-pbs train-2-gpu.sh
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --output=log/slurm-%j.out
#SBATCH --export=ALL
#SBATCH --time="48:00:00"
#SBATCH --mem=100G

### for CMU rocks cluster ###
#PBS -q standard
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -V
#PBS -l nodes=1:ppn=1

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

stage=1

# Set paths to various datasets
swbd=/path/to/LDC97S62
fisher_dirs="/path/to/LDC2004T19/fe_03_p1_tran/ /path/to/LDC2005T19/fe_03_p2_tran/" # Set to "" if you don't have the fisher corpus
eval2000_dirs="/path/to/LDC2002S09/hub5e_00 /path/to/LDC2002T43"

# Set paths to various datasets
swbd="/oasis/projects/nsf/cmu131/fmetze/LDC97S62"
fisher_dirs="/oasis/projects/nsf/cmu139/yajie/LDC/LDC2004T19/fe_03_p1_tran/ /oasis/projects/nsf/cmu131/fmetze/LDC2005T19/FE_03_P2_TRAN/" # Set to "" if you don't have the fisher corpus
eval2000_dirs="/oasis/projects/nsf/cmu131/fmetze/LDC2002S09/hub5e_00 /oasis/projects/nsf/cmu139/yajie/LDC/LDC2002T43"

# CMU Rocks
swbd=/data/ASR4/babel/ymiao/CTS/LDC97S62
fisher_dirs="/data/ASR5/babel/ymiao/Install/LDC/LDC2004T19/fe_03_p1_tran/ /data/ASR5/babel/ymiao/Install/LDC/LDC2005T19/fe_03_p2_tran/"
eval2000_dirs="/data/ASR4/babel/ymiao/CTS/LDC2002S09/hub5e_00 /data/ASR4/babel/ymiao/CTS/LDC2002T43"

. parse_options.sh

if [ $stage -le 1 ]; then
  echo =====================================================================
  echo "             Data Preparation and FST Construction                 "
  echo =====================================================================
  # Use the same data preparation script from Kaldi
  local/swbd1_data_prep.sh $swbd  || exit 1;

  # Construct the phoneme-based lexicon
  local/swbd1_prepare_phn_dict.sh || exit 1;

  # Compile the lexicon and token FSTs
  utils/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;

  # Train and compile LMs.
  local/swbd1_train_lms.sh data/local/train/text data/local/dict_phn/lexicon.txt data/local/lm $fisher_dirs || exit 1;

  # Compile the language-model FST and the final decoding graph TLG.fst
  local/swbd1_decode_graph.sh data/lang_phn data/local/dict_phn/lexicon.txt || exit 1;

  # Data preparation for the eval2000 set
  local/eval2000_data_prep.sh $eval2000_dirs
fi

if [ $stage -le 2 ]; then
  echo =====================================================================
  echo "                    FBank Feature Generation                       "
  echo =====================================================================
  fbankdir=fbank

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  steps/make_fbank.sh --cmd "$train_cmd" --nj 32 data/train exp/make_fbank/train $fbankdir || exit 1;
  utils/fix_data_dir.sh data/train || exit;
  steps/compute_cmvn_stats.sh data/train exp/make_fbank/train $fbankdir || exit 1;

  steps/make_fbank.sh --cmd "$train_cmd" --nj 10 data/eval2000 exp/make_fbank/eval2000 $fbankdir || exit 1;
  utils/fix_data_dir.sh data/eval2000 || exit;
  steps/compute_cmvn_stats.sh data/eval2000 exp/make_fbank/eval2000 $fbankdir || exit 1;

  # Use the first 4k sentences as dev set, around 5 hours
  utils/subset_data_dir.sh --first data/train 4000 data/train_dev
  n=$[`cat data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last data/train $n data/train_nodev

  # Create a smaller training set by selecting the first 100k utterances, around 110 hours
  utils/subset_data_dir.sh --first data/train_nodev 100000 data/train_100k
  local/remove_dup_utts.sh 200 data/train_100k data/train_100k_nodup

  # Finally the full training set, around 286 hours
  local/remove_dup_utts.sh 300 data/train_nodev data/train_nodup
fi

if [ $stage -le 3 ]; then
  echo =====================================================================
  echo "                Network Training with the 110-Hour Set             "
  echo =====================================================================
  # Specify network structure and generate the network topology
  input_feat_dim=120   # dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
  lstm_layer_num=4     # number of LSTM layers
  lstm_cell_dim=320    # number of memory cells in every LSTM layer

  dir=exp_110h/train_phn_l${lstm_layer_num}_c${lstm_cell_dim}
  mkdir -p $dir

  target_num=`cat data/lang_phn/units.txt | wc -l`; target_num=$[$target_num+1]; #  #targets = #labels + 1 (the blank)

  # Output the network topology
  utils/model_topo.py --input-feat-dim $input_feat_dim --lstm-layer-num $lstm_layer_num \
    --lstm-cell-dim $lstm_cell_dim --target-num $target_num \
    --fgate-bias-init 1.0 > $dir/nnet.proto || exit 1;

  # Label sequences; simply convert words into their label indices
  utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_100k_nodup/text "<unk>" | gzip -c - > $dir/labels.tr.gz
  utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_dev/text "<unk>" | gzip -c - > $dir/labels.cv.gz

  # Train the network with CTC. Refer to the script for details about the arguments
  steps/train_ctc_parallel.sh --add-deltas true --num-sequence 10 --frame-num-limit 25000 \
    --learn-rate 0.00004 --report-step 1000 --halving-after-epoch 12 \
    data/train_100k_nodup data/train_dev $dir || exit 1;

  echo =====================================================================
  echo "                            Decoding                               "
  echo =====================================================================
  # decoding
  for lm_suffix in sw1_tg sw1_fsh_tgpr; do
    steps/decode_ctc_lat.sh --cmd "$decode_cmd" --nj 20 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.6 \
      data/lang_phn_${lm_suffix} data/eval2000 $dir/decode_eval2000_${lm_suffix} || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  echo =====================================================================
  echo "                  Network Training with the Full Set               "
  echo =====================================================================
  input_feat_dim=120   # dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
  lstm_layer_num=5     # number of LSTM layers
  lstm_cell_dim=320    # number of memory cells in every LSTM layer

  dir=exp/train_phn_l${lstm_layer_num}_c${lstm_cell_dim}
  mkdir -p $dir

  target_num=`cat data/lang_phn/units.txt | wc -l`; target_num=$[$target_num+1]; # #targets = #labels + 1 (the blank)

  # Output the network topology
  utils/model_topo.py --input-feat-dim $input_feat_dim --lstm-layer-num $lstm_layer_num \
    --lstm-cell-dim $lstm_cell_dim --target-num $target_num \
    --fgate-bias-init 1.0 > $dir/nnet.proto || exit 1;

  # Label sequences; simply convert words into their label indices
  utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_nodup/text "<unk>" | gzip -c - > $dir/labels.tr.gz
  utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_dev/text "<unk>" | gzip -c - > $dir/labels.cv.gz

  # Train the network with CTC. Refer to the script for details about the arguments
  steps/train_ctc_parallel.sh --add-deltas true --num-sequence 20 --frame-num-limit 25000 \
    --learn-rate 0.00004 --report-step 1000 --halving-after-epoch 12 \
    data/train_nodup data/train_dev $dir || exit 1;

  echo =====================================================================
  echo "                            Decoding                               "
  echo =====================================================================
  # decoding
  for lm_suffix in sw1_tg sw1_fsh_tgpr; do
    steps/decode_ctc_lat.sh --cmd "$decode_cmd" --nj 20 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.6 \
      data/lang_phn_${lm_suffix} data/eval2000 $dir/decode_eval2000_${lm_suffix} || exit 1;
  done
fi
