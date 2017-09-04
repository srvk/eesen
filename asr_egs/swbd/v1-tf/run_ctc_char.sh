### for CMU rocks cluster ###
#PBS -M ramon.sanabria.teixidor@gmail.com
#PBS -q gpu
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -N lm_swbd_full
#PBS -V
#PBS -l walltime=48:00:00
#PBS -l nodes=1:ppn=1


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

stage=5

fisher_dirs="/path/to/LDC2004T19/fe_03_p1_tran/ /path/to/LDC2005T19/fe_03_p2_tran/" # Set to "" if you don't have the fisher corpus
eval2000_dirs="/path/to/LDC2002S09/hub5e_00 /path/to/LDC2002T43"

# CMU Rocks
swbd=/data/ASR4/babel/ymiao/CTS/LDC97S62
fisher_dirs="/data/ASR5/babel/ymiao/Install/LDC/LDC2004T19/fe_03_p1_tran/ /data/ASR5/babel/ymiao/Install/LDC/LDC2005T19/fe_03_p2_tran/"
eval2000_dirs="/data/ASR4/babel/ymiao/CTS/LDC2002S09/hub5e_00 /data/ASR4/babel/ymiao/CTS/LDC2002T43"




. parse_options.sh

if [ $stage -le 1 ]; then
  echo =====================================================================
  echo "                       Data Preparation                            "
  echo =====================================================================

  # Use the same datap prepatation script from Kaldi
  local/swbd1_data_prep.sh $swbd  || exit 1;

  # Represent word spellings using a dictionary-like format
  local/swbd1_prepare_char_dict.sh || exit 1;

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
  steps/compute_cmvn_stats.sh data/train exp/make_fbank/train $fbankdir || exit 1;
  utils/fix_data_dir.sh data/train || exit;

  steps/make_fbank.sh --cmd "$train_cmd" --nj 10 data/eval2000 exp/make_fbank/eval2000 $fbankdir || exit 1;
  steps/compute_cmvn_stats.sh data/eval2000 exp/make_fbank/eval2000 $fbankdir || exit 1;
  utils/fix_data_dir.sh data/eval2000 || exit;

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

if [ $stage -le 4 ]; then
  echo =====================================================================
  echo "                Training AM with the Full Set                      "
  echo =====================================================================

  # Specify network structure and generate the network topology
  input_feat_dim=120   # dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
  lstm_layer_num=4     # number of LSTM layers
  lstm_cell_dim=320    # number of memory cells in every LSTM layer
  model=deepbilstm
  window=3
  norm=false

  dir=exp/train_char_l${lstm_layer_num}_c${lstm_cell_dim}_m${model}_w${window}_n${norm}

  mkdir -p $dir

  echo generating train labels...

  python ./local/swbd1_prepare_char_dict_tf.py --text_file ./data/train_nodup/text --output_units ./data/local/dict_char/units.txt --output_labels $dir/labels.tr --lower_case --ignore_noises

  echo generating cv labels...

  python ./local/swbd1_prepare_char_dict_tf.py --text_file ./data/train_dev/text --input_units ./data/local/dict_char/units.txt --output_labels $dir/labels.cv
  exit

  # Train the network with CTC. Refer to the script for details about the arguments
  steps/train_ctc_tf.sh --num-sequence 16 --learn-rate 0.01 --half_after 6 --model $model --window $window --norm $norm --continue_ckpt /data/ASR5/ramons_2/sinbad_projects/youtube_project/am/eesen_20170714/asr_egs/swbd/v1-tf/exp/train_char_l4_c320_mdeepbilstm_w3_nfalse/model/epoch11.ckpt \
    data/train_nodup data/train_dev $dir || exit 1;

fi

if [ $stage -le 5 ]; then
  echo =====================================================================
  echo "             Char RNN LM Training with the Full Set                "
  echo =====================================================================
  fisher_dir_a="/data/ASR5/babel/ymiao/Install/LDC/LDC2004T19/fe_03_p1_tran/"
  fisher_dir_b="/data/ASR5/babel/ymiao/Install/LDC/LDC2005T19/fe_03_p2_tran/"

  embed_size=64   # dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
  batch_size=32
  drop_out=0.5
  lstm_layer_num=1     # number of LSTM layers
  lstm_cell_dim=1024    # number of memory cells in every LSTM layer
  optimizer="adam"    # number of memory cells in every LSTM layer

  fisher_text_dir="./data/fisher/"

  dir=exp/train_lm_char_l${lstm_layer_num}_c${lstm_cell_dim}_e${embed_size}_d${drop_out}_o${optimizer}/

  mkdir -p $dir
  mkdir -p ./data/local/dict_char_lm/

  echo ""
  echo creating labels files from train...
  echo ""

  python ./local/swbd1_prepare_char_dict_tf.py --text_file ./data/train_nodup/text --input_units ./data/local/dict_char/units.txt --output_units ./data/local/dict_char_lm/units.txt --output_labels $dir/labels.tr --lm

  echo ""
  echo creating labels files from cv...
  echo ""

  python ./local/swbd1_prepare_char_dict_tf.py --text_file ./data/train_dev/text --input_units ./data/local/dict_char_lm/units.txt --output_labels $dir/labels.cv --lm

  echo ""
  echo generating fisher_data...
  echo ""

  ./local/swbd1_create_fisher_text.sh ./data/fisher/ $fisher_dir_a $fisher_dir_b

  echo ""
  echo fisher data generated in $fisher_text_dir...
  echo ""

  echo ""
  echo creating labels files from fisher...
  echo ""

  python ./local/swbd1_prepare_char_dict_tf.py --text_file ./data/fisher/text --input_units ./data/local/dict_char_lm/units.txt --output_labels $dir/labels.fisher --lm

  echo ""
  echo fisher labels generated in $dir/labels.fisher
  echo ""

  echo ""
  echo fusing swbd data with fisher data...
  echo ""

  cat $dir/labels.fisher >> $dir/labels.tr

  echo ""
  echo training with full swbd text...
  echo ""

  ./steps/train_char_lm.sh --train_dir $dir --nembed $embed_size --nlayer $lstm_layer_num --nhidden $lstm_cell_dim --batch_size $batch_size --nepoch 100 --train_labels $dir/labels.tr --cv_labels $dir/labels.cv --drop_out $drop_out --optimizer ${optimizer}


fi

