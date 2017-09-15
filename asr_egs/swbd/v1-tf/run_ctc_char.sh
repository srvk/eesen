### for CMU rocks cluster ###
#PBS -M ramon.sanabria.teixidor@gmail.com
#PBS -q gpu
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -N eesen_tf_swbd_pipeline_char
#PBS -V
#PBS -l walltime=48:00:00
#PBS -l nodes=1:ppn=1


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

stage=4

fisher_dirs="/path/to/LDC2004T19/fe_03_p1_tran/ /path/to/LDC2005T19/fe_03_p2_tran/" # Set to "" if you don't have the fisher corpus
eval2000_dirs="/path/to/LDC2002S09/hub5e_00 /path/to/LDC2002T43"

# CMU Rocks
#swbd=/data/ASR4/babel/ymiao/CTS/LDC97S62
#fisher_dirs="/data/ASR5/babel/ymiao/Install/LDC/LDC2004T19/fe_03_p1_tran/ /data/ASR5/babel/ymiao/Install/LDC/LDC2005T19/fe_03_p2_tran/"
#eval2000_dirs="/data/ASR4/babel/ymiao/CTS/LDC2002S09/hub5e_00 /data/ASR4/babel/ymiao/CTS/LDC2002T43"

. parse_options.sh


#acoustic model parameters
am_nlayer=4
am_ncell_dim=320
am_model=deepbilstm
am_window=3
am_norm=false


#language model parameters
fisher_dir_a="/data/ASR5/babel/ymiao/Install/LDC/LDC2004T19/fe_03_p1_tran/"
fisher_dir_b="/data/ASR5/babel/ymiao/Install/LDC/LDC2005T19/fe_03_p2_tran/"

lm_embed_size=64
lm_batch_size=32
lm_nlayer=1
lm_ncell_dim=320
lm_drop_out=0.5
lm_optimizer="adam"

fisher_text_dir="./data/fisher/"


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


  dir=exp/train_char_l${am_nlayer}_c${am_ncell_dim}_m${am_model}_w${am_window}_n${am_norm}

  mkdir -p $dir

  echo generating train labels...

  python ./local/swbd1_prepare_char_dict_tf.py --text_file ./data/train_nodup/text --output_units ./data/local/dict_char/units.txt --output_labels $dir/labels.tr --lower_case --ignore_noises || exit 1

  echo generating cv labels...

  python ./local/swbd1_prepare_char_dict_tf.py --text_file ./data/train_dev/text --input_units ./data/local/dict_char/units.txt --output_labels $dir/labels.cv || exit 1

  # Train the network with CTC. Refer to the script for details about the arguments
  steps/train_ctc_tf.sh --nlayer $am_nlayer --nhidden $am_ncell_dim  --batch_size 16 --learn_rate 0.01 --half_after 6 --model $am_model --window $am_window --continue_ckpt ./exp/train_char_l4_c320_mdeepbilstm_w3_nfalse/model/epoch08.ckpt --norm $am_norm data/train_nodup data/train_dev $dir || exit 1;

  exit


  echo =====================================================================
  echo "                   Decoding eval200 using AM                      "
  echo =====================================================================

  epoch=epoch14.ckpt
  filename=$(basename "$epoch")
  name_exp="${filename%.*}"

  data=./data/eval2000/
  weights=$dir/model/$epoch
  config=$dir/model/config.pkl
  results=$dir/results/$name_exp

  ./steps/decode_ctc_am_tf.sh --config $config --data $data --weights $weights --results $results

fi

if [ $stage -le 5 ]; then
  echo =====================================================================
  echo "             Char RNN LM Training with the Full Set                "
  echo =====================================================================

  dir=exp/train_lm_char_l${lstm_layer_num}_c${lstm_cell_dim}_e${lm_embed_size}_d${drop_out}_o${optimizer}/

  mkdir -p $dir
  mkdir -p ./data/local/dict_char_lm/

  echo ""
  echo creating labels files from train...
  echo ""

  python ./local/swbd1_prepare_char_dict_tf.py --text_file ./data/train_nodup/text --input_units ./data/local/dict_char/units.txt --output_units ./data/local/dict_char_lm/units.txt --output_labels $dir/labels.tr --lm

  echo ""
  echo creating word list from train...
  echo ""

  python ./local/swbd1_prepare_word_list_tf.py --text_file ./data/train_nodup/text --output_word_list $dir/words.tr --ignore_noises


  echo ""
  echo creating labels files from cv...
  echo ""

  python ./local/swbd1_prepare_char_dict_tf.py --text_file ./data/train_dev/text --input_units ./data/local/dict_char_lm/units.txt --output_labels $dir/labels.cv --lm

  echo ""
  echo creating word list from cv...
  echo ""

  python ./local/swbd1_prepare_word_list_tf.py --text_file ./data/train_dev/text --output_word_list $dir/words.cv --ignore_noises

  echo ""
  echo generating fisher_data...
  echo ""

  ./local/swbd1_create_fisher_text.sh ./data/fisher/ $fisher_dir_a $fisher_dir_b


  echo ""
  echo creating labels files from fisher...
  echo ""

  python ./local/swbd1_prepare_char_dict_tf.py --text_file ./data/fisher/text --input_units ./data/local/dict_char_lm/units.txt --output_labels $dir/labels.fisher --lm

  echo ""
  echo creating word list from fisher...
  echo ""

  python ./local/swbd1_prepare_word_list_tf.py --text_file ./data/fisher/text --output_word_list $dir/words.fisher --ignore_noises


  echo ""
  echo fusing swbd data with fisher data...
  echo ""

  cat $dir/labels.fisher >> $dir/labels.tr

  echo ""
  echo fusing words files...
  echo ""

  cat $dir/words.fisher > $dir/words
  cat $dir/words.cv >> $dir/words
  cat $dir/words.tr >> $dir/words

  echo ""
  echo training with full swbd text...
  echo ""

  #./steps/train_char_lm.sh --train_dir $dir --nembed $lm_embed_size --nlayer $lm_nlayer --nhidden $lm_ncell_dim --batch_size $lm_batch_size --nepoch 100 --train_labels $dir/labels.tr --cv_labels $dir/labels.cv --drop_out $lm_drop_out --optimizer ${lm_optimizer}

fi

if [ $stage -le 6 ]; then
  echo =====================================================================
  echo "             	Decode Eval 2000 (AM + (char) LM)                  "
  echo =====================================================================


  #./steps/decode_ctc_am_tf.sh --config /data/ASR5/sdalmia_1/fall2017/swbd/v1-tf/exp/train_char_l4_c320_mdeepbilstm_w3_nfalse/model/config.pkl --data ./data/eval2000/ --weights /data/ASR5/sdalmia_1/fall2017/swbd/v1-tf/exp/train_char_l4_c320_mdeepbilstm_w3_nfalse/model/epoch04.ckpt --results ./results/am/

fi
