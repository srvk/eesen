### for CMU rocks cluster ###
#PBS -M ramon.sanabria.teixidor@gmail.com
#PBS -q gpu
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -N eesen_tf_swbd_pipeline_phn
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

am_nproj=60
am_ninitproj=80
am_nfinalproj=100

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

  # Construct the phoneme-based lexicon
  local/swbd1_prepare_phn_dict.sh || exit 1;

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


  dir=exp/train_phn_l${am_nlayer}_c${am_ncell_dim}_m${am_model}_w${am_window}_n${am_norm}_p${am_nproj}_ip${am_ninitproj}_fp${am_ninitproj}

  mkdir -p $dir

  echo generating train labels...

  python ./local/swbd1_prepare_phn_dict_tf.py --phn_lexicon ./data/local/dict_phn/lexicon.txt --text_file ./data/train_nodup/text --output_units ./data/local/dict_phn/units.txt --output_labels $dir/labels.tr --ignore_noises || exit 1

  echo generating cv labels...

  python ./local/swbd1_prepare_phn_dict_tf.py --phn_lexicon ./data/local/dict_phn/lexicon.txt --text_file ./data/train_dev/text --output_units ./data/local/dict_phn/units.txt --output_labels $dir/labels.cv --ignore_noises || exit 1

  # Train the network with CTC. Refer to the script for details about the arguments
  steps/train_ctc_tf.sh --nlayer $am_nlayer --nhidden $am_ncell_dim  --batch_size 16 --learn_rate 0.005 --half_after 6 --model $am_model --window $am_window --ninitproj $am_ninitproj --nproj $am_nproj --nfinalproj $am_nfinalproj --norm $am_norm data/train_nodup data/train_dev $dir || exit 1;

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

