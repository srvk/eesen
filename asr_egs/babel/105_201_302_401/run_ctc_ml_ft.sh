### for CMU rocks cluster ###
#PBS -M ramon.sanabria.teixidor@gmail.com
#PBS -q gpu
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -N babel_ml_l6_c360_p0w3_ka
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
am_nlayer=6
am_ncell_dim=360
am_model=deepbilstm
am_window=3
am_projection=0
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

  dir=exp/train_char_phn_ml_l${am_nlayer}_c${am_ncell_dim}_m${am_model}_w${am_window}_n${am_norm}_p${am_projection}_0.02_e6_tur

  mkdir -p $dir
  all_language=("data_ml/302-kazakh-flp")

steps/train_ctc_tf_ml.sh --nlayer $am_nlayer --nhidden $am_ncell_dim --continue_ckpt exp/train_char_phn_ml_l6_c360_mdeepbilstm_w3_nfalse_p0_0.02/model/epoch06.ckpt  --batch_size  16 --nproj $am_projection --window $am_window --import_config exp/train_char_phn_ml_l6_c360_mdeepbilstm_w3_nfalse_p0_0.02/model/config.pkl --force_lr_epoch_ckpt true --learn_rate 0.005 --half_after 4 --model $am_model --norm  $am_norm "${all_language[@]}" $dir || exit 1;

  exit

fi


