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

stage=1

fisher_dirs="/data/corpora/swb/fe_03_p1_tran/ /data/corpora/swb/fe_03_p2_tran/" # Set to "" if you don't have the fisher corpus
eval2000_dirs="/data/corpora/swb/hub5e_00 /data/corpora/swb/2000_hub5_eng_eval_tr"
swbd=/data/corpora/swb/swb1

feattype=fbank_pitch

# CMU Rocks
#swbd=/data/ASR4/babel/ymiao/CTS/LDC97S62
#fisher_dirs="/data/ASR5/babel/ymiao/Install/LDC/LDC2004T19/fe_03_p1_tran/ /data/ASR5/babel/ymiao/Install/LDC/LDC2005T19/fe_03_p2_tran/"
#eval2000_dirs="/data/ASR4/babel/ymiao/CTS/LDC2002S09/hub5e_00 /data/ASR4/babel/ymiao/CTS/LDC2002T43"

. parse_options.sh


#acoustic model parameters
am_nlayer=5

#am_nproj=60
#am_nproj=80  # use this setting for the bigger net in RESULTS
am_nproj=100  # use this setting for the bigger net in RESULTS
am_ninitproj=80
am_nfinalproj=100

#am_ncell_dim=320
#am_ncell_dim=400  # use this setting for the bigger net in RESULTS
am_ncell_dim=480  # use this setting for the bigger net in RESULTS

am_model=deepbilstm
am_window=3
am_norm=true

#lr_rate=0.05
lrscheduler="newbob"
lr_spec="lr_rate=0.05,half_after=3"


#language model parameters
fisher_dir_a="/data/corpora/swb/fe_03_p1_tran/"
fisher_dir_b="/data/corpora/swb/fe_03_p2_tran/"

lm_embed_size=64
lm_batch_size=16
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


  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  steps/make_${feattype}.sh --cmd "$train_cmd" --nj 4 data/train exp/make_${feattype}/train $feattype || exit 1;
  steps/compute_cmvn_stats.sh data/train exp/make_${feattype}/train $feattype || exit 1;
  utils/fix_data_dir.sh data/train || exit;

  steps/make_${feattype}.sh --cmd "$train_cmd" --nj 4 data/eval2000 exp/make_${feattype}/eval2000 $feattype || exit 1;
  steps/compute_cmvn_stats.sh data/eval2000 exp/make_${feattype}/eval2000 $feattype || exit 1;
  utils/fix_data_dir.sh data/eval2000 || exit;

  # Use the first 4k sentences as dev set, around 5 hours
  utils/subset_data_dir.sh --first data/train 4000 data/train_dev
  n=$[`cat data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last data/train $n data/train_nodev

  # Create a smaller training set by selecting the first 100k utterances, around 110 hours
  #utils/subset_data_dir.sh --first data/train_nodev 100000 data/train_100k
  #local/remove_dup_utts.sh 200 data/train_100k data/train_100k_nodup

  # Finally the full training set, around 286 hours
  local/remove_dup_utts.sh 300 data/train_nodev data/train_nodup
fi

# global used by stages 3, 4, 5, 6
dir=exp/train_phn_${feattype}_l${am_nlayer}_c${am_ncell_dim}_m${am_model}_w${am_window}_n${am_norm}_p${am_nproj}_ip${am_ninitproj}_fp${am_ninitproj}

if [ $stage -le 3 ]; then
  echo =====================================================================
  echo "                Setting up local labels/decode graph               "
  echo =====================================================================

  mkdir -p $dir

  ## TO DO: move this under data prep
  echo generating train labels...

  # note: use of --ignore_noises means that we need to syncronize lexicon
  mkdir -p ./data/local/dict_phn_nonoise

  python ./local/swbd1_prepare_phn_dict_tf.py --phn_lexicon ./data/local/dict_phn/lexicon.txt --text_file ./data/train_nodup/text --output_units ./data/local/dict_phn_nonoise/units.txt --output_labels $dir/labels.tr --ignore_noises || exit 1

  egrep -v ' (spn|nsn|lau)' ./data/local/dict_phn/lexicon.txt > ./data/local/dict_phn_nonoise/lexicon.txt
  utils/sym2int.pl -f 2- data/local/dict_phn_nonoise/units.txt < data/local/dict_phn_nonoise/lexicon.txt > data/local/dict_phn_nonoise/lexicon_numbers.txt
  utils/ctc_compile_dict_token.sh data/local/dict_phn_nonoise data/local/lang_phn_tmp data/lang_phn
  local/swbd1_train_lms.sh data/local/train/text data/local/dict_phn/lexicon.txt data/local/lm $fisher_dirs
  local/swbd1_decode_graph_tf.sh data/lang_phn data/local/dict_phn_nonoise/lexicon.txt

  echo generating cv labels...

  python ./local/swbd1_prepare_phn_dict_tf.py --phn_lexicon ./data/local/dict_phn/lexicon.txt --text_file ./data/train_dev/text --output_units ./data/local/dict_phn_nonoise/units.txt --output_labels $dir/labels.cv --ignore_noises || exit 1

  echo generating test labels... [only for TER calculation on test set]

  python ./local/swbd1_prepare_phn_dict_tf.py --phn_lexicon ./data/local/dict_phn/lexicon.txt --text_file ./data/eval2000/text --output_units ./data/local/dict_phn_nonoise/units.txt --output_labels ./data/eval2000/label_phn.test --ignore_noises || exit 1

fi

if [ $stage -le 4 ]; then
  echo =====================================================================
  echo "                      Training CTC network               "
  echo =====================================================================

  # Train the network with CTC. Refer to the script for details about the arguments
  ( 
  steps/train_ctc_tf.sh --l2 0.001 --batch_norm true --nlayer $am_nlayer --nhidden $am_ncell_dim --lrscheduler newbob --lr_spec $lr_spec --model $am_model  --ninitproj $am_ninitproj --nproj $am_nproj --nfinalproj  $am_nfinalproj data/train_nodup data/train_dev $dir 2>&1 || exit 1 
  ) | tee $dir/train.log 

fi

# globals used by stages 5

epoch=final.ckpt
filename=$(basename "$epoch")
name_exp="${filename%.*}"

data=./data/eval2000/
weights=$dir/model/$epoch
config=$dir/model/config.pkl
results=$dir/results_${name_exp}


if [ $stage -le 5 ]; then

  echo =====================================================================
  echo "            Decoding eval2000 using AM + WFST decoder                     "
  echo =====================================================================

  for lm_suffix in sw1_tg sw1_fsh_tgpr; do
      for bs in 4.0 5.0 6.0 7.0; do
	  ./steps/decode_ctc_lat_tf.sh \
	      --model $weights \
	      --nj 8 \
	      --blank_scale $bs \
	      ./data/lang_phn_${lm_suffix} \
	      ${data} \
	      ${results}_bs${bs}_${lm_suffix}
	  done
  done
fi






