### for CMU rocks cluster ###
#PBS -M ramon.sanabria.teixidor@gmail.com
#PBS -q gpu
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -N how_to_am_480h2_l5_c200_p0
#PBS -V
#PBS -l walltime=48:00:00
#PBS -l nodes=1:ppn=1


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

stage=3

fisher_dirs="/path/to/LDC2004T19/fe_03_p1_tran/ /path/to/LDC2005T19/fe_03_p2_tran/" # Set to "" if you don't have the fisher corpus
eval2000_dirs="/path/to/LDC2002S09/hub5e_00 /path/to/LDC2002T43"

# CMU Rocks
swbd=/data/ASR4/babel/ymiao/CTS/LDC97S62

#TODO fisher datat is currentlu hardcoded (need to deal with that)
fisher_dir_a="/data/ASR5/babel/ymiao/Install/LDC/LDC2004T19/fe_03_p1_tran/"
fisher_dir_b="/data/ASR5/babel/ymiao/Install/LDC/LDC2005T19/fe_03_p2_tran/"


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

if [ $stage -le 3 ]; then
  echo =====================================================================
  echo "                Network Training with the 90-Hour Set             "
  echo =====================================================================
  # Specify network structure and generate the network topology
  nlayer=5                # number of layers
  nhidden=200              # cells in each layer and direction
  nproj=0

  dir=exp/train_am_char_480h2_l${nlayer}_c${nhidden}_p${nproj}

  mkdir -p $dir

  echo generating train labels...

  #python ./local/swbd1_prepare_dicts_tf.py --text_file ./data/train_tr95/text --input_units ./data/local/dict_char/units.txt --output_labels $dir/labels.tr --lower_case
  cp /data/ASR5/ramons_2/sinbad_projects/youtube_project/am/eesen_20170714/asr_egs/how_to/no_adapted/exp/train_lm_char_480hl1_c1024_e1024_d0.3_oadam/labels.tr $dir/


  echo generating cv labels...

  cp /data/ASR5/ramons_2/sinbad_projects/youtube_project/am/eesen_20170714/asr_egs/how_to/no_adapted/exp/train_lm_char_480hl1_c1024_e1024_d0.3_oadam/labels.cv $dir/
  #python ./local/swbd1_prepare_dicts_tf.py --text_file ./data/train_cv05/text --input_units ./data/local/dict_char/units.txt --output_labels $dir/labels.cv --lower_case


  # Train the network with CTC. Refer to the script for details about the arguments
  steps/train_ctc_tf.sh --batch_size 16 --learn-rate 0.02 --half_after 6 --nlayer $nlayer --nproj $nproj --nhidden $nhidden --max_iters 25 ./data/train_480h_tr95_2/ ./data/train_480h_cv05/ ./$dir || exit 1;

fi




