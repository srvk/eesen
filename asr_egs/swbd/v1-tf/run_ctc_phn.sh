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

stage=0

fisher_dir_a="/PATH/TO/FISHER/TRANSCRIPTS/LDC2004T19/fe_03_p1_tran/"
fisher_dir_b="/PATH/TO/FISHER/TRANSCRIPTS/LDC2005T19/fe_03_p2_tran/"
fisher_dirs="$fisher_dir_a $fisher_dir_b" # Set to "" if you don't have the fisher corpus
eval2000_dirs="/PATH/TO/LDC2002S09/hub5e_00 /PATH/TO/LDC2002T43/2000_hub5_eng_eval_tr"
swbd="/PATH/TO/LDC97S62/swb1"

n_parallel_jobs=4  #set this to something that your filesystem can handle - robust systems can do 10-32

. parse_options.sh


#acoustic model parameters
am_nlayer=4

am_nproj=60
am_ninitproj=80
am_nfinalproj=100

am_ncell_dim=320

am_model=deepbilstm
am_window=3
am_norm=true


#language model parameters

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
  steps/make_fbank.sh --cmd "$train_cmd" --nj $n_parallel_jobs data/train exp/make_fbank/train $fbankdir || exit 1;
  steps/compute_cmvn_stats.sh data/train exp/make_fbank/train $fbankdir || exit 1;
  utils/fix_data_dir.sh data/train || exit;

  steps/make_fbank.sh --cmd "$train_cmd" --nj $n_parallel_jobs data/eval2000 exp/make_fbank/eval2000 $fbankdir || exit 1;
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

# global used by stages 3, 4, 5, 6
dir=exp/train_phn_l${am_nlayer}_c${am_ncell_dim}_m${am_model}_w${am_window}_n${am_norm}_p${am_nproj}_ip${am_ninitproj}_fp${am_ninitproj}

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
  utils/sym2int.pl -f 2- data/local/dict_phn/units.txt < data/local/dict_phn_nonoise/lexicon.txt > data/local/dict_phn_nonoise/lexicon_numbers.txt
  utils/ctc_compile_dict_token.sh data/local/dict_phn_nonoise data/local/lang_phn_tmp data/lang_phn
  local/swbd1_train_lms.sh data/local/train/text data/local/dict_phn/lexicon.txt data/local/lm $fisher_dirs
  local/swbd1_decode_graph_tf.sh data/lang_phn data/local/dict_phn_nonoise/lexicon.txt

  echo generating cv labels...

  python ./local/swbd1_prepare_phn_dict_tf.py --phn_lexicon ./data/local/dict_phn/lexicon.txt --text_file ./data/train_dev/text --output_units ./data/local/dict_phn_nonoise/units.txt --output_labels $dir/labels.cv --ignore_noises || exit 1

fi

if [ $stage -le 4 ]; then
  echo =====================================================================
  echo "                      Training CTC network               "
  echo =====================================================================

  # Train the network with CTC. Refer to the script for details about the arguments
  steps/train_ctc_tf.sh --nlayer $am_nlayer --nhidden $am_ncell_dim --lr_rate 0.005 --model $am_model  --ninitproj $am_ninitproj --nproj $am_nproj --nfinalproj $am_nfinalproj data/train_nodup data/train_dev $dir 2>&1 | tee $dir/train.log  || exit 1;

fi

# globals used by stages 5, 6
# TO DO: pick optimal by looking at Validation logs
epoch=epoch22.ckpt
filename=$(basename "$epoch")
name_exp="${filename%.*}"

data=./data/eval2000/
weights=$dir/model/$epoch
config=$dir/model/config.pkl
results=$dir/results/$name_exp


if [ $stage -le 5 ]; then

  echo =====================================================================
  echo "                   Decoding eval200 using AM                      "
  echo =====================================================================

  ./steps/decode_ctc_am_tf.sh --config $config --data $data --weights $weights --results $results

fi

if [ $stage -le 6 ]; then
  echo =====================================================================
  echo "                   Decoding eval200 using AM+decoder                "
  echo =====================================================================

  for lm_suffix in sw1_tg sw1_fsh_tgpr; do
      #steps/decode_ctc_lat_tf.sh --cmd "$decode_cmd" --nj 20 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.6 --norm-vars=$am_norm \
      #data/lang_phn_${lm_suffix} data/eval2000 $dir/decode_eval2000_${lm_suffix} || exit 1;

      results_lm=${results}/log_soft_prob_${lm_suffix}
      mkdir -p ${results_lm}

    # TO FIX: this hard-codes some stuff that should really be parameters
    # TO FIX: this probably wants to call decode_ctc_lat_tf.sh as above
    #        particularly to parallelize this decoding - would need to decode am in parallel above or split results
      ../../../src/decoderbin/latgen-faster \
	  --max-active=7000 \
	  --max-mem=50000000 \
	  --beam=17.0 \
	  --lattice-beam=8.0 \
	  --acoustic-scale=0.6 \
	  --allow-partial=true \
	  --word-symbol-table=data/lang_phn_${lm_suffix}/words.txt \
	  data/lang_phn_${lm_suffix}/TLG.fst \
	  ark:${results}/log_soft_prob_no_target_name.ark \
	  "ark:|gzip -c > ${results_lm}/lat.1.gz"

    local/score_sclite.sh --min-acwt 1 --max-acwt 10 --acwt-factor 0.6 --cmd ${decode_cmd} data/eval2000 data/lang_phn_${lm_suffix} ${results_lm}

    grep Sum ${results_lm}/score_*/eval2000.ctm.filt.sys | utils/best_wer.sh > ${results_lm}/RESULTS
    grep Sum ${results_lm}/score_*/eval2000.ctm.swbd.filt.sys | utils/best_wer.sh >> ${results_lm}/RESULTS
    grep Sum ${results_lm}/score_*/eval2000.ctm.callhm.filt.sys | utils/best_wer.sh >> ${results_lm}/RESULTS
    cat ${results_lm}/RESULTS

  done

fi






