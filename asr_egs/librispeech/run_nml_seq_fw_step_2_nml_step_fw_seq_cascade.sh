# Adapted from Kaldi librispeech and Eesen WSJ recipes by Jayadev Billa (2017)

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
[ -f path.sh ] && . ./path.sh;

stage=4
exp=nml_seq_fw_step_2_nml_step_fw_seq_cascade
exp_base=./exp/$exp # ./exp dir must be created before running.
data=$exp_base/data
lm_data=$data/lm #data/local/lm
lm_tmp=$exp_base/lm_tmp
dict_dir=$exp_base/dict #data/local/dict
lang_dir=$exp_base/lang #data/lang
feats_tmpdir=./tmp # this should ideally be a tmp dir local to the machine.
train_dir=$exp_base/train_lstm   # working directory

dict_name=librispeech_phn_reduced_dict.txt
dict_type="char"
frame_limit=25000
train_seq_parallel=10

## In a cascade approach, we initially train the network with a particular
## set of dropout parameters and then use the resulting model as an initial
## model for training with a different of parameters.
init_iter=206
half_after_epoch=522
aug_iter=522
max_iter=550

fb_conf=$exp_base/fbconf


# create directories and copy relevant files
mkdir -p $exp_base/{data,lm,lm_tmp,dict,lang}
cp config/$dict_name $exp_base
cp config/fbconf-{8,10,11} $exp_base

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

echo =====================================================================
echo "Started run @ ", `date`
echo =====================================================================

if [ $stage -le 1 ]; then
  echo =====================================================================
  echo "             Data Preparation                                      "
  echo =====================================================================

  # download the 100hr training data and test sets.
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
      local/download_and_untar.sh $data $data_url $part || exit 1;
  done

  # download the LM resources
  local/download_lm.sh $lm_url $lm_data || exit 1;

  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
      # use underscore-separated names in data directories.
      local/data_prep.sh $data/LibriSpeech/$part $exp_base/$(echo $part | sed s/-/_/g) || exit 1;
  done
fi

if [ $stage -le 2 ]; then
  echo =====================================================================
  echo "                 Prepare dictionary and FST                    "
  echo =====================================================================

  ## See Kaldi librispeech recipe for additional information/context

  # Normally dict is in $lm_data but for this sequence of experiments well provide the dict
  # in $exp_base
  local/ls_prepare_phoneme_dict.sh $exp_base $dict_dir $dict_name || exit 1;

  # Compile the lexicon and token FSTs
  # usage: utils/ctc_compile_dict_token.sh <dict-src-dir> <tmp-dir> <lang-dir>"
  utils/ctc_compile_dict_token.sh --dict-type $dict_type --space-char "<SPACE>" \
    $dict_dir $lang_dir/tmp $lang_dir || exit 1;

  # Compile the language-model FST and the final decoding graph TLG.fst
  local/ls_decode_graph.sh $lang_dir $lm_data $lm_tmp/tmp || exit 1;
fi

if [ $stage -le 3 ]; then
  echo =====================================================================
  echo "                    FBank Feature Generation                       "
  echo =====================================================================
  # Split the whole training data into training (95%) and cross-validation (5%) sets
  # utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train_si284 data/train_tr95 data/train_cv05 || exit 1
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 $exp_base/train_clean_100  $exp_base/train_tr95 $exp_base/train_cv05 || exit 1

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  fbankdir=fbank

  # Only apply different transforms to train
  set=train_tr95
  for vtlnw in 0.8 1.0 1.2; do
  	for spkrate in 8 10 11; do
  	  steps/make_fbank_mult.sh --vtln true --vtln-warp $vtlnw --tag ${spkrate}_$vtlnw \
                   --cmd "$train_cmd" --nj 14 --fbank-config ${fb_conf}-${spkrate} $exp_base/$set $exp_base/make_fbank/$set $exp_base/$fbankdir || exit 1;
    done
  done
  utils/fix_data_dir.sh $exp_base/$set || exit;
  steps/compute_cmvn_stats_mult.sh --tag 10_1.0 $exp_base/$set $exp_base/make_fbank/$set $exp_base/$fbankdir || exit 1;

  # Use standard feature extraction for test sets.
  # A simple extension would be to apply the different transforms to test and combine with ROVER.
  # this should provide an improvement albeit with need for multiple decodings.
  set=train_cv05
  steps/make_fbank.sh --cmd "$train_cmd" --nj 14 --fbank-config ${fb_conf}-10 $exp_base/$set $exp_base/make_fbank/$set $exp_base/$fbankdir || exit 1;
  utils/fix_data_dir.sh $exp_base/$set || exit;
  steps/compute_cmvn_stats.sh $exp_base/$set $exp_base/make_fbank/$set $exp_base/$fbankdir || exit 1;

  for set in  dev_clean test_clean dev_other test_other ; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 14 --fbank-config ${fb_conf}-10  $exp_base/$set $exp_base/make_fbank/$set $exp_base/$fbankdir || exit 1;
    utils/fix_data_dir.sh $exp_base/$set || exit;
    steps/compute_cmvn_stats.sh $exp_base/$set $exp_base/make_fbank/$set $exp_base/$fbankdir || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  echo =====================================================================
  echo "                        Network Training                           "
  echo =====================================================================

  # setup directories
  mkdir -p $train_dir/nnet


  # Copy network topology to nnet.proto
  cp config/nnet.proto.$exp $exp_base/
  cp config/nnet.proto.$exp  $train_dir/nnet.proto

  # Label sequences; simply convert words into their label indices
  utils/prep_ctc_trans.py $lang_dir/lexicon_numbers.txt \
    $exp_base/train_tr95/text "<UNK>" "<SPACE>" | gzip -c - > $train_dir/labels.tr.gz
  utils/prep_ctc_trans.py $lang_dir/lexicon_numbers.txt \
    $exp_base/train_cv05/text "<UNK>" "<SPACE>" | gzip -c - > $train_dir/labels.cv.gz

  # Train the network with CTC. Refer to the script for details about the arguments
  # train the starter model
  steps/train_ctc_parallel_mult.sh --tags "10_1.0 8_1.0 11_1.0 8_0.8 10_1.2 10_1.0 11_0.8 8_1.2 10_0.8 11_1.2" \
    --skip true --splice true  --splice-opts "--left-context=1 --right-context=1" --skip-frames 3 --skip-offset 1 \
    --add-deltas true --num-sequence $train_seq_parallel --halving-after-epoch $half_after_epoch --frame-num-limit $frame_limit \
    --feats-tmpdir $feats_tmpdir --max-iters $init_iter --learn-rate 0.00004 --report-step 1000 --write-final false \
    $exp_base/train_tr95 $exp_base/train_cv05 $train_dir || exit 1;

  # Initialize model with revised parameters from starter model.
  mv  $train_dir/nnet/nnet.iter$init_iter  $train_dir/nnet/nnet.iter${init_iter}_starter
  net-change-model --forwarddrop=0.2 --forwardseq=true --nmldrop=true --recurrentdrop=0.2 --recurrentstep=true \
                $train_dir/nnet/nnet.iter${init_iter}_starter $train_dir/nnet/nnet.iter$init_iter 

  # Train the final model with data augmentation
  steps/train_ctc_parallel_mult.sh --tags "10_1.0 8_1.0 11_1.0 8_0.8 10_1.2 10_1.0 11_0.8 8_1.2 10_0.8 11_1.2" \
    --skip true --splice true  --splice-opts "--left-context=1 --right-context=1" --skip-frames 3 --skip-offset 1 \
    --add-deltas true --num-sequence $train_seq_parallel --halving-after-epoch $half_after_epoch --frame-num-limit $frame_limit \
    --feats-tmpdir $feats_tmpdir --max-iters $aug_iter --learn-rate 0.00004 --report-step 1000 --write-final false \
    $exp_base/train_tr95 $exp_base/train_cv05 $train_dir || exit 1;

  # for halfing just use the regular features 10_1.0
  steps/train_ctc_parallel_mult.sh --tags "10_1.0" \
    --skip true --splice true  --splice-opts "--left-context=1 --right-context=1" --skip-frames 3 --skip-offset 1 \
    --add-deltas true --num-sequence $train_seq_parallel --halving-after-epoch $half_after_epoch --frame-num-limit $frame_limit \
    --feats-tmpdir $feats_tmpdir --max-iters $max_iter --learn-rate 0.00004 --report-step 1000 \
    $exp_base/train_tr95 $exp_base/train_cv05 $train_dir || exit 1;
fi

if [ $stage -le 5 ]; then
  echo =====================================================================
  echo "                            Decoding                               "
  echo =====================================================================

  # Decoding with the librispeech dict
  for test in test_clean test_other dev_clean dev_other; do
      for lm_suffix in tgsmall tgmed; do
          steps/decode_ctc_lat_splicefeat.sh --cmd "$decode_cmd" --nj 10 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.9 \
                  --skip true --splice true --splice-opts "--left-context=1 --right-context=1" --skip-frames 3 --skip-offset 1 \
                                  ${lang_dir}_test_${lm_suffix} $exp_base/$test $train_dir/decode_${test}_${lm_suffix} || exit 1;
      done
  done

fi
