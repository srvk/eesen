### for CMU rocks cluster ###
#PBS -M ramon.sanabria.teixidor@gmail.com
#PBS -q gpu
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -N viavoice_af_multitask_mt
#PBS -V
#PBS -l walltime=48:00:00
#PBS -l nodes=1:ppn=1


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

. parse_options.sh



stage=2

#acoustic model parameters
am_nlayer=4
am_ncell_dim=320
am_model=deepbilstm
am_window=3
am_norm=false

if [ $stage -le 2 ]; then
  echo =====================================================================
  echo "                    FBank Feature Generation                       "
  echo =====================================================================
  fbankdir=fbank_pitch_11

  which sox
  which sox
  which sox
  which sox
  which sox
  which sox

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  for set in train_11 test_11; do
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --paste-length-tolerance 10000  data/$set exp/make_fbank_pitch_f11/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank_11/$set $fbankdir || exit 1;
  done

  # Split the whole training data into training (95%) and cross-validation (5%) sets
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train data/train_tr95_f11 data/train_cv05_f11 || exit 1
fi

exit

if [ $stage -le 4 ]; then
  echo =====================================================================
  echo "                       Viavoice Training AM                       "
  echo =====================================================================


  dir=exp/train_af_l${am_nlayer}_c${am_ncell_dim}_m${am_model}_w${am_window}_n${am_norm}_mt

  mkdir -p $dir

  echo ""
  echo generating train labels articulators...

  python ./local/viavoice_trans_phn_af_dict_tf.py --dict_af ./data/dict_af/dict_con_disct_phn_to_arcticulator.txt --lexicon_phn data/local/dict_phn/lexicon.txt --ignore_noises --input_text ./data/train_tr95/text --output_units ./data/dict_af/units_articulators.txt --output_labels $dir/labels_articulators.tr || exit 1

  echo generating cv labels articulators...

  python ./local/viavoice_trans_phn_af_dict_tf.py --dict_af ./data/dict_af/dict_con_disct_phn_to_arcticulator.txt --lexicon_phn data/local/dict_phn/lexicon.txt --ignore_noises --input_text ./data/train_cv05/text --output_units ./data/dict_af/units_articulators.txt --output_labels $dir/labels_articulators.cv || exit 1

  echo ""
  echo generating train labels articulatory location...

  python ./local/viavoice_trans_phn_af_dict_tf.py --dict_af ./data/dict_af/dict_con_disct_phn_to_articulatory_location.txt --lexicon_phn data/local/dict_phn/lexicon.txt --ignore_noises --input_text ./data/train_tr95/text --output_units ./data/dict_af/units_articulatory_location.txt --output_labels $dir/labels_articulatory_location.tr || exit 1

  echo generating cv labels articulatory location...

  python ./local/viavoice_trans_phn_af_dict_tf.py --dict_af ./data/dict_af/dict_con_disct_phn_to_articulatory_location.txt --lexicon_phn data/local/dict_phn/lexicon.txt --ignore_noises --input_text ./data/train_cv05/text --output_units ./data/dict_af/units_articulatory_location.txt --output_labels $dir/labels_articulatory_location.cv || exit 1


  echo ""
  echo generating train labels voiced/voicedless...

  python ./local/viavoice_trans_phn_af_dict_tf.py --dict_af ./data/dict_af/dict_con_disct_phn_to_voiced_voiceless.txt --lexicon_phn data/local/dict_phn/lexicon.txt --ignore_noises --input_text ./data/train_tr95/text --output_units ./data/dict_af/units_voiced_voiceless.txt --output_labels $dir/labels_voiced_voiceless.tr || exit 1

  echo generating cv labels voiced/voicedless...

  python ./local/viavoice_trans_phn_af_dict_tf.py --dict_af ./data/dict_af/dict_con_disct_phn_to_voiced_voiceless.txt --lexicon_phn data/local/dict_phn/lexicon.txt --ignore_noises --input_text ./data/train_cv05/text --output_units ./data/dict_af/units_voiced_voiceless.txt --output_labels $dir/labels_voiced_voiceless.cv || exit 1

  echo ""
  echo generating train labels mouth opening...

  python ./local/viavoice_trans_phn_af_dict_tf.py --dict_af ./data/dict_af/dict_voc_disct_phn_to_mouth_opening.txt --lexicon_phn data/local/dict_phn/lexicon.txt --ignore_noises --input_text ./data/train_tr95/text --output_units ./data/dict_af/units_mouth_opening.txt --output_labels $dir/labels_mouth_opening.tr || exit 1

  echo generating cv labels mouth opening...

  python ./local/viavoice_trans_phn_af_dict_tf.py --dict_af ./data/dict_af/dict_voc_disct_phn_to_mouth_opening.txt --lexicon_phn data/local/dict_phn/lexicon.txt --ignore_noises --input_text ./data/train_cv05/text --output_units ./data/dict_af/units_mouth_opening.txt --output_labels $dir/labels_mouth_opening.cv || exit 1

  echo ""
  echo generating train labels mouth rounding...

  python ./local/viavoice_trans_phn_af_dict_tf.py --dict_af ./data/dict_af/dict_voc_disct_phn_to_rounding.txt --lexicon_phn data/local/dict_phn/lexicon.txt --ignore_noises --input_text ./data/train_tr95/text --output_units ./data/dict_af/units_mouth_rounding.txt --output_labels $dir/labels_mouth_rounding.tr || exit 1

  echo generating cv labels mouth rounding...

  python ./local/viavoice_trans_phn_af_dict_tf.py --dict_af ./data/dict_af/dict_voc_disct_phn_to_rounding.txt --lexicon_phn data/local/dict_phn/lexicon.txt --ignore_noises --input_text ./data/train_cv05/text --output_units ./data/dict_af/units_mouth_rounding.txt --output_labels $dir/labels_mouth_rounding.cv || exit 1

  echo ""
  echo generating train labels tong position...

  python ./local/viavoice_trans_phn_af_dict_tf.py --dict_af ./data/dict_af/dict_voc_disct_phn_to_tong_position.txt --lexicon_phn data/local/dict_phn/lexicon.txt --ignore_noises --input_text ./data/train_tr95/text --output_units ./data/dict_af/units_tong_position.txt --output_labels $dir/labels_tong_position.tr || exit 1

  echo generating cv labels tong position...

  python ./local/viavoice_trans_phn_af_dict_tf.py --dict_af ./data/dict_af/dict_voc_disct_phn_to_tong_position.txt --lexicon_phn data/local/dict_phn/lexicon.txt --ignore_noises --input_text ./data/train_cv05/text --output_units ./data/dict_af/units_tong_position.txt --output_labels $dir/labels_tong_position.cv || exit 1

  echo generating train labels

  # Train the network with CTC. Refer to the script for details about the arguments
  steps/train_ctc_tf.sh --nlayer $am_nlayer --nhidden $am_ncell_dim  --batch_size 16 --learn_rate 0.02 --half_after 6 --model $am_model --window $am_window  --norm $am_norm data/train_tr95/ data/train_cv05/ $dir || exit 1;


fi

