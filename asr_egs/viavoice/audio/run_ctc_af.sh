### for CMU rocks cluster ###
#PBS -M ramon.sanabria.teixidor@gmail.com
#PBS -q gpu
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -N eesen_tf_swbd_pipeline
#PBS -V
#PBS -l walltime=48:00:00
#PBS -l nodes=1:ppn=1


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

. parse_options.sh


#acoustic model parameters
am_nlayer=4
am_ncell_dim=320
am_model=deepbilstm
am_window=3
am_norm=false


if [ $stage -le 4 ]; then
  echo =====================================================================
  echo "                Training AM with the Full Set                      "
  echo =====================================================================


  dir=exp/train_af_l${am_nlayer}_c${am_ncell_dim}_m${am_model}_w${am_window}_n${am_norm}

  mkdir -p $dir

  echo generating train af...

  python ./local/viavoice_prepare_char_dict_af.py --original_units data/lang_phn/units.txt --original_lexicon data/lang_phn/lexicon_numbers.txt  --af_folder ./data/dict_af/ --af_folder ./data/dict_af/ || exit 1

  echo generating cv labels...

  python ./local/viavoice_prepare_labels_from_lexicon.py --lexicon data/lang_phn/lexicon_con_disct_phn_to_arcticulator.txt  --text_file --ignore_noises
  
  python ./local/viavoice_prepare_labels_from_lexicon.py --lexicon data/lang_phn/lexicon_con_disct_phn_to_arcticulator.txt  --text_file --ignore_noises
  
  python ./local/viavoice_prepare_labels_from_lexicon.py --lexicon data/lang_phn/lexicon_con_disct_phn_to_arcticulator.txt  --text_file --ignore_noises

  python ./local/viavoice_prepare_labels_from_lexicon.py --lexicon data/lang_phn/lexicon_con_disct_phn_to_arcticulator.txt  --text_file --ignore_noises


  echo generating train labels

  # Train the network with CTC. Refer to the script for details about the arguments
  steps/train_ctc_tf.sh --nlayer $am_nlayer --nhidden $am_ncell_dim  --batch_size 16 --learn_rate 0.02 --half_after 6 --model $am_model --window $am_window --debug true --norm $am_norm data/train_nodup data/train_dev $dir || exit 1;


fi

