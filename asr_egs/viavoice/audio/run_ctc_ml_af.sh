### for CMU rocks cluster ###
#PBS -M ramon.sanabria.teixidor@gmail.com
#PBS -q gpu
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -N viavoice_af_multitask_ml
#PBS -V
#PBS -l walltime=48:00:00
#PBS -l nodes=1:ppn=1


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

. parse_options.sh



stage=4
#acoustic model parameters
am_nlayer=4
am_ncell_dim=320
am_model=deepbilstm
am_window=3
am_norm=false

if [ $stage -le 4 ]; then
  echo =====================================================================
  echo "                       Viavoice Training AM                       "
  echo =====================================================================


  dir=exp/train_af_l${am_nlayer}_c${am_ncell_dim}_m${am_model}_w${am_window}_n${am_norm}_ml

  mkdir -p $dir

  all_language=("./data_ml/articulators/" "./data_ml/articulatory_location/" "./data_ml/mouth_opening/" "./data_ml/rounding/" "./data_ml/tong_position/" "./data_ml/voiced_voiceless/")

  echo generating train labels

  steps/train_ctc_tf_ml.sh --nlayer $am_nlayer --nhidden $am_ncell_dim  --batch_size 16 --learn_rate 0.02 --half_after 6 --model $am_model --window $am_window --norm $am_norm "${all_language[@]}" $dir || exit 1;


fi

