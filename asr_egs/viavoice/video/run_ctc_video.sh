### for CMU rocks cluster ###
#PBS -M ramon.sanabria.teixidor@gmail.com
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -N viavoice_video
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
am_model=arc_net
am_window=3
am_norm=false

if [ $stage -le 3 ]; then
  echo =====================================================================
  echo "                       Viavoice Training AM                       "
  echo =====================================================================

   echo creating ./data/train_video_tr95 ...
   mkdir -p ./data/train_video_tr95

   python ./local/vivoice_create_video_feats.py --feats_scp ./data/train_tr95/feats.scp --video_dir ./videos --output ./data/train_video_tr95/feats_video.scp  --video_format avi

   echo creating ./data/train_video_cv05 ...
   mkdir -p ./data/train_video_cv05

   python ./local/vivoice_create_video_feats.py --feats_scp ./data/train_cv05/feats.scp --video_dir ./videos --output ./data/train_video_cv05/feats_video.scp --video_format avi
   exit
fi


if [ $stage -le 4 ]; then
  echo =====================================================================
  echo "                       Viavoice Training AM                       "
  echo =====================================================================

  dir=exp/train_video_af_l${am_nlayer}_c${am_ncell_dim}_m${am_model}_w${am_window}_n${am_norm}_ml

  mkdir -p $dir

  echo generating train labels
  all_language=("./data_video_ml/articulators/")

  steps/train_ctc_tf_video_ml.sh --nlayer $am_nlayer --nhidden $am_ncell_dim  --batch_size 16 --learn_rate 0.02 --half_after 6 --model $am_model --window $am_window --norm $am_norm "${all_language[@]}" $dir || exit 1;


fi

