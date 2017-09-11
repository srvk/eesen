results=""

subsampled_utt=0

batch_size=16

online_storage=false

config=""
#/data/ASR5/ramons_2/sinbad_projects/lorelei/full_amharic_cv_tigrinya/exp/new_full_clean_cv_clean/l6_h200_p100_f0_achen_n_9/model/config.pkl

weights=""
#/data/ASR5/ramons_2/sinbad_projects/lorelei/full_amharic_cv_tigrinya/exp/new_full_clean_cv_clean/l6_h200_p100_f0_achen_n_9/model/epoch11.ckpt


data=""


echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;

. utils/parse_options.sh || exit 1;

if $online_storage ; then

      online_storage="--online_storage"
else
      online_storage=""
fi

if [ "$subsampled_utt" -gt "0" ]; then

      subsampled_utt="--subsampled_utt $subsampled_utt"
else
      subsampled_utt=""
fi

mkdir -p $results

norm_vars=true
compute_ter=false
tmpdir=`mktemp -d `

feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:$data/feats.scp ark:- |"

copy-feats "${feats}" ark,scp:$tmpdir/f.ark,$tmpdir/test_local.scp

if $compute_ter; then
    compute_ter="--compute_ter"
    python ./utils/clean_length.py --scp_in  $tmpdir/test_local.scp --labels $tmpdir/labels.test --subsampling 3 --scp_out $tmpdir/test_local.scp $subsampled_utt_utt
    cp $data/labels.test
else
    force_lr_epoch_ckpt=""
fi


python -m test --data_dir $tmpdir --results_dir $results --train_config $config --trained_weights $weights --batch_size $batch_size --temperature 1 $online_storage

#python /data/ASR5/ramons_2/sinbad_projects/youtube_project/am/eesen_20170714/tf/ctc-am/test.py --data_dir $tmpdir --results_dir $results --train_config $config --trained_weights $weight --batch_size 1 --temperature 1


