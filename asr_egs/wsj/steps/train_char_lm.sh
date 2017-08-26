## Begin configuration section
train_tool="python -m lm_train" # the command for training; by default, we use the

train_dir=
units_file=
import_config=

tmpdir=`mktemp -d`
trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT ERR


# network size
nembed=
nhidden=
nlayer=
nepoch=
train_labels=
cv_labels=
continue_ckpt=

concat_sat=
learn_rate=0
batch_size=0

max_iters=25

do_shuf=true

debug=

. utils/parse_options.sh || exit 1;

cp $train_labels $tmpdir/labels.tr
cp $cv_labels $tmpdir/labels.cv


if [ "$continue_ckpt" != "" ]; then
    continue_ckpt="--continue_ckpt $continue_ckpt"
else
    continue_ckpt=""
fi

if [ "$import_config" != "" ]; then
    import_config="--import_config $import_config"
else
    import_config=""
fi

if [ "$debug" != "" ]; then
    debug=--debug
else
    debug=""
fi

if [ "$concat_sat" != "" ]; then
    cat $concat_sat | /data/ASR5/ramons_2/sinbad_projects/eesen/src/featbin/copy-feats ark,t:- ark,scp:$tmpdir/sat_local.ark,$tmpdir/sat_local.scp

    import_config="--concat_sat"
else
    import_config=""
fi


$train_tool --lr_rate $learn_rate --batch_size $batch_size \
    --nhidden $nhidden --nlayer $nlayer --nembed $nembed --nepoch $nepoch \
    --data_dir $tmpdir $debug --train_dir $train_dir --optimizer Adam \
    --units_file $units_file $continue_ckpt $import_config $concat_sat || exit 1;
