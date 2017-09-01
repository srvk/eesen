## Begin configuration section
train_tool="python -m lm_train" # the command for training; by default, we use the

train_dir=
import_config=


tmpdir=`mktemp -d`
trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT ERR


# network architecture
nembed=
nhidden=
nlayer=
nepoch=
train_labels=
cv_labels=
continue_ckpt=

#netwrok training
drop_out=
#adam sdg for now
optimizer=


learn_rate=0
max_iters=25
batch_size=
do_shuf=true

#extra things
debug=

#sat
num_sat_layers=
num_sat_dim=
concat_sat=""
fuse_sat=""

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

    concat_sat="--concat_sat"
else
    concat_sat=""
fi

if [ "$fuse_sat" != "" ]; then
    cat $fuse_sat | /data/ASR5/ramons_2/sinbad_projects/eesen/src/featbin/copy-feats ark,t:- ark,scp:$tmpdir/sat_local.ark,$tmpdir/sat_local.scp

    fuse_sat="--fuse_sat"
else
    fuse_sat=""
fi

if [ "$num_sat_layers" != "" ]; then
    num_sat_layers="--num_sat_layers $num_sat_layers"
else
    num_sat_layers=""
fi

if [ "$num_sat_dim" != "" ]; then
    num_sat_dim="--num_sat_dim $num_sat_dim"
else
    num_sat_dim=""
fi



$train_tool --lr_rate $learn_rate --batch_size $batch_size \
    --nhidden $nhidden --nlayer $nlayer --nembed $nembed --nepoch $nepoch \
    --drop_out $drop_out --data_dir $tmpdir $debug --train_dir $train_dir --optimizer $optimizer \
    $continue_ckpt $import_config $concat_sat $num_satLayers $num_sat_dim|| exit 1;
