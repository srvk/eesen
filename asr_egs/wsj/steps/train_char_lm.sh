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

learn_rate=0.02
batch_size=16

max_iters=25

do_shuf=true

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

$train_tool --lr_rate $learn_rate --batch_size $batch_size \
    --nhidden $nhidden --nlayer $nlayer --nembed $nembed --nepoch $nepoch \
    --data_dir $tmpdir --train_dir $train_dir --optimizer Adam \
    --units_file $units_file $continue_ckpt $import_config || exit 1;
