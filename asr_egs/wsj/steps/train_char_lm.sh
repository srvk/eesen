## Begin configuration section
train_tool="python -m lm_train" # the command for training; by default, we use the

train_dir=
units_file=

tmpdir=`mktemp -d`
trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT ERR


# network size
nembed=
nhidden=
nlayer=

learn_rate=0.02
batch_size=16

max_iters=25

do_shuf=true

. utils/parse_options.sh || exit 1;

cp $train_dir/labels.tr $tmpdir/labels.tr
cp $train_dir/labels.cv $tmpdir/labels.cv


$train_tool --lr_rate $learn_rate --batch_size $batch_size \
    --nhidden $nhidden --nlayer $nlayer --nembed $nembed --nepoch $max_iters \
    --data_dir $tmpdir --train_dir $train_dir --optimizer Adam \
    --units_file $units_file || exit 1;
