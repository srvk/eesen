## Begin configuration section
train_tool="python -m main_rnn" # the command for training; by default, we use the
train_opts="--store_model"
train_dir=

. utils/parse_options.sh || exit 1;

tmpdir=`mktemp -d`
trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT ERR

cp $train_dir/labels.tr $tmpdir/labels.tr
cp $train_dir/labels.cv $tmpdir/labels.cv


# network size
nembed=100
nhidden=320
nlayer=2

train_file=$tmpdir/labels.tr
dev_file=$tmpdir/labels.cv

learn_rate=0.02
batch_size=16

max_iters=25

do_shuf=true

. utils/parse_options.sh || exit 1;


$train_tool $train_opts --lr_rate $learn_rate --batch_size $batch_size \
    --nhidden $nhidden --nlayer $nlayer --nembed $nembed --nepoch $max_iters \
    --dev_file $tmpdir/labels.cv --train_file $tmpdir/labels.tr --train_dir $train_dir || exit 1;
