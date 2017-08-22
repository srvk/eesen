## Begin configuration section
train_tool="python -m main_rnn" # the command for training; by default, we use the
train_opts="--store_model"

cp $dir/labels.tr $tmpdir/labels.tr
cp $dir/labels.cv $tmpdir/labels.cv


# network size
embed_size=100               
hidden_size=320              
num_layers=0                 

train_file=$tmpdir/labels.tr 
dev_file=$tmpdir/labels.cv 

learn_rate=0.02
batch_size=16

max_iters=25          

do_shuf=true

. utils/parse_options.sh || exit 1;

$train_tool $train_opts --lr_rate $learn_rate --batch_size $num_sequence --l2 $l2 \
    --nhidden $nhidden --nlayer $nlayer --nembed $feat_proj --nepoch $max_iters \
    --train_dir $dir --data_dir $tmpdir || exit 1;
