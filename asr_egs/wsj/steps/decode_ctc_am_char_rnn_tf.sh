#main calls and arguments
train_tool="python -m lm_test.py"

[ -f path.sh ] && . ./path.sh;

lm_weights=
lm_config=
ctc_probs=
blank_scaling=
results_filename=
units_file=
lexicon_file=
beam_size=
insertion_bonus=
decoding_strategy=
lm_weights_ckpt=
lm_weight=

. utils/parse_options.sh || exit 1;


lm_weights_ckpt="--lm_weights_ckpt $lm_weights_ckpt"
lm_weight="--lm_weight $lm_weight"
lm_config="--lm_config $lm_config"

results_filename="--results_filename $results_filename"
units_file="--units_file $units_file"
lexicon_file="--lexicon_file $lexicon_file"
beam_size="--beam_size $beam_size"
insertion_bonus="--insertion_bonus $insertion_bonus"
decoding_strategy="--decoding_strategy $decoding_strategy"

blank_scaling="--blank_scaling $decoding_strategy"


tmpdir=`mktemp -d `

trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT ERR

copy-feats scp:$ctc_probs ark,scp:$tmpdir/f.ark,$tmpdir/test_local.scp

#source activate tensorflow_cpu

python -m lm_test $results_filename $units_file --ctc_probs_scp $tmpdir/test_local.scp $lexicon_list $insertion_bonus $decoding_strategy $lm_config $lexicon_file $lm_weights_ckpt $lm_weight --batch_size 1



