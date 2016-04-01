#!/bin/bash

#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -V
#PBS -l nodes=1:ppn=8

. ./path.sh
. ./cmd.sh

[ -f ./local.sh ] && . ./local.sh

. ./utils/parse_options.sh

    # Specify network structure and generate the network topology
    input_feat_dim=`feat-to-dim scp:data/dev/feats.scp  ark,t:|awk 'NR==1 { print 3*$2 }'`
    # dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
    lstm_layer_num=5     # number of LSTM layers
    lstm_cell_dim=320    # number of memory cells in every LSTM layer

    dir=exp/train_phn_l${lstm_layer_num}_c${lstm_cell_dim}
    tst=$dir/decode_dev_v1

    for set in dev test; do
        tst=$dir/decode_${set}_v1
	steps/decode_ctc_lat.sh --cmd "$decode_cmd" --nj 8 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.6 \
	    data/lang_phn_test data/$set $tst

        [ -f data/$set/glm ] || (cd data/$set; ln -s /data/ASR4/babel/cmilo/sctk-2.4.8/src/test_suite/example.glm glm)
        local/score_sclite.sh --cmd run.pl --min_acwt 1 --max_acwt 9  data/$set data/lang_phn_test $tst
    done;
