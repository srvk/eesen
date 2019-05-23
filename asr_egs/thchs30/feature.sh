#!/bin/bash
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

. parse_options.sh

echo =====================================================================
echo "                    FBank Feature Generation                       "
echo =====================================================================
fbankdir=fbank

# Generate the fbank features; by default 40-dimensional fbanks on each frame
#make train fbank
steps/make_fbank.sh --cmd "$train_cmd" --nj 32 data/train exp/make_fbank/train $fbankdir || exit 1;
utils/fix_data_dir.sh data/train || exit;
steps/compute_cmvn_stats.sh data/train exp/make_fbank/train $fbankdir || exit 1;
echo -e "\n"

#make test fbank
steps/make_fbank.sh --cmd "$train_cmd" --nj 10 data/test exp/make_fbank/test $fbankdir || exit 1;
utils/fix_data_dir.sh data/test || exit;
steps/compute_cmvn_stats.sh data/test exp/make_fbank/test $fbankdir || exit 1;
echo -e "\n"

#make dev fbank
steps/make_fbank.sh --cmd "$train_cmd" --nj 10 data/dev exp/make_fbank/dev $fbankdir || exit 1;
utils/fix_data_dir.sh data/dev || exit;
steps/compute_cmvn_stats.sh data/dev exp/make_fbank/dev $fbankdir || exit 1;
echo -e "\n"
