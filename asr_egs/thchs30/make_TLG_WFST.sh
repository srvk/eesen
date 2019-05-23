#!/bin/bash
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

. parse_options.sh
H=`pwd`
corpus_dir=$H/corpus

echo =====================================================================
echo "                       TLG WFST Construction                       "
echo =====================================================================
#Data preparation
local/thchs-30_data_prep.sh $H $corpus_dir

# Construct the phoneme-based dict.
# We get 216 tokens, representing phonemes with tonality.
local/thchs-30_prepare_phn_dict.sh || exit 1;
# Compile the lexicon and token FSTs
utils/ctc_compile_dict_token.sh --dict-type "phn" data/dict_phn data/lang_tmp data/lang || exit 1;

# Train and compile LMs. 
#local/thchs-30_train_lms.sh corpus/train/text data/dict_phn/lexicon.txt data/language_model || exit 1;

# Compile the language-model FST and the final decoding graph TLG.fst
local/thchs-30_decode_graph.sh data/language_model data/lang data/search_Graph || exit 1;
