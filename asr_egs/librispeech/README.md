
This dir contains fully fledged recipes to build end-to-end ASR systems using
the Librispeech 100hr corpus. These scripts will start by downloading 
the relevant training data and language models, then start/complete training 
and decoding.

There are two recipes, illustrating max perturbation, stochastic and cascade 
dropout combination for a phoneme based system

run_nml_seq_fw_seq_tw.sh  - max perturbation + stochastic dropout combo
 
run_nml_seq_fw_step_2_nml_step_fw_seq_cascade.sh - max perturbation + cascade
                                                       dropout combo

NOTE:

- please create/link exp and tmp directories prior to running scripts.
- these take a *long* time to run.

For dropout and max perturbation, please cite:
- "Improving LSTM-CTC based ASR performance in domains with limited training data", Jayadev Billa (https://arxiv.org/pdf/1707.00722.pdf)
