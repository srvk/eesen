README
--

Run the code using run_rnnlm.sh by setting the hyper-parameters. 
It requires 4 files - 
1) train.txt - eg. 105-turkish-flp/data/train/text
2) dev.txt - eg. 105-turkish-flp/data/cv_05/text
3) units.txt - eg. 105-turkish-flp/data/local/dict_phn/units.txt
4) lexicon_numbers.txt - eg. 105-turkish-flp/data/local/dict_phn/lexicon_numbers.txt

#DEFAULT PARAMETERS
batch_size=16
emb_size=64
hidden_size=1000
num_layers=1
drop_emb=1.0

more details in /data/ASR5/sdalmia_1/rnnlm

If using a different dataset change prepare_data
