#!/bin/bash
#
# RunMe.sh -  script to produce error analysis spreadsheet features.csv
# for use at http://speechkitchen.org/error-analysis-tools,
# from results of running EESEN-tedlium from 

. ./path.sh

# Globals
# 6 gives the best WER in default EESEN-tedlium
lmweight=6
experiment=train_l4_c320
decode_folder=decode_test_pruned.lm3

exp=`pwd`/../exp/$experiment
testtext=`pwd`/../data/test/text
scorefile=$exp/$decode_folder/score_${lmweight}_0/ctm.filt.filt.raw

# Links
ln -s ../data/local/dict_nosp/lexicon.txt .

# unpack the .sph files in wav.scp as local .wav files
while IFS=" " read -r -a myArray
do
 sph2pipe -f wav -p ${myArray[5]} > ${myArray[0]}.wav
 echo ${myArray[5]}
done < ../data/test/wav.scp

#./sctk-2.4.8/bin/sclite -h hyp.txt -r ref.txt -i rm -o rsum

# creates feat.txt, computing praat feats & appending SNR
# 1. computes praat features
# 2. computes SNR
# 3. appends them together by line into feat.txt
python scripts/unifiedFeat.py

# creates accuracy.txt
# containing accuracy insertion deletion substitution words
python scripts/accuracy.py $scorefile > accuracy.txt

# compute speaking rate
python scripts/spr.py $testtext > spr.txt

paste -d " " feat.txt spr.txt > foo.txt
paste -d " " foo.txt accuracy.txt > features.txt

sed -i '1s/^/filename originalduration speakingduration begintime endtime gradelevel intensity f0 f1 f2 f3 snr spr score accuracy insertion deletion substitution words\n/' features.txt

# always make csv file
#if  [[ $1 = "--csv" ]]; then
  sed 's/[[:space:]]\+/,/g' features.txt > features.csv
#fi

# cleanup
rm -rf ./temp convert.sh *.wav hyp.txt hyp.txt.raw feat.txt feat2.txt spr.txt foo.txt features.txt accuracy.txt ref.txt tmp.ref.* # *.py
