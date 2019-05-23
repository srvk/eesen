#!/bin/bash
#Copyright 2016  Tsinghua University (Author: Dong Wang, Xuewei Zhang).  Apache 2.0.

#This script pepares the data directory for thchs30 recipe. 
#It reads the corpus and get wav.scp and transcriptions.

dir=$1
corpus_dir=$2


cd $dir
echo ========================================
echo "            Data Preparation    "
echo ========================================
echo "creating data/{train,dev,test}"
mkdir -p data/{train,dev,test}

#create wav.scp, utt2spk.scp, spk2utt.scp, text
(
for x in train dev test; do
  echo "cleaning data/$x"
  cd $dir/data/$x
  rm -rf wav.scp utt2spk spk2utt word.txt text
  echo "preparing scps and text in data/$x"
  for nn in `find  $corpus_dir/$x/*.wav | sort -u | xargs -i basename {} .wav`; do
      echo $nn $corpus_dir/$x/$nn.wav >> wav.scp
      echo $nn $nn >> utt2spk
      echo $nn $nn >> spk2utt
      echo $nn `sed -n 1p $corpus_dir/data/$nn.wav.trn` >> word.txt
  done 
  cp word.txt text
done
) || exit 1
echo " Data prepration succeeded "
echo -e "\n"
