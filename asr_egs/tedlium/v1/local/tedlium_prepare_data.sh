#!/bin/bash
#
# Copyright  2014 Nickolay V. Shmyrev 
#            2014 Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# To be run from one directory above this script.

data_dir=db/TEDLIUM_release1

. path.sh
. parse_options.sh || exit 1;

export LC_ALL=C

sph2pipe=$EESEN_ROOT/tools/sph2pipe_v2.5/sph2pipe
[ ! -x $sph2pipe ] \
  && echo "Could not execute the sph2pipe program at $sph2pipe" && exit 1;

# Prepare: test, train,
for set in dev test train; do
  dir=data/$set
  mkdir -p $dir

  # Merge transcripts into a single 'stm' file, do some mappings:
  # - <F0_M> -> <o,f0,male> : map dev stm labels to be coherent with train + test,
  # - <F0_F> -> <o,f0,female> : --||--
  # - (2) -> null : remove pronunciation variants in transcripts, keep in dictionary
  # - <sil> -> null : remove marked <sil>, it is modelled implicitly (in kaldi)
  # - (...) -> null : remove utterance names from end-lines of train
  # - it 's -> it's : merge words that contain apostrophe (if compound in dictionary, local/join_suffix.py)
  { # Add STM header, so sclite can prepare the '.lur' file
    echo ';;
;; LABEL "o" "Overall" "Overall results"
;; LABEL "f0" "f0" "Wideband channel"
;; LABEL "f2" "f2" "Telephone channel"
;; LABEL "male" "Male" "Male Talkers"
;; LABEL "female" "Female" "Female Talkers"
;;'
    # Process the STMs
    cat ${data_dir}/$set/stm/*.stm | sort -k1,1 -k2,2 -k4,4n | \
      sed -e 's:<F0_M>:<o,f0,male>:' \
          -e 's:<F0_F>:<o,f0,female>:' \
          -e 's:([0-9])::g' \
          -e 's:<sil>::g' \
          -e 's:([^ ]*)$::' | \
      awk '{ $2 = "A"; print $0; }'
  } | local/join_suffix.py ${data_dir}/TEDLIUM.*.dic > data/$set/stm 

  # Prepare 'text' file
  # - {NOISE} -> [NOISE] : map the tags to match symbols in dictionary
  cat $dir/stm | grep -v -e 'ignore_time_segment_in_scoring' -e ';;' | \
    awk '{ printf ("%s-%07d-%07d", $1, $4*100, $5*100); 
           for (i=7;i<=NF;i++) { printf(" %s", $i); } 
           printf("\n"); 
         }' | tr '{}' '[]' | sort -k1,1 > $dir/text || exit 1

  # Prepare 'segments', 'utt2spk', 'spk2utt'
  cat $dir/text | cut -d" " -f 1 | awk -F"-" '{printf("%s %s %07.2f %07.2f\n", $0, $1, $2/100.0, $3/100.0)}' > $dir/segments
  cat $dir/segments | awk '{print $1, $2}' > $dir/utt2spk
  cat $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt
  
  # Prepare 'wav.scp', 'reco2file_and_channel' 
  cat $dir/spk2utt | awk -v set=$set -v pwd=$PWD -v sph2pipe=$sph2pipe -v d=$data_dir '{ printf("%s %s -f wav -p %s/%s/%s/sph/%s.sph |\n", $1, sph2pipe, pwd, d, set, $1); }' > $dir/wav.scp
  cat $dir/wav.scp | awk '{ print $1, $1, "A"; }' > $dir/reco2file_and_channel
  
  # Create empty 'glm' file
  echo ';; empty.glm
  [FAKE]     =>  %HESITATION     / [ ] __ [ ] ;; hesitation token
  ' > data/$set/glm

  # Check that data dirs are okay!
  utils/validate_data_dir.sh --no-feats $dir || exit 1
done

