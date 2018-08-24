#!/usr/bin/env python

# Copyright 2015       Yajie Miao    (Carnegie Mellon University)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This python script converts the word-based transcripts into label sequences. The labels are
# represented by their indices. 

import sys

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print "Usage: {0} <lexicon_file> <trans_file> <unk_word>".format(sys.argv[0])
        print "e.g., lm_utils/prep_ctc_trans.py data/lang/lexicon_numbers.txt data/train/text <UNK>"
        print "<lexicon_file> - the lexicon file in which entries have been represented by indices"
        print "<trans_file>   - the word-based transcript file"
        print "<unk_word>     - the word which represents OOVs in transcripts"
        exit(1)

    dict_file = sys.argv[1]
    trans_file = sys.argv[2]
    unk_word = sys.argv[3]

    # read the lexicon into a dictionary data structure
    fread = open(dict_file,'r')
    dict = {}
    for line in fread.readlines():
        line = line.replace('\n','')
        splits = line.split(' ')  # assume there are no multiple spaces
        word = splits[0]
        letters = ''
        for n in range(1, len(splits)):
            letters += splits[n] + ' '
        dict[word] = letters.strip()
    fread.close()

    # assume that each line is formatted as "uttid word1 word2 word3 ...", with no multiple spaces appearing
    fread = open(trans_file,'r')
    for line in fread.readlines():
        out_line = ''
        line = line.replace('\n','').strip();
        splits = line.split(' ');
    
        out_line += splits[0] + ' '
        for n in range(1, len(splits)):
            try:
              out_line += dict[splits[n]] + ' '
            except Exception:
              out_line += dict[unk_word] + ' '
        print out_line.strip()
