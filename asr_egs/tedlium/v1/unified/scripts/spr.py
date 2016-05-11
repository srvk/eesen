from os import listdir
from os.path import isfile, join
import re
import sys

onlywavs = [ f for f in sorted(listdir("./")) if f.endswith('.wav') ]
oovlist = []
# Holds a list of speaking durations (one per wav file)
wavdurlist = []
lex_map = {}
# List of words and their phonemes
with open("lexicon.txt") as lf:
  for line in lf:
    # for example:  "phoneme F OW N IY M"
    array = line.split()
    # number of phonemes
    lex_map[array[0]] = len(array) -1

# Match lines in feat.txt, e.g:
# AimeeMullins_2009P.wav  1318.22 1318.22 0  1318.22 4 65.81 209.21   1280.46  2595.22  4018.85 6 5 -0.6421
# captures speaking duration (the third string, the second number)
regex2 = re.compile(".*\s\s\d+\.\d+\s(\d+\.\d+)\s.*")
for wav in onlywavs:
  #print "wav: ", wav
  with open("feat.txt") as ff:
    for line in ff:
      #print "line: ", line
      result2 = regex2.search(line)
      if result2:
        wavdurlist.append(float(result2.group(1)))

idx = 0
for wav in onlywavs:
  wav = wav.replace(".wav","")
  regex = re.compile(wav+"-\d*-\d*\s*(.*)")  
  # Match lines from test text that begin with, e.g: 
  # TomWujec_2010U-0033857-0034512
  # capture the (sentence-ish) text that follows each line
  with open(sys.argv[1]) as fx:
    count = 0
    # count all sentences for this speaker (wav)
    for line in fx:
      result = regex.search(line)
      if result:
        for word in result.group(1).strip().split():
          if word in lex_map:
            # count the phonemes in the word
            count += lex_map[word]
          else:
            oovlist.append(word)
    print "{0:.2f}".format(count/wavdurlist[idx])
  idx += 1
