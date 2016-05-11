from os import listdir
from os.path import isfile, join
import re
import sys

onlywavs = [ f for f in sorted(listdir("./")) if f.endswith('.wav') ]

for wav in onlywavs:
  wav = wav[:wav.index("_")]
  wav = wav.lower()
  # read from the score file *.ctm.filt.filt.raw
  #
  #| SPKR  | # Snt  # Wrd | Corr Sub Del Ins Err  S.Err | NCE    |
  #| Joe_1 |  129    2897 | 2437 388 72  117 577  122   | -0.169 |
  # group             (1)   (2)  (3) (4) (5) 
  regex = re.compile("\s*\|\s*"+wav+".*\s*\|\s*[0-9]*\s*([0-9]*)\s*\|\s*([0-9]*)\s*([0-9]*)\s*([0-9]*)\s*([0-9]*)\s*[0-9]*\s*[0-9]*\s*\|\s*")  
  with open(sys.argv[1]) as fx:
    for line in fx:
      result = regex.search(line)
      if result:
        words = float(result.group(1))
        subs  = float(result.group(3))
        dels  = float(result.group(4))
        ins   = float(result.group(5))
        wer = "{0:.2f}".format(((subs + dels + ins) / words))
        accuracy = "{0:.2f}".format(float(result.group(2))/float(result.group(1)))

        print wer + " " + accuracy + " " + str(ins) + " " + str(dels) + " " + str(subs) + " " + str(words)
        break

