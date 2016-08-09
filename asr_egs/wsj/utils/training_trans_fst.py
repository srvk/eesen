#!/usr/bin/env python

# Apache 2.0

import sys

fread = open(sys.argv[1], 'r')

for entry in fread.readlines():
    entry = entry.replace('\n','').strip()
    fields = entry.split(' ')
    uttid = fields[0]
   
    for n in range(1, len(fields)):
      print str(n-1) + ' ' + str(n) + ' ' + fields[n] + ' ' + fields[n]

    print str(n) + ' ' + '0' + ' ' + '0' + ' ' + '0'  # assume that <eps> is 0 in words.txt    

print '0'

fread.close()
