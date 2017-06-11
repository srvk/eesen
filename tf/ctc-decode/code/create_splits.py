import os
import numpy
import kaldi_io
import random
import sys
file = sys.argv[1]
d = { key:mat for key,mat in kaldi_io.read_mat_ark(file)}
kes = d.keys()
random.shuffle(kes)
size = len(d)
spt = int(sys.argv[2])

dd = size/spt
os.makedirs('new_split_'+sys.argv[2])
for i in range(spt):
    ark_file = str(i+1)+'.ark'
    dic = { k:d[k] for k in kes[i*dd:(i+1)*dd] }
    with open('new_split_'+sys.argv[2]+'/'+ark_file,'w') as f:
        for key,mat in dic.iteritems():
            kaldi_io.write_mat(f, mat, key=key)

print 'heee', i
ark_file = str(i+1)+'.ark'
dic = { k:d[k] for k in kes[i*dd:] }
with open('new_split_'+sys.argv[2]+'/'+ark_file,'w') as f:
    for key,mat in dic.iteritems():
        kaldi_io.write_mat(f, mat, key=key)
