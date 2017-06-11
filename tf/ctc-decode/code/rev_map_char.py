import numpy
import pdb
import ast
import io
import sys

f = io.open(sys.argv[1],encoding ='utf-8').readlines()
raw_dat = dict()

for fil in f:
    fi = fil.strip().split(' ',1)
    if fi[0] not in raw_dat:
        try:
            temp = ast.literal_eval(fi[1])
            raw_dat[fi[0]] = ''.join(temp)
            # pdb.set_trace()
        except:
            print "Error reading CTM"
            pdb.set_trace()
print len(raw_dat)

f = io.open('./data/dev10h/segments',encoding ='utf-8').readlines()
seg_dic = dict()
for ele in f:
    el = ele.strip().split(' ')
    if el[0] not in seg_dic:
        seg_dic[el[0]] = el[1:]
    else:
        print "Noob"

# it seems we could filter the examples here ...
f = io.open('./data/dev10h/segments',encoding ='utf-8').readlines()
sp = ", ' ', "
ccc = 0
f1 = io.open(sys.argv[2],'w',encoding ='utf-8')
for ele in f:
    elem = ele.strip().split(' ')
    elem = elem[0]
    temp = seg_dic[elem][0] + ' 1'
    start = float(seg_dic[elem][1])+ 0.01
    end = float(seg_dic[elem][2]) - 0.01
    # pdb.set_trace()
    if elem in raw_dat:
        # pdb.set_trace()
        pred = raw_dat[elem].split(' ')
        count = len(pred)
        dur = (end-start)/count
        time = start
        for pr in pred:
            # tt = '[' + pr + ']'
            # pdb.set_trace()
            try:
                f1.write(temp + ' ' + str(round(time,2)) + ' '+ str(round(dur,2)) + ' ' + pr +' 1.00\n')
            except:
                print "Error writing output"
                pdb.set_trace()
            # temp += ' ' + rev_lex_dict[tt]
            time += dur
    else:
        ccc+=1
        f1.write(temp + ' ' + str(round(start,2)) + ' '+ str(round((end-start),2)) + ' <hes> 1.00\n')
print "CCC = ", ccc
