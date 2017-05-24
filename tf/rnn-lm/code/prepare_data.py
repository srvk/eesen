import io
from collections import Counter, defaultdict
from itertools import count
import random
import math
import sys
import pdb
import numpy as np

def prep_data(config):

    train_file = config['train_file']
    dev_file = config['dev_file']
    units_file = config['units_file']
    lexicon_file = config['lexicon_file']
    w2i = defaultdict(count(0).next)
    sos = '<s>'
    eos = '<s>'
    space = ' '
    train_fil = io.open(train_file, encoding='utf-8').readlines()
    dev_fil = io.open(dev_file, encoding='utf-8').readlines()
    units = io.open(units_file, encoding='utf-8').readlines()
    w2i['<s>']
    for i in units[8:]:
        w2i[i.strip().split()[0]]
    w2i[eos]
    w2i[space]

    lexicon = io.open(lexicon_file, encoding='utf-8').readlines()
    lexicon_dict = {}
    for i in lexicon:
        lexicon_dict[i.strip().split()[0]] = i.strip().split()[1:]
    train_list = []
    useless_list =['<nsn>','<hes>','<int>','<lgh>','<cgh>','<br>','<spn>','<sil>', '(())','<noise>','<spoken_noise>','<unk>','<breath>','<cough>','<foreign>','<hes>','<int>','<laugh>','<silence>','<unk>','<v-noise>']

    for tr in train_fil:
        temp = list()
        temp.append(w2i['<s>'])
        words = tr.strip().split(' ')[1:]
        twords = list()
        for wo in words:
            if wo.lower() not in useless_list:
                twords.append(wo)
        for w in twords:
            for ch in lexicon_dict[w]:
                temp.append(int(ch)-8)
                if(int(ch)<=8):
                    print('MISTAKE!')
                    sys.exit(1)
            if(w != twords[-1]):
                temp.append(w2i[space])
        temp.append(w2i['<s>'])
        if(len(temp) > 3 ):
            train_list.append(temp)

    dev_list = []
    for tr in dev_fil:
        temp = list()
        temp.append(w2i['<s>'])
        words = tr.strip().split(' ')[1:]
        twords = list()
        for wo in words:
            if wo.lower() not in useless_list:
                twords.append(wo)
        for w in twords:
            for ch in lexicon_dict[w]:
                temp.append(int(ch)-8)
                if(int(ch)<=8):
                    print('MISTAKE!')
                    sys.exit(1)
            if(w != twords[-1]):
                temp.append(w2i[space])
        temp.append(w2i['<s>'])
        if(len(temp)>3):
            dev_list.append(temp)

    train = train_list
    nwords = len(w2i)
    test = dev_list
    S = w2i[eos]
    assert (nwords == len(w2i))
    print('N Words ', nwords)
    train.sort(key=lambda x: len(x), reverse=True)
    test.sort(key=lambda x: len(x), reverse=True)
    return train, test, nwords, S
