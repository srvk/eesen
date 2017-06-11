#coding: utf-8
from __future__ import print_function
import argparse
import math
import kaldi_io
from fileutils.kaldi import readScp
import numpy as np
from multiprocessing import Pool
import pdb
from searchAlgo import beam_search, greedy_search
from lm_util import lm_util
import sys
from collections import Counter, defaultdict
from itertools import count
import io
import json
from prepare_data import *
from build_tree import *


print("decode_new_tr.py - version information follows:")
try:
    print(sys.version)
    print(tf.__version__)
except:
    print("tf.py: could not get version information for logging")


BLANK = u'<eps>'
BLANK_ID = 0

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Decode Character Distribution of CTC"
    )
    parser.add_argument(
        '--arkFile', default="lol",
        help="path to ctc likelihoods file (.ark)"
    )
    parser.add_argument(
        '--ctmOut', default="lol.ctm",
        help="output file to write output to"
    )
    parser.add_argument(
        '--lmFile', default="/data/ASR5/sdalmia_1/turkish_langmodel/redo_char_saved_models/model9.ckpt",
        help="path to the language model file"
    )
    parser.add_argument(
        '--beamSize', type=int, default=40,
        help="size of the beam in beam search"
    )
    parser.add_argument(
        '--insertionBonus', type=float, default=0.6,
        help="insertion bonus"
    )
    parser.add_argument(
        '--lmWeight', type=float, default=1.0,
        help="language model weight"
    )
    parser.add_argument(
        '--blankFudge', type=float, default=0.5,
        help="blank fudge factor"
    )
    parser.add_argument(
        '--show', type=int, default=0,
        help="show greedy or no"
    )
    parser.add_argument(
        '--nBestOutput', type=int, default=1,
        help="output at most N utterances of the beam"
    )

    return parser


def createConfig(args):
    config = {
        'arkFile' : args.arkFile ,
        'beamSize' : args.beamSize ,
        'insertionBonus' : args.insertionBonus ,
        'lmWeight' : args.lmWeight ,
        'blankFudge' : args.blankFudge ,
        'nBestOutput' : args.nBestOutput ,
        'show' : args.show ,
        'beamSize' : args.beamSize ,
        'lmFile' : args.lmFile ,
        'ctmOut' : args.ctmOut ,
    }

    lmFile_config_path = config['lmFile'].rsplit('/',1)[0] + '/../config'
    lm_config = json.load(open(lmFile_config_path))
    config['lm_config'] = lm_config

    print(config)
    return config




if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    config = createConfig(args)
    config, lex_dict = prep_data(config)

    #print(config)
    # pdb.set_trace()
    # util_model = lm_util(config)
    lm_function = lm_util(config).get_prob
    w2i = config['w2i']

    # hacky

    expansion_characters = [u' ']

    # create map from integer to char
    id_to_char = {BLANK_ID: BLANK}
    with io.open(config['lm_config']['units_file'], encoding='utf-8') as filename:
        for line in filename:
            # lin = line.decode('UTF-8')
            l = line.split()
            assert len(l) == 2
            # if len(l[0]) > 1:
            #     print("WARN: ignoring {}".format(l[0]))
            #     continue
            id_to_char[int(l[1])] = l[0]
    # invert dict
    char_to_id = {v: k for k, v in id_to_char.items()}
    print(id_to_char)

    kk = w2i.keys()
    candidates = list()
    for k in kk:
        if k != ' ' and k!='<s>' and k not in expansion_characters:
            candidates.append(char_to_id[k])
    candidates.append(0)
    candidates.sort()
    ch_to_id = {}

    for i in range(len(candidates)) :
        ch_to_id[id_to_char[candidates[i]]] = i
    id_to_ch = {v: k for k, v in ch_to_id.items()}

    trie = Trie()
    for i in lex_dict:
        word = ''.join([id_to_ch[int(cc)-8] for cc in lex_dict[i]])
        trie.insert(word)
    # pdb.set_trace()


    def decode(mat):
        # sanity check
        check_sum = 0.0
        for x in mat[0]:
            check_sum += math.exp(x)
        # pdb.set_trace()
        assert abs(1.0 - check_sum) < 0.01

        beam = beam_search(mat, lm_function, ch_to_id, config['insertionBonus'], config['lmWeight'], config['beamSize'], trie, expansion_chars=expansion_characters, blank_fudge=config['blankFudge'])
        return beam

    print("DECODING FOR: {}\n".format(config['arkFile']))
    #arc_file = kaldi_io.read_mat_ark(config['arkFile'])
    matrices, utterances = readScp(config['arkFile'])
    # pdb.set_trace()

    with open(config['ctmOut'], mode="w", buffering=1) as f:
        #for key, mat in arc_file:
        for mat, key in zip(matrices, utterances):

            temp  = mat[:,candidates]
            row_sums = temp.sum(axis=1)
            new_mat = temp / row_sums[:, np.newaxis]
            new_mat = np.log(new_mat)

            if config['show'] == 1:
                a= greedy_search(new_mat, id_to_ch)
                f.write("greedy: {}\n".format(a))

            beam = decode(new_mat)

            for i, utterance in enumerate(beam):
                if i >= config['nBestOutput']:
                    break
                if(len(utterance)>0 and utterance[-1] == ' '):
                    utterance = utterance[:-1]
                s = "{} {}\n".format(key, utterance)
                f.write(s)
            print('done ',key)
    print("finished decoding")
