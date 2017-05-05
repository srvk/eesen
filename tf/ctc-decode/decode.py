#!/usr/bin/env python3
#coding: utf-8

import argparse
import math
import kaldi_io
import numpy as np
from multiprocessing import Pool
import dynet as dy

from searchAlgorithms import beam_search, greedy_search
from lm_util import LM_Wrapper, get_char_dicts

BLANK = "<eps>"
SPACE = " "
EOS = "</s>"
BLANK_ID = 0

def create_parser():
    # lower case, no space, fucked up....
    path = "/Users/thomaszenkel/GitRepos/MasterThesis/bin/CTC/train_char_l5_c320_3s-0/"
    # upper case
    path = "/Users/thomaszenkel/GitRepos/MasterThesis/bin/CTC/ABC/"


    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Decode Character Distribution of CTC"
    )

    parser.add_argument(
        '--arkFile', default=path + "/eval2000/ctc_peak.1.ark",
        help="path to ctc likelihoods file (.ark)"
    )

    parser.add_argument(
        '--ctmOut', default="results.ctm",
        help="output file to write output to"
    )

    parser.add_argument(
        '--unitsFile', default=path+"/units.txt"
    )

    parser.add_argument(
        '--lmFile', default="dynetLM/lstm2048Adam.model",
        help="path to the language model file"
    )

    parser.add_argument(
        '--beamSize', type=int, default=20,
        help="size of the beam in beam search"
    )

    parser.add_argument(
        '--insertionBonus', type=float, default=0.4,
        help="insertion bonus"
    )

    parser.add_argument(
        '--lmWeight', type=float, default=1.5,
        help="language model weight"
    )

    parser.add_argument(
        '--merge', type=int, default=1,
        help="merge strings in the beam which are the same, merge should be 0 or 1"
    )

    parser.add_argument(
        '--show', type=int, default=0,
        help="merge strings in the beam which are the same, merge should be 0 or 1"
    )

    parser.add_argument(
        '--nBestOutput', type=int, default=5,
        help="output at most N utterances of the beam"
    )

    parser.add_argument(
        '--expand', default="",
        help="characters which can be added during the search, split character for string is §, example: ' §.§,'"
    )

    # dynet fix
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-gpu')

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(args)

    # hacky
    merge = True
    if args.merge == 0:
        merge = False

    # deal with [''] array returned for empty string
    expansion_characters = list(filter(lambda x: len(x) > 0, (args.expand.split("§"))))

    # create map from integer to char
    id_to_char = {BLANK_ID: BLANK}
    with open(args.unitsFile, "r") as filename:
        for line in filename:
            l = line.split()
            assert len(l) == 2
            if len(l[0]) > 1:
                print("WARN: ignoring {}".format(l[0]))
                continue
            id_to_char[int(l[1])] = l[0]
    # invert dict
    char_to_id = {v: k for k, v in id_to_char.items()}
    print(id_to_char)

    model = dy.Model()
    # not sure about the [0] stuff
    lstm = model.load(args.lmFile)[0]
    char_to_int, int_to_char = get_char_dicts()
    wrapper = LM_Wrapper(char_to_int, lstm)
    lm_function = wrapper.get_prob

    def decode(mat):
        # sanity check
        check_sum = 0.0
        for x in mat[0]:
            check_sum += math.exp(x)
        assert abs(1.0 - check_sum) < 0.01

        beam = beam_search(mat, lm_function, char_to_id, args.insertionBonus, args.lmWeight, args.beamSize,
                           merge=merge, expansion_chars=expansion_characters)
        return beam

    print("DECODING FOR: {}\n".format(args.arkFile))
    arc_file = kaldi_io.read_mat_ark(args.arkFile)

    with open(args.ctmOut, mode="w", buffering=1) as f:
        for key, mat in arc_file:
            if args.show == 1:
                print("greedy: {}".format(greedy_search(mat, id_to_char)))
            beam = decode(mat)
            for i, utterance in enumerate(beam):
                if i >= args.nBestOutput:
                    break
                s = "{} {}\n".format(key, utterance)
                if args.show == 1:
                    print(s, end="")
                else:
                    f.write(s)

    print("finished decoding")
