#!/usr/bin/env python3

import argparse
import kaldi_io
import os
from searchAlgorithms import greedy_search

BLANK = "<eps>"
SPACE = " "
EOS = "</s>"
BLANK_ID = 0


def create_parser():
    # upper case
    path = "/Users/thomaszenkel/GitRepos/MasterThesis/bin/CTC/ABC/"


    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Decode Character Distribution of CTC"
    )

    parser.add_argument(
        '--arkFileDir', default=path + "/eval2000/",
        help="path to ctc likelihoods file (.ark)"
    )

    parser.add_argument(
        '--ctmOut', default="Results/greedyTest.txt",
        help="output file to write output to"
    )

    parser.add_argument(
        '--unitsFile', default=path+"/units.txt"
    )

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(args)

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

    with open(args.ctmOut, mode="w") as f:
        for arkFile in os.listdir(args.arkFileDir):
            print("DECODING FOR: {}".format(arkFile))
            ark_file = kaldi_io.read_mat_ark(args.arkFileDir + "/" + arkFile)
            for key, mat in ark_file:
                utt = greedy_search(mat, id_to_char)
                f.write("{} {}\n".format(key, utt))

    print("finished decoding")