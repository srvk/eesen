#coding: utf-8
from __future__ import print_function
from lm_utils.lm_fileutils import debug
from lm_utils.lm_data_structures.trie import Trie
import numpy as np
import lm_constants
import argparse
import math
import os
import sys

def main_parser():
    #TODO add option to not get .arks and only ter
    parser = argparse.ArgumentParser(description='Test TF-Eesen Language Model (Char RNN)')

    #io options
    parser.add_argument('--data_dir', default = "", help = "like data_dir for training script")
    parser.add_argument('--results_filename', help="where the results (in CTM format will be written)")
    parser.add_argument('--units_file', help = "like data_dir for training script")
    parser.add_argument('--lexicon_file', default = "", help = "giving lexicon will add close vocabulary behaviour to the model")

    #previous models options
    parser.add_argument('--lm_cpkt', help="weight file for lm model")
    parser.add_argument('--beam_size', type=int, default=40, help="weight file for lm model")
    parser.add_argument('--insertion_bonus', type=float, default=0.6, help="inserion bonus")
    parser.add_argument('--gen_priors', default = False, action='store_true', help="preview priors during decoding")
    parser.add_argument('--blank_scaling', type=float, default=1.0, help="blank_scaling")
    parser.add_argument('--n_best_output', default = False, action='store_true', help="output N best utterances")

    #decoding options
    parser.add_argument('--decoding_strategy', default = "beam_search", help = "type of decoding to apply to data")
    parser.add_argument('--train_config', default = "", help = "model to load for evaluation")
    parser.add_argument('--trained_weights', default = "", help = "model to load for evaluation")

    #computing options
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--temperature', default = 1, type=float, help='temperature used in softmax')
    parser.add_argument('--use_priors', default = False, action='store_true', help='if --use_priors it will take ')
    parser.add_argument('--compute_ter', default = False, action='store_true', help='if --compute_ter the labels will be taken from data_dir (label_phn.test)and ter will be computed')

    return parser


def create_test_config(args):

    config = {

        #io options
        lm_constants.CONFIG_TAGS_TEST.DATA_DIR : args.data_dir,
        lm_constants.CONFIG_TAGS_TEST.RESULTS_FILENAME : args.resuls_filename,
        lm_constants.CONFIG_TAGS_TEST.UNITS_FILE : args.units_file,
        lm_constants.CONFIG_TAGS_TEST.LEXICON_FILE : args.lexicon_file,

        #computing options
        lm_constants.CONFIG_TAGS_TEST.LM_CKPT : args.lm_cpkt,
        lm_constants.CONFIG_TAGS_TEST.BEAM_SIZE : args.beam_size,
        lm_constants.CONFIG_TAGS_TEST.INSERTION_BONUS: args.insertion_bonus,
        lm_constants.CONFIG_TAGS_TEST.GEN_PRIORS: args.gen_priors,
        lm_constants.CONFIG_TAGS_TEST.NBEST_OUTPUT: args.n_best_output,


    }

    return config

def get_units(units_path):

    units_char_to_id={}

    if(os.path.isfile(units_path)):
        with open(units_path, 'r') as input:
            for line in input:
                units_char_to_id[line.split()[0]]=[int(element) for element in line.split()[1:]]
    else:
        print("Path to units txt does not exist")
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    units_id_to_char = {v: k for k, v in units_char_to_id.items()}

    return units_char_to_id, units_id_to_char

#it will read a lis of words (i.e. one word per line)
def get_words(lexicon_path):

    words=[]

    if(os.path.isfile(lexicon_path)):
        with open(lexicon_path, 'r') as input:
            for line in input:
                words.append(line.replace("\n",""))

    else:
        print("Path to words.txt does not exist")
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    return words

if __name__ == "__main__":

    parser = main_parser()
    args = parser.parse_args()
    config = create_test_config(args)

    units_char_to_id, units_id_to_char = get_units(config[lm_constants.CONFIG_TAGS_TEST.UNITS_FILE])


    if(config[lm_constants.CONFIG_TAGS_TEST.TYPE_OF_DECODING] != ""):

        if(config[lm_constants.CONFIG_TAGS_TEST.LEXICON_FILE] != ""):
            print("Using lexicon (trie) for decoding...")
            print("charging lexicon "+config[lm_constants.CONFIG_TAGS_TEST.LEXICON_FILE]+"...")
            words = get_words(config[lm_constants.CONFIG_TAGS_TEST.LEXICON_FILE])

            trie = Trie()

            for word in words:
                #word = lex_dict[i]
                word = [units_char_to_id[cc] for cc in word]
                trie.insert(word)
        else:
            print("Decoding will be performed without a lexicon (trie) ...")


    print("DECODING FOR: {}\n".format(config['arkFile']))
    arc_file = kaldi_io.read_mat_ark(config['arkFile'])

    with open(config['ctmOut'], mode="w", buffering=1) as f:
        for key, mat in arc_file:
            #pdb.set_trace()
            temp  = mat[:,candidates]
            temp[:,0] = temp[:,0]*config['bs']
            # pdb.set_trace()
            row_sums = temp.sum(axis=1)
            new_mat = temp / row_sums[:, np.newaxis]
            new_mat = np.log(new_mat)
            # pdb.set_trace()

            if config['show'] == 1:
                a= greedy_search(new_mat, id_to_ch)
                f.write("greedy-bs-0.5 {}: {}\n".format(key,a))
                # a= greedy_search(mat[:,1:], temp_d)
                # f.write("greedy-2 {}: {}\n".format(key,a))
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