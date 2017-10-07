#coding: utf-8
from __future__ import print_function
from lm_utils.lm_fileutils import debug
from lm_utils.lm_data_structures.trie import Trie
from lm_utils.lm_fileutils.kaldi import read_scp_info_dic
from lm_utils.lm_fileutils.kaldi import readMatrixByOffset
from lm_decoder.decoder_factory import create_decoder
import numpy as np
import pickle
import lm_constants
import argparse
import math
import os
import sys
import pdb
def main_parser():
    #TODO add option to not get .arks and only ter
    parser = argparse.ArgumentParser(description='Test TF-Eesen Language Model (Char RNN)')

    #io options
    parser.add_argument('--results_filename', type=str, required=True, help="where the results (in CTM format will be written)")

    parser.add_argument('--units_file', type=str, required=True, help = "units.txt to perfrom the conversion form int to char")

    parser.add_argument('--lexicon_file', type=str, required=True, help = "giving lexicon will add close vocabulary behaviour to the model")

    parser.add_argument('--decoding_strategy', default = "beam_search", help = "either greedy search or beam search")

    parser.add_argument('--insertion_bonus', type=float, default=0.6, help="inserion bonus")

    parser.add_argument('--beam_size', type=int, default=40, help="size of the beam that we will use")

    parser.add_argument('--batch_size', type=int, default=16, help="size of the batches used during testing")

    parser.add_argument('--lm_weight', type=float, default=0.5, help = "language model weights")

    parser.add_argument('--lm_weights_ckpt', type=str, required=True, help = "language model weights")

    parser.add_argument('--lm_config', type=str, required=True, help = "config.pkl from the language model")

    parser.add_argument('--blank_scaling', type=float,  default=1.0, help="blank_scaling")

    parser.add_argument('--ctc_probs_scp', type=str, required=True, help="ctc likelihoods in scp format")


    return parser


def create_test_config(args):

    config = pickle.load(open(args.lm_config, "rb"))
    config_new = {

        #io options
        lm_constants.CONFIG_TAGS_TEST.LM_WEIGHT : args.lm_weight,
        lm_constants.CONFIG_TAGS_TEST.LM_WEIGHTS_CKPT : args.lm_weights_ckpt,
        lm_constants.CONFIG_TAGS_TEST.RESULTS_FILENAME : args.results_filename,
        lm_constants.CONFIG_TAGS_TEST.UNITS_FILE : args.units_file,
        lm_constants.CONFIG_TAGS_TEST.LEXICON_FILE : args.lexicon_file,
        lm_constants.CONFIG_TAGS_TEST.TYPE_OF_DECODING : args.decoding_strategy,

        #computing options
        lm_constants.CONFIG_TAGS_TEST.BEAM_SIZE : args.beam_size,
        lm_constants.CONFIG_TAGS_TEST.INSERTION_BONUS: args.insertion_bonus,
        lm_constants.CONFIG_TAGS_TEST.BLANK_SCALING: args.blank_scaling,
        lm_constants.CONFIG_TAGS_TEST.CTC_PROBS: args.ctc_probs_scp,

    }


    config.update(config_new)

    return config

def get_units(units_path):

    units_char_to_id={}
    flag_eos_find = False

    if(os.path.isfile(units_path)):
        with open(units_path, 'r') as input:
            for line in input:
                if(line.split()[0] == lm_constants.SPECIAL_CARACTERS.EOS):
                    flag_eos_find = True

                if(line.strip().split(' ')[0] == lm_constants.SPECIAL_CARACTERS.SPACE_SYMBOL):
                    units_char_to_id[lm_constants.SPECIAL_CARACTERS.SPACE] = int(line.strip().split(' ')[1])
                else:
                    units_char_to_id[line.strip().split(' ')[0]] = int(line.strip().split(' ')[1])
    else:
        print("Path to units txt does not exist")
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    if(not flag_eos_find):
        print("Path eos symbol was not found ("+lm_constants.SPECIAL_CARACTERS.EOS+")")
        print(debug.get_debug_info())
        print("exiting...")

    print(units_char_to_id)
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

    config[lm_constants.CONFIG_TAGS_TEST.C2I]=units_char_to_id


    if(config[lm_constants.CONFIG_TAGS_TEST.TYPE_OF_DECODING] != ""):

        if(config[lm_constants.CONFIG_TAGS_TEST.LEXICON_FILE] != ""):
            print("Using lexicon (trie) for decoding...")
            print("charging lexicon "+config[lm_constants.CONFIG_TAGS_TEST.LEXICON_FILE]+"...")
            words = get_words(config[lm_constants.CONFIG_TAGS_TEST.LEXICON_FILE])

            trie = Trie()
            for word in words:
                #word = lex_dict[i]
                #print(word)
                word = [cc for cc in word]
                trie.insert(word)
        else:
            print("Decoding will be performed without a lexicon (trie) ...")


    # if(config[lm_constants.CONFIG_TAGS_TEST.COMPUTE_WER]):
    #     print("Computing wer.")
    #     print("reading labels file form data_dir...")
    # else:
    #     print("Not computing wer.")


    # print(80 * "-")
    # print("reading sat...")
    # print(80 * "-")

    test_softprobs_path = config[lm_constants.CONFIG_TAGS_TEST.CTC_PROBS]

    test_softprobs = read_scp_info_dic(test_softprobs_path)

    print("data read.")
    print(80 * "-")
    print(80 * "-")
    print("about to start testing with the following config:")
    print(config)

    print("creating decoder...")
    decoder = create_decoder(config)

    #only one line will be stored in memory
    with open(config[lm_constants.CONFIG_TAGS_TEST.RESULTS_FILENAME], mode="w", buffering=1) as output_file:
        for key, info in test_softprobs.items():

            #copying mat
            mat_softprob = readMatrixByOffset(info[1],info[2])
            m_mat_softprobs = np.copy(mat_softprob)

            #reescaling blank
            m_mat_softprobs[:, 0] = m_mat_softprobs[:, 0] * config[lm_constants.CONFIG_TAGS_TEST.BLANK_SCALING]

            #normalize
            row_sums = m_mat_softprobs.sum(axis=1)

            #normalitzation
            #we are applying the normalization
            new_mat = m_mat_softprobs / row_sums[:, np.newaxis]

            #we need softprobs (no log applied)
            log_soft_probs = np.log(new_mat)

            #list of decoded sequences
            decoded_sequences = decoder.decode(log_soft_probs, units_char_to_id, trie)[0]

            #TODO add nbest list
            if(len(decoded_sequences) > 0 and decoded_sequences[-1] == lm_constants.SPECIAL_CARACTERS.SPACE):
                decoded_sequences = decoded_sequences[:-1]

            final_sentence = "{} {}\n".format(key, decoded_sequences)
            output_file.write(final_sentence)
            print('done ',key)
    print("finished decoding")