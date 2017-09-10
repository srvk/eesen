#coding: utf-8
from __future__ import print_function
from lm_utils.lm_fileutils import debug
from lm_utils.lm_data_structures.trie import Trie
from lm_utils.lm_fileutils.kaldi import read_scp_info_dic
from lm_decoder.decoder_factory import create_decoder
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
    parser.add_argument('--units_file', help = "units.txt to perfrom the conversion form int to char")
    parser.add_argument('--lexicon_file', default = "", help = "giving lexicon will add close vocabulary behaviour to the model")

    #previous models options
    parser.add_argument('--lm_cpkt', help="weight file for lm model")
    parser.add_argument('--beam_size', type=int, default=40, help="weight file for lm model")
    parser.add_argument('--insertion_bonus', type=float, default=0.6, help="inserion bonus")
    parser.add_argument('--gen_priors', default = False, action='store_true', help="preview priors during decoding")
    parser.add_argument('--n_best_output', default = False, action='store_true', help="output N best utterances")

    #decoding options
    parser.add_argument('--decoding_strategy', default = "beam_search", help = "type of decoding to apply to data")
    parser.add_argument('--config_ckpt', default = "", help = "model to load for evaluation")
    parser.add_argument('--weights_ckpt', default = "", help = "model to load for evaluation")
    parser.add_argument('--blank_scaling', type=float, default=1.0, help="blank_scaling")

    #computing options
    parser.add_argument('--temperature', default = 1, type=float, help='temperature used in softmax')
    parser.add_argument('--compute_wer', default = False, action='store_true', help='if --compute_ter the labels will be taken from data_dir (label_phn.test)and ter will be computed')

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
        lm_constants.CONFIG_TAGS_TEST.BLANK_SCALING: args.blank_scaling,

        parser.add_argument('--blank_scaling', type=float, default=1.0, help="blank_scaling")

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

def translate_sentence_id_to_char (dict_id_to_char, sentence):

    translated_sentence=[]

    for element in sentence:
        translated_sentence.append(dict_id_to_char[element])

    return translated_sentence

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


    if(config[lm_constants.CONFIG_TAGS_TEST.COMPUTE_WER]):
        print("Computing wer.")
        print("reading labels file form data_dir...")
    else:
        print("Not computing wer.")


    print(80 * "-")
    print("Reading sat...")
    print(80 * "-")

    test_logprobs_path = os.path.join(config[lm_constants.CONF_TAGS.DATA_DIR], lm_constants.FILE_NAMES.TEST_LOGPORBS_SCP)

    test_logprobs = read_scp_info_dic(test_logprobs_path)

    print("data read.")
    print(80 * "-")
    print(80 * "-")
    print("about to start testing with the following config:")
    print(config)

    print("creating decoder...")
    decoder = create_decoder(config)

    #only one line will be stored in memory
    with open(config[lm_constants.CONFIG_TAGS_TEST.RESULTS_FILENAME], mode="w", buffering=1) as output_file:
        for key, mat in test_logprobs:

            #copying mat
            m_mat = mat
            #reescaling blank
            m_mat[:,0] = m_mat[:,0]*config['bs']

            row_sums = m_mat.sum(axis=1)

            #normalitzation + log?
            new_mat = m_mat / row_sums[:, np.newaxis]
            new_mat = np.log(new_mat)

            #TODO return a list of decoded sequences
            decoded_sequences = decoder(new_mat)

            for decoded_sequence in decoded_sequences:
                final_utt = translate_sentence_id_to_char(decoded_sequence)
                s = "{} {}\n".format(key, final_utt)

                f.write(s)
            print('done ',key)
    print("finished decoding")