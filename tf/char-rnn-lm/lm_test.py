#coding: utf-8
from __future__ import print_function
from lm_utils.lm_fileutils import debug
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
    parser.add_argument('--lexicon', default = "", help = "giving lexicon will add close vocabulary behaviour to the model")

    #previous models options
    parser.add_argument('--lm_cpkt', help="weight file for lm model")
    parser.add_argument('--beam_size', type=int, default=40, help="weight file for lm model")
    parser.add_argument('--insertion_bonus', type=float, default=0.6, help="inserion bonus")
    parser.add_argument('--gen_priors', default = False, action='store_true', help="preview priors during decoding")
    parser.add_argument('--blank_scaling', type=float, default=1.0, help="blank_scaling")
    parser.add_argument('--n_best_output', default = False, action='store_true', help="output N best utterances")

    #decoding options
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

        #computing options
        lm_constants.CONFIG_TAGS_TEST.LM_CKPT : args.lm_cpkt,
        lm_constants.CONFIG_TAGS_TEST.BEAM_SIZE : args.beam_size,
        lm_constants.CONFIG_TAGS_TEST.INSERTION_BONUS: args.insertion_bonus,
        lm_constants.CONFIG_TAGS_TEST.GEN_PRIORS: args.gen_priors,
        lm_constants.CONFIG_TAGS_TEST.NBEST_OUTPUT: args.n_best_output,


    }

    return config

def get_units(units_path):
    units_dic={}

    if(os.path.isfile(units_path)):
        with open(units_path, 'r') as input:
            for line in input:
                units_dic[line.split()[0]]=line.split()[1]
    else:
        print("Path to units txt does not exist")
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    return units_dic



if __name__ == "__main__":

    parser = main_parser()
    args = parser.parse_args()
    config = create_test_config(args)




    char_to_id = {v: k for k, v in id_to_char.items()}
    print(id_to_char)
    #pdb.set_trace()
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

    #######################################

    id_to_ch = {v: k for k, v in ch_to_id.items()}
    #pdb.set_trace()
    trie = Trie()
    for i in lex_dict:
        #word = lex_dict[i]
        word = [id_to_char[int(cc)] for cc in lex_dict[i]]
        trie.insert(word)
    # pdb.set_trace()



    print("DECODING FOR: {}\n".format(config['arkFile']))
    arc_file = kaldi_io.read_mat_ark(config['arkFile'])
    #HACK!!!!!!
    candidates[9:] = [t-1 for t in candidates[9:] ]
    #pdb.set_trace()
    #temp_d = {char_to_id[k]-1:k for k in char_to_id.keys()}
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