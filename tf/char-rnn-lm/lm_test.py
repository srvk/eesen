#coding: utf-8
from __future__ import print_function
import lm_constants
import argparse
import math
import numpy as np

def main_parser():
    #TODO add option to not get .arks and only ter
    parser = argparse.ArgumentParser(description='Test TF-Eesen Language Model (Char RNN)')

    #io options
    parser.add_argument('--data_dir', default = "", help = "like data_dir for training script")
    parser.add_argument('--results_filename', help="where the results (in CTM format will be written)")

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





if __name__ == "__main__":
    parser = main_parser()
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
    filename = pickle.load(open('data/units.pkl','rb'))
    rev_filename = {j:k for k,j in filename.iteritems()}
    for line in rev_filename:
        # lin = line.decode('UTF-8')
        if rev_filename[line] != ' ' and rev_filename[line] !='<s>' :
            id_to_char[line] = rev_filename[line]
        #id_to_char[int(l[1])] = l[0]
    # invert dict
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
    id_to_ch = {v: k for k, v in ch_to_id.items()}
    #pdb.set_trace()
    trie = Trie()
    for i in lex_dict:
        #word = lex_dict[i]
        word = [id_to_char[int(cc)] for cc in lex_dict[i]]
        trie.insert(word)
    # pdb.set_trace()


    def decode(mat):
        # sanity check
        check_sum = 0.0
        for x in mat[0]:
            check_sum += math.exp(x)
        # pdb.set_trace()
        assert abs(1.0 - check_sum) < 0.01

        beam = beam_search(mat, lm_function, ch_to_id, config['insertionBonus'], config['lmWeight'], config['beamSize'], trie, expansion_chars=expansion_characters)
        return beam

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