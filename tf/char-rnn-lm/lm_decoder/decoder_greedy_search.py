import numpy as np
from itertools import groupby
import lm_constants

class GreedySearch:

    def __init__(self, config):
        print("construction")

    def decode(self, log_ctc_prob, char_to_int, trie=None):
        # to make if comparision shorter
        decoded_seq = []

        for t in range(len(log_ctc_prob)):
            arg_max = np.argmax(log_ctc_prob[t])
            decoded_seq.append(arg_max)
        ids = [x[0] for x in groupby(decoded_seq)]
        rev_c2i = {k:j for j,k in char_to_int.iteritems()}
        cleaned_decoded_seq = [rev_c2i[x] for x in filter(lambda x: x != lm_constants.IDS.BLANK_ID, ids)]

        #filter(lambda a: a != 0,[x[0] for x in groupby(numpy.argmax(tmat,axis=1))])

        return [cleaned_decoded_seq]
