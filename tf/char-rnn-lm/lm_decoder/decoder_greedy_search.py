import numpy as np
from itertools import groupby
import lm_constants

class BeamSearch:

    def __init__(self, config):
        print("construction")
        self.__blank_id = config[lm_constants.CONFIG_TAGS_TEST.BLANK_ID]

    def decode(self, log_ctc_prob):
        # to make if comparision shorter
        decoded_seq = []

        for t in range(len(log_ctc_prob)):
            arg_max = np.argmax(log_ctc_prob[t])
            decoded_seq.append(arg_max)
        ids = [x[0] for x in groupby(decoded_seq)]
        cleaned_decoded_seq = [x for x in filter(lambda x: x != self.__blank_id, ids)]

        return [cleaned_decoded_seq]
