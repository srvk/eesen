
from lm_decoder.decoder_greedy_search import Beam_Search
from lm_decoder.decoder_greedy_search import Greedy_Search

import lm_constants


def create_decoder(config):
    if(config[lm_constants.DECODING_TYPE.BEAM_SEARCH]):

        print("creating beam search decoder...")
        return Beam_Search(config)

    elif(config[lm_constants.DECODING_TYPE.GREEDY_SEARCH]):

        print("creating greedy search decoder...")
        return Greedy_Search(config)
