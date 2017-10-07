from utils.fileutils import debug
import sys

from lm_decoder.decoder_greedy_search import GreedySearch
from lm_decoder.decoder_beam_search import BeamSearch

import lm_constants


def create_decoder(config):
    if(config[lm_constants.CONFIG_TAGS_TEST.TYPE_OF_DECODING] == lm_constants.DECODE_STRATEGY_NAMES.BEAM_SEARCH):

        print("creating beam search decoder...")

        return BeamSearch(config)

    elif(config[lm_constants.CONFIG_TAGS_TEST.TYPE_OF_DECODING] == lm_constants.DECODE_STRATEGY_NAMES.GREEDY_SEARCH):

        print("creating greedy search decoder...")
        return GreedySearch(config)
    else:
        print("Path to words.txt does not exist")
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()
