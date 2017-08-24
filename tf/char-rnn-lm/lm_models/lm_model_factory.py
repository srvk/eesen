from lm_models.rnn import *
from utils.fileutils import debug
import sys
import lm_constants

#it returns an object lm_reader that internaly will manage all the data
#client will be agnostic for the internals
def lm_create_model(config):

    if config[lm_constants.CONF_TAGS.MODEL] == lm_constants.MODEL_NAME.RNN:
        return RNN(config)
    else:
        print("model selected not existing")
        print(debug.get_debug_info())
        print("exiting...\n")
        sys.exit()