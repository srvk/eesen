from models_16.deep_bilstm import *
from models_16.deep_bilstm_test import *

#it returns an object lm_reader that internaly will manage all the data
#client will be agnostic for the internals
def create_model(config):

    if config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.DEEP_BILSTM:
        return DeepBidirRNN(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.DEEP_BILSTM_TEST:
        return DeepBidirRNNTest(config)
    else:
        print("model selected not existing")
        print(debug.get_debug_info())
        print("exiting...\n")
        sys.exit()

