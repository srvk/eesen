from models.achen import *
from models.achen_conv import *
from models.achen_sum import *
from models.convnet import *
from models.deep_bilstm import *
#from models.arcnet_video import *


#it returns an object lm_reader that internaly will manage all the data
#client will be agnostic for the internals
def create_model(config):

    if config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.DEEP_BILSTM:
        return DeepBidirRNN(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.DEEP_BILSTM_RELU:
        return DeepBidirRNNRelu(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.ACHEN:
        return Achen(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.ACHEN_SUM:
        return AchenSum(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.CONVNET:
        return ConvNet(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.ACHEN_CONV:
        return AchenConv(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.ARCNET_VIDEO:
        return ArcNetVideo(config)
    elif config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.ARCNET_VIDEO:
        print("ARCNET underconstruction (fused) video is almost there...")
        print(debug.get_debug_info())
        print("exiting...\n")
        sys.exit()
        #return ArcNet(config)
    else:
        print("model selected not existing")
        print(debug.get_debug_info())
        print("exiting...\n")
        sys.exit()

