class MODEL_NAME:
    RNN="rnn"
    CNN_RNN="rnn"

class CONF_TAGS:

    #architecture arguments
    LSTM_TYPE = "lstm_type"
    MODEL = "model"
    NLAYERS="nlayer"
    NHIDDEN="nhidden"
    BATCH_NORM = "batch_norm"
    NEMBEDS = "input_feats_dim"
    DROPOUT = "drop_out"

    #runtime arguments
    NEPOCH="nepoch"
    LR_RATE="lr_rate"

    #io dir
    DATA_DIR = "data_dir"
    RESULTS_DIR = "results_dir"
    STORE_MODEL = "store_model"

#names
class DEFAULT_NAMES:
    MODEL_DIR_NAME = "model"
    SAT_DIR_NAME = "sat"



