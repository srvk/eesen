class MODEL_NAME:
    RNN="rnn"
    CNN_RNN="cnn_rnn"

class SPECIAL_CARACTERS:
    SPACE=" "

class CONF_TAGS:

    #developer arguments
    DEBUG = "debug"

    #architecture arguments
    EMBEDS_SIZE = "embed_size"
    NHIDDEN="nhidden"
    NLAYERS="nlayer"
    LSTM_TYPE = "lstm_type"
    MODEL = "model"
    BATCH_NORM = "batch_norm"
    BATCH_SIZE = "batch_size"
    DROPOUT = "drop_out"
    NUM_TARGETS = "num_targets"

    #runtime arguments
    NEPOCH="nepoch"
    RANDOM_SEED="random_seed"
    LR_RATE="lr_rate"
    OPTIMIZER="optimizer"

    #io dir
    DATA_DIR = "data_dir"
    TRAIN_DIR = "train_dir"

    CONTINUE_CKPT = "continue_ckpt"
    IMPORTED_CONFIG = "imported_config"

    DO_SHUF="do_shuf"

    #sat arguments
    SAT_CONF="sat_conf"
    SAT_SATGE="sat_stage"
    NUM_SAT_LAYERS="num_sat_layers"
    NUM_SAT_DIM="sat_feat_dim"

class CONFIG_TAGS_TEST:
    #io options
    RESULTS_FILENAME="results_filename"
    DATA_DIR="data_dir"

    #previous models options
    LM_CKPT = "lm_cpkt"
    TRAIN_CONFIG="train_config"

    #output options
    BEAM_SIZE = "beam_size"
    INSERTION_BONUS = "insertion_bonus"
    GEN_PRIORS = "gen_priors"
    NBEST_OUTPUT = "nbest_output"

    #decoding options
    BATCH_SIZE="batch_size"
    TEMPERATURE="temperature"
    USE_PRIORS="use_priors"
    COMPUTE_TER="compute_ter"

#names
class DEFAULT_NAMES:
    MODEL_DIR_NAME = "model"
    SAT_DIR_NAME = "sat"

class FILE_NAMES:
    CONFIG_PKL = "config.pkl"

    TR_X = "labels.tr"
    CV_X = "labels.cv"

    SAT = "sat_local.scp"

class SAT_SATGES:
    UNADAPTED = "non_adapted"
    TRAIN_SAT = "train_sat"
    FINE_TUNE = "fine_tune"
    CONCAT = "concat"
    FUSE = "fuse"

class SCOPES:
    SPEAKER_ADAPTAION = "speaker_adaptation"
    SAT_FUSE = "sat_concatenation"
    SAT_SHIFT = "sat_shift"
    OUTPUT = "output_layers"

class OPTIMIZERS:
    ADAM="adam"
    SDG="sdg"


