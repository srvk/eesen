

class DEFAULT_FILENAMES:
    SAT="sat_local"


class SCOPES:
    SPEAKER_ADAPTAION="speaker_adaptation"
    OUTPUT="output_layers"

#names
class DEFAULT_NAMES:
    NO_LANGUAGE_NAME = "no_name_language"
    NO_TARGET_NAME = "no_target_name"
    MODEL_DIR_NAME = "model"
    SAT_DIR_NAME = "sat"

class CONF_TAGS:

    #general arguments
    DEBUG="debug"
    STORE_MODEL = "store_model"
    DATA_DIR = "data_dir"
    TRAIN_DIR = "train_dir"
    MODEL_DIR = "model_dir"

    #io arguments
    CONTINUE_CKPT="continue_ckpt"
    BATCH_SIZE="batch_size"
    DO_SHUF="do_shuf"
    ONLINE_AUGMENT_CONF = "online_augment_conf"

    #architecture arguments
    LSTM_TYPE = "lstm_type"
    NPROJ = "nproj"
    L2="l2"
    NLAYERS="nlayer"
    NHIDDEN="nhidden"
    CLIP="clip"
    BATCH_NORM = "batch_norm"
    FEAT_PROJ = "feat_proj"
    GRAD_OPT="grad_opt"
    LANGUAGE_SCHEME = "target_scheme"
    INPUT_FEATS_DIM = "input_feats_dim"

    #runtime arguments
    NEPOCH="nepoch"
    LR_RATE="lr_rate"
    HALF_PERIOD="half_period"
    HALF_RATE="half_rate"
    HALF_AFTER="half_after"

    #training conf
    RANDOM_SEED="random_seed"

    #sat arguments
    SAT="sat"
    APPLY_SAT="apply_sat"
    NUM_SAT_LAYERS="num_sat_layers"
    SAT_FEAT_DIM="sat_feat_dim"

class AUGMENTATION:
    WINDOW="win"
    FACTOR="factor"
    ROLL="roll"

class LSTM_TYPE:
    CUDNN= "cudnn_lstm"
    FUSE= "fuse_lstm"
    NATIVE= "native_lstm"
