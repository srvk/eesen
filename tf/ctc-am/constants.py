

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
    SAT_CONF="sat_conf"
    SAT_SATGE="sat_stage"
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

class SAT_SATGES:
    UNADAPTED = "non_adapted"
    TRAIN_SAT = "train_sat"
    FINE_TUNE = "fine_tune"

class CONFIG_TAGS_TEST:
    TEST_CONFIG = "test_config"

    EVAL_CONFIG_FILE = "eval_config_file"
    WEIGHT_FILE = "eval_weights"

    BATCH_SIZE = "batch_size"
    TEMPERATURE = "temperature"
    RESULTS_DIR = "results_dir"

    PRIORS = "priors"
    TER_REF_LABELS = "ter_ref_labels"

    TEST_SCP_FILE = "test_scp_file"

