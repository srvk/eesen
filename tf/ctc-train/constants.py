

#TODO put everything inside a TAG class

#names
NO_LANGUAGE_NAME = "no_name_language"
NO_TARGET_NAME = "no_target_name"

#configuartion tags
LANGUAGE_SCHEME = "target_scheme"
INPUT_FEATS_DIM = "input_feats_dim"
BATCH_SIZE = "batch_size"
LSTM_TYPE = "lstm_type"
DATA_DIR = "data_dir"
ONLINE_AUGMENT_CONF = "online_augment_conf"
MODEL_DIR = "model_dir"
TRAIN_DIR = "train_dir"
STORE_MODEL = "store_model"
BATCH_NORM = "batch_norm"
FEAT_PROJ = "feat_proj"
NPROJ = "n_proj"



#training conf
NEPOCH="nepoch"
CLIP="clip"
HALF_PERIOD="half_period"
HALF_RATE="half_rate"
HALF_AFTER="half_after"
LR_RATE="lr_rate"
DO_SHUF="do_shuf"
GRAD_OPT="grad_opt"
BATCH_SIZE="batch_size"
RANDOM_SEED="random_seed"
DEBUG="debug"
CONTINUE_CKPT="continue_ckpt"



#architecture config
L2="l2"
NLAYERS="nlayer"
NHIDDEN="nhidden"

#adaptation
SAT="sat"
ADAPT_STAGE="adapt_stage"
SAT_FEAT_DIM="sat_feat_dim"
NUM_SAT_LAYERS="num_sat_layers"
ADAPT_ORG_PATH="adapt_org_path"


class ADAPTATION_STAGES:
    UNADAPTED="unadapted"

class AUGMENTATION:
    WINDOW="win"
    FACTOR="factor"
    ROLL="roll"
