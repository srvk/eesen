import argparse
import sys
import os
import lm_constants
import pickle

from lm_reader.lm_labels_reader.labels_reader import LabelsReader
from lm_reader.lm_feats_reader.feats_reader import FeatsReaderKaldi

from lm_utils.lm_fileutils import debug
from lm_tf.lm_tf_train import *


# from lm_reader.lm_feats_reader.lm_feats_reader import FeatsReaderKald

def mainParser():
    parser = argparse.ArgumentParser(description='Train TF-RNN_LM')

    parser.add_argument('--debug', default = False, action='store_true', help='activate debug mode')

    parser.add_argument('--batch_size', default = 16, type=int, help='batch size')

    parser.add_argument('--data_dir', help = "train data loc")
    parser.add_argument('--train_dir', help = "directory where logs and weights will be stored")

    parser.add_argument('--lexicon_file', default = "./data/lexicon_char_system.txt", help = "lexicon data loc")
    parser.add_argument('--nepoch', default = 20, type=int, help='#epoch')

    # parser.add_argument('--l2', default = 0.0, type=float, help='l2 normalization')
    # parser.add_argument('--clip', default = 0.1, type=float, help='gradient clipping')

    parser.add_argument('--nlayer', default = 1, type=int, help='#layer')
    parser.add_argument('--nhidden', default = 1000, type=int, help='dimension of hidden units in single direction')
    parser.add_argument('--nembed', default = 64, type=int, help='embedding size')
    parser.add_argument('--drop_out', default = 1.0, type=float, help='dropout ((min)1.0 - (max)0) probability')

    parser.add_argument('--optimizer', default= 'Adam', help='Training Optimizer')

    parser.add_argument('--lr_rate', default=0, type=float,help='0 for default parameters')

    #TODO continute_pkt
    parser.add_argument('--continue_ckpt', default = "", help='continue experiment with the weight path given')
    parser.add_argument('--import_config', default = "", help='import configuration from a previous experiment. The configuation will be merged with the flags given.')

    parser.add_argument('--no_shuffle', default=True, dest='do_shuf', action='store_false', help='shuf batches before training')

    #sat arguments
    parser.add_argument('--train_sat', default = False, help='apply and train a sat layer')
    parser.add_argument('--concat_sat', default = False, help='apply and train a sat layer')
    parser.add_argument('--fuse_sat', default = False, help='apply and train a sat layer')
    parser.add_argument('--num_sat_layers', default = 2, type=int, help='number of sat layers for sat module')
    parser.add_argument('--num_sat_dim', default = 100, type=int, help='number of sat layers for sat module')

    #model
    parser.add_argument('--model', default = "rnn", help='number of sat layers for sat module')

    return parser

def merge_external_config(config):

    imported_config = pickle.load(open(config[lm_constants.CONF_TAGS.IMPORTED_CONFIG], "rb"))
    config[lm_constants.CONF_TAGS.NUM_TARGETS] = int(imported_config[lm_constants.CONF_TAGS.NUM_TARGETS])

def createConfig(args):

    config ={

    lm_constants.CONF_TAGS.EMBEDS_SIZE : args.nembed,
    lm_constants.CONF_TAGS.NLAYERS : args.nlayer,
    lm_constants.CONF_TAGS.NHIDDEN : args.nhidden,
    lm_constants.CONF_TAGS.DROPOUT : args.drop_out,
    lm_constants.CONF_TAGS.NEPOCH : args.nepoch,
    lm_constants.CONF_TAGS.TRAIN_DIR : args.train_dir,
    lm_constants.CONF_TAGS.BATCH_SIZE : args.batch_size,
    lm_constants.CONF_TAGS.OPTIMIZER : args.optimizer,
    lm_constants.CONF_TAGS.LR_RATE : args.lr_rate,

    lm_constants.CONF_TAGS.CONTINUE_CKPT : args.continue_ckpt,
    lm_constants.CONF_TAGS.IMPORTED_CONFIG : args.import_config,

    lm_constants.CONF_TAGS.DO_SHUF : args.do_shuf,
    lm_constants.CONF_TAGS.DATA_DIR : args.data_dir,
    lm_constants.CONF_TAGS.RANDOM_SEED : 15213,
    lm_constants.CONF_TAGS.MODEL : args.model,

    lm_constants.CONF_TAGS.DEBUG : args.debug,

    lm_constants.CONF_TAGS.NUM_SAT_LAYERS : args.num_sat_layers,
    lm_constants.CONF_TAGS.NUM_SAT_DIM : args.num_sat_dim,

    }

    if(not args.train_sat and not args.concat_sat and not args.fuse_sat):
        config[lm_constants.CONF_TAGS.SAT_SATGE] = lm_constants.SAT_SATGES.UNADAPTED


    #creating and setting train dir (where model and logs are stored)
    if(os.path.exists(config[lm_constants.CONF_TAGS.TRAIN_DIR])):
        config[lm_constants.CONF_TAGS.TRAIN_DIR] = os.path.join(config[lm_constants.CONF_TAGS.TRAIN_DIR], lm_constants.DEFAULT_NAMES.MODEL_DIR_NAME)
        if(not os.path.exists(config[lm_constants.CONF_TAGS.TRAIN_DIR])):
            os.makedirs(config[lm_constants.CONF_TAGS.TRAIN_DIR])
    else:
        print("Error: window can not be even currently : "+config[lm_constants.CONF_TAGS.TRAIN_DIR])
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    check_and_gen_sat_config(args, config)

    return config

def check_and_gen_sat_config(args, config):

    if(args.train_sat):
        if(args.concat_sat or args.fuse_sat):
            print("Error: only one sat method can be applied")
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()
        config[lm_constants.CONF_TAGS.SAT_SATGE] = lm_constants.SAT_SATGES.TRAIN_SAT

    if(args.concat_sat):
        if(args.train_sat or args.fuse_sat):
            print("Error: only one sat method can be applied")
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()
        config[lm_constants.CONF_TAGS.SAT_SATGE] = lm_constants.SAT_SATGES.CONCAT

    if(args.fuse_sat):
        if(args.train_sat or args.concat_sat):
            print("Error: only one sat method can be applied")
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()
        config[lm_constants.CONF_TAGS.SAT_SATGE] = lm_constants.SAT_SATGES.FUSE

def main():


    parser = mainParser()
    args = parser.parse_args()
    config = createConfig(args)


    print("about to train with the following configuration:")
    print(80 * "-")
    for key, element in config.items():
        print(str(key)+" "+str(element))

    print(80 * "-")
    print(80 * "-")
    print("reading data")
    print(80 * "-")
    print("reading tr_x...")
    print(80 * "-")


    tr_x_path_file = os.path.join(config[lm_constants.CONF_TAGS.DATA_DIR], lm_constants.FILE_NAMES.TR_X)
    tr_x = LabelsReader(tr_x_path_file, config[lm_constants.CONF_TAGS.BATCH_SIZE])

    print("reading cv_x...")
    print(80 * "-")
    cv_x_path_file = os.path.join(config[lm_constants.CONF_TAGS.DATA_DIR], lm_constants.FILE_NAMES.CV_X)

    cv_x = LabelsReader(cv_x_path_file, config[lm_constants.CONF_TAGS.BATCH_SIZE])

    if(cv_x.get_num_diff_labels() > tr_x.get_num_diff_labels()):
        print("Warning: number of targets has changed between sets (e.g. train and validation)")
        print("tr_x tarnget number will change from "+tr_x.get_num_diff_labels()+" to "+cv_x.get_num_diff_labels())
        tr_x.update_num_diff_labels(cv_x.get_num_diff_labels())
        print(debug.get_debug_info())
        print(80 * "-")

    config[lm_constants.CONF_TAGS.NUM_TARGETS] = tr_x.get_num_diff_labels()

    if config[lm_constants.CONF_TAGS.SAT_SATGE] != lm_constants.SAT_SATGES.UNADAPTED:
        print("reading sat...")
        print(80 * "-")

        tr_sat_path_file = os.path.join(config[lm_constants.CONF_TAGS.DATA_DIR], lm_constants.FILE_NAMES.SAT)
        tr_sat = FeatsReaderKaldi(tr_sat_path_file, tr_x.get_uttid())

        cv_sat_path_file = os.path.join(config[lm_constants.CONF_TAGS.DATA_DIR], lm_constants.FILE_NAMES.SAT)
        cv_sat = FeatsReaderKaldi(cv_sat_path_file, cv_x.get_uttid())

        config[lm_constants.CONF_TAGS.SAT_FEAT_DIM] = int(tr_sat.get_feat_dim())

        data = (tr_x, cv_x, tr_sat, cv_sat)

    else:
        data = (tr_x, cv_x, None, None)

    if(config[lm_constants.CONF_TAGS.IMPORTED_CONFIG] != ""):
        print("merging some config from previous experiment...")
        merge_external_config(config)

    pickle.dump(config, open(os.path.join(config[lm_constants.CONF_TAGS.TRAIN_DIR], lm_constants.FILE_NAMES.CONFIG_PKL), "wb"))

    print("data read.")
    print(80 * "-")
    print(80 * "-")
    train(data, config)


if __name__ == "__main__":

    print(80 * "-")
    print("Eesen TF (Char RNN) library:", os.path.realpath(__file__))
    print("cwd:", os.getcwd(), "version:")
    try:
        print(sys.version)
        print(tf.__version__)
    except:
        print("tf.py: could not get version information for logging")
    print(80 * "-")

    main()
