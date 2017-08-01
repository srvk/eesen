#!/usr/bin/env python

"""
this project has been wrtien following this naming convention:

https://google.github.io/styleguide/pyguide.html#naming
plus mutable vars in function (that are actually changes m_*)

"""

# -----------------------------------------------------------------
#   Main script
# -----------------------------------------------------------------

import argparse
import os
import constants
import os.path
import pickle
import sys
from eesen import Eesen
from utils.checkers import set_checker

from reader.feats_reader import feats_reader_factory
from reader.labels_reader import labels_reader_factory


# -----------------------------------------------------------------
#   Parser and Configuration
# -----------------------------------------------------------------

def main_parser():
    parser = argparse.ArgumentParser(description='Train TF-Eesen Model')

    parser.add_argument('--lstm_type', default="cudnn", help = "lstm type: cudnn, fuse, native")
    parser.add_argument('--store_model', default=False, dest='store_model', action='store_true', help='store model')
    parser.add_argument('--augment', default=False, dest='augment', action='store_true', help='do internal data augmentation')
    parser.add_argument('--eval', default=False, dest='eval', action='store_true', help='enable evaluation mode')
    parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='enable debug mode')
    parser.add_argument('--mix', default=False, dest='mix', action='store_true', help='do external data augmentation')
    parser.add_argument('--noshuffle', default=True, dest='do_shuf', action='store_false', help='do not shuffle training samples')
    parser.add_argument('--eval_model', default = "", help = "model to load for evaluation")
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--data_dir', default = "./tmp", help = "data dir")



    parser.add_argument('--use_kaldi_io', default=False, action='store_true', help='Do not use Kaldi IO library')

    #TODO this is infered with the name of the files included
    parser.add_argument('--mode', default='kaldi', action='store_true', help='IO data format. options [kaldi, hdf5]')

    parser.add_argument('--counts_file', default = "label.counts", help = "data dir")
    parser.add_argument('--nepoch', default = 30, type=int, help='#epoch')
    parser.add_argument('--lr_rate', default = 0.03, type=float, help='learning rate')
    parser.add_argument('--l2', default = 0.0, type=float, help='l2 normalization')
    parser.add_argument('--clip', default = 0.1, type=float, help='gradient clipping')
    parser.add_argument('--nlayer', default = 5, type=int, help='#layer')
    parser.add_argument('--nhidden', default = 320, type=int, help='dimension of hidden units in single direction')
    parser.add_argument('--nproj', default = 0, type=int, help='dimension of projection units, set to 0 if no projection needed')
    parser.add_argument('--feat_proj', default = 0, type=int, help='dimension of feature projection units, set to 0 if no projection needed')
    parser.add_argument('--batch_norm', default = False, dest='batch_norm', action='store_true', help='add batch normalization to FC layers')
    parser.add_argument('--half_period', default = 10, type=int, help='half period in epoch of learning rate')
    parser.add_argument('--half_rate', default = 0.5, type=float, help='halving factor')
    parser.add_argument('--half_after', default = 0, type=int, help='halving becomes enabled after this many epochs')
    parser.add_argument('--temperature', default = 1, type=float, help='temperature used in softmax')
    parser.add_argument('--grad_opt', default = "grad", help='optimizer: grad, adam, momentum, cuddnn only work with grad')
    parser.add_argument('--train_dir', default = "log", help='log and model (output) dir')
    parser.add_argument('--continue_ckpt', default = "", help='continue this experiment')

    #SAT arguments
    parser.add_argument('--num_sat_layers', default = 2, type=int, help='continue this experiment')
    parser.add_argument('--adapt_stage', default = 'unadapted', help='Stage of adatpation process. Three possibilities: train_adapt, fine_tune and unadapted. Default: unadapted')
    parser.add_argument('--adapt_org_path', default ="", help='path to the model that we will use as starter')

    return parser

def create_sat_config(args):

    sat={}

    sat[constants.NUM_SAT_LAYERS]=args.num_sat_layers
    sat[constants.ADAPT_STAGE]=args.adapt_stage
    sat[constants.ADAPT_ORG_PATH]=args.adapt_org_path

    return sat

def create_online_argu_config(args):

    #TODO enter the values using a conf file or something
    online_augment_config={}
    online_augment_config[constants.AUGMENTATION.WINDOW]=3
    online_augment_config[constants.AUGMENTATION.FACTOR]=3
    online_augment_config[constants.AUGMENTATION.ROLL]=False

    return online_augment_config

def create_global_config(args):

    config = {
        #training conf
        constants.NEPOCH: args.nepoch,
        constants.CLIP: args.clip,
        constants.HALF_PERIOD: args.half_period,
        constants.HALF_RATE: args.half_rate,
        constants.HALF_AFTER: args.half_after,
        constants.LR_RATE: args.lr_rate,
        constants.DO_SHUF: args.do_shuf,
        constants.GRAD_OPT: args.grad_opt,
        constants.BATCH_SIZE: args.batch_size,
        constants.RANDOM_SEED: 15213,
        constants.DEBUG: False,

        #architecture config
        constants.L2: args.l2,
        constants.NLAYERS: args.nlayer,
        constants.NHIDDEN: args.nhidden,

        #TODO this can be joined with one argument
        constants.NPROJ: args.nproj,
        constants.FEAT_PROJ: args.feat_proj,

        constants.BATCH_NORM: args.batch_norm,
        constants.LSTM_TYPE: args.lstm_type,

        #directories
        constants.TRAIN_DIR: args.train_dir,
        constants.STORE_MODEL: args.store_model,
        constants.DATA_DIR: args.data_dir,

        #adptation
        constants.ADAPT_STAGE: args.adapt_stage,
        constants.NUM_SAT_LAYERS: args.num_sat_layers,
        constants.ADAPT_ORG_PATH: args.adapt_org_path
    }

    config[constants.SAT] = create_sat_config(args)
    config[constants.ONLINE_AUGMENT_CONF] = create_online_argu_config(args)

    if len(args.continue_ckpt):
        config[constants.CONTINUE_CKPT] = args.continue_ckpt


    return config

# -----------------------------------------------------------------
#   Main part
# -----------------------------------------------------------------

def main():

    #TODO construct a factory/helper to load everything by just looking at data_dir

    parser = main_parser()
    args = parser.parse_args()
    config = create_global_config(args)


    #load training feats
    print("processing tr_x")
    tr_x = feats_reader_factory.create_reader('train', 'kaldi', config)

    #load training targets
    print("processing tr_y")
    tr_y = labels_reader_factory.create_reader('train', 'txt', config, tr_x.get_batches_id())

    #create reader for labels
    print("processing cv_x")
    cv_x = feats_reader_factory.create_reader('cv', 'kaldi', config)

    #create reader for labels
    print("processing cv_y")
    cv_y = labels_reader_factory.create_reader('cv', 'txt', config, cv_x.get_batches_id())

    #set config (targets could change)
    config[constants.INPUT_FEATS_DIM] = cv_x.get_num_dim()
    config[constants.TARGET_SCHEME] = cv_y.get_target_scheme()

    #checking that all sets are consitent
    set_checker.check_sets(cv_x, cv_y, tr_x, tr_y)


    if config[constants.ADAPT_STAGE] != constants.ADAPTATION_STAGES.UNADAPTED:

        cv_sat = feats_reader_factory.create_reader('sat', 'kaldi', config, cv_x.get_batches_id())
        tr_sat = feats_reader_factory.create_reader('sat', 'kaldi', config, tr_x.get_batches_id())
        data = (cv_x, tr_x, cv_y, tr_y, cv_sat, tr_sat)
        config[constants.SAT_FEAT_DIM] = tr_sat.get_num_dim()
    else:

        data = (cv_x, tr_x, cv_y, tr_y)

    config[constants.MODEL_DIR] = os.path.join(config[constants.TRAIN_DIR], "model")

    #create folder for storing experiment
    if not os.path.exists(config[constants.MODEL_DIR]):
        os.makedirs(config[constants.MODEL_DIR])
    pickle.dump(config, open(os.path.join(config[constants.TRAIN_DIR], "config.pkl"), "wb"))

    #log of expriment configuration
    sys.stdout.flush()
    print(80 * "-")
    print("experiment configuration:")
    print(80 * "-")
    for k, v in config.items():
        print(k, v)

    #start the acutal training

    eesen=Eesen()

    eesen.train(data, config)

if __name__ == "__main__":
    main()

