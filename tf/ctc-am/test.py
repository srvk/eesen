#!/usr/bin/env python


"""
this project has been wrtien following this naming convention:

https://google.github.io/styleguide/pyguide.html#naming

"""

# -----------------------------------------------------------------
#   Main script
# -----------------------------------------------------------------

import os
from utils.fileutils import debug
import argparse
import constants
import sys
from eesen import Eesen
import pickle

from reader.sat_reader import sat_reader_factory
from reader.feats_reader import feats_reader_factory
from reader.labels_reader import labels_reader_factory

# -----------------------------------------------------------------
#   Function definitions
# -----------------------------------------------------------------

def load_prior(prior_path, nclass):
    priors = []

    print ("loading priors loaded for path: "+(prior_path))
    print ("with nclass: "+str(nclass))

    return priors

def generate_priors(data_dir, language_scheme):

    priors_scheme={}

    for language_id, target_scheme in language_scheme.iter():
        if(language_scheme == 1):
            language_dir=data_dir
        else:
            language_dir=os.path.join(data_dir,language_id)
        priors_scheme[language_dir]={}

        for target_id, nclass in target_scheme.iter():
            priors_path=os.path.join(language_dir,target_id+".priors")
            priors=load_prior(priors_path, nclass)
            priors_scheme[language_dir][target_id]=priors

    return priors_scheme

# -----------------------------------------------------------------
#   Parser and Configuration
# -----------------------------------------------------------------

def main_parser():
    #TODO add option to not get .arks and only ter
    parser = argparse.ArgumentParser(description='Test TF-Eesen Model')

    #io options
    parser.add_argument('--data_dir', default = "", help = "like data_dir for training script")
    parser.add_argument('--results_dir', default = "log", help='log and results dir')
    parser.add_argument('--save_every_batch', default = -1, type=int, help='log and results dir')

    #train configuration options
    parser.add_argument('--train_config', default = "", help = "model to load for evaluation")
    parser.add_argument('--trained_weights', default = "", help = "model to load for evaluation")

    #computing options
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--temperature', default = 1, type=float, help='temperature used in softmax')
    parser.add_argument('--use_priors', default = False, action='store_true', help='if --use_priors it will take ')
    parser.add_argument('--compute_ter', default = False, action='store_true', help='if --compute_ter the labels will be taken from data_dir (label_phn.test)and ter will be computed')

    return parser

def create_test_config(args, language_scheme):

    config_test = {}

    #io dir
    config_test[constants.CONFIG_TAGS_TEST.DATA_DIR] = args.data_dir
    config_test[constants.CONFIG_TAGS_TEST.RESULTS_DIR] = args.results_dir

    #train configuration
    config_test[constants.CONFIG_TAGS_TEST.TRAINED_WEIGHTS] = args.trained_weights
    config_test[constants.CONFIG_TAGS_TEST.TRAIN_CONFIG] = args.train_config

    #computing options
    config_test[constants.CONFIG_TAGS_TEST.TEMPERATURE] = args.temperature
    config_test[constants.CONFIG_TAGS_TEST.COMPUTE_TER] = args.compute_ter
    config_test[constants.CONFIG_TAGS_TEST.USE_PRIORS] = args.use_priors
    config_test[constants.CONFIG_TAGS_TEST.BATCH_SIZE] = args.batch_size
    if(config_test[constants.CONFIG_TAGS_TEST.USE_PRIORS]):
        config_test[constants.CONFIG_TAGS_TEST.PRIORS_SCHEME] = generate_priors(config_test[constants.CONFIG_TAGS_TEST.DATA_DIR], language_scheme)

    return config_test

def check_paths(args):

    #mandatory
    if not os.path.exists(args.train_config):
        print("Error: train_config does not correspond to a valid path: "+args.train_config)
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    if not os.path.exists(args.trained_weights+".index"):
        print("Error: eval_weights does not correspond to a valid path: "+args.eval_weights)
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    if not os.path.exists(args.data_dir):
        print("Error: test_data does not correspond to a valid path: "+args.data_dir)
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

def main():

    parser = main_parser()
    args = parser.parse_args()

    check_paths(args)

    config = pickle.load(open(args.train_config, "rb"))

    config_test = create_test_config(args, config[constants.CONF_TAGS.LANGUAGE_SCHEME])

    config.update(config_test)

    print(80 * "-")
    print("reading testing set")
    print(80 * "-")
    print(80 * "-")
    print("test_x:")
    print(80 * "-")

    #load training feats
    test_x = feats_reader_factory.create_reader('test', 'kaldi', config)

    print(80 * "-")

    if(config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]):
        print(80 * "-")
        print("test_y (for ter computation):")
        print(80 * "-")
        test_y = labels_reader_factory.create_reader('test', 'txt', config, test_x.get_batches_id())
    else:
        test_y = None

    if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
            != constants.SAT_SATGES.UNADAPTED:

        print("tr_sat:")
        print(80 * "-")
        tr_sat = sat_reader_factory.create_reader('kaldi', config, test_x.get_batches_id())
        print(80 * "-")

    else:
        tr_sat = None

    data=(test_x, test_y, tr_sat)

    eesen = Eesen()
    eesen.test(data, config)

if __name__ == "__main__":
    main()

