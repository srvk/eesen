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
from reader.feats_reader import feats_reader_factory

# -----------------------------------------------------------------
#   Function definitions
# -----------------------------------------------------------------

def load_prior(prior_paths, nclass):
    priors = []

    if(len(prior_paths.split(":")) != len(nclass)):
        print("Error: Wrong number or prior paths given. Only "+str(len(prior_paths.split(":")))+" paths were given and "+str(len(nclass))+" needed")
        sys.exit()

    for idx, prior_path in enumerate(prior_paths.split(":")):
        with open(prior_path, "r") as f:
            for line in f.readlines():
                parts = [int(x) for x in line.strip().split(" ")[1:-1]]
                counts = parts[1:]
                counts.append(parts[0])
                cnt_sum = reduce(lambda x, y: x + y, counts)
                prior = [float(x) / cnt_sum for x in counts]
            if(len(prior) != nclass[idx]):
                print("Error: Wrong number of elements given in prior file number "+str(idx)+" it has "+str(len(prior))+" and it should have "+str(nclass[idx]))

        priors.append(prior)
    return priors


# -----------------------------------------------------------------
#   Parser and Configuration
# -----------------------------------------------------------------

def main_parser():
    parser = argparse.ArgumentParser(description='Test TF-Eesen Model')

    parser.add_argument('--eval_config', default = "", help = "model to load for evaluation")
    parser.add_argument('--eval_weights', default = "", help = "model to load for evaluation")

    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--temperature', default = 1, type=float, help='temperature used in softmax')
    parser.add_argument('--results_dir', default = "log", help='log and model (output) dir')

    parser.add_argument('--use_priors', default = "", help='path to priors file (if more than one use :)')
    parser.add_argument('--compute_ter', default = "", help='compute ter using labels file. --compute_ter path/to/labels.test')
    parser.add_argument('--test_scp_file', default = "", help = "data dir")

    return parser

def check_priors(priors, language_scheme):

    #TODO here we need to code a sanity check of priors vs language scheme (from the trained model)
    print("checking priors")
    return True

def generate_priors(priors):

    #TODO here we will generate a dict of priors according the input
    print("checking priors")
    return True

def generate_test_labels(test_):

def create_test_config(args, language_scheme):

    config = {}
    config[constants.CONFIG_TAGS_TEST.TEMPERATURE] = args.temperature

    if(args.use_priors != ""):
        check_priors(args.use_priors, language_scheme)
        config[constants.CONFIG_TAGS_TEST.PRIORS] = generate_priors(args.use_priors)

    if(args.use_priors != ""):
        config[constants.CONFIG_TAGS_TEST.RESULTS_DIR] = args.results_dir

    config[constants.CONFIG_TAGS_TEST.WEIGHT_FILE] = args.eval_config
    config[constants.CONFIG_TAGS_TEST.EVAL_CONFIG_FILE] = args.eval_config
    config[constants.CONFIG_TAGS_TEST.BATCH_SIZE] = args.batch_size
    config[constants.CONFIG_TAGS_TEST.TEMPERATURE] = args.temperature
    config[constants.CONFIG_TAGS_TEST.RESULTS_DIR] = args.results_dir


    parser.add_argument('--compute_ter', default = "", help='compute ter using labels file. --compute_ter path/to/labels.test')
    parser.add_argument('--test_scp_file', default = "", help = "data dir")
    return config

def check_paths(args):

    #mandatory
    if not os.path.exists(args.eval_config):
        print("Error: eval_config does not correspond to a valid path: "+args.import_config)
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    if not os.path.exists(args.eval_weights):
        print("Error: eval_weights does not correspond to a valid path: "+args.import_config)
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    if not os.path.exists(args.test_data):
        print("Error: test_data does not correspond to a valid path: "+args.import_config)
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    #optionals
    if(args.use_priors != ""):
        if not os.path.exists(args.use_priors):
            print("Error: path_config does not correspond to a valid path: "+args.import_config)
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()

    if(args.compute_ter != ""):
        if not os.path.exists(args.compute_ter):
            print("Error: path_config does not correspond to a valid path: "+args.import_config)
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()
def main():

    parser = main_parser()
    args = parser.parse_args()

    check_paths(args)

    config = create_test_config(args.eval_config)

    config[]

    config_imported = pickle.load(open(args.import_config, "rb"))

    config.update(config_imported)

    config[constants.CONF_TAGS.SAT_CONF] = create_sat_config(args, config_imported)

    config[constants.CONF_TAGS.ONLINE_AUGMENT_CONF] = create_online_arg_config()
    print(80 * "-")
    print("reading testing set")
    print(80 * "-")
    print(80 * "-")
    print("tr_x:")
    print(80 * "-")

    #load training feats
    tr_x = feats_reader_factory.create_reader('train', 'kaldi', config)


    sys.exit()
    if(len(nclass) != len(config["nclass"])):
        print("Error. Number of labels provided not correct. "+str(len(nclass))+" provided "+str(len(config["nclass"]))+" needed")
        sys.exit()

    eesen = Eesen()
    eesen.test(data, config)

if __name__ == "__main__":
    main()

