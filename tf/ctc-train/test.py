#!/usr/bin/env python


"""
this project has been wrtien following this naming convention:

https://google.github.io/styleguide/pyguide.html#naming

"""

# -----------------------------------------------------------------
#   Main script
# -----------------------------------------------------------------

import argparse
import sys
from functools import reduce

import constants
from eesen import Eesen
from reader.sat_reader import feats_reader_factory


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


def add_args_test_config(args, config):

    config["temperature"] = args.temperature
    config["prior"] = load_prior(args.counts_file, config["nclass"])
    config["train_path"] = args.data_dir
    config["adapt_stage"] = 'unadapted'

    return config

# -----------------------------------------------------------------
#   Parser and Configuration
# -----------------------------------------------------------------

def main_parser():
    parser = argparse.ArgumentParser(description='Test TF-Eesen Model')

    parser.add_argument('--eval_model', default = "", help = "model to load for evaluation")
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--test_data', default = "./tmp", help = "data dir")
    parser.add_argument('--counts_file', default = "label.counts", help = "data dir")
    parser.add_argument('--temperature', default = 1, type=float, help='temperature used in softmax')
    parser.add_argument('--train_dir', default = "log", help='log and model (output) dir')

    return parser

def main():

    parser = main_parser()
    args = parser.parse_args()


    config
    test_x = feats_reader_factory.create_reader('cv', 'kaldi', config[constants.DATA_DIR], args.lstm_type, online_augment_config, args.batch_size)

    config = add_args_test_config(args, config)

    if(len(nclass) != len(config["nclass"])):
        print("Error. Number of labels provided not correct. "+str(len(nclass))+" provided "+str(len(config["nclass"]))+" needed")
        sys.exit()

    eesen = Eesen()
    eesen.test(data, config)

if __name__ == "__main__":
    main()

