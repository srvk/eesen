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
import os.path
import pickle
import sys
from eesen import Eesen

from reader.feats_reader import feats_reader_factory
from reader.labels_reader import labels_reader_factory


# -----------------------------------------------------------------
#   Parser and Configuration
# -----------------------------------------------------------------

def main_parser():
    parser = argparse.ArgumentParser(description='Train TF-Eesen Model')

    parser.add_argument('--lstm_type', default="cudnn", help = "lstm type: cudnn, fuse, native")
    parser.add_argument('--store_model', default=False, dest='store_model', action='store_true', help='store model')
    parser.add_argument('--eval', default=False, dest='eval', action='store_true', help='enable evaluation mode')
    parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='enable debug mode')
    parser.add_argument('--augment', default=False, dest='augment', action='store_true', help='do internal data augmentation')
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

    sat["num_sata_layers"]=args.num_sat_layers
    sat["adapt_stage"]=args.adapt_stage
    sat["adapt_org_path"]=args.adapt_org_path

    return sat

def create_online_argu_config(args):

    #TODO enter the values using a conf file or something
    online_augment_config={}
    online_augment_config["win"]=3
    online_augment_config["factor"]=3
    online_augment_config["roll"]=False

    return online_augment_config

def create_global_config(args):

    config = {
        #training conf
        "nepoch": args.nepoch,
        "clip": args.clip,
        "half_period": args.half_period,
        "half_rate": args.half_rate,
        "half_after": args.half_after,
        "lr_rate": args.lr_rate,
        "do_shuf": args.do_shuf,
        "grad_opt": args.grad_opt,
        "batch_size": args.batch_size,
        "random_seed": 15213,
        "debug": False,

        #architecture config
        "l2": args.l2,
        "nlayer": args.nlayer,
        "nhidden": args.nhidden,
        "nproj": args.nproj,
        "feat_proj": args.feat_proj,
        "batch_norm": args.batch_norm,
        "lstm_type": args.lstm_type,

        #directories
        "train_dir": args.train_dir,
        "store_model": args.store_model,
        "data_dir": args.data_dir,

        #augmentation
        "augment": args.augment,

        #adptation
        "adapt_stage": args.adapt_stage,
        "num_sat_layers": args.num_sat_layers,
        "adapt_org_path": args.adapt_org_path
    }

    config["sat"] = create_sat_config(args)
    config["online_augment"] = create_online_argu_config(args)

    if len(args.continue_ckpt):
        config["continue_ckpt"] = args.continue_ckpt


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
    tr_x = feats_reader_factory.create_reader('train','kaldi', config)


    #load training targets
    tr_y = labels_reader_factory.create_reader('train','txt', config, tr_x.get_batches_id())

    #create reader for labels
    cv_x = feats_reader_factory.create_reader('cv','kaldi', config)

    #create reader for labels
    cv_y = labels_reader_factory.create_reader('cv','txt', config, cv_x.get_batches_id())

    config["input_feat_dim"] = tr_x.get_num_dim()
    config["target_scheme"] = tr_y.get_target_scheme()

    if config["adapt_stage"] != 'unadapted':

        cv_sat = feats_reader_factory.create_reader('sat', 'kaldi', config, cv_x.get_batches_id())
        tr_sat = feats_reader_factory.create_reader('sat', 'kaldi', config, tr_x.get_batches_id())
        data = (cv_x, tr_x, cv_y, tr_y, cv_sat, tr_sat)

        config["sat_feat_dim"] = tr_sat.get_num_dim()

    else:
        data = (cv_x, tr_x, cv_y, tr_y)

    config["model_dir"] = os.path.join(config["train_dir"],"model")

    #create folder for storing experiment
    if not os.path.exists(config["model_dir"]):
        os.makedirs(config["model_dir"])
    pickle.dump(config, open(os.path.join(config["train_dir"],"config.pkl"), "wb"))

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

