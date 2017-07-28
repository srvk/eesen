#!/usr/bin/env python


"""
this project has been wrtien following this naming convention:

https://google.github.io/styleguide/pyguide.html#naming

"""

# -----------------------------------------------------------------
#   Main script
# -----------------------------------------------------------------

import argparse
import os
import os.path
import pickle
import sys
from functools import reduce

from reader.feats_reader import feats_reader_factory
from reader.labels_reader import labels_reader_factory


# -----------------------------------------------------------------
#   Function definitions
# -----------------------------------------------------------------

#TODO add this as a reader
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

    parser.add_argument('--mode', default='kaldi', action='store_true', help='IO data format. options [kaldi, hdf5]')

    parser.add_argument('--h5_mode', default=False, action='store_true', help='Enable reading HDF5 files')
    parser.add_argument('--h5_train', help='h5 train data', type=str, default=None)
    parser.add_argument('--h5_valid', help='h5 valid data', type=str, default=None)
    parser.add_argument('--h5_input_dim', default=None, type=int, help='Size of input features')
    parser.add_argument('--h5_input_feat', default=None, type=str, help='Name of input feature(s)')
    parser.add_argument('--h5_target', default=None, type=str, help='Name of target feature')
    parser.add_argument('--h5_labels', default=None, help='JSON-file containing all characters for prediction')
    parser.add_argument('--h5_uttSkip', default=None, type=str, help='Skip these utts')
    parser.add_argument('--h5_filter', default=None, type=str, help='evaluate only speakers containing this string')
    parser.add_argument('--h5_spkList', default=None, type=str, help='File(s) containing list of speakers')
    parser.add_argument('--h5_mapping', default=None, type=str, help='Token mapping file(s)')
    parser.add_argument('--h5_augment_feat', default=None, type=str, help='Name of feature for augmentation')
    parser.add_argument('--h5_augment_size', default=None, type=int, help='Size of feature for augmentation')

    #TODO include that in the normal labels with :
    parser.add_argument('--extra_labels', help = "extra labels (e.g. phn set when your main target are char) IMPORTANT: the feature files are the master")

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
    parser.add_argument('--adapt_path', default = "", help='root path where all the adpatation vectors are')
    parser.add_argument('--adapt_reader_type', default = 'csv_matrix_folder_first', help='fromat of the speaker. Thee possibilities: kaldi_file, csv_folder, csv_matrix_folder_first, csv_matrix_folder_last')
    parser.add_argument('--adapt_stage', default = 'unadapted', help='Stage of adatpation process. Three possibilities: train_adapt, fine_tune and unadapted. Default: unadapted')
    parser.add_argument('--adapt_dim', default = 1024, type=int,  help='continue this experiment')
    parser.add_argument('--num_sat_layers', default = 2, type=int, help='continue this experiment')
    parser.add_argument('--adapt_org_path', default ="", help='path to the model that we will use as starter')

    return parser

def create_config_eval(args):

def create_config_train(args, nfeat, target_scheme, train_path):

    if(args.adapt_stage == "train_adapt" or args.adapt_stage == "fine_tune"):
        if (args.adapt_reader_type == 'kaldi_file' or args.adapt_reader_type == 'csv_folder' or args.adapt_reader_type == 'csv_matrix_folder_first'):
            print("Reader type of adapation file has a wrong option: "+args.adapt_reader_type)
            print("Valid options: kaldi_file, csv_folder, csv_matrix_folder_first, csv_matrix_folder_last")
            sys.exit()

        if not (args.adapt_stage == 'train_adapt' or args.adapt_stage == 'fine_tune'):
            print("Adaptation satge has a wrong option: "+args.adapt_stage)
            sys.exit()

    else:
        args.adapt_path=''
        args.adapt_stage='unadapted'
        args.adapt_dim=0
        args.num_sat_layers=0
        args.adapt_org_path=""

    config = {
        "nfeat": nfeat,
        "target_scheme": target_scheme,
        "nepoch": args.nepoch,
        "lr_rate": args.lr_rate,
        "l2": args.l2,
        "clip": args.clip,
        "nlayer": args.nlayer,
        "nhidden": args.nhidden,
        "nproj": args.nproj,
        "feat_proj": args.feat_proj,
        "batch_norm": args.batch_norm,
        "do_shuf": args.do_shuf,
        "lstm_type": args.lstm_type,
        "half_period": args.half_period,
        "half_rate": args.half_rate,
        "half_after": args.half_after,
        "grad_opt": args.grad_opt,
        "batch_size": args.batch_size,
        "train_path": train_path,
        "store_model": args.store_model,
        "random_seed": 15213,
        "temperature": args.temperature,
        "h5_mode": args.h5_mode,
        "use_kaldi_io": args.use_kaldi_io,
        "mix": args.mix,
        "augment": args.augment,

        "adapt_path": args.adapt_path,
        "adapt_reader_type": args.adapt_reader_type,
        "adapt_stage": args.adapt_stage,
        "adapt_dim": args.adapt_dim,
        "num_sat_layers": args.num_sat_layers,
        "adapt_org_path": args.adapt_org_path
    }


    if len(args.continue_ckpt):
        config["continue_ckpt"] = args.continue_ckpt
    for k, v in config.items():
        print(k, v)
    sys.stdout.flush()
    model_dir = config["train_path"] + "/model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    pickle.dump(config, open(config["train_path"] + "/model/config.pkl", "wb"))

    return config




def create_global_config(args):

    config = {
        #test_config
        "eval_model": args.eval_model,
        "config_path": os.path.join(args.eval_model,"config.pkl"),
        "temperature": args.temperature,
    }

    return config

def main():

    parser = main_parser()
    args = parser.parse_args()

    #TODO this will be arranged in two scripts (train and test)
    #load validation reader (cv can be used as test set when eval)
    online_augment_config={}
    online_augment_config["win"]=3
    online_augment_config["factor"]=3
    online_augment_config["roll"]=None

    cv_x = feats_reader_factory.create_reader('cv','kaldi', args.data_dir, args.lstm_type,online_augment_config, args.batch_size)

    #create reader for labels
    cv_y = labels_reader_factory.create_reader('cv','txt', args, cv_x.get_batches_id())

    if args.eval:
        config = create_config_eval(args)
        if(len(nclass) != len(config["nclass"])):
            print("Error. Number of labels provided not correct. "+str(len(nclass))+" provided "+str(len(config["nclass"]))+" needed")
            sys.exit()
        config["temperature"] = args.temperature
        config["prior"] = LoadPrior(args.counts_file, config["nclass"])
        config["train_path"] = args.data_dir
        config["adapt_stage"] = 'unadapted'
        eesen_20170714.eval(cv_data, config, args.eval_model)

    else:

        #TODO check
        #load training feats
        tr_x = feats_reader_factory.create_reader('train','kaldi', args.data_dir, args.lstm_type,online_augment_config, args.batch_size)

        #load training targets
        tr_y = labels_reader_factory.create_reader('train','txt', args, tr_x.get_batches_id())

        #TODO when two scripts created. we should take a look to create_config_train
        #TODO this tr_y.get_num_dim() will have to have a dic of dic with all the output structure TO KNOW which language are we training
        config = create_config_train(args, tr_x.get_num_dim(), tr_y.get_target_scheme(), args.data_dir)

        #if we need adaptation
        if config["adapt_stage"] != 'unadapted':
            tr_x = reader_factory.create_reader('sat', 'feats', 'kaldi', args)
            sat=create_reader(args, 'sat', 'kaldi')
            data = (cv_x, tr_x, sat, cv_y, tr_y)
        else:
            data = (cv_x, tr_x, None, cv_y, tr_y)

        eesen_20170714.train(data, config)

if __name__ == "__main__":
    main()

