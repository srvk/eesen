#!/usr/bin/env python


import sys, os, os.path
# -----------------------------------------------------------------
#   Main script
# -----------------------------------------------------------------


import pickle, re
import argparse
import numpy as np
from fileutils.kaldi import readScpInfo
from multiprocessing import Pool
from functools import partial
import tf


# -----------------------------------------------------------------
#   Function definitions
# -----------------------------------------------------------------

def load_labels(dir, files=['labels.tr', 'labels.cv']):
    """
    Load a set of labels in (local) Eesen format
    """
    mapLabel = lambda x: x - 1
    labels = {}
    m = 0

    for filename in files:
        with open(os.path.join(dir, filename), "r") as f:
            for line in f:
                tokens = line.strip().split()
                labels[tokens[0]] = [mapLabel(int(x)) for x in tokens[1:]]
                if max(labels[tokens[0]]) > m:
                    m = max(labels[tokens[0]])

    return m+2, labels

def get_batch_info(feat_info, label_dict, start, height):
    """
    feat_info: uttid, arkfile, offset, feat_len, feat_dim
    """
    max_label_len = 0
    xinfo, yidx, yval = [], [], []
    for i in range(height):
        uttid, arkfile, offset, feat_len, feat_dim = feat_info[start + i]
        label = label_dict[uttid]
        max_label_len = max(max_label_len, len(label))
        xinfo.append((arkfile, offset, feat_len, feat_dim))
        for j in range(len(label)):
            yidx.append([i, j])
            yval.append(label[j])
    
    yshape = np.array([height, max_label_len], dtype = np.int32)
    yidx = np.asarray(yidx, dtype = np.int32)
    yval = np.asarray(yval, dtype = np.int32)

    return xinfo, yidx, yval, yshape

def make_batches_info(feat_info, label_dict, batch_size):
    batch_x, batch_y = [], []
    L = len(feat_info)
    uttids = [x[0] for x in feat_info]
    for idx in range(0, L, batch_size):
        height = min(batch_size, L - idx)
        xinfo, yidx, yval, yshape = get_batch_info(feat_info, label_dict, idx, height)
        batch_x.append(xinfo)
        batch_y.append((yidx, yval, yshape))
    return batch_x, batch_y, uttids

def make_even_batches_info(feat_info, label_dict, batch_size):
    """
    CudnnLSTM requires batches of even sizes
    feat_info: uttid, arkfile, offset, feat_len, feat_dim
    """
    batch_x, batch_y = [], []
    L = len(feat_info)
    uttids = [x[0] for x in feat_info]

    idx = 0
    while idx < L:
        # find batch with even size, and with maximum size of batch_size
        j = idx + 1
        target_len = feat_info[idx][3]
        while j < min(idx + batch_size, L) and feat_info[j][3] == target_len: 
            j += 1
        xinfo, yidx, yval, yshape = get_batch_info(feat_info, label_dict, idx, j - idx)
        batch_x.append(xinfo)
        batch_y.append((yidx, yval, yshape))
        idx = j
    return batch_x, batch_y, uttids

def load_feat_info(args, part):
    data_dir = args.data_dir
    batch_size = args.batch_size
    nclass, label_dict = load_labels(data_dir)

    x, y = None, None
    features, labels, uttids = [], [], []
    filename = os.path.join(data_dir, "%s_local.scp" % (part))
    if args.debug:
        feat_info = readScpInfo(filename, 1000)
    else:
        feat_info = readScpInfo(filename)
    nfeat = feat_info[0][4]
    feat_info = sorted(feat_info, key = lambda x: x[3])
    if args.lstm_type == "cudnn":
        x, y, uttids = make_even_batches_info(feat_info, label_dict, batch_size)
    else:
        x, y, uttids = make_batches_info(feat_info, label_dict, batch_size)
    return nclass, nfeat, (x, y, uttids)

def load_prior(prior_path):
    prior = None
    with open(prior_path, "r") as f:
        for line in f:
            parts = map(int, line.split(" ")[1:-1])
            counts = parts[1:]
            counts.append(parts[0])
            cnt_sum = reduce(lambda x, y: x + y, counts)
            prior = [float(x) / cnt_sum for x in counts]
    return prior

def get_output_folder(parent_dir):
    exp_name = "dbr"
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, exp_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


# -----------------------------------------------------------------
#   Parser and Configuration
# -----------------------------------------------------------------

def mainParser():
    parser = argparse.ArgumentParser(description='Train TF-Eesen Model')

    parser.add_argument('--lstm_type', default="cudnn", help = "lstm type: cudnn, fuse, native")
    parser.add_argument('--store_model', default=False, dest='store_model', action='store_true', help='store model')
    parser.add_argument('--eval', default=False, dest='eval', action='store_true', help='enable evaluation mode')
    parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='enable debug mode')
    parser.add_argument('--eval_model', default = "", help = "model to load for evaluation")
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--data_dir', default = "/data/ASR5/fmetze/eesen/asr_egs/swbd/v1/tmp.LHhAHROFia/T22/", help = "data dir")
    parser.add_argument('--counts_file', default = "/data/ASR5/fmetze/eesen/asr_egs/swbd/v1/label.counts", help = "data dir")
    parser.add_argument('--nepoch', default = 30, type=int, help='#epoch')
    parser.add_argument('--lr_rate', default = 0.03, type=float, help='learning rate')
    parser.add_argument('--l2', default = 0.0, type=float, help='l2 normalization')
    parser.add_argument('--clip', default = 0.1, type=float, help='gradient clipping')
    parser.add_argument('--nlayer', default = 5, type=int, help='#layer')
    parser.add_argument('--nhidden', default = 320, type=int, help='dimesnion of hidden units in single direction')
    parser.add_argument('--nproj', default = 0, type=int, help='dimension of projection units in single direction, set to 0 if no projection needed')
    parser.add_argument('--half_period', default = 10, type=int, help='half period in epoch of learning rate')
    parser.add_argument('--temperature', default = 1, type=float, help='temperature used in softmax')
    parser.add_argument('--grad_opt', default = "grad", help='optimizer: grad, adam, momentum, cuddnn only work with grad')
    parser.add_argument('--train_dir', default = "log", help='log and model (output) dir')
    parser.add_argument('--continue_ckpt', default = "", help='continue this experiment')

    return parser

def readConfig(args):
    config_path = os.path.dirname(args.eval_model) + "/config.pkl"
    config = pickle.load(open(config_path, "rb"))
    config["temperature"] = args.temperature
    config["prior"] = load_prior(args.counts_file)
    config["lstm_type"] = "cudnn"
    if len(args.continue_ckpt):
        config["continue_ckpt"] = args.continue_ckpt
    for k, v in config.items():
        print(k, v)
    sys.stdout.flush()
    return config

def createConfig(args, nfeat, nclass, train_path):
    config = {
        "nfeat": nfeat,
        "nclass": nclass,
        "nepoch": args.nepoch,
        "lr_rate": args.lr_rate,
        "l2": args.l2,
        "clip": args.clip,
        "nlayer": args.nlayer,
        "nhidden": args.nhidden,
        "nproj": args.nproj,
        "lstm_type": args.lstm_type,
        "half_period": args.half_period,
        "grad_opt": args.grad_opt,
        "batch_size": args.batch_size,
        "train_path": train_path,
        "store_model": args.store_model,
        "random_seed": 15213
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


# -----------------------------------------------------------------
#   Main part
# -----------------------------------------------------------------

def main():
    parser = mainParser()
    args = parser.parse_args()

    nclass, nfeat, cv_data = load_feat_info(args, 'cv')
    if len(args.continue_ckpt):
        train_path = os.path.join(args.train_dir, os.path.dirname(os.path.dirname(args.continue_ckpt)))
    else:
        train_path = get_output_folder(args.train_dir)

    if args.eval:
        config = readConfig(args)
        config["temperature"] = args.temperature
        config["prior"] = load_prior(args.counts_file)
        tf.eval(cv_data, config, args.eval_model)
    else:
        config = createConfig(args, nfeat, nclass, train_path)

        _, _, tr_data = load_feat_info(args, 'train')
        cv_xinfo, cv_y, _ = cv_data
        tr_xinfo, tr_y, _ = tr_data
        data = (cv_xinfo, tr_xinfo, cv_y, tr_y)

        tf.train(data, config)


if __name__ == "__main__":
    main()

    #python3 /pylon2/ir3l68p/metze/eesen-tf/tf/tf1/main.py --store_model --nhidden 240 --nproj 0 --train_dir log --data_dir ../v1-30ms-arlberg/tmp.LHhAHROFia/T22 --nlayer 5
    # python /data/ASR5/fmetze/eesen-tf/tf/tf1/main.py --store_model --nhidden 240 --train_dir log --data_dir tmp.pgJ1QN1au3/T24/ --nlayer 5

    # I got this to work with build #485, http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-linux-gpu/
    # pip install http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-linux-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-linux/lastBuild/artifact/pip_test/whl/tensorflow_gpu-1.head-cp27-cp27mu-manylinux1_x86_64.whl
