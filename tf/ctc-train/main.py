#!/usr/bin/env python

#
# to debug:
#   import pdb; pdb.set_trace()
#

# -----------------------------------------------------------------
#   Main script
# -----------------------------------------------------------------

import sys, os, os.path
import pickle, re
import argparse
import numpy as np
from fileutils.kaldi import readScpInfo
from multiprocessing import Pool
from functools import partial, reduce
import tf
#, tensorflow
try:
    from h5_Reader import H5Dataset
except:
    pass

#import pdb; pdb.set_trace()


# -----------------------------------------------------------------
#   Function definitions
# -----------------------------------------------------------------

def load_labels(dir, files=['labels.tr', 'labels.cv'], nclass=0):
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
                try:
                    # this can be empty
                    if max(labels[tokens[0]]) > m:
                        m = max(labels[tokens[0]])
                except:
                    pass

    # sanity check - did we provide a value, and the actual is different?
    if nclass > 0 and m+2 != nclass:
        print("Warning: provided nclass=", nclass, " while observed nclass=", m+2, "in", dir, files)
        m = nclass-2
    else:
        print("Load labels in", dir, files, "nclass=", m+2)
    return m+2, labels

def get_batch_info(feat_info, label_dicts, start, height):
    """
    feat_info: uttid, arkfile, offset, feat_len, feat_dim
    """
    max_label_len = []

    xinfo, yidx, yval, uttid = [], [], [], []

    for count_label in range(len(label_dicts)):
        yidx.append([])
        yval.append([])
        max_label_len.append(0)

    for i in range(height):
        uttid, arkfile, offset, feat_len, feat_dim, a_info = feat_info[start + i]
        # uttid_aux, arkfile, offset, feat_len, feat_dim, a_info = feat_info[start + i]
        # xinfo.append((arkfile, offset, feat_len, feat_dim, a_info))
        # uttid.append(uttid_aux)

        for count_label, label_dict in enumerate(label_dicts):
            label = label_dict[uttid_aux]
            max_label_len[count_label] = max(max_label_len[count_label], len(label))
            for j in range(len(label)):
                yidx[count_label].append([i, j])
                yval[count_label].append(label[j])
        xinfo.append((arkfile, offset, feat_len, feat_dim, a_info))

    yshape_r=[]
    yidx_r=[]
    yval_r=[]

    for count_label, _ in enumerate(label_dicts):
        yshape_r.append(np.array([height, max_label_len[count_label]], dtype = np.int32))
        yidx_r.append(np.asarray(yidx[count_label], dtype = np.int32))
        yval_r.append(np.asarray(yval[count_label], dtype = np.int32))

    return xinfo, yidx_r, yval_r, yshape_r, uttid

def make_batches_info(feat_info, label_dicts, batch_size):
    batch_x, batch_y, uttids = [], [], []
    L = len(feat_info)

    for idx in range(0, L, batch_size):
        height = min(batch_size, L - idx)
        xinfo, yidx, yval, yshape, uttid = get_batch_info(feat_info, label_dicts, idx, height)
        batch_x.append(xinfo)
        uttids.append(uttid)

        batch_y_element=[]
        for idx, _ in enumerate(label_dicts):
            element=((yidx[idx], yval[idx], yshape[idx]))
            batch_y.append(element)

        batch_y.append(batch_y_element)

    return batch_x, batch_y, uttids

def make_even_batches_info(feat_info, label_dicts, batch_size):
    """
    CudnnLSTM requires batches of even sizes
    feat_info: uttid, arkfile, offset, feat_len, feat_dim
    """
    batch_x, batch_y, uttids = [], [], []
    L = len(feat_info)
    uttids = [x[0] for x in feat_info]
    idx, c = 0, 0

    while idx < L:
        # find batch with even size, and with maximum size of batch_size
        j = idx + 1
        target_len = feat_info[idx][3]
        while j < min(idx + batch_size, L) and feat_info[j][3] == target_len:
            j += 1
        xinfo, yidx, yval, yshape, uttid = get_batch_info(feat_info, label_dicts, idx, j - idx)
        batch_x.append(xinfo)
        uttids.append(uttid)
        batch_y_element=[]

        for idx, _ in enumerate(label_dicts):
            element = ((yidx[idx], yval[idx], yshape[idx]))
            #elementensor = tensorflow.SparseTensor(tensorflow.cast(yidx[idx], tensorflow.int64), tensorflow.cast(yval[idx], tensorflow.int32), tensorflow.cast(yshape[idx], tensorflow.int64))
            #b = tensorflow.sparse_to_dense(elementensor.indices,elementensor.dense_shape,elementensor.values,validate_indices=True)
            #d = tensorflow.sparse_transpose(elementensor,[1,0,2])
            batch_y_element.append(element)

        #print("begin validate")
        #y = batch_y_element[0]
        #a = tensorflow.SparseTensor(y[0],y[1],y[2])
        #b = tensorflow.sparse_to_dense(a.indices,a.dense_shape,a.values,validate_indices=True)

        batch_y.append(batch_y_element)
        idx = j
        #print("BATCH-INFO",c)
        #print("X",xinfo)
        #print("Y",batch_y_element)
        c+=1
    return batch_x, batch_y, uttids

def load_feat_info(args, part, nclass=0):
    nclass_all=[]
    label_dicts=[]

    data_dir = args.data_dir
    batch_size = args.batch_size
    nclass, label_dict = load_labels(data_dir, nclass=nclass)

    nclass_all.append(nclass)
    label_dicts.append(label_dict)

    if(args.extra_labels):
        extra_labels=args.extra_labels
        extra_dirs=extra_labels.split(':')
        for extra_dir in extra_dirs:
            print("Extra dir:", extra_dir)
            # Fixme - 47 or 388 for chars or bpe300 during testing
            nclass, label_dict = load_labels(extra_dir)
            nclass_all.append(nclass)
            label_dicts.append(label_dict)

    x, y = None, None
    features, labels, uttids = [], [], []
    filename = os.path.join(data_dir, "%s_local.scp" % (part))
    if args.debug:
        feat_info = readScpInfo(filename, 1000)
    else:
        feat_info = readScpInfo(filename)
    nfeat = feat_info[0][4]

    if args.augment:
        if args.h5_augment_size is None:
            factor = 3
        else:
            factor = args.h5_augment_size
        if args.h5_input_dim is None:
            win = 3
        else:
            win = args.h5_input_dim
        nfeat *= win
        print("Augmenting data x", factor, "from", filename, "#features=", nfeat, "#classes=", nclass_all, "#examples=", len(feat_info))
        feat_info = [(tup[0], tup[1], tup[2], (tup[3]+factor-1-shift) // factor, win*tup[4], (shift, factor, win))
                     for shift in range(factor) for tup in feat_info]
    else:
        feat_info = [tup+(None,) for tup in feat_info]
    feat_info = sorted(feat_info, key = lambda x: x[3])

    if args.lstm_type == "cudnn":
        x, y, uttids = make_even_batches_info(feat_info, label_dicts, batch_size)
    else:
        x, y, uttids = make_batches_info(feat_info, label_dicts, batch_size)

    return nclass_all, nfeat, (x, y, uttids)

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
    parser.add_argument('--augment', default=False, dest='augment', action='store_true', help='do internal data augmentation')
    parser.add_argument('--mix', default=False, dest='mix', action='store_true', help='do external data augmentation')
    parser.add_argument('--noshuffle', default=True, dest='do_shuf', action='store_false', help='do not shuffle training samples')
    parser.add_argument('--eval_model', default = "", help = "model to load for evaluation")
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--data_dir', default = "./tmp", help = "data dir")
    parser.add_argument('--use_kaldi_io', default=False, action='store_true', help='Do not use Kaldi IO library')

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
    parser.add_argument('--nclass', default = 0, type=int, help='dimensionality, if not auto-detecable from labels')

    #SAT arguments
    parser.add_argument('--adapt_path', default = "", help='root path where all the adpatation vectors are')
    parser.add_argument('--adapt_reader_type', default = 'csv_matrix_folder_first', help='fromat of the speaker. Thee possibilities: kaldi_file, csv_folder, csv_matrix_folder_first, csv_matrix_folder_last')
    parser.add_argument('--adapt_stage', default = 'unadapted', help='Stage of adatpation process. Three possibilities: train_adapt, fine_tune and unadapted. Default: unadapted')
    parser.add_argument('--adapt_dim', default = 1024, type=int,  help='continue this experiment')
    parser.add_argument('--num_sat_layers', default = 2, type=int, help='continue this experiment')
    parser.add_argument('--adapt_org_path', default ="", help='path to the model that we will use as starter')

    return parser

def readConfig(args):
    config_path = os.path.dirname(args.eval_model) + "/config.pkl"

    config = pickle.load(open(config_path, "rb"))
    config["temperature"] = args.temperature
    config["use_kaldi_io"] = args.use_kaldi_io
    config["augment"] = args.augment
    config["mix"] = args.mix
    config["batch_norm"] = args.batch_norm
    if len(args.continue_ckpt):
        config["continue_ckpt"] = args.continue_ckpt
    for k, v in config.items():
        print(k, v)
    sys.stdout.flush()
    return config

def createConfig(args, nfeat, nclass, train_path):

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
        "nclass": nclass,
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

# -----------------------------------------------------------------
#   Main part
# -----------------------------------------------------------------

def main():
    parser = mainParser()
    args = parser.parse_args()

    if args.h5_mode:
        valid_dataset = H5Dataset(args, input_file=args.h5_valid)
        valid_dataset.readData()
        valid_dataset.make_even_batches(args.batch_size)
        nclass, nfeat, cv_data = valid_dataset.load_feat_info()
    else:
        valid_dataset = None
        # can specify the value of nclass here (373,688)
        nclass, nfeat, cv_data = load_feat_info(args, 'cv', nclass=args.nclass)
    train_path = get_output_folder(args.train_dir)

    if args.eval:
        config = readConfig(args)
        if(len(nclass) != len(config["nclass"])):
            print("Error. Number of labels provided not correct. "+str(len(nclass))+" provided "+str(len(config["nclass"]))+" needed")
            sys.exit()
        config["temperature"] = args.temperature
        config["prior"] = load_prior(args.counts_file, config["nclass"])
        config["train_path"] = args.data_dir
        config["adapt_stage"] = 'unadapted'
        tf.eval(cv_data, config, args.eval_model)

    else:
        config = createConfig(args, nfeat, nclass, train_path)

        train_dataset = None
        if args.h5_mode:
            train_dataset = H5Dataset(args, input_file=args.h5_train)
            train_dataset.readData()
            train_dataset.make_even_batches(args.batch_size)
            _, _, tr_data = train_dataset.load_feat_info()
            tr_xinfo, tr_y, _ = tr_data

        elif config["mix"]:
            print('Fixing data dir for mixing')
            tr_xinfo = {}; tr_y = {}
            p = args.data_dir
            for epoch in range(0,args.nepoch):
                args.data_dir = p+"/X"+str(epoch)
                _, _, tr_data = load_feat_info(args, 'train')
                tr_xinfo[epoch], tr_y[epoch], _ = tr_data
            args.data_dir = p

        else:
            _, _, tr_data = load_feat_info(args, 'train')
            tr_xinfo, tr_y, tr_id = tr_data
            print(tr_id[0])
            print(len(tr_id))

        # this needs to be cleaned up for H5 support
        cv_xinfo, cv_y, cv_id = cv_data

        data = (cv_xinfo, tr_xinfo, cv_y, tr_y, valid_dataset, train_dataset, cv_id, tr_id)

        tf.train(data, config)


if __name__ == "__main__":
    main()

    #python3 /pylon2/ir3l68p/metze/eesen-tf/tf/tf1/main.py --store_model --nhidden 240 --nproj 0 --train_dir log --data_dir ../v1-30ms-arlberg/tmp.LHhAHROFia/T22 --nlayer 5
    # python /data/ASR5/fmetze/eesen-tf/tf/tf1/main.py --store_model --nhidden 240 --train_dir log --data_dir tmp.pgJ1QN1au3/T24/ --nlayer 5

    # I got this to work with build #485, http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-linux-gpu/
    # pip install http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-linux-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-linux/lastBuild/artifact/pip_test/whl/tensorflow_gpu-1.head-cp27-cp27mu-manylinux1_x86_64.whl
