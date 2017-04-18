import sys, os, os.path, time
import numpy as np
from fileutils.kaldi import readArk
import tf

def load_haitian_label(DATA_DIR):
    labels = {}
    # Maps 2 to 0 and 7~40 to 1~34
    mapLabel = lambda x: 0 if x == 2 else x - 6
    for filename in ["labels.tr", "labels.cv"]:
        with open(os.path.join(DATA_DIR, filename), "r") as f:
            for line in f:
                tokens = line.strip().split()
                labels[tokens[0]] = [mapLabel(int(x)) for x in tokens[1:]]
    return 36, labels

def load_swbd_label(DATA_DIR):
    labels = {}
    # Maps 1 to 45 to 0 to 44 
    mapLabel = lambda x: x - 1
    for filename in ["labels.tr", "labels.cv"]:
        with open(os.path.join(DATA_DIR, filename), "r") as f:
            for line in f:
                tokens = line.strip().split()
                labels[tokens[0]] = [mapLabel(int(x)) for x in tokens[1:]]
    return 46, labels

def get_batch(feats, labels, start, height):
    max_feat_len = max(len(feats[start+i]) for i in range(height))
    max_label_len = max(len(labels[start+i]) for i in range(height))
    tmpx = np.zeros((height, max_feat_len, feats[start].shape[-1]), np.float32)
    yshape = np.array([height, max_label_len], dtype = np.int32)
    yidx, yval = [], []
    for i in range(height):
        feat, label = feats[start+i], labels[start+i]
        tmpx[i, :len(feat), :] = feat
        for j in range(len(label)):
            yidx.append([i, j])
            yval.append(label[j])

    yidx = np.asarray(yidx, dtype = np.int32)
    yval = np.asarray(yval, dtype = np.int32)
    return tmpx, yidx, yval, yshape

def make_batches(feats, labels, BATCH_SIZE):
    batch_x, batch_y = [], []
    # L = min(len(feats), 10 * BATCH_SIZE)
    L = len(feats)
    feats, labels = zip(*sorted(zip(feats, labels), key = lambda x: x[0].shape[0]))
    for start in range(0, L, BATCH_SIZE):
        height = min(BATCH_SIZE, L - start)
        tmpx, yidx, yval, yshape = get_batch(feats, labels, start, height)
        batch_x.append(tmpx)
        batch_y.append((yidx, yval, yshape))
    return batch_x, batch_y

def make_even_batches(feats, labels, BATCH_SIZE):
    """
    CudnnLSTM requires batches of even sizes
    """
    batch_x, batch_y = [], []
    # L = min(len(feats), 10 * BATCH_SIZE)
    L = len(feats)
    feats, labels = zip(*sorted(zip(feats, labels), key = lambda x: x[0].shape[0]))
    idx = 0
    while idx < L:
        # find batch with even size, and with maximum size of BATCH_SIZE
        j = idx + 1
        target_len = feats[idx].shape[0]
        while j < min(idx + BATCH_SIZE, L) and feats[j].shape[0] == target_len: 
            j += 1
        tmpx, yidx, yval, yshape = get_batch(feats, labels, idx, j - idx)
        batch_x.append(tmpx)
        batch_y.append((yidx, yval, yshape))
        idx = j
    return batch_x, batch_y

def load_feat(use_cudnn, DATA_DIR, BATCH_SIZE):
    # nclass, label_dict = load_label()
    nclass, label_dict = load_swbd_label(DATA_DIR)

    cv_x, cv_y, tr_x, tr_y = [], [], [], []
    for part in ["cv", "train"]:
    # for part in ["cv"]:
        features = []
        labels = []

        for i in [0, 1, 2]:
        # for i in [0]:
            filename = os.path.join(DATA_DIR, "%s%d.ark" % (part, i))
            part_features, uttids = readArk(filename)
            part_labels = [label_dict["%dx%s" % (i, x)] for x in uttids]
            features += part_features
            labels += part_labels

        if part == "cv":
            if use_cudnn: 
                cv_x, cv_y = make_even_batches(features, labels, BATCH_SIZE)
            else:
                cv_x, cv_y = make_batches(features, labels, BATCH_SIZE)
        elif part == "train":
            if use_cudnn: 
                tr_x, tr_y = make_even_batches(features, labels, BATCH_SIZE)
            else:
                tr_x, tr_y = make_batches(features, labels, BATCH_SIZE)
    print("Finish loading data")
    return nclass, (cv_x, tr_x, cv_y, tr_y)
    # return cv_x, cv_x, cv_y, cv_y

if __name__ == "__main__":
    use_cudnn = True
    BATCH_SIZE = 32
    DATA_DIR = "/data/ASR5/fmetze/eesen/asr_egs/swbd/v1/tmp.LHhAHROFia/T22/"
    nclass, data = load_feat(use_cudnn, DATA_DIR, BATCH_SIZE)
    nfeat = data[0][0].shape[-1]
    config = {
        "nfeat": nfeat, 
        "nclass": nclass, 
        "nepoch": 30,
        "lr_rate": 3e-2,
        "l2": 0,
        "clip": 1e-1,
        "nlayer": 5,
        "nhidden": 320,
        "nproj": 160,
	"cudnn": use_cudnn,
        "half_period": 30,
        "grad_opt": "grad",
        "batch_size": BATCH_SIZE
    }
    print(config)
    tf.train(data, config)
