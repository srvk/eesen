import random, sys
import numpy as np
from fileutils.kaldi import writeArk, readMatrixByOffset

def read_batch(xinfo):
    """
    xinfo: arkfile, offset, feat_len, feat_dim, augment
    """
    height = len(xinfo)
    max_feat_len = max(x[2] for x in xinfo)
    tmpx = None
    i = 0

    #print("Batch",height,xinfo[0][2],max_feat_len)
    for arkfile, offset, feat_len, feat_dim, augment in xinfo:
        feat = readMatrixByOffset(arkfile, offset)
        if not augment is None:
            # data augmentation
            #shift = augment[0]
            #stride = augment[1]
            stride=3
            shift=augment
            if stride is 3:
                feat = np.concatenate((np.roll(feat,1,axis=0), feat, np.roll(feat,-1,axis=0)), 1)[shift::stride,]
            else:
                print("stride not supported", stride)
                exit()

        if feat_len != feat.shape[0] or feat_dim != feat.shape[1]:
            print("invalid shape",feat_len,feat.shape[0],feat_dim,feat.shape[1], augment)
            exit()
        if tmpx is None:
            tmpx = np.zeros((height, max_feat_len, feat_dim), np.float32)
        tmpx[i, :feat_len, :] = feat
        i += 1
    return tmpx

def run_reader(q, xinfo, ys, do_shuf):
    idx_shuf = list(range(len(xinfo)))
    if do_shuf:
        random.shuffle(idx_shuf)
    for i in idx_shuf:
        x = read_batch(xinfo[i])
        y = ys[i]
        q.put((x, y))
    q.put(None)
