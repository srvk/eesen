import random, sys
import numpy as np
from fileutils.kaldi import writeArk, readMatrixByOffset

def read_batch(xinfo):
    """
    xinfo: arkfile, offset, feat_len, feat_dim[, augment_info]
    """
    height = len(xinfo)
    max_feat_len = max(x[2] for x in xinfo)
    tmpx = None
    i = 0
    for x in xinfo:
        if len(x) is 4:
            arkfile, offset, feat_len, feat_dim = x
            augment_info = None
        else:
            arkfile, offset, feat_len, feat_dim, augment_info = x

        feat = readMatrixByOffset(arkfile, offset)

        if not augment_info is None:
            # data augmentation
            offset = augment_info[0]
            stride = augment_info[1]
            # subsampling -> feat[2::3,] has a stride of 3 and an offset of 2
            # splicing -> numpy.concatenate((numpy.roll(a,1,axis=0),a,numpy.roll(a,-1,axis=0)),1)[1::3,]
            feat = np.concatenate((np.roll(feat,-1), feat, np.roll(feat,1)),1)[offset::stride,]

        if feat_len != feat.shape[0] or feat_dim != feat.shape[1]:
            print("invalid shape",feat_len,feat.shape[0],feat_dim,feat.shape[1])
            exit()

        if tmpx is None:
            tmpx = np.zeros((height, max_feat_len, feat_dim), np.float32)
        tmpx[i, :feat_len, :] = feat
        i += 1
    #print("read mini-batch ",arkfile,offset,feat_len)
    return tmpx

def run_reader(q, xinfo, ys, do_shuf, epoch=1):
    idx_shuf = list(range(len(xinfo)))
    if do_shuf and epoch > 0:
        random.shuffle(idx_shuf)
        print("Shuffling for epoch",epoch+1)
    for i in idx_shuf:
        x = read_batch(xinfo[i])
        y = ys[i]
        q.put((x, y))
    q.put(None)
