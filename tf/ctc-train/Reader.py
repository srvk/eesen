import random, sys, os
import numpy as np
from fileutils.kaldi import writeArk, readMatrixByOffset

def read_batch(xinfo, roll = False):
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
            shift = augment[0]
            stride = augment[1]
            win = augment[2]
            #stride=3
            #shift=augment
            if win == 1:
                feat = feat[shift::stride,]
            elif win == 2:
                feat = np.concatenate((np.roll(feat,1,axis=0), feat), axis=1)[shift::stride,]
            elif win == 3:
                feat = np.concatenate((np.roll(feat,1,axis=0), feat, np.roll(feat,-1,axis=0)), axis=1)[shift::stride,]
            elif win == 5:
                feat = np.concatenate((np.roll(feat,2,axis=0), np.roll(feat,1,axis=0), feat, np.roll(feat,-1,axis=0), np.roll(feat,-2,axis=0)), axis=1)[shift::stride,]
            elif win == 7:
                feat = np.concatenate((np.roll(feat,3,axis=0), np.roll(feat,2,axis=0), np.roll(feat,1,axis=0), feat, np.roll(feat,-1,axis=0), np.roll(feat,-2,axis=0), np.roll(feat,-3,axis=0)), axis=1)[shift::stride,]
            else:
                print("win not supported", win)
                exit()

        if feat_len != feat.shape[0] or feat_dim != feat.shape[1]:
            print("invalid shape",feat_len,feat.shape[0],feat_dim,feat.shape[1], augment)
            exit()
        if roll:
            feat = np.roll(feat, random.randrange(-2,2,1), axis = 0)
        if tmpx is None:
            tmpx = np.zeros((height, max_feat_len, feat_dim), np.float32)
        tmpx[i, :feat_len, :] = feat
        i += 1
    return tmpx

def read_adapt_csv_matrix_folder(uttids, visual_vector_path, ndims_sat, read_first):
    first=False
    count =0
    batch_vector=np.zeros((len(uttids), 1, ndims_sat))
    first=True
    for uttid in uttids:
        if not os.path.isfile(os.path.join(visual_vector_path, uttid)):
            print(uttid+' not present in '+visual_vector_path)
            sys.exit()
        all_vectors = np.loadtxt(os.path.join(visual_vector_path, uttid), delimiter=',')
        if read_first:
            feat=all_vectorsa[-1,:]
        else:
            feat=all_vectors[0,:]
        batch_vector[count,0,:]=feat
        count=count+1
    return batch_vector


#TODO check if this works
def read_adapt_csv_folder(uttids, visual_vector_path, ndims_sat):
    first=False
    count =0
    batch_vector=np.zeros((len(uttids), 1, ndims_sat))
    first=True
    for uttid in uttids:
        if not os.path.isfile(os.path.join(visual_vector_path, uttid)):
            print(uttid+' not present in '+visual_vector_path)
    count =0
    batch_vector=np.zeros((len(uttids), 1, 100))
    first=True
    with open(visual_vector_path) as f:
        reader = csv.reader((line.strip() for line in f), delimiter=' ')
        visual_vectors = dict(reader)

    for uttid in uttids:
        if uttid in visual_vectors:
            feat = readMatrixByOffset(visual_vectors[uttid].split(":")[0], int(visual_vectors[uttid].split(":")[1]))
            feat=feat.reshape((1, ndims_sat))
            batch_vector[count,0,:]=feat
        else:
            print("visual vector "+uttids+" not find")
            sys.exit()
        count=count+1
    return batch_vector

def run_reader(q, xinfo, ys, uttids, do_shuf, ndims_sat=0, visual_vector_path="", reader_type="", roll=False):

    idx_shuf = list(range(len(xinfo)))
    if do_shuf:
        random.shuffle(idx_shuf)
    for i in idx_shuf:
        x = read_batch(xinfo[i], roll)
        y = ys[i]
        if(visual_vector_path == ""):
            q.put((x, y))
        else:
            if(reader_type=="kaldi_file"):
                adapt_vector=read_adapt_kaldi(uttids[i], visual_vector_path, ndims_sat)
            elif(reader_type=="csv_folder"):
                adapt_vector=read_adapt_csv_folder(uttids[i], visual_vector_path, ndims_sat)
            elif(reader_type=="csv_matrix_folder_first"):
                adapt_vector=read_adapt_csv_matrix_folder(uttids[i], visual_vector_path, ndims_sat, True)
            elif(reader_type=="csv_matrix_folder_last"):
                adapt_vector=read_adapt_csv_matrix_folder(uttids[i], visual_vector_path, ndims_sat, False)
            q.put((x, y, adapt_vector))
    q.put(None)
