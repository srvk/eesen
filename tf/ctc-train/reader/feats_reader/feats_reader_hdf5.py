import h5py
import json
import numpy as np
import random
from reader import Reader

class FeatsReaderHDF5(Reader):

    def __init__(self, filename, args, uttids=None):




        #getting filename will al kaldi pointers to ark
        filename = os.path.join(data_dir, "%s_local.hdf5" % (filename))
        self.hdf5_file = h5py.File(path, 'r')


        if(not uttids):
            #get all values and sort by lenngth (foth element)
            feat_info = sorted(___get_utt_and_values(filename), key = lambda x: x[2])

            #CudnnLSTM requires batches of even sizes
            if args.lstm_type == "cudnn":
                self.batches_x, self.uttids = make_even_batches_info(feat_info, args.batch_size
)
            else:
                self.batches_x, self.uttids = make_batches_info(feat_info, args.batch_size
)
        else:
            self.uttids = uttids
            self.batches_x = order_feat_info(feat_dict_info, batches_id)


        self.batches_utt_ids = batches_utt_ids

    #It creates an even number of batches: recieves all the data and batch size
    def make_even_batches_info(feat_info, batch_size):
        """
        feat_info: uttid, arkfile, offset, feat_len, feat_dim
        """
        batch_x, uttids = [], []
        L = len(feat_info)
        idx = 0
        while idx < L:
            # find batch with even size, and with maximum size of batch_size
            j = idx + 1
            target_len = feat_info[idx][3]
            while j < min(idx + batch_size, L) and feat_info[j][3] == target_len:
                j += 1
            xinfo, uttid = get_batch_info(feat_info, idx, j - idx)
            batch_x.append(xinfo)
            uttids.append(uttid)
            batch_y_element=[]

            idx = j
        return batch_x, uttids


    def get_size(self):
        return len(all_batches)

    def read(self, idx):
        current_batch=[]
        i=0
        tmpx=None
        for element in self.batches_utt_ids[idx]:
            feat=self.hdf5_file[element]
            if not augment is None:
                #TODO other values of strides currently hardcoded
                stride=3
                shift=augment
                if stride is 3:
                    feat=super(ReaderHDF5, self).three_stride(feat)
                else:
                    print("stride not supported", stride)
                    exit()

            if tmpx is None:
                tmpx = np.zeros((height, max_feat_len, feat_dim), np.float32)

            tmpx[i, :feat_len, :] = feat
        current_batch.append()

    def __get_utt_and_values(filename):
        #get features and values
        return None

