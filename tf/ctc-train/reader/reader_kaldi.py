import random, sys, os
import numpy as np
from fileutils.kaldi import read_scp_info_dic
from fileutils.kaldi import read_scp_info
from fileutils.kaldi import readMatrixByOffset
from reader import Reader

class ReaderKaldi(Reader):
    #constructor of Reader_Kaldi. info_set: (train, cv, sat), batches_id: order of the batches
    def __init__ (self, info_set, args, batches_id=None):

        #getting filename will al kaldi pointers to ark
        filename = os.path.join(args.data_dir, "%s_local.scp" % (info_set))

        #if batches_id has been generated previously this will be used as template
        if(batches_id):
            print("creating "+info_set+"batches using and external template...")
            #getting feat in dict format (faster to search)
            feat_dict_info = read_scp_info_dic(filename)

            self.batches_x = self.__order_feat_info(feat_dict_info, batches_id)
        else:
            print("ordering "+info_set+" batches...")
            #getting feat in list format no need to search anything
            feat_dict_info = read_scp_info(filename)

            self.batches_x, self.batches_id = self.__create_ordered_batches(feat_dict_info, args)


    #getter number of feature dimension. Just taking the size of the first
    def get_num_dim (self):
        return self.batches_x[0][0][3]

    #get number of batches
    def get_num_batches (self):
        return len(self.batches_x)

    #get number of batches
    def get_batches_id (self):
        return self.batches_id

    #read batch idx. Input: batch index. Output: batch read with feats
    def read (self, idx):
        i=0
        tmpx=None
        for arkfile, offset, feat_len, feat_dim, augment in self.batches_x[idx]:
            feat = readMatrixByOffset(arkfile, offset)
            if not augment is None:
                #TODO other values of strides currently hardcoded
                stride=3
                shift=augment
                if stride is 3:
                    feat=super(Reader, self).three_stride(feat)
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

    #it creates batches and returns a template of batch_ids that will be used externally to createate other readers (or maybe something else)
    def __create_ordered_batches(self, feat_info, args):

        #sort the list by length
        feat_info = sorted(feat_info, key = lambda x: x[2])

        #CudnnLSTM requires batches of even sizes
        if args.lstm_type == "cudnn":
            batches_x, batches_id = self.__make_even_batches_info(feat_info, args.batch_size)
        else:
            batches_x, batches_id = self.__make_batches_info(feat_info, args.batch_size)

        return batches_x, batches_id

    #It creates an even number of batches: recieves all the data and batch size
    def __make_even_batches_info (self, feat_info, batch_size):
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
            xinfo, uttid = self.__get_batch_info(feat_info, idx, j - idx)
            batch_x.append(xinfo)
            uttids.append(uttid)
            idx = j
        return batch_x, uttids

    def __make_batches_info (self, feat_info, batch_size):
        print("still not copied")

    #contruct one batch of data
    def __get_batch_info (self, feat_info, start, height):
        """
        feat_info: uttid, arkfile, offset, feat_len, feat_dim
        """
        xinfo, uttid = [], []

        for i in range(height):
            uttid_aux, arkfile, offset, feat_len, feat_dim = feat_info[start + i]
            xinfo.append((arkfile, offset, feat_len, feat_dim))
            uttid.append(uttid_aux)

        return xinfo, uttid

    #recieve a dictionary and a batch strucute and it orders everything up
    def __order_feat_info (self, feat_dict_info, batches_id):
        batches_xinfo=[]
        for batch_id in batches_id:
            batch_xinfo=[]
            for sample in batch_id:
                batch_xinfo.append(feat_dict_info[sample])
            batches_xinfo.append(batch_xinfo)

        return batches_xinfo

