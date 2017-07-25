import random, sys, os
import numpy as np
from fileutils.kaldi import read_scp_info_dic
from fileutils.kaldi import read_scp_info
from fileutils.kaldi import readMatrixByOffset
from feats_reader import FeatsReader

class FeatsReaderKaldi(FeatsReader):
    #constructor of Reader_Kaldi. info_set: (train, cv, sat), batches_id: order of the batches
    def __init__ (self, info_set, data_dir, lstm_type, online_augment_config, batch_size):

        #constructing parent class and creating self.list_files and stroing self.info_set
        super(FeatsReaderKaldi, self).__init__(info_set, data_dir, online_augment_config, "scp")

        print("ordering "+info_set+" batches...")

        print(self.list_files)

        #getting feat in list format no need to search anything
        feat_dict_info = read_scp_info(self.list_files[0])

        self.batches_x, self.batches_id = self.__create_ordered_batches(feat_dict_info, lstm_type, batch_size)


    #TODO this is just an scheme of how to deal with a mix env
    #TODO arguments will be either the path or a number that can be the posisiton in the list
    def change_source (source_position):

        if(self.info_set == 'cv'):
            print("this option is not available for this type of info_set (cv)")
            print("exiting...")
            sys.exit()

        #sanity check
        if(len(self.filenames) < source_positon-1):
            print(str(source_positon)+" does not exists for this current source")
            print("get_num_augmented_folders() will provide this information for you")
            print("exiting...")
            sys.exit()

        #getting feat in dict format (faster to search)
        feat_dict_info = read_scp_info_dic(self.list_files[source_position])

        self.batches_x = self.__order_feat_info(feat_dict_info, self.batches_id)

    #TODO here we will need to indicate which language are we looking for
    def get_num_augmented_folders():
        return len(self.list_files)

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
    def read (self, idx, roll=False):
        i=0
        tmpx=None

        number_of_utt=len(self.batches_x[idx])
        max_utt_len = max(x[2] for x in self.batches_x[idx])

        for arkfile, offset, feat_len, feat_dim, augment in self.batches_x[idx]:

            feat = readMatrixByOffset(arkfile, offset)

            feat = self.augmenter.augment(feat, augment)

            #sanity check that the augmentation is ok
            if feat_len != feat.shape[0] or feat_dim != feat.shape[1]:
                print("invalid shape",feat_len,feat.shape[0],feat_dim,feat.shape[1], augment)
                sys.exit()

            if tmpx is None:
                tmpx = np.zeros((number_of_utt, max_utt_len, feat_dim), np.float32)


            tmpx[i, :feat_len, :] = feat
            i += 1

        return tmpx

    #it creates batches and returns a template of batch_ids that will be used externally to createate other readers (or maybe something else)
    def __create_ordered_batches(self, feat_info, lstm_type, batch_size):

        #augmenting data
        feat_info=self.augmenter.preprocess(feat_info)

        #sort the list by length
        feat_info = sorted(feat_info, key = lambda x: x[3])

        #CudnnLSTM requires batches of even sizes
        if lstm_type == "cudnn":
            batches_x, batches_id = self.__make_even_batches_info(feat_info, batch_size)
        else:
            batches_x, batches_id = self.__make_batches_info(feat_info, batch_size)

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
            uttid_aux, arkfile, offset, feat_len, feat_dim, augment = feat_info[start + i]
            xinfo.append((arkfile, offset, feat_len, feat_dim, augment))
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

