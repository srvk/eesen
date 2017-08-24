import sys
import constants
import numpy as np
from utils.fileutils import debug
from feats_reader import FeatsReader
from utils.fileutils.kaldi import readMatrixByOffset
from utils.fileutils.kaldi import read_scp_info
from utils.fileutils.kaldi import read_scp_info_dic


class FeatsReaderKaldi(FeatsReader):

    def __init__ (self, info_set, config, batches_id = None):

        self.feat_dict_info = read_scp_info_dic(self.list_files[0])


    def read(uttids)
        """
        feat_info: uttid, arkfile, offset, feat_len, feat_dim
        """
        batch=[]
        for element in uttids:

            arkfile, offset = self.feat_dict_info[element[0]].split()
            feat = readMatrixByOffset(arkfile, offset)

            batch.append(feat)

        return batch
