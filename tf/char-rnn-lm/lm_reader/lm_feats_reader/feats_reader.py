from utils.fileutils import readMatrixByOffset
from utils.fileutils import read_scp_info_dic

class FeatsReaderKaldi(object):

    def __init__ (self, path_to_scp, uttids):

        self.batches=[]

        print('reading scp as dict...')
        feat_dict_info = read_scp_info_dic(path_to_scp)

        self.prepare_batches(uttids, feat_dict_info)

    def get_feat_dim(self):

        return self.batches[0][0][4]

    def read(self, idx):
        """
        feat_info: uttid, arkfile, offset, feat_len, feat_dim
        """
        batch=[]
        for element in self.batches[idx]:
            uttid, arkfile, offset, _, _ = element
            feat = readMatrixByOffset(arkfile, offset)
            batch.append(feat)

        return batch

    def prepare_batches(self, uttids, feat_dict_info):

        for batch in uttids:
            tmp_batch_uttid=[]
            for uttid in batch:
                tmp_batch_uttid.append(feat_dict_info[uttid])
            self.batches.append(tmp_batch_uttid)


