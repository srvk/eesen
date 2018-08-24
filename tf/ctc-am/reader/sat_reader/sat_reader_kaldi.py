import sys
import constants
import numpy as np
from utils.fileutils import debug
from reader.sat_reader.sat_reader import SatReader

from utils.fileutils.kaldi import readMatrixByOffset
from utils.fileutils.kaldi import read_scp_info_dic


class SatReaderKaldi(SatReader):

    def __init__ (self, config, batches_id):

        self.__config = config

        #constructing parent class and creating self.list_files and stroing self._info_set
        super(SatReaderKaldi, self).__init__(self.__config[constants.CONF_TAGS.DATA_DIR], "scp")

        #getting feat in list format no need to search anything
        feat_dict_info_languages = self.__read_dict_info_languages()

        print("ordering (from batch_id) sat batches... \n")
        self._batches_sat = self.__order_feat_info(feat_dict_info_languages, batches_id)

    def update_batches_id(self, batches_id):

        print("reordering sat batches...")

        #reordering stuff
        #TODO we should do this by using dict that contains all paths and dicts
        feat_dict_info = read_scp_info_dic(self.list_files[0])
        self._batches_sat = self.__order_feat_info(feat_dict_info, batches_id)

    #read batch idx. Input: batch index. Output: batch read with feats
    def read (self, idx):

        current_batch=[]

        #TODO there is no length
        #TODO remove uttid asap(just sanitychek)
        for uttid, arkfile, offset, _, _ in self._batches_sat[idx]:

            feat = readMatrixByOffset(arkfile, offset)
            current_batch.append(feat)

        return current_batch

    def __read_dict_info_languages(self):

        feat_dict_info_languages = {}

        for language, scp_path in self._language_scheme.iteritems():
            print("preparing dictionary for "+language+"...\n")
            feat_dict_info_languages[language] = read_scp_info_dic(scp_path[0])

        return feat_dict_info_languages


    #recieve a dictionary and a batch strucute and it orders everything up
    def __order_feat_info (self, feat_dict_info, batches_id):

        #all batches
        batches_xinfo=[]

        for batch_id in batches_id:

            #get language of the batch
            batch_language = batch_id[1]
            if(batch_language not in feat_dict_info):
                print("Error: language: "+batch_language+" not present in sat files")
                print(debug.get_debug_info())
                print("exiting... \n")


            #create new batch
            batch_xinfo=[]

            for uttid in batch_id[0]:

                if(uttid not in feat_dict_info[batch_language]):
                    print("Error: uttid: "+uttid+" not present in sat file of language: "+batch_language)
                    print(debug.get_debug_info())
                    print("exiting... \n")

                #get sat vector according to language and uutid
                batch_xinfo.append(feat_dict_info[batch_language][uttid])
            batches_xinfo.append(batch_xinfo)

        return batches_xinfo

