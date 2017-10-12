import sys
import constants
import numpy as np
from utils.fileutils import debug
from reader.feats_reader.feats_reader import FeatsReader
from utils.fileutils.kaldi import readMatrixByOffset
from utils.fileutils.kaldi import read_scp_info


class FeatsReaderKaldi(FeatsReader):

    def __init__ (self, info_set, config):

        self.__config = config

        #constructing parent class and creating self.list_files and stroing self._info_set
        super(FeatsReaderKaldi, self).__init__(info_set, self.__config[constants.CONF_TAGS.DATA_DIR],
                                               self.__config[constants.CONF_TAGS.ONLINE_AUGMENT_CONF], "scp")

        #getting feat in dict format
        feat_dict_info_languages = self.__read_dict_info_languages()

        print("ordering all languages (from scratch) "+info_set+" batches... \n")
        self._batches_x, self._batches_id = self.__create_ordered_batches_all_languages(feat_dict_info_languages, config[constants.CONF_TAGS.LSTM_TYPE], config[constants.CONF_TAGS.BATCH_SIZE])

    #TODO check augmentation this idea is kinda ok, but should take a closer look
    def change_source (self, new_source_positions):

        if(self._info_set != 'train'):
            print("this option is not available for this type of info_set (cv)")
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()

        #we need to recreat and refill the dictionary becuase we removed it before
        feat_dict_info_languages={}
        for language_id, scp_path in self._language_augment_scheme.items():
            print(80 * "-")
            print("preparing dictionary for "+language_id+"...\n")
            feat_dict_info_languages[language_id] = read_scp_info(scp_path[new_source_positions[language_id]])

        print(80 * "-")
        self._batches_x, self._batches_id = self.__create_ordered_batches_all_languages(feat_dict_info_languages,
                                                                                        self.__config[constants.CONF_TAGS.LSTM_TYPE],
                                                                                        self.__config[constants.CONF_TAGS.BATCH_SIZE])

    #read batch idx. Input: batch index. Output: batch read with feats
    def read(self, idx):

        i=0
        tmpx=None

        number_of_utt=len(self._batches_x[idx])
        max_utt_len = max(x[2] for x in self._batches_x[idx])

        #TODO remove this asap (just sanity check)
        uttid_check=[]

        #TODO remove uttid asap(just sanitychek)
        for arkfile, offset, feat_len, feat_dim, augment, uttid in self._batches_x[idx]:

            feat = readMatrixByOffset(arkfile, offset)

            feat = self._augmenter.augment(feat, augment)

            #sanity check that the augmentatbacion is ok
            if feat_len != feat.shape[0] or feat_dim != feat.shape[1]:
                print("invalid shape", feat_len, feat.shape[0], feat_dim,feat.shape[1], augment)
                print(debug.get_debug_info())
                print("exiting...")
                sys.exit()

            if tmpx is None:
                tmpx = np.zeros((number_of_utt, max_utt_len, feat_dim), np.float32)

            tmpx[i, :feat_len, :] = feat
            uttid_check.append(uttid)
            i += 1

        return (tmpx, uttid_check)

    def __create_ordered_batches_all_languages(self, feat_dict_info_languages, lstm_type, batch_size):

        all_zipped_batches = []
        #https://stackoverflow.com/questions/7529376/pythonic-way-to-mix-two-lists

        for language, feat_dict_info in feat_dict_info_languages.items():

            batch_x_language, batch_id_language = self.__create_ordered_batches(feat_dict_info, lstm_type, batch_size)


            #coloring every batch with its language
            batch_id_language_c=list(zip(batch_id_language, [language]*len(batch_id_language)))

            all_zipped_batches=all_zipped_batches+list(zip(batch_x_language, batch_id_language_c))

        #unzip
        batch_x, batch_id = list(zip(*all_zipped_batches))

        return batch_x, batch_id

    def __read_dict_info_languages(self):

        feat_dict_info_languages = {}

        for language, scp_path in self._language_augment_scheme.items():

            print("preparing dictionary for "+language+"...\n")
            feat_dict_info_languages[language] = read_scp_info(scp_path[0])
            if(len(feat_dict_info_languages[language]) == 0):
                print("feature file ("+scp_path[0]+") for language: "+language+" is void")
                print(debug.get_debug_info())
                print("exiting...")
                sys.exit()

        return feat_dict_info_languages

    #it creates batches and returns a template of batch_ids that will be used externally to createate other readers (or maybe something else)
    def __create_ordered_batches(self, feat_info, lstm_type, batch_size):

        #augmenting data
        feat_info=self._augmenter.preprocess(feat_info)

        #sort the list by length
        feat_info = sorted(feat_info, key = lambda x: x[3])

        #CudnnLSTM requires batches of even sizes
        if lstm_type == "cudnn":
            batches_x, batches_id = self.__make_even_batches_info(feat_info, batch_size)
        else:
            #TODO try to work on that also asap
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
        return None, None

    #contruct one batch of data
    def __get_batch_info (self, feat_info, start, height):
        """
        feat_info: uttid, arkfile, offset, feat_len, feat_dim
        """
        xinfo, uttid = [], []

        for i in range(height):
            uttid_aux, arkfile, offset, feat_len, feat_dim, augment = feat_info[start + i]
            xinfo.append((arkfile, offset, feat_len, feat_dim, augment, uttid_aux))
            uttid.append(uttid_aux)

        return xinfo, uttid

