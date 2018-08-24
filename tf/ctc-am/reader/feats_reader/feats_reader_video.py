import sys
import imageio
import os

import numpy as np
from reader.feats_reader.feats_reader import FeatsReader

import constants
from utils.fileutils import debug
from utils.fileutils.kaldi import readMatrixByOffset
from utils.fileutils.kaldi import read_scp_info


class FeatsReaderVideo(FeatsReader):

    def __init__ (self, info_set, config):

        self.__config = config

        #constructing parent class and creating self.list_files and stroing self._info_set
        super(FeatsReaderVideo, self).__init__(info_set, self.__config[constants.CONF_TAGS.DATA_DIR],
                                               self.__config[constants.CONF_TAGS.ONLINE_AUGMENT_CONF], "video")

        #getting feat in dict format
        feat_dict_info_languages = self.__read_dict_info_languages()

        print("ordering all languages (from scratch) "+info_set+" batches... \n")
        self._batches_x, self._batches_id =\
            self.__create_ordered_batches_all_languages(feat_dict_info_languages, config[constants.CONF_TAGS.LSTM_TYPE], config[constants.CONF_TAGS.BATCH_SIZE])


    #TODO check augmentation this idea is kinda ok, but should take a closer look
    def change_source (self, new_source_positions):
        print("change source (augmentation mix) is not needed by feats_reader_video")
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    def get_num_dim (self):
        return self.__dim

    #read batch idx. Input: batch index. Output: batch read with feats
    def read (self, idx):

        i=0
        tmpx=None

        number_of_utt=len(self._batches_x[idx])

        uttid_check=[]

        vids = []
        vid_seq_lens = []
        for uttid, path, feat_len in self._batches_x[idx]:
            uttid_check.append(uttid)

            video = imageio.mimread(path)
            vids.append(np.array(video, dtype=np.float32) / 255.0)
            vid_seq_lens.append(vids[-1].shape[0])


        vid_seq_lens = np.array(vid_seq_lens)
        video_size = vids[-1].shape[1:]

        tmpx = np.zeros([number_of_utt, vid_seq_lens.max(), ] + list(video_size) )
        for i_vid, vid in enumerate(vids):
            t = vid.shape[0]
            tmpx[i_vid, :t, ...] = vid

        return (tmpx, uttid_check)



    def __create_ordered_batches_all_languages(self, feat_dict_info_languages, lstm_type, batch_size):

        all_ziped_batches = []
        #https://stackoverflow.com/questions/7529376/pythonic-way-to-mix-two-lists

        for language, feat_dict_info in feat_dict_info_languages.iteritems():

            batch_x_language, batch_id_language = self.__create_ordered_batches(feat_dict_info, lstm_type, batch_size)


            #coloring every batch with its language
            batch_id_language_c=zip(batch_id_language, [language]*len(batch_id_language))

            all_ziped_batches=all_ziped_batches+zip(batch_x_language, batch_id_language_c)

        #unzip
        batch_x, batch_id = zip(*all_ziped_batches)

        return batch_x, batch_id

    def __read_dict_info_languages(self):

        feat_dict_info_languages = {}

        for language, scp_path in self._language_augment_scheme.iteritems():
            print("preparing dictionary for "+language+"...\n")

            feat_dict_info_languages[language] = self.__read_video(scp_path[0])

            if(len(feat_dict_info_languages[language]) == 0):
                print("feature file ("+scp_path[0]+") for language: "+language+" is void")
                print(debug.get_debug_info())
                print("exiting...")
                sys.exit()

        return feat_dict_info_languages

    def __read_video(self, path):

        parsed_scp = self.__parse_scp(path)
        data_paths=[]

        failed_read_videos = 0

        self.__dim = (int(parsed_scp[0][3]), int(parsed_scp[0][4]))

        for exemplar in parsed_scp:

            if os.path.isfile(exemplar[1]):  # only add videos whos path exist

                # try:
                data_paths.append((exemplar[0], exemplar[1], int(exemplar[2])))

                # except:
                #     failed_read_videos += 1

                if(self.__dim[0] != int(exemplar[3]) or self.__dim[1] != int(exemplar[4])):
                    print("dimensions are not consistent")
                    print(debug.get_debug_info())
                    print("exiting...")
                    sys.exit()

            else:
                print("video file ("+exemplar[1]+") does not exist")
                print(debug.get_debug_info())
                print("exiting...")
                sys.exit()

        # print("All paths processed. "+str(failed_read_videos)+" can not be read.")
        return data_paths

    def __parse_scp(self, filename):
        '''
        Reads Filename, returns list where each element is:
            - A list of the space separated strings from the file line
        '''
        with open(filename, 'r') as f:
            res = [tuple(line.strip().split()) for line in f]
        return res


    #it creates batches and returns a template of batch_ids that will be used externally to createate other readers (or maybe something else)
    def __create_ordered_batches(self, feat_info, lstm_type, batch_size):

        #augmenting data
        #feat_info=self._augmenter.preprocess(feat_info)

        #sort the list by length
        feat_info = sorted(feat_info, key = lambda x: x[2])

        #CudnnLSTM requires batches of even sizes
        if lstm_type == "cudnn":
            batches_x, batches_id = self.__make_even_batches_info(feat_info, batch_size)
        else:
            print("not implemented yet...")
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()
            #TODO try to work on that also asap
            #batches_x, batches_id = self.__make_batches_info(feat_info, batch_size)

        return batches_x, batches_id

    #It creates an even number of batches: recieves all the data and batch size
    def __make_even_batches_info (self, feat_info, batch_size):
        """
        feat_info: uttid, path, feat_len
        """
        batch_x, uttids = [], []
        L = len(feat_info)
        idx = 0

        while idx < L:
            # find batch with even size, and with maximum size of batch_size
            j = idx + 1
            target_len = feat_info[idx][2]

            while j < min(idx + batch_size, L) and feat_info[j][2] == target_len:
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
        feat_info: uttid, path, feat_len
        """
        xinfo, uttid = [], []

        for i in range(height):
            uttid_aux, path, feat_len = feat_info[start + i]
            xinfo.append((uttid_aux, path, feat_len))
            uttid.append(uttid_aux)

        return xinfo, uttid

