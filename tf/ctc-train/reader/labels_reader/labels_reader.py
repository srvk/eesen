import os
import sys
import numpy as np
import constants
from utils.fileutils import debug


class LabelsReader(object):

    def __init__(self, info_set, m_conf, batches_id):

        #temporal dictonary that stores all paths from all targets dict[language][target_1]=path_target_1
        all_languages_labels_files={}

        #temporal dictornary to store all targets dict[language][target_1]=dict_target_1
        all_languages_labels_dicts={}

        #permanent list to store the number of classes of each language dict[language][target_1]=number_target_1
        self.__target_scheme={}

        #hack to get the correct name for the file
        if(info_set=='train'):
            info_set='tr'

        if(self.__is_multiple_languages(m_conf["data_dir"])):
            print("multilanguage setup detected (in labels)... \n")
            self.__read_files_multiple_langues(m_conf[constants.DATA_DIR], info_set, all_languages_labels_files)

        else:
            print("unilanguage setup detected (in labels)... \n")
            self.__read_one_language(m_conf[constants.DATA_DIR], constants.NO_LANGUAGE_NAME, info_set, all_languages_labels_files)

        #load all dicts
        #iterate over languages
        for language, labels_path_dic in all_languages_labels_files.iteritems():

            self.__target_scheme[language] = {}
            all_languages_labels_dicts[language] = {}

            #iterate over all targets of this language
            for target_id, target_path in labels_path_dic.iteritems():

                #get number of targets and dictioanries (utt -> sequence)
                ntarget, label_dict = self._load_dict(target_path)

                self.__target_scheme[language][target_id] = ntarget
                all_languages_labels_dicts[language][target_id] = label_dict

        self.batches_y = self.__order_labels(all_languages_labels_dicts, batches_id)

        if(constants.TARGET_SCHEME in m_conf):
            self.__update_config(m_conf)

    #getter
    def get_target_scheme(self):
        return self.__target_scheme

    #this will assure that we always have the maximum number of labels
    def __update_config(self, m_conf):
        for language in self.__target_scheme:
            for target in self.__target_scheme[language]:
                conf_value = m_conf[constants.TARGET_SCHEME][language][target]
                local_value = self.__target_scheme[language][target]

                if conf_value > local_value:
                    print("Warning: number of targets has changed between sets (e.g. train and validation)")
                    self.__target_scheme[constants.TARGET_SCHEME][language][target]=conf_value

    def __is_multiple_languages(self, data_dir):
        for filename in os.listdir(data_dir):
            if (filename.startswith('labels')):
                return False
        return True

    def __read_files_multiple_langues(self, data_dir, info_set, m_all_languages_labels_files):

        for language_name in os.listdir(data_dir):
            self.__read_one_language(data_dir, language_name, info_set, m_all_languages_labels_files)

    def __read_one_language(self, data_dir, language_name, info_set, m_all_languages_labels_files):

        m_all_languages_labels_files[language_name]={}

        for filename in os.listdir(data_dir):

            if (filename.startswith('labels') and filename.endswith('.'+info_set)):
                #filename will: 'labels.tr' or 'labels.cv'
                target_id=filename.replace("labels_","").replace("labels","").replace('.'+info_set,"")
                if(target_id==""):
                    target_id=constants.NO_TARGET_NAME

                m_all_languages_labels_files[language_name][target_id] = os.path.join(data_dir, filename)

    def __order_labels(self, all_languages_labels_dicts, batches_id):

        #final batches list
        batches_y=[]

        #iterate over all batches
        for batch_id in batches_id:

            #declare counters and target batches
            #yidx: index list of a sparse matrix
            #yval: list of values that corresponds to the previous index list
            #max_label_len: maximum length value in the batch
            yidx, yval, max_label_len = {}, {}, {}

            #getting batch language of the batch
            #(just for clarification)
            batch_language=batch_id[1]

            #initialize counters and target batches
            #note that we are taking the langugae from batch_id
            for target_id, _ in all_languages_labels_dicts[batch_language].iteritems():
                yidx[target_id]=[]
                yval[target_id]=[]
                max_label_len[target_id]=0

            #iterate over all element of a batch (note that utterance are in position 0)
            for i, uttid in enumerate(batch_id[0]):

                #iterate over all target dictionaries (languages)
                for target_id, label_dict in all_languages_labels_dicts[batch_id[1]].iteritems():

                    #getting taget sequence from the current dictionary
                    label = label_dict[uttid]

                    #getting the max number of previous or current length
                    max_label_len[target_id] = max(max_label_len[target_id], len(label))

                    #fill the sparse batche (yidx: index, yval: corresponding value to this index)
                    for j in range(len(label)):
                        yidx[target_id].append([i, j])
                        yval[target_id].append(label[j])

            #construct the final batch
            batch_y={}
            for target_id, label_dict in all_languages_labels_dicts[batch_id[1]].iteritems():
                yshape_np = np.array([len(batch_id), max_label_len[target_id]], dtype = np.int32)
                yidx_np = np.asarray(yidx[target_id], dtype = np.int32)
                yval_np = np.asarray(yval[target_id], dtype = np.int32)
                batch_y[target_id]=((yidx_np, yval_np, yshape_np))

            #add the final batch to the inner list
            batches_y.append((batch_y, batch_id[0]))

        return batches_y

    def _load_dict(self, target_path):

        print("labels_reader is a virtual class can not be contructed by it self")
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

        return None, None

