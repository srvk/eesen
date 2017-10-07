import os
import sys
import numpy as np
import constants
from utils.fileutils import debug


class LabelsReader(object):

    def __init__(self, info_set, m_conf, batches_id, language_scheme = None):

        #temporal dictonary that stores all paths from all targets dict[language][target_1]=path_target_1
        self.__all_languages_labels_files={}

        #permanent list to store the number of classes of each language dict[language][target_1]=number_target_1
        self.__language_scheme={}

        #temporal dictornary to store all targets dict[language][target_1]=dict_target_1
        all_languages_labels_dicts={}

        #hack to get the correct name for the file
        if(info_set=='train'):
            info_set='tr'

        if(self.__is_multiple_languages(m_conf["data_dir"])):
            print("multilanguage setup detected (in labels)... \n")
            self.__read_files_multiple_langues(m_conf[constants.CONF_TAGS.DATA_DIR],
                                               info_set)

        else:
            print("unilanguage setup detected (in labels)... \n")
            self.__read_one_language(m_conf[constants.CONF_TAGS.DATA_DIR],
                                     constants.DEFAULT_NAMES.NO_LANGUAGE_NAME,
                                     info_set)


        #load all dicts
        #iterate over languages
        for language, labels_path_dic in self.__all_languages_labels_files.items():

            self.__language_scheme[language] = {}
            all_languages_labels_dicts[language] = {}

            #iterate over all targets of this language
            for target_id, target_path in labels_path_dic.items():

                #get number of targets and dictioanries (utt -> sequence)
                ntarget, label_dict = self._load_dict(target_path)

                self.__language_scheme[language][target_id] = ntarget
                all_languages_labels_dicts[language][target_id] = label_dict

        if(language_scheme != None):
            self.__language_scheme = language_scheme

            #fill it up with void dicts
            for language_id, target_scheme in self.__language_scheme.items():

                if(language_id not in all_languages_labels_dicts):

                    all_languages_labels_dicts[language_id] = {}
                    for target_id, _ in target_scheme.items():
                        all_languages_labels_dicts[language_id][target_id]={}

        self.__batches_y = self.__order_labels(all_languages_labels_dicts, batches_id)

    def read(self, idx_batch):

        if(idx_batch > len(self.__batches_y) - 1):

            print("idx_batch for labels is out of bounds")
            print(debug.get_debug_info())
            print("exiting...\n")
            sys.exit()
        return self.__batches_y[idx_batch]

    #getter
    def get_language_scheme(self):
        return self.__language_scheme

    def update_batches_id(self, batches_id):

        #temporal dictornary to store all targets dict[language][target_1]=dict_target_1
        all_languages_labels_dicts={}

        #iterate over languages
        for language, labels_path_dic in self.__all_languages_labels_files.items():

            self.__language_scheme[language] = {}
            all_languages_labels_dicts[language] = {}

            #iterate over all targets of this language
            for target_id, target_path in labels_path_dic.items():

                #get number of targets and dictioanries (utt -> sequence)
                ntarget, label_dict = self._load_dict(target_path)

                self.__language_scheme[language][target_id] = ntarget
                all_languages_labels_dicts[language][target_id] = label_dict

        self.__batches_y = self.__order_labels(all_languages_labels_dicts, batches_id)

    def set_number_targets(self, language, target_id, number_targets):
        self.__language_scheme[language][target_id] = number_targets

    #this will assure that we always have the maximum number of labels
    def __update_config(self, m_conf):
        for language in self.__language_scheme:
            for target in self.__language_scheme[language]:
                conf_value = m_conf[constants.CONF_TAGS.LANGUAGE_SCHEME][language][target]
                local_value = self.__language_scheme[language][target]

                if conf_value > local_value:
                    print("Warning: number of targets has changed between sets (e.g. train and validation)")
                    self.__language_scheme[constants.CONF_TAGS.LANGUAGE_SCHEME][language][target]=conf_value

    def __is_multiple_languages(self, data_dir):

        is_labels = False
        is_feats = False
        is_folder = False

        for filename in os.listdir(data_dir):
            if (filename.startswith('labels')):
                is_labels = True

            if (filename.startswith('labels')):
                is_labels =  True

            if (os.path.isdir(os.path.join(data_dir,filename)) and not filename.startswith('augment_')):
                is_folder = True

        if(is_labels and is_feats and not is_folder):
            return False
        elif(is_folder):
            if(is_labels or is_feats):
                print("strange multilingual setup found in data_dir:")
                print("(several languages but scp or labels in root data_dir (?))")
                for filename in os.listdir(data_dir):
                    print(filename)
                print(debug.get_debug_info())
                print("exiting...\n")
                sys.exit()

            else:
                return True

    def __read_files_multiple_langues(self, data_dir, info_set):

        for language_name in os.listdir(data_dir):
            self.__read_one_language(os.path.join(data_dir,language_name), language_name, info_set)

    def __read_one_language(self, language_dir, language_name, info_set):

        self.__all_languages_labels_files[language_name]={}

        file_find=False

        for filename in os.listdir(language_dir):

            if (filename.startswith('labels') and filename.endswith('.'+info_set)):
                target_id=filename.replace("labels_","").replace("labels","").replace('.'+info_set,"")
                if(target_id==""):
                    target_id=constants.DEFAULT_NAMES.NO_TARGET_NAME

                self.__all_languages_labels_files[language_name][target_id] = os.path.join(language_dir, filename)
                file_find=True

        if(not file_find):
            print("no label files fins in " + language_dir + " with info_set: " + info_set)
            print(debug.get_debug_info())
            print("exiting...\n")
            sys.exit()

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
            for language_id, target_scheme in all_languages_labels_dicts.items():
                yidx[language_id]={}
                yval[language_id]={}
                max_label_len[language_id]={}

                for target_id, _ in target_scheme.items():
                    yidx[language_id][target_id]=[]
                    yval[language_id][target_id]=[]
                    max_label_len[language_id][target_id]=0

            #iterate over all element of a batch (note that utterance are in position 0)
            for i, uttid in enumerate(batch_id[0]):

                #iterate over all target dictionaries (languages)
                for language_id, language_dict in all_languages_labels_dicts.items():

                    #iterate over all targets
                    for target_id, label_dict in language_dict.items():

                        #if it is the correct one we fill everything ok
                        if(language_id == batch_language):
                            #getting taget sequence from the current dictionary
                            label = label_dict[uttid]

                            #getting the max number of previous or current length
                            max_label_len[language_id][target_id] = max(max_label_len[language_id][target_id], len(label))

                            #fill the sparse batche (yidx: index, yval: corresponding value to this index)
                            for j in range(len(label)):
                                yidx[language_id][target_id].append([i, j])
                                yval[language_id][target_id].append(label[j])

                        #else we create a fake one
                        else:
                            #creating fake labels
                            max_label_len[language_id][target_id]=1
                            yidx[language_id][target_id].append([i,0])
                            yval[language_id][target_id].append(0)


            #construct the final batch
            batch_y={}
            for language_id, target_scheme in all_languages_labels_dicts.items():
                batch_y[language_id]={}

                for target_id, _ in target_scheme.items():

                    yshape_np = np.array([len(batch_id[0]), max_label_len[language_id][target_id]], dtype = np.int32)
                    yidx_np = np.asarray(yidx[language_id][target_id], dtype = np.int32)
                    yval_np = np.asarray(yval[language_id][target_id], dtype = np.int32)

                    batch_y[language_id][target_id]=((yidx_np, yval_np, yshape_np))

            #add the final batch to the inner list
            batches_y.append((batch_y, batch_language))

        return batches_y

    def _load_dict(self, target_path):

        print("lm_labels_reader is a virtual class can not be contructed by it self")
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

        return None, None

