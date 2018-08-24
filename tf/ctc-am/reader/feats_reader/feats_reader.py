import os
import sys

import constants
from reader.augmenter.augmenter import Augmenter
from utils.fileutils import debug


class FeatsReader(object):

    """This is a virtual class for reading input features (and also sat vectors altough is out TODO list to move this).

    In order to use this class as parent three methods should be implemented (update_batches_id, change_source, read).
    Check the documentation for more detailed explanation.

    .. note::

       You can not instantiate this class without an implementation all non implemtned methods kill the program

    """

    def __init__(self, info_set, data_dir, online_augment_config, extension):

        #list of all folders
        self._language_augment_scheme = {}
        self._info_set = info_set
        self._extension = extension
        self._augmenter = Augmenter(online_augment_config)
        self._batches_x = None
        self._batches_id = None

        if (info_set == 'train'):

            self.__read_train(data_dir)

        elif (info_set =='cv'):

            self.__read_test(data_dir)

        elif (info_set =='test'):

            self.__read_test(data_dir)

        else:

            print("Error: info_set var has an unkonwn value: "+info_set)
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()

    def change_source (self, new_source_positions):
        print("def change_source (self, source_position) is a virutal function need to be caled from an instance that has implemented it")
        print(debug.get_debug_info())
        print("exiting... \n")
        sys.exit()

    def read (self, idx):
        print("def read (self, idx) is a virutal function need to be caled from an instance that has implemented it")
        print(debug.get_debug_info())
        print("exiting... \n")
        sys.exit()

    def get_language_augment_scheme(self):
        return self._language_augment_scheme

    #getter number of feature dimension. Just taking the size of the first
    def get_num_dim (self):
        return self._batches_x[0][0][3]

    #get number of batches
    def get_num_batches (self):
        return len(self._batches_x)

    #get batch id structure
    def get_batches_id (self):
        return self._batches_id

    def __read_train(self, data_dir):

        #this counts allow us to see which is the structure of data_dir (how many augmentations?, etc.)
        count_augmented_folder, count_scp_files = self.__count_status_files(data_dir)

        if(count_augmented_folder == 0 and count_scp_files == 0):

            print("multilingual set up detected on train set ("+str(len(os.listdir(data_dir)))+" languages) \n")
            self.__process_multilingual_folder(data_dir, False)

        else:
            self.__process_language_folder(data_dir, constants.DEFAULT_NAMES.NO_LANGUAGE_NAME, False)

    def __read_test(self, data_dir):


        count_augmented_folder, count_scp_files = self.__count_status_files(data_dir)

        if(count_augmented_folder == 0 and count_scp_files == 0):

            print("multilingual set up detected on test set ( "+str(len(os.listdir(data_dir)))+" languages )... \n")
            self.__process_multilingual_folder(data_dir, True)

        else:

            print("unilingual set up detected on test or  set language... \n")
            self.__process_language_folder(data_dir, constants.DEFAULT_NAMES.NO_LANGUAGE_NAME, True)

    def __process_multilingual_folder(self, data_dir, is_test):

        for language_folder in os.path.os.listdir(data_dir):
            self.__process_language_folder(os.path.join(data_dir, language_folder), language_folder, is_test)

    def __process_language_folder(self, data_dir, language_name, is_test):

        count_augmented_folder, count_scp_files = self.__count_status_files(data_dir)

        #is train
        #there is sample augmentation
        if(count_augmented_folder > 0 and not is_test):
            print("augmented data (mix) found for language "+language_name+"... \n")
            self.__read_augmented_folder(data_dir, self._info_set, self._extension, language_name)
        else:
            #is train
            if(not is_test):
                print("non augmented (mix) training set found for language: "+language_name+" ... \n")
            else:
                print(self._info_set+" (feats) found for language: "+language_name+" ... \n")
            self._language_augment_scheme[language_name] = [self.__read_folder(data_dir, self._info_set, self._extension)]

    def __count_status_files(self, data_dir):

        count_augmented_folder = 0
        count_scp_files = 0

        for augmented_folder in os.listdir(data_dir):
            path_augmented=os.path.join(data_dir,augmented_folder)
            if os.path.isdir(path_augmented) and augmented_folder.startswith("augment_"):
                count_augmented_folder += 1
            if not os.path.isdir(path_augmented) and augmented_folder.endswith(".scp"):
                count_scp_files += 1

        return count_augmented_folder, count_scp_files

    def __read_augmented_folder(self, data_dir, info_set, extension, language_name):

        self._language_augment_scheme[language_name]=[]

        count_augment = 0
        for augmented_dirname in os.listdir(data_dir):

            augmented_path = os.path.join(data_dir, augmented_dirname)

            #it is an augmented folder
            if os.path.isdir(augmented_path) and augmented_dirname.startswith("augment"):
                feats_file = self.__read_folder(augmented_path, info_set, extension)
                if(feats_file):
                    self._language_augment_scheme[language_name].append(feats_file)
                else:
                    print("No feats.scp encountered in  "+augmented_path+" for language "+language_name)
                    print(debug.get_debug_info())
                    print("exiting... \n")
                    sys.exit()

    def __read_folder(self, data_dir, info_set, extension):

        path_to_local = os.path.join(data_dir, "%s_local.%s" % (info_set, extension))

        if(os.path.isfile(path_to_local)):
            return path_to_local
        else:
            print("path: "+path_to_local+" does not exists. It is needed by "+info_set)
            print(debug.get_debug_info())
            print("exiting... \n")
            sys.exit()
            return None








