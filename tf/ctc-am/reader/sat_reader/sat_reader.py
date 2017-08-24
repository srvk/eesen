import os
import sys

import constants
from utils.fileutils import debug


class SatReader(object):

    """This is a virtual class for reading input features (and also sat vectors altough is out TODO list to move this).

    In order to use this class as parent three methods should be implemented (update_batches_id, change_source, read).
    Check the documentation for more detailed explanation.

    .. note::

       You can not instantiate this class without an implementation all non implemtned methods kill the program

    """

    def __init__(self, data_dir, extension):

        #list of all folders
        self._language_scheme = {}
        self._extension = extension
        self._batches_sat = None

        self.__read_sat(data_dir)

    def update_batches_id(self, batches_id):
        print("def update_batches_id(self, batches_id) is a virutal function need to be caled from an instance that has implemented it")
        print(debug.get_debug_info())
        print("exiting... \n")
        sys.exit()

    def read (self, idx):
        print("def read (self, idx) is a virutal function need to be caled from an instance that has implemented it")
        print(debug.get_debug_info())
        print("exiting... \n")
        sys.exit()

    def get_language_scheme(self):
        return self._language_scheme

    #getter number of feature dimension. Just taking the size of the first
    def get_num_dim (self):
        return self._batches_sat[0][0][4]

    #get number of batches
    def get_num_batches (self):
        return len(self._batches_sat)

    def __read_sat(self, data_dir):

        count_augmented_folder, count_scp_files = self.__count_status_files(data_dir)

        if(count_augmented_folder == 0 and count_scp_files == 0):

            print("multilingual set up detected on sat set ( "+str(len(os.listdir(data_dir)))+" languages )... \n")
            self.__process_multilingual_folder(data_dir)

        else:

            print("unilingual set up detected on sat set language... \n")
            self.__process_language_folder(data_dir, constants.DEFAULT_NAMES.NO_LANGUAGE_NAME)

    #TODO maybe there should be either only one sat (one directory up)
    #TODO or just augment directly
    def __process_multilingual_folder(self, data_dir):

        for language_name in os.path.os.listdir(data_dir):
            self.__process_language_folder(os.path.join(data_dir, language_name), language_name)

    def __process_language_folder(self, data_dir, language_name):

        count_augmented_folder, count_scp_files = self.__count_status_files(data_dir)

        #there is sample augmentation
        if(count_augmented_folder > 0):
            print("sat augmented data (mix) found for language "+language_name+"... \n")
            self.__read_augmented_folder(data_dir, self._extension, language_name)
        else:
            print("non augmented (mix) sat set found for language: "+language_name+" ... \n")
            self._language_scheme[language_name] = [self.__read_sat_folder(data_dir, self._extension)]

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

    def __read_augmented_folder(self, data_dir, extension, language_name):

        self._language_scheme[language_name]=[]

        count_augment = 0
        for augmented_dirname in os.listdir(data_dir):

            augmented_path = os.path.join(data_dir, augmented_dirname)

            #it is an augmented folder
            if os.path.isdir(augmented_path) and augmented_dirname.startswith("augment"):
                feats_file = self.__read_sat_folder(augmented_path, extension)
                if(feats_file):
                    self._language_scheme[language_name].append(feats_file)
                else:
                    print("No sat.scp encountered in  "+augmented_path+" for language "+language_name)
                    print(debug.get_debug_info())
                    print("exiting... \n")
                    sys.exit()

    def __read_sat_folder(self, data_dir, extension):

        path_to_local = os.path.join(data_dir, "sat_local.%s" % (extension))

        if(os.path.isfile(path_to_local)):
            return path_to_local
        else:
            print("path: "+path_to_local+" does not exists. It is needed by sat")
            print(debug.get_debug_info())
            print("exiting... \n")
            sys.exit()
            return None








