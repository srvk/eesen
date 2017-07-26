import os
import sys
from reader.augmenter.augmenter import Augmenter

class FeatsReader(object):

    def __init__(self, info_set, data_dir, online_augment_config, extension):

        #list of all folders
        self.list_files=[]
        self.info_set=info_set
        self.augmenter=Augmenter(online_augment_config)

        count_augmented_folder=0
        count_scp_files=0

        if (info_set == 'train'):

            for augmented_folder in os.listdir(data_dir):
                path_augmented=os.path.join(data_dir,augmented_folder)
                if os.path.isdir(path_augmented) and augmented_folder.startswith("augment_"):
                    count_augmented_folder += 1
                if not os.path.isdir(path_augmented) and augmented_folder.endswith(".scp"):
                    count_scp_files += 1

            #there is sample augmentation
            if(count_augmented_folder > 0):
                self.list_files=self.read_augmented_folder(data_dir, info_set, extension)

            #there is not sample augmentation
            else:
                print("not augmented data (mix) found ...")
                self.list_files.append(self.read_folder(data_dir, info_set, extension))

        elif (info_set == 'cv'):
            path_feats=self.read_folder(data_dir, info_set, extension)

            if(path_feats):
                self.list_files.append(path_feats)
            else:
                print("No feats.scp encountered in "+path_feats+" info_set: "+info_set)
                print("exiting")
                sys.exit()

        else:
            print("info_set is not valid ("+info_set+")")
            print("exiting...")
            sys.exit()

    def read_augmented_folder(self, data_dir, info_set, extension):

        list_files=[]
        for augmented_dirname in os.listdir(data_dir):

            augmented_path = os.path.join(data_dir,augmented_dirname)

            #it is an augmented folder
            if os.path.isdir(augmented_path) and augmented_dirname.startswith("augment"):
                feats_file = self.read_folder(augmented_path, info_set, extension)
                if(feats_file):
                    list_files.append(feats_file)

        if(len(list_files)==0):
            print("No feats.scp encountered in any augmented path")
            print("exiting")
            sys.exit()

        return list_files


    def read_folder(self, data_dir, info_set, extension):

        path_to_local=os.path.join(data_dir, "%s_local.%s" % (info_set, extension))
        if(os.path.isfile(path_to_local)):
            return path_to_local
        else:
            return None








