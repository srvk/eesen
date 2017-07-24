import os
from reader.augment.augmentor import Augmentor

class FeatsReader(object):

    def __init__(self, data_dir, info_set, online_augment_config):

        #list of all folders
        self.list_files=[]
        self.info_set=info_set
        self.augmentor=Augmentor(online_augment_config)

        count_augmented_folder=0
        count_scp_files=0

        if (info_set == 'tr'):

            for augmented_folder in os.listdir(data_dir):
                if os.path.isdir(filename) and filename.startswith("augment"):
                    augmnet_folder += 1

            #there is sample augmentation
            if(augment_folder > 0):
                list_folders=read_augmentated_folders(data_dir, info_set)

            #there is not sample augmentation
            else:
                list_folders=read_scp_folder(data_dir, info_set)
        elif (info_set == 'cv'):

            self.list_folders.append(read_scp_folder(data_dir, info_set))

        else:
            print("info_set is not valid("+info_set+")")
            print("exiting...")
            sys.exit()



    def read_augmentated_folder(data_dir, info_set):

        for augmented_dirname in os.listdir(data_dir):

            augmented_path = os.path.join(data_dir,augmented_dirname)

            #it is an augmented folder
            if os.path.isdir(augmented_path) and augmented_dirname.startswith("augment"):
                self.list_files.append(read_scp_folder(augmented_path, info_set))


    def read_scp_folder(data_dir, info_set):

        return os.path.join(args.data_dir, "%s_local.scp" % (info_set))







