import numpy as np
import sys, random
import random
import sys
import numpy as np
import constants
from utils.fileutils.debug import get_debug_info

class Augmenter(object):
    def __init__(self, online_augment_config):
        self.online_augment_config=online_augment_config

    def preprocess(self, feat_info):

        #TODO @florian here you can have more room to play with augmentation option
        if(self.online_augment_config[constants.AUGMENTATION.WINDOW] and self.online_augment_config[constants.AUGMENTATION.WINDOW]):

            print("Augmenting data x" + str(self.online_augment_config[constants.AUGMENTATION.SUBSAMPLING])+" and win "+str(self.online_augment_config[constants.AUGMENTATION.WINDOW])+"...\n")

            factor=self.online_augment_config[constants.AUGMENTATION.SUBSAMPLING]

            win=self.online_augment_config[constants.AUGMENTATION.WINDOW]

            feat_info = [
                (tup[0], tup[1], tup[2], (int(tup[3]) + factor - 1 - shift) // factor, win * tup[4], (shift, factor, win))
                for shift in range(factor) for tup in feat_info]
        else:

            feat_info = [tup+(None,) for tup in feat_info]

        return feat_info

    def augment(self, feat, augment):

        shift = augment[0]
        stride = augment[1]
        win = augment[2]
        #stride=3
        #shift=augment

        if win == 1:
            augmented_feats = feat[shift::stride,]
        elif win == 2:
            augmented_feats = np.concatenate((np.roll(feat,1,axis=0), feat), axis=1)[shift::stride,]
        elif win == 3:
            augmented_feats = np.concatenate((np.roll(feat,1,axis=0), feat, np.roll(feat,-1,axis=0)), axis=1)[shift::stride,]
        elif win == 5:
            augmented_feats = np.concatenate((np.roll(feat,2,axis=0), np.roll(feat,1,axis=0), feat, np.roll(feat,-1,axis=0), np.roll(feat,-2,axis=0)), axis=1)[shift::stride,]
        elif win == 7:
            augmented_feats = np.concatenate((np.roll(feat,3,axis=0), np.roll(feat,2,axis=0), np.roll(feat,1,axis=0), feat, np.roll(feat,-1,axis=0), np.roll(feat,-2,axis=0), np.roll(feat,-3,axis=0)), axis=1)[shift::stride,]
        elif win == 9:
            augmented_feats = np.concatenate(
                (np.roll(feat,4,axis=0),
                 np.roll(feat,3,axis=0),
                 np.roll(feat,2,axis=0),
                 np.roll(feat,1,axis=0), feat,
                 np.roll(feat,-1,axis=0),
                 np.roll(feat,-2,axis=0),
                 np.roll(feat,-3,axis=0),
                 np.roll(feat,-4,axis=0)
                 ), axis=1)[shift::stride,]
        else:
            print("win not supported", win)
            print(get_debug_info())
            print("exiting...")
            sys.exit()

        #applying roll to augmented feats
        if self.online_augment_config["roll"]:
            augmented_feats = np.roll(augmented_feats, random.randrange(-2,2,1), axis = 0)

        return augmented_feats

