#
# Halvsies learning rate schedule (from default tf_train schedule)
#
# Author: Eric Fosler-Lussier

# This implements a constant training schedule
# 

import constants
from lrscheduler.lrscheduler import LRScheduler
import re

class Constantlr(LRScheduler):
    def __init__(self, config):
        LRScheduler.__init__(self,config)
        self.__config = config
        self.__lr_rate = self.__config[constants.CONF_TAGS.LR_RATE]
        self.__epoch = 1
        self.__nepoch = self.__config[constants.CONF_TAGS.NEPOCH]
        if (self.__config[constants.CONF_TAGS.LR_SPEC] is not ""):
            self.__parse_spec()
        
    # process spec string
    def __parse_spec(self):
        for spec in self.__config[constants.CONF_TAGS.LR_SPEC].split(","):
            var, value = spec.split("=")
            try:
                if (var == "lr_rate"):
                    self.__lr_rate=float(value)
                elif (var == "nepoch"):
                    self.__nepoch=int(value)
                else:
                    print("Unknown option %s in Constantlr lr_spec." % var)
                    print('  lr_spec="%s"' % self.__config[constants.CONF_TAGS.LR_SPEC])
                    print('  example options: lr_rate=0.05,nepoch=10')
                    sys.exit()
            except ValueError:
                print("Constantlr lr_spec: invalid type for var %s" % var)
                print('  lr_spec="%s"' % self.__config[constants.CONF_TAGS.LR_SPEC])
                print('  example options: lr_rate=0.05,nepoch=10')
                sys.exit()
        

    # TO DO: handle restarts
    def initialize_training(self):
        self.__set_status("LRScheduler.Constantlr: initialized")
        return self.__epoch, self.__lr_rate

    def __set_status(self,string):
        self.__status=string

    def get_status(self):
        return self.__status

    def update_lr_rate(self, cv_ters):
        self.__epoch = self.__epoch + 1

        should_stop = False 
        restore = None

        if (self.__epoch >= self.__nepoch+1):
            self.__set_status("LRScheduler.Constantlr: reached last epoch, ending training")
            should_stop=True
        else:
            self.__set_status("LRScheduler.Constantlr: continuing training")
        
        return self.__epoch, self.__lr_rate, should_stop, restore

    def set_epoch(self, epoch):
        self.__epoch = epoch

    def resume_from_log(self):
        # this is pretty simple - allow new lr to replace old, so only need the epoch set
        alpha = int(re.match(".*epoch([-+]?\d+).ckpt", self.__config[constants.CONF_TAGS.CONTINUE_CKPT]).groups()[0])
        self.__epoch = alpha + 1

