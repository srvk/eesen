#
# Halvsies learning rate schedule (from default tf_train schedule)
#
# Author: Eric Fosler-Lussier

# This implements a constant training schedule
# 

import constants
from lrscheduler.lrscheduler import LRScheduler

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
                print("Halvsies lr_spec: invalid type for var %s" % var)
                print('  lr_spec="%s"' % self.__config[constants.CONF_TAGS.LR_SPEC])
                print('  example options: lr_rate=0.05,nepoch=10')
                sys.exit()
        

    # TO DO: handle restarts
    def initialize_training(self):
        return self.__epoch, self.__lr_rate

    def update_lr_rate(self, cv_ters):
        self.__epoch = self.__epoch + 1

        should_stop = False 
        restore = None

        if (self.__epoch+1 >= self.__nepoch):
            print("LRScheduler.Constantlr: reached last epoch, ending training")
            should_stop=True
            
        return self.__epoch, self.__lr_rate, should_stop, restore

