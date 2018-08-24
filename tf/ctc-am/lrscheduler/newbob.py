#
# Halvsies learning rate schedule (from default tf_train schedule)
#
# Author: Eric Fosler-Lussier

# This implements a training schedule that depends on the average token error rate (TER) of the CV set
# 
#  - initial learning rate is given by LR_RATE in config
#  - learning rate stays constant for the first HALF_AFTER epochs
#  - learn rate remains constant until no improvement (or improvement less than threshold)
#  - learn rate halves until no improvement (or improvement less than threshold)

import constants
from lrscheduler.lrscheduler import LRScheduler

class Newbob(LRScheduler):
    def __init__(self, config):
        LRScheduler.__init__(self,config)
        self.__config = config
        self.__lr_rate = self.__config[constants.CONF_TAGS.LR_RATE]
        self.__nepoch = self.__config[constants.CONF_TAGS.NEPOCH]
        self.__half_after = self.__config[constants.CONF_TAGS.HALF_AFTER]
        self.__phase = 0  # 0: burn in (constant), 1: constant until threshold, 2: ramping
        self.__epoch = 1
        self.__best_avg_ters = 1000000000000000000.0
        self.__best_epoch = None
        self.__ramp_threshold = 0.0
        self.__stop_threshold = 0.0
        self.__min_lr_rate = self.__config[constants.CONF_TAGS.MIN_LR_RATE]
        self.__half_rate = self.__config[constants.CONF_TAGS.HALF_RATE]
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
                elif (var == "half_after"):
                    self.__half_after=int(value)
                elif (var == "ramp_threshold"):
                    self.__ramp_threshold=float(value)
                elif (var == "stop_threshold"):
                    self.__stop_threshold=float(value)
                elif (var == "min_lr_rate"):
                    self.__min_lr_rate=float(value)
                elif (var == "half_rate"):
                    self.__half_rate=float(value)
                else:
                    print("Unknown option %s in Newbob lr_spec." % var)
                    print('  lr_spec="%s"' % self.__config[constants.CONF_TAGS.LR_SPEC])
                    print('  example options: lr_rate=0.05,nepoch=30,half_after=8,ramp_threshold=0.0,stop_threshold=0.0,min_lr_rate=0.0005,half_rate=0.5')
                    sys.exit()
            except ValueError:
                print("Newbob lr_spec: invalid type for var %s" % var)
                print('  lr_spec="%s"' % self.__config[constants.CONF_TAGS.LR_SPEC])
                print('  example options: lr_rate=0.05,nepoch=30,half_after=8,ramp_threshold=0.0,stop_threshold=0.0,min_lr_rate=0.0005,half_rate=0.5')
                sys.exit()
                

    # TO DO: handle restarts
    def initialize_training(self):
        return self.__epoch, self.__lr_rate

    def update_lr_rate(self, cv_ters):
        # when there are multiple ters, compute average
        avg_ters = self.__compute_avg_ters(cv_ters)

        should_stop = False 
        restore = None

        # original learning rate scheduler runs for a fixed number of iterations
        if (self.__epoch+1 >= self.__nepoch):
            print("LRScheduler.Newbob: reached last epoch, ending training")
            should_stop=True

        if (self.__epoch < self.__half_after):
            if (not should_stop):
                print("LRScheduler.Newbob: not updating learning rate for first %s epochs" % str(self.__half_after))
            if (self.__epoch + 1 <= self.__half_after):
                self.__best_avg_ters = avg_ters
                self.__best_epoch = self.__epoch
                self.__phase = 1  # start constant
            self.__epoch = self.__epoch+1
            return self.__epoch, self.__lr_rate, should_stop, restore


        elif (self.__lr_rate <= self.__min_lr_rate):
            if (not should_stop):
                print("LRScheduler.Newbob: not updating learning rate, currently at minimum ", self.__min_lr_rate)

        elif (self.__phase == 1):
            if (self.__best_avg_ters - avg_ters >= self.__ramp_threshold):
                if (not should_stop):
                    print("LRScheduler.Newbob: learning rate remaining constant %.4g, TER improved %.1f%% from epoch %d" % (self.__lr_rate, 100.0*(self.__best_avg_ters-avg_ters), self.__best_epoch))
            else:
                if (not should_stop):
                    self.__lr_rate = self.__lr_rate * self.__half_rate
                    if (self.__lr_rate < self.__min_lr_rate):
                        self.__lr_rate = self.__min_lr_rate
                    self.__phase = 2
                    print("LRScheduler.Newbob: beginning ramping to learn rate %.4g, TER difference %.1f%% under threshold %.1f%% from epoch %d" % (self.__lr_rate, 100.0*(self.__best_avg_ters-avg_ters), self.__ramp_threshold, self.__best_epoch))
                    if (self.__best_avg_ters <= avg_ters):
                        restore = self.__best_epoch
                    
        else:  # phase == 2
            if (self.__best_avg_ters - avg_ters >= self.__stop_threshold):
                self.__lr_rate = self.__lr_rate * self.__half_rate
                if (self.__lr_rate < self.__min_lr_rate):
                    self.__lr_rate = self.__min_lr_rate
                if (not should_stop):
                    print("LRScheduler.Newbob: learning rate ramping to %.4g, TER improved %.1f%% from epoch %d" % (self.__lr_rate, 100.0*(self.__best_avg_ters-avg_ters), self.__best_epoch))
            else:
                print("LRScheduler.Newbob: stopping training, TER difference %.1f%% under threshold %.1f%% from epoch %d" % (100.0*(self.__best_avg_ters-avg_ters), self.__ramp_threshold, self.__best_epoch))
                should_stop = True
                if (self.__best_avg_ters <= avg_ters):
                    restore = self.__best_epoch


        if(self.__best_avg_ters > avg_ters):
            self.__best_avg_ters = avg_ters
            self.__best_epoch = self.__epoch

        self.__epoch=self.__epoch+1

        return self.__epoch, self.__lr_rate, should_stop, restore


    def __compute_avg_ters(self, ters):
        nters=0
        avg_ters = 0.0
        for language_id, target_scheme in ters.items():
            for target_id, ter in target_scheme.items():
                if(ter > 0):
                    avg_ters += ter
                    nters+=1
        avg_ters /= float(nters)

        return avg_ters
