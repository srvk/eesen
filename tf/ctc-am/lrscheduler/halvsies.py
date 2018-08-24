#
# Halvsies learning rate schedule (from default tf_train schedule)
#
# Author: Eric Fosler-Lussier

# This implements a training schedule that depends on the average token error rate (TER) of the CV set
# 
#  - initial learning rate is given by LR_RATE in config
#  - learning rate stays constant for the first HALF_AFTER epochs
#  - until minimum learning rate is reached:
import constants
from lrscheduler.lrscheduler import LRScheduler
import re

class Halvsies(LRScheduler):
    def __init__(self, config):
        LRScheduler.__init__(self,config)
        self.__config = config
        self.__lr_rate = self.__config[constants.CONF_TAGS.LR_RATE]
        self.__epoch = 1
        self.__best_avg_ters = 1000000000000000000.0
        self.__best_epoch = None
        self.__nepoch = self.__config[constants.CONF_TAGS.NEPOCH]
        self.__half_after = self.__config[constants.CONF_TAGS.HALF_AFTER]
        self.__half_rate = self.__config[constants.CONF_TAGS.HALF_RATE]
        self.__min_lr_rate = self.__config[constants.CONF_TAGS.MIN_LR_RATE]
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
                elif (var == "min_lr_rate"):
                    self.__min_lr_rate=float(value)
                elif (var == "half_rate"):
                    self.__half_rate=float(value)
                else:
                    print("Unknown option %s in Halvsies lr_spec." % var)
                    print('  lr_spec="%s"' % self.__config[constants.CONF_TAGS.LR_SPEC])
                    print('  example options: lr_rate=0.05,nepoch=30,half_after=8,min_lr_rate=0.0005,half_rate=0.5')
                    sys.exit()
            except ValueError:
                print("Halvsies lr_spec: invalid type for var %s" % var)
                print('  lr_spec="%s"' % self.__config[constants.CONF_TAGS.LR_SPEC])
                print('  example options: lr_rate=0.05,nepoch=30,half_after=8,min_lr_rate=0.0005,half_rate=0.5')
                sys.exit()
                
        

    # TO DO: handle restarts
    def initialize_training(self):
        self.__status = "LRScheduler.Halvsies: initialized"
        return self.__epoch, self.__lr_rate

    def __set_status(self,string):
        self.__status=string

    def get_status(self):
        return self.__status

    def update_lr_rate(self, cv_ters):
        # when there are multiple ters, compute average
        avg_ters = self.__compute_avg_ters(cv_ters)

        should_stop = False 
        restore = None

        # original learning rate scheduler runs for a fixed number of iterations
        if (self.__epoch >= self.__nepoch):
            self.__set_status("LRScheduler.Halvsies: reached last epoch, ending training")
            should_stop=True

        if (self.__epoch < self.__half_after):
            if (not should_stop):
                self.__set_status("LRScheduler.Halvsies: not updating learning rate for first %s epochs" % str(self.__half_after))
            if (self.__epoch + 1 <= self.__half_after):
                self.__best_avg_ters = avg_ters
                self.__best_epoch = self.__epoch
            self.__epoch = self.__epoch+1
            #print(self.get_status())
            return self.__epoch, self.__lr_rate, should_stop, restore


        elif (self.__lr_rate <= self.__min_lr_rate):
            if (not should_stop):
                self.__set_status("LRScheduler.Halvsies: not updating learning rate, currently at minimum ", self.__min_lr_rate)

        elif (self.__best_avg_ters > avg_ters):
            if (not should_stop):
                self.__set_status("LRScheduler.Halvsies: not updating learning rate, TER improved %.1f%% from epoch %d" % (100.0*(self.__best_avg_ters-avg_ters), self.__best_epoch))

        else:
            self.__lr_rate = self.__lr_rate * self.__half_rate
            if (self.__lr_rate < self.__min_lr_rate):
                self.__lr_rate = self.__min_lr_rate

            if (should_stop):
                self.__set_status("LRScheduler.Halvsies: TER worse by %.1f%% from %.1f%% in epoch %d, will restore" % (100.0*(avg_ters-self.__best_avg_ters), 100.0*self.__best_avg_ters, self.__best_epoch))
            else:
                self.__set_status("LRScheduler.Halvsies: TER worse by %.1f%% from %.1f%% in epoch %d, updating learning rate to %.4g" % (100.0*(avg_ters-self.__best_avg_ters), 100.0*self.__best_avg_ters, self.__best_epoch, self.__lr_rate))
            restore = self.__best_epoch

        if(self.__best_avg_ters > avg_ters):
            self.__best_avg_ters = avg_ters
            self.__best_epoch = self.__epoch

        self.__epoch=self.__epoch+1

        #print(self.get_status())
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

    def set_epoch(self, epoch):
        self.__epoch = epoch

    def resume_from_log(self):
        alpha = int(re.match(".*epoch([-+]?\d+).ckpt", self.__config[constants.CONF_TAGS.CONTINUE_CKPT]).groups()[0])
        
        self.__epoch = alpha + 1
        self.__best_epoch = alpha  # technically this is not correct, should search logs

        num_val = 0
        acum_val = 0
        
        with open(self.__config[constants.CONF_TAGS.CONTINUE_CKPT].replace(".ckpt",".log")) as input_file:
            for line in input_file:
                if (constants.LOG_TAGS.VALIDATE in line):
                    acum_val += float(line.split()[4].replace("%,",""))
                    num_val += 1
                        
        self.__best_avg_ters=(acum_val / num_val)/100.0



