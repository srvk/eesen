#                                                                                                                                                                                                                                             # LRScheduler Base Class
#                                                                                                                                                                                                                                             
# Author: Eric Fosler-Lussier                                                                                                                                                                                                                 

# Provides base definition for class, common auxiliary functions
 
class LRScheduler():

    def __init__(self, config):
        pass

    def initialize_training(self):
        pass

    def update_lr_rate(self, cv_ters):
        pass

    def set_epoch(self, epoch):
        pass

    def resume_from_log(self):
        pass

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

