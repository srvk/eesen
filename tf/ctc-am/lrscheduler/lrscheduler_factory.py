#
# lrscheduler_factory.py: creates lrschedulers
#
# Author: Eric Fosler-Lussier

from lrscheduler.halvsies import *
from lrscheduler.newbob import *
from lrscheduler.constantlr import *
import constants


#it returns an object lrscheduler that internaly will manage all the data
#client will be agnostic for the internals
def create_lrscheduler(config):

    if config[constants.CONF_TAGS.LRSCHEDULER] == constants.LRSCHEDULER_NAME.HALVSIES:
        return Halvsies(config)
    elif config[constants.CONF_TAGS.LRSCHEDULER] == constants.LRSCHEDULER_NAME.NEWBOB:
        return Newbob(config)
    elif config[constants.CONF_TAGS.LRSCHEDULER] == constants.LRSCHEDULER_NAME.CONSTANTLR:
        return Constantlr(config)
    else:
        print("lrscheduler selected does not exist")
        print(debug.get_debug_info())
        print("exiting...\n")
        sys.exit()

