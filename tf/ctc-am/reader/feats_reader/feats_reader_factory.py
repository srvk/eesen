import sys

from .feats_reader_kaldi import FeatsReaderKaldi
#from .feats_reader_video import FeatsReaderVideo
from utils.fileutils import debug


#it returns an object lm_reader that internaly will manage all the data
#client will be agnostic for the internals
#TODO we need to create a separate sat lm_reader
def create_reader(info_set, info_format, config):

    #TODO deduce file type
    if(not (info_set == "train" or info_set == "cv" or info_set == "test")):
        print("Error: info_set ( "+info_set+" ) is not contemplated")
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

    #sanity check for feats
    #read features with kaldi format
    if info_format == "kaldi": return FeatsReaderKaldi(info_set, config)
    if info_format == "video": return FeatsReaderVideo(info_set, config)

    else:
        print("Error: "+info_format+" is not defined as \"info_format\" in \"info_set\": "+info_set)
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

