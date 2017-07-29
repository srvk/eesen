import sys
from fileutils import debug
from feats_reader_kaldi import FeatsReaderKaldi


#it returns an object reader that internaly will manage all the data
#client will be agnostic for the internals
#TODO we need to create a separate sat reader
def create_reader(info_set, info_format, config, batches_id = None):

    #TODO deduce file type
    if(not (info_set == "sat" or info_set == "train" or info_set == "cv")):
        print("Error: info_set ( "+info_set+" ) is not contemplated")
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()
    else:
        if(info_set == "sat" and not batches_id):

            print("Error: info_set ( "+info_set+" ) need a batch_id structure for its proper consturction")
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()

        elif((info_set == "train" or info_set == "cv") and batches_id):

            print("Error: info_set ( "+info_set+" ) does not require a batch_id because it will create it")
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()

    #sanity check for feats
    #read features with kaldi format
    if info_format == "kaldi":  return FeatsReaderKaldi(info_set, config, batches_id)

    else:
        print("Error: "+info_format+" is not defined as \"info_format\" in \"info_set\": "+info_set)
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()

