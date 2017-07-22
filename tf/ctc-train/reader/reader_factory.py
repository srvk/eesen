from reader_kaldi import ReaderKaldi
from reader_hdf5 import ReaderHDF5
from reader_labels import ReaderLabels

#it returns an object reader that internaly will manage all the data
#client will be agnostic for the internals
def create_reader(info_set, info_type, info_format, args = None, batches_id = None):

    #TODO deduce file type

    #sanity check for feats
    if(info_type == "feats"):
        #read features with kaldi format
        if info_format == "kaldi":  return ReaderKaldi(info_set, args, batches_id)

        #read features with hdf5 format
        elif info_format == "hdf5": return ReaderHdf5(info_set, args, batches_id)

        else:
            print(info_format+" is not defined as \"info_format\" in \"info_type\": "+info_type)
            return None

    #sanity check for labels
    elif(info_type == "labels"):
        #read labels with txt format
        if info_format == "txt":  return ReaderLabels(info_set, args, batches_id)

        else:
            print(info_format+" is not defined as \"info_format\" in \"info_type\": "+info_type)
            return None
    else:
        print(info_set+" is not defined as \"info_set\"")
        return None
