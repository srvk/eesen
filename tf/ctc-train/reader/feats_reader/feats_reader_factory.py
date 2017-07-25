from feats_reader_kaldi import FeatsReaderKaldi
from feats_reader_hdf5 import FeatsReaderHDF5

#it returns an object reader that internaly will manage all the data
#client will be agnostic for the internals
def create_reader(info_set, info_format, data_dir, lstm_type, online_augment_config, batch_size):

    #TODO deduce file type

    #sanity check for feats
    #read features with kaldi format
    if info_format == "kaldi":  return FeatsReaderKaldi(info_set, data_dir, lstm_type, online_augment_config, batch_size)

    #read features with hdf5 format
    # elif info_format == "hdf5": return FeatsReaderHDF5(info_set, args)

    else:
        print(info_format+" is not defined as \"info_format\" in \"info_type\": "+info_type)
        return None

