import sys

from reader.labels_reader.labels_reader_txt import LabelsReaderTxt
from utils.fileutils import debug


def create_reader(info_set, info_format, conf, batches_id, language_scheme = None):

    #read labels with txt format
    if info_format == "txt": return LabelsReaderTxt(info_set, conf, batches_id, language_scheme)

    else:
        print(info_format+" is not defined as \"info_format\" in \"info_set\": "+info_set)
        print(debug.get_debug_info())
        print("exiting...")
        sys.exit()
