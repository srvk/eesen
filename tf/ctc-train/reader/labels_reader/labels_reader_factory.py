from labels_reader_txt import LabelsReaderTxt

def create_reader(info_set, info_format, conf, batches_id):

    #read labels with txt format
    if info_format == "txt":  return LabelsReaderTxt(info_set, conf, batches_id)

    else:
        print(info_format+" is not defined as \"info_format\" in \"info_type\": "+info_type)
        return None
