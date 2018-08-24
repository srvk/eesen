import sys, os
import pdb
import numpy as np
from reader.labels_reader.labels_reader import LabelsReader

class LabelsReaderTxt(LabelsReader):

    def __init__(self, info_set, conf, batches_id, language_scheme):

        #constructing parent class and creating self.list_files and stroing self.__info_set
        super(LabelsReaderTxt, self).__init__(info_set, conf, batches_id, language_scheme)

    #get filenmae an
    def _load_dict(self, filename, nclass=0):
        """
        Load a set of labels in (local) Eesen format
        """
        mapLabel = lambda x: x - 1
        labels = {}
        m = 0

        with open(filename, "r") as f:
            for line in f:
                tokens = line.strip().split()
                labels[tokens[0]] = [mapLabel(int(x)) for x in tokens[1:]]
                try:
                    # this can be empty
                    if max(labels[tokens[0]]) > m:
                        m = max(labels[tokens[0]])
                except:
                    pass

        # sanity check - did we provide a value, and the actual is different?
        if nclass > 0 and m+2 != nclass:
            print("Warning: provided nclass=", nclass, " while observed nclass=", m+2)
            m = nclass-2
        return m+2, labels







