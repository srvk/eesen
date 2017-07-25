import sys, os
import pdb
import numpy as np

class LabelsReaderTxt():
    def __init__(self, info_set, args, batches_id):

        #permanent list to store the number of classes of each language
        self.target_scheme={}

        #temporary dctionary from tag to targets
        all_label_dicts={}

        #hack to get the correct name for the file
        if(info_set=='train'):
            info_set='tr'

        #getting all labels files in a dictonary (key: target_name, value: filename)
        all_labels_file={}
        for filename in os.listdir(args.data_dir):
            if (filename.startswith('labels_') and filename.endswith('.'+info_set)):
                #filename will: 'labels.tr' or 'labels.cv'
                target_id=filename.replace("labels_","").replace('.'+info_set,"")
                if(target_id==""):
                    target_id="no_name_target"

                all_labels_file[target_id] = os.path.join(args.data_dir,filename)

        #loading all labels in a dictionary and number of classes
        for target_id, path_to_target in all_labels_file.iteritems():
            ntarget, label_dict = self.__load_labels(path_to_target)
            self.target_scheme[target_id] = ntarget
            all_label_dicts[target_id] = label_dict

        if(not batches_id):
            print("Error: batches_id need to be provided in label reader")
            sys.exit()
        else:
            self.batches_y = self.__order_labels(all_label_dicts, batches_id)

    #getter
    def get_target_scheme(self):
        return self.target_scheme

    #read batch idx. Input: batch index. Output: batch with the number of languages available
    #TODO this has been done in case some day we need a online reader online (e.g. ReaderLabelKaldi, ReaderLabelHDF5)
    def read(self, idx):
        return self.batches_y[idx]

    #get filenmae an
    def __load_labels(self, filename, nclass=0):
        """
        Load a set of labels in (local) Eesen format
        """

        # mapLabel = lambda x: x - 1
        mapLabel = lambda x: x - 0
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


    def __order_labels(self, all_label_dicts, batches_id):

        #final batches list
        batches_y=[]

        #iterate over all batches
        for batch_id in batches_id:

            #declare counters and target batches
            #yidx: index list of a sparse matrix
            #yval: list of values that corresponds to the previous index list
            #max_label_len: maximum length value in the batch
            yidx, yval, max_label_len = {}, {}, {}

            #initialize counters and target batches
            for target_id, _ in all_label_dicts.iteritems():
                yidx[target_id]=[]
                yval[target_id]=[]
                max_label_len[target_id]=0

            #iterate over all element of a batch
            for i, uttid in enumerate(batch_id):

                #iterate over all target dictionaries (languages)
                for target_id, label_dict in all_label_dicts.iteritems():

                    #getting taget sequence from the current dictionary
                    label = label_dict[uttid]

                    #getting the max number of previous or current length
                    max_label_len[target_id] = max(max_label_len[target_id], len(label))

                    #fill the sparse batche (yidx: index, yval: corresponding value to this index)
                    for j in range(len(label)):
                        yidx[target_id].append([i, j])
                        yval[target_id].append(label[j])

            #construct the final batch
            batch_y={}
            for target_id, label_dict in all_label_dicts.iteritems():
                yshape_np = np.array([len(batch_id), max_label_len[target_id]], dtype = np.int32)
                yidx_np = np.asarray(yidx[target_id], dtype = np.int32)
                yval_np = np.asarray(yval[target_id], dtype = np.int32)
                batch_y[target_id]=((yidx_np, yval_np, yshape_np))

            #add the final batch to the inner list
            batches_y.append(batch_y)

        return batches_y






