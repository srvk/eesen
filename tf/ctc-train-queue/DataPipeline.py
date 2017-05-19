import tensorflow as tf
import numpy as np

import os
import pdb
import progressbar
import sys
from fileutils.kaldi import readScp_by_id


'''
Experimental data pipeline to directly load videos
'''
class DataPipeline:
    '''
    Setup Queue and data preprocessor
    '''
    def __init__(self, data_dir, labels_file, batch_size, num_threads):

        self.data_dir=data_dir
        self._setup_data_pipeline(data_dir, labels_file, batch_size, num_threads)

    def encode_labels(self, labels):
        '''
        Serializes labels into sparse matrices in preparation for CTC

        Reference:
        http://stackoverflow.com/q/42578532
        '''
        serialized_labels = []

        print("sparsing... "+self.data_dir)

        bar = progressbar.ProgressBar(maxval=len(labels), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' (', progressbar.ETA(), ') '])
        bar.start()
        count=0
        for values in labels:

            # create SparseMatrix data
            indices = [[i] for i in range(0, len(values))]
            shape = [len(values)]

            label = tf.SparseTensorValue(indices, values, shape)
            label = tf.convert_to_tensor_or_sparse_tensor(label)

            # Must serialize for minibatching
            label = tf.serialize_sparse(label)
            serialized_labels.append(label)

            bar.update(count)
            count=count+1

        bar.finish()
        labels_out = tf.convert_to_tensor_or_sparse_tensor(serialized_labels)
        print("end sparsing...")
        return labels_out

    def read_label_file(self, label_filename, utterance_order):
        '''
        Expects a file where each line is an exemplar of format (kaldi style):
        {SPEAKER}.{FILE}-{META}-{META} {SPACE SEPARATED 1-INDEXED SEQUENCE}
        Where the gif files are located in:
        Returns a list of labels
          Each label is properly encoded by encode_labels().
          Each element is a list of 0-indexed integer labels.
          *****This allows NUM_CLASSES to be the blank symbol during CTC.*****
        '''
        final_labels=[[] for i in range(len(utterance_order))]

        with open(label_filename, 'rb') as f:
            for line in f:
                utt_id, labels= line.rstrip().split(' ', 1)
                final_labels[utterance_order[utt_id]]=[int(s)-1 for s in labels.split() if not s=='']


        return self.encode_labels(final_labels)


    def get_data_from_queue(self, input_queue):

        '''
        Reads a concrete utterance (using an scp line) from disk
        Consumes a single filename and label.
        Resizes video as necessary.
        Returns an video tensor and a label.
        video range [0,1]
        '''
        def read_scp(scp_line):
            return np.asarray(readScp_by_id(scp_line), dtype=np.float32)

        #the first component of the queue contains a pure line of tae pool of scp
        utterance = tf.py_func(read_scp, [input_queue[0]], [tf.float32])

        #TF need to know this in definition time
        utterance[0].set_shape([None, 40])

        #the second component contains the pure label sequence
        labels = input_queue[1]

        return utterance[0], labels

    def batch_ops(self):
        return self.audio_batch, self.label_batch

    def get_scp_file(self, scp_file):
        final_list=[]
        utterance_id={}
        count =0
        for element in open(scp_file).readlines():
            id=element.split()[0]
            utterance_id[id]=count
            final_list.append(element.split()[1])
            count=count+1
        return utterance_id, final_list


    def _setup_data_pipeline(self, scp_file, labels_file, batch_size, num_threads):
        '''
        Partitions data and sets up data queues
        Based off of code written:
        http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/
        '''
        #http://stackoverflow.com/questions/33932901/whats-the-purpose-of-tf-app-flags-in-tensorflow

        utterance_order, scp_list= self.get_scp_file(scp_file)

        labels = self.read_label_file(labels_file, utterance_order)

        print("converting labels and scp to tensors...")
        labels_t = tf.convert_to_tensor(labels)
        scp_list_t = tf.convert_to_tensor(scp_list)
        print("convertion done")

        ################
        # Create Queue #
        ################
        #the queue will peoduce in a infinite fashon
        input_queue = tf.train.slice_input_producer(
                [scp_list_t, labels_t],
                shuffle=False,
                name='train_producer',
                capacity=batch_size*5)

        ############################
        # Define Data Retrieval Op #
        ############################
        # these input queues automatically dequeue 1 slice
        utterance_feat, label = self.get_data_from_queue(input_queue)

        ################
        # Minibatching #
        ################
        #create an **OPERATION** for the feature vectors and labels
        self.audio_batch, label_batch = tf.train.batch(
                [utterance_feat, label],
                batch_size=batch_size,
                num_threads=num_threads,
                dynamic_pad=True)
        self.label_batch=tf.deserialize_many_sparse(label_batch, tf.int32)
