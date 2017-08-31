import os
import sys
import numpy as np
import io

class LabelsReader(object):

    def __init__(self, path, batch_size):

        self.batches=[]

        self.__num_diff_labels = 0

        self.__read_batches(path, batch_size)

    def read(self, idx):

        lengths, utt_id, batch = zip(*self.batches[idx])

        #TODO check this hardcoded 42
        batch_padded = [self.__pad(example, self.__num_diff_labels, max(lengths)) for example in batch]

        return utt_id, batch_padded, lengths


    def get_uttid(self):

        uttid_batches=[]

        for idx in range(len(self.batches)):

            utt_batch, _, _ = self.read(idx)
            uttid_batches.append(utt_batch)

        return uttid_batches

    def get_num_batches(self):

        return len(self.batches)

    def get_num_diff_labels(self):

        return self.__num_diff_labels

    def update_num_diff_labels(self, new_num_diff_labels):

        self.__num_diff_labels = new_num_diff_labels

    def __read_batches(self, path, batch_size):


        lines_file = io.open(path, encoding='utf-8').readlines()
        tmp_list=[]

        dict_seen_char={}

        for line in lines_file:

            id_utt=line.split()[0]
            lst=line.split()[1:]

            tmp=[int(i)-1 for i in lst]
            for char in tmp:
                if char not in dict_seen_char:
                    dict_seen_char[char]=""
                    self.__num_diff_labels += 1


            tmp_list.append((len(tmp), id_utt, tmp))

        #TODO check this if correct
        #self.__num_diff_labels += 1

        tmp_list.sort(key=lambda x: x[0], reverse=True)


        idx = 0
        while idx < len(tmp_list):
            j = idx + 1

            while j < min(idx + batch_size, len(tmp_list)):
                j += 1

            batch = self.__make_batch(tmp_list, idx, j - idx)

            if(len(batch) == batch_size):
                self.batches.append(batch)

            idx = j


    def __make_batch (self, all_batches, start, height):

        batch=[]
        for idx in range(height):
            batch.append(all_batches[start+idx])
        return batch


    def __pad(self, seq, element, length):

        assert len(seq) <= length
        r = seq + [element] * (length - len(seq))
        assert len(r) == length

        return r

