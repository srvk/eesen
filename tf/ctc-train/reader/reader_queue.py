import random, sys, os
import numpy as np
from fileutils.kaldi import writeArk, readMatrixByOffset

def run_reader_queue(queue, reader_x, reader_y, do_shuf, reader_sat=None):

    idx_shuf = list(range(reader_x.get_num_batches()))
    if do_shuf:
        random.shuffle(idx_shuf)
    for idx_batch in idx_shuf:
        x = reader_x.read(idx_batch)
        y = reader_y.read(idx_batch)
        if(reader_sat):
            sat=reader_sat.read(idx_batch)
            queue.put((x, y, sat))
        else:
            queue.put((x, y))
    queue.put(None)

