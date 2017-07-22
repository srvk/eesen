import random, sys, os
import numpy as np
from fileutils.kaldi import writeArk, readMatrixByOffset

def run_reader_queue(queue, reader_x, ys, reader_sat=None):

    print("hello run reader queue")
    idx_shuf = list(range(len(xinfo)))
    if do_shuf:
        random.shuffle(idx_shuf)
    for idx_batch in idx_shuf:
        x = reader_x.read(idx_batch)
        y = ys[idx_batch]
        if(reader_sat):
            sat=reader_sat.read(idx_batch)
            q.put((x, y, sat))
        else:
            q.put((x, y))
    q.put(None)

