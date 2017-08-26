import random

def run_reader_queue(queue, reader_x, do_shuf, reader_sat=None):

    idx_shuf = list(range(reader_x.get_num_batches()))

    if do_shuf:
        random.shuffle(idx_shuf)

    for idx_batch in idx_shuf:

        print(idx_batch)

        _, len, x = reader_x.read(idx_batch)

        if reader_sat:
            sat = reader_sat.read(idx_batch)
            queue.put((len, x, sat))
        else:
            queue.put((len, x))

    queue.put(None)

