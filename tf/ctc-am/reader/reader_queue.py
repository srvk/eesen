import random


#TODO we should do something like: def run_reader_queue(queue, do_shuf, reader_x=None, reader_y=None, reader_sat=None):
#TODO after changing this signature all calls have to be made using the key_id of argument

def run_reader_queue(queue, reader_x, reader_y, do_shuf, is_debug, seed, reader_sat= None):

    random.seed(seed)

    idx_shuf = list(range(reader_x.get_num_batches()))

    if do_shuf:
        random.shuffle(idx_shuf)

    count = 0
    for idx_batch in idx_shuf:
        x = reader_x.read(idx_batch)

        if(reader_y):
            y = reader_y.read(idx_batch)

        if(reader_sat):
            if(reader_y):
                #x, y, sat
                sat=reader_sat.read(idx_batch)
                queue.put((x, y, sat))

            else:
                #x, sat (for testing)
                sat=reader_sat.read(idx_batch)
                queue.put((x, sat))
        else:
            if(reader_y):
                #x, y
                queue.put((x, y))
            else:
                #x (for training)
                queue.put(x)

        if(count > 10 and is_debug):
            break
        count = count +1
    queue.put(None)

