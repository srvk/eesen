from __future__ import print_function
from lm_reader.reader_queue import run_reader_queue
from lm_models.rnn import *
from lm_models.lm_model_factory import lm_create_model

import math
import time
import re
import random
import time
from multiprocessing import Process, Queue
import lm_constants
import numpy as np
import os

#hide pool alocator warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train(all_readers, config):

    tf.set_random_seed(config[lm_constants.CONF_TAGS.RANDOM_SEED])
    random.seed(config[lm_constants.CONF_TAGS.RANDOM_SEED])

    #defining the model
    model = lm_create_model(config)
    lr_rate = config[lm_constants.CONF_TAGS.LR_RATE]


    #starting tf session...
    with tf.Session() as sess:


        tf.global_variables_initializer().run()

        data_queue = Queue(config[lm_constants.CONF_TAGS.BATCH_SIZE])

        tr_x, cv_x, tr_sat, cv_sat = all_readers

        saver, alpha = restore_weights(config, sess)

        for epoch in range(alpha, config[lm_constants.CONF_TAGS.NEPOCH]):

            print(80 * "-")
            print("Epoch "+str(epoch)+" starting ...")
            print(80 * "-")
            tic = time.time()

            train_cost, ntrain, ntrain_w = train_epoch(sess, model, config, data_queue, epoch, lr_rate, tr_x, tr_sat)

            cv_cost, ncv, ncv_w = eval_epoch(sess, model, config, data_queue, cv_x, cv_sat)

            generate_logs(config, cv_cost, ncv, ncv_w, train_cost, ntrain, ntrain_w, epoch, lr_rate, tic)

            saver.save(sess, "%s/epoch%02d.ckpt" % (config[lm_constants.CONF_TAGS.TRAIN_DIR], epoch))

            #TODO LM update LR
            #lr_rate, best_avg_ters, best_epoch = self.__update_lr_rate(epoch, cv_ters, best_avg_ters, best_epoch, saver)

            print("Epoch "+str(epoch)+" done.")
            print(80 * "-")


def train_epoch(sess, model, config, data_queue, epoch,  lr_rate, tr_x, tr_sat):

    p = Process(target = run_reader_queue, args = (data_queue, tr_x, config[lm_constants.CONF_TAGS.DEBUG], config[lm_constants.CONF_TAGS.DO_SHUF], tr_sat))

    p.start()

    train_cost, ntrain, ntrain_w = 0.0, 0, 0

    while True:

        data = data_queue.get()
        if data is None:
            break

        if(config[lm_constants.CONF_TAGS.SAT_SATGE] == lm_constants.SAT_SATGES.UNADAPTED):

            batch_len, batch_x = data
            batch_sat=None

        else:
            batch_len, batch_x, batch_sat = data


        feed_dict = prepare_feed_dict(model, config, batch_x, batch_len, 1.0, batch_sat)

        batch_cost, _ = sess.run([model.loss, model.optimizer], feed_dict = feed_dict)

        train_cost, ntrain, ntrain_w = update_counters(train_cost, ntrain, ntrain_w, batch_cost, batch_len)

    p.join()
    p.terminate()

    return train_cost, ntrain, ntrain_w

def eval_epoch(sess, model, config, data_queue, cv_x, cv_sat):

    cv_cost, ncv, ncv_w = 0.0, 0, 0

    p = Process(target = run_reader_queue, args = (data_queue, cv_x, config[lm_constants.CONF_TAGS.DEBUG], config[lm_constants.CONF_TAGS.DO_SHUF], cv_sat))

    p.start()

    while True:

        data = data_queue.get()

        if data is None:
            break

        if(config[lm_constants.CONF_TAGS.SAT_SATGE] == lm_constants.SAT_SATGES.UNADAPTED):

            batch_len, batch_x = data
            batch_sat=None
        else:
            batch_len, batch_x, batch_sat = data

        feed_dict = prepare_feed_dict(model, config, batch_x, batch_len, 1.0, batch_sat)

        #third argument is to keep the state during training
        batch_cost = sess.run(model.loss, feed_dict = feed_dict)

        cv_cost, ncv, ncv_w, = update_counters(cv_cost, ncv, ncv_w, batch_cost, batch_len)

    return cv_cost, ncv, ncv_w

#self.__update_counters(train_ters, train_cost, ntrain, ntr_labels, batch_ters, batch_cost, batch_size, data[1])
def update_counters(acum_cost, acum_utt, acum_words, batch_cost, batch_len):

    batch_size = len(batch_len)
    total_num_units = sum(batch_len) - batch_size

    #update counters
    acum_words += total_num_units
    acum_cost += (batch_cost * total_num_units)
    acum_utt += batch_size

    return acum_cost, acum_utt, acum_words

def restore_weights(config, sess):

    if config[lm_constants.CONF_TAGS.CONTINUE_CKPT] != "":

        print(80 * "-")
        print("restoring weights....")
        print(80 * "-")
        if(config[lm_constants.CONF_TAGS.SAT_SATGE] != lm_constants.SAT_SATGES.UNADAPTED):
            print("partial restoring....")
            var_list=[]
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                if ("Shift" not in var.name):
                    var_list.append(var)
        else:
            print("total restoring....")
            var_list=[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

        saver = tf.train.Saver(max_to_keep = config[lm_constants.CONF_TAGS.NEPOCH], var_list=var_list)
        saver.restore(sess, config[lm_constants.CONF_TAGS.CONTINUE_CKPT])
        alpha = int(re.match(".*epoch([-+]?\d+).ckpt", config[lm_constants.CONF_TAGS.CONTINUE_CKPT]).groups()[0])
    else:
        saver = tf.train.Saver(max_to_keep = config[lm_constants.CONF_TAGS.NEPOCH])
        alpha=0

    return saver, alpha


def prepare_feed_dict(model, config, batch_x, batch_len, keep_prob, batch_sat = None):

    feed_dict = {}

    feed_dict[model.x_input] = batch_x
    feed_dict[model.x_lens] = batch_len
    feed_dict[model.state] = np.zeros((len(batch_x), 2*config[lm_constants.CONF_TAGS.NLAYERS]*config[lm_constants.CONF_TAGS.NHIDDEN]))
    feed_dict[model.drop_out] = keep_prob

    if(batch_sat):
        feed_dict[model.sat_input] = batch_sat

    return feed_dict

def generate_logs(config, cv_cost, ncv, ncv_w, train_cost, ntrain, ntrain_w, epoch, lr_rate, tic):

    with open("%s/epoch%02d.log" % (config[lm_constants.CONF_TAGS.TRAIN_DIR], epoch + 1), 'w') as fp:

        print("Epoch %d finished in %.0f minutes, learning rate: %.4g (%s)" % (epoch, (time.time() - tic)/60.0, lr_rate, config[lm_constants.CONF_TAGS.OPTIMIZER]))
        fp.write("Time: %.0f minutes, lrate: %.4g\n" % ((time.time() - tic)/60.0, lr_rate))
        print("\t\t Train cost: %.1f, ppl: %.1f, #examples: %d, #tokens: %d\n" % (train_cost/ntrain_w, math.exp(train_cost/ntrain_w), ntrain, ntrain_w))
        print("\t\t Validate cost: %.1f, ppl: %.1f, #examples: %d, #tokens: %d\n" % (cv_cost/ncv_w, math.exp(cv_cost/ncv_w), ncv, ncv_w))
        fp.write("\t\tTrain cost: %.1f, ppl: %.1f, #examples: %d, #tokens: %d\n" % (train_cost/ntrain_w, math.exp(train_cost/ntrain_w), ntrain, ntrain_w))
        fp.write("\t\tValidate cost: %.1f, ppl: %.1f, #examples: %d #tokens: %d\n" % (cv_cost/ncv_w, math.exp(cv_cost/ncv_w), ncv, ncv_w))
