from __future__ import print_function
from lm_reader.reader_queue import run_reader_queue
from lm_models.rnn import *
from lm_models.lm_model_factory import lm_create_model

import math
import re
import random
import time
from multiprocessing import Process, Queue
import lm_constants
import numpy as np


def train(all_readers,config):

    start_ = time.time()
    tf.set_random_seed(config[lm_constants.CONF_TAGS.RANDOM_SEED])
    random.seed(config[lm_constants.CONF_TAGS.RANDOM_SEED])

    #defining the model
    model =lm_create_model(config)


    #starting tf session...
    with tf.Session() as sess:


        tf.global_variables_initializer().run()

        data_queue = Queue(config[lm_constants.CONF_TAGS.BATCH_SIZE])

        tr_x, cv_x, tr_sat, cv_sat = all_readers

        saver, alpha = restore_weights(config, sess)

        for epoch in range(alpha, config[lm_constants.CONF_TAGS.NEPOCH]):

            ntrain_w = 0
            ntrain = 0

            train_losses = []


            train_cost, train_ters, ntrain = train_epoch(sess, model, data_queue, epoch,  lr_rate, tr_x, tr_y, tr_sat)


            dev_start = time.time()
            test_losses = []
            test_words = 0
            print('Testing on dev set...')

            if(config['adaptation_stage'] == "unadapated"):
                p = Process(target = run_reader_queue, args = (data_queue, cv_x, config["do_shuf"]))
            else:
                print("non addapted cv!!")
                p = Process(target = run_reader_queue, args = (data_queue, cv_x, config["do_shuf"], cv_sat))

            p.start()

            p.join()
            p.terminate()

            nll = sum(test_losses) / test_words

            print('ITER %d, Dev Loss: %.4f, ppl=%.4f wps: %.4f' % (it,nll, math.exp(nll), test_words/(time.time() - dev_start)))

            save_path = saver.save(sess, model_dir + 'model' + str(it) + '.ckpt')
            print('Model SAVED for ITER - ' , ITER)


def train_epoch(sess, model, epoch, data_queue, lr_rate, tr_x, tr_y, tr_sat):

    if(config[lm_constants.CONF_TAGS.SAT_SATGE] == lm_constants.SAT_SATGES.UNADAPTED):
        p = Process(target = run_reader_queue, args = (data_queue, tr_x, config[lm_constants.CONF_TAGS.DO_SHUF]))
    else:
        p = Process(target = run_reader_queue, args = (data_queue, tr_x, config[lm_constants.CONF_TAGS.DO_SHUF], tr_sat))

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


        feed_dict = prepare_feed_dict(model, config, batch_x, batch_len, batch_sat)

        train_cost, _ = sess.run([model.loss, model.optimizer], feed_dict = feed_dict)

        #train_cost, train_ters, ntrain = self.__train_epoch(epoch, lr_rate, tr_x, tr_y, tr_sat)

        update_counters(train_cost, )

        tot_words = sum(batch_len) - len(batch_len)

        train_losses.append(train_loss * tot_words)
        train_words += tot_words

    p.join()
    p.terminate()

def eval_epoch(sess, model, config, data_queue):

    while True:

        data = data_queue.get()
        if data is None:
            break

        if(config[lm_constants.CONF_TAGS.SAT_SATGE] == lm_constants.SAT_SATGES.UNADAPTED):

            batch_len, batch_x = data
            batch_sat=None

        else:
            batch_len, batch_x, batch_sat = data


        prepare_feed_dict(model, config, batch_x, batch_len, batch_sat)

        #third argument is to keep the state during training
        test_loss = sess.run(model.loss, feed_dict={model.x_input: batch_x,
                                                    model.x_lens: batch_len,
                                                    model.sat_input: batch_sat,
                                                    model.state:np.zeros((len(batch_len), 2*num_layers*hidden_size)),
                                                    model.keep_prob : 1.0})
        tot_words = sum(batch_len) - len(batch_len)
        test_losses.append(test_loss * tot_words)
        test_words += tot_words


#self.__update_counters(train_ters, train_cost, ntrain, ntr_labels, batch_ters, batch_cost, batch_size, data[1])
def update_counters(acum_ter, acum_cost, acum_utt, acum_words, batch_ter, batch_cost, batch_size):




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


def prepare_feed_dict(model, config, batch_x, batch_len, batch_sat = None):

    feed_dict = {}

    feed_dict[model.x_input] = batch_x
    feed_dict[model.x_lens] = batch_len
    feed_dict[model.state] = np.zeros((len(batch_x), 2*config[lm_constants.CONF_TAGS.NLAYERS]*config[lm_constants.CONF_TAGS.NHIDDEN]))
    feed_dict[model.drop_out] = config[lm_constants.CONF_TAGS.DROPOUT]

    if(batch_sat):
        feed_dict[model.sat_input] = batch_sat

    return feed_dict

def generate_logs(self, config, cv_ters, cv_cost, ncv, train_ters, train_cost, ntrain, epoch, lr_rate, tic):


    with open("%s/epoch%02d.log" % (config[lm_constants.CONF_TAGS.TRAIN_DIR], epoch + 1), 'w') as fp:

        fp.write("Time: %.0f minutes, lrate: %.4g\n" % ((time.time() - tic)/60.0, lr_rate))
        print("\t\t Train    cost: %.1f, ter: %.1f%%, #example: %d" % (train_cost, 100.0*train_ters, ntrain))
        print("\t\t Validate cost: %.1f, ter: %.1f%%, #example: %d" % (cv_cost, 100.0*cv_ter, ncv))
        fp.write("\t\tTrain    cost: %.1f, ter: %.1f%%, #example: %d\n" % (train_cost, 100.0*train_ters, ntrain))
        fp.write("\t\tValidate cost: %.1f, ter: %.1f%%, #example: %d\n" % (cv_cost, 100.0*cv_ter, ncv))
