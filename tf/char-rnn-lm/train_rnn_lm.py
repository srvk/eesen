from __future__ import print_function
from RNN_Model import *
import tensorflow as tf
import os
import time
import random
import numpy as np
import sys
import math
from reader import labels_reader
from multiprocessing import Process, Queue
from reader.reader_queue import run_reader_queue


def train(all_readers,config):

    start_ = time.time()
    tf.set_random_seed(config["random_seed"])
    random.seed(config["random_seed"])

    #defining the model
    model = RNN_Model(config)
    nepoch = config["nepoch"]

    model_dir = config["exp_path"]

    if (model_dir.strip()[-1] == '/'):
        model_dir=model_dir[:-1]

    model_dir=model_dir+ "/saved_model/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    non_addapted_model_dir=config["exp_path"]+"/saved_model/"

    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]


    with tf.Session() as sess:


        if(config['adaptation_stage'] == "adapt_sat"):

            print("preparing to construct and train adaptation module")
            var_list=[]
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                if ("Shift" not in var.name):
                    var_list.append(var)
        else:
            print("train/fine-tune the complete model")
            var_list=[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

        saver = tf.train.Saver(max_to_keep = nepoch, var_list=var_list)

        for element in var_list:
            print(element.name)

        # writer = tf.summary.FileWriter(config["train_path"], sess.graph)
        tf.global_variables_initializer().run()

        if config['weight_path'] != "":
            saver.restore(sess, config['weight_path'])
            #if adaptation start again the counts
            config['cont']=0

        elif config['cont'] > 0:
            saver.restore(sess, non_addapted_model_dir + 'model' + str(config['cont'])+'.ckpt')


        data_queue = Queue(config["batch_size"])

        print('startup time: %r' % (time.time() - start_))

        i = dev_time = 0
        start_train = time.time()

        tr_x, cv_x, tr_sat, cv_sat = all_readers

        for ITER in range(nepoch):

            train_words = 0
            train_losses = []

            if(config['adaptation_stage'] == "unadapated"):
                p = Process(target = run_reader_queue, args = (data_queue, tr_x, config["do_shuf"]))
            else:
                p = Process(target = run_reader_queue, args = (data_queue, tr_x, config["do_shuf"], tr_sat))

            p.start()

            it = ITER+config['cont']+1

            start = time.time()

            while True:

                data = data_queue.get()

                if data is None:
                    break

                if(config['adaptation_stage'] == "unadapted"):

                    batch_len, batch_x = data
                    batch_sat=None

                else:
                    batch_len, batch_x, batch_sat = data

                train_loss, _ = sess.run([model.loss, model.optimizer], feed_dict={model.x_input: batch_x,
                                                                                   model.x_lens: batch_len,
                                                                                   model.sat_input: batch_sat,
                                                                                   model.state:np.zeros((len(batch_x), 2*num_layers*hidden_size)),
                                                                                   model.keep_prob : config['drop_emb']})
                print("single:")
                print(train_loss)
                #remove end of sentence for every sentence (or begining of sentence and interpret the las one as ".")
                tot_words = sum(batch_len) - len(batch_len)
                train_losses.append(train_loss * tot_words)
                train_words += tot_words

            print(train_words)
            print(train_losses)
            print(sum(train_losses))
            print('ITER %d, Train Loss: %.4f wps: %.4f' % ( it, sum(train_losses) / train_words, train_words / (time.time() - start)))

            p.join()
            p.terminate()

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

            while True:

                data = data_queue.get()
                if data is None:
                    break

                batch_len, batch_x, batch_sat = data

                #third argument is to keep the state during training
                test_loss = sess.run(model.loss, feed_dict={model.x_input: batch_x,
                                                            model.x_lens: batch_len,
                                                            model.sat_input: batch_sat,
                                                            model.state:np.zeros((len(batch_len), 2*num_layers*hidden_size)),
                                                            model.keep_prob : 1.0})
                tot_words = sum(batch_len) - len(batch_len)
                test_losses.append(test_loss * tot_words)
                test_words += tot_words

            p.join()
            p.terminate()

            nll = sum(test_losses) / test_words

            print('ITER %d, Dev Loss: %.4f, ppl=%.4f wps: %.4f' % (it,nll, math.exp(nll), test_words/(time.time() - dev_start)))

            save_path = saver.save(sess, model_dir + 'model' + str(it) + '.ckpt')
            print('Model SAVED for ITER - ' , ITER)
