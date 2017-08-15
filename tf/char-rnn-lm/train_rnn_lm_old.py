from __future__ import print_function
from RNN_Model import *
import tensorflow as tf
import os
import time
import random
import numpy as np
import sys
import math

def pad(seq, element, length):
    assert len(seq) <= length
    r = seq + [element] * (length - len(seq))
    assert len(r) == length
    return r

def train(data,config):
    start_ = time.time()
    tf.set_random_seed(config["random_seed"])
    random.seed(config["random_seed"])
    model = RNN_Model(config)
    nepoch = config["nepoch"]
    model_dir = config["exp_path"] + "/saved_model/"
    batch_size = config["batch_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    saver = tf.train.Saver(max_to_keep=nepoch)

    train = data['train']
    test = data['test']
    S = data['eos']

    if batch_size != 0:
        train_order = [x * batch_size for x in range((len(train) - 1) / batch_size + 1)]
        test_order = [x * batch_size for x in range((len(test) - 1) / batch_size + 1)]
    else:
        train_order = range(len(train))
        test_order = range(len(test))



    with tf.Session() as sess:
        # writer = tf.summary.FileWriter(config["train_path"], sess.graph)
        tf.global_variables_initializer().run()
        # if "continue_ckpt" in config:
        #     alpha = int(re.match(".*epoch([-+]?\d+).ckpt", config["continue_ckpt"]).groups()[0])
        #     print ("continue_ckpt", alpha, model_dir)
        #     saver.restore(sess, "%s/epoch%02d.ckpt" % (model_dir, alpha))
        train_losses = []
        print('startup time: %r' % (time.time() - start_))
        i = all_time = dev_time = all_tagged = train_words = 0
        start_train = time.time()
        for ITER in range(nepoch):
            random.shuffle(train_order)
            start_ = time.time()
            for i, sid in enumerate(train_order, start=1):
                if i % int(500 / batch_size) == 0:
                    print('Updates so far: %d Loss: %f wps: %f' % (
                    i - 1, sum(train_losses) / train_words, train_words / (time.time() - start_)))
                    all_tagged += train_words
                    train_losses = []
                    train_words = 0
                    all_time = time.time() - start_train
                    start_ = time.time()
                if i % int(30000 / batch_size) == 0 :
                    dev_start = time.time()
                    test_losses = []
                    test_words = 0
                    all_time += time.time() - start_train
                    print('Testing on dev set...')
                    for tid in test_order:
                        t_examples = test[tid:tid + batch_size]
                        x_lens_in = [len(example) for example in t_examples]
                        x_in = [pad(example, S, max(x_lens_in)) for example in t_examples]
                        test_loss = sess.run(model.loss, feed_dict={model.x_input: x_in, model.x_lens: x_lens_in, model.state:np.zeros((len(x_lens_in), 2*num_layers*hidden_size))})
                        tot_words = sum(x_lens_in) - len(x_lens_in)
                        test_losses.append(test_loss * tot_words)
                        test_words += tot_words
                    nll = sum(test_losses) / test_words
                    dev_time += time.time() - dev_start
                    train_time = time.time() - start_train - dev_time
                    print('nll=%.4f, ppl=%.4f, time=%.4f, words_per_sec=%.4f' % (
                nll, math.exp(nll), train_time, all_tagged / train_time), file=sys.stderr)
                    start_ = start_ + (time.time() - dev_start)

                # train on sent
                examples = train[sid: sid + batch_size]
                x_lens_in = [len(example) for example in examples]
                if x_lens_in.count(x_lens_in[0]) != len(x_lens_in):
                    x_in =[pad(example, S, max(x_lens_in)) for example in examples]
                else:
                    x_in = examples
                train_loss, _ = sess.run([model.loss, model.optimizer], feed_dict={model.x_input: x_in, model.x_lens: x_lens_in, model.state:np.zeros((len(x_lens_in), 2*num_layers*hidden_size))})
                tot_words = sum(x_lens_in) - len(x_lens_in)
                train_losses.append(train_loss * tot_words)
                train_words += tot_words
            save_path = saver.save(sess, model_dir + 'model' + str(ITER) + '.ckpt')
            print('Model SAVED ' , ITER)
