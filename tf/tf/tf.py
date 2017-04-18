import tensorflow as tf
from DeepBidirRNN import *
import numpy as np
import time
import random
import sys, os

def restore(data):
    idxes = vals = shape = []
    if isinstance(data, tf.SparseTensorValue):
        idxes = data.indices
        vals = data.values
        shape = data.dense_shape
    else:
        idxes, vals, shape = data
    arr = np.zeros(shape, np.int32)
    for i in range(len(vals)):
        arr[tuple(idxes[i])] = vals[i]
    return arr

def info(s):
    s = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + s
    print(s)

def get_label_len(label):
    idx, _, _ = label
    return len(idx)

def eval(data, config, model_path): 
    model = DeepBidirRNN(config)
    cv_x, cv_y = data
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        ncv, ncv_label = 0, 0
        cv_cost = cv_wer = 0.0
        for i in range(len(cv_x)):
            batch_size = len(cv_x[i])
            ncv += batch_size
            ncv_label += get_label_len(cv_y[i])
            feed = {model.feats: cv_x[i], model.labels: cv_y[i]}
            batch_cost, batch_wer, dstr = sess.run(
                [model.cost, model.wer, model.decoded], feed)
            cv_cost += batch_cost * batch_size
            cv_wer += batch_wer
            label = restore(cv_y[i])
            decode = restore(dstr[0])
            assert len(label) == len(decode)
            for i in range(len(label)):
                print(label[i])
                print(decode[i])
                print()

        cv_cost /= ncv
        cv_wer /= float(ncv_label)
        print("cost: %.4f, cer: %.4f, #example: %d" % (cv_cost, cv_wer, ncv))

def train(data, config):
    tf.set_random_seed(15213)
    random.seed(15213)
    model = DeepBidirRNN(config)
    cv_x, tr_x, cv_y, tr_y = data
    for var in tf.trainable_variables():
        print(var)
    sys.stdout.flush()

    nepoch = config["nepoch"]
    init_lr_rate = config["lr_rate"]
    half_period = config["half_period"]
    use_cudnn = config["cudnn"]
    idx_shuf = range(len(tr_x))
    model_dir = "model/l%d-h%d-p%d" % (config["nlayer"], config["nhidden"], config["nproj"])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    saver = tf.train.Saver(max_to_keep=nepoch)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(nepoch):
            lr_rate = init_lr_rate * (0.5 ** (epoch / half_period)) 
            tic = time.time()
            random.shuffle(idx_shuf)
            ntrain, ntr_label = 0, 0
            train_cost = train_wer = 0.0
            for i in idx_shuf: 
                batch_size = len(tr_x[i])
                ntrain += batch_size
                ntr_label += get_label_len(tr_y[i])
                feed = {model.feats: tr_x[i], model.labels: tr_y[i], model.lr_rate: lr_rate}
                batch_cost, batch_wer, _ = sess.run(
                    [model.cost, model.wer, model.opt], feed)
                train_cost += batch_cost * batch_size 
                train_wer += batch_wer

            # # export cudnn model
            # if use_cudnn:
                # # feed used here only for passing shape information
                # cudnn_weight, cudnn_bias = sess.run(
                    # [model.cudnn_weight, model.cudnn_bias], {}) 
                # print "cudnn model"
                # for w in cudnn_weight:
                    # print type(w)
                    # print w.shape

            train_cost /= ntrain
            train_wer /= float(ntr_label)

            ncv, ncv_label = 0, 0
            cv_cost = cv_wer = 0.0
            for i in range(len(cv_x)):
                batch_size = len(cv_x[i])
                ncv += batch_size
                ncv_label += get_label_len(cv_y[i])
                feed = {model.feats: cv_x[i], model.labels: cv_y[i], model.lr_rate: lr_rate}
                batch_cost, batch_wer = sess.run([model.cost, model.wer], feed)
                cv_cost += batch_cost * batch_size
                cv_wer += batch_wer
            cv_cost /= ncv
            cv_wer /= float(ncv_label)
            # saver.save(sess, "%s/epoch%02d.ckpt" % (model_dir, epoch + 1))

            info("Epoch %d finished in %.2f seconds, learning rate: %.4f" % (epoch + 1, time.time() - tic, lr_rate))
            print("Train cost: %.4f, cer: %.4f, #example: %d" % (train_cost, train_wer, ntrain))
            print("Validate cost: %.4f, cer: %.4f, #example: %d" % (cv_cost, cv_wer, ncv))
            print(80 * "-")
            sys.stdout.flush()
