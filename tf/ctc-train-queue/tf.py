import tensorflow as tf
import math
import os
from DeepBidirRNN import *
import numpy as np
from fileutils.kaldi import writeArk
from multiprocessing import Process, Queue
import sys, os, re, time, random
from fileutils.kaldi import writeArk
#, readMatrixByOffset
from data_pipeline import DataPipeline
#from Reader import run_reader

def info(s):
    s = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + s
    print(s)

def get_label_len(label):
    idx, _, _ = label
    return len(idx)

def save_scalar(step, name, value, writer):
    """Save a scalar value to tensorboard.
      Parameters
      ----------
      step: int
        Training step (sets the position on x-axis of tensorboard graph.
      name: str
        Name of variable. Will be the name of the graph in tensorboard.
      value: float
        The value of the variable at this step.
      writer: tf.FileWriter
        The tensorboard FileWriter instance.
      """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = float(value)
    summary_value.tag = name
    writer.add_summary(summary, step)

def full_join_training(config, path_cv_x, path_cv_y, path_tr_x, path_tr_y, nfeats, number_epochs, number_threads):

    assert (os.path.exists(path_tr_x) is True), "path: "+str(path_tr_x)+" does not exist"
    assert (os.path.exists(path_cv_x) is True), "path: "+str(path_cv_x)+" does not exist"
    assert (os.path.exists(path_tr_y) is True), "path: "+str(path_tr_x)+" does not exist"
    assert (os.path.exists(path_cv_y) is True), "path: "+str(path_cv_y)+" does not exist"

    log_freq = 10
    init_lr_rate = config["lr_rate"]
    half_period = config["half_period"]
    model_dir = os.path.join(config["train_path"], "model")
    log_dir = os.path.join(config["train_path"], "log")
    batch_size = config["batch_size"]

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tf.reset_default_graph()

    #setting the data queues
    with tf.device("/cpu:0"):
        eval_data_pipeline = DataPipeline(path_cv_x, path_cv_y, batch_size, nfeats, 8)
        cv_x_op, cv_y_op = eval_data_pipeline.batch_ops()

        train_data_pipeline = DataPipeline(path_tr_x, path_tr_y, batch_size, nfeats, 8)
        tr_x_op, tr_y_op = train_data_pipeline.batch_ops()

    #set up the model on gpu
    with tf.device('/gpu:0'):
        model = DeepBidirRNN(config, tr_x_op, tr_y_op, cv_x_op, cv_y_op)
        sys.stdout.flush()

    soft_prob = []
    log_soft_prob = []
    log_like = []
    batch_size = config["batch_size"]

    num_cv = len(open(path_cv_x, 'r').readlines())

    cv_uttids=[]
    for element in open(path_cv_x, 'r').readlines():
        cv_uttids.append(element.split()[0])

    saver = tf.train.Saver()
    session_config = tf.ConfigProto(allow_soft_placement=True)


    init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())

    with tf.Session(config=session_config) as sess:
        current_epoch=1
        sess.run(init)

        ncv, ncv_label = 0, 0
        cv_cost = cv_wer = 0.0

        coord = tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)

        writer = tf.summary.FileWriter(config["train_path"], sess.graph)

        #defining variables for each epoch
        lr_rate = init_lr_rate * (0.5 ** (current_epoch / half_period))
        tic = time.time()

        num_tr = len(open(path_tr_x, 'r').readlines())
        num_cv = len(open(path_cv_x, 'r').readlines())
        train_step=0

        for current_epoch in range(number_epochs):
            cv_file=open(log_dir+"/cv."+"iter"+str(current_epoch)+".log", 'w')
            tr_file=open(log_dir+"/tr."+"iter"+str(current_epoch)+".log", 'w')

            for is_training in [True, False]:
                if is_training:
                    acum_samples_tr = 0
                    train_cost = train_wer = 0.0
                    number_iterations=int(math.ceil(num_tr/batch_size))
                    print("TRAINING STARTED")
                else:
                    acum_samples_cv = 0
                    test_cost = test_wer = 0.0
                    number_iterations=int(math.ceil(num_cv/batch_size))
                    print("EVAL STARTED")

                for iteration in range(1, number_iterations+2):

                    if coord.should_stop():
                        break

                    #feed data to placeholder, optimize and update wer
                    if(is_training):

                        feed = {model.lr_rate: lr_rate, model.is_training: is_training}

                        batch_cost, batch_wer, _ , num_samples = sess.run([model.cost, model.wer, model.opt, model.num_samples], feed)

                        train_cost = train_cost + batch_cost
                        train_wer = train_wer + batch_wer
                        train_acum_cost = train_cost /iteration
                        train_acum_wer = train_wer /iteration
                        acum_samples_tr = acum_samples_tr+num_samples

                        #print(str(num_samples)+" from "+str(acum_samples_tr)+" of "+str(num_tr))

                        if (iteration % log_freq == 0):
                            log_tr="After "+str(acum_samples_tr)+" sequences: "+str(round(float(train_acum_cost),2))+" Obj      TokenAcc = "+str(round(float(100*(1-train_acum_wer)),2))+" %"
                            tr_file.write(log_tr)
                            print(log_tr)

                    else:

                        feed = {model.temperature: config["temperature"], model.prior: config["prior"], model.is_training: is_training}
                        batch_cost, batch_wer, batch_soft_prob, batch_log_soft_prob, batch_log_like, batch_seq_len, num_samples = sess.run([model.cost, model.wer, model.softmax_prob, model.log_softmax_prob, model.log_likelihood, model.seq_len, model.num_samples], feed)
                        test_cost = test_cost + batch_cost
                        test_wer = test_wer + batch_wer
                        test_acum_cost = test_wer/iteration
                        test_acum_wer = train_wer/iteration
                        acum_samples_cv=acum_samples_cv + num_samples

                        #print(str(num_samples)+" from "+str(acum_samples_cv)+" of "+str(num_cv))


                        if (iteration % log_freq == 0):
                            log_ev="After "+str(acum_samples_cv)+" sequences: "+str(round(float(test_acum_cost),2))+" Obj      TokenAcc = "+str(round(float(100*(1-test_acum_wer)),2))+" %"
                            tr_file.write(log_ev)
                            print(log_ev)

            if config["store_model"]:
                saver.save(sess, "%s/epoch%02d.ckpt" % (model_dir, current_epoch + 1))
            print("TOKEN_ACCURACY >>"+str(round(float(test_acum_wer*100),2))+"%<<")

            coord.request_stop()
            coord.join(threads)




