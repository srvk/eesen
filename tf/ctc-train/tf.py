import tensorflow as tf
from DeepBidirRNN import *
import numpy as np
from multiprocessing import Process, Queue
import sys, os, re, time, random
from fileutils.kaldi import writeArk, readMatrixByOffset
from Reader import run_reader
import itertools

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

def eval(data, config, model_path): 
    model = DeepBidirRNN(config)
    cv_xinfo, cv_y, cv_uttids = data
    saver = tf.train.Saver()
    soft_prob = []
    log_soft_prob = []
    log_like = []

    def mat2list(a, seq_len):
        # roll to match the output of essen code, blank label first
        return [np.roll(a[i, :seq_len[i], :], 1, axis = 1) for i in range(len(a))]

    with tf.Session() as sess:
        print (model_path)
        saver.restore(sess, model_path)

        ncv, ncv_label = 0, 0
        cv_cost = cv_wer = 0.0
        data_queue = Queue(config["batch_size"])
        Process(target = run_reader, args = (data_queue, cv_xinfo, cv_y)).start()
        while True:
            data = data_queue.get()
            if data is None:
                break
            xbatch, ybatch = data
            batch_size = len(xbatch)
            ncv += batch_size
            ncv_label += get_label_len(ybatch)
            feed = {model.feats: xbatch, model.labels: ybatch,
                model.temperature: config["temperature"], model.prior: config["prior"]}
            batch_cost, batch_wer, batch_soft_prob, batch_log_soft_prob, batch_log_like, batch_seq_len = \
                sess.run([model.cost, model.wer, model.softmax_prob,
                model.log_softmax_prob, model.log_likelihood, model.seq_len], feed)
            cv_cost += batch_cost * batch_size
            cv_wer += batch_wer
            soft_prob += mat2list(batch_soft_prob, batch_seq_len)
            log_soft_prob += mat2list(batch_log_soft_prob, batch_seq_len)
            log_like += mat2list(batch_log_like, batch_seq_len)

        cv_cost /= ncv
        cv_wer /= float(ncv_label)
        print("cost: %.4f, cer: %.4f, #example: %d" % (cv_cost, cv_wer, ncv))
        root_path = config["train_path"]
        #with open(root_path + "/soft_prob.ark",'w') as f:
        #    for key,mat in itertools.izip(soft_prob, cv_uttids):
        #        kaldi_io.write_mat(f, mat, key=key)
        #with open(root_path + "/log_soft_prob.ark",'w') as f:
        #    for key,mat in itertools.izip(log_soft_prob, cv_uttids):
        #        kaldi_io.write_mat(f, mat, key=key)
        #with open(root_path + "/log_like.ark",'w') as f:
        #    for key,mat in itertools.izip(log_like, cv_uttids):
        #        kaldi_io.write_mat(f, mat, key=key)
        writeArk(root_path + "/soft_prob.ark", soft_prob, cv_uttids)
        writeArk(root_path + "/log_soft_prob.ark", log_soft_prob, cv_uttids)
        writeArk(root_path + "/log_like.ark", log_like, cv_uttids)

def train(data, config):
    tf.set_random_seed(config["random_seed"])
    random.seed(config["random_seed"])
    model = DeepBidirRNN(config)
    cv_xinfo, tr_xinfo, cv_y, tr_y = data
    for var in tf.trainable_variables():
        print(var)
    sys.stdout.flush()

    log_freq = 100
    nepoch = config["nepoch"]
    init_lr_rate = config["lr_rate"]
    half_period = config["half_period"]
    model_dir = config["train_path"] + "/model" 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    saver = tf.train.Saver(max_to_keep=nepoch)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(config["train_path"], sess.graph)
        tf.global_variables_initializer().run()

        alpha = 0
        # restore a training
        if "continue_ckpt" in config:
            alpha = int(re.match(".*epoch([-+]?\d+).ckpt", config["continue_ckpt"]).groups()[0])
            print ("continue_ckpt", alpha, model_dir)
            saver.restore(sess, "%s/epoch%02d.ckpt" % (model_dir, alpha))
        
        data_queue = Queue(config["batch_size"])

        for epoch in range(alpha,nepoch):
            lr_rate = init_lr_rate * (0.5 ** (epoch / half_period)) 
            tic = time.time()
            ntrain, ntr_label, train_step = 0, 0, 0
            train_cost = train_wer = 0.0
            ntrain_batch = len(tr_xinfo)
            ncv_batch = len(cv_xinfo)

            Process(target = run_reader, args = (data_queue, tr_xinfo, tr_y, config["do_shuf"], epoch)).start()
            while True:
                data = data_queue.get()
                if data is None:
                    break
                xbatch, ybatch = data
                batch_size = len(xbatch)
                ntrain += batch_size
                ntr_label += get_label_len(ybatch)
                feed = {model.feats: xbatch, model.labels: ybatch, model.lr_rate: lr_rate}
                batch_cost, batch_wer, _ = sess.run(
                    [model.cost, model.wer, model.opt], feed)
                train_cost += batch_cost * batch_size
                train_wer += batch_wer
                if train_step % log_freq == 0:
                    global_step = train_step + ntrain_batch * epoch
                    save_scalar(global_step, "train/batch_cost", batch_cost, writer)
                    save_scalar(global_step, "train/batch_ce", batch_wer, writer)
                train_step += 1

            train_cost /= ntrain
            train_wer /= float(ntr_label)
            save_scalar(epoch, "train/epoch_cost", train_cost, writer)
            save_scalar(epoch, "train/epoch_cer", train_wer, writer)

            ncv, ncv_label, cv_step = 0, 0, 0
            cv_cost = cv_wer = 0.0
            Process(target = run_reader, args = (data_queue, cv_xinfo, cv_y, False)).start()
            while True:
                data = data_queue.get()
                if data is None:
                    break
                xbatch, ybatch = data
                batch_size = len(xbatch)
                ncv += batch_size
                ncv_label += get_label_len(ybatch)
                feed = {model.feats: xbatch, model.labels: ybatch, model.lr_rate: lr_rate}
                batch_cost, batch_wer = sess.run([model.cost, model.wer], feed)
                cv_cost += batch_cost * batch_size
                cv_wer += batch_wer
                if cv_step % log_freq == 0:
                    global_step = cv_step + ncv_batch * epoch
                    save_scalar(global_step, "test/batch_cost", batch_cost, writer)
                    save_scalar(global_step, "test/batch_ce", batch_wer, writer)
                cv_step += 1

            cv_cost /= ncv
            cv_wer /= float(ncv_label)
            save_scalar(epoch, "test/epoch_cost", cv_cost, writer)
            save_scalar(epoch, "test/epoch_cer", cv_wer, writer)
            if config["store_model"]:
                saver.save(sess, "%s/epoch%02d.ckpt" % (model_dir, epoch + 1))

            info("Epoch %d finished in %.2f seconds, learning rate: %.4f" % (epoch + 1, time.time() - tic, lr_rate))
            print("Train cost: %.1f, ter: %.3f, #example: %d" % (train_cost, train_wer, ntrain))
            print("Validate cost: %.1f, ter: %.3f, #example: %d" % (cv_cost, cv_wer, ncv))
            print(80 * "-")
            sys.stdout.flush()
