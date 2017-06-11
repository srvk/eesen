import tensorflow as tf
from DeepBidirRNN import *
import numpy as np
from multiprocessing import Process, Queue
import sys, os, re, time, random, functools
from fileutils.kaldi import writeArk, writeScp, readMatrixByOffset
from Reader import run_reader
from itertools import islice

print("tf.py - version information follows:")
try:
    print(sys.version)
    print(tf.__version__)
except:
    print("tf.py: could not get version information for logging")


# -----------------------------------------------------------------
#   Main part
# -----------------------------------------------------------------

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

    def chunk(it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    with tf.Session() as sess:
        print ("load_ckpt", model_path)
        saver.restore(sess, model_path)

        ncv, ncv_label = 0, 0
        cv_cost = cv_wer = 0.0
        data_queue = Queue(config["batch_size"])
        p=Process(target = run_reader, args = (data_queue, cv_xinfo, cv_y, False))
        p.start()
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

        p.join()
        p.terminate()

        cv_wer /= float(ncv_label)
        print("Eval cost: %.1f, ter: %.3f, #example: %d" % (cv_cost, cv_wer, ncv))
        root_path = config["train_path"]
        if config["augment"]:
            # let's average the three(?) sub-sampled outputs
            S1 = {}; P1 = {}; L1 = {}; S2 = {}; P2 = {}; L2 = {}; U = []
            for u, s, p, l in zip(cv_uttids, soft_prob, log_soft_prob, log_like):
                if not u in S1:
                    S1[u] = s; P1[u] = p; L1[u] = l
                elif not u in S2:
                    S2[u] = s; P2[u] = p; L2[u] = l
                else:
                    L = min(S1[u].shape[0],S2[u].shape[0],s.shape[0])
                    if S1[u].shape[0] > L:
                        S1[u] = S1[u][0:L][:]; P1[u] = P1[u][0:L][:]; L1[u] = L1[u][0:L][:]
                    if S2[u].shape[0] > L:
                        S2[u] = S2[u][0:L][:]; P2[u] = P2[u][0:L][:]; L2[u] = L2[u][0:L][:]
                    if s.shape[0] > L:
                        s     =     s[0:L][:]; p     =     p[0:L][:]; l     =     l[0:L][:]
                    S1[u]=(s+S1[u]+S2[u])/3; P1[u]=(p+P1[u]+P2[u])/3; L1[u]=(l+L1[u]+L2[u])/3
                    del S2[u]; del P2[u]; del L2[u]
                    U.append(u)
            soft_prob = []; log_soft_prob = []; log_like = []
            for u in U:
                soft_prob += [S1[u]]
                log_soft_prob += [P1[u]]
                log_like += [L1[u]]

        # let's write scp and ark files for our data
        writeScp(os.path.join(root_path, "soft_prob.scp"), U,
                 writeArk(os.path.join(root_path, "soft_prob.ark"), soft_prob, U))
        writeScp(os.path.join(root_path, "log_soft_prob.scp"), U,
                 writeArk(os.path.join(root_path, "log_soft_prob.ark"), log_soft_prob, U))
        writeScp(os.path.join(root_path, "log_like.scp"), U,
                 writeArk(os.path.join(root_path, "log_like.ark"), log_like, U))

def train(data, config):
    tf.set_random_seed(config["random_seed"])
    random.seed(config["random_seed"])
    model = DeepBidirRNN(config)
    cv_xinfo, tr_xinfo, cv_y, tr_y = data
    for var in tf.trainable_variables():
        print(var)

    debug=False
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
            print ("continue_ckpt", alpha, model_dir, config["continue_ckpt"])
            saver.restore(sess, config["continue_ckpt"])
        print(80 * "-")
        sys.stdout.flush()

        data_queue = Queue(config["batch_size"])
        for epoch in range(alpha,nepoch):
            lr_rate = init_lr_rate * (0.5 ** (epoch / half_period)) 
            tic = time.time()
            ntrain, ntr_label, train_step = 0, 0, 0
            train_cost = train_wer = 0.0
            ntrain_batch = len(tr_xinfo)
            ncv_batch = len(cv_xinfo)

            p=Process(target = run_reader, args = (data_queue, tr_xinfo, tr_y, config["do_shuf"]))
            p.start()
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
                if debug:
                    print("batch",train_step,"of",ntrain_batch,"size",batch_size,
                          "queue",data_queue.empty(),data_queue.full(),data_queue.qsize())
                train_step += 1

            p.join()
            p.terminate()

            train_cost /= ntrain
            train_wer /= float(ntr_label)
            save_scalar(epoch, "train/epoch_cost", train_cost, writer)
            save_scalar(epoch, "train/epoch_cer", train_wer, writer)

            ncv, ncv_label, cv_step = 0, 0, 0
            cv_cost = cv_wer = 0.0
            p=Process(target = run_reader, args = (data_queue, cv_xinfo, cv_y, False))
            p.start()
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

            p.join()
            p.terminate()

            cv_cost /= ncv
            cv_wer /= float(ncv_label)
            save_scalar(epoch, "test/epoch_cost", cv_cost, writer)
            save_scalar(epoch, "test/epoch_cer", cv_wer, writer)
            if config["store_model"]:
                saver.save(sess, "%s/epoch%02d.ckpt" % (model_dir, epoch + 1))
                with open("%s/epoch%02d.log" % (model_dir, epoch + 1), 'w') as fp:
                    fp.write("Time: %.2f seconds, lrate: %.4f\n" % (time.time() - tic, lr_rate))
                    fp.write("Train cost: %.1f, ter: %.3f, #example: %d\n" % (train_cost, train_wer, ntrain))
                    fp.write("Validate cost: %.1f, ter: %.3f, #example: %d\n" % (cv_cost, cv_wer, ncv))

            info("Epoch %d finished in %.2f seconds, learning rate: %.4f" % (epoch + 1, time.time() - tic, lr_rate))
            print("Train cost: %.1f, ter: %.3f, #example: %d" % (train_cost, train_wer, ntrain))
            print("Validate cost: %.1f, ter: %.3f, #example: %d" % (cv_cost, cv_wer, ncv))
            print(80 * "-")
            sys.stdout.flush()
