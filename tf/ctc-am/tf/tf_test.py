import os
import constants
import sys
import time
from itertools import islice
from multiprocessing import Process, Queue

import numpy as np
import tensorflow as tf
from models.deep_bilstm import *

from utils.fileutils.kaldi import writeArk, writeScp

try:
    from h5_Reader import h5_run_reader
    import kaldi_io
except:
    pass

class Test():

    def info(s):
        s = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + s
        print(s)

    def get_label_len(label):
        idx, _, _ = label
        return len(idx)

    def test(data, config):

        model = DeepBidirRNN(config)

        cv_x, cv_y, cv_sat = data

        saver = tf.train.Saver()

        soft_probs = {}
        log_soft_probs = {}
        log_likes = {}
        logits = {}


        for target_key, _ in config["target_scheme"].iteritems():
            soft_probs[target_key] = []
            log_soft_probs[target_key] = []
            log_likes[target_key]=[]
            logits[target_key]=[]

        def mat2list(a, seq_len):
            # roll to match the output of essen code, blank label first
            return [np.roll(a[i, :seq_len[i], :], 1, axis = 1) for i in range(len(a))]

        def chunk(it, size):
            it = iter(it)
            return iter(lambda: tuple(islice(it, size)), ())

        with tf.Session() as sess:

            print ("load_ckpt", model_dir)

            saver.restore(sess, model_path)


            ncv = 0
            cv_cost = 0.0
            cv_ters = [0] * len(nclass)
            ncv_labels = [0] * len(nclass)

            data_queue = Queue(config["batch_size"])

            p = Process(target = run_reader, args = (data_queue, cv_xinfo, cv_y, False))
            p.start()

            while True:

                ncv, cv_step = 0, 0
                data = data_queue.get()
                if data is None:
                    break

                xbatch, ybatch = data
                batch_size = len(xbatch)
                ncv += batch_size

                for idx, y_element_batch in enumerate(ybatch):
                    ncv_labels[idx] += get_label_len(y_element_batch)

                feed = {i: y for i, y in zip(model.labels, ybatch)}

                feed_priors={i: y for i, y in zip(model.priors, config["prior"])}

                feed.update(feed_priors)

                feed[model.feats] = xbatch
                feed[model.temperature] = config["temperature"]
                feed[model.is_training] = False

                batch_cost, batch_ters, batch_soft_probs, batch_log_soft_probs, batch_log_likes, batch_seq_len, batch_logits = sess.run([model.cost,
                    model.ters, model.softmax_probs, model.log_softmax_probs, model.log_likelihoods, model.seq_len, model.logits], feed)

                cv_cost += batch_cost * batch_size
                for idx, _ in enumerate(nclass):
                    cv_ters[idx] += batch_ters[idx]
                    soft_probs[idx] += mat2list(batch_soft_probs[idx], batch_seq_len)
                    log_soft_probs[idx] += mat2list(batch_log_soft_probs[idx], batch_seq_len)
                    log_likes[idx] += mat2list(batch_log_likes[idx], batch_seq_len)
                    logits[idx] += mat2list(batch_logits[idx], batch_seq_len)

            p.join()
            p.terminate()

            # for all classes
            cv_cost = cv_cost/ncv
            for idx, _ in enumerate(nclass):
                cv_ters[idx] /= float(ncv_labels[idx])

                if config["augment"]:
                    # let's average the three(?) sub-sampled outputs
                    S1 = {}; P1 = {}; L1 = {}; O1 = {}; S2 = {}; P2 = {}; L2 = {}; O2 = {}; U = []
                    for u, s, p, l, o in zip(cv_uttids, soft_probs[idx], log_soft_probs[idx], log_likes[idx], logits[idx]):
                        if not u in S1:
                            S1[u] = s; P1[u] = p; L1[u] = l; O1[u] = o
                        elif not u in S2:
                            S2[u] = s; P2[u] = p; L2[u] = l; O2[u] = o
                        else:
                            L = min(S1[u].shape[0],S2[u].shape[0],s.shape[0])
                            if S1[u].shape[0] > L:
                                S1[u] = S1[u][0:L][:]; P1[u] = P1[u][0:L][:]; L1[u] = L1[u][0:L][:]; O1[u] = O1[u][0:L][:]
                            if S2[u].shape[0] > L:
                                S2[u] = S2[u][0:L][:]; P2[u] = P2[u][0:L][:]; L2[u] = L2[u][0:L][:]; O2[u] = O2[u][0:L][:]
                            if s.shape[0] > L:
                                s     =     s[0:L][:]; p     =     p[0:L][:]; l     =     l[0:L][:]; o     =     o[0:L][:]
                            S1[u]=(s+S1[u]+S2[u])/3; P1[u]=(p+P1[u]+P2[u])/3; L1[u]=(l+L1[u]+L2[u])/3; O1[u]=(o+O1[u]+O2[u])/3
                            del S2[u]; del P2[u]; del L2[u]; del O2[u]
                            U.append(u)
                    soft_prob = []; log_soft_prob = []; log_like = []; logit = []
                    for u in U:
                        soft_prob += [S1[u]]
                        log_soft_prob += [P1[u]]
                        log_like += [L1[u]]
                        logit += [O1[u]]

                # Output
                if (len(nclass) > 1):
                    z = "."+str(idx)
                    print("Eval cost: %.1f, ter: %.3f, #example: %d (language %s)" %
                          (cv_cost, cv_ters[idx], ncv, str(idx)))
                else:
                    z = ""
                    print("Eval cost: %.1f, ter: %.3f, #example: %d" %
                          (cv_cost, cv_ters[idx], ncv))

                # let's write scp and ark files for our data
                root_path = config["train_path"]
                if config["use_kaldi_io"]:
                    # this is quite slow ...
                    with open(os.path.join(root_path, "logit_new.ark"), 'wb') as f:
                        for key,mat in zip(U,logit):
                              kaldi_io.write_mat(f, mat, key=key)
                else:
                    writeScp(os.path.join(root_path, "soft_prob"+z+".scp"), U,
                             writeArk(os.path.join(root_path, "soft_prob"+z+".ark"), soft_prob, U))
                    writeScp(os.path.join(root_path, "log_soft_prob"+z+".scp"), U,
                             writeArk(os.path.join(root_path, "log_soft_prob"+z+".ark"), log_soft_prob, U))
                    writeScp(os.path.join(root_path, "log_like"+z+".scp"), U,
                             writeArk(os.path.join(root_path, "log_like"+z+".ark"), log_like, U))
                    writeScp(os.path.join(root_path, "logit"+z+".scp"), U,
                             writeArk(os.path.join(root_path, "logit"+z+".ark"), logit, U))


