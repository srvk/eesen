import tensorflow as tf
from DeepBidirRNN import *
import numpy as np
from multiprocessing import Process, Queue
import sys, os, re, time, random
from fileutils.kaldi import writeArk, readMatrixByOffset
from Reader import run_reader

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

    nclass= config["nclass"]

    model = DeepBidirRNN(config)
    cv_xinfo, cv_y, cv_uttids = data
    saver = tf.train.Saver()


    soft_probs = []
    log_soft_probs = []
    log_likes = []
    logits = []

    for idx, _ in enumerate(nclass):
        soft_probs.append([])
        log_soft_probs.append([])
        log_likes.append([])
        logits.append([])


    def mat2list(a, seq_len):
        # roll to match the output of essen code, blank label first
        return [np.roll(a[i, :seq_len[i], :], 1, axis = 1) for i in range(len(a))]

    with tf.Session() as sess:
        print ("load_ckpt", model_path)
        saver.restore(sess, model_path)

        ncv = 0
        cv_cost = 0.0

        cv_cers = [0] * len(nclass)
        ncv_labels = [0] * len(nclass)

        data_queue = Queue(config["batch_size"])


        p=Process(target = run_reader, args = (data_queue, cv_xinfo, cv_y, False))
        p.start()

        count =0
        while True:
            count=count+1

            data = data_queue.get()
            if data is None:
                break

            xbatch, ybatch = data
            batch_size = len(xbatch)
            ncv += batch_size

            for idx, y_element_batch in enumerate(ybatch):
                 ncv_labels[idx] += get_label_len(y_element_batch)

            feed = {i: y for i, y in zip(model.labels, ybatch)}

            feed[model.feats] = xbatch
            feed[model.temperature] = config["temperature"]
            feed[model.prior] = config["prior"]

            batch_cost, batch_cers, batch_soft_probs, batch_log_soft_probs, batch_log_likes, batch_seq_len, batch_logits = sess.run([model.cost, model.cers, model.softmax_probs,
                model.log_softmax_probs, model.log_likelihoods, model.seq_len, model.logits], feed)

            cv_cost += batch_cost * batch_size
            for idx, _ in enumerate(nclass):
                cv_cers[idx] += batch_cers[idx]
                soft_probs[idx] += mat2list(batch_soft_probs[idx], batch_seq_len)
                log_soft_probs[idx] += mat2list(batch_log_soft_probs[idx], batch_seq_len)
                log_likes[idx] += mat2list(batch_log_likes[idx], batch_seq_len)
                logits[idx] += mat2list(batch_logits[idx], batch_seq_len)

        p.join()
        p.terminate()

        root_path = config["train_path"]
        cv_cost=cv_cost/ncv
        for idx, _ in enumerate(nclass):
            cv_cers[idx] /= float(ncv_labels[idx])
            if(len(nclass) > 1):
                print("Eval cost: %.1f, ter: %.3f, #example: %d language "+str(idx)+"]" % (cv_cost, cv_cers[idx], ncv))
            else:
                print("Eval cost: %.1f, ter: %.3f, #example: %d" % (cv_cost, cv_cers[idx], ncv))
            writeArk(root_path + "/soft_prob_"+str(idx)+".ark", soft_probs[idx], cv_uttids)

            writeArk(root_path + "/log_soft_prob_"+str(idx)+".ark", log_soft_probs[idx], cv_uttids)
            writeArk(root_path + "/logits_"+str(idx)+".ark", logits[idx], cv_uttids)

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

    nclass= config["nclass"]

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
        print(80 * "-")
        sys.stdout.flush()

        data_queue = Queue(config["batch_size"])
        for epoch in range(alpha,nepoch):
            lr_rate = init_lr_rate * (0.5 ** (epoch / half_period))
            tic = time.time()

            ntrain, train_step = 0, 0
            train_cost = 0.0

            ntr_label = [0] * len(nclass)
            train_cer = [0] * len(nclass)

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

                for idx, y_element_batch in enumerate(ybatch):
                    ntr_label[idx] += get_label_len(y_element_batch)


                feed = {i: y for i, y in zip(model.labels, ybatch)}
                feed[model.feats] = xbatch
                feed[model.lr_rate] = lr_rate


                batch_cost, batch_cers, _ = sess.run(
                    [model.cost, model.cers, model.opt], feed)

                train_cost += batch_cost * batch_size

                for idx, cers in enumerate(batch_cers):
                    train_cer[idx] += cers

                if train_step % log_freq == 0:
                    global_step = train_step + ntrain_batch * epoch

                    save_scalar(global_step, "train/batch_cost", batch_cost, writer)

                if debug:
                    print("batch",train_step,"of",ntrain_batch,"size",batch_size,
                          "queue",data_queue.empty(),data_queue.full(),data_queue.qsize())
                train_step += 1

            p.join()
            p.terminate()

            train_cost /= ntrain

            save_scalar(epoch, "train/epoch_cost", train_cost, writer)
            for idx, ind_cer in enumerate(train_cer):
                train_cer[idx] = train_cer[idx]/float(ntr_label[idx])
                save_scalar(epoch, "train/epoch_cer", train_cer[idx], writer)

            ncv, cv_step = 0, 0
            ncv_label = [0] * len(nclass)

            cv_cost = 0.0
            cv_cer = [0] * len(nclass)

            p=Process(target = run_reader, args = (data_queue, cv_xinfo, cv_y, False))
            p.start()
            while True:

                data = data_queue.get()
                if data is None:
                    break

                xbatch, ybatch = data


                print(ybatch)
                sys.exit()
                batch_size = len(xbatch)

                ncv += batch_size

                for idx, y_element_batch in enumerate(ybatch):
                    ncv_label[idx] += get_label_len(y_element_batch)

                feed = {i: y for i, y in zip(model.labels, ybatch)}
                feed[model.feats] = xbatch
                feed[model.lr_rate] = lr_rate

                batch_cost, batch_cers = sess.run([model.cost, model.cers], feed)

                for idx, cer in enumerate(batch_cers):
                    cv_cer[idx] += cer

                cv_cost += batch_cost * batch_size


                if cv_step % log_freq == 0:
                    global_step = cv_step + ncv_batch * epoch
                    save_scalar(global_step, "test/batch_cost", batch_cost, writer)
                    for idx, _ in enumerate(nclass):
                        save_scalar(global_step, "test/batch_cer", batch_cers[idx], writer)
                cv_step += 1

            p.join()
            p.terminate()

            cv_cost /= ncv
            save_scalar(epoch, "test/epoch_cost", cv_cost, writer)
            for idx, _ in enumerate(cv_cer):
                cv_cer[idx] = cv_cer[idx]/float(ncv_label[idx])
                save_scalar(epoch, "test/epoch_cer", cv_cer[idx], writer)


            if config["store_model"]:
                saver.save(sess, "%s/epoch%02d.ckpt" % (model_dir, epoch + 1))
                with open("%s/epoch%02d.log" % (model_dir, epoch + 1), 'w') as fp:
                    fp.write("Time: %.2f seconds, lrate: %.4f" % (time.time() - tic, lr_rate))

                    for idx, _ in enumerate(nclass):
                        if(len(nclass) > 1):
                            fp.write("Train cost (language "+str(idx)+"): %.1f, ter: %.3f, #example: %d" % (train_cost, train_cer[idx], ntrain))
                            fp.write("Validate cost: %.1f, ter: %.3f, #example: %d [language "+str(idx)+"]" % (cv_cost, cv_cer[idx], ncv))
                        else:
                            fp.write("Train cost: %.1f, ter: %.3f, #example: %d" % (train_cost, train_cer[idx], ntrain))
                            fp.write("Validate cost: %.1f, ter: %.3f, #example: %d" % (cv_cost, cv_cer[idx], ncv))


            info("Epoch %d finished in %.2f seconds, learning rate: %.4f" % (epoch + 1, time.time() - tic, lr_rate))

            for idx, _ in enumerate(nclass):
                if(len(nclass) > 1):
                    print("Train cost ( language "+str(idx)+"): %.1f, ter: %.3f, #example: %d" % (train_cost, train_cer[idx], ntrain))
                    print("Validate cost ( language "+str(idx)+"): %.1f, ter: %.3f, #example: %d" % (cv_cost, cv_cer[idx], ncv))
                else:
                    print("Train cost ( language "+str(idx)+"): %.1f, ter: %.3f, #example: %d" % (train_cost, train_cer[idx], ntrain))
                    print("Validate cost ( language "+str(idx)+"): %.1f, ter: %.3f, #example: %d" % (cv_cost, cv_cer[idx], ncv))
                    print(80 * "-")
                    sys.stdout.flush()
