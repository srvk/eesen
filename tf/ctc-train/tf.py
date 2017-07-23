import tensorflow as tf
from DeepBidirRNN import *
import numpy as np
from multiprocessing import Process, Queue
import sys, os, re, time, random, functools
from fileutils.kaldi import writeArk, writeScp, readMatrixByOffset
from itertools import islice
from Reader import run_reader
try:
    from h5_Reader import h5_run_reader
except:
    pass
try:
    import kaldi_io
except:
    pass

print(80 * "-")
print("Eesen TF library:", os.path.realpath(__file__))
print("cwd:", os.getcwd(), "version:")
try:
    print(sys.version)
    print(tf.__version__)
except:
    print("tf.py: could not get version information for logging")
print(80 * "-")


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
    """ Evaluate the model
    """

    model = DeepBidirRNN(config)
    cv_xinfo, cv_y, cv_uttids = data
    saver = tf.train.Saver()

    soft_probs = []
    log_soft_probs = []
    log_likes = []
    logits = []
    nclass= config["nclass"]
    for idx, _ in enumerate(nclass):
        soft_probs.append([])
        log_soft_probs.append([])
        log_likes.append([])
        logits.append([])

    def mat2list(a, seq_len):
        # roll to match the output of essen code, blank label first
        return [np.roll(a[i, :seq_len[i], :], 1, axis = 1) for i in range(len(a))]

    def chunk(it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth=True
    with tf.Session(config=tfconfig) as sess:
        print ("load_ckpt", model_path)

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
            #feed[model.prior] = config["prior"]
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
                with open(os.path.join(root_path, "logit"+z+".ark"), 'wb') as f:
                    for key,mat in zip(U,logit):
                          kaldi_io.write_mat(f, mat, key=key)
                #with open(os.path.join(root_path, "log_soft_prob"+z+".ark"), 'wb') as f:
                #    for key,mat in zip(U,log_soft_prob):
                #          kaldi_io.write_mat(f, mat, key=key)
            else:
                writeScp(os.path.join(root_path, "soft_prob"+z+".scp"), U,
                         writeArk(os.path.join(root_path, "soft_prob"+z+".ark"), soft_prob, U))
                writeScp(os.path.join(root_path, "log_soft_prob"+z+".scp"), U,
                         writeArk(os.path.join(root_path, "log_soft_prob"+z+".ark"), log_soft_prob, U))
                writeScp(os.path.join(root_path, "log_like"+z+".scp"), U,
                         writeArk(os.path.join(root_path, "log_like"+z+".ark"), log_like, U))
                writeScp(os.path.join(root_path, "logit"+z+".scp"), U,
                         writeArk(os.path.join(root_path, "logit"+z+".ark"), logit, U))

def train(data, config):
    """ Train the model
    """
    tf.set_random_seed(config["random_seed"])
    random.seed(config["random_seed"])
    model = DeepBidirRNN(config)
    cv_xinfo, tr_xinfo, cv_y, tr_y, valid_dataset, train_dataset, cv_id, tr_id = data

    for var in tf.trainable_variables():
        print(var)

    debug=False
    log_freq = 100
    nepoch = config["nepoch"]
    init_lr_rate = config["lr_rate"]
    half_period = config["half_period"]
    half_rate = config["half_rate"]
    half_after = config["half_after"]
    model_dir = config["train_path"] + "/model"
    nclass = config["nclass"]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    saver = tf.train.Saver(max_to_keep=nepoch)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(config["train_path"], sess.graph)
        tf.global_variables_initializer().run()
        alpha = 0

        # restore a training
        if "continue_ckpt" in config or config["adapt_stage"] == "fine_tune":

            alpha = int(re.match(".*epoch([-+]?\d+).ckpt", config["continue_ckpt"]).groups()[0])
            print ("continue_ckpt", alpha, model_dir, config["continue_ckpt"])

            saver = tf.train.Saver(max_to_keep=nepoch)
            if(config["adapt_stage"] == "fine_tune"):
                saver.restore(sess, config["adapt_org_path"])
            else:
                saver.restore(sess, config["continue_ckpt"])
                if alpha > half_after:
                    half_after = alpha

        elif config["adapt_org_path"] != "" and config["adapt_stage"] == "train_adapt":

            train_vars_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            train_vars=[]
            for var in train_vars_all:
                if not "sat" in var.name:
                    train_vars.append(var)

            saver = tf.train.Saver(max_to_keep=nepoch, var_list=train_vars)
            saver.restore(sess, config["adapt_org_path"])

        print(80 * "-")
        sys.stdout.flush()

        data_queue = Queue(config["batch_size"])
        for epoch in range(alpha,nepoch):
            if epoch > half_after:
                new_lr_rate = init_lr_rate * (half_rate ** ((epoch - half_after) // half_period))
                try:
                    lr_rate
                except NameError:
                    # lr_rate has not been defined yet
                    pass
                else:
                    if new_lr_rate != lr_rate and cv_ters[0] > best_ter and config["store_model"]:
                        # if we reduce the learning rate, we want to do it from the 
                        # previously best configuration (on language 0, for now)
                        # only works if we store the models
                        print ("load_ckpt", best_idx+1, 100.0*best_ter, epoch+1, 100.0*cv_ters[0], new_lr_rate)
                        saver.restore(sess, "%s/epoch%02d.ckpt" % (model_dir, best_idx+1))
                    
                lr_rate = new_lr_rate
            else:
                lr_rate = init_lr_rate

            tic = time.time()

            ntrain, train_step = 0, 0
            train_cost = 0.0
            ntr_label = [0] * len(nclass)
            train_ter = [0] * len(nclass)

            if config["h5_mode"]:

                ntrain_batch = len(train_dataset.batches)
                ncv_batch = len(valid_dataset.batches)
                p = Process(target = h5_run_reader, args = (data_queue, train_dataset, False if epoch == alpha else config["do_shuf"], tr_id))
            elif config["mix"]:

                ntrain_batch = len(tr_xinfo[epoch])
                ncv_batch = len(cv_xinfo)
                p = Process(target = run_reader, args = (data_queue, tr_xinfo[epoch], tr_y[epoch], tr_id[epoch], config["do_shuf"], config["adapt_dim"], config["adapt_path"], config["adapt_reader_type"]))
            else:

                ntrain_batch = len(tr_xinfo)
                ncv_batch = len(cv_xinfo)
                p = Process(target = run_reader, args = (data_queue, tr_xinfo, tr_y, tr_id, config["do_shuf"], config["adapt_dim"], config["adapt_path"], config["adapt_reader_type"], config["roll"]))

            p.start()
            while True:
                data = data_queue.get()
                if data is None:
                    break

                if config["adapt_stage"] == 'unadapted':
                    xbatch, ybatch = data
                else:
                    xbatch, ybatch, sat_batch = data
                batch_size = len(xbatch)
                ntrain += batch_size

                for idx, y_element_batch in enumerate(ybatch):
                    ntr_label[idx] += get_label_len(y_element_batch)

                feed = {i: y for i, y in zip(model.labels, ybatch)}
                feed[model.feats] = xbatch
                feed[model.lr_rate] = lr_rate
                feed[model.is_training] = True

                if config["adapt_stage"] != 'unadapted':
                    feed[model.sat] = sat_batch


                batch_cost, batch_ters, _ = sess.run(
                    [model.cost, model.ters, model.opt], feed)
                train_cost += batch_cost * batch_size

                for idx, ters in enumerate(batch_ters):
                    train_ter[idx] += ters

                if train_step % log_freq == 0:
                    global_step = train_step + ntrain_batch * epoch
                    save_scalar(global_step, "train/batch_cost", batch_cost, writer)
                    save_scalar(global_step, "train/batch_ter", batch_ters[0], writer)
                if debug:
                    print("epoch={} batch={}/{} size={} batch_cost={} batch_wer={}".format(epoch,
                        train_step, ntrain_batch, batch_size, batch_cost, batch_ters[0]/get_label_len(y_element_batch)))
                    print("batch",train_step,"of",ntrain_batch,"size",batch_size,
                        "queue",data_queue.empty(),data_queue.full(),data_queue.qsize())
                train_step += 1

            p.join()
            p.terminate()

            # save the training progress
            train_cost /= ntrain
            save_scalar(epoch, "train/epoch_cost", train_cost, writer)
            for idx, ind_ter in enumerate(train_ter):
                train_ter[idx] = train_ter[idx]/float(ntr_label[idx])
                save_scalar(epoch, "train/epoch_ter{}".format(idx), train_ter[idx], writer)

            # now for validation
            ncv, cv_step = 0, 0
            ncv_label = [0] * len(nclass)
            cv_cost = 0.0
            cv_ters = [0] * len(nclass)

            if config["h5_mode"]:
                p=Process(target = h5_run_reader, args = (data_queue, valid_dataset, False, cv_id))
            else:
                p=Process(target = run_reader, args = (data_queue, cv_xinfo, cv_y, cv_id, False, config["adapt_dim"], config["adapt_path"], config["adapt_reader_type"]))
            p.start()
            while True:
                data = data_queue.get()
                if data is None:
                    break

                xbatch, ybatch = data
                batch_size = len(xbatch)
                ncv += batch_size
                for idx, y_element_batch in enumerate(ybatch):
                    ncv_label[idx] += get_label_len(y_element_batch)

                feed = {i: y for i, y in zip(model.labels, ybatch)}
                feed[model.feats] = xbatch
                feed[model.lr_rate] = lr_rate
                feed[model.is_training] = False

                batch_cost, batch_ters = sess.run([model.cost, model.ters], feed)

                for idx, ter in enumerate(batch_ters):
                    cv_ters[idx] += ter
                cv_cost += batch_cost * batch_size

                if cv_step % log_freq == 0:
                    global_step = cv_step + ncv_batch * epoch
                    save_scalar(global_step, "test/batch_cost", batch_cost, writer)
                    for idx, _ in enumerate(nclass):
                        save_scalar(global_step, "test/batch_ter{}".format(idx), batch_ters[idx], writer)
                cv_step += 1

            p.join()
            p.terminate()

            # logging
            cv_cost /= ncv
            save_scalar(epoch, "test/epoch_cost", cv_cost, writer)
            for idx, _ in enumerate(cv_ters):
                cv_ters[idx] = cv_ters[idx]/float(ncv_label[idx])
                save_scalar(epoch, "test/epoch_ter", cv_ters[idx], writer)

            # let's remember the best iteration so far
            try:
                best_ter
            except NameError:
                best_ter = cv_ters[0]; best_idx = epoch
            else:
                if cv_ters[0] < best_ter:
                    best_ter = cv_ters[0]; best_idx = epoch
                
            if config["store_model"]:
                saver.save(sess, "%s/epoch%02d.ckpt" % (model_dir, epoch + 1))
                with open("%s/epoch%02d.log" % (model_dir, epoch + 1), 'w') as fp:
                    fp.write("Time: %.0f minutes, lrate: %.4g\n" %
                             ((time.time() - tic)/60.0, lr_rate))
                    if (len(nclass) > 1):
                        for idx, _ in enumerate(nclass):
                            fp.write("Train    cost: %.1f, ter: %.1f%%, #example: %d (language %s)\n" %
                                     (train_cost, 100.0*train_ter[idx], ntrain, str(idx)))
                            fp.write("Validate cost: %.1f, ter: %.1f%%, #example: %d (language %s)\n" %
                                     (cv_cost, 100.0*cv_ters[idx], ncv, str(idx)))
                    else:
                        fp.write("Train    cost: %.1f, ter: %.1f%%, #example: %d\n" %
                                 (train_cost, 100.0*train_ter[0], ntrain))
                        fp.write("Validate cost: %.1f, ter: %.1f%%, #example: %d\n" %
                                 (cv_cost, 100.0*cv_ters[0], ncv))

            info("Epoch %d finished in %.0f minutes, learning rate: %.4g" %
                 (epoch + 1, (time.time() - tic)/60.0, lr_rate))
            if (len(nclass) > 1):
                for idx, _ in enumerate(nclass):
                    print("Train    cost: %.1f, ter: %.1f%%, #example: %d (language %s)" %
                          (train_cost, 100.0*train_ter[idx], ntrain, str(idx)))
                    print("Validate cost: %.1f, ter: %.1f%%, #example: %d (language %s)" %
                          (cv_cost, 100.0*cv_ters[idx], ncv, str(idx)))
            else:
                print("Train    cost: %.1f, ter: %.1f%%, #example: %d" %
                      (train_cost, 100.0*train_ter[0], ntrain))
                print("Validate cost: %.1f, ter: %.1f%%, #example: %d" %
                      (cv_cost, 100.0*cv_ters[0], ncv))
            print(80 * "-")
            sys.stdout.flush()
