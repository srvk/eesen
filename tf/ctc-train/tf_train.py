from models.deep_bilstm import *
from multiprocessing import Process, Queue
import sys, os, re, time, random
from reader.reader_queue import run_reader_queue
from reader.feats_reader import feats_reader_factory
from reader.labels_reader import labels_reader_factory
from random import randint



class Train():

    def __init__(self, config):
        self.config=config
        self.model = DeepBidirRNN(config)
        self.data_queue = Queue(self.config["batch_size"])

    def train_impl(self, data):

            #set random seed so that models can be reproduced
            tf.set_random_seed(config["random_seed"])
            random.seed(config["random_seed"])
            #construct the model acoring to config
            cv_x, tr_x, cv_y, tr_y, cv_sat, tr_sat = data

            #TODO take into consideration the number of mix augmentation
            number_augmented_folder=tr_x.get_num_augmented_folders()

            model_dir = os.path.join(self.config["train_path"],"model")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            with tf.Session() as sess:

                tf.global_variables_initializer().run()

                # restore a training
                saver, alpha = self.__restore_weights()

                for epoch in range(alpha, self.config["nepoch"]):

                    if epoch > self.config["half_after"]:
                        lr_rate = self.config["lr_rate"] * (self.config["half_rate"] ** ((epoch - self.config["half_after"]) // self.config["half_period"]))
                    else:
                        lr_rate = self.config["lr_rate"]

                    #start timer...
                    tic = time.time()

                    #training...
                    train_cost, train_cers, ntrain = self.__train_epoch(sess, epoch, tr_x, tr_y, tr_sat)

                    if self.config["store_model"]:
                        self.__store_weights(sess, saver)

                    #evaluate on validation...
                    cv_cost, cv_cers, ncv = self.__eval_epoch(cv_x, cv_y, cv_sat)

                    self.info("Epoch %d finished in %.0f minutes, learning rate: %.4g" % (epoch + 1, (time.time() - tic)/60.0, lr_rate))

                    for target_key, cv_cer in cv_cers.iteritems():
                        if(len(cv_cers.values()) > 1):
                            print("T: %s" % (target_key))
                        print("\t Train    cost: %.1f, ter: %.1f%%, #example: %d" % (train_cost, 100.0*train_cers[target_key], ntrain))
                        print("\t Validate cost: %.1f, ter: %.1f%%, #example: %d" % (cv_cost, 100.0*cv_cer, ncv))


                    if(tr_x.get_num_augmented_folders > 1):
                        tr_x, tr_y, tr_sat = self.__update_sets(tr_x)


    def __update_sets(self, tr_x):

        tr_x.change_source(randint(0,tr_x.get_batches_id()-1)

        if self.config["adapt_stage"] == "unadapted":
            tr_sat=None
        else:
            tr_sat= feats_reader_factory.create_reader('sat', 'kaldi', tr_x.get_batches_id())

        tr_y=labels_reader_factory.create_reader()

        return tr_x, tr_y, tr_sat

    def __train_epoch(self, sess, epoch, tr_x, tr_y, tr_sat):

        #initializing samples, steps and cost counters
        batch_counter, ntrain, train_cost = 0, 0, 0

        #initializinzing dictionaries that will count
        train_cers, ntr_labels = {}, {}

        #TODO change all iteritems for iter for python 3.0
        #TODO try to do an utils for this kind of functions
        for target_id, _ in self.config["half_period"].iteritems():
            ntr_labels[target_id] = 0
            train_cers[target_id] = 0

        if self.config["adapt_stage"] == "unadapted":
            p = Process(target = run_reader_queue, args = (self.data_queue, tr_x, tr_y, self.config["do_shuf"]))
        else:
            p = Process(target = run_reader_queue, args = (self.data_queue, tr_x , tr_y, self.config["do_shuf"], tr_sat))

        p.start()

        #training starting...
        while True:

            #pop from queue
            data = self.data_queue.get()

            #finish if there no more batches
            if data is None:
                break

            #getting the feed
            feed, batch_size = self.__prepare_feed(self, data)

            #run over a batch
            batch_cost, batch_cers = sess.run([self.model.cost, self.model.cers], feed)

            #updating values...
            train_cers, train_cost, ntrain, ntr_labels = self.__update_counters(train_cers, train_cost, ntrain, ntr_labels, batch_cers, batch_cost, batch_size, data[1])

            #update counters
            for target_key, batch_tr_cer in batch_cers.iteritems():
                train_cers[target_key] += batch_tr_cer

            #print if in debug mode
            if self.config["debug"] == True:
                self.__print_counts_debug(epoch, batch_counter, batch_counter, batch_cost, batch_size, batch_cers)

            batch_counter += 1
            ntrain = batch_size


        p.join()
        p.terminate()

        #averaging counters
        train_cost /= ntrain
        for target_id, train_cer in train_cers.iteritems():
            train_cers[target_id] = train_cer/float(ntr_labels[target_id])

        return train_cost, train_cers, ntrain

    def __eval_epoch(self, sess, cv_x, cv_y, cv_sat):


        #initializing counters and dicts
        ncv, cv_step, cv_cost = 0.0, 0.0, 0.0
        ncv_labels, cv_cers, ncv_label = {}, {}, {}

        for target_id, _ in self.config["target_scheme"].iteritems():
            ncv_labels[target_id] = 0
            cv_cers[target_id]= 0
            ncv_label[target_id]=0

        if self.config["adapt_stage"] == "unadapted":
            p = Process(target = run_reader_queue, args = (self.data_queue, cv_x, cv_y, False))
        else:
            p = Process(target = run_reader_queue, args = (self.data_queue, cv_x , cv_y, False, cv_sat))

        #training starting...
        while True:

            #get batch
            data = self.data_queue.get()

            #if it is empty exit...
            if data is None:
                break

            #getting the feed..
            feed, batch_size = self.__prepare_feed__(self, data)

            #processing a batch...
            batch_cost, batch_cers = sess.run([self.model.cost, self.model.cers], feed)

            #updating values...
            cv_cers, cv_cost, ncv, ncv_labels = self.__update_counters__(cv_cers, cv_cost, ncv, ncv_labels, batch_cers, batch_cost, batch_size, data[1])

        #terminating the queue
        p.join()
        p.terminate()

        #averaging counters
        cv_cost /= ncv
        for target_id, cv_cer in cv_cers.iteritems():
            cv_cers[target_id] = cv_cer/float(ncv_labels[target_id])

        return cv_cost, cv_cers, ncv

    def __update_counters(self, acum_cers, acum_cost, acum_samples, acum_labels, batch_cers, batch_cost, batch_size, ybatch):


        for target_key, cer in batch_cers.iteritems():
            acum_cers[target_key] += cer
            acum_labels[target_key] += ybatch[target_key]

        acum_cost += batch_cost * batch_size
        acum_samples += batch_size

        return acum_cers, acum_cost, acum_samples, acum_labels


    def __restore_weights(self, sess):

        alpha = 0

        saver = tf.train.Saver(max_to_keep=self.config["nepoch"])

        if "continue_ckpt" in self.config or self.config["adapt_stage"] == "fine_tune":

            alpha = int(re.match(".*epoch([-+]?\d+).ckpt", self.config["continue_ckpt"]).groups()[0])
            if(self.config["adapt_stage"] == "fine_tune"):
                saver.restore(sess, self.config["adapt_org_path"])
            else:
                saver.restore(sess, self.config["continue_ckpt"])

        #restoring all variables that should be loaded during adaptation stage (all of them except adaptation layer)
        elif self.config["adapt_org_path"] != "" and self.config["adapt_stage"] == "train_adapt":

            train_vars_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            train_vars=[]
            for var in train_vars_all:
                if not "sat" in var.name:
                    train_vars.append(var)

            saver = tf.train.Saver(max_to_keep=self.config["nepoch"], var_list=train_vars)

            #TODO we should consider to throw a warning and only load needed weights
            saver.restore(sess, self.config["adapt_org_path"])

        return saver, alpha



    def __store_weights(self, sess, saver):
        saver.save(sess, "%s/epoch%02d.ckpt" % (model_dir, epoch + 1))
        with open("%s/epoch%02d.log" % (model_dir, epoch + 1), 'w') as fp:
            fp.write("Time: %.0f minutes, lrate: %.4g\n" % ((time.time() - tic)/60.0, lr_rate))

            for target_key, cv_cer in cv_cers.iteritems():
                fp.write("Targets %s" % (target_key))
                fp.write("Train    cost: %.1f, ter: %.1f%%, #example: %d\n" % (train_cost, 100.0*train_cers[target_key], ntrain))
                fp.write("Validate cost: %.1f, ter: %.1f%%, #example: %d\n" % (cv_cost, 100.0*cv_cer, ncv))

    def __print_counts_debug(self, epoch, batch_counter, total_number_batches, batch_cost, batch_size, batch_cers):

        print("epoch={} batch={}/{} size={} batch_cost={}".format(epoch, train_step, total_number_batches, batch_size, batch_cost))
        print("batch",train_step,"of",total_number_batches,"size",batch_size,
              "queue",self.data_queue.empty(),self.data_queue.full(),self.data_queue.qsize())

        print("cers:")
        for target_key, batch_tr_cer in batch_cers.iteritems():
            print(target_key+" : "+str(batch_tr_cer))

    def __prepare_feed(self, data, lr_rate):

        if self.config["adapt_stage"] == 'unadapted':
            x_batch, ybatch = data
        else:
            x_batch, y_batch, sat_batch = data

        batch_size = len(x_batch)

        y_batch_list=[]
        for _, value in y_batch.iteritems():
            y_batch_list.append(value)

        feed = {i: y for i, y in zip(self.model.labels, y_batch_list)}

        feed[self.model.feats] = x_batch
        feed[self.model.lr_rate] = lr_rate
        feed[self.model.is_training] = True

        if self.config["adapt_stage"] != 'unadapted':
            feed[self.model.sat] = sat_batch

        return feed, batch_size

    def __info(self, s):
        s = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + s
        print(s)

    def __get_label_len(label):
        idx, _, _ = label
        return len(idx)
