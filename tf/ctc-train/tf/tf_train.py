from models.deep_bilstm import *
from multiprocessing import Process, Queue
import sys, os, re, time, random
import pdb
import constants
from reader.reader_queue import run_reader_queue
from reader.feats_reader import feats_reader_factory
from reader.labels_reader import labels_reader_factory
from random import randint
import numpy as np


class Train():

    def __init__(self, config):
        self.__config = config
        self.__model = DeepBidirRNN(config)
        self.__sess = tf.Session()
        self.max_targets_layers = 0
        for language_id, target_scheme in self.__config[constants.LANGUAGE_SCHEME].iteritems():
                if(self.max_targets_layers < len(target_scheme)):
                    self.max_targets_layers = len(target_scheme)

    def train_impl(self, data):

            #set random seed so that models can be reproduced
            tf.set_random_seed(self.__config[constants.RANDOM_SEED])
            random.seed(self.__config[constants.RANDOM_SEED])

            #construct the __model acoring to __config
            if(self.__config[constants.ADAPT_STAGE] == constants.ADAPTATION_STAGES.UNADAPTED):
                cv_x, tr_x, cv_y, tr_y = data
                tr_sat=None
                cv_sat=None
            else:
                cv_x, tr_x, cv_y, tr_y, cv_sat, tr_sat = data

            if not os.path.exists(self.__config[constants.MODEL_DIR]):
                os.makedirs(self.__config[constants.MODEL_DIR])

            #initialize variables of our model
            self.__sess.run(tf.global_variables_initializer())

            # restore a training
            saver, alpha = self.__restore_weights()

            #initialize counters
            best_avg_ters = float("inf")
            best_epoch = 0
            lr_rate = self.__config[constants.LR_RATE]

            for epoch in range(alpha, self.__config[constants.NEPOCH]):

                #start timer...
                tic = time.time()

                print("training!")
                #training...
                train_cost, train_ters, ntrain = self.__train_epoch(epoch, lr_rate, tr_x, tr_y, tr_sat)
                print("epoch done!")

                if self.__config[constants.STORE_MODEL]:
                    saver.save(self.__sess, "%s/epoch%02d.ckpr" % (self.__config[constants.MODEL_DIR], epoch + 1))

                #evaluate on validation...
                print("eval starting")
                cv_cost, cv_ters, ncv = self.__eval_epoch(cv_x, cv_y, cv_sat)
                print("eval ended")

                #print results
                self.__generate_logs(cv_ters, cv_cost, ncv, train_ters, train_cost, ntrain, epoch, lr_rate, tic)

                #change set if needed (mix augmentation)
                if(tr_x.get_num_augmented_folders() > 1):
                    self.__update_sets(tr_x, tr_y, tr_sat)

                #update lr_rate if needed
                lr_rate, best_avg_ters, best_epoch = self.__update_lr_rate(epoch, cv_ters, best_avg_ters, best_epoch, saver)

    def __compute_avg_ters(self, ters):

        avg_ters = 0.0
        for _, ter in ters.iteritems():
            avg_ters += ter
        avg_ters /= len(ters.values())

        return avg_ters

    def __update_lr_rate(self, epoch, cv_ters, best_avg_ters, best_epoch, saver):

        avg_ters = self.__compute_avg_ters(cv_ters)

        if epoch > self.__config["half_after"]:

            new_lr_rate = self.__config["lr_rate"] * (self.__config["half_rate"] ** ((epoch - self.__config["half_after"]) // self.__config["half_period"]))

            if new_lr_rate != self.__config["lr_rate"] and avg_ters > best_avg_ters and self.__config["store_model"]:

                print ("load_ckpt", best_epoch+1, 100.0*best_avg_ters, epoch+1, 100.0*avg_ters, new_lr_rate)
                print ("load_ckpt", best_epoch+1, 100.0*best_avg_ters, epoch+1, 100.0*avg_ters, new_lr_rate)
                saver.restore(self.__sess, "%s/epoch%02d.ckpt" % (self.__config["__model_dir"], best_epoch+1))

            lr_rate = new_lr_rate
        else:
            lr_rate = self.__config["lr_rate"]

        if(best_avg_ters > avg_ters):
            best_avg_ters = avg_ters
            best_epoch = epoch

        return lr_rate, best_avg_ters, best_epoch

    def __update_sets(self, m_tr_x, m_tr_y, m_tr_sat):

        #get a random folder of all previously provided
        m_tr_x.change_source(randint(0, m_tr_x.get_num_augmented_folders()-1))

        #reorganize sat reader (augmentation might change the order)
        if m_tr_sat:
            m_tr_sat.update_batches_id(m_tr_x.get_batches_id())

        #reorganize label reader (augmentation might change the order)
        m_tr_y.update_batches_id(m_tr_x.get_batches_id())

    def __train_epoch(self, epoch, lr_rate, tr_x, tr_y, tr_sat):

        #init data_queue
        data_queue = Queue(self.__config["batch_size"])

        #initializing samples, steps and cost counters
        batch_counter, ntrain, train_cost = 0, 0, 0

        #initializinzing dictionaries that will count
        train_ters, ntr_labels = {}, {}

        #TODO change all iteritems for iter for python 3.0
        #TODO try to do an utils for this kind of functions
        for language_id, target_scheme in self.__config[constants.LANGUAGE_SCHEME].iteritems():
            ntr_labels[language_id]={}
            train_ters[language_id]={}
            for target_id, _ in target_scheme.iteritems():
                ntr_labels[language_id][target_id] = 0
                train_ters[language_id][target_id] = 0

        if self.__config["adapt_stage"] == "unadapted":
            p = Process(target = run_reader_queue, args = (data_queue, tr_x, tr_y, self.__config["do_shuf"]))
        else:
            p = Process(target = run_reader_queue, args = (data_queue, tr_x , tr_y, self.__config["do_shuf"], tr_sat))

        #start queue ...
        p.start()

        #training starting...
        while True:

            #pop from queue
            data = data_queue.get()

            #finish if there no more batches
            if data is None:
                break

            feed, batch_size, index_correct_lan = self.__prepare_feed(data, lr_rate)

            batch_cost, batch_ters, _ = self.__sess.run([self.__model.cost[index_correct_lan], self.__model.ters[index_correct_lan], self.__model.opt[index_correct_lan]], feed)

            train_cost += batch_cost * batch_size

            #updating values...
            train_ters, train_cost, ntrain, ntr_labels = self.__update_counters(train_ters, train_cost, ntrain, ntr_labels, batch_ters, batch_cost, batch_size, data[1])

            #print if in debug mode
            if self.__config["debug"] == True:
                self.__print_counts_debug(epoch, batch_counter, batch_counter, batch_cost, batch_size, batch_ters, data_queue)
                batch_counter += 1

        p.join()
        p.terminate()

        #averaging counters
        train_cost /= ntrain

        for target_id, train_ter in train_ters.iteritems():

            train_ters[target_id] = train_ter/float(ntr_labels[target_id])

        return train_cost, train_ters, ntrain

    def __eval_epoch(self, cv_x, cv_y, cv_sat):

        #init data_queue
        data_queue = Queue(self.__config["batch_size"])

        #initializing counters and dicts
        ncv, cv_step, cv_cost = 0.0, 0.0, 0.0
        ncv_labels, cv_ters = {}, {}, {}

        for language_id, language_scheme in self.__config[constants.LANGUAGE_SCHEME].iteritems():
            for target_id, _ in self.__config[constants.LANGUAGE_SCHEME].iteritems():
                ncv_labels[target_id] = 0
                cv_ters[target_id]= 0
                ncv_labels[target_id]=0

        if self.__config["adapt_stage"] == "unadapted":
            p = Process(target = run_reader_queue, args = (data_queue, cv_x, cv_y, self.__config["do_shuf"]))
        else:
            p = Process(target = run_reader_queue, args = (data_queue, cv_x , cv_y, self.__config["do_shuf"], cv_sat))

        #starting the queue...
        p.start()

        #training starting...
        count = 0
        while True:

            #get batch
            data = data_queue.get()

            #if it is empty exit...
            if data is None:
                break
            count += 1

            #getting the feed..
            feed, batch_size = self.__prepare_feed(data)

            #processing a batch...
            batch_cost, batch_ters = self.__sess.run([self.__model.cost, self.__model.ters], feed)

            #updating values...
            cv_ters, cv_cost, ncv, ncv_labels = self.__update_counters(cv_ters, cv_cost, ncv, ncv_labels, batch_ters, batch_cost, batch_size, data[1])

        #terminating the queue
        p.join()
        p.terminate()

        #averaging counters
        cv_cost /= ncv
        for target_id, cv_ter in cv_ters.iteritems():
            cv_ters[target_id] = cv_ter/float(ncv_labels[target_id])

        return cv_cost, cv_ters, ncv

    def __update_counters(self, acum_ters, acum_cost, acum_samples, acum_labels, batch_ters, batch_cost, batch_size, ybatch):

        #https://stackoverflow.com/questions/835092/python-dictionary-are-keys-and-values-always-the-same-order
        #TODO although this should be changed for now is a workaround

        for language_id, target_scheme in self.__config[constants.LANGUAGE_SCHEME].iteritems():
            if(ybatch == language_id):
                for target_id, _ in target_scheme.iteritems():
                    #note that ybatch[0] contains targets and ybathc[1] contains language_id
                    acum_ters[language_id][target_id] += ter
                    acum_labels[language_id][target_id] += self.__get_label_len(ybatch[0])

        acum_cost += batch_cost * batch_size
        acum_samples += batch_size

        return acum_ters, acum_cost, acum_samples, acum_labels


    def __restore_weights(self):

        alpha = 0

        saver = tf.train.Saver(max_to_keep=self.__config["nepoch"])

        if "continue_ckpt" in self.__config or self.__config["adapt_stage"] == "fine_tune":

            alpha = int(re.match(".*epoch([-+]?\d+).ckpt", self.__config["continue_ckpt"]).groups()[0])

            #check adaptation stage
            if(self.__config["adapt_stage"] == "fine_tune"):
                saver.restore(self.__sess, self.__config["adapt_org_path"])
            else:
                saver.restore(self.__sess, self.__config["continue_ckpt"])

        #restoring all variables that should be loaded during adaptation stage (all of them except adaptation layer)
        elif self.__config["adapt_org_path"] != "" and self.__config["adapt_stage"] == "train_adapt":

            train_vars_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            train_vars=[]
            for var in train_vars_all:
                if not "sat" in var.name:
                    train_vars.append(var)

            saver = tf.train.Saver(max_to_keep=self.__config["nepoch"], var_list=train_vars)

            #TODO we should consider to throw a warning and only load needed weights
            saver.restore(self.__sess, self.__config["adapt_org_path"])

        return saver, alpha

    def __generate_logs(self, cv_ters, cv_cost, ncv, train_ters, train_cost, ntrain, epoch, lr_rate, tic):

        self.__info("Epoch %d finished in %.0f minutes, learning rate: %.4g" % (epoch + 1, (time.time() - tic)/60.0, lr_rate))

        with open("%s/epoch%02d.log" % (self.__config["model_dir"], epoch + 1), 'w') as fp:
            fp.write("Time: %.0f minutes, lrate: %.4g\n" % ((time.time() - tic)/60.0, lr_rate))

            for target_key, cv_ter in cv_ters.iteritems():
                if(len(cv_ters) > 1):
                    fp.write("Target: %s" % (target_key))
                    print("Target: %s" % (target_key))
                fp.write("Train    cost: %.1f, ter: %.1f%%, #example: %d\n" % (train_cost, 100.0*train_ters[target_key], ntrain))
                fp.write("Validate cost: %.1f, ter: %.1f%%, #example: %d\n" % (cv_cost, 100.0*cv_ter, ncv))

                print("\t Train    cost: %.1f, ter: %.1f%%, #example: %d" % (train_cost, 100.0*train_ters[target_key], ntrain))
                print("\t Validate cost: %.1f, ter: %.1f%%, #example: %d" % (cv_cost, 100.0*cv_ter, ncv))


    def __print_counts_debug(self, epoch, batch_counter, total_number_batches, batch_cost, batch_size, batch_ters, data_queue):

        print("epoch={} batch={}/{} size={} batch_cost={}".format(epoch, batch_counter, total_number_batches, batch_size, batch_cost))
        print("batch",batch_counter,"of",total_number_batches,"size",batch_size,
              "queue",data_queue.empty(),data_queue.full(),data_queue.qsize())

        print("ters:")
        for target_key, batch_tr_ter in batch_ters.iteritems():
            print(target_key+" : "+str(batch_tr_ter))

    def __prepare_feed(self, data, lr_rate = None):

        if self.__config["adapt_stage"] == 'unadapted':
            x_batch, y_batch = data
        else:
            x_batch, y_batch, sat_batch = data

        #TODO remove uttid asap(just sanitychek)
        utt_id = x_batch[1]
        x_batch = x_batch[0]

        batch_size = len(x_batch)

        count=0
        for language_id, language_scheme in self.__config[constants.LANGUAGE_SCHEME].iteritems():
            if (language_id == y_batch[1]):
                index_correct_lan=count
            count+=1

        y_batch_list = []
        count = 0

        for _, value in y_batch[0].iteritems():
            for _, value in value.iteritems():
                y_batch_list.append(value)

        if(len(y_batch_list) < self.max_targets_layers):
           for count in range(self.max_targets_layers- len(y_batch_list)):
               y_batch_list.append(y_batch_list[0])

        #eventhough self.__model.labels will be equal or grater we will use only until_batch_list
        feed = {i: y for i, y in zip(self.__model.labels, y_batch_list)}

        #TODO remove this prelimenary approaches
        feed[self.__model.feats] = x_batch

        #it is training
        if(lr_rate):
            feed[self.__model.lr_rate] = lr_rate
            feed[self.__model.is_training] = True
        else:
            feed[self.__model.is_training] = False

        if self.__config["adapt_stage"] != 'unadapted':
            feed[self.__model.sat] = sat_batch

        return feed, batch_size, index_correct_lan

    def __info(self, s):
        s = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + s
        print(s)

    def __get_label_len(self, label):
        idx, _, _ = label
        return len(idx)
