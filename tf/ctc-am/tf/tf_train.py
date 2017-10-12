from multiprocessing import Process, Queue
import sys
from models.model_factory import create_model
import os, re, time, random
import tensorflow as tf
import pdb
import constants
from reader.reader_queue import run_reader_queue
from random import randint
from collections import deque


class Train():

    def __init__(self, config):

        self.__config = config
        self.__model = create_model(config)

        self.__sess = tf.Session()
        self.max_targets_layers = 0
        self.__ter_buffer = [float('inf'), float('inf')]

        self.last_mult_lr_rate = 0

        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
                if(self.max_targets_layers < len(target_scheme)):
                    self.max_targets_layers = len(target_scheme)

    def train_impl(self, data):

            #set random seed so that lm_models can be reproduced
            tf.set_random_seed(self.__config[constants.CONF_TAGS.RANDOM_SEED])
            random.seed(self.__config[constants.CONF_TAGS.RANDOM_SEED])

            #construct the __model acoring to __config
            if(self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] != constants.SAT_TYPE.UNADAPTED):
                cv_x, tr_x, cv_y, tr_y, cv_sat, tr_sat = data
            else:
                cv_x, tr_x, cv_y, tr_y = data
                tr_sat=None
                cv_sat=None

            if not os.path.exists(self.__config[constants.CONF_TAGS.MODEL_DIR]):
                os.makedirs(self.__config[constants.CONF_TAGS.MODEL_DIR])

            #initialize variables of our model
            self.__sess.run(tf.global_variables_initializer())

            # restore a training
            saver, alpha, best_avg_ters = self.__restore_weights()

            #initialize counters
            best_epoch = alpha

            lr_rate = self.__config[constants.CONF_TAGS.LR_RATE]

            if(alpha > 1):
                if(self.__config[constants.CONF_TAGS.FORCE_LR_EPOCH_CKPT]):
                    lr_rate = self.__config[constants.CONF_TAGS.LR_RATE]
                    alpha=1
                    self.__ter_buffer = [float('inf'), float('inf')]
                    best_epoch=1
                else:
                    lr_rate = self.__compute_new_lr_rate(alpha)

            for epoch in range(alpha, self.__config[constants.CONF_TAGS.NEPOCH]):

                #log start
                print(80 * "-")
                print("Epoch "+str(epoch)+" starting ... ( lr_rate: "+str(lr_rate)+")")
                print(80 * "-")

                #start timer...
                tic = time.time()

                #training...
                train_cost, train_ters, ntrain = self.__train_epoch(epoch, lr_rate, tr_x, tr_y, tr_sat)

                if self.__config[constants.CONF_TAGS.STORE_MODEL]:
                    saver.save(self.__sess, "%s/epoch%02d.ckpt" % (self.__config[constants.CONF_TAGS.MODEL_DIR], epoch))

                #evaluate on validation...
                cv_cost, cv_ters, ncv = self.__eval_epoch(cv_x, cv_y, cv_sat)

                self.__generate_logs(cv_ters, cv_cost, ncv, train_ters, train_cost, ntrain, epoch, lr_rate, tic)

                #update lr_rate if needed
                lr_rate, best_avg_ters, best_epoch = self.__update_lr_rate(epoch, cv_ters, best_avg_ters, best_epoch, saver, lr_rate)

                print("Epoch "+str(epoch)+" done.")
                print(80 * "-")
                #change set if needed (mix augmentation)
                self.__update_sets(tr_x, tr_y, tr_sat)


    def __compute_avg_ters(self, ters):

        nters=0
        avg_ters = 0.0
        for language_id, target_scheme in ters.items():
            for target_id, ter in target_scheme.items():
                if(ter > 0):
                    avg_ters += ter
                    nters+=1
        avg_ters /= float(nters)

        return avg_ters

    def __compute_new_lr_rate(self, epoch):


        if epoch > self.__config[constants.CONF_TAGS.HALF_AFTER]:

            diff_epoch= int(float(epoch + 1) / float(self.__config[constants.CONF_TAGS.HALF_AFTER]))

            result = self.__config[constants.CONF_TAGS.LR_RATE] * (self.__config[constants.CONF_TAGS.HALF_RATE] ** (diff_epoch))

            #new_lr_rate = self.__config["lr_rate"] * (self.__config["half_rate"] ** ((epoch - self.__config["half_after"]) // self.__config["half_period"]))

            return result

        else:
            return self.__config[constants.CONF_TAGS.LR_RATE]



    def __update_lr_rate(self, epoch, cv_ters, best_avg_ters, best_epoch, saver, lr_rate):

        avg_ters = self.__compute_avg_ters(cv_ters)

        if (best_avg_ters > avg_ters):
            print("Improved ter by %.1f%% over previous minimum %.1f%% in epoch %d, not updating learning rate" % (100.0*(best_avg_ters-avg_ters), 100.0*self.__ter_buffer[1], best_epoch))
            update_lr=False
        else:
            print("ter worsened by %.1f%% from previous minimum %.1f%% in epoch %d, updating learning rate" % (100.0*(avg_ters-best_avg_ters), 100.0*self.__ter_buffer[1], best_epoch))
            update_lr=True

        # if epoch > self.__config[constants.CONF_TAGS.HALF_AFTER]:
        if update_lr:

            print("updating learning rate...")

            print("from: "+str(lr_rate))

            lr_rate = lr_rate / 2

            print("to: "+str(lr_rate))
            #new_lr_rate = self.__config["lr_rate"] * (self.__config["half_rate"] ** ((epoch - self.__config["half_after"]) // self.__config["half_period"]))
            #new_lr_rate = self.__config["lr_rate"] * (self.__config["half_rate"] ** ((epoch - self.__config["half_after"])))
            #lr_rate = new_lr_rate

            #
            #     print ("load_ckpt", best_epoch+1, 100.0*best_avg_ters, epoch+1, 100.0*avg_ters, new_lr_rate)
            #     print ("load_ckpt", best_epoch+1, 100.0*best_avg_ters, epoch+1, 100.0*avg_ters, new_lr_rate)
            #     saver.restore(self.__sess, "%s/epoch%02d.ckpt" % (self.__config["model_dir"], best_epoch+1))

            #new_lr_rate = init_lr_rate * (half_rate ** ((epoch - half_after) // half_period))

            #diff_epoch= int(float(epoch+1) / float(self.__config[constants.CONF_TAGS.HALF_AFTER]))

            #new_lr_rate = self.__config[constants.CONF_TAGS.LR_RATE] * (self.__config[constants.CONF_TAGS.HALF_RATE] ** (diff_epoch))
                         # * (self.__config[constants.CONF_TAGS.HALF_RATE] ** (diff_epoch))


            #if lr_rate != new_lr_rate:

            print("about to restore model from "+str(best_epoch)+" epoch")
            epoch_name = "/epoch%02d.ckpt" % (best_epoch)
            best_epoch_path = self.__config[constants.CONF_TAGS.MODEL_DIR] + epoch_name

            if(os.path.isfile(best_epoch_path+".index")):
                print("epoch "+str(best_epoch)+" found. ")
                saver.restore(self.__sess, "%s/epoch%02d.ckpt" % (self.__config["model_dir"], best_epoch))
            else:
                print("epoch "+str(best_epoch)+" NOT found. restoring can not be done. ("+best_epoch_path+")")

        if(best_avg_ters > avg_ters):
            best_avg_ters = avg_ters
            best_epoch = epoch

        self.__ter_buffer[0]=self.__ter_buffer[1]
        self.__ter_buffer[1]=avg_ters

        return lr_rate, best_avg_ters, best_epoch

    def __update_sets(self, m_tr_x, m_tr_y, m_tr_sat):

        print(80 * "-")
        print("checking update of epoch...")
        #this is fundamentally wrong
        if(self.__check_needed_mix_augmentation(m_tr_x)):
            dic_sources={}

            #randomizing over all lanaguages (aka geting a number for each language)
            for language_id, lan_aug_folders in m_tr_x.get_language_augment_scheme().items():
                new_src = randint(0, len(lan_aug_folders)-1)
                dic_sources[language_id] = new_src

            print(80 * "-")
            print("changing tr_x sources...")
            print(80 * "-")
            #get a random folder of all previously provided
            m_tr_x.change_source(dic_sources)

            #reorganize sat lm_reader (augmentation might change the order)
            if m_tr_sat:
                print(80 * "-")
                print("updating tr_sat batch order...")
                m_tr_sat.update_batches_id(m_tr_x.get_batches_id())

            print(80 * "-")
            print("changing tr_y batch order... ")
            #reorganize label lm_reader (augmentation might change the order)
            m_tr_y.update_batches_id(m_tr_x.get_batches_id())
        else:
            print("augmentation is not needed.")

    print(80 * "-")

    def __check_needed_mix_augmentation(self, tr_x):
        for target_id, augment_dirs in tr_x.get_language_augment_scheme().items():
            if(len (augment_dirs) > 1):
                return True
        return False

    def __train_epoch(self, epoch, lr_rate, tr_x, tr_y, tr_sat):

        #init data_queue
        data_queue = Queue(self.__config["batch_size"])

        #initializing samples, steps and cost counters
        batch_counter = 0

        #initializinzing dictionaries that will count
        train_ters, ntr_labels, ntrain, train_cost = {}, {}, {}, {}

        #TODO change all iteritems for iter for python 3.0
        #TODO try to do an lm_utils for this kind of functions
        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():

            ntr_labels[language_id] = {}
            train_ters[language_id] = {}
            train_cost[language_id] = {}

            ntrain[language_id] = 0

            for target_id, _ in target_scheme.items():
                ntr_labels[language_id][target_id] = 0
                train_ters[language_id][target_id] = 0
                train_cost[language_id][target_id] = 0

        if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:
            p = Process(target = run_reader_queue, args = (data_queue, tr_x , tr_y, self.__config["do_shuf"], False, tr_sat))
        else:
            p = Process(target = run_reader_queue, args = (data_queue, tr_x, tr_y, self.__config["do_shuf"], False))

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


            batch_cost, batch_ters, _  = self.__sess.run([self.__model.debug_costs[index_correct_lan],
                                                       self.__model.ters[index_correct_lan],
                                                       self.__model.opt[index_correct_lan]],
                                                    feed)

            #updating values...
            self.__update_counters(train_ters, train_cost, ntrain, ntr_labels, batch_ters, batch_cost, batch_size, data[1])

            #print if in debug mode
            if self.__config[constants.CONF_TAGS.DEBUG] == True:

                self.__print_counts_debug(epoch, batch_counter, tr_x.get_num_batches(), batch_cost, batch_size, batch_ters, data_queue)
                batch_counter += 1
        p.join()
        p.terminate()

        #averaging counters
        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            for target_id, _ in target_scheme.items():
                if(ntrain[language_id] != 0):
                    train_cost[language_id][target_id] = train_cost[language_id][target_id] / float(ntrain[language_id])

        for language_id, target_scheme in train_ters.items():
            for target_id, train_ter in target_scheme.items():
                if(ntr_labels[language_id][target_id] != 0):
                    train_ters[language_id][target_id] = train_ter/float(ntr_labels[language_id][target_id])

        return train_cost, train_ters, ntrain

    def __eval_epoch(self, cv_x, cv_y, cv_sat):

        #init data_queue
        data_queue = Queue(self.__config[constants.CONF_TAGS.BATCH_SIZE])

        #initializing counters and dicts
        ncv_labels, cv_ters, cv_cost, ncv = {}, {}, {}, {}

        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            ncv_labels[language_id] = {}
            cv_ters[language_id] = {}
            cv_cost[language_id] = {}

            ncv[language_id] = 0

            for target_id, _ in target_scheme.items():
                ncv_labels[language_id][target_id] = 0
                cv_ters[language_id][target_id]= 0
                cv_cost[language_id][target_id] = 0

        if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:
            p = Process(target = run_reader_queue, args = (data_queue, cv_x , cv_y,
                                                           self.__config[constants.CONF_TAGS.DO_SHUF], False, cv_sat))
        else:
            p = Process(target = run_reader_queue, args = (data_queue, cv_x, cv_y,
                                                           self.__config[constants.CONF_TAGS.DO_SHUF], False))

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
            feed, batch_size, index_correct_lan = self.__prepare_feed(data)


            #processing a batch...
            batch_cost, batch_ters, = self.__sess.run([self.__model.debug_costs[index_correct_lan], self.__model.ters[index_correct_lan]], feed)

            #updating values...
            self.__update_counters(cv_ters, cv_cost, ncv, ncv_labels, batch_ters, batch_cost, batch_size, data[1])

        #terminating the queue
        p.join()
        p.terminate()

        #averaging counters
        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            for target_id, _ in target_scheme.items():
                if(ncv[language_id] != 0):
                    cv_cost[language_id][target_id] = cv_cost[language_id][target_id] / float(ncv[language_id])

        for language_id, target_scheme in cv_ters.items():
            for target_id, cv_ter in target_scheme.items():
                if(ncv_labels[language_id][target_id] != 0):
                    cv_ters[language_id][target_id] = cv_ter/float(ncv_labels[language_id][target_id])

        return cv_cost, cv_ters, ncv

    def __update_counters(self, m_acum_ters, m_acum_cost, m_acum_samples, m_acum_labels,
                          batch_ters, batch_cost, batch_size, ybatch):

        #https://stackoverflow.com/questions/835092/python-dictionary-are-keys-and-values-always-the-same-order
        #TODO although this should be changed for now is a workaround

        for idx_lan, (language_id, target_scheme) in enumerate (self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items()):
            if(ybatch[1] == language_id):
                for idx_tar, (target_id, _) in enumerate(target_scheme.items()):

                    #note that ybatch[0] contains targets and ybathc[1] contains language_id
                    m_acum_ters[language_id][target_id] += batch_ters[idx_tar]
                    m_acum_labels[language_id][target_id] += self.__get_label_len(ybatch[0][language_id][target_id])
                    if(batch_cost[idx_tar] != float('Inf')):
                        m_acum_cost[language_id][target_id] += batch_cost[idx_tar] * batch_size

        m_acum_samples[ybatch[1]] += batch_size

    def __restore_weights(self):

        alpha = 1
        best_avg_ters = float('Inf')

        if self.__config[constants.CONF_TAGS.CONTINUE_CKPT]:

            print(80 * "-")
            print("restoring weights....")
            print(80 * "-")
            #restoring all variables that should be loaded during adaptation stage (all of them except adaptation layer)
            if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                    != constants.SAT_TYPE.UNADAPTED and \
                    (self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                            == constants.SAT_SATGES.TRAIN_SAT or self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                            == constants.SAT_SATGES.TRAIN_DIRECT) and not \
                    self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.CONTINUE_CKPT_SAT]:

                print("partial restoring....")
                vars_to_load=[]
                for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    if (not constants.SCOPES.SPEAKER_ADAPTAION in var.name):
                        vars_to_load.append(var)

                print("var list:")
                for var in vars_to_load:
                    print(var.name)

                saver = tf.train.Saver(max_to_keep=self.__config[constants.CONF_TAGS.NEPOCH], var_list=vars_to_load)
                saver.restore(self.__sess, self.__config[constants.CONF_TAGS.CONTINUE_CKPT])

                #lets track all the variables again...
                alpha = int(re.match(".*epoch([-+]?\d+).ckpt", self.__config[constants.CONF_TAGS.CONTINUE_CKPT]).groups()[0])

                alpha += 1

            else:

                vars_to_load=[]
                if(self.__config[constants.CONF_TAGS.DIFF_NUM_TARGET_CKPT]):
                    print("partial restoring....")
                    print("var list:")
                    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                        if(constants.SCOPES.OUTPUT not in var.name):
                            vars_to_load.append(var)
                else:
                    print("total restoring....")
                    print("var list:")
                    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                        vars_to_load.append(var)

                for var in vars_to_load:
                    print(var)

                saver = tf.train.Saver(max_to_keep=self.__config[constants.CONF_TAGS.NEPOCH], var_list=vars_to_load)
                saver.restore(self.__sess, self.__config[constants.CONF_TAGS.CONTINUE_CKPT])

                alpha = int(re.match(".*epoch([-+]?\d+).ckpt", self.__config[constants.CONF_TAGS.CONTINUE_CKPT]).groups()[0])

                num_val = 0
                acum_val = 0

                with open(self.__config[constants.CONF_TAGS.CONTINUE_CKPT].replace(".ckpt",".log")) as input_file:
                    for line in input_file:
                        if (constants.LOG_TAGS.VALIDATE in line):
                            acum_val += float(line.split()[4].replace("%,",""))
                            num_val += 1

                    self.__ter_buffer[len(self.__ter_buffer) - 1]= acum_val / num_val

                    best_avg_ters=acum_val / num_val

                if(alpha > 1):

                    new_log=self.__config[constants.CONF_TAGS.CONTINUE_CKPT][:-7]+"%02d" % (alpha-1,)+".log"

                    with open(new_log) as input_file:
                        for line in input_file:
                            if (constants.LOG_TAGS.VALIDATE in line):
                                acum_val += float(line.split()[4].replace("%,",""))
                                num_val += 1

                    self.__ter_buffer[0]= acum_val / num_val

                alpha += 1

            print(80 * "-")

        #we want to store everyhting
        saver = tf.train.Saver(max_to_keep=self.__config[constants.CONF_TAGS.NEPOCH])


        return saver, alpha, best_avg_ters

    def __generate_logs(self, cv_ters, cv_cost, ncv, train_ters, train_cost, ntrain, epoch, lr_rate, tic):

        self.__info("Epoch %d finished in %.0f minutes, learning rate: %.4g" % (epoch, (time.time() - tic)/60.0, lr_rate))

        with open("%s/epoch%02d.log" % (self.__config["model_dir"], epoch), 'w') as fp:
            fp.write("Time: %.0f minutes, lrate: %.4g\n" % ((time.time() - tic)/60.0, lr_rate))

            for language_id, target_scheme in cv_ters.items():
                if(len(cv_ters) > 1):
                    print("Language: "+language_id)
                    fp.write("Language: "+language_id)

                for target_id,  cv_ter in target_scheme.items():
                    if(len(target_scheme) > 1):
                        print("\tTarget: %s" % (target_id))
                        fp.write("\tTarget: %s" % (target_id))
                    print("\t\t Train    cost: %.1f, ter: %.1f%%, #example: %d" % (train_cost[language_id][target_id], 100.0*train_ters[language_id][target_id], ntrain[language_id]))
                    print("\t\t"+constants.LOG_TAGS.VALIDATE+" cost: %.1f, ter: %.1f%%, #example: %d" % (cv_cost[language_id][target_id], 100.0*cv_ter, ncv[language_id]))
                    fp.write("\t\tTrain    cost: %.1f, ter: %.1f%%, #example: %d\n" % (train_cost[language_id][target_id], 100.0*train_ters[language_id][target_id], ntrain[language_id]))
                    fp.write("\t\t"+constants.LOG_TAGS.VALIDATE+" cost: %.1f, ter: %.1f%%, #example: %d\n" % (cv_cost[language_id][target_id], 100.0*cv_ter, ncv[language_id]))

    def __print_counts_debug(self, epoch, batch_counter, total_number_batches, batch_cost, batch_size, batch_ters, data_queue):

        print("epoch={} batch={}/{} size={} batch_cost={}".format(epoch, batch_counter, total_number_batches, batch_size, batch_cost))
        print("batch ",batch_counter," of ",total_number_batches,"size ",batch_size,
              "queue ",data_queue.empty(),data_queue.full(),data_queue.qsize())

        print("ters: ")
        print(batch_ters)

    def __prepare_feed(self, data, lr_rate = None):

        if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:
            x_batch, y_batch, sat_batch = data
        else:
            x_batch, y_batch = data


        if self.__config[constants.CONF_TAGS.DEBUG] == True:
            print("")
            print("the following batch_id is prepared to be processed...")
            print(x_batch[1])
            print("size batch x:")
            for element in x_batch[0]:
                print(element.shape)

            print("sizes batch y:")
            for language_id, target in y_batch[0].items():
                for target_id, content in target.items():
                    print(content[2])

            print("")

        #it contains the actuall value of x
        x_batch = x_batch[0]

        batch_size = len(x_batch)

        current_lan_index = 0
        for language_id, language_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            if (language_id == y_batch[1]):
                index_correct_lan = current_lan_index
            current_lan_index += 1

        y_batch_list = []
        for _, value in y_batch[0].items():
            for _, value in value.items():
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
            feed[self.__model.is_training_ph] = True
        else:
            feed[self.__model.is_training_ph] = False

        if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:
            feed[self.__model.sat] = sat_batch

        return feed, batch_size, index_correct_lan

    def __info(self, s):
        s = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + s
        print(s)

    def __get_label_len(self, label):
        idx, _, _ = label
        return len(idx)
