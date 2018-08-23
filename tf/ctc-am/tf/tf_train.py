from multiprocessing import Process, Queue
import sys
import tensorflow as tf
if(tf.__version__ == "1.6.0"):
    from models_16.model_factory import create_model
else:
    from models.model_factory import create_model

import os, re, time, random
import tensorflow as tf
import pdb
import constants
from reader.reader_queue import run_reader_queue
from random import randint
from collections import deque
import numpy as np
from utils.fileutils.kaldi import writeArk, writeScp

class Train():

    def __init__(self, config):

        self.__config = config
        self.__model = create_model(config)

        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        self.__sess = tf.Session(config=config)
        self.__sess = tf.Session()

        self.max_targets_layers = 0
        self.__ter_buffer = [float('inf'), float('inf')]

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
                    lr_rate = self.__config[constants.CONF_TAGS.LR_RATE]
                    #lr_rate = self.__compute_new_lr_rate(alpha)

            for epoch in range(alpha, self.__config[constants.CONF_TAGS.NEPOCH]):

                #log start
                self.__info("Epoch %d starting, learning rate: %.4g" % (epoch, lr_rate))

                #start timer...
                tic = time.time()

                #training...
                train_cost, train_ters, ntrain = self.__train_epoch(epoch, lr_rate, tr_x, tr_y, tr_sat)

                if self.__config[constants.CONF_TAGS.STORE_MODEL]:
                    saver.save(self.__sess, "%s/epoch%02d.ckpt" % (self.__config[constants.CONF_TAGS.MODEL_DIR], epoch))

                #evaluate on validation...
                cv_cost, cv_ters, ncv = self.__eval_epoch(epoch, cv_x, cv_y, cv_sat)

                self.__generate_logs(cv_ters, cv_cost, ncv, train_ters, train_cost, ntrain, epoch, lr_rate, tic)

                #update lr_rate if needed
                lr_rate, best_avg_ters, best_epoch = self.__update_lr_rate(epoch, cv_ters, best_avg_ters, best_epoch, saver, lr_rate)

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


    # TODO: we should really only use compute_lr_rate OR update_lr_rate - this is confusing
    def __update_lr_rate(self, epoch, cv_ters, best_avg_ters, best_epoch, saver, lr_rate):

        avg_ters = self.__compute_avg_ters(cv_ters)

        if (epoch < self.__config[constants.CONF_TAGS.HALF_AFTER] or lr_rate <= self.__config[constants.CONF_TAGS.MIN_LR_RATE]):
            print("not updating learning rate, parameters", self.__config[constants.CONF_TAGS.HALF_AFTER], self.__config[constants.CONF_TAGS.MIN_LR_RATE])

        elif (best_avg_ters > avg_ters):
            print("not updating learning rate, ter down %.1f%% from epoch %d" % (100.0*(best_avg_ters-avg_ters), best_epoch))

        else:
            lr_rate = lr_rate * self.__config[constants.CONF_TAGS.HALF_RATE]
            if (lr_rate < self.__config[constants.CONF_TAGS.MIN_LR_RATE]):
                lr_rate = self.__config[constants.CONF_TAGS.MIN_LR_RATE]

            print("ter up by %.1f%% from %.1f%% in epoch %d, updating learning rate to %.4g" % (100.0*(avg_ters-best_avg_ters), 100.0*self.__ter_buffer[1], best_epoch, lr_rate))

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

            epoch_name = "/epoch%02d.ckpt" % (best_epoch)
            best_epoch_path = self.__config[constants.CONF_TAGS.MODEL_DIR] + epoch_name

            if(os.path.isfile(best_epoch_path+".index")):
                print("restoring model from epoch "+str(best_epoch))
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
        #print("checking update of epoch...")
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
            pass
            #print("augmentation is not needed.")

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
            p = Process(target = run_reader_queue, args = (data_queue, tr_x , tr_y,
                                                           self.__config[constants.CONF_TAGS.DO_SHUF],
                                                           self.__config[constants.CONF_TAGS.DEBUG],
                                                           self.__config[constants.CONF_TAGS.RANDOM_SEED] + epoch,
                                                           tr_sat))

        else:
            p = Process(target = run_reader_queue, args = (data_queue,
                                                           tr_x,
                                                           tr_y,
                                                           self.__config[constants.CONF_TAGS.DO_SHUF],
                                                           self.__config[constants.CONF_TAGS.DEBUG],
                                                           self.__config[constants.CONF_TAGS.RANDOM_SEED] + epoch))
        #start queue ...
        p.start()

        #training starting...
        while True:

            #pop from queue
            data = data_queue.get()

            #finish if there no more batches
            if data is None:
                break

            feed, batch_size, index_correct_lan, _ = self.__prepare_feed(data, lr_rate)


            batch_cost, batch_edit_distance, _  = self.__sess.run([self.__model.debug_costs[index_correct_lan],
                                                       self.__model.ters[index_correct_lan],
                                                       self.__model.opt[index_correct_lan]],
                                                    feed)

            #updating values...
            self.__update_counters(train_ters, train_cost, ntrain, ntr_labels, batch_edit_distance, batch_cost, batch_size, data[1])

            #print if in debug mode
            if self.__config[constants.CONF_TAGS.DEBUG] == True:

                self.__print_counts_debug(epoch, batch_counter, tr_x.get_num_batches(), batch_cost, batch_size, batch_edit_distance, data_queue)
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


    def __create_result_containers(self, config):

        batches_id={}
        logits = {}

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            batches_id[language_id] = []
            logits[language_id] = {}
            for target_id, _ in target_scheme.items():
                logits[language_id][target_id]=[]

        return logits, batches_id


    def __eval_epoch(self, epoch, cv_x, cv_y, cv_sat):

        #init data_queue
        data_queue = Queue(self.__config[constants.CONF_TAGS.BATCH_SIZE])

        # if we are going to be saving the forward pass, initialize here
        if(self.__config[constants.CONF_TAGS.DUMP_CV_FWD]):
            logits, batches_id = self.__create_result_containers(self.__config)

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
                                                           self.__config[constants.CONF_TAGS.DO_SHUF],
                                                           False,
                                                           0,
                                                           cv_sat))
        else:
            p = Process(target = run_reader_queue, args = (data_queue,
                                                           cv_x,
                                                           cv_y,
                                                           self.__config[constants.CONF_TAGS.DO_SHUF],
                                                           0,
                                                           False))

        #print("Starting forward pass...")

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

            #print('  Batch %d' % count)

            #getting the feed..
            feed, batch_size, index_correct_lan, batch_id = self.__prepare_feed(data)

            if True:
                request_list=[
                    self.__model.debug_costs[index_correct_lan],
                    self.__model.ters[index_correct_lan],
                    self.__model.decodes[index_correct_lan]
                    ]
                if(self.__config[constants.CONF_TAGS.DUMP_CV_FWD]):
                    request_list.append(self.__model.logits)
                    request_list.append(self.__model.seq_len)
                    batch_cost, batch_ters, batch_decodes, batch_logits, batch_seq_len, = self.__sess.run(request_list, feed)
                    self.__update_probs_containers(self.__config, batch_id, batches_id, batch_seq_len, batch_logits, logits)
                else:
                    #batch_cost, batch_ters, batch_decodes, _ = self.__sess.run(request_list, feed)
                    batch_cost, batch_ters, _ = self.__sess.run(request_list, feed)

            else:
            #processing a batch...
                #batch_cost, batch_ters, batch_decodes, = self.__sess.run([self.__model.debug_costs[index_correct_lan], self.__model.ters[index_correct_lan]], feed)
                batch_cost, batch_ters, _ = self.__sess.run([self.__model.debug_costs[index_correct_lan], self.__model.ters[index_correct_lan]], feed)
        
            #print(feed)

            if (self.__config[constants.CONF_TAGS.DEBUG]):
                for cur_id, cur_decode in zip(batch_id, batch_decodes[0]):
                    # add one to convert from tf (blank==last) back to our labeling scheme (blank==0)
                    decode_list=[str(i+1) for i in cur_decode if i>=0]
                    decode_string=' '.join(decode_list)
                    print('DECODE epoch %d:%s %s' % (epoch, cur_id, decode_string))


            #updating values...
            self.__update_counters(cv_ters, cv_cost, ncv, ncv_labels, batch_ters, batch_cost, batch_size, data[1])

            #print ('  ... done',flush=True)

        #terminating the queue
        p.join()
        p.terminate()
        #print("Completed forward pass...",flush=True)

        #averaging counters
        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            for target_id, _ in target_scheme.items():
                if(ncv[language_id] != 0):
                    cv_cost[language_id][target_id] = cv_cost[language_id][target_id] / float(ncv[language_id])

        for language_id, target_scheme in cv_ters.items():
            for target_id, cv_ter in target_scheme.items():
                if(ncv_labels[language_id][target_id] != 0):
                    cv_ters[language_id][target_id] = cv_ter/float(ncv_labels[language_id][target_id])
        
        if (self.__config[constants.CONF_TAGS.DUMP_CV_FWD]):
            if(self.__config[constants.CONF_TAGS.ONLINE_AUGMENT_CONF][constants.AUGMENTATION.SUBSAMPLING] > 0):
                batches_id = self.__average_over_augmented_data(self.__config, batches_id, logits)

            self.__store_results(self.__config, batches_id, logits, epoch)

        return cv_cost, cv_ters, ncv

    def __update_probs_containers(self, config,
                                  batch_id, m_batches_id,
                                  batch_seq_len,
                                  batch_logits, m_logits):

        language_idx=0
        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            m_batches_id[language_id] += batch_id
            target_idx=0
            for target_id, num_targets in target_scheme.items():
                m_logits[language_id][target_id] += self.__mat2list(batch_logits[language_idx][target_idx], batch_seq_len)
                target_idx += 1
            language_idx += 1


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

    def __average_over_augmented_data(self, config, m_batches_id, m_logits):
        #new batch structure
        new_batch_id = {}

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            new_batch_id[language_id] = {}

            for target_id, num_targets in target_scheme.items():
                S={}; P={}; O={}
                    #iterate over all utterances of a concrete language
                for utt_id, s in zip(m_batches_id[language_id], m_logits[language_id][target_id]):
                        #utt did not exist. Lets create it
                        if not utt_id in S:
                            S[utt_id] = [s]

                        #utt exists. Lets concatenate
                        #elif(config[constants.CONFIG_TAGS_TEST.SUBSAMPLED_UTT] == 0):
                        else:
                            #S[utt_id] += [s]
                            pass

                S, _ = self.__shrink_and_average(S)

                m_logits[language_id][target_id] = []
                new_batch_id[language_id][target_id] = []

                #iterate over all uttid again
                for idx, (utt_id, _) in enumerate(S.items()):
                    m_logits[language_id][target_id] += [S[utt_id]]
                    new_batch_id[language_id][target_id].append(utt_id)

        return new_batch_id

    def __shrink_and_average(self, S, L=None):

        avg_S={}; avg_P={}; avg_L={}; avg_O={}

        for utt_id, _ in S.items():

            #computing minimum L
            min_length = sys.maxsize
            #sys.maxint
            for utt_prob in S[utt_id]:
                if(utt_prob.shape[0] < min_length):
                    min_length = utt_prob.shape[0]

            for idx, (utt_prob) in enumerate(S[utt_id]):
                if(utt_id not in avg_S):

                    avg_S[utt_id] = S[utt_id][idx][0:min_length][:]/float(len(S[utt_id]))

                    if(L):
                        avg_L[utt_id] = L[utt_id][0:min_length][:]/float(len(L[utt_id]))
                else:
                    avg_S[utt_id] += S[utt_id][idx][0:min_length][:]/float(len(S[utt_id]))

                    if(L):
                        avg_L[utt_id] += L[utt_id][0:min_length][:]/float(len(L[utt_id]))
        return avg_S, avg_L

    def __store_results(self, config, uttids, logits, epoch):

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            if(len(config[constants.CONF_TAGS.LANGUAGE_SCHEME]) > 1):
                results_dir = os.path.join(config[constants.CONF_TAGS.MODEL_DIR], language_id)
            else:
                results_dir = config[constants.CONF_TAGS.MODEL_DIR]
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            for target_id, _ in target_scheme.items():
                    writeScp(os.path.join(results_dir, "epoch%02d_cv_logit_%s.scp" % (epoch, target_id)), 
                             uttids[language_id][target_id],
                             writeArk(os.path.join(results_dir, "epoch%02d_cv_logit_%s.ark" % (epoch,target_id)), 
                                      logits[language_id][target_id], 
                                      uttids[language_id][target_id]))


    
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

        #we want to store everything
        saver = tf.train.Saver(max_to_keep=self.__config[constants.CONF_TAGS.NEPOCH])


        return saver, alpha, best_avg_ters

    def __generate_logs(self, cv_ters, cv_cost, ncv, train_ters, train_cost, ntrain, epoch, lr_rate, tic):

        self.__info("Epoch %d finished in %.0f minutes" % (epoch, (time.time() - tic)/60.0))

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
                    print("\t\tTrain    cost: %.1f, ter: %.1f%%, #example: %d" % (train_cost[language_id][target_id], 100.0*train_ters[language_id][target_id], ntrain[language_id]))
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
        batch_id= x_batch[1]
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
        #TODO fix this and allow passing parameter?
        feed[self.__model.temperature] = 1.0 # float(config[constants.CONF_TAGS_TEST.TEMPERATURE])

        #it is training
        if(lr_rate):
            feed[self.__model.lr_rate] = lr_rate
            feed[self.__model.is_training_ph] = True
        else:
            feed[self.__model.is_training_ph] = False

        if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:
            feed[self.__model.sat] = sat_batch

        return feed, batch_size, index_correct_lan, batch_id

    def __info(self, s):
        s = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + s
        print(s)

    def __get_label_len(self, label):
        idx, _, _ = label
        return len(idx)

    def __mat2list(self, a, seq_len):
        # roll to match the output of essen code, blank label first
        return [np.roll(a[i, :seq_len[i], :], 1, axis = 1) for i in range(len(a))]
