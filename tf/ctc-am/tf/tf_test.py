import os
import constants
import sys
import time
from itertools import islice
from multiprocessing import Process, Queue

from reader.reader_queue import run_reader_queue
import numpy as np
import tensorflow as tf
from models.deep_bilstm import *

from utils.fileutils.kaldi import writeArk, writeScp


class Test():

    def test(self, data, config):

        self.__model = DeepBidirRNN(config)

        cv_x, cv_y, cv_sat = data

        saver = tf.train.Saver()


        with tf.Session() as sess:


            print(80 * "-")
            print("restoring weights from path:")
            print (self.__config[constants.CONFIG_TAGS_TEST.TRANED_WEIGHTS])

            saver.restore(sess, self.__config[constants.CONFIG_TAGS_TEST.TRANED_WEIGHTS])

            print("weights restored")
            print(80 * "-")

            soft_probs, log_soft_probs, log_likes, logits = self.__create_result_containers(config)
            ntest, test_costs, test_ters, ntest_labels = self.__create_counter_containers(config)

            process, data_queue = self.__generate_queue(config, data)
            process.start()

            while True:

                data = data_queue.get()
                if data is None:
                    break

                feed, batch_size, index_correct_lan = self.__prepare_feed(data)

                if(config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]):

                    if(config[constants.CONF_TAGS.SAT_SATGE] != constants.SAT_SATGES.UNADAPTED):
                        _, batch_y = data
                    else:
                        batch_y, _, _ = data

                    batch_cost, batch_ters, batch_soft_probs, batch_log_soft_probs, batch_log_likes, batch_seq_len, batch_logits = \
                        sess.run([self.__model.cost,
                                  self.__model.ters,
                                  self.__model.softmax_probs,
                                  self.__model.log_softmax_probs,
                                  self.__model.log_likelihoods,
                                  self.__model.seq_len,
                                  self.__model.logits],
                                 feed)

                    self.__update_counters(test_ters, test_costs, ntest, ntest_labels,
                                           batch_ters, batch_cost, batch_size, batch_y)
                else:

                        batch_soft_probs, batch_log_soft_probs, batch_log_likes, batch_seq_len, batch_logits = \
                        sess.run([self.__model.softmax_probs,
                                  self.__model.log_softmax_probs,
                                  self.__model.log_likelihoods,
                                  self.__model.seq_len,
                                  self.__model.logits],
                                 feed)

                self.__update_probs_containers(config, config, index_correct_lan, batch_soft_probs,
                                               batch_log_soft_probs, batch_log_likes, batch_logits,
                                               soft_probs, log_soft_probs, log_likes, logits)

            process.join()
            process.terminate()


            if(config[constants.CONF_TAGS.ONLINE_AUGMENT_CONF]):
                self.__average_over_augmented_data(self, )


            # for all classes
            cv_cost = cv_cost/ncv
            for idx, _ in enumerate(nclass):
                cv_ters[idx] /= float(ncv_labels[idx])

                if config["augment"]:

                __print_logs()
                __store_results()

    def __average_over_augmented_data(self, config, uttids, soft_probs, log_soft_probs, log_likes, logits):

        for language_id,  target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME]:
            for target_id,  number_targets in target_scheme:
                # let's average the three(?) sub-sampled outputs
                S1 = {}; P1 = {}; L1 = {}; O1 = {}; S2 = {}; P2 = {}; L2 = {}; O2 = {}; U = []
                for u, s, p, l, o in zip(uttids, soft_probs[language_id][target_id], log_soft_probs[language_id][target_id],
                                         log_likes[language_id][target_id], logits[language_id][target_id]):

                    #we store all results utterance wise
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



    def __update_probs_containers(self, config, correct_language_idx, batch_soft_probs,
                                  batch_log_soft_probs, batch_log_likes, batch_logits, batch_seq_len,
                                  m_soft_probs, m_log_soft_probs, m_log_likes, m_logits):


        language_current_idx=0
        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME]:
            idx=0
            if(language_current_idx == correct_language_idx):
                for target_id, num_targets in target_scheme:
                    m_soft_probs[language_id][target_id] += self.__mat2list(batch_soft_probs[idx], batch_seq_len)
                    m_log_soft_probs[language_id][target_id] += self.__mat2list(batch_log_soft_probs[idx], batch_seq_len)
                    m_log_likes[language_id][target_id] += self.__mat2list(batch_log_likes[idx], batch_seq_len)
                    m_logits[language_id][target_id] += self.__mat2list(batch_logits[idx], batch_seq_len)
                idx=idx+1
            language_current_idx+=1

    def __update_counters(self, batch_ter, batch_cost, batch_size, ybatch, m_acum_ters, m_acum_cost,m_acum_samples, m_acum_labels):


        #https://stackoverflow.com/questions/835092/python-dictionary-are-keys-and-values-always-the-same-order
        #TODO although this should be changed for now is a workaround

        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].iteritems():
            if(ybatch[1] == language_id):
                for idx, (target_id, _) in enumerate(target_scheme.iteritems()):
                    #note that ybatch[0] contains targets and ybathc[1] contains language_id
                    m_acum_ters[language_id][target_id] += m_batch_ters[idx]
                    m_acum_labels[language_id][target_id] += self.__get_label_len(ybatch[0][language_id][target_id])

        m_acum_cost[ybatch[1]] += m_batch_cost * batch_size
        m_acum_samples[ybatch[1]] += batch_size

    def __create_counter_containers(self, config):

        ntest = {}
        test_cost = {}
        test_ter = {}
        ntest_labels = {}

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].iteritems():
            ntest[language_id] = 0
            test_cost[language_id] = 0
            test_ter[language_id] = {}
            ntest_labels[language_id] = {}
            for target_id, _ in target_scheme.iteritems():
                test_ter[language_id][target_id] = 0
                ntest_labels[language_id][target_id] = 0

        return ntest, test_cost, test_ter, ntest_labels

    def __create_result_containers(self, config):

        soft_probs = {}
        log_soft_probs = {}
        log_likes = {}
        logits = {}


        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].iteritems():
            soft_probs[language_id] = {}
            soft_probs[language_id] = {}
            log_soft_probs[language_id] = {}
            log_likes[language_id] = {}
            logits[language_id] = {}
            for target_id, _ in target_scheme.iteritems():
                soft_probs[language_id][target_id] = []
                log_soft_probs[language_id][target_id] = []
                log_likes[language_id][target_id]=[]
                logits[language_id][target_id]=[]

        return soft_probs, log_soft_probs, log_likes, logits

    def __print_logs(self, config, test_cost, test_ters, ntest):

        for language_id, target_scheme in test_ters.iteritems():
            if(len(test_ters) > 1):
                fp = open(os.path.join(config[constants.CONFIG_TAGS_TEST.RESULTS_DIR],"test.log"), "w")
                print("Language: "+language_id)
                fp.write("Language: "+language_id)

            else:
                fp = open(os.path.join(config[constants.CONFIG_TAGS_TEST.RESULTS_DIR],"test.log"), "w")

            for target_id,  test_ter in target_scheme.iteritems():
                if(len(target_scheme) > 1):
                    print("\tTarget: %s" % (target_id))
                    fp.write("\tTarget: %s" % (target_id))
                print("\t\t Test cost: %.1f, ter: %.1f%%, #example: %d" % (test_cost[language_id], 100.0*test_ter, ntest[language_id]))
                fp.write("\t\tTest cost: %.1f, ter: %.1f%%, #example: %d\n" % (test_cost[language_id], 100.0*test_ter, ntest[language_id]))
        fp.close()


    def __store_results(self, config, utterances, soft_prob, log_soft_prob, logit, log_like=None):

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].iteritems():
            if(len(config[constants.CONF_TAGS.LANGUAGE_SCHEME]) > 1):
                results_dir = os.path.join(config[constants.CONFIG_TAGS_TEST.RESULTS_DIR], language_id)
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
            else:
                results_dir = config[constants.CONFIG_TAGS_TEST.RESULTS_DIR]

            for target_id, _ in target_scheme.iteritems():

                    if(config[constants.CONFIG_TAGS_TEST.USE_PRIORS]):
                        writeScp(os.path.join(results_dir, "log_like_"+target_id+".scp"), utterances,
                                 writeArk(os.path.join(results_dir, "log_like_"+target_id+".ark"), log_like[language_id][target_id], utterances))

                    writeScp(os.path.join(results_dir, "soft_prob_"+target_id+".scp"), utterances,
                             writeArk(os.path.join(results_dir, "soft_prob_"+target_id+".ark"), soft_prob[language_id][target_id], utterances))
                    writeScp(os.path.join(results_dir, "log_soft_prob_"+target_id+".scp"), utterances,
                             writeArk(os.path.join(results_dir, "log_soft_prob_"+target_id+".ark"), log_soft_prob[language_id][target_id], utterances))
                    writeScp(os.path.join(results_dir, "logit_"+target_id+".scp"), utterances,
                             writeArk(os.path.join(results_dir, "logit"+target_id+".ark"), logit[language_id][target_id], utterances))


    def __update_counters(self, m_acum_ters, m_acum_cost, m_acum_samples, m_acum_labels,
                          m_batch_ters, m_batch_cost, batch_size, ybatch):

        #https://stackoverflow.com/questions/835092/python-dictionary-are-keys-and-values-always-the-same-order
        #TODO although this should be changed for now is a workaround

        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].iteritems():
            if(ybatch[1] == language_id):
                for idx, (target_id, _) in enumerate(target_scheme.iteritems()):
                    #note that ybatch[0] contains targets and ybathc[1] contains language_id
                    m_acum_ters[language_id][target_id] += m_batch_ters[idx]
                    m_acum_labels[language_id][target_id] += self.__get_label_len(ybatch[0][language_id][target_id])

        m_acum_cost[ybatch[1]] += m_batch_cost * batch_size
        m_acum_samples[ybatch[1]] += batch_size

    def __generate_queue (self, config, data):

        test_x, test_y, test_sat = data

        data_queue = Queue(config[constants.CONFIG_TAGS_TEST.BATCH_SIZE])

        if test_y:
            if test_sat:
                #x, y, sat
                return Process(run_reader_queue, data_queue, test_x , test_y, test_sat), data_queue
            else:
                #x, y
                return Process(run_reader_queue, data_queue, test_x , test_y, None), data_queue
        else:
            if test_sat:
                #x, sat
                return Process(run_reader_queue, data_queue, test_x , None, test_sat), data_queue
            else:
                #x
                return Process(run_reader_queue, data_queue, test_x , None, None), data_queue

    def __prepare_feed(self, data, config):

        if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                != constants.SAT_SATGES.UNADAPTED:
            x_batch, y_batch, sat_batch = data
        else:
            x_batch, y_batch = data

        x_batch = x_batch[0]

        batch_size = len(x_batch)

        current_lan_index = 0
        for language_id, language_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].iteritems():
            if (language_id == y_batch[1]):
                index_correct_lan = current_lan_index
            current_lan_index += 1

        #we just convert from dict to a list
        y_batch_list = []
        #every batch has the structue of:
        # batch{language_1;{target_1: labels, target_2: labels},
        # language_2;{target_1: labels, target_2: labels}]
        for language_id, batch_targets in y_batch[0].iteritems():
            for targets_id, batch_targets in batch_targets.iteritems():
                y_batch_list.append(batch_targets)


        #y_batch_list containts [batch_target_1, batch_target_2, ...,, batch_target_n]
        #we need to fill all the target container of the __model
        #the target container has a capacity of self.max_targets_layers
        if(len(y_batch_list) < self.max_targets_layers):
            for count in range(self.max_targets_layers- len(y_batch_list)):
                y_batch_list.append(y_batch_list[0])

        #eventhough self.__model.labels will be equal or grater we will use only until_batch_list
        feed = {i: y for i, y in zip(self.__model.labels, y_batch_list)}

        #TODO remove this prelimenary approaches
        feed[self.__model.feats] = x_batch

        feed[self.__model.is_training] = False

        if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                != constants.SAT_SATGES.UNADAPTED:
            feed[self.__model.sat] = sat_batch


        #TODO we need to add priors also:
        #feed_priors={i: y for i, y in zip(model.priors, config["prior"])}

        return feed, batch_size, index_correct_lan

    def __info(s):
        s = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + s
        print(s)

    def __get_label_len(label):
        idx, _, _ = label
        return len(idx)

    def __mat2list(a, seq_len):
        # roll to match the output of essen code, blank label first
        return [np.roll(a[i, :seq_len[i], :], 1, axis = 1) for i in range(len(a))]

    def __chunk(it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())
