import os
import constants
import sys
import time
from itertools import islice
from multiprocessing import Process, Queue
from models.model_factory import create_model

from reader.reader_queue import run_reader_queue
import numpy as np
import tensorflow as tf
from models.deep_bilstm import *
from models.achen import *

from utils.fileutils.kaldi import writeArk, writeScp


class Test():

    def test_impl(self, data, config):
        print(80 * "-")
        print(80 * "-")
        print("loading config path:")

        print(80 * "-")
        print(80 * "-")
        print("config: ")
        for key, value in config.items():
            print(str(key)+" : "+str(value))

        self.__model = create_model(config)

        with tf.Session() as sess:

            saver = tf.train.Saver()

            print("restoring weights from path:")
            print (config[constants.CONFIG_TAGS_TEST.TRAINED_WEIGHTS])

            saver.restore(sess, config[constants.CONFIG_TAGS_TEST.TRAINED_WEIGHTS])

            print("weights restored")
            print(80 * "-")
            print(80 * "-")
            print("start decoding...")
            soft_probs, log_soft_probs, log_likes, logits, batches_id = self.__create_result_containers(config)

            ntest, test_costs, test_ters, ntest_labels = self.__create_counter_containers(config)

            process, data_queue, test_x = self.__generate_queue(config, data)
            process.start()

            while True:

                data = data_queue.get()
                if data is None:
                    break

                feed, batch_size, index_correct_lan, batch_id, y_batch = self.__prepare_feed(data, config)

                request_list = self.__prepare_request_list(config)

                #TODO clean this up
                batch_ters = None
                batch_log_likes = None

                if(config[constants.CONFIG_TAGS_TEST.COMPUTE_TER] and config[constants.CONFIG_TAGS_TEST.USE_PRIORS]):
                    batch_soft_probs, batch_log_soft_probs, batch_seq_len, batch_logits, batch_cost, batch_ters, batch_log_likes= \
                        sess.run(request_list, feed)

                elif(config[constants.CONFIG_TAGS_TEST.COMPUTE_TER] and not config[constants.CONFIG_TAGS_TEST.USE_PRIORS]):
                    batch_soft_probs, batch_log_soft_probs, batch_seq_len, batch_logits, batch_cost, batch_ters = \
                        sess.run(request_list, feed)

                elif(not config[constants.CONFIG_TAGS_TEST.COMPUTE_TER] and config[constants.CONFIG_TAGS_TEST.USE_PRIORS]):
                    batch_soft_probs, batch_log_soft_probs, batch_seq_len, batch_logits, batch_log_likes= \
                        sess.run(request_list, feed)

                elif(not config[constants.CONFIG_TAGS_TEST.COMPUTE_TER] and not config[constants.CONFIG_TAGS_TEST.USE_PRIORS]):

                    batch_soft_probs, batch_log_soft_probs, batch_seq_len, batch_logits = \
                        sess.run(request_list, feed)

                #TODO the sum should be done online
                self.__update_probs_containers(config, batch_id, batches_id, batch_seq_len, batch_soft_probs, soft_probs,
                                  batch_log_soft_probs, log_soft_probs, batch_logits, logits, batch_log_likes, log_likes)


                if(config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]):
                    self.__update_counters(config, batch_size, ntest, y_batch, ntest_labels, batch_ters, test_ters, batch_cost, test_costs)

            process.join()
            process.terminate()
            print("done decoding")
            print(80 * "-")
            print(80 * "-")


            if(config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]):

                for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
                    for target_id, _ in target_scheme.items():
                        test_costs[language_id][target_id] = test_costs[language_id][target_id] / float(ntest[language_id])

                for language_id, target_scheme in test_ters.iteritems():
                    for target_id, cv_ter in target_scheme.iteritems():
                        test_ters[language_id][target_id] = cv_ter/float(ntest_labels[language_id][target_id])

                self.__print_logs(config, test_costs, test_ters, ntest)

            if(config[constants.CONF_TAGS.ONLINE_AUGMENT_CONF][constants.AUGMENTATION.SUBSAMPLING] > 0):

                batches_id = self.__average_over_augmented_data(config, batches_id, soft_probs, log_soft_probs, log_likes, logits)

            self.__store_results(config, batches_id, soft_probs, log_soft_probs, log_likes, logits)

    def __prepare_request_list(self, config):

                request_list = [
                    self.__model.softmax_probs,
                    self.__model.log_softmax_probs,
                    self.__model.seq_len,
                    self.__model.logits]

                if(config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]):
                    request_list.append(self.__model.ters)
                    request_list.append(self.__model.debug_costs)

                if(config[constants.CONFIG_TAGS_TEST.USE_PRIORS]):
                    request_list.append(self.__model.log_likelihoods)

                return request_list

    def __average_over_augmented_data(self, config, m_batches_id, m_soft_probs, m_log_soft_probs, m_log_likes, m_logits):

        #new batch structure
        new_batch_id = {}

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            new_batch_id[language_id] = {}

            for target_id, num_targets in target_scheme.items():
                if(config[constants.CONFIG_TAGS_TEST.USE_PRIORS]):

                    S={}; P={}; L={}; O={};
                    #iterate over all utterances of a concrete language
                    for utt_id, s, p, l, o in zip(m_batches_id[language_id], m_soft_probs[language_id][target_id], m_log_soft_probs[language_id][target_id],
                                                  m_log_likes[language_id][target_id], m_logits[language_id][target_id]):
                        #utt did not exist. Lets create it
                        if not utt_id in S:
                            S[utt_id] = [s]; P[utt_id] = [p]; L[utt_id] = [l]; O[utt_id] = [o]

                        #utt exists. Lets concatenate
                        else:
                            S[utt_id] += [s]; P[utt_id] += [p]; L[utt_id] += [l]; O[utt_id] += [o]

                    S, P, O, L = self.__shrink_and_average(S, P, O, L)

                else:

                    S={}; P={}; O={}
                    #iterate over all utterances of a concrete language
                    for utt_id, s, p, o in zip(m_batches_id[language_id], m_soft_probs[language_id][target_id], m_log_soft_probs[language_id][target_id],
                                               m_logits[language_id][target_id]):
                        #utt did not exist. Lets create it
                        if not utt_id in S:
                            S[utt_id] = [s]; P[utt_id] = [p]; O[utt_id] = [o]

                        #utt exists. Lets concatenate
                        else:
                            S[utt_id] += [s]; P[utt_id] += [p]; O[utt_id] += [o]

                    S, P, O, _ = self.__shrink_and_average(S, P, O)

                m_soft_probs[language_id][target_id] = []
                m_log_soft_probs[language_id][target_id] = []
                m_logits[language_id][target_id] = []
                new_batch_id[language_id][target_id] = []

                if(config[constants.CONFIG_TAGS_TEST.USE_PRIORS]):
                    m_log_likes[language_id][target_id] = []

                #iterate over all uttid again
                for idx, (utt_id, _) in enumerate(S.items()):
                    m_soft_probs[language_id][target_id] += [S[utt_id]]
                    m_log_soft_probs[language_id][target_id] += [P[utt_id]]
                    m_logits[language_id][target_id] += [O[utt_id]]
                    new_batch_id[language_id][target_id].append(utt_id)

                    if(config[constants.CONFIG_TAGS_TEST.USE_PRIORS]):
                        m_log_likes[language_id][target_id] += [L[utt_id]]

        return new_batch_id

    def __shrink_and_average(self, S, P, O, L=None):

        avg_S={}; avg_P={}; avg_L={}; avg_O={}

        for utt_id, _ in S.items():

            #computing minimum L
            min_length = sys.maxint
            for utt_prob in S[utt_id]:
                if(utt_prob.shape[0] < min_length):
                    min_length = utt_prob.shape[0]

            for idx, (utt_prob) in enumerate(S[utt_id]):
                if(utt_id not in avg_S):

                    avg_S[utt_id] = S[utt_id][idx][0:min_length][:]/float(len(S[utt_id]))
                    avg_P[utt_id] = P[utt_id][idx][0:min_length][:]/float(len(P[utt_id]))
                    avg_O[utt_id] = O[utt_id][idx][0:min_length][:]/float(len(O[utt_id]))

                    if(L):
                        avg_L[utt_id] = L[utt_id][0:min_length][:]/float(len(L[utt_id]))
                else:
                    avg_S[utt_id] += S[utt_id][idx][0:min_length][:]/float(len(S[utt_id]))
                    avg_P[utt_id] += P[utt_id][idx][0:min_length][:]/float(len(P[utt_id]))
                    avg_O[utt_id] += O[utt_id][idx][0:min_length][:]/float(len(O[utt_id]))

                    if(L):
                        avg_L[utt_id] += L[utt_id][0:min_length][:]/float(len(L[utt_id]))
        return avg_S, avg_P, avg_O, avg_L


    def __update_probs_containers(self, config,
                                  batch_id, m_batches_id,
                                  batch_seq_len,
                                  batch_soft_probs, m_soft_probs,
                                  batch_log_soft_probs , m_log_soft_probs,
                                  batch_logits, m_logits,
                                  batch_log_likes, m_log_likes):

        language_idx=0
        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            m_batches_id[language_id] += batch_id
            target_idx=0
            for target_id, num_targets in target_scheme.items():
                m_soft_probs[language_id][target_id] += self.__mat2list(batch_soft_probs[language_idx][target_idx], batch_seq_len)
                m_log_soft_probs[language_id][target_id] += self.__mat2list(batch_log_soft_probs[language_idx][target_idx], batch_seq_len)
                m_logits[language_id][target_id] += self.__mat2list(batch_logits[language_idx][target_idx], batch_seq_len)

                if(config[constants.CONFIG_TAGS_TEST.USE_PRIORS]):
                    m_log_likes[language_id][target_id] += self.__mat2list(batch_log_likes[language_idx][target_idx], batch_seq_len)
                target_idx += 1
            language_idx += 1

    def __create_counter_containers(self, config):

        ntest = {}
        test_cost = {}
        test_ters = {}
        ntest_labels = {}

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].iteritems():
            ntest[language_id] = 0
            test_cost[language_id] = {}
            test_ters[language_id] = {}
            ntest_labels[language_id] = {}
            for target_id, _ in target_scheme.iteritems():
                test_ters[language_id][target_id] = 0
                test_cost[language_id][target_id] = 0
                ntest_labels[language_id][target_id] = 0

        return ntest, test_cost, test_ters, ntest_labels

    def __create_result_containers(self, config):

        batches_id={}
        soft_probs = {}
        log_soft_probs = {}
        log_likes = {}
        logits = {}

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].iteritems():
            batches_id[language_id] = []
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

        return soft_probs, log_soft_probs, log_likes, logits, batches_id

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
                print("\t\t Test cost: %.1f, ter: %.1f%%, #example: %d" % (test_cost[language_id][target_id], 100.0*test_ter, ntest[language_id]))
                fp.write("\t\tTest cost: %.1f, ter: %.1f%%, #example: %d\n" % (test_cost[language_id][target_id], 100.0*test_ter, ntest[language_id]))
        fp.close()


    def __store_results(self, config, uttids, soft_probs, log_soft_probs, log_likes, logits):

        for language_id, target_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].iteritems():
            if(len(config[constants.CONF_TAGS.LANGUAGE_SCHEME]) > 1):
                results_dir = os.path.join(config[constants.CONFIG_TAGS_TEST.RESULTS_DIR], language_id)
            else:
                results_dir = config[constants.CONFIG_TAGS_TEST.RESULTS_DIR]
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            for target_id, _ in target_scheme.iteritems():

                    if(config[constants.CONFIG_TAGS_TEST.USE_PRIORS]):
                        writeScp(os.path.join(results_dir, "log_like_"+target_id+".scp"), uttids[language_id][target_id],
                                 writeArk(os.path.join(results_dir, "log_like_"+target_id+".ark"), log_likes[language_id][target_id], uttids[language_id][target_id]))

                    writeScp(os.path.join(results_dir, "soft_prob_"+target_id+".scp"), uttids[language_id][target_id],
                             writeArk(os.path.join(results_dir, "soft_prob_"+target_id+".ark"), soft_probs[language_id][target_id], uttids[language_id][target_id]))

                    writeScp(os.path.join(results_dir, "log_soft_prob_"+target_id+".scp"), uttids[language_id][target_id],
                             writeArk(os.path.join(results_dir, "log_soft_prob_"+target_id+".ark"), log_soft_probs[language_id][target_id], uttids[language_id][target_id]))

                    writeScp(os.path.join(results_dir, "logit_"+target_id+".scp"), uttids[language_id][target_id],
                             writeArk(os.path.join(results_dir, "logit_"+target_id+".ark"), logits[language_id][target_id], uttids[language_id][target_id]))


    def __update_counters(self, config, batch_size, m_acum_samples, ybatch, m_acum_labels, batch_ters, m_acum_ters, batch_cost, m_acum_cost):


        #https://stackoverflow.com/questions/835092/python-dictionary-are-keys-and-values-always-the-same-order
        #TODO although this should be changed for now is a workaround

        for idx_lan, (language_id, target_scheme) in enumerate(config[constants.CONF_TAGS.LANGUAGE_SCHEME].iteritems()):
            if(ybatch[1] == language_id):

                for idx_tar, (target_id, _) in enumerate(target_scheme.iteritems()):
                    #note that ybatch[0] contains targets and ybathc[1] contains language_id
                    m_acum_ters[language_id][target_id] += batch_ters[idx_lan][idx_tar]
                    m_acum_labels[language_id][target_id] += self.__get_label_len(ybatch[0][language_id][target_id])
                    m_acum_cost[language_id][target_id] += batch_cost[idx_lan][idx_tar] * batch_size

        m_acum_samples[ybatch[1]] += batch_size



    #important to note that shuf = False (to be able to reuse uttid)
    def __generate_queue (self, config, data):

        test_x, test_y, test_sat = data

        data_queue = Queue(config[constants.CONFIG_TAGS_TEST.BATCH_SIZE])

        if test_y:
            if test_sat:
                #x, y, sat
                process = Process(target= run_reader_queue, args= (data_queue, test_x, test_y, False, False, test_sat))
            else:
                #x, y
                process = Process(target = run_reader_queue, args= (data_queue, test_x, test_y, False, False, None))
        else:
            if test_sat:
                #x, sat
                process = Process(target = run_reader_queue, args= (data_queue, test_x, None, False, False, test_sat))
            else:
                #x
                process =  Process(target = run_reader_queue, args = (data_queue, test_x, None, False, False, None))

        return process, data_queue, test_x


    def __prepare_feed(self, data, config):

        #TODO solve this
        y_batch = None
        if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                != constants.SAT_SATGES.UNADAPTED:
            x_batch, y_batch, sat_batch = data
        elif config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]:
            x_batch, y_batch = data
        else:
            x_batch = data

        #getting the actual x_batch that is in position 0
        batch_size = len(x_batch[0])

        current_lan_index = 0

        index_correct_lan = None
        if config[constants.CONFIG_TAGS_TEST.COMPUTE_TER]:

            #we just convert from dict to a list
            y_batch_list = []
            # batch{language_1;{target_1: labels, target_2: labels},
            # language_2;{target_1: labels, target_2: labels}]
            for language_id, batch_targets in y_batch[0].items():
                for targets_id, batch_targets in batch_targets.items():
                    y_batch_list.append(batch_targets)

            for language_id, language_scheme in config[constants.CONF_TAGS.LANGUAGE_SCHEME].iteritems():
                if (language_id == y_batch[1]):
                    index_correct_lan = current_lan_index
                current_lan_index += 1

            #eventhough self.__model.labels will be equal or grater we will use only until_batch_list
            feed = {i: y for i, y in zip(self.__model.labels, y_batch_list)}

        else:
            feed = {}

        #TODO remove this prelimenary approaches
        #getting the actual x_batch that is in position 0
        feed[self.__model.feats] = x_batch[0]
        feed[self.__model.temperature] = float(config[constants.CONFIG_TAGS_TEST.TEMPERATURE])
        feed[self.__model.is_training_ph] = False

        if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                != constants.SAT_SATGES.UNADAPTED:
            feed[self.__model.sat] = sat_batch


        if(config[constants.CONFIG_TAGS_TEST.USE_PRIORS]):
            #TODO we need to add priors also:
            #feed_priors={i: y for i, y in zip(model.priors, config["prior"])}
            print(config[constants.CONFIG_TAGS_TEST.PRIORS_SCHEME])

        #return feed, batch_size, correct_index, uttid_batch
        return feed, batch_size, index_correct_lan, x_batch[1], y_batch

    def __info(self, s):
        s = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + s
        print(s)

    def __get_label_len(self, label):
        idx, _, _ = label
        return len(idx)

    def __mat2list(self, a, seq_len):

        # roll to match the output of essen code, blank label first
        return [np.roll(a[i, :seq_len[i], :], 1, axis = 1) for i in range(len(a))]

    def __chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())
