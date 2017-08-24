import sys
import constants
import tensorflow as tf

from models.core import lstm_factory
from models.core import cnn_factory

from lm_utils.lm_fileutils import debug


class DeepBidirRNN:

    def length(self, sequence):
        with tf.variable_scope("seq_len"):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
            length = tf.reduce_sum(used, axis=1)
            length = tf.cast(length, tf.int32)
        return length

    def my_sat_layers(self, num_sat_layers, adapt_dim, nfeat, outputs, scope):

        with tf.variable_scope(scope):
            for i in range(num_sat_layers-1):
                with tf.variable_scope("layer%d" % i):
                    outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = adapt_dim)

            with tf.variable_scope("last_sat_layer"):
                outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = nfeat)

        return outputs

    #TODO check that non expected var names have output in the name
    def __generate_all_gradients(self, language_scheme, optimizer, clip):

        language_ref=[]
        all_gradients=[]
        for language_id in language_scheme:
            language_ref.append(language_id)
            vars=[]
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                if("logits" not in var.name):
                    vars.append(var)
                elif(language_id in var.name):
                    vars.append(var)
            gvs=optimizer.compute_gradients(self.cost, var_list=vars)
            crapped_gvs = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in gvs]
            all_gradients.append(optimizer.apply_gradients(crapped_gvs))

        return language_ref, all_gradients

    def __init__(self, config):

        nfeat = config[constants.INPUT_FEATS_DIM]
        nhidden = config[constants.NHIDDEN]
        language_scheme = config[constants.LANGUAGE_SCHEME]
        l2 = config[constants.L2]
        nlayer = config[constants.NLAYERS]
        clip = config[constants.CLIP]
        nproj = config[constants.NPROJ]
        batch_norm = config[constants.BATCH_NORM]
        lstm_type = config[constants.LSTM_TYPE]
        grad_opt = config[constants.GRAD_OPT]

        adaptation_stage = config[constants.ADAPT_STAGE]


        if adaptation_stage == constants.ADAPTATION_STAGES.TRAIN_ADAPTATION:

            num_sat_layers = int(config[constants.NUM_SAT_LAYERS])
            adapt_dim = int(config[constants.NUM_SAT_LAYERS])
            self.is_trainable_sat=False

        elif adaptation_stage == constants.ADAPTATION_STAGES.FINE_TUNE:

            num_sat_layers = int(config[constants.NUM_SAT_LAYERS])
            adapt_dim = int(config[constants.NUM_SAT_LAYERS])
            self.is_trainable_sat=True

        elif adaptation_stage == constants.ADAPTATION_STAGES.UNADAPTED:
            self.is_trainable_sat=True

        try:
            featproj = config[constants.FEAT_PROJ]
        except:
            featproj = 0


        # build the graph
        self.lr_rate = tf.placeholder(tf.float32, name = "learning_rate")[0]
        self.feats = tf.placeholder(tf.float32, [None, None, nfeat], name = "feats")
        self.temperature = tf.placeholder(tf.float32, name = "temperature")
        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
        self.labels=[]

        # try:
            #TODO can not do xrange directly?
            #TODO iterterm vs iter python 3 vs 2

            # this is because of Python2 vs 3
            # self.labels = [tf.sparse_placeholder(tf.int32)
                           # for _ in xrange(len(target_scheme.values()))]

        #for now we will create the maximum sparse_placeholder needed
        #TODO try to come out with a niter solution
        max_targets_layers=0
        for language_id, language_target_dict in language_scheme.items():
                if(max_targets_layers < len(language_target_dict)):
                    max_targets_layers = len(language_target_dict)

        for language_id, target_scheme in language_scheme.items():
            for target_id, _ in target_scheme.items():
                self.labels.append(tf.sparse_placeholder(tf.int32))

        # except:
            # self.labels = [tf.sparse_placeholder(tf.int32)
                           # for _ in range(len(target_scheme.values()))]

        # try:
            #TODO deal with priors
            # this is because of Python2 vs 3

            # self.priors = [tf.placeholder(tf.float32)
                           # for _ in xrange(len(target_scheme.values()))]

        # self.priors = {key : tf.placeholder(tf.float32)
                       # for (key, value) in target_scheme.iteritems()}

        #TODO for now only taking into consideration the labels. Languages will be needed
        self.priors=[]
        for language_id, target_scheme in language_scheme.items():
            for target_id, _ in target_scheme.items():
                self.priors.append(tf.placeholder(tf.float32, name="place_holder_"+language_id+"_"+target_id))
        # except:
            # self.priors = [tf.placeholder(tf.float32)
                           # for _ in range(len(target_scheme.values()))]

        self.seq_len = self.length(self.feats)

        output_size = 2 * nhidden if nproj == 0 else nproj
        batch_size = tf.shape(self.feats)[0]
        outputs = tf.transpose(self.feats, (1, 0, 2), name = "feat_transpose")

        if adaptation_stage != constants.ADAPTATION_STAGES.UNADAPTED:
            #SAT
            with tf.variable_scope("sat"):
                self.sat = tf.placeholder(tf.float32, [None, 1, adapt_dim], name = "sat")
                sat_t=tf.transpose(self.sat, (1, 0, 2), name = "sat_transpose")
                learned_sat = self.my_sat_layers(num_sat_layers, adapt_dim,  nfeat, sat_t, "sat_layers")
                outputs=tf.add(outputs, learned_sat, name="shift")

        if batch_norm:
            outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, decay=0.9, is_training=self.is_training, updates_collections=None)

        if featproj > 0:
            outputs = tf.contrib.layers.fully_connected(
                activation_fn = None, inputs = outputs, num_outputs = featproj,
                scope = "input_fc", biases_initializer = tf.contrib.layers.xavier_initializer())


        if lstm_type == "cudnn":

            outputs = self.my_cudnn_lstm(outputs, batch_size, nlayer, nhidden, nfeat, nproj,  "cudnn_lstm", batch_norm)
        elif lstm_type == "fuse":
            outputs = self.my_fuse_block_lstm(outputs, batch_size, nlayer, nhidden, nfeat, nproj, "fuse_lstm")
        else:
            outputs = self.my_native_lstm(outputs, batch_size, nlayer, nhidden, nfeat, nproj, "native_lstm")


        with tf.variable_scope("optimizer"):
            optimizer = None
            # TODO: cudnn only supports grad, add check for this
            if grad_opt == "grad":
                optimizer = tf.train.GradientDescentOptimizer(self.lr_rate)
            elif grad_opt == "adam":
                optimizer = tf.train.AdamOptimizer(self.lr_rate)
            elif grad_opt == "momentum":
                optimizer = tf.train.MomentumOptimizer(self.lr_rate, 0.9)


        self.opt = []
        self.ters = []
        self.cost = []

        count=0
        for language_id, language_target_dict in language_scheme.items():
            losses=[]
            tmp_ter=[]

            for target_id, num_targets in language_target_dict.items():
                scope="output_fc_"+language_id+"_"+target_id
                logit = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs=num_targets, scope = scope, biases_initializer = tf.contrib.layers.xavier_initializer())
                loss = tf.nn.ctc_loss(labels=self.labels[count], inputs=logit, sequence_length=self.seq_len)
                losses.append(loss)

                decoded, log_prob = tf.nn.ctc_greedy_decoder(logit, self.seq_len)
                ter = tf.reduce_sum(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.labels[count], normalize = False), name = "ter")
                tmp_ter.append(ter)

                count=count+1

            self.ters.append(tmp_ter)


            var_list_new = self.get_variables_by_lan(language_id)

            with tf.variable_scope("loss"):
                regularized_loss = tf.add_n([tf.nn.l2_loss(v) for v in var_list_new])

            tmp_cost=tf.reduce_mean(losses) + l2 * regularized_loss
            gvs = optimizer.compute_gradients(tmp_cost, var_list = var_list_new)
            capped_gvs = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in gvs]

            #at end  of the day we will just pick up:
            #cost: averaged cost of all targets of a language
            #opt: activate the optimitzation over all the var_list (new_var_list) of a language
            #ter: list of target ters in each language. When we get a language we get all ter targets
            self.cost.append(tmp_cost)
            self.opt.append(optimizer.apply_gradients(capped_gvs))



    def get_variables_by_lan(self, current_name):

        train_vars=[]
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            if("output_fc" not in var.name):
                train_vars.append(var)
            elif(current_name in var.name):
                train_vars.append(var)

        return train_vars

