import constants
import tensorflow as tf
from utils.fileutils import debug
import sys


class DeepBidirRNN:

    def kl_divergence_with_logits(self, q_logits, num_targets):
        """Returns weighted KL divergence between distributions q and p.
        https://github.com/tensorflow/models/blob/master/research/adversarial_text/adversarial_losses.py
        Args:
          q_logits: logits for 1st argument of KL divergence shape
                    [batch_size, num_timesteps, num_classes]
          p_logits: logits for 2nd argument of KL divergence with same shape q_logits.
          weights: 1-D float tensor with shape [batch_size, num_timesteps].
                   Elements should be 1.0 only on end of sequences
        Returns:
          KL: float scalar.
        """
        p_logits = tf.fill(
            tf.shape(q_logits), float(1.0/float(num_targets)), name="homogenousdistribution")

        p_logits = tf.transpose(p_logits, (1, 0, 2), name = "feat_transpose")

        q_logits = tf.transpose(q_logits, (1, 0, 2), name = "feat_transpose")

        q = tf.nn.softmax(q_logits)
        kl = tf.reduce_sum(
            q * (tf.nn.log_softmax(q_logits) - p_logits), -1)


        return kl


    def length(self, sequence):
        with tf.variable_scope("seq_len"):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
            length = tf.reduce_sum(used, axis=1)
            length = tf.cast(length, tf.int32)
        return length


    def my_cudnn_lstm(self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope, batch_norm, dropout, is_training = True):
        """
        outputs: time, batch_size, feat_dim
        """
        if (nlayer == 0):
            sys.exit()
            
        with tf.variable_scope(scope):
            if (nproj > 0):
                ninput = nfeat
                for i in range(nlayer):
                    with tf.variable_scope("layer%d" % i):

                        cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(1,
                                                                     nhidden,
                                                                     'linear_input',
                                                                     'bidirectional',
                                                                     dropout,
                                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                                     bias_initializer=tf.contrib.layers.xavier_initializer())


                        outputs, _output_h = cudnn_model(outputs, None, True)

                        #biases initialized in 0 (default)
                        #weights initialized with xavier (default)
                        outputs = tf.contrib.layers.fully_connected(
                            activation_fn=None,
                            inputs=outputs,
                            num_outputs=nproj,
                            scope="intermediate_projection")


                        if batch_norm:
                            outputs = tf.contrib.layers.batch_norm(outputs,
                            scope = "bn", center=True, scale=True, decay=0.9,
                            is_training=self.is_training_ph, updates_collections=None)

            else:
                cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(nlayer,
                                                             nhidden,
                                                             'linear_input',
                                                             'bidirectional',
                                                             dropout,
                                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                             bias_initializer=tf.contrib.layers.xavier_initializer())

                outputs, _output_h = cudnn_model(outputs, None, True)

                if batch_norm:
                    outputs = tf.contrib.layers.batch_norm(outputs,
                        scope = "bn", center=True, scale=True, decay=0.9,
                        is_training=self.is_training_ph, updates_collections=None)

        return outputs

    def my_fuse_block_lstm(self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope):
        """
        output: time, batch_size, feat_dim
        """
        with tf.variable_scope(scope):
            for i in range(nlayer):
                with tf.variable_scope("layer%d" % i):
                    with tf.variable_scope("fw_lstm"):
                        fw_lstm = tf.contrib.rnn.LSTMBlockFusedCell(nhidden, cell_clip = 0)
                        fw_out, _ = fw_lstm(outputs, dtype=tf.float32, sequence_length = self.seq_len)
                    with tf.variable_scope("bw_lstm"):
                        bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(tf.contrib.rnn.LSTMBlockFusedCell(nhidden, cell_clip = 0))
                        bw_out, _ = bw_lstm(outputs, dtype=tf.float32, sequence_length = self.seq_len)
                    outputs = tf.concat_v2([fw_out, bw_out], 2, name = "output")
                    # outputs = tf.concat([fw_out, bw_out], 2, name = "output")
                    if nproj > 0:
                        outputs = tf.contrib.layers.fully_connected(
                            activation_fn = None, inputs = outputs,
                            num_outputs = nproj, scope = "projection")
        return outputs

    def my_native_lstm(self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope):
        """
        outputs: time, batch_size, feat_dim
        """
        with tf.variable_scope(scope):
            for i in range(nlayer):
                with tf.variable_scope("layer%d" % i):
                    if nproj > 0:
                        cell = tf.contrib.rnn.LSTMCell(nhidden, num_proj = nproj, state_is_tuple = True)
                    else:
                        cell = tf.contrib.rnn.BasicLSTMCell(nhidden, state_is_tuple = True)
                    # outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, outputs,
                    # self.seq_len, swap_memory=True, time_major = True, dtype = tf.float32)
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, outputs,
                        self.seq_len, time_major = True, dtype = tf.float32)
                    # also some API change
                    outputs = tf.concat_v2(values = outputs, axis = 2, name = "output")
                    # outputs = tf.concat(values = outputs, axis = 2, name = "output")
            # for i in range(nlayer):
                # with tf.variable_scope("layer%d" % i):
                    # cell = tf.contrib.rnn.LSTMBlockCell(nhidden)
                    # if nproj > 0:
                        # cell = tf.contrib.rnn.OutputProjectionWrapper(cell, nproj)
                    # outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell,
                        # outputs, self.seq_len, swap_memory=True, dtype = tf.float32, time_major = True)
                    # # outputs = tf.concat_v2(outputs, 2, name = "output")
                    # outputs = tf.concat(outputs, 2, name = "output")
                    # outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, outputs, self.seq_len, dtype = tf.float32)
        return outputs

    def my_sat_layers(self, num_sat_layers, adapt_dim, nfeat, outputs):

        for i in range(num_sat_layers-1):
            with tf.variable_scope("layer%d" % i):
                outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = adapt_dim)

        with tf.variable_scope("last_sat_layer"):
            outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = nfeat)

        return outputs


    def my_sat_module(self, config, input_feats, input_sat):


        if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                == constants.SAT_SATGES.TRAIN_SAT:

            self.is_trainable_sat=False

        with tf.variable_scope(constants.SCOPES.SPEAKER_ADAPTAION):


            if(config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] == constants.SAT_TYPE.CONCAT):

                with tf.variable_scope(constants.SCOPES.SAT_FUSE):
                    sat_input = tf.tile(input_sat, tf.stack([tf.shape(input_feats)[0], 1, 1]))
                    outputs = tf.concat([input_feats, sat_input], 2)

                    return self.my_sat_layers(
                                    config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.NUM_SAT_LAYERS],
                                    config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_FEAT_DIM],
                                    config[constants.CONF_TAGS.INPUT_FEATS_DIM],
                                           outputs)

            elif(config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] == constants.SAT_TYPE.SHIFT):

                    learned_sat = self.my_sat_layers(
                        config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.NUM_SAT_LAYERS],
                                                     config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_FEAT_DIM],
                                                     config[constants.CONF_TAGS.INPUT_FEATS_DIM],
                                                    input_sat)

                    return tf.add(input_feats, learned_sat, name="shift")
            else:

                print("this sat type ("+str(config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE])+") was not contemplates")
                print(debug.get_debug_info())
                print("exiting...")
                sys.exit()

    def __init__(self, config):

        nfeat = config[constants.CONF_TAGS.INPUT_FEATS_DIM]

        nhidden = config[constants.CONF_TAGS.NHIDDEN]
        language_scheme = config[constants.CONF_TAGS.LANGUAGE_SCHEME]
        l2 = config[constants.CONF_TAGS.L2]
        nlayer = config[constants.CONF_TAGS.NLAYERS]
        clip = config[constants.CONF_TAGS.CLIP]
        nproj = config[constants.CONF_TAGS.NPROJ]
        dropout = config[constants.CONF_TAGS.DROPOUT]
        clip_norm = config[constants.CONF_TAGS.CLIP_NORM]
        kl_weight = config[constants.CONF_TAGS.KL_WEIGHT]

        batch_norm = config[constants.CONF_TAGS.BATCH_NORM]
        lstm_type = config[constants.CONF_TAGS.LSTM_TYPE]
        grad_opt = config[constants.CONF_TAGS.GRAD_OPT]

        tf.set_random_seed(config[constants.CONF_TAGS.RANDOM_SEED])


        if(constants.CONF_TAGS.INIT_NPROJ in config):
            init_nproj = config[constants.CONF_TAGS.INIT_NPROJ]
        else:
            init_nproj = 0

        if(constants.CONF_TAGS.FINAL_NPROJ in config):
            finalfeatproj = config[constants.CONF_TAGS.FINAL_NPROJ]
        else:
            finalfeatproj = 0

        if(constants.CONFIG_TAGS_TEST in config):
            self.is_training = False
        else:
            self.is_training = True

        self.is_trainable_sat=True

        try:
            featproj = config["feat_proj"]
        except:
            featproj = 0

        # build the graph
        self.lr_rate = tf.placeholder(tf.float32, name = "learning_rate")[0]
        self.feats = tf.placeholder(tf.float32, [None, None, nfeat], name = "feats")
        self.temperature = tf.placeholder(tf.float32, name = "temperature")
        self.is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")
        self.opt = []


        self.labels=[]
        self.priors=[]

        #optional outputs for test
        self.ters = []
        self.costs = []
        self.debug_costs = []

        #mantadory outpus for test
        self.softmax_probs = []
        self.log_softmax_probs = []
        self.log_likelihoods = []
        self.seq_len = []
        self.logits = []

        #creating enough placeholders for out graph
        for language_id, target_scheme in language_scheme.items():
            for target_id, _ in target_scheme.items():
                self.labels.append(tf.sparse_placeholder(tf.int32))
                self.priors.append(tf.placeholder(tf.float32))

        self.seq_len = self.length(self.feats)

        batch_size = tf.shape(self.feats)[0]
        outputs = tf.transpose(self.feats, (1, 0, 2), name = "feat_transpose")

        if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:

            self.sat = tf.placeholder(tf.float32, [None, 1, config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_FEAT_DIM]], name="sat")
            sat_t = tf.transpose(self.sat, (1, 0, 2), name="sat_transpose")

            outputs = self.my_sat_module(config, outputs, sat_t)

        if batch_norm:
            outputs = tf.contrib.layers.batch_norm(outputs, scope = "bn", center=True, scale=True, decay=0.9, is_training=self.is_training_ph, updates_collections=None)


        if init_nproj > 0:
            outputs = tf.contrib.layers.fully_connected(
                activation_fn = None, inputs = outputs, num_outputs = init_nproj,
                scope = "init_projection")

        if lstm_type == "cudnn":
            outputs = self.my_cudnn_lstm(outputs, batch_size, nlayer, nhidden, nfeat, nproj,  "cudnn_lstm", batch_norm, dropout, self.is_training)
        elif lstm_type == "fuse":
            outputs = self.my_fuse_block_lstm(outputs, batch_size, nlayer, nhidden, nfeat, nproj, "fuse_lstm")
        else:
            outputs = self.my_native_lstm(outputs, batch_size, nlayer, nhidden, nfeat, nproj, "native_lstm")



        #we should first verify this....
        #T time
        #B batch
        #D dimension of the output
        # (T,B,D) => (B,T,D)

        attention_size=3
        with tf.variable_scope("attention"):
            w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
            b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
            u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        outputs = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)



        if finalfeatproj > 0:
            outputs = tf.contrib.layers.fully_connected(
                activation_fn = None, inputs = outputs, num_outputs = finalfeatproj,
                scope = "final_projection")

        with tf.variable_scope("optimizer"):
            optimizer = None
            # TODO: cudnn only supports grad, but I think that adam should also work
            if grad_opt == "grad":
                optimizer = tf.train.GradientDescentOptimizer(self.lr_rate)
            elif grad_opt == "adam":
                optimizer = tf.train.AdamOptimizer(self.lr_rate)
            elif grad_opt == "momentum":
                optimizer = tf.train.MomentumOptimizer(self.lr_rate, 0.9)

        count=0

        print(80 * "-")
        print("preparing model variables...")
        print(80 * "-")


        for language_id, language_target_dict in language_scheme.items():


            tmp_ctc_cost, tmp_debug_cost, tmp_ter, tmp_logits, tmp_softmax_probs, tmp_log_softmax_probs, tmp_log_likelihoods = [], [], [], [], [], [], []
            tmp_kl_cost = []

            with tf.variable_scope(constants.SCOPES.OUTPUT):
                for target_id, num_targets in language_target_dict.items():

                    scope = "output_fc"

                    if(len(language_scheme.items()) > 1):
                        scope=scope+"_"+language_id
                    if(len(language_target_dict.items()) > 1):
                        scope=scope+"_"+target_id

                    if batch_norm:
                        outputs = tf.contrib.layers.batch_norm(outputs, scope = scope+"_bn", center=True, scale=True, decay=0.9,
                                                             is_training=self.is_training_ph, updates_collections=None)


                    logit = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs,
                                                              num_outputs=num_targets,
                                                              scope = scope,
                                                              biases_initializer = tf.contrib.layers.xavier_initializer(),
                                                              trainable=self.is_trainable_sat)

                    #######
                    #here logits: time, batch, num_classes+1
                    #######

                    kl_loss = self.kl_divergence_with_logits(logit, num_targets)

                    ctc_loss = tf.nn.ctc_loss(labels=self.labels[count], inputs=logit, sequence_length=self.seq_len)

                    tmp_kl_cost.append(kl_loss)
                    tmp_ctc_cost.append(ctc_loss)
                    tmp_debug_cost.append(tf.reduce_mean(ctc_loss))

                    decoded, log_prob = tf.nn.ctc_greedy_decoder(logit, self.seq_len)

                    ter = tf.reduce_sum(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.labels[count], normalize = False))
                    tmp_ter.append(ter)

                    #storing outputs
                    tran_logit = tf.transpose(logit, (1, 0, 2)) * self.temperature
                    tmp_logits.append(tran_logit)

                    softmax_prob = tf.nn.softmax(tran_logit, dim=-1, name=None)
                    tmp_softmax_probs.append(softmax_prob)

                    log_softmax_prob = tf.log(softmax_prob)
                    tmp_log_softmax_probs.append(log_softmax_prob)

                    log_likelihood = log_softmax_prob - tf.log(self.priors[count])
                    tmp_log_likelihoods.append(log_likelihood)



                    count=count+1

            #preparing variables to optimize
            if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                == constants.SAT_SATGES.TRAIN_SAT:
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=constants.SCOPES.SPEAKER_ADAPTAION)
            else:
                if(len(language_scheme.items()) > 1):
                    var_list = self.get_variables_by_lan(language_id)
                else:
                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


            if(self.is_training):
                print(80 * "-")
                if(len(language_scheme.items()) > 1):
                    print("for language: "+language_id)
                print("following variables will be optimized: ")
                print(80 * "-")
                for var in var_list:
                    print(var)
                print(80 * "-")
            else:
                print(80 * "-")
                print("testing... no variables will be optimized.")
                print(80 * "-")

            with tf.variable_scope("loss"):
                regularized_loss = tf.add_n([tf.nn.l2_loss(v) for v in var_list])

            #reduce the mean of all targets of current language(language_id)
            tmp_ctc_cost = tf.reduce_mean(tmp_ctc_cost) + l2 * regularized_loss

            tmp_final_cost = (1-kl_weight)*tf.reduce_mean(tmp_ctc_cost)-(kl_weight*tf.reduce_mean(tmp_kl_cost))+(l2 * regularized_loss)

            self.debug_costs.append(tmp_debug_cost)
            self.costs.append(tmp_final_cost)
            self.ters.append(tmp_ter)
            self.logits.append(tmp_logits)
            self.softmax_probs.append(tmp_softmax_probs)

            self.log_softmax_probs.append(tmp_log_softmax_probs)
            self.log_likelihoods.append(tmp_log_likelihoods)

            #gvs = optimizer.compute_gradients(tmp_ctc_cost, var_list=var_list)
            gvs = optimizer.compute_gradients(tmp_final_cost, var_list=var_list)

            if(clip_norm):
                capped_gvs = [(tf.clip_by_norm(grad, clip), var) for grad, var in gvs]
            else:
                capped_gvs = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in gvs]

            #at end  of the day we will decide whch optimizer to call:
            #each one will optimize over all targets of the selected language (correct idx)
            self.opt.append(optimizer.apply_gradients(capped_gvs))

            print(80 * "-")

    def get_variables_by_lan(self, current_name):

        train_vars=[]
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            if("output_fc" not in var.name):
                train_vars.append(var)
            elif(current_name in var.name):
                train_vars.append(var)

        return train_vars

