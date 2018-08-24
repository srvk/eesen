import tensorflow as tf
import constants
import numpy as np

from core.layers import layers

FLAGS = tf.app.flags.FLAGS

class ArcNetVideo:

    '''
    LipNet Model
    '''
    def length(self, sequence):
        with tf.variable_scope("seq_len"):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=[2,3,4]))
            length = tf.reduce_sum(used, axis=1)
            length = tf.cast(length, tf.int32)
        return length



    def my_cudnn_lstm(self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope, batch_norm, is_training = True):
        """
        outputs: time, batch_size, feat_dim
        """
        if(nlayer > 0 ):
            with tf.variable_scope(scope):
                if nproj > 0:
                    ninput = nfeat
                    for i in range(nlayer):
                        with tf.variable_scope("layer%d" % i):

                            cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(1, nhidden, ninput, direction = 'bidirectional')
                            params_size_t = cudnn_model.params_size()
                            input_h = tf.zeros([2, batch_size, nhidden], dtype = tf.float32, name = "init_lstm_h")
                            input_c = tf.zeros([2, batch_size, nhidden], dtype = tf.float32, name = "init_lstm_c")
                            bound = tf.sqrt(6. / (nhidden + nhidden))
                            cudnn_params = tf.Variable(tf.random_uniform([params_size_t], -bound, bound), validate_shape = False, name = "params", trainable=self.is_trainable_sat)
                            #TODO is_training=is_training should be changed!
                            outputs, _output_h, _output_c = cudnn_model(is_training=is_training,
                                                                        input_data=outputs, input_h=input_h, input_c=input_c,
                                                                        params=cudnn_params)
                            outputs = tf.contrib.layers.fully_connected(
                                activation_fn = None, inputs = outputs,
                                num_outputs = nproj, scope = "projection")

                            if(batch_norm):
                                outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, decay=0.9, is_training=self.is_training_ph, updates_collections=None)

                            ninput = nproj
                else:
                    cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(nlayer, nhidden, nfeat, direction = 'bidirectional')
                    params_size_t = cudnn_model.params_size()
                    input_h = tf.zeros([nlayer * 2, batch_size, nhidden], dtype = tf.float32, name = "init_lstm_h")
                    input_c = tf.zeros([nlayer * 2, batch_size, nhidden], dtype = tf.float32, name = "init_lstm_c")
                    bound = tf.sqrt(6. / (nhidden + nhidden))

                    cudnn_params = tf.Variable(tf.random_uniform([params_size_t], -bound, bound),
                                               validate_shape = False, name = "params", trainable=self.is_trainable_sat)

                    outputs, _output_h, _output_c = cudnn_model(is_training=is_training,input_data=outputs,
                                                                input_h=input_h, input_c=input_c,params=cudnn_params)

                    if(batch_norm):
                        outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, decay=0.9, is_training=self.is_training_ph, updates_collections=None)

        return outputs

    def my_fuse_block_lstm(self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope):
        """
        outputs: time, batch_size, feat_dim
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

    def my_cnn(self, x, batch_size, is_training):

        with tf.variable_scope('video'):
            CONV_KEEP_PROB = 0.7
            CONV_ACTIVATION = 'relu'

            # assume input is 5D and has shape:
            # (batch, n_frames, video_height, video_width, video_channels)
            # filters=[depth, height, width, in_channels, out_channels],
            st_net = []  # spatiotemporal
            v_conv_1, weights, biases = layers.conv3d(
                input=x,
                shape=(3, 5, 5, 3, 32),
                strides=[1, 1, 2, 2, 1],
                padding='SAME',
                activation=CONV_ACTIVATION,
                name='v_conv_1')
            st_net.append(v_conv_1)

            # Even though it says 3d, we're using it only spatially
            st_net.append(tf.nn.max_pool3d(
                input=st_net[-1],
                ksize=[1, 1, 2, 2, 1],
                strides=[1, 1, 2, 2, 1],
                padding='SAME',
                name='v_pool_1'
            ))

            st_net.append(layers.batch_norm(st_net[-1],
                                            phase=is_training,
                                            name='bn_1'))

            st_net.append(layers.channel_dropout(
                input=st_net[-1],
                keep_prob=CONV_KEEP_PROB,
                is_training=is_training,
                name='v_dropout_1',
            ))

            st_net.append(layers.conv3d(
                input=st_net[-1],
                shape=(3, 5, 5, 32, 64),
                strides=[1, 1, 1, 1, 1],
                padding='SAME',
                activation=CONV_ACTIVATION,
                name='v_conv_2')[0])

            st_net.append(tf.nn.max_pool3d(
                input=st_net[-1],
                ksize=[1, 1, 2, 2, 1],
                strides=[1, 1, 2, 2, 1],
                padding='SAME',
                name='v_pool_2'
            ))

            st_net.append(layers.batch_norm(
                st_net[-1],
                phase=is_training,
                name='bn_2'))

            st_net.append(layers.channel_dropout(
                input=st_net[-1],
                keep_prob=CONV_KEEP_PROB,
                is_training=is_training,
                name='v_dropout_2',
            ))

            st_net.append(layers.conv3d(
                input=st_net[-1],
                shape=(3, 3, 3, 64, 96),
                strides=[1, 1, 1, 1, 1],
                padding='SAME',
                activation=CONV_ACTIVATION,
                name='v_conv_3')[0])

            st_net.append(tf.nn.max_pool3d(
                input=st_net[-1],
                ksize=[1, 1, 2, 2, 1],
                strides=[1, 1, 2, 2, 1],
                padding='SAME',
                name='v_pool_3'
            ))

            st_net.append(layers.batch_norm(
                st_net[-1],
                phase=is_training,
                name='bn_3'))

            st_net.append(layers.channel_dropout(
                input=st_net[-1],
                keep_prob=CONV_KEEP_PROB,
                is_training=is_training,
                name='v_dropout_3',
            ))

            # Flatten the video branch output into a feature vector
            # The "Output" layer of the video branch.
            # Output shape: (batch, frames, features)
            fan_in = int(np.prod(st_net[-1].get_shape()[2:]))
            shape = [batch_size, -1, fan_in]

            st_net.append(tf.reshape(st_net[-1], shape))

        # The "Output" layer of the video branch.
        return st_net[-1]

    def __init__(self, config):

        nfeat = config[constants.CONF_TAGS.INPUT_FEATS_DIM]

        nhidden = config[constants.CONF_TAGS.NHIDDEN]
        language_scheme = config[constants.CONF_TAGS.LANGUAGE_SCHEME]
        l2 = config[constants.CONF_TAGS.L2]
        nlayer = config[constants.CONF_TAGS.NLAYERS]
        clip = config[constants.CONF_TAGS.CLIP]
        nproj = config[constants.CONF_TAGS.NPROJ]

        if(constants.CONF_TAGS.FINAL_NPROJ in config):
            finalfeatproj = config[constants.CONF_TAGS.FINAL_NPROJ]
        else:
            finalfeatproj = 0

        batch_norm = config[constants.CONF_TAGS.BATCH_NORM]
        lstm_type = config[constants.CONF_TAGS.LSTM_TYPE]
        grad_opt = config[constants.CONF_TAGS.GRAD_OPT]

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


        #shape=(BATCH_SIZE, None, VIDEO_SIZE[0], VIDEO_SIZE[1], 3), name='x_video')
        shape=(None, None, nfeat[0], nfeat[1], 3)

        self.feats = tf.placeholder(tf.float32, shape, name = "feats")

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

        outputs = self.my_cnn(self.feats, batch_size, self.is_training_ph)

        outputs = tf.transpose(outputs, (1, 0, 2), name = "feat_transpose")

        if lstm_type == "cudnn":
            outputs = self.my_cudnn_lstm(outputs, batch_size, nlayer, nhidden, int(outputs.get_shape()[-1]), nproj,  "cudnn_lstm", batch_norm, self.is_training)
        elif lstm_type == "fuse":
            print("not implemented")
            print("exiting....")
            sys.exit()
        else:
            print("not implemented")
            print("exiting....")
            sys.exit()


        if finalfeatproj > 0:
            outputs = tf.contrib.layers.fully_connected(
                activation_fn = None, inputs = outputs, num_outputs = finalfeatproj,
                scope = "input_fc", biases_initializer = tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("optimizer"):
            optimizer = None
            # TODO: cudnn only supports grad, add check for this
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

            tmp_cost, tmp_debug_cost, tmp_ter, tmp_logits, tmp_softmax_probs, tmp_log_softmax_probs, tmp_log_likelihoods = [], [], [], [], [], [], []

            with tf.variable_scope(constants.SCOPES.OUTPUT):
                for target_id, num_targets in language_target_dict.items():

                    scope="output_fc_"+language_id+"_"+target_id
                    logit = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs,
                                                              num_outputs=num_targets,
                                                              scope = scope,
                                                              biases_initializer = tf.contrib.layers.xavier_initializer(),
                                                              trainable=self.is_trainable_sat)

                    loss = tf.nn.ctc_loss(labels=self.labels[count], inputs=logit, sequence_length=self.seq_len)
                    tmp_cost.append(loss)
                    tmp_debug_cost.append(tf.reduce_mean(loss))

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
                var_list = self.get_variables_by_lan(language_id)

            if(self.is_training):
                print(80 * "-")
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
            tmp_cost = tf.reduce_mean(tmp_cost) + l2 * regularized_loss


            self.debug_costs.append(tmp_debug_cost)
            self.costs.append(tmp_cost)
            self.ters.append(tmp_ter)
            self.logits.append(tmp_logits)
            self.softmax_probs.append(tmp_softmax_probs)
            self.log_softmax_probs.append(tmp_log_softmax_probs)
            self.log_likelihoods.append(tmp_log_likelihoods)

            gvs = optimizer.compute_gradients(tmp_cost, var_list=var_list)

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

