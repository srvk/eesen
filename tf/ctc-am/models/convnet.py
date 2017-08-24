import constants
import tensorflow as tf


class ConvNet:

    def length(self, sequence):
        with tf.variable_scope("seq_len"):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
            length = tf.reduce_sum(used, axis=1)
            length = tf.cast(length, tf.int32)
        return length


    #def my_cnn (self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope, batch_norm, is_training = True):

    #    if(nlayer > 0 ):


    def __init__(self, config):

        nfeat = config[constants.CONF_TAGS.INPUT_FEATS_DIM]
        nhidden = config[constants.CONF_TAGS.NHIDDEN]
        language_scheme = config[constants.CONF_TAGS.LANGUAGE_SCHEME]
        l2 = config[constants.CONF_TAGS.L2]
        nlayer = config[constants.CONF_TAGS.NLAYERS]
        clip = config[constants.CONF_TAGS.CLIP]
        nproj = config[constants.CONF_TAGS.NPROJ]
        batch_norm = config[constants.CONF_TAGS.BATCH_NORM]
        lstm_type = config[constants.CONF_TAGS.LSTM_TYPE]
        grad_opt = config[constants.CONF_TAGS.GRAD_OPT]

        if(constants.CONFIG_TAGS_TEST in config):
            self.is_training = False
        else:
            self.is_training = True

        if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                != constants.SAT_SATGES.UNADAPTED:
            num_sat_layers = config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.NUM_SAT_LAYERS]
            adapt_dim = config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_FEAT_DIM]
            self.is_trainable_sat=False

        else:
            self.is_trainable_sat=True

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

        output_size = 2 * nhidden if nproj == 0 else nproj
        batch_size = tf.shape(self.feats)[0]
        #outputs = tf.transpose(self.feats, (1, 0, 2), name = "feat_transpose")

        if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_SATGE] \
                != constants.SAT_SATGES.UNADAPTED:
            #SAT
            with tf.variable_scope(constants.SCOPES.SPEAKER_ADAPTAION):
                self.sat = tf.placeholder(tf.float32, [None, 1, adapt_dim], name = "sat")

                sat_t=tf.transpose(self.sat, (1, 0, 2), name = "sat_transpose")

                learned_sat = self.my_sat_layers(num_sat_layers, adapt_dim,  nfeat, sat_t, "sat_layers")

        with tf.variable_scope("input"):
            # outputs = tf.transpose(self.feats, (0, 1, 2), name = "feat_transpose") # (B,T,F) -> (B,F,T)
            outputs = tf.expand_dims(self.feats, -1) # (B,T,F) -> (B,T,F,1)
            # self.shape_0 = tf.shape(outputs)




        ################################CONV1##################################################
        with tf.name_scope('conv1_0') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 1, 64], dtype=tf.float32, stddev=1e-1))

            outputs = tf.nn.conv2d(outputs, kernel, [1, 1, 1, 1], padding='SAME',  use_cudnn_on_gpu=True, name="conv0") # (B,T,F,1) -> (B,T,F,32)

            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=True, name='biases_conv0')

            outputs= tf.nn.bias_add(outputs, biases)

            outputs = tf.nn.relu(outputs, name="relu_conv0")
            # self.shape_1 = tf.shape(outputs)

        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 10, 64, 64], dtype=tf.float32, stddev=1e-1))

            outputs = tf.nn.conv2d(outputs, kernel, [1, 1, 1, 1], padding='SAME',  use_cudnn_on_gpu=True, name="conv1") # (B,T,F,1) -> (B,T,F,32)

            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases_conv1')

            outputs= tf.nn.bias_add(outputs, biases)

            outputs = tf.nn.relu(outputs, name="relu_conv1")
            # self.shape_1 = tf.shape(outputs)
            # outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, decay=0.9, is_training=self.is_training, updates_collections=None)

            outputs = tf.nn.max_pool(outputs, ksize=[1, 1, 2, 1],strides=[1, 1, 2, 1],padding='SAME',name='pool1') # (B,T,F,32)
            # self.shape_2 = tf.shape(outputs)

        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 40, 64, 128], dtype=tf.float32,
                                                         stddev=1e-1))

            outputs = tf.nn.conv2d(outputs, kernel, [1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True, name="conv2") # (B,T,F/2+1,32) -> (B,T,F/2+1,32)

            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=True, name='biases_conv2')
            # bias=tf.Variable(tf.random_normal([64]), name="bias_conv2")

            outputs= tf.nn.bias_add(outputs, biases)

            outputs = tf.nn.relu(outputs, name="relu_conv2")
            # self.shape_3 = tf.shape(outputs)
            # pool1
            outputs = tf.nn.max_pool(outputs, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME', name='pool2') # (B,T,F,32) -> (B,T,(F/2+1)/2+1,32)
            # self.shape_4 = tf.shape(outputs)

        with tf.name_scope('conv1_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([20, 1, 128, 128], dtype=tf.float32,
                                                         stddev=1e-1))

            outputs = tf.nn.conv2d(outputs, kernel, [1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True, name="conv3") # (B,T,F/2+1,32) -> (B,T,F/2+1,32)

            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=True, name='biases_conv3')
            # bias=tf.Variable(tf.random_normal([64]), name="bias_conv2")

            outputs= tf.nn.bias_add(outputs, biases)

            outputs = tf.nn.relu(outputs, name="relu_conv3")


        outputs = tf.reshape(outputs, [tf.shape(outputs)[0], tf.shape(outputs)[1], (int((int(nfeat / 2) + 1) / 2) + 1) * 128])
        outputs = tf.transpose(outputs, (1, 0, 2), name="feat_transpose")


        if batch_norm:
            outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, decay=0.9, is_training=self.is_training_ph, updates_collections=None)


        with tf.variable_scope("optimizer"):
            optimizer = None
            # TODO: cudnn only supports grad, add check for this
            if grad_opt == "grad":
                optimizer = tf.train.GradientDescentOptimizer(self.lr_rate)
            elif grad_opt == "adam":
                optimizer = tf.train.AdamOptimizer(self.lr_rate)
            elif grad_opt == "momentum":
                optimizer = tf.train.MomentumOptimizer(self.lr_rate, 0.9)


        print(80 * "-")
        print("preparing model variables...")
        print(80 * "-")


        count=0
        for language_id, language_target_dict in language_scheme.items():

            tmp_cost, tmp_debug_cost, tmp_ter, tmp_logits, tmp_softmax_probs, tmp_log_softmax_probs, tmp_log_likelihoods = [], [], [], [], [], [], []

            with tf.variable_scope(constants.SCOPES.OUTPUT):
                for target_id, num_targets in language_target_dict.items():

                    scope="output_fc_"+language_id+"_"+target_id

                    outputs = tf.contrib.layers.fully_connected(
                    activation_fn = tf.nn.relu, inputs = outputs, num_outputs = 1000, scope = "input_fc0", biases_initializer = tf.contrib.layers.xavier_initializer())
                    outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, decay=0.9, is_training=self.is_training_ph, updates_collections=None)
                    #outputs = tf.nn.dropout(outputs, self.drop_out)
                    outputs = tf.contrib.layers.fully_connected( activation_fn = tf.nn.relu, inputs = outputs, num_outputs = 500, scope = "input_fc", biases_initializer = tf.contrib.layers.xavier_initializer())
                    outputs = tf.contrib.layers.fully_connected( activation_fn = tf.nn.relu, inputs = outputs, num_outputs = 500, scope = "input_fc_2", biases_initializer = tf.contrib.layers.xavier_initializer())
                    outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, decay=0.9, is_training=self.is_training_ph, updates_collections=None)
                    #outputs = tf.nn.dropout(outputs, self.drop_out)

                    logit = tf.contrib.layers.fully_connected(activation_fn=None, inputs=outputs, num_outputs=num_targets, scope=scope,
                                                              biases_initializer=tf.contrib.layers.xavier_initializer())

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

