import tensorflow as tf

class DeepBidirRNN:
    def length(self, sequence):
        with tf.variable_scope("seq_len"):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
            length = tf.reduce_sum(used, axis=1)
            length = tf.cast(length, tf.int32)
        return length

    def my_cudnn_lstm(self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope, batch_norm, is_training = True):
        """
        outputs: time, batch_size, feat_dim
        """
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
                            outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True,decay=0.9, is_training=self.is_training,  updates_collections=None)

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
                    outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True,decay=0.9, is_training=self.is_training,  updates_collections=None)

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
                    outputs = tf.concat(values = [fw_out, bw_out], axis = 2, name = "output")
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
                    outputs = tf.concat(values = outputs, axis = 2, name = "output")
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

    def my_sat_layers(self, num_sat_layers, adapt_dim, nfeat, outputs, scope):

        with tf.variable_scope(scope):
            for i in range(num_sat_layers-1):
                with tf.variable_scope("layer%d" % i):
                    outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = adapt_dim)

            with tf.variable_scope("last_sat_layer"):
                outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = nfeat)

        return outputs


    def __init__(self, config):
        nfeat = config["nfeat"]
        nhidden = config["nhidden"]
        nclasses = config["nclass"]
        l2 = config["l2"]
        nlayer = config["nlayer"]
        clip = config["clip"]
        nproj = config["nproj"]
        batch_norm = config["batch_norm"]


        adaptation_stage = config["adapt_stage"]

        if adaptation_stage == 'train_adapt':

            num_sat_layers = int(config["num_sat_layers"])
            adapt_dim = int(config["adapt_dim"])
            self.is_trainable_sat=False

        elif adaptation_stage == 'fine_tune':

            num_sat_layers = int(config["num_sat_layers"])
            adapt_dim = config["adapt_dim"]
            self.is_trainable_sat=True

        elif adaptation_stage == 'unadapted':
            self.is_trainable_sat=True

        try:
            featproj = config["feat_proj"]
        except:
            featproj = 0
        lstm_type = config["lstm_type"]
        grad_opt = config["grad_opt"]

        # build the graph
        self.lr_rate = tf.placeholder(tf.float32, name = "learning_rate")[0]
        self.feats = tf.placeholder(tf.float32, [None, None, nfeat], name = "feats")
        self.temperature = tf.placeholder(tf.float32, name = "temperature")
        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
        self.labels = [tf.sparse_placeholder(tf.int32)
                       for _ in range(len(nclasses))]                        
        self.prior = [tf.placeholder(tf.float32, nclass, name = "prior"+str(count))
                      for count, nclass in enumerate(nclasses)]
        self.seq_len = self.length(self.feats)

        output_size = 2 * nhidden if nproj == 0 else nproj
        batch_size = tf.shape(self.feats)[0]
        outputs = tf.transpose(self.feats, (1, 0, 2), name = "feat_transpose")

        if adaptation_stage != 'unadapted':
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

        logits=[]
        for count_label, _ in enumerate(nclasses):
            scope = "output_fc"
            if len(nclasses) > 1:
                scope = "output_fc_"+str(count_label)
            logit = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = nclasses[count_label], scope = scope, biases_initializer = tf.contrib.layers.xavier_initializer())
            if batch_norm:
                logit = tf.contrib.layers.batch_norm(logit, center=True, scale=True, decay=0.9, is_training=self.is_training,  updates_collections=None)
            logits.append(logit)

        with tf.variable_scope("loss"):
            regularized_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        losses=[]
        for idx, logit in enumerate(logits):
            loss = tf.nn.ctc_loss(labels=self.labels[idx], inputs=logit, sequence_length=self.seq_len), ignore_longer_outputs_than_inputs=True)
            losses.append(loss)

        self.cost = tf.reduce_mean(losses) + l2 * regularized_loss

        self.softmax_probs=[]
        self.log_softmax_probs=[]
        self.log_likelihoods=[]
        self.logits=[]

        with tf.variable_scope("eval_output"):
            for idx, logit in enumerate(logits):
                tran_logit = tf.transpose(logit, (1, 0, 2)) * self.temperature
                self.logits.append(tran_logit)

                softmax_prob = tf.nn.softmax(tran_logit, dim=-1, name=None)
                self.softmax_probs.append(softmax_prob)

                log_softmax_prob = tf.log(softmax_prob)
                self.log_softmax_probs.append(log_softmax_prob)

                log_likelihood = log_softmax_prob - tf.log(self.priors[idx])
                self.log_likelihoods.append(log_likelihood)

        with tf.variable_scope("optimizer"):
            optimizer = None
            # TODO: cudnn only supports grad, add check for this
            if grad_opt == "grad":
                optimizer = tf.train.GradientDescentOptimizer(self.lr_rate)
            elif grad_opt == "adam":
                optimizer = tf.train.AdamOptimizer(self.lr_rate)
            elif grad_opt == "momentum":
                optimizer = tf.train.MomentumOptimizer(self.lr_rate, 0.9)

            train_vars=[]
            if(not self.is_trainable_sat):
                train_vars_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                for var in train_vars_all:
                    if "sat" in var.name:
                        train_vars.append(var)
            else:
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            gvs = optimizer.compute_gradients(self.cost, var_list=train_vars)

            capped_gvs = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in gvs]
            self.opt = optimizer.apply_gradients(capped_gvs)

        self.decodes=[]
        self.log_probs=[]
        self.ters=[]

        for idx, _ in enumerate(nclasses):
            decoded, log_prob = tf.nn.ctc_greedy_decoder(logits[idx], self.seq_len)
            ter = tf.reduce_sum(
                tf.edit_distance(tf.cast(decoded[0], tf.int32), self.labels[idx] , normalize = False), name = "ter")

            self.decodes.append(decoded)
            self.log_probs.append(log_prob)
            self.ters.append(ter)
