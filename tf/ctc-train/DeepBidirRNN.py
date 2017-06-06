import tensorflow as tf

class DeepBidirRNN:
    def length(self, sequence):
        with tf.variable_scope("seq_len"):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
            length = tf.reduce_sum(used, axis=1)
            length = tf.cast(length, tf.int32)
        return length

    def my_cudnn_lstm(self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope, is_training = True):
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
                        cudnn_params = tf.Variable(tf.random_uniform([params_size_t], -bound, bound),
                            validate_shape = False, name = "params")
                        outputs, _output_h, _output_c = cudnn_model(is_training=is_training,
                            input_data=outputs, input_h=input_h, input_c=input_c,
                            params=cudnn_params)
                        outputs = tf.contrib.layers.fully_connected(
                            activation_fn = None, inputs = outputs,
                            num_outputs = nproj, scope = "projection")
                        ninput = nproj
            else:
                cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(nlayer, nhidden, nfeat, direction = 'bidirectional')
                params_size_t = cudnn_model.params_size()
                input_h = tf.zeros([nlayer * 2, batch_size, nhidden], dtype = tf.float32, name = "init_lstm_h")
                input_c = tf.zeros([nlayer * 2, batch_size, nhidden], dtype = tf.float32, name = "init_lstm_c")
                bound = tf.sqrt(6. / (nhidden + nhidden))
                cudnn_params = tf.Variable(tf.random_uniform([params_size_t], -bound, bound),
                    validate_shape = False, name = "params")
                outputs, _output_h, _output_c = cudnn_model(is_training=is_training,
                    input_data=outputs, input_h=input_h, input_c=input_c,
                    params=cudnn_params)
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

    def __init__(self, config):
        nfeat = config["nfeat"]
        nhidden = config["nhidden"]
        nclasses = config["nclass"]
        l2 = config["l2"]
        nlayer = config["nlayer"]
        clip = config["clip"]
        nproj = config["nproj"]
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

        self.labels = [tf.sparse_placeholder(tf.int32)
          for _ in xrange(len(nclasses))]

        self.priors =[tf.placeholder(tf.float32, nclass)
          for count, nclass in enumerate(nclasses)]

        self.seq_len = self.length(self.feats)

        output_size = 2 * nhidden if nproj == 0 else nproj

        batch_size = tf.shape(self.feats)[0]

        outputs = tf.transpose(self.feats, (1, 0, 2), name = "feat_transpose")

        if featproj > 0:
            outputs = tf.contrib.layers.fully_connected(
                activation_fn = None, inputs = outputs, num_outputs = featproj,
                scope = "input_fc", biases_initializer = tf.contrib.layers.xavier_initializer())
        if lstm_type == "cudnn":
            outputs = self.my_cudnn_lstm(outputs, batch_size, nlayer, nhidden, nfeat, nproj, "cudnn_lstm")
        elif lstm_type == "fuse":
            outputs = self.my_fuse_block_lstm(outputs, batch_size, nlayer, nhidden, nfeat, nproj, "fuse_lstm")
        else:
            outputs = self.my_native_lstm(outputs, batch_size, nlayer, nhidden, nfeat, nproj, "native_lstm")

        logits=[]
        for count_label, _ in enumerate(nclasses):
            logit = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = nclasses[count_label], scope = "output_fc_"+str(count_label), biases_initializer = tf.contrib.layers.xavier_initializer())

            logits.append(logit)

        with tf.variable_scope("loss"):
            regularized_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        losses=[]
        for idx, logit in enumerate(logits):
            loss = tf.nn.ctc_loss(labels=self.labels[idx], inputs=logit, sequence_length=self.seq_len)
            losses.append(loss)

        self.cost = tf.reduce_mean(losses) + l2 * regularized_loss

        self.softmax_probs=[]
        self.log_softmax_probs=[]
        self.log_likelihoods=[]

        with tf.variable_scope("eval_output"):
            tran_logit = tf.transpose(logit, (1, 0, 2)) * self.temperature

            softmax_prob = tf.nn.softmax(tran_logit, dim=-1, name=None)
            self.softmax_probs.append(softmax_prob)

            log_softmax_prob = tf.log(softmax_prob)
            self.log_softmax_probs.append(log_softmax_prob)

            log_likelihood = log_softmax_prob - tf.log(self.priors[0])
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
            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in gvs]
            self.opt = optimizer.apply_gradients(capped_gvs)

        self.decodes=[]
        self.log_probs=[]
        self.cers=[]

        for idx, _ in enumerate(nclasses):
            decoded, log_prob = tf.nn.ctc_greedy_decoder(logit, self.seq_len)
            cer = tf.reduce_sum(
                tf.edit_distance(tf.cast(decoded[0], tf.int32), self.labels[idx] , normalize = False), name = "cer")

            self.decodes.append(decoded)
            self.log_probs.append(log_prob)
            self.cers.append(cer)

