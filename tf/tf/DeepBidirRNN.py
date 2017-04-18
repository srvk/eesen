import tensorflow as tf

class DeepBidirRNN:
    def length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def __init__(self, config):
        nfeat = config["nfeat"]
        nhidden = config["nhidden"]
        nclass = config["nclass"]
        l2 = config["l2"]
        nlayer = config["nlayer"]
        clip = config["clip"]
        nproj = config["nproj"]
        use_cudnn = config["cudnn"]
        grad_opt = config["grad_opt"]

        # build the graph
        self.lr_rate = tf.placeholder(tf.float32)[0]
        self.feats = tf.placeholder(tf.float32, [None, None, nfeat], name = "feats")
        self.labels = tf.sparse_placeholder(tf.int32, name = "labels")
        seq_len = self.length(self.feats)

        output_size = 2 * nhidden if nproj == 0 else 2 * nproj
        # output_size = 1 * nhidden
        batch_size = tf.shape(self.feats)[0]
        outputs = []
        cudnn_model = None
        cudnn_params = None
        if use_cudnn:
            outputs = tf.transpose(self.feats, (1, 0, 2))
            if nproj > 0:
                ninput = nfeat
                for i in range(nlayer):
                    with tf.variable_scope("layer%d" % i):
                        cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(1, nhidden, ninput, direction = 'bidirectional')
                        # cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(1, nhidden, nfeat)
                        params_size_t = cudnn_model.params_size()
                        input_h = tf.zeros([2, batch_size, nhidden], dtype = tf.float32)
                        input_c = tf.zeros([2, batch_size, nhidden], dtype = tf.float32)
                        bound = tf.sqrt(6. / (nhidden + nhidden))
                        cudnn_params = tf.Variable(tf.random_uniform([params_size_t], -bound, bound), validate_shape = False)
                        outputs, _output_h, _output_c = cudnn_model(is_training=True,
                            input_data=outputs,
                            input_h=input_h,
                            input_c=input_c,
                            params=cudnn_params)

                        proj_bound = tf.sqrt(3. / (nhidden + nproj)) 
                        Wproj = tf.Variable(tf.random_uniform([2 * nhidden, 2 * nproj], -proj_bound, proj_bound), name = "projection")
                        outputs = tf.reshape(outputs, [-1, 2 * nhidden])
                        outputs = tf.matmul(outputs, Wproj) 
                        outputs = tf.reshape(outputs, [-1, batch_size, output_size])
                        ninput = output_size
            else:
                cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(nlayer, nhidden, nfeat, direction = 'bidirectional')
                # cudnn_model = tf.contrib.cudnn_rnn.CudnnLSTM(nlayer, nhidden, nfeat)
                params_size_t = cudnn_model.params_size()
                input_h = tf.zeros([nlayer * 2, batch_size, nhidden], dtype = tf.float32)
                input_c = tf.zeros([nlayer * 2, batch_size, nhidden], dtype = tf.float32)
                bound = tf.sqrt(6. / (nhidden + nhidden))
                cudnn_params = tf.Variable(tf.random_uniform([params_size_t], -bound, bound), validate_shape = False)
                outputs, _output_h, _output_c = cudnn_model(is_training=True,
                    input_data=outputs,
                    input_h=input_h,
                    input_c=input_c,
                    params=cudnn_params)
            outputs = tf.transpose(outputs, (1, 0, 2))
        else:
            cell = []
            outputs = self.feats
            for i in range(nlayer):
                with tf.variable_scope("layer%d" % i):
                    cell = None
                    if nproj > 0:
                        cell = tf.contrib.rnn.LSTMCell(nhidden, num_proj = nproj, state_is_tuple = True,
                           initializer = tf.contrib.layers.xavier_initializer())
                    else:
                        cell = tf.contrib.rnn.BasicLSTMCell(nhidden, state_is_tuple = True)
                    # outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, outputs, seq_len, dtype = tf.float32)
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, outputs, seq_len, swap_memory=True, dtype = tf.float32)
                    outputs = tf.concat_v2(outputs, 2)
            # if nproj > 0:
                # for i in range(nlayer):
                    # with tf.variable_scope("layer%d" % i):
                        # cell = tf.contrib.rnn.LSTMCell(nhidden, num_proj = nproj, state_is_tuple = True,
                           # initializer = tf.contrib.layers.xavier_initializer())
                        # outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, outputs, seq_len, dtype = tf.float32)
                        # outputs = tf.concat_v2(outputs, 2)
            # else:
                # for i in range(nlayer):
                    # with tf.variable_scope("layer%d" % i):
                        # cell = tf.contrib.rnn.BasicLSTMCell(nhidden, state_is_tuple = True)
                        # outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, outputs, seq_len, dtype = tf.float32)
                        # outputs = tf.concat_v2(outputs, 2)

        outputs = tf.reshape(outputs, [-1, output_size])

        with tf.variable_scope("fc"):
            bound = tf.sqrt(6. / (output_size + nclass))
            W = tf.Variable(tf.random_uniform([output_size, nclass], -bound, bound), name ="Weight")
            b = tf.Variable(tf.constant(0., shape = [nclass]), name = "bias")
        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [batch_size, -1, nclass])

        logits = tf.transpose(logits, (1, 0, 2))
        loss = tf.nn.ctc_loss(self.labels, logits, seq_len)
        regularized_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.cost = tf.reduce_mean(loss) + l2 * regularized_loss
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

        if use_cudnn:
            self.cudnn_weight, self.cudnn_bias = cudnn_model.params_to_canonical(cudnn_params)

        self.decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
        self.wer = tf.reduce_sum(
            tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels,
            normalize = False))
