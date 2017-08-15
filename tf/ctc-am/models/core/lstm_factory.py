import tensorflow as tf
import constants


class LstmFactory:

    def get_lstm(self, lstm_type, outputs, batch_size, nlayer, nhidden, nfeat, nproj, batch_norm):
        if(lstm_type == constants.LSTM_TYPE.CUDNN):
            return self.my_cudnn_lstm(outputs, batch_size, nlayer, nhidden, nfeat, nproj,  "cudnn_lstm", batch_norm)
        if(lstm_type == constants.LSTM_TYPE.FUSE):
            return self.my_fuse_block_lstm(outputs, batch_size, nlayer, nhidden, nfeat, nproj, "fuse_lstm")

        if(lstm_type == constants.LSTM_TYPE.):
            return self.my_native_lstm(outputs, batch_size, nlayer, nhidden, nfeat, nproj, "native_lstm")

    def __my_cudnn_lstm(self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope, batch_norm, is_training = True):
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

    def __my_fuse_block_lstm(self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope):
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

    def ___my_native_lstm(self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope):
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

