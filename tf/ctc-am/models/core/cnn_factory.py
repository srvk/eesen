import tf

class CnnFactory:
    def get_cnn(self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope, batch_norm, is_training = True):

        return self.__my_cnn(outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope, batch_norm, is_training)

    def __my_cnn (self, outputs, batch_size, nlayer, nhidden, nfeat, nproj, scope, batch_norm, is_training = True):

        if(nlayer > 0 ):
            with tf.variable_scope("input"):
                # outputs = tf.transpose(self.feats, (0, 1, 2), name = "feat_transpose") # (B,T,F) -> (B,F,T)
                outputs = tf.expand_dims(self.feats, -1)  # (B,T,F) -> (B,T,F,1)
                # self.shape_0 = tf.shape(outputs)
                ################################CONV1##################################################
            with tf.name_scope('conv1_0') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 1, 32], dtype=tf.float32,
                                                         stddev=1e-1))

                outputs = tf.nn.conv2d(outputs, kernel, [1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True,
                                       name="conv0")  # (B,T,F,1) -> (B,T,F,32)

                biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                                     trainable=True, name='biases_conv0')

                outputs = tf.nn.bias_add(outputs, biases)

                outputs = tf.nn.relu(outputs, name="relu_conv0")
                # self.shape_1 = tf.shape(outputs)

            with tf.name_scope('conv1_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 10, 32, 32], dtype=tf.float32,
                                                         stddev=1e-1))

                outputs = tf.nn.conv2d(outputs, kernel, [1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True,
                                       name="conv1")  # (B,T,F,1) -> (B,T,F,32)

                biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                                     trainable=True, name='biases_conv1')

                outputs = tf.nn.bias_add(outputs, biases)

                outputs = tf.nn.relu(outputs, name="relu_conv1")
                # self.shape_1 = tf.shape(outputs)
                # outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, decay=0.9, is_training=self.is_training, updates_collections=None)

                outputs = tf.nn.max_pool(outputs, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME',
                                         name='pool1')  # (B,T,F,32)
                # self.shape_2 = tf.shape(outputs)

                ################################CONV1##################################################
            with tf.name_scope('conv1_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([1, 40, 32, 64], dtype=tf.float32,
                                                         stddev=1e-1))

                outputs = tf.nn.conv2d(outputs, kernel, [1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True,
                                       name="conv2")  # (B,T,F/2+1,32) -> (B,T,F/2+1,32)

                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=True, name='biases_conv2')
                # bias=tf.Variable(tf.random_normal([64]), name="bias_conv2")

                outputs = tf.nn.bias_add(outputs, biases)

                outputs = tf.nn.relu(outputs, name="relu_conv2")
                # self.shape_3 = tf.shape(outputs)
                # pool1
                outputs = tf.nn.max_pool(outputs, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME',
                                         name='pool2')  # (B,T,F,32) -> (B,T,(F/2+1)/2+1,32)
                # self.shape_4 = tf.shape(outputs)

            with tf.name_scope('conv1_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([20, 1, 64, 64], dtype=tf.float32,
                                                         stddev=1e-1))

                outputs = tf.nn.conv2d(outputs, kernel, [1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True,
                                       name="conv3")  # (B,T,F/2+1,32) -> (B,T,F/2+1,32)

                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=True, name='biases_conv3')
                # bias=tf.Variable(tf.random_normal([64]), name="bias_conv2")

                outputs = tf.nn.bias_add(outputs, biases)

                outputs = tf.nn.relu(outputs, name="relu_conv3")
                # tf.Print(tf.shape(outputs))
                # outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, decay=0.9, is_training=self.is_training, updates_collections=None)

                # outputs = tf.transpose(outputs, (0,1,3), name = "feat_transpose")
            outputs = tf.reshape(outputs,
                                 [tf.shape(outputs)[0], tf.shape(outputs)[1], (int((int(nfeat / 2) + 1) / 2) + 1) * 64])
            outputs = tf.transpose(outputs, (1, 0, 2), name="feat_transpose")
            # self.shape_out = tf.shape(outputs)
            # outputs = tf.reshape(outputs,[tf.shape(outputs)[0],tf.shape(outputs)[1],nfeat*64])
            # outputs=tf.squeeze(outputs, [3])

            # outputs = tf.transpose(outputs, (2,0,1), name = "feat_transpose")

            # outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, decay=0.9, is_training=self.is_training, updates_collections=None)

            outputs = tf.contrib.layers.fully_connected(
                activation_fn=tf.nn.relu, inputs=outputs, num_outputs=600,
                scope="input_fc0", biases_initializer=tf.contrib.layers.xavier_initializer())
            # self.shape_0 = tf.shape(outputs)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            # self.shape_1 = tf.shape(outputs)
            ################################END CONV##################################################
            ################################FC PART##################################################

            outputs = tf.contrib.layers.fully_connected(
                activation_fn=tf.nn.relu, inputs=outputs, num_outputs=300,
                scope="input_fc", biases_initializer=tf.contrib.layers.xavier_initializer())

            # outputs = tf.nn.dropout(outputs,self.drop_out)


            # outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, decay=0.9, is_training=self.is_training, updates_collections=None)


            outputs = tf.contrib.layers.fully_connected(
                activation_fn=tf.nn.relu, inputs=outputs, num_outputs=300,
                scope="input_fc_2", biases_initializer=tf.contrib.layers.xavier_initializer())

            outputs = tf.nn.dropout(outputs, self.keep_prob)

            # outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, decay=0.9, is_training=self.is_training, updates_collections=None)
        return outputs
