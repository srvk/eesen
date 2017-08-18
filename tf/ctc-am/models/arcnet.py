import tensorflow as tf
import numpy as np

from core.lipnet_layers import layers
import summary_utils
import video_utils

import pdb

FLAGS = tf.app.flags.FLAGS

class ArcNet:
    '''
    LipNet Model
    '''

    #def __init__(self, config):
    def __init__(self, x, y, phase):
        '''
        Sets up network enough to do a forward pass.
        x - input
        y - output
        phase - boolean where True is train, False is validate
        '''
        self.N_RNN_UNITS = 512
        self.N_RNN_LAYERS = 3
        self.FRAME_SHAPE = [None, 50, 100, 3]
        self.CONV_KEEP_PROB = 0.5
        self.RECURRENT_KEEP_PROB = 0.9
        self.CONV_ACTIVATION = 'leaky_relu'

        ########
        # Misc #
        ########
        self.global_step = tf.get_variable('global_step', dtype=tf.int32, trainable=False,
                initializer=0)
        self.learning_rate = FLAGS.initial_lr
        self.phase = phase

        ####################
        # I/O placeholders #
        ####################
        # Expects videos of dimension 100 wide, 50 tall, 3 color channels
        self.x = x
        self.x.set_shape([None]+self.FRAME_SHAPE)
        self.y = tf.to_int32(y)

        ###############
        # Main Layers #
        ###############
        # Used for future recurrent calculations
        self.sequence_length = video_utils.get_sequence_length(self.x)
        with tf.variable_scope('video_branch'):
            self._video_branch()
        with tf.variable_scope('recurrent_branch'):
            self._recurrent_layers()

        ######################
        # Define Collections #
        ######################
        self.video_branch_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                "video_branch")
        self.recurrent_branch_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                "recurrent_branch")

    def get_global_step(self):
        return self.global_step

    def inference(self):
        '''
        Returns the output of a forward pass of the network (Tensor).
        '''
        #ToDo clean up this debugging info
        dense_decode = tf.sparse_tensor_to_dense(self.decoded[0], default_value=FLAGS.num_classes)
        return dense_decode, self.log_probabilities

    def loss(self):
        '''
        Returns the loss output (Tensor).
        '''
        # CTC_Loss internally softmaxes
        # The last label must be the blank symbol
        # Preprocess_collapse_repeated removes repeated consecutive labels
        # See https://www.tensorflow.org/versions/r0.10/api_docs/python/nn/conectionist_temporal_classification__ctc_
        #
        # Some documentation:
        # inputs: 3-D float Tensor sized [max_time x batch_size x num_classes]. The logits.
        # labels: An int32 SparseTensor. labels.indices[i, :] == [b, t] means labels.values[i]
        #         stores the id for (batch b, time t). See core/ops/ctc_ops.cc for more details.
        self.loss = tf.nn.ctc_loss(
                labels=self.y,
                inputs=self.ctc_ready_fc_1,
                sequence_length=self.sequence_length)

        self.cost = tf.reduce_mean(self.loss)

        dense_y = tf.sparse_tensor_to_dense(self.y, default_value=FLAGS.num_classes)
        return self.cost, dense_y, self.sequence_length

    def optimize(self):
        '''
        Returns the Training Operation (op).
        '''
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Apply Gradient Clipping
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5.0)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                global_step=self.global_step)
        return self.train_op

    def evaluate(self):
        '''
        Returns the count of correct classifications (Tensor).
        '''
        self.accuracy = 1 - tf.reduce_mean(tf.edit_distance(
                tf.to_int32(self.decoded[0]), self.y))
        return self.accuracy

    #####################
    # Private Functions #
    #####################
    def _video_branch(self):
        '''
        Based off of LipNet architecture
        '''
        # Preprocess input videos
        mean, _ = tf.nn.moments(self.x, axes=[1], keep_dims=True)
        _, variance = tf.nn.moments(self.x, axes=[1,2,3], keep_dims=True)
        self.normalized_videos = (self.x - mean) / tf.sqrt(variance)
        tf.summary.image('normalized', self.normalized_videos[0,...], max_outputs=12)

        # assume input is 5D and has shape:
        # (batch, n_frames, video_height, video_width, video_channels)
        # filters=[depth, height, width, in_channels, out_channels],
        self.v_conv_1, weights, biases = layers.conv3d(
                input=self.normalized_videos,
                shape=(3,5,5,3,32),
                strides=[1,1,2,2,1],
                padding='SAME',
                activation=self.CONV_ACTIVATION,
                name='v_conv_1')

        # Add a TensorBoard image summary for conv1 kernels
        #summary_utils.conv_visual_summary(weights, name='v_conv_1')

        # Even though it says 3d, we're using it only spatially
        self.v_pool_1 = tf.nn.max_pool3d(
                input=self.v_conv_1,
                ksize=[1,1,2,2,1],
                strides=[1,1,2,2,1],
                padding='SAME',
                name='v_pool_1'
                )

        self.v_bn_1 = layers.batch_norm(self.v_pool_1,
                        phase=self.phase,
                        name='bn_1')

        self.v_dropout_1 = layers.channel_dropout(
                input=self.v_bn_1,
                keep_prob=self.CONV_KEEP_PROB,
                is_training=self.phase,
                name='v_dropout_1',
                )

        self.v_conv_2, weights, biases = layers.conv3d(
                input=self.v_dropout_1,
                shape=(3,5,5,32,64),
                strides=[1,1,1,1,1],
                padding='SAME',
                activation=self.CONV_ACTIVATION,
                name='v_conv_2')

        self.v_pool_2 = tf.nn.max_pool3d(
                input=self.v_conv_2,
                ksize=[1,1,2,2,1],
                strides=[1,1,2,2,1],
                padding='SAME',
                name='v_pool_2'
                )

        self.v_bn_2 = layers.batch_norm(self.v_pool_2,
                        phase=self.phase,
                        name='bn_2')

        self.v_dropout_2 = layers.channel_dropout(
                input=self.v_bn_2,
                keep_prob=self.CONV_KEEP_PROB,
                is_training=self.phase,
                name='v_dropout_2',
                )

        self.v_conv_3, weights, biases = layers.conv3d(
                input=self.v_dropout_2,
                shape=(3,3,3,64,96),
                strides=[1,1,1,1,1],
                padding='SAME',
                activation=self.CONV_ACTIVATION,
                name='v_conv_3')

        self.v_pool_3 = tf.nn.max_pool3d(
                input=self.v_conv_3,
                ksize=[1,1,2,2,1],
                strides=[1,1,2,2,1],
                padding='SAME',
                name='v_pool_3'
                )

        self.v_bn_3 = layers.batch_norm(self.v_pool_3,
                        phase=self.phase,
                        name='bn_3')

        self.v_dropout_3 = layers.channel_dropout(
                input=self.v_bn_3,
                keep_prob=self.CONV_KEEP_PROB,
                is_training=self.phase,
                name='v_dropout_3',
                )

        # Flatten the video branch output into a feature vector
        # The "Output" layer of the video branch.
        # Output shape: (batch, frames, features)
        fan_in = int(np.prod(self.v_dropout_3.get_shape()[2:]))
        shape = [self.v_dropout_3.get_shape().as_list()[0], -1, fan_in]

        self.v_flat_1 = tf.reshape(self.v_dropout_3, shape)

        tf.summary.histogram('video_feature', self.v_flat_1)

    def _recurrent_layers(self):
        self.fc_0, weights, biases = layers.fc_broadcast(
                input=self.v_flat_1,
                units=1024,
                activation='leaky_relu',
                name='fc_0')

        self.bi_gru_outs_2, self.bi_gru_states_2 = layers.multi_layer_bi_rnn(
                input=self.fc_0,
                n_layers=self.N_RNN_LAYERS,
                units=self.N_RNN_UNITS,
                keep_prob=self.RECURRENT_KEEP_PROB,
                sequence_length=self.sequence_length,
                phase=self.phase,
                name='bi_recurrent_2'
                )

        self.fc_1, weights, biases = layers.fc_broadcast(
                input=self.bi_gru_outs_2,
                units=FLAGS.num_classes+1,
                activation='linear',
                name='fc_1')

        # Transpose the last FC layer to be in form:
        # (frames, batch, n_classes)
        # Its what CTC functions expect
        self.ctc_ready_fc_1 = tf.transpose(self.fc_1, (1, 0, 2))

        # Generate Predictions
        self.decoded, self.log_probabilities = tf.nn.ctc_greedy_decoder(
                inputs=self.ctc_ready_fc_1,
                sequence_length=self.sequence_length,
                merge_repeated=True
                )
