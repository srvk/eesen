import tensorflow as tf

from utils.imageutils.summary_utils import variable_summaries

'''
Uses https://github.com/NickShahML/tensorflow_with_latest_papers.git
'''

def activate(x, type):
    '''
    Activation Function Selector
    '''
    type = type.lower()
    return {'relu': tf.nn.relu(x, name='relu'),
            'leaky_relu': leaky_relu(x),
            'linear': x,
            'none':x,
            }.get(type, x)

def leaky_relu(x, alpha=0.18):
    '''
    Applys leaky relu activation.
    '''
    return tf.maximum(alpha*x,x)

class SharedGRUCell(tf.contrib.rnn.GRUCell):
    '''
    GRUCell that can weight share
    Source: Ishamael, 2016
    http://stackoverflow.com/a/39134388
    '''
    def __init__(self, num_units, input_size=None, activation=tf.nn.tanh):
        tf.contrib.rnn.GRUCell.__init__(self, num_units, input_size, activation)
        self.my_scope = None

    def __call__(self, a, b):
        if self.my_scope == None:
            self.my_scope = tf.get_variable_scope()
        else:
            self.my_scope.reuse_variables()
        return tf.contrib.rnn.GRUCell.__call__(self, a, b, self.my_scope)

def channel_dropout(name, input, keep_prob=0.5, seed=None, is_training=False):
    '''
    Randomly Drops entire channels
    '''
    noise_shape = [tf.shape(input)[0], tf.shape(input)[1], 1, 1, tf.shape(input)[4]]

    keep_prob = tf.cond(tf.equal(is_training, True),
                lambda: tf.constant(keep_prob),
                lambda: tf.constant(1.0))

    return tf.nn.dropout(input, keep_prob=keep_prob, noise_shape=noise_shape,
            seed=seed, name=name)

def batch_norm(x, phase, name='bn'):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps; or 5D BTHWD
        phase:       boolean tf.Variabe, true indicates training phase
        name:        string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        ndims = len(x_shape)
        n_out = x_shape[-1]
        beta = tf.get_variable('beta', shape=[n_out], dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable('gamma', shape=[n_out], dtype=tf.float32,
                initializer=tf.constant_initializer(1.0))

	if ndims==4:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        elif ndims==5:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2,3], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def multi_layer_bi_rnn(name, input, units, n_layers,
        sequence_length=None,
        keep_prob=1,
        dtype=tf.float32):
    '''
    Creates a multilayer birdirectional dynamic rnn with GRU cells

    If sequence_length is none, the entire batch length is considered
    the sequence_length (good for batch size 1)

    Otherwise sequence length should be a vector of size [batch_size]

    Input is a Tensor of shape: (Batch size x time_steps x features)
    '''
    with tf.variable_scope(name):
        fw_cell = tf.contrib.rnn.GRUCell(num_units=units)
        bw_cell = tf.contrib.rnn.GRUCell(num_units=units)

        fw_multicell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.GRUCell(num_units=units),
                    input_keep_prob=keep_prob) for _ in range(n_layers)])
        bw_multicell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.GRUCell(num_units=units),
                    input_keep_prob=keep_prob) for _ in range(n_layers)])

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_multicell,
                cell_bw=bw_multicell,
                dtype=dtype,
                sequence_length=sequence_length,
                inputs=input)
    return tf.concat(outputs,2), states

def bi_rnn(name, input, units,
        sequence_length,
        dtype=tf.float32):
    '''
    Creates a bidirectional dynamic rnn with sharable GRU cells

    Input is a Tensor of shape: (Batch size x time_steps x features)
    '''
    def bi_rnn_helper(input, units, sequence_length, dtype):
        with tf.variable_scope('forward_cell'):
            forward_cell = SharedGRUCell(
                    num_units=units,
                    activation=tf.nn.tanh)
        with tf.variable_scope('backward_cell'):
            backward_cell = SharedGRUCell(
                    num_units=units,
                    activation=tf.nn.tanh)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward_cell,
                cell_bw=backward_cell,
                dtype=dtype,
                sequence_length=sequence_length,
                inputs=input)
        return outputs, states

    with tf.variable_scope(name) as scope:
        try:
            outputs, states = bi_rnn_helper(input, units, sequence_length, dtype)
        except ValueError:
            scope.reuse_variables()
            outputs, states = bi_rnn_helper(input, units, sequence_length, dtype)
    return tf.concat(outputs, 2), states


def conv2d(name, input, shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        strides=[1,1,1,1],
        padding='SAME',
        activation='relu'):
    '''
    2D convolution Layer with smart variable reuse.
    '''

    def conv2d_helper(input, shape, dtype, initializer, strides, padding, activation):
        kernel = tf.get_variable('weights', shape=shape, dtype=dtype,
                initializer=initializer)
        biases = tf.get_variable('biases', shape=shape[-1], dtype=dtype,
                initializer=tf.constant_initializer(0.0))

        variable_summaries(kernel, 'kernels')
        variable_summaries(biases, 'biases')

        conv = tf.nn.conv2d(input, kernel, strides, padding=padding)
        biased_conv = tf.nn.bias_add(conv, biases)
        output = activate(biased_conv, type=activation)
        return output, kernel, biases

    with tf.variable_scope(name) as scope:
        try:
            return conv2d_helper(input, shape, dtype, initializer,
                    strides, padding, activation)
        except ValueError:
            scope.reuse_variables()
            return conv2d_helper(input, shape, dtype, initializer,
                    strides, padding, activation)

def conv3d(name, input, shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        strides=[1,1,1,1,1],
        padding='SAME',
        activation='relu'):
    '''
    3D convolution Layer with smart variable reuse.
    Input: Tensor (batch, frames, in_height, in_width, in_channels)
    Shape: 1D Tensor to describe kernel shape
           [filter_depth, filter_height, filter_width, in_channels, out_channels]
    Stride: 1D Tensor
           [1, temporal, spatial_height, spatial_width, 1]
    '''

    def conv3d_helper(input, shape, dtype, initializer, strides, padding, activation):
        kernel = tf.get_variable('weights', shape=shape, dtype=dtype,
                initializer=initializer)
        biases = tf.get_variable('biases', shape=shape[-1], dtype=dtype,
                initializer=tf.constant_initializer(0.0))

        variable_summaries(kernel, 'kernels')
        variable_summaries(biases, 'biases')

        # [filter_depth, filter_height, filter_width, in_channels, out_channels]
        conv = tf.nn.conv3d(input, kernel, strides, padding=padding)
        biased_conv = tf.nn.bias_add(conv, biases)
        output = activate(biased_conv, type=activation)
        return output, kernel, biases

    with tf.variable_scope(name) as scope:
        try:
            return conv3d_helper(input, shape, dtype, initializer,
                    strides, padding, activation)
        except ValueError:
            scope.reuse_variables()
            return conv3d_helper(input, shape, dtype, initializer,
                    strides, padding, activation)

def fc_broadcast(name, input, units,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        activation='relu'):
    '''
    Broadcasts fully connected layer across dims 2+
    '''
    input_shape = input.get_shape().as_list()

    # Flatten
    input_flat = tf.reshape(input, [-1, input_shape[-1]])

    # Perform Inner Product
    result, weights, biases = fc(name, input_flat, units, dtype, initializer, activation)

    # Reshape back into original shape
    output = tf.reshape(result, [input_shape[0], -1, units])
    return output, weights, biases


def fc(name, input, units,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        activation='relu'):
    '''
    Input must be 2D (batch, features)
    Applys a fully connected layer
    '''
    def fc_helper(input, units, initializer, activation):
        # This must be a non tensor value present at graph creation
        # Used to determine 'Weights' first dimension
        fan_in = input.get_shape()[1]

        weights = tf.get_variable('weights', shape=(fan_in, units), dtype=dtype, initializer=initializer)
        biases = tf.get_variable('biases', shape=(units), dtype=dtype,
                initializer=tf.constant_initializer(0.0))

        variable_summaries(weights, 'weights')
        variable_summaries(biases, 'biases')

        pre_activate = tf.nn.bias_add(tf.matmul(input, weights), biases)
        output = activate(pre_activate, type=activation)
        return output, weights, biases

    with tf.variable_scope(name) as scope:
        try:
            return fc_helper(input, units, initializer, activation)
        except ValueError:
            scope.reuse_variables()
            return fc_helper(input, units, initializer, activation)

