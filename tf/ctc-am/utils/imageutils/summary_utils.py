import tensorflow as tf
from math import sqrt

def variable_summaries(var, name=''):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).

    Example inputs: Layer weights, biases, kernel, etc
    """
    with tf.name_scope('summaries_' + name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def put_kernels_on_grid(kernel, pad = 1):

    '''
    Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:   tensor of shape [Y, X, NumChannels, NumKernels]
      pad:      number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].

    Source:
    https://gist.github.com/kukuruza/03731dc494603ceab0c5
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant([[pad,pad],[pad, pad],[0,0],[0,0]]),
            mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7

def conv_visual_summary(kernel, name=''):
    '''
    Adds a visualization of convolution filters to tensorboard.
    Really only makes sense on the first layer
    '''
    if name=='':
        summary_name = 'visualization'
    else:
        summary_name = name + '/visualization'

    kernel_shape = kernel.get_shape().as_list()
    ndims = len(kernel_shape)
    if ndims==4:
        n_images=1
        grid = put_kernels_on_grid(kernel)
    elif ndims==5:
        n_images=kernel_shape[0]
        gridify = lambda x: put_kernels_on_grid(x)
        grid = tf.map_fn(gridify, kernel, back_prop=False)
        grid = tf.squeeze(grid)

    tf.summary.image(summary_name, grid, max_outputs=n_images)

