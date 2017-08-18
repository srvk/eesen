import tensorflow as tf
from tensorflow.python.framework import tensor_shape

# For safety, always assume video must be in [0,1] range

def get_sequence_length(input):
    '''
    Computes the number of valid frames in a video sequence
    From: https://danijar.com/variable-sequence-lengths-in-tensorflow/

    Input:
    sequence - 3D tensor of shape (batch, frames, frame_data)
    If greater than 3D, flattens to a 3D tensor.

    Output:
    length - 1D tensor of valid frame length for each sample
    '''
    fan_in = tf.to_int32(tf.reduce_prod(tf.shape(input)[2:]))
    input_flat = tf.reshape(input, [-1, tf.shape(input)[1], fan_in])

    # used is a 2D tensor mask that has 1 for valid frames and
    # 0 for padded frames.
    used = tf.sign(tf.reduce_max(tf.abs(input_flat), reduction_indices=2))

    # Sum up over the frame dimension.  Now its a 1D tensor
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)

    return length


def subtract_video_channel_mean(video):
    '''
    Expects a 4d tensor
    (frames, height, width, channels)
    Subtracts average channel.
    No std_dev normalization
    '''
    video = tf.to_float(tf.convert_to_tensor(video, name='video'))
    channel_means = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(video,
            axis=0, keep_dims=True),
            axis=1, keep_dims=True),
            axis=2, keep_dims=True)
    return video - channel_means

def adjust_brightness(video, delta):
    """
    Adjust video brightness
    Returns:
      The brightness-adjusted video.
    """
    return tf.map_fn(lambda frame: tf.image.adjust_brightness(frame, delta), video)

def adjust_contrast(video, contrast_factor):
    """
    Adjust video contrast
    Returns:
      The contrast-adjusted video.
    """
    return tf.map_fn(lambda frame: tf.image.adjust_contrast(frame, contrast_factor), video)

def adjust_hue(video, delta):
    """
    Adjust video hue
    Returns:
      The hue-adjusted video.
    """
    return tf.map_fn(lambda frame: tf.image.adjust_hue(frame, delta), video)

def adjust_saturation(video, saturation_factor):
    """
    Adjust video saturation
    Returns:
      The saturation-adjusted video.
    """
    return tf.map_fn(lambda frame: tf.image.adjust_saturation(frame, saturation_factor), video)

def random_saturation(video, lower, upper, seed=None):
    """Adjust the saturation of an RGB video by a random factor.
    Args:
      image: RGB video.
      lower: float.  Lower bound for the random saturation factor.
      upper: float.  Upper bound for the random saturation factor.
      seed: An operation-specific seed. It will be used in conjunction
        with the graph-level seed to determine the real seeds that will be
        used in this operation. Please see the documentation of
        set_random_seed for its interaction with the graph-level random seed.
    Returns:
      Adjusted image(s), same shape and DType as `image`.
    Raises:
      ValueError: if `upper <= lower` or if `lower < 0`.
    """
    if upper <= lower:
        raise ValueError('upper must be > lower.')
    if lower < 0:
        raise ValueError('lower must be non-negative.')
    # Pick a float in [lower, upper]
    saturation_factor = tf.random_uniform([], lower, upper, seed=seed)
    return adjust_saturation(video, saturation_factor)

def random_hue(video, max_delta, seed=None):
    """Adjust the hue of an RGB video by a random factor.
    Args:
      image: RGB video.
      max_delta: float.  Maximum value for the random delta.
      seed: An operation-specific seed. It will be used in conjunction
        with the graph-level seed to determine the real seeds that will be
        used in this operation. Please see the documentation of
        set_random_seed for its interaction with the graph-level random seed.
    Returns:
      4-D float tensor of shape `[frames, height, width, channels]`.
    Raises:
      ValueError: if `max_delta` is invalid.
    """
    if max_delta > 0.5:
        raise ValueError('max_delta must be <= 0.5.')
    if max_delta < 0:
      raise ValueError('max_delta must be non-negative.')
    delta = tf.random_uniform([], -max_delta, max_delta, seed=seed)
    return adjust_hue(video, delta)

def random_brightness(video, max_delta, seed=None):
    """Adjust the brightness of a video by a random factor.
    Args:
      video: A video.
      max_delta: float, must be non-negative.
      seed: A Python integer. Used to create a random seed. See
        @{tf.set_random_seed}
        for behavior.
    Returns:
      The brightness-adjusted video.
    Raises:
      ValueError: if `max_delta` is negative.
    """
    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')
    delta = tf.random_uniform([], -max_delta, max_delta, seed=seed)
    return adjust_brightness(video, delta)

def random_contrast(video, lower, upper, seed=None):
    """Adjust the contrast of a video by a random factor.
    Args:
      video: An video tensor with 3 or more dimensions.
      lower: float.  Lower bound for the random contrast factor.
      upper: float.  Upper bound for the random contrast factor.
      seed: A Python integer. Used to create a random seed. See
        @{tf.set_random_seed}
        for behavior.
    Returns:
      The contrast-adjusted tensor.
    Raises:
      ValueError: if `upper <= lower` or if `lower < 0`.
    """
    if upper <= lower:
        raise ValueError('upper must be > lower.')
    if lower < 0:
        raise ValueError('lower must be non-negative.')
    # Generate a float in [lower, upper]
    contrast_factor = tf.random_uniform([], lower, upper, seed=seed)
    return adjust_contrast(video, contrast_factor)

def random_color_augmentation(video):
    '''
    Randomly augments the video color attributes in random operation order
    (brightness, saturation, hue, contrast)

    To replace random_color_augmentation()

    Expects input to be in [0,1] Range
    '''
    BRIGHTNESS_MAX_DELTA = 28.0/255.0
    SATURATION_LOWER = 0.5
    SATURATION_UPPER = 1.5
    HUE_MAX_DELTA = 0.15
    CONTRAST_LOWER = 0.5
    CONTRAST_UPPER = 1.5

    # Ideally the order of these operations would be random
    # for every image, but I don't know how to do that
    order = tf.convert_to_tensor([1,2,3,4])
    order = tf.random_shuffle(order,seed=None)

    def body(op_idx, video):
        video = tf.cond(tf.equal(order[op_idx], 1),
                lambda: random_brightness(video,
                            max_delta=BRIGHTNESS_MAX_DELTA),
                lambda: video)
        video = tf.cond(tf.equal(order[op_idx], 2),
                lambda:  random_saturation(video,
                            lower=SATURATION_LOWER, upper=SATURATION_UPPER),
                lambda: video)
        video = tf.cond(tf.equal(order[op_idx], 3),
                lambda: random_hue(video,
                            max_delta=HUE_MAX_DELTA),
                lambda: video)
        video = tf.cond(tf.equal(order[op_idx], 4),
                lambda: random_contrast(video,
                            lower=CONTRAST_LOWER, upper=CONTRAST_UPPER),
                lambda: video)
        return op_idx + 1, video

    def cond(op_idx, video):
        return op_idx < 4

    # Loop through the augmentation ops
    i = tf.constant(0) # counter for augment operations
    _, video = tf.while_loop(cond, body,
            loop_vars=[i, video])

    # Augmentations may produce something outside the [0.0,1.0] range
    video = tf.clip_by_value(video, 0.0, 1.0)
    return video

def random_flip_left_right(video, seed=None):
    '''
    Randomly flips an entire video of form (frames, height, width, channels)
    either left or right
    '''
    video = tf.convert_to_tensor(video, name='video')
    uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
    mirror_cond = tf.less(uniform_random, .5)
    result = tf.cond(mirror_cond,
                     lambda: tf.reverse(video, [2]),
                     lambda: video)
    return fix_video_flip_shape(video, result)

def random_flip_up_down(video, seed=None):
    '''
    Randomly flips an entire video of form (frames, height, width, channels)
    either left or right
    '''
    video = tf.convert_to_tensor(video, name='video')
    uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
    mirror_cond = tf.less(uniform_random, .5)
    result = tf.cond(mirror_cond,
                     lambda: tf.reverse(video, [1]),
                     lambda: video)
    return fix_video_flip_shape(video, result)


def fix_video_flip_shape(video, result):
    """Set the shape to 3 dimensional if we don't know anything else.
    Args:
      image: original image size
      result: flipped or transformed image
    Returns:
      An image whose shape is at least None,None,None.
    """

    video_shape = video.get_shape()
    if video_shape == tensor_shape.unknown_shape():
        result.set_shape([None, None, None, None])
    else:
        result.set_shape(video_shape)
    return result