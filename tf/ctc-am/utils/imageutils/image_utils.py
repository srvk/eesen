import tensorflow as tf

# For safety, always assume image must be in [0,1] range

def crop_and_resize_proper(image, out_shape):
    '''
    Rescales image by:
    1) 0-Padding Image to a square
    2) Cropping center to a square using the average dim-length
    3) Rescale to out_shape
    '''
    im_shape = tf.shape(image)[0:2]
    crop_len = tf.to_int32(tf.reduce_mean(im_shape))
    im_square = tf.image.resize_image_with_crop_or_pad(image, crop_len, crop_len)
    result = tf.image.resize_images(im_square, out_shape[0:2])
    return result

def random_crop_and_resize_proper(image, out_shape):
    '''
    Randomly crops a square thats has sidelength:
        SIDELENGTH_PERCENTAGE*${shortest_side}.
    Then resizes to out_shape
    '''
    im_shape = tf.shape(image)[0:2]
    n_channels = tf.shape(image)[2]
    crop_len = tf.to_int32(tf.reduce_mean(im_shape))
    im_square = tf.image.resize_image_with_crop_or_pad(image, crop_len, crop_len)
    rand_crop = tf.random_crop(im_square, size=[crop_len, crop_len, n_channels])
    resized = tf.image.resize_images(rand_crop, out_shape[0:2])
    return resized

def random_color_augmentation(image):
    '''
    Randomly augments the image color attributes
    (brightness, saturation, hue, contrast)

    Expects input to be in [0,1] Range
    '''
    BRIGHTNESS_MAX_DELTA = 32.0/255.0
    SATURATION_LOWER = 0.5
    SATURATION_UPPER = 1.5
    HUE_MAX_DELTA = 0.2
    CONTRAST_LOWER = 0.5
    CONTRAST_UPPER = 1.5

    # Ideally the order of these operations would be random
    # for every image, but I don't know how to do that
    image =  tf.image.random_brightness(image,
            max_delta=BRIGHTNESS_MAX_DELTA)
    image =  tf.image.random_saturation(image,
            lower=SATURATION_LOWER, upper=SATURATION_UPPER)
    image = tf.image.random_hue(image,
            max_delta=HUE_MAX_DELTA)
    image = tf.image.random_contrast(image,
            lower=CONTRAST_LOWER, upper=CONTRAST_UPPER)

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image
