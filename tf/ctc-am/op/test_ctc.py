import tensorflow as tf

ctc_sinbad = tf.load_op_library('./ctc_sin_bad.so')

with tf.Session(''):
    print(dir(ctc_sinbad))
