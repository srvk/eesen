from tf.tf_train import Train
from tf.tf_test import Test
import tensorflow as tf
import os, sys

print(80 * "-")
print("Eesen TF library:", os.path.realpath(__file__))
print("cwd:", os.getcwd(), "version:")
try:
    print(sys.version)
    print(tf.__version__)
except:
    print("tf.py: could not get version information for logging")
print(80 * "-")

class Eesen(object):

    def train(self, data, config):

        train = Train(config)
        train.train_impl(data)


    def test(self, data, config):
        test = Test()
        test.test_impl(data, config)






