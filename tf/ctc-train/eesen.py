from tf.tf_train import Train
import tensorflow as tf
import os, sys

class Eesen(object):

    def train(self, data, config):

        print(80 * "-")
        print("Eesen TF library:", os.path.realpath(__file__))
        print("cwd:", os.getcwd(), "version:")
        try:
            print(sys.version)
            print(tf.__version__)
        except:
            print("tf.py: could not get version information for logging")
        print(80 * "-")

        train=Train(config)
        train.train_impl(data)


    def test(self, data, config):
        print("not implemented yet")






