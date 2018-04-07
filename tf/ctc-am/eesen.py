from tf.tf_train import Train
from tf.tf_test import Test
import tensorflow as tf
import os, sys, time, subprocess


print(80 * "-")
print(sys.version)
print("now:", time.strftime("%a %Y-%m-%d %H:%M:%S"))
try:
    print("tf:", tf.__version__)
    print("env:", sys.executable)
    print("cwd:", os.getcwd())
    f=os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    print("library:", f)
    print("git:", os.popen("git --git-dir="+os.path.join(f, ".git")+" --work-tree="+f+" describe --dirty --tags --all --always").read().strip())
except:
    print(os.path.basename(__file__)+": error getting version information for logging")
print(80 * "-")


class Eesen(object):

    def train(self, data, config):
        train = Train(config)
        train.train_impl(data)

    def test(self, data, config):
        test = Test()
        test.test_impl(data, config)
