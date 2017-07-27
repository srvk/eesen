from tf.tf_train import Train

class Eesen():

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

        train_impl=Train(config)
        train_impl.train(data)


    def test(self, data, config):

        train_impl(data, config)



