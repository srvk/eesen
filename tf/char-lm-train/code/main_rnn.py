import argparse
import json
import pdb
from train_rnn_lm import *
from prepare_traindata import *


print("main_rnn.py - version information follows:")
try:
    print(sys.version)
    print(tf.__version__)
except:
    print("tf.py: could not get version information for logging")


def mainParser():
    parser = argparse.ArgumentParser(description='Train TF-RNN_LM')
    parser.add_argument('--batch_size', default = 16, type=int, help='batch size')
    parser.add_argument('--train_file', default = "./data/turkish_train_text", help = "train data loc")
    parser.add_argument('--dev_file', default = "./data/turkish_dev_text", help = "dev data loc")
    parser.add_argument('--units_file', default = "./data/units_char_system.txt", help = "units data loc")
    parser.add_argument('--lexicon_file', default = "./data/lexicon_char_system.txt", help = "lexicon data loc")
    parser.add_argument('--nepoch', default = 20, type=int, help='#epoch')
    # parser.add_argument('--l2', default = 0.0, type=float, help='l2 normalization')
    # parser.add_argument('--clip', default = 0.1, type=float, help='gradient clipping')
    parser.add_argument('--num_layers', default = 1, type=int, help='#layer')
    parser.add_argument('--hidden_size', default = 1000, type=int, help='dimension of hidden units in single direction')
    parser.add_argument('--embed_size', default = 64, type=int, help='embedding size')
    parser.add_argument('--start_idx', default = 1, type=int, help='ignore leading fields in transcriptions')
    parser.add_argument('--drop_emb', default = 1.0, type=float, help='embedding (1.0-dropout) probability')
    # parser.add_argument('--gpu_id', default = 1, type=int, help='gpu to use')

    return parser

def createConfig(args):
    config = {
        'embed_size' : args.embed_size ,
        'drop_emb' : args.drop_emb ,
        'hidden_size' : args.hidden_size ,
        'num_layers' : args.num_layers ,
        'nepoch' : args.nepoch ,
        'lexicon_file' : args.lexicon_file ,
        'units_file' : args.units_file ,
        'dev_file' : args.dev_file ,
        'train_file' : args.train_file ,
        'batch_size' : args.batch_size
    }
    config['random_seed']= 15213
    model_name = 'train_l' + str(config['num_layers'])+ '_e' + str(config['embed_size']) + '_h' + str(config['hidden_size']) + '_b' + str(config['batch_size']) + '_d' + str(config['drop_emb'])

    config['exp_path'] = './exp/' + model_name + '/'
    if not os.path.exists(config["exp_path"]):
        os.makedirs(config["exp_path"])

    with open( config['exp_path']+ 'config', 'w') as fp:
        fp.write(json.dumps(config, indent=4, sort_keys=True))
        fp.close()
    print config
    return config

def main():
    parser = mainParser()
    args = parser.parse_args()
    config = createConfig(args)
    data = dict()
    data['train'], data['test'], config['nwords'], data['eos'] = prep_data(config, startidx=args.start_idx)
    train(data,config)
    # pdb.set_trace()


if __name__ == "__main__":
    main()
