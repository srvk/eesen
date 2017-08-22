import argparse
import json
import pdb
from reader.labels_reader.labels_reader import LabelsReader
from reader.feats_reader.feats_reader import FeatsReaderKaldi



from train_rnn_lm import *

def mainParser():
    parser = argparse.ArgumentParser(description='Train TF-RNN_LM')
    parser.add_argument('--batch_size', default = 16, type=int, help='batch size')

    parser.add_argument('--train_file', default = "./data/turkish_train_text", help = "train data loc")
    parser.add_argument('--dev_file', default = "./data/turkish_dev_text", help = "dev data loc")
    parser.add_argument('--store_model', action='store_true', default=False)

    parser.add_argument('--train_dir', default="", type=str, help= "directory where all training related files will be stored")
    parser.add_argument('--data_dir', default="",type=str, help= "directory where all training related files will be stored")


    parser.add_argument('--units_file', default = "./data/units_char_system.txt", help = "units data loc")
    parser.add_argument('--lexicon_file', default = "./data/lexicon_char_system.txt", help = "lexicon data loc")
    parser.add_argument('--nepoch', default = 20, type=int, help='#epoch')
    # parser.add_argument('--l2', default = 0.0, type=float, help='l2 normalization')
    # parser.add_argument('--clip', default = 0.1, type=float, help='gradient clipping')
    parser.add_argument('--nlayer', default = 1, type=int, help='#layer')
    parser.add_argument('--nhidden', default = 1000, type=int, help='dimension of hidden units in single direction')
    parser.add_argument('--nembed', default = 64, type=int, help='embedding size')
    parser.add_argument('--drop_emb', default = 1.0, type=float, help='embedding (1.0-dropout) probability')

    parser.add_argument('--optimizer', default= 'Adam', help='Training Optimizer')
    parser.add_argument('--lr_rate', default=0, type=float,help='0 for default parameters')
    parser.add_argument('--continue_ckpt', default=0,type=int, help='continue_training with model number')

    parser.add_argument('--noshuffle', default=True, dest='do_shuf', action='store_false', help='shuf batches before training')

    parser.add_argument('--num_sat_layers', default=0, type=int, help='Number of layer for speaker adaptation part')
    parser.add_argument('--apply_sat', default = "unadapted", help = "options: adaptat_sat, fine-tune")
    parser.add_argument('--visual_path', default = "visual adaptation", help = "where visual vectors are")

    return parser

def create_config(args):

    config = {
        'embed_size' : args.nembed ,
        'nhidden_size' : args.nhidden,
        'num_layers' : args.nlayer ,

        'drop_emb' : args.drop_emb ,
        'nepoch' : args.nepoch ,
        'optimizer' : args.optimizer,
        'lr' : args.lr_rate,
        'do_shuf' : args.do_shuf,
        'batch_size' : args.batch_size,

        'dev_file' : args.dev_file ,
        'train_file' : args.train_file ,
        'sat_path': args.visual_path,

        'store_model' : args.store_model,
        'continue_ckpt' : args.continue_ckpt,

        'train_dir' : args.train_dir,

        'num_sat_layers' : args.num_sat_layers,
        'apply_sat' : args.adaptation_stage
    }
    config['random_seed']= 15213

    model_name = 'train_l' + str(config['num_layers'])+ '_e' + str(config['embed_size']) + '_h' \
                 + str(config['hidden_size']) + '_b' + str(config['batch_size']) + '_d' + str(config['drop_emb']) \
                 + str(config['adaptation_stage'])

    config['exp_path'] = './exp/' + model_name + '/'

    if not os.path.exists(config["exp_path"]):
        os.makedirs(config["exp_path"])

    pickle.dump(config, open(os.path.join(config['exp_path'], "config.pkl"), "wb"))

    return config

def main():


    parser = mainParser()
    args = parser.parse_args()
    config = create_config(args)

    #TODO we should think this
    config['adaptation_stage']= "unadapted"



    print("training a model with the following configuration:")
    for key, valeu in config.items():
        print(key+" "+value)

    print("reading labels")
    tr_x = LabelsReader(os.path.join(config['data_file'], "labels.tr"), config['batch_size'])
    cv_x = LabelsReader(os.path.join(config['data_file'], "labels.cv"), config['batch_size'])

    config['ntokens'] = tr_x.get_num_diff_labels()

    if config['adaptation_stage'] != "unadapted":
        print("reading adaptation vectors")
        tr_sat = FeatsReaderKaldi(config['sat_path'], tr_x.get_uttid())
        cv_sat = FeatsReaderKaldi(config['sat_path'], cv_x.get_uttid())

        config['dim_visual'] = int(tr_sat.get_feat_dim())

        data = (tr_x, cv_x, tr_sat, cv_sat)

    else:
        data = (tr_x, cv_x, None, None)

    train(data, config)


if __name__ == "__main__":
    main()
