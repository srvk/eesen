import argparse
import json
import pdb

from reader.labels_reader.labels_reader import LabelsReader
#from reader.feats_reader.feats_reader import FeatsReaderKaldi



from train_rnn_lm import *

def mainParser():
    parser = argparse.ArgumentParser(description='Train TF-RNN_LM')
    parser.add_argument('--batch_size', default = 16, type=int, help='batch size')

    parser.add_argument('--train_file', default = "./data/turkish_train_text", help = "train data loc")
    parser.add_argument('--dev_file', default = "./data/turkish_dev_text", help = "dev data loc")
    parser.add_argument('--train_dir', help = "directory where logs and weights will be stored")

    parser.add_argument('--store_model', action='store_true', default=False)


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

    #TODO continute_pkt
    parser.add_argument('--cont', default=0,type=int, help='continue_training with model number')

    parser.add_argument('--noshuffle', default=True, dest='do_shuf', action='store_false', help='shuf batches before training')
    parser.add_argument('--lmweights', default="",type=str, help='Weight path. If used sat layers will be trained. If not used will not be charged')
    parser.add_argument('--num_sat_layers', default=0, type=int, help='Number of layer for speaker adaptation part')
    parser.add_argument('--adaptation_stage', default = "unadapted", help = "options: adaptat_sat, fine-tune")
    parser.add_argument('--visual_path', default = "visual adaptation", help = "where visual vectors are")

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
        'proto' : args.proto ,
        'batch_size' : args.batch_size,
        'optimizer' : args.optimizer,
        'lr' : args.lr,
        'cont' : args.cont,

        'do_shuf' : args.do_shuf,

        'sat_path': args.visual_path,
        'weight_path' : args.lmweights,
        'num_sat_layers' : args.num_sat_layers,
        'adaptation_stage' : args.adaptation_stage
    }
    config['random_seed']= 15213

    model_name = 'train_l' + str(config['num_layers'])+ '_e' + str(config['embed_size']) + '_h' \
                 + str(config['hidden_size']) + '_b' + str(config['batch_size']) + '_d' + str(config['drop_emb']) \
                 + str(config['adaptation_stage'])

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

    print("reading labels")
    tr_x = LabelsReader(config['train_file'], config['batch_size'])
    cv_x = LabelsReader(config['dev_file'], config['batch_size'])

    config['nwords'] = tr_x.get_num_diff_labels()

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
