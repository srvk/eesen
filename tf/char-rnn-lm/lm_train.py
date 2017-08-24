import argparse
import json
import lm_constants

from lm_reader.lm_labels_reader.labels_reader import LabelsReader
from lm_tf.lm_tf_train import *


# from lm_reader.lm_feats_reader.lm_feats_reader import FeatsReaderKald

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
    parser.add_argument('--continute_pkt', default=0,type=int, help='continue_training with model number')

    parser.add_argument('--noshuffle', default=True, dest='do_shuf', action='store_false', help='shuf batches before training')
    parser.add_argument('--lmweights', default="",type=str, help='Weight path. If used sat layers will be trained. If not used will not be charged')




    #sat arguments
    parser.add_argument('--apply_sat', default = False, help='apply and train a sat layer')
    parser.add_argument('--concat_sat', default = False, help='apply and train a sat layer')
    parser.add_argument('--num_sat_layers', default = 2, type=int, help='number of sat layers for sat module')

    return parser

def createConfig(args):

    config = {
        lm_constants.CONF_TAGS.NEMBEDS : args.nembed,
        lm_constants.CONF_TAGS.NLAYERS : args.nlayer,
        lm_constants.CONF_TAGS.NHIDDEN : args.nhidden,

        lm_constants.CONF_TAGS.STORE_MODEL : args.store_model,

        'drop_emb' : args.drop_emb ,
        'nepoch' : args.nepoch ,
        'lexicon_file' : args.lexicon_file ,
        'units_file' : args.units_file ,
        'dev_file' : args.dev_file ,
        'train_file' : args.train_file ,

        'batch_size' : args.batch_size,
        'optimizer' : args.optimizer,
        'lr' : args.lr_rate,
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

    return config

def main():


    parser = mainParser()
    args = parser.parse_args()
    config = createConfig(args)

    print("about to train with the following configuration:")
    print(80 * "-")
    for key, element in config.items():
        print(str(key)+" "+str(element))

    print(80 * "-")
    print(80 * "-")
    print("reading data")
    print(80 * "-")
    print("reading tr_x...")
    print(80 * "-")
    tr_x = LabelsReader(config['train_file'], config['batch_size'])

    print("reading cv_x...")
    print(80 * "-")
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

    print("data read.")
    print(80 * "-")
    print(80 * "-")
    train(data, config)


if __name__ == "__main__":
    main()
