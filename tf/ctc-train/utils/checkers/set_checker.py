from utils.fileutils import debug
import constants

def check_sets(cv_x, cv_y, tr_x, tr_y):
    #here we should check that cv and tr have the same languages

    for language in cv_x.get_language_augment_scheme():
        if((language not in cv_y.get_language_scheme()) or (language not in tr_x.get_language_augment_scheme()) or (language not in tr_y.get_language_scheme())):
            print("Error: in language: "+language+" not find in cv_y, tr_x or tr_y\n")
            print(debug.get_debug_info())
            print("exiting... \n")

    print("languages checked... \n")

    for language,  targets_dic in tr_y.get_language_scheme().iteritems():
        for target_id, number_targets in targets_dic.iteritems():
            if target_id not in cv_y.get_language_scheme()[language]:
                print("Error: target: "+target_id+" not find in tr_y\n")
                print(debug.get_debug_info())
                print("exiting... \n")

            if(number_targets != cv_y.get_language_scheme()[language][target_id]):
                print("Error: number of targets ("+str(number_targets)+") from tr_y in language: "+str(language)+" in target: "+str(target_id)+"is different form cv_y")
                print(debug.get_debug_info())
                print("exiting... \n")

    print("number of targets checked... \n")


def check_test(config, test_x, sat_x = None):

    if(config[constants.INPUT_FEATS_DIM] != test_x.get_num_dim()):
        print("Error: input dimension from model loaded("+str(config[constants.INPUT_FEATS_DIM])+") is not the same as input_feats ("+str(test_x.get_num_dim())+")")
        print(debug.get_debug_info())
        print("exiting... \n")

    if(sat_x):
        if(config[constants.SAT_FEAT_DIM] != sat_x.get_num_dim()):
            print("Error: input sat dimension from model loaded("+str(config[constants.SAT_FEAT_DIM])+") is not the same as input_feats ("+str(sat_x.get_num_dim())+")")
            print(debug.get_debug_info())
            print("exiting... \n")


