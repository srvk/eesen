from utils.fileutils import debug
import os, sys
import constants

def check_sets_training(cv_x, cv_y, tr_x, tr_y, tr_sat = None, cv_sat = None):

    #here we should check that cv and tr have the same languages

    if (tr_sat == None and cv_sat == None):
        for language in cv_x.get_language_augment_scheme():
            if((language not in cv_y.get_language_scheme()) or
                   (language not in tr_x.get_language_augment_scheme()) or
                   (language not in tr_y.get_language_scheme())):
                print("Error: in language: "+language+" not find in cv_y, tr_x or tr_y\n")
                print(debug.get_debug_info())
                print("exiting... \n")
                sys.exit()
    elif(tr_sat != None and cv_sat != None):

        for language in cv_x.get_language_augment_scheme():
            if((language not in cv_y.get_language_scheme()) or
            (language not in tr_x.get_language_augment_scheme()) or
            (language not in tr_y.get_language_scheme()) or
            (language not in tr_sat.get_language_scheme()) or
            (language not in cv_sat.get_language_scheme())):

                print("Error: in language: "+language+" not find in cv_y, tr_x or tr_y\n")
                print(debug.get_debug_info())
                print("exiting... \n")
                sys.exit()

        print("languages checked ...")
        print("(cv_x vs cv_y vs tr_x vs tr_y vs sat_x)")
        print("correct\n")
    else:
        print("Argument error in check_sets_training cv_sat and tr_sat should have same content structure (e.g. cv_sat = None, tr_sat = None)")
        print("Now cv_sat = "+str(type(cv_sat))+" and tr_sat = "+str(type(tr_sat)))
        print(debug.get_debug_info())
        print("exiting... \n")
        sys.exit()

    for language,  targets_dic in tr_y.get_language_scheme().items():
        for target_id, number_targets in targets_dic.items():
            if target_id not in cv_y.get_language_scheme()[language]:
                print("Error: target: "+target_id+" not find in tr_y\n")
                print(debug.get_debug_info())
                print("exiting... \n")
                sys.exit()

            if(number_targets != cv_y.get_language_scheme()[language][target_id]):
                print(80 * "*")
                print(80 * "*")
                print("WARINING!: number of targets ("+str(number_targets)+") from tr_y ("+str(cv_y.get_language_scheme()[language][target_id])+") in language: "+str(language)+" in target: "+str(target_id)+"is different form cv_y")
                print(debug.get_debug_info())
                print("replicating biggest one...")
                if(number_targets > cv_y.get_language_scheme()[language][target_id]):
                    cv_y.set_number_targets(language, target_id, number_targets)
                else:
                    tr_y.set_number_targets(language, target_id, cv_y.get_language_scheme()[language][target_id])
                print(80 * "*")
                print(80 * "*")



        print("languages checked ...")
        print("(cv_x vs cv_y vs tr_x vs tr_y)")

def check_sets_testing(config, test_x, sat_x = None):

    if(config[constants.INPUT_FEATS_DIM] != test_x.get_num_dim()):
        print("Error: input dimension from model loaded("+str(config[constants.INPUT_FEATS_DIM])+") is not the same as input_feats ("+str(test_x.get_num_dim())+")")
        print(debug.get_debug_info())
        print("exiting... \n")
        sys.exit()

    if(sat_x):
        if(config[constants.SAT_FEAT_DIM] != sat_x.get_num_dim()):
            print("Error: input sat dimension from model loaded("+str(config[constants.SAT_FEAT_DIM])+") is not the same as input_feats ("+str(sat_x.get_num_dim())+")")
            print(debug.get_debug_info())
            print("exiting... \n")
            sys.exit()


def check_sat_exist(config, tr_x):

    language_general_flag = False
    for language in tr_x.get_language_augment_scheme():

        if (len(tr_x.get_language_augment_scheme()) > 1):
            lan_path = os.path.join(config[constants.CONF_TAGS.DATA_DIR], language)
        else:
            lan_path = config[constants.CONF_TAGS.DATA_DIR]

        language_flag = False
        for lan_file in os.listdir(lan_path):
            if(os.path.splitext(lan_file)[0] == constants.DEFAULT_FILENAMES.SAT):
                print("speaker adaptation vector found for language: "+language+"\n")
                language_flag = True
                language_general_flag = True

        if (language_general_flag):
            if (not language_flag):
                print("Error: inconsisten structure over language on sat (local_sat.scp) files\n")
                print(debug.get_debug_info())
                print("exiting... \n")
                sys.exit()
                return False
            else:
                return True
