from utils.fileutils import debug

def check_sets(cv_x, cv_y, tr_x, tr_y):
    #here we should check that cv and tr have the same languages

    for language in cv_x.get_id_augmented_folders():
        if((language not in cv_y.get_target_scheme()) or (language not in tr_x.get_id_augmented_folders()) or (language not in tr_y.get_target_scheme())):
            print("Error: in language: "+language+" not find in cv_y, tr_x or tr_y\n")
            print(debug.get_debug_info())
            print("exiting... \n")

    print("languages checked... \n")

    for language,  targets_dic in tr_y.get_target_scheme().iteritems():
        for target_id, number_targets in targets_dic.iteritems():
            if target_id not in cv_y.get_target_scheme()[language]:
                print("Error: target: "+target_id+" not find in tr_y\n")
                print(debug.get_debug_info())
                print("exiting... \n")

            if(number_targets != cv_y.get_target_scheme()[language][target_id]):
                print("Error: number of targets ("+str(number_targets)+") from tr_y in language: "+str(language)+" in target: "+str(target_id)+"is different form cv_y")
                print(debug.get_debug_info())
                print("exiting... \n")

    print("number of targets checked... \n")

