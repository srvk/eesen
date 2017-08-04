import tf

def train_impl(self, data):

        #set random seed so that models can be reproduced
        tf.set_random_seed(self.__config[constants.RANDOM_SEED])
        random.seed(self.__config[constants.RANDOM_SEED])

        #construct the __model acoring to __config
        if(self.__config[constants.ADAPT_STAGE] == constants.ADAPTATION_STAGES.UNADAPTED):
            cv_y, tr_y = data
            tr_sat=None
            cv_sat=None
        else:
            cv_y, tr_y, cv_sat, tr_sat = data

        if not os.path.exists(self.__config[constants.MODEL_DIR]):
            os.makedirs(self.__config[constants.MODEL_DIR])

        #initialize variables of our model
        self.__sess.run(tf.global_variables_initializer())

        # restore a training
        saver, alpha = self.__restore_weights()

        #initialize counters
        best_avg_ters = float("inf")
        best_epoch = 0
        lr_rate = self.__config[constants.LR_RATE]

        for epoch in range(alpha, self.__config[constants.NEPOCH]):

            #start timer...
            tic = time.time()

            #training...
            train_cost, train_ters, ntrain = self.__train_epoch(epoch, lr_rate, tr_x, tr_y, tr_sat)

            if self.__config[constants.STORE_MODEL]:
                saver.save(self.__sess, "%s/epoch%02d.ckpr" % (self.__config[constants.MODEL_DIR], epoch + 1))

            #evaluate on validation...
            cv_cost, cv_ters, ncv = self.__eval_epoch(cv_x, cv_y, cv_sat)

            #print results
            self.__generate_logs(cv_ters, cv_cost, ncv, train_ters, train_cost, ntrain, epoch, lr_rate, tic)

            #change set if needed (mix augmentation)
            if(tr_x.get_num_augmented_folders() > 1):
                self.__update_sets(tr_x, tr_y, tr_sat)

            #update lr_rate if needed
            lr_rate, best_avg_ters, best_epoch = self.__update_lr_rate(epoch, cv_ters, best_avg_ters, best_epoch, saver)
