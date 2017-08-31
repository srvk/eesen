import tensorflow as tf
import lm_constants
import sys
from lm_utils.lm_fileutils import debug

class RNN:
    def __init__(self,config):
        embed_size =config[lm_constants.CONF_TAGS.EMBEDS_SIZE]

        nwords = config[lm_constants.CONF_TAGS.NUM_TARGETS]+1
        hidden_size = config[lm_constants.CONF_TAGS.NHIDDEN]
        num_layers = config[lm_constants.CONF_TAGS.NLAYERS]

        lr_rate = config[lm_constants.CONF_TAGS.LR_RATE]

        #common arguments
        self.x_input = tf.placeholder(tf.int32, [None, None], name="x_input")
        self.x_lens = tf.placeholder(tf.int32, [None], name = 'x_lens')
        self.state = tf.placeholder(tf.float32, shape=[None,2*num_layers*hidden_size], name="state")
        self.drop_out = tf.placeholder(tf.float32)


        with tf.variable_scope("Embedding"):

            self.embedding = tf.get_variable("embedding", [nwords, embed_size])
            self.x_embs = tf.nn.embedding_lookup(self.embedding, self.x_input)
            self.x_embs_drop = tf.nn.dropout(self.x_embs, self.drop_out)

        if(config[lm_constants.CONF_TAGS.SAT_SATGE] != lm_constants.SAT_SATGES.UNADAPTED):
            self.sat_input = tf.placeholder(tf.float32, [None, None, config[lm_constants.CONF_TAGS.NUM_SAT_DIM]], name="sat_input")
            sat_embs = self.my_sat_module(self.x_embs, self.sat_input, config)
            self.x_embs_drop = tf.nn.dropout(sat_embs, self.drop_out)

        with tf.variable_scope("RNN"):

            self.cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=False)
            self.drop_cell = tf.contrib.rnn.DropoutWrapper( self.cell, output_keep_prob=self.drop_out)
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * num_layers, state_is_tuple=False)

            #input shifted features
            #TODO undo the sidechain
            self.outputs, self.next_state = tf.nn.dynamic_rnn(self.cell, self.x_embs_drop, sequence_length=self.x_lens, dtype=tf.float32, initial_state=self.state)

            self.output = tf.reshape(tf.concat(self.outputs,1), [-1, hidden_size],name='output')


        with tf.variable_scope("Affine"):
            self.W_sm = tf.Variable(tf.random_uniform([hidden_size, nwords]))
            self.b_sm = tf.Variable(tf.random_uniform([nwords]))
            self.logits = tf.matmul(tf.squeeze(self.output), self.W_sm) + self.b_sm
            self.softmax = tf.nn.softmax(self.logits, dim=-1, name='softmax_final')

        with tf.variable_scope("Loss"):
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[:-1], labels=tf.reshape(self.x_input, [-1])[1:])
            self.loss = tf.reduce_mean(self.losses)
            self.ppl = tf.exp(self.loss)

        if config[lm_constants.CONF_TAGS.SAT_SATGE] == lm_constants.SAT_SATGES.TRAIN_SAT:
            var_list=[]
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                if (lm_constants.SCOPES.SAT_SHIFT in var.name):
                    var_list.append(var)
        else:
            var_list=[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]


        if config[lm_constants.CONF_TAGS.OPTIMIZER] == lm_constants.OPTIMIZERS.ADAM:

            if(lr_rate==0):
                self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, var_list=var_list)
            else:
                self.optimizer = tf.train.AdamOptimizer(epsilon=lr_rate).minimize(self.loss, var_list=var_list)

        elif config[lm_constants.CONF_TAGS.OPTIMIZER] ==  lm_constants.OPTIMIZERS.SDG:
            if(lr_rate==0):
                self.optimizer = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss, var_list=var_list)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss, var_list=var_list)
        else:
            print("invalid optimizer id: "+config[lm_constants.CONF_TAGS.OPTIMIZER])
            print(debug.get_debug_info())
            print("exiting...")
            sys.exit()


    def my_sat_module(self, input_feats, sat_input, config):

            with tf.variable_scope(lm_constants.SCOPES.SPEAKER_ADAPTAION):
                if(config[lm_constants.CONF_TAGS.SAT_SATGE] == lm_constants.SAT_SATGES.CONCAT):

                    #sat_input.set_shape([None, input_feats.get_shape()[1], config[lm_constants.CONF_TAGS.SAT_FEAT_DIM]])
                    sat_input = tf.tile(sat_input, tf.stack([1, tf.shape(input_feats)[1], 1]))

                    return tf.concat([input_feats, sat_input], 2)

                elif(config[lm_constants.CONF_TAGS.SAT_SATGE] == lm_constants.SAT_SATGES.FUSE):
                    with tf.variable_scope(lm_constants.SCOPES.SAT_FUSE):
                        sat_input = tf.tile(sat_input, tf.stack([1, tf.shape(input_feats)[1], 1]))
                        outputs = tf.concat([input_feats, sat_input], 2)
                        for i in range(config[lm_constants.CONF_TAGS.NUM_SAT_LAYERS]-1):
                            outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = config[lm_constants.CONF_TAGS.SAT_FEAT_DIM])
                        outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = config[lm_constants.CONF_TAGS.EMBEDS_SIZE])
                        return outputs

                else:
                    outputs=sat_input
                    with tf.variable_scope(lm_constants.SCOPES.SAT_SHIFT):
                        for i in range(config[lm_constants.CONF_TAGS.NUM_SAT_LAYERS]-1):
                            outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = config[lm_constants.CONF_TAGS.SAT_FEAT_DIM])
                        outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = config[lm_constants.CONF_TAGS.EMBEDS_SIZE])
                        tf.add(input_feats, outputs)


