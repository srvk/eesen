import tensorflow as tf

class RNN_Model:
    def __init__(self,config):
        embed_size =config['embed_size']
        nwords = config['nwords']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        lr_rate = config['lr']

        dim_sat = config['dim_visual']

        number_of_sat_layers=config['num_sat_layers']

        self.x_input = tf.placeholder(tf.int32, [None, None], name="x_input")


        if(config["apply"]):
            self.sat_input = tf.placeholder(tf.float32, [None, None, dim_sat], name="sat_input")

        self.x_lens = tf.placeholder(tf.int32, [None], name = 'x_lens')
        self.state = tf.placeholder(tf.float32, shape=[None,2*num_layers*hidden_size], name="state")

        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope("Embedding"):

            self.embedding = tf.get_variable("embedding", [nwords, embed_size])

            #self.embeddings = tf.Variable(tf.random_uniform([nwords, embed_size, -1.0, 1.0))

            self.x_embs = tf.nn.embedding_lookup(self.embedding, self.x_input)
            self.x_embs_drop = tf.nn.dropout(self.x_embs, self.keep_prob)

        with tf.variable_scope("Shift"):

            outputs=self.sat_input

            for i in range(number_of_sat_layers-1):

                outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = dim_sat)

            outputs = tf.contrib.layers.fully_connected(activation_fn = None, inputs = outputs, num_outputs = embed_size)

        #    self.x_embs_drop=tf.add(outputs, self.x_embs, name="shift")


        with tf.variable_scope("RNN"):

            self.cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=False)
            #self.drop_cell = tf.contrib.rnn.DropoutWrapper( self.cell, output_keep_prob=self.keep_prob)
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * num_layers, state_is_tuple=False)

            #input shifted features
            #TODO undo the sidechain
            self.outputs, self.next_state = tf.nn.dynamic_rnn(self.cell, outputs, sequence_length=self.x_lens, dtype=tf.float32,initial_state=self.state)
            self.output = tf.reshape(tf.concat(self.outputs,1), [-1, hidden_size],name='output')

        with tf.variable_scope("Affine"):
            self.W_sm = tf.Variable(tf.random_uniform([hidden_size, nwords]))
            self.b_sm = tf.Variable(tf.random_uniform([nwords]))
            self.logits = tf.matmul(tf.squeeze(self.output), self.W_sm) + self.b_sm
            self.softmax = tf.nn.softmax(self.logits, dim=-1, name='softmax_final')

        with tf.variable_scope("Loss"):
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[:-1], labels=tf.reshape(self.x_input, [-1])[1:])
            self.loss = tf.reduce_mean(self.losses)

        if config['adaptation_stage'] == 'adapt_sat':
            var_list=[]
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                if ("Shift" in var.name):
                    var_list.append(var)

        if config['adaptation_stage'] == 'fine_tune':
            var_list=[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

        if config['optimizer'] == 'Adam':

            if(lr_rate==0):
                self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, var_list=var_list)
            else:
                self.optimizer = tf.train.AdamOptimizer(epsilon=lr_rate).minimize(self.loss, var_list=var_list)

        elif config['optimizer'] == 'SGD':
            if(lr_rate==0):
                self.optimizer = tf.train.GradientDescentOptimizer().minimize(self.loss, var_list=var_list)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss, var_list=var_list)




