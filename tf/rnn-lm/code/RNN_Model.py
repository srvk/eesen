import tensorflow as tf

class RNN_Model:
    def __init__(self,config):
        embed_size =config['embed_size']
        nwords = config['nwords']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        #grad_opt = config['grad_opt']
        # gpu_id = config['gpu_id']
        drop_emb = config['drop_emb']
        self.x_input = tf.placeholder(tf.int32, [None, None], name="x_input")
        self.x_lens = tf.placeholder(tf.int32, [None], name = 'x_lens')
        self.state = tf.placeholder(tf.float32, shape=[None,2*num_layers*hidden_size], name="state")
        with tf.variable_scope("Embedding"):
            self.embedding = tf.get_variable("embedding", [nwords, embed_size])
            self.x_embs = tf.nn.embedding_lookup(self.embedding, self.x_input)
            self.x_embs_drop = tf.nn.dropout(self.x_embs, drop_emb)

        with tf.variable_scope("RNN"):
            self.cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=False)
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * num_layers, state_is_tuple=False)
            self.outputs, self.next_state = tf.nn.dynamic_rnn(self.cell, self.x_embs_drop, sequence_length=self.x_lens, dtype=tf.float32,initial_state=self.state)
            self.output = tf.reshape(tf.concat(self.outputs,1), [-1, hidden_size],name='output')

        with tf.variable_scope("Affine"):
            self.W_sm = tf.Variable(tf.random_uniform([hidden_size, nwords]))
            self.b_sm = tf.Variable(tf.random_uniform([nwords]))
            self.logits = tf.matmul(tf.squeeze(self.output), self.W_sm) + self.b_sm
            self.softmax = tf.nn.softmax(self.logits, dim=-1, name='softmax_final')

        with tf.variable_scope("Loss"):
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[:-1], labels=tf.reshape(self.x_input, [-1])[1:])
            self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)




