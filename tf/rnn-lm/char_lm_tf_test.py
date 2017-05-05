from __future__ import print_function
import time

start_ = time.time()

from collections import Counter, defaultdict
from itertools import count
import random
import math
import sys
import argparse
import pdb
import numpy as np
import tensorflow as tf
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(gpu=True)
parser.add_argument('MB_SIZE', type=int, help='minibatch size')
parser.add_argument('EMBED_SIZE', type=int, help='embedding size')
parser.add_argument('HIDDEN_SIZE', type=int, help='hidden size')
#parser.add_argument('SPARSE', type=int, help='sparse update 0/1')  # sparse updates by default in tensorflow
parser.add_argument('TIMEOUT', type=int, help='timeout in seconds')
args = parser.parse_args()

NUM_LAYERS = 1

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file = 'train.txt'
test_file = 'dev.txt'
w2i = defaultdict(count(0).next)
sos = '<s>'
eos = '<s>'
space = ' '
train_fil = open('train.txt').readlines()
units = open('units.txt').readlines()
w2i['<s>']
for i in units:
    w2i[i.strip().split()[0]]
# w2i[eos]
w2i[space]
lexicon = open('lexicon_numbers.txt').readlines()
lexicon_dict = {}
for i in lexicon:
    lexicon_dict[i.strip().split()[0]] = i.strip().split()[1:]
train_list = []
for tr in train_fil:
    temp = list()
    temp.append(w2i['<s>'])
    words = tr.strip().split(' ')[1:]
    for w in words:
        for ch in lexicon_dict[w] :
            temp.append(int(ch))
        temp.append(w2i[space])
    temp.append(w2i['<s>'])
    train_list.append(temp)

dev_list = []
for tr in train_fil:
    temp = list()
    temp.append(w2i['<s>'])
    words = tr.strip().split(' ')[1:]
    for w in words:
        for ch in lexicon_dict[w] :
            temp.append(int(ch))
        temp.append(w2i[space])
    temp.append(w2i['<s>'])
    dev_list.append(temp)


train = train_list
nwords = len(w2i)
test = dev_list
S = w2i[eos]
# E = w2i[sos]
assert (nwords == len(w2i))

train.sort(key=lambda x: len(x), reverse=True)
test.sort(key=lambda x: len(x), reverse=True)

if args.MB_SIZE != 0:
    train_order = [x * args.MB_SIZE for x in range((len(train) - 1) / args.MB_SIZE + 1)]
    test_order = [x * args.MB_SIZE for x in range((len(test) - 1) / args.MB_SIZE + 1)]
else:
    train_order = range(len(train))
    test_order = range(len(test))


def pad(seq, element, length):
    assert len(seq) <= length
    r = seq + [element] * (length - len(seq))
    assert len(r) == length
    return r



# if args.gpu:
#     cpu_or_gpu = '/gpu:0'
# else:
#     cpu_or_gpu = '/cpu:0'
# #pdb.set_trace()
# with tf.device(cpu_or_gpu):
    # Lookup parameters for word embeddings
WORDS_LOOKUP = tf.Variable(tf.random_uniform([nwords, 1, args.EMBED_SIZE], -1.0, 1.0))

# Word-level LSTM (configurable number of layers, input is unspecified,
# but will be equal to the embedding dim, output=128)
# cell = tf.nn.rnn_cell.BasicLSTMCell(args.HIDDEN_SIZE, forget_bias=0.0, state_is_tuple=True)
cell = tf.contrib.rnn.LSTMCell(args.HIDDEN_SIZE, forget_bias=0.0, state_is_tuple=False)
cell = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS, state_is_tuple=False)

# input sentence placeholder
x_input = tf.placeholder(tf.int32, [None, None], name="x_input")
x_lens = tf.placeholder(tf.int32, [None], name='x_lens')

x_embs = tf.squeeze(tf.nn.embedding_lookup(WORDS_LOOKUP, x_input), axis=2)
# Hack to fix shape so dynamic_rnn will accept this as input
x_embs.set_shape([None, None, args.EMBED_SIZE])

# Actually run the RNN
state = tf.placeholder(tf.float32, shape=[None,2*args.HIDDEN_SIZE], name="state")
# pdb.set_trace()
outputs, next_state = tf.nn.dynamic_rnn(cell, x_embs, sequence_length=x_lens, dtype=tf.float32,initial_state=state)


# Affine transform
output = tf.reshape(tf.concat(outputs,1), [-1, args.HIDDEN_SIZE])

W_sm = tf.Variable(tf.random_uniform([args.HIDDEN_SIZE, nwords]))
b_sm = tf.Variable(tf.random_uniform([nwords]))
logits = tf.matmul(output, W_sm) + b_sm

# logits = tf.matmul(tf.squeeze(output), W_sm) + b_sm
softmax = tf.nn.softmax(logits, dim=-1, name='softmax_final')
# Compute categorical loss
# Don't predict the first input (<s>), and don't worry about the last output (after we've input </s>)
# losses = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs[:-1], x_input[1:])
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[:-1], labels=tf.reshape(x_input, [-1])[1:])
loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer().minimize(loss)

print('Graph created.', file=sys.stderr)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
model_path='/home/sdalmia/asr-test/lorelei-audio/egs/asr/s5c/201-haitian-flp/langmodel/saved_models/model4.ckpt'

sess= tf.Session()
sess.run(init)
saver.restore(sess, model_path)
def get_prob(cha,context):
    temp = list()
    for ch in context:
        if ch not in w2i:
            temp.append(7)
        else:
            temp.append(w2i[ch])

    t_examples = [temp]
    x_lens_in = [len(example) for example in t_examples]
    x_in = [pad(example, S, max(x_lens_in)) for example in t_examples]
    # [[2]] , [1]
    try:
        soft = sess.run(softmax, feed_dict={x_input: x_in, x_lens: x_lens_in,state:np.zeros((len(x_lens_in), 2*args.HIDDEN_SIZE))})
    except:
        pdb.set_trace()
    if (cha not in w2i):
        return 1.0 / math.pow(10.0, 20.0)
    else:
        return soft[-1][w2i[cha]]



cha = 'n'
context = 'tente'
print(get_prob(cha,context))
pdb.set_trace()
#pickle.dump(w2i,open('w2i.pkl','wb'))
#pdb.set_trace()
