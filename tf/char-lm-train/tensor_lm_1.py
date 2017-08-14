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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(gpu=False)
parser.add_argument('MB_SIZE', type=int, help='minibatch size')
parser.add_argument('EMBED_SIZE', type=int, help='embedding size')
parser.add_argument('HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('SPARSE', type=int, help='sparse update 0/1')  # sparse updates by default in tensorflow
parser.add_argument('TIMEOUT', type=int, help='timeout in seconds')
args = parser.parse_args()

NUM_LAYERS = 1

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file = 'train.txt'
test_file = 'dev.txt'
w2i = defaultdict(count(0).next)
eos = '<s>'

train_fil = open('train.txt').readlines()
units = open('units.txt').readlines()
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
    temp.append(w2i['<s>'])
    dev_list.append(temp)
w2i['<s>']
for i in units:
    w2i[i.strip().split()[0]]

# def read(fname):
#     """
#     Read a file where each line is of the form "word1 word2 ..."
#     Yields lists of the form [word1, word2, ...]
#     """
#     with file(fname) as fh:
#         for line in fh:
#             sent = [w2i[eos]]
#             sent += [w2i[x] for x in line.strip().split()]
#             sent.append(w2i[eos])
#             yield sent
#
#
train = train_list
nwords = len(w2i)
test = dev_list
S = w2i[eos]
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


if args.gpu:
    cpu_or_gpu = '/gpu:0'
else:
    cpu_or_gpu = '/cpu:0'
pdb.set_trace()
with tf.device(cpu_or_gpu):
    # Lookup parameters for word embeddings
    WORDS_LOOKUP = tf.Variable(tf.random_uniform([nwords, 1, args.EMBED_SIZE], -1.0, 1.0))

    # Word-level LSTM (configurable number of layers, input is unspecified,
    # but will be equal to the embedding dim, output=128)
    # cell = tf.nn.rnn_cell.BasicLSTMCell(args.HIDDEN_SIZE, forget_bias=0.0, state_is_tuple=True)
    cell = tf.contrib.rnn.LSTMCell(args.HIDDEN_SIZE, forget_bias=0.0, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS, state_is_tuple=True)

    # input sentence placeholder
    x_input = tf.placeholder(tf.int32, [None, None], name="x_input")
    x_lens = tf.placeholder(tf.int32, [None], name='x_lens')

    x_embs = tf.squeeze(tf.nn.embedding_lookup(WORDS_LOOKUP, x_input), axis=2)
    # Hack to fix shape so dynamic_rnn will accept this as input
    x_embs.set_shape([None, None, args.EMBED_SIZE])

    # Actually run the RNN
    outputs, _ = tf.nn.dynamic_rnn(cell, x_embs, sequence_length=x_lens, dtype=tf.float32)


    # Affine transform
    output = tf.reshape(tf.concat(outputs,1), [-1, args.HIDDEN_SIZE])

    W_sm = tf.Variable(tf.random_uniform([args.HIDDEN_SIZE, nwords]))
    b_sm = tf.Variable(tf.random_uniform([nwords]))
    logits = tf.matmul(tf.squeeze(output), W_sm) + b_sm

    # Compute categorical loss
    # Don't predict the first input (<s>), and don't worry about the last output (after we've input </s>)
    # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs[:-1], x_input[1:])
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[:-1], labels=tf.reshape(x_input, [-1])[1:])
    loss = tf.reduce_mean(losses)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    saver = tf.train.Saver()
    print('Graph created.', file=sys.stderr)

# pdb.set_trace()
sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
tf.global_variables_initializer().run()
print('Session initialized.', file=sys.stderr)

train_losses = []
print('startup time: %r' % (time.time() - start_))
i = all_time = dev_time = all_tagged = train_words = 0
start_train = time.time()
for ITER in range(10):
    random.shuffle(train_order)
    start_ = time.time()
    for i, sid in enumerate(train_order, start=1):
        if i % int(500 / args.MB_SIZE) == 0:
            print('Updates so far: %d Loss: %f wps: %f' % (
            i - 1, sum(train_losses) / train_words, train_words / (time.time() - start_)))
            all_tagged += train_words
            train_losses = []
            train_words = 0
            all_time = time.time() - start_train
            start_ = time.time()
        if i % int(10000 / args.MB_SIZE) == 0 or all_time > args.TIMEOUT:
            save_path = saver.save(sess, "MODEL_"+str(ITER)+".ckpt")
            dev_start = time.time()
            test_losses = []
            test_words = 0
            all_time += time.time() - start_train
            print('Testing on dev set...')
            for tid in test_order:
                t_examples = test[tid:tid + args.MB_SIZE]
                x_lens_in = [len(example) for example in t_examples]
                x_in = [pad(example, S, max(x_lens_in)) for example in t_examples]
                test_loss = sess.run(loss, feed_dict={x_input: x_in, x_lens: x_lens_in})
                tot_words = sum(x_lens_in) - len(
                    x_lens_in)  # Subtract out <s> from the denominator - to be in line with other toolkits
                test_losses.append(test_loss * tot_words)
                test_words += tot_words
            nll = sum(test_losses) / test_words
            dev_time += time.time() - dev_start
            train_time = time.time() - start_train - dev_time
            print('nll=%.4f, ppl=%.4f, time=%.4f, words_per_sec=%.4f' % (
            nll, math.exp(nll), train_time, all_tagged / train_time), file=sys.stderr)
            start_ = start_ + (time.time() - dev_start)
            if all_time > args.TIMEOUT:
                sys.exit(0)
        # train on sent
        examples = train[sid: sid + args.MB_SIZE]
        x_lens_in = [len(example) for example in examples]
        if x_lens_in.count(x_lens_in[0]) != len(x_lens_in):
            x_in = [pad(example, S, max(x_lens_in)) for example in examples]
        else:
            x_in = examples
        
        train_loss, _ = sess.run([loss, optimizer], feed_dict={x_input: x_in, x_lens: x_lens_in})
        tot_words = sum(x_lens_in) - len(x_lens_in)  # Subtract out <s> from the denominator
        train_losses.append(train_loss * tot_words)
        train_words += tot_words
	#save_path = saver.save(sess, "MODEL_"+ITER+".ckpt")
