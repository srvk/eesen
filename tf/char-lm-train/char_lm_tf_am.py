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
import cPickle
from itertools import chain
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
# train_file = 'train.txt'
# test_file = 'dev.txt'
w2i = defaultdict(count(0).next)
sos = '<s>'
eos = '<s>'
space = ' '
w2i[sos]
w2i[space]
S = w2i[eos]
# sos = '<s>'
# eos = '<s>'
# space = ' '
# train_fil = open('am/train.txt').readlines()
# dev_fil = open('am/dev.txt').readlines()
# units = open('units_char.txt').readlines()
# w2i['<s>']
# for i in units:
#     w2i[i.strip().split()[0]]
# # w2i[eos]
# w2i[space]

# lexicon = open('lexicon_numbers_char.txt').readlines()
# lexicon_dict = {}
# for i in lexicon:
#     lexicon_dict[i.strip().split()[0]] = i.strip().split()[1:]
# train_list = []


# for tr in train_fil:
#     temp = list()
#     temp.append(w2i['<s>'])
#     words = tr.strip().split(' ')[1:]
#     for w in words:
#         for ch in lexicon_dict[w]:
#             temp.append(int(ch))
#         temp.append(w2i[space])
#     temp.append(w2i['<s>'])
#     train_list.append(temp)



# dev_list = []
# for tr in dev_fil:
#     temp = list()
#     temp.append(w2i['<s>'])
#     words = tr.strip().split(' ')[1:]
#     for w in words:
#         for ch in lexicon_dict[w]:
#             temp.append(int(ch))
#         temp.append(w2i[space])
#     temp.append(w2i['<s>'])
#     dev_list.append(temp)


# train = train_list
# nwords = len(w2i)
# test = dev_list
# S = w2i[eos]
# # E = w2i[sos]
# assert (nwords == len(w2i))

train = cPickle.load(open('train_am_reduced'))
# train_old.sort(key=lambda x: len(x), reverse=True)
# train = train_old[:-85]
# random.seed(123)
random.shuffle(train)
test = train[:25000]
train = train[50000:]
nwords = len(set(chain(*train)))
# pdb.set_trace()
assert (nwords >= 36),"Missing chars"
# pdb.set_trace()

train.sort(key=lambda x: len(x), reverse=True)
test.sort(key=lambda x: len(x), reverse=True)
print("Training Size = ", len(train))
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
# outputs, _ = tf.nn.dynamic_rnn(cell, x_embs, sequence_length=x_lens, dtype=tf.float32)

# reset_state = cell.zero_state(args.MB_SIZE, dtype=tf.float32)

state = tf.placeholder(tf.float32, shape=[None,2*args.HIDDEN_SIZE], name="state")
# pdb.set_trace()
outputs, next_state = tf.nn.dynamic_rnn(cell, x_embs, sequence_length=x_lens, dtype=tf.float32,initial_state=state)


# Affine transform
output = tf.reshape(tf.concat(outputs,1), [-1, args.HIDDEN_SIZE])

W_sm = tf.Variable(tf.random_uniform([args.HIDDEN_SIZE, nwords]))
b_sm = tf.Variable(tf.random_uniform([nwords]))
logits = tf.matmul(tf.squeeze(output), W_sm) + b_sm
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
model_path='/home/sdalmia/asr-test/lorelei-audio/egs/asr/s5c/201-haitian-flp/langmodel/saved_models/'
with tf.Session() as sess:
    sess.run(init)

    train_losses = []
    print('startup time: %r' % (time.time() - start_))
    i = all_time = dev_time = all_tagged = train_words = 0
    start_train = time.time()
    for ITER in range(10):
        start_ = time.time()
        for i, sid in enumerate(train_order, start=1):
            if i % int(1000 / args.MB_SIZE) == 0:
                print('Updates so far: %d Loss: %f wps: %f' % (
                i - 1, sum(train_losses) / train_words, train_words / (time.time() - start_)))
                all_tagged += train_words
                train_losses = []
                train_words = 0
                all_time = time.time() - start_train
                start_ = time.time()
            if i % int(30000 / args.MB_SIZE) == 0 or all_time > args.TIMEOUT:
                dev_start = time.time()
                test_losses = []
                test_words = 0
                all_time += time.time() - start_train
                print('Testing on dev set...')
                for tid in test_order:
                    t_examples = test[tid:tid + args.MB_SIZE]
                    x_lens_in = [len(example) for example in t_examples]
                    x_in = [pad(example, S, max(x_lens_in)) for example in t_examples]
                    # if(len(x_lens) < 16):
                    #     pdb.set_trace()
                    # else:
                    #     print('ha')
                    test_loss = sess.run(loss, feed_dict={x_input: x_in, x_lens: x_lens_in,state:np.zeros((len(x_lens_in), 2*args.HIDDEN_SIZE))})
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
                    print('TIMEOUT!!')
                    sys.exit(0)
            # train on sent
            examples = train[sid: sid + args.MB_SIZE]
            x_lens_in = [len(example) for example in examples]
            if x_lens_in.count(x_lens_in[0]) != len(x_lens_in):
                x_in =[pad(example, S, max(x_lens_in)) for example in examples]
            else:
                x_in = examples

            train_loss, _ = sess.run([loss, optimizer], feed_dict={x_input: x_in, x_lens: x_lens_in, state:np.zeros((len(x_lens_in), 2*args.HIDDEN_SIZE))})
            tot_words = sum(x_lens_in) - len(x_lens_in)  # Subtract out <s> from the denominator
            train_losses.append(train_loss * tot_words)
            train_words += tot_words
        save_path = saver.save(sess, model_path + 'model' + str(ITER) + '.ckpt')
        random.shuffle(train_order)
