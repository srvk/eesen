#coding: utf-8

from __future__ import print_function
import argparse
import math
import kaldi_io
import numpy as np
from multiprocessing import Pool
# import dynet as dy
import tensorflow as tf
import pdb
from searchAlgo import beam_search, greedy_search
from lm_util import LM_Wrapper, get_char_dicts
import sys
from collections import Counter, defaultdict
from itertools import count
from repoze.lru import lru_cache

BLANK = "<eps>"
SPACE = " "
EOS = "</s>"
SOS = "<s>"
BLANK_ID = 0

NUM_LAYERS = 1
argus = {'MB_SIZE':16,'HIDDEN_SIZE':1024,'EMBED_SIZE':64}
# format of files: each line is "word1/tag2 word2/tag2 ..."
# train_file = 'train.txt'
# #test_file = 'dev.txt'
# w2i = defaultdict(count(0).next)
# eos = '<s>'


f = open('amh-dr-xsampa.wordlist.lex.x').readlines()
w2i = defaultdict(count(0).next)
# fl = f[170:]
# Ss = set()
dic = dict()
S = set()
lex_dic = dict()
sos = '<s>'
eos = '<s>'
space = ' '
w2i[sos]
w2i[space]
for lines in f:
        line = lines.strip().split(' ')
        if(line[0] in lex_dic):
            pdb.set_trace()
        lex_dic[line[0]] = list()
        for w in line[1:]:
            lex_dic[line[0]].append(w2i[w])
            if (w not in dic):
                dic[w] = 0
            dic[w]+=1



# train = cPickle.load(open('train_am_reduced'))
# test = train[:25000]
# train = train[50000:]
nwords = len(w2i)
# test = dev_list
S = w2i[eos]
assert (nwords == len(w2i))

# train.sort(key=lambda x: len(x), reverse=True)
# test.sort(key=lambda x: len(x), reverse=True)

# if argus['MB_SIZE'] != 0:
#     train_order = [x * argus['MB_SIZE'] for x in range((len(train) - 1) / argus['MB_SIZE'] + 1)]
#     test_order = [x * argus['MB_SIZE'] for x in range((len(test) - 1) / argus['MB_SIZE'] + 1)]
# else:
#     train_order = range(len(train))
#     test_order = range(len(test))


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
WORDS_LOOKUP = tf.Variable(tf.random_uniform([nwords, 1, argus['EMBED_SIZE']], -1.0, 1.0))

# Word-level LSTM (configurable number of layers, input is unspecified,
# but will be equal to the embedding dim, output=128)
# cell = tf.nn.rnn_cell.BasicLSTMCell(args.HIDDEN_SIZE, forget_bias=0.0, state_is_tuple=True)
cell = tf.contrib.rnn.LSTMCell(argus['HIDDEN_SIZE'], forget_bias=0.0, state_is_tuple=False)
cell = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS, state_is_tuple=False)

# input sentence placeholder
x_input = tf.placeholder(tf.int32, [None, None], name="x_input")
x_lens = tf.placeholder(tf.int32, [None], name='x_lens')

x_embs = tf.squeeze(tf.nn.embedding_lookup(WORDS_LOOKUP, x_input), axis=2)
# Hack to fix shape so dynamic_rnn will accept this as input
x_embs.set_shape([None, None, argus['EMBED_SIZE']])

# Actually run the RNN
state = tf.placeholder(tf.float32, shape=[None,2*argus['HIDDEN_SIZE']], name="state")
# pdb.set_trace()
outputs, next_state = tf.nn.dynamic_rnn(cell, x_embs, sequence_length=x_lens, dtype=tf.float32,initial_state=state)

# Affine transform
output = tf.reshape(tf.concat(outputs,1,name='outputs_lstm'), [-1, argus['HIDDEN_SIZE']],name='output_lstm')

W_sm = tf.Variable(tf.random_uniform([argus['HIDDEN_SIZE'], nwords]))
b_sm = tf.Variable(tf.random_uniform([nwords]))
logits = tf.matmul(output, W_sm) + b_sm
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
model_path='/home/sdalmia/asr-test/lorelei-audio/egs/asr/s5c/201-haitian-flp/langmodel/saved_models/model2.ckpt'
state_cache = 1024*16
beam_dict = {}

sess= tf.Session()
sess.run(init)
saver.restore(sess, model_path)
def get_soft(context,state_p=None):
    # print("In get_soft")
    temp = list()
    for ch in context:
        if ch not in w2i:
            temp.append(7)
        else:
            temp.append(w2i[ch])
    if(context == ''):
        temp.append(0)
    t_examples = [temp]
    x_lens_in = [len(example) for example in t_examples]
    x_in = [pad(example, S, max(x_lens_in)) for example in t_examples]
    # pdb.set_trace()

    if(state_p is None):
        soft,out,stat = sess.run([softmax,output,next_state], feed_dict={x_input: x_in, x_lens: x_lens_in,state:np.zeros((len(x_lens_in), 2*argus['HIDDEN_SIZE']))})
    else:
        soft,out,stat = sess.run([softmax,output,next_state], feed_dict={x_input: x_in, x_lens: x_lens_in,state:state_p})
    # except:
    #     pdb.set_trace()
    return soft[-1], out[-1], stat

@lru_cache(maxsize=1024 * 1024)
def get_probs(string):
    # print("In get_probs")
    # check this for tensorflow
    global beam_dict
    if len(beam_dict) > state_cache:

        beam_dict={}

    # assert len(string) > 0 and "this leads to an error, because we'll call output() on an initial rnn state..."
    stri = string[:]
    leng = 0
    while(string[:leng] not in beam_dict):
        leng-=1
        if(-1*leng >= len(string)):
            break
    # print("Haha")

    if(stri[:leng] == ''):
        # print("Yo3")
        state=None
        probs,out,state = get_soft(string)
        # pdb.set_trace()
        beam_dict[string] = [out,state]
    elif(string[:leng] == string):
        pdb.set_trace()
        # print("Yo1")
        [output,state] = beam_dict[string[:leng]]
        probs = softmax.eval(feed_dict={'output_lstm':output})
    else:
        # print("Yo2")
        [output,state] = beam_dict[string[:leng]]
        probs,out,state = get_soft(string[leng:],state)
        beam_dict[string] = [out,state]


    # output = _get_s(string)

    # probs = softmax.eval(feed_dict={'output_lstm':output})
    # sess.run(softmax, feed_dict={})
    return probs

def get_prob(cha, context):
    # assert len(context) > 0 and "please padd your stuff"
    # print("In get_prob")
    if (cha not in w2i):
        return 1.0 / math.pow(10.0, 20.0)
    else:
        # return soft[-1][w2i[cha]]
        probs = get_probs(context)
        return probs[w2i[cha]]


def create_parser():
    # lower case, no space, fucked up....
    path = "/Users/thomaszenkel/GitRepos/MasterThesis/bin/CTC/train_char_l5_c320_3s-0/"
    # upper case
    path = "/Users/thomaszenkel/GitRepos/MasterThesis/bin/CTC/ABC/"
    path = "/data/ASR5/fmetze/asr-test/lorelei-audio/egs/asr/s5c/504-amharic-v1-tf/log/dbr-run21-lor17dryrun-eval/"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Decode Character Distribution of CTC"
    )

    parser.add_argument(
        '--arkFile', default=path + "soft_prob.ark",
        help="path to ctc likelihoods file (.ark)"
    )

    parser.add_argument(
        '--ctmOut', default="dbr-run21-lor17dryrun-eval.ctm",
        help="output file to write output to"
    )

    parser.add_argument(
        '--unitsFile', default=path+"../../../105-201-302-401-v1/data/l5_c320_n300_h0.8_share_tr/units.txt"
    )

    parser.add_argument(
        '--lmFile', default="dynetLM/lstm2048Adam.model",
        help="path to the language model file"
    )

    parser.add_argument(
        '--beamSize', type=int, default=20,
        help="size of the beam in beam search"
    )

    parser.add_argument(
        '--insertionBonus', type=float, default=0.4,
        help="insertion bonus"
    )

    parser.add_argument(
        '--lmWeight', type=float, default=1.5,
        help="language model weight"
    )

    parser.add_argument(
        '--merge', type=int, default=1,
        help="merge strings in the beam which are the same, merge should be 0 or 1"
    )

    parser.add_argument(
        '--show', type=int, default=0,
        help="merge strings in the beam which are the same, merge should be 0 or 1"
    )

    parser.add_argument(
        '--nBestOutput', type=int, default=3,
        help="output at most N utterances of the beam"
    )

    parser.add_argument(
        '--expand', default="",
        help="characters which can be added during the search, split character for string is §, example: ' §.§,'"
    )

    # dynet fix
    # parser.add_argument('--dynet-mem')
    # parser.add_argument('--dynet-gpu')

    return parser





if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(args)
    #pdb.set_trace()
    # hacky
    merge = True
    if args.merge == 0:
        merge = False

    # deal with [''] array returned for empty string
    expansion_characters = list(filter(lambda x: len(x) > 0, (args.expand.split("§"))))
    expansion_characters = [' ']
    # create map from integer to char
    id_to_char = {BLANK_ID: BLANK}
    with open(args.unitsFile, "r") as filename:
        for line in filename:
            l = line.split()
            assert len(l) == 2
            # if len(l[0]) > 1:
            #     print("WARN: ignoring {}".format(l[0]))
            #     continue
            id_to_char[int(l[1])] = l[0]
    # invert dict
    char_to_id = {v: k for k, v in id_to_char.items()}
    print(id_to_char)

    kk = w2i.keys()
    candidates = list()
    for k in kk:
        if k != ' ' and k!='<s>':
            candidates.append(char_to_id[k])
    candidates.append(0)
    candidates.sort()
    ch_to_id = {}
    for i in range(len(candidates)) :
        ch_to_id[id_to_char[candidates[i]]] = i
    id_to_ch = {v: k for k, v in ch_to_id.items()}

    # model = dy.Model()
    # not sure about the [0] stuff
    # lstm = model.load(args.lmFile)[0]
    # char_to_int, int_to_char = get_char_dicts()
    #wrapper = LM_Wrapper(char_to_int, lstm)
    #lm_function = wrapper.get_prob
    # char to int -> w2i karna hai
    lm_function = get_prob

    def decode(mat):
        # sanity check
        check_sum = 0.0
        for x in mat[0]:
            check_sum += math.exp(x)
        # pdb.set_trace()
        assert abs(1.0 - check_sum) < 0.01

        beam = beam_search(mat, lm_function, ch_to_id, args.insertionBonus, args.lmWeight, args.beamSize,
                           merge=merge, expansion_chars=expansion_characters)
        return beam

    print("DECODING FOR: {}\n".format(args.arkFile))

    # print(get_prob('n','tente'))
    # pdb.set_trace()
    arc_file = kaldi_io.read_mat_ark(args.arkFile)

    # labelcounts = open('/home/sdalmia/asr-test/lorelei-audio/egs/asr/s5c/201-haitian-flp/exp/train_l6_c140-70_n60_h0.7_v7v/label.counts').read().strip()
    # prior = list()
    # for x in labelcounts.split(' '):
    #     if x != '[' and x != ']':
    #         prior.append(float(x))
    # prior = np.array(prior)
    # sumprior = prior.sum()
    # tprior = prior / sumprior
    # tprior = np.log(tprior)


    with open(args.ctmOut, mode="w", buffering=1) as f:
        for key, mat in arc_file:
            # if args.show == 1:
            #     print("greedy: {}".format(greedy_search(mat, id_to_char)))
            # temp = mat+tprior
            temp  = mat[:,candidates]
            row_sums = temp.sum(axis=1)
            new_matrix = temp / row_sums[:, np.newaxis]
            new_matrix = np.log(new_matrix)
            # pdb.set_trace()

            beam = decode(new_matrix)
            for i, utterance in enumerate(beam):
                if i >= args.nBestOutput:
                    break
                s = "{} {}\n".format(key, utterance)
                if args.show == 1:
                    print(s, end="")
                else:
                    f.write(s)
            print('done ',key)

    print("finished decoding")

