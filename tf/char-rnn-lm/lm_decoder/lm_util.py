from lm_models.rnn   import *
from repoze.lru import lru_cache
from lm_models.rnn import RNN

import math
import tensorflow as tf
import pdb
import numpy as np
import lm_constants

beam_dict ={}

def pad(seq, element, length):
    assert len(seq) <= length
    r = seq + [element] * (length - len(seq))
    assert len(r) == length
    return r

class LmUtil:
    def __init__(self,config):

        self.model = RNN(config)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        model_path=config[lm_constants.CONFIG_TAGS_TEST.LM_WEIGHTS_CKPT]
        self.state_cache = 1024*64

        self.sess= tf.Session()
        self.sess.run(init)
        self.saver.restore(self.sess, model_path)

        self.S = config[lm_constants.CONFIG_TAGS_TEST.C2I][lm_constants.SPECIAL_CARACTERS.EOS]

        self.c2i = config[lm_constants.CONFIG_TAGS_TEST.C2I]

        self.hidden_size = config[lm_constants.CONF_TAGS.NHIDDEN]
        self.num_layers = config[lm_constants.CONF_TAGS.NLAYERS]


    def get_soft(self,sequence,state_p=None):
        # print("In get_soft")
        temp = list()
        for ch in sequence:
            if ch not in self.c2i:
                print("WARNING!!!!")
                pdb.set_trace()
                temp.append(7)
            else:
                temp.append(self.c2i[ch])
        if(len(sequence) == 0):
            temp.append(0)
        t_examples = [temp]
        x_lens_in = [len(example) for example in t_examples]
        x_in = [pad(example, self.S, max(x_lens_in)) for example in t_examples]
        # pdb.set_trace()
        if(x_lens_in[0] == 1 and len(x_lens_in) == 1):
            x_in = x_in+x_in
            x_lens_in = x_lens_in + x_lens_in
        if(state_p is None):
            # pdb.set_trace()
            soft,out,stat = self.sess.run([self.model.softmax, self.model.output, self.model.next_state], feed_dict={self.model.x_input: x_in, self.model.x_lens: x_lens_in, self.model.state:np.zeros((len(x_lens_in), 2*self.num_layers*self.hidden_size)), self.model.drop_out:1.0})
        else:
            # pdb.set_trac[e()
                        # x_in = x_in+x_in
                        # x_lens_in = x_lens_in + x_lens_in]
            # print "Hi"
            soft,out,stat = self.sess.run([self.model.softmax, self.model.output, self.model.next_state], feed_dict={ self.model.x_input: x_in, self.model.x_lens: x_lens_in, self.model.state: state_p , self.model.drop_out:1.0})
        # except:
        # pdb.set_trace()
        print soft[-1]
        return soft[-1], out[-1], stat[:1]

    @lru_cache(maxsize=1024 * 1024)
    def get_probs(self,sequence):
        seq = list(sequence)
        global beam_dict
        if len(beam_dict) > self.state_cache:
            beam_dict={}
        # assert len(string) > 0 and "this leads to an error, because we'll call output() on an initial rnn state..."
        stri = seq[:]
        leng = None
        # print seq
        while(tuple(seq[:leng]) not in beam_dict):
            if(leng==None):
                leng=-1
            leng-=1
            if(-1*leng >= len(seq)):
                break
        if(len(stri[:leng]) == 0):
            # print("Yo3")
            state=None
            probs,out,state = self.get_soft(seq)
            # pdb.set_trace()
            beam_dict[tuple(seq)] = [out,state]
        elif(seq[:leng] == seq):
            pdb.set_trace()
            # print("Yo1")
            #[output,state] = self.beam_dict[seq[:leng]]
            #probs = softmax.eval(feed_dict={'output_lstm':output})
        else:
            # print("Yo2")
            [output,state] = beam_dict[tuple(seq[:leng])]
            probs,out,state = self.get_soft(seq[leng:],state)
            beam_dict[tuple(seq)] = [out,state]

        return probs

    def get_prob(self,cha, context):
        # assert len(context) > 0 and "please padd your stuff"
        # print("In get_prob")
        # pdb.set_trace()
        # print("Get prob : " + cha + " ", context)
        if (cha not in self.c2i.keys()):
            print cha
            print self.c2i.keys()
            print ("CHECK!!!!!!")
            return 1.0 / math.pow(10.0, 20.0)
        else:
            # return soft[-1][w2i[cha]]
            probs = self.get_probs(tuple(context))
            return probs[self.c2i[cha]]