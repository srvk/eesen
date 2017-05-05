from keras.models import Model

from keras.callbacks import *
from functools import lru_cache

from keras.layers import Input, Embedding, LSTM, Dense, merge, Reshape, Lambda,\
    Flatten, GRU
from keras.layers.core import GRU_Stateless
from collections import defaultdict
import math



def bits_per_unit(y_pred, y_true):
    return K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))/K.log(2)


class CharacterLM:
    # handles loading model, calculating stuff, hidden state change,

    def __init__(self, path):

        self.batch_size = 1
        self.embedding_size = 32
        self.hidden_size = 2048

        self.UNK = "<unk>"
        self.SOS = "<s>"
        self.EOS = "</s>"
        self.char_to_id = self._get_char_to_int()
        self.unk_prob = 1.0 / math.pow(10.0, 20.0)

        # Inputs
        char_inp = Input(batch_shape=(self.batch_size, 1), dtype='uint8', name='input')
        hidden = Input(batch_shape=(self.batch_size, self.hidden_size), name="hidden")

        # calc embedding
        embedding_function = Embedding(len(self.char_to_id), self.embedding_size, name="embedding_1", init="zero")(char_inp)
        char_embedding = Flatten()(embedding_function)

        # calc hidden
        rnn = GRU_Stateless(self.hidden_size, consume_less="gpu", name="gru_1", init="zero")([char_embedding, hidden])

        # calc output
        output = Dense(len(self.char_to_id), activation="softmax", name="output", init="zero")(hidden)


        # (char, hidden) -> hidden
        self.model_hidden = Model(input=[char_inp, hidden], output=rnn)
        # self.model_hidden.summary()

        # hidden -> probs
        self.model_softmax = Model(input=hidden, output=output)
        # self.model_softmax.summary()

        # load weights
        self.model_hidden.load_weights(path, by_name=True)
        self.model_softmax.load_weights(path, by_name=True)


    def _get_char_to_int(self):
        TOKENS = " '-abcdefghijklmnopqrstuvwxyz"

        char_to_int = defaultdict(lambda: len(char_to_int))
        for x in [self.EOS, self.SOS, self.UNK]:
            char_to_int[x]
        for char in TOKENS:
            char_to_int[char]
        return char_to_int

    @lru_cache(1024*1024)
    def _get_hidden(self, string):
        if (len(string) == 0):
            # do first step with StartOfSentence char
            old_hidden = np.zeros((self.batch_size, self.hidden_size))
            X = np.empty((self.batch_size, 1), dtype="int32")
            X[0, 0] = self.char_to_id.get(self.SOS)
            hidden = self.model_hidden.predict_on_batch([X, old_hidden])
            return hidden
        else:
            input_char = string[-1]
            X = np.empty((self.batch_size, 1), dtype="int32")
            UNK_ID = self.char_to_id[self.UNK]
            X[0,0] = self.char_to_id.get(input_char, UNK_ID)

            context = string[:-1]
            old_hidden = self._get_hidden(context)
            hidden = self.model_hidden.predict_on_batch([X, old_hidden])

            return hidden

    @lru_cache(1024*1024)
    def _get_probs(self, context):
        hidden = self._get_hidden(context)
        probs = self.model_softmax.predict_on_batch(hidden)[0]
        return probs

    def get_prob(self, char, context):
        # print("character: {} context: {} ".format(char, context))
        char_id = self.char_to_id.get(char)
        if not char_id:
            return self.unk_prob
        if len(context) > 0 and not self.char_to_id.get(context[-1]):
            return self.unk_prob

        probablities = self._get_probs(context)
        prob = probablities[char_id]
        assert prob >= 0.0
        assert prob <= 1.01
        return prob
