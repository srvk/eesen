import dynet as dy
import random
from functools import lru_cache

class LSTM(dy.Saveable):
    # that's a fix to make loading from a file work
    # probably not the best solution
    state_cache = 1024*16
    beam_dict = {}

    def __init__(self, model, vocab_size, embedding_size = 64, hidden_dim = 1024, num_layers = 2):
        self.embedding = model.add_lookup_parameters((vocab_size, embedding_size))
        self.lstm = dy.VanillaLSTMBuilder(num_layers, embedding_size, hidden_dim, model)
        self.W = model.add_parameters((vocab_size, hidden_dim))
        self.b = model.add_parameters(vocab_size)

        self._renew_cg()


    def _renew_cg(self):
        """
        use this function instead of calling dy.renew_cg() to avoid bugs in the future
        :return:
        """
        dy.renew_cg()
        LSTM.beam_dict = {}

    def _get_s(self, string):
        if string in LSTM.beam_dict:
            return LSTM.beam_dict[string]
        if len(string) == 0:
            return self.lstm.initial_state()

        assert len(string) > 0 and "beam dict was not initialized correctly"
        context, char = string[:-1], string[-1]
        prev = self._get_s(context)
        s = prev.add_input(self.embedding[char])

        LSTM.beam_dict[string] = s
        return s

    @lru_cache(1024 * 1024)
    def get_probs(self, string):
        if len(LSTM.beam_dict) > LSTM.state_cache:
            self._renew_cg()

        assert len(string) > 0 and "this leads to an error, because we'll call output() on an initial rnn state..."
        s = self._get_s(string)

        W = dy.parameter(self.W)
        b = dy.parameter(self.b)

        # does this actually help?
        dy.cg_checkpoint()
        probs_sym = dy.softmax(W * s.output() + b)
        probs = probs_sym.value()
        dy.cg_revert()

        return probs

    def get_prob(self, char, context):
        assert len(context) > 0 and "please padd your stuff"
        probs = self.get_probs(tuple(context))
        return probs[char]

    # sentences [[s1w1, s1w2, s1w3], [s2w1, ...],...]
    def loss_of_sentences(self, sentences):
        assert len(sentences) > 0
        assert len(sentences[0]) > 1

        # transpose it to get
        batches_of_chars = list(map(list, zip(*sentences)))
        self._renew_cg()

        W = dy.parameter(self.W)
        b = dy.parameter(self.b)
        losses = []

        s = self.lstm.initial_state()
        # zip chops of last element
        for char, next_char in zip(batches_of_chars, batches_of_chars[1:]):
            # print(char, next_char)
            embedding = dy.lookup_batch(self.embedding, char)
            s = s.add_input(embedding)
            probs = dy.softmax(W * s.output() + b)
            picked_probs = dy.pick_batch(probs, next_char)
            minus_log = -dy.log(picked_probs)
            loss = dy.sum_batches(minus_log)
            losses.append(loss)
        loss = dy.esum(losses)
        return loss

    # generate from model:
    def generate(self, begin_of_sentence=[0], length=100):
        assert len(begin_of_sentence) > 0

        def sample(probs):
            rnd = random.random()
            for i, p in enumerate(probs):
                rnd -= p
                if rnd <= 0: break
            return i

        # setup the sentence
        self._renew_cg()

        W = dy.parameter(self.W)
        b = dy.parameter(self.b)

        out = []
        out_prob = []

        s0 = self.lstm.initial_state()
        s0.output()
        s = s0.add_input(self.embedding[begin_of_sentence[0]])

        for i in range(1, length):
            probs_sym = dy.softmax(W * s.output() + b)
            probs = probs_sym.vec_value()

            if i < len(begin_of_sentence):
                next_char = begin_of_sentence[i]
            else:
                next_char = sample(probs)

            s = s.add_input(self.embedding[next_char])
            out.append(next_char)
            out_prob.append(probs)
        self._renew_cg()
        return out, out_prob

    def get_components(self):
        return self.embedding, self.lstm, self.W, self.b

    def restore_components(self, components):
        self.embedding, self.lstm, self.W, self.b = components