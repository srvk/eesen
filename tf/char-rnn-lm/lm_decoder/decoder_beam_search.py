import pdb
import numpy as np
from itertools import groupby
from collections import namedtuple
from lm_util import LmUtil

import math
import lm_constants
BLANK = u'<eps>'
BLANK_ID = 0
LOG_ZERO = -999999999.0
LOG_ZERO_THRESHOLD = -100.0
LOG_ONE = 0.0

class BeamSearch:

    def __init__(self, config):

        self.__lm_util = LmUtil(config)
        self.__insertion_bonus = config[lm_constants.CONFIG_TAGS_TEST.INSERTION_BONUS]
        self.__lm_weight = config[lm_constants.CONFIG_TAGS_TEST.LM_WEIGHT]
        self.__beam_size = config[lm_constants.CONFIG_TAGS_TEST.BEAM_SIZE]

        self.__beam_entry = namedtuple('HeapEntry', ['key', 'sequence', 'log_prob', 'log_prob_blank', 'log_prob_nonblank'], verbose=False)
        self.__candidate_entry = namedtuple('CandidateEntry', ['sequence', 'log_prob_blank', 'log_prob_nonblank'], verbose=False)

        self.__expansion_characters=[" "]

    # log_ctc_prob[time][chars]
    #TODO expansion caracters are comas and what ever you need to use
    #TODO candidate entryis is just names
    #TODO log_get_prob

    def decode(self, log_ctc_prob, char_to_int, trie):

        beam = []
        candidates = []
        # pdb.set_trace()

        # start element
        candidate_entry = self.__candidate_entry([], LOG_ONE, LOG_ZERO)
        beam.append(self.get_beam_entry(candidate_entry, self.__insertion_bonus, self.__lm_weight))

        # for each time step
        for t in range(len(log_ctc_prob)):
            #several sequence of characters
            #get the probability of space given the character
            for candidate_entry in beam:
                # Handle the case where string stays the same
                # (1) blank symbol
                log_prob_blank = candidate_entry.log_prob + log_ctc_prob[t][lm_constants.IDS.BLANK_ID]

                # (2) repeat character
                log_prob_nonblank = LOG_ZERO
                if len(candidate_entry.sequence) > 0 and candidate_entry.sequence[-1] not in self.__expansion_characters:
                    char = candidate_entry.sequence[-1]

                    if char not in char_to_int.keys():
                        print(candidate_entry.sequence)

                    char_id = char_to_int[char]

                    log_prob_nonblank = candidate_entry.log_prob + log_ctc_prob[t][char_id]
                candidates.append(self.__candidate_entry(candidate_entry.sequence, log_prob_blank, log_prob_nonblank))
            # this is space essentially
            for char in self.__expansion_characters:
                for candidate_entry in beam:
                    if (len(candidate_entry.sequence) == 0):
                        continue
                    # pdb.set_trace()
                    u_t_s = self.upper_to_string(candidate_entry.sequence)
                    try:
                        word = u_t_s[len(u_t_s) - u_t_s[::-1].index(u' '):]
                    except ValueError:
                        word = u_t_s[:]

                    #if the expansion character is possible (after the previous sequence of beam)
                    #here we search if the word exists or is incomplete
                    add_space = trie.search(word)
                    if (not add_space):
                        continue
                    weighted_log_lm = math.log(self.__lm_util.get_prob(char, u_t_s)) * self.__lm_weight
                    # pdb.set_trace()
                    log_prob_nonblank = candidate_entry.log_prob + weighted_log_lm
                    log_prob_blank = LOG_ZERO
                    candidates.append(
                        self.__candidate_entry(candidate_entry.sequence + [char], log_prob_blank, log_prob_nonblank))
            # TODO: break if we cannot get better than the worst element in the queue
            for candidate_entry in beam:
                # print(heap_entry.key)
                # handle other chars
                #TODO check char to int
                #you are iterating over all units of the AM
                for c in char_to_int.keys():
                    if c == BLANK:
                        continue
                    u_t_s = self.upper_to_string(candidate_entry.sequence)
                    try:
                        l_word = u_t_s[len(u_t_s) - u_t_s[::-1].index(u' '):]
                    except ValueError:
                        l_word = u_t_s[:]

                    #check if it exist
                    l_word.append(c)
                    in_vocab = trie.startsWith(l_word)
                    if (not in_vocab):
                        continue

                    #we get the LM probability of adding this character given the seq
                    weighted_log_lm = math.log(self.__lm_util.get_prob(c, u_t_s)) * self.__lm_weight * 0.5
                    # get the AM probability
                    ctc_log_prob = log_ctc_prob[t][char_to_int[c]]

                    #if the last character in the sequence is not the same as chatacter added
                    if (len(candidate_entry.sequence) == 0) or (not c == candidate_entry.sequence[-1]):
                        #candidate is one candidate in the beam
                        #we are adding normally
                        #log_prob: the total probability
                        log_prob_nonblank = candidate_entry.log_prob + weighted_log_lm + ctc_log_prob

                    else:

                        # double character, so we need an epsilon before
                        #log_prob_blank: probability of blank being present at the end of the sequence in the candidate entry
                        log_prob_nonblank = candidate_entry.log_prob_blank + weighted_log_lm + ctc_log_prob

                    # log prob blank is zero because we consider a new character and NO blank
                    log_prob_blank = LOG_ZERO

                    candidates.append(
                        self.__candidate_entry(candidate_entry.sequence + [c], log_prob_blank, log_prob_nonblank))

            # reset beam
            beam = []
            # merge same beam entries
            assert len(candidates) > 0
            # pdb.set_trace()
            candidates.sort(key=lambda x: str(x.sequence))

            log_prob_blank = LOG_ZERO
            log_prob_nonblank = LOG_ZERO
            string = candidates[0].sequence

            for i, candidate_entry in enumerate(candidates):
                if candidate_entry.sequence != string:

                    #creating a candidate entry
                    #lm_weigh and insertion bouns are just contants added to this tupple
                    beam_entry = self.get_beam_entry(self.__candidate_entry(string, log_prob_blank, log_prob_nonblank),
                                                self.__insertion_bonus, self.__lm_weight)

                    #adding each candidate to final list (that will be rescored and cut)
                    beam.append(beam_entry)
                    string = candidate_entry.sequence
                    log_prob_blank = LOG_ZERO
                    log_prob_nonblank = LOG_ZERO
                log_prob_blank = self.add_log_prob(log_prob_blank, candidate_entry.log_prob_blank)
                log_prob_nonblank = self.add_log_prob(log_prob_nonblank, candidate_entry.log_prob_nonblank)
                # do not forget the last one
                if i + 1 == len(candidates):
                    beam_entry = self.get_beam_entry(candidate_entry, self.__insertion_bonus, self.__lm_weight)
                    beam.append(beam_entry)

            candidates = []

            beam.sort(key=lambda x: -x.key)

            # apply beam
            #here we just take a beam from the whole graph
            beam = beam[:self.__beam_size]
        return [self.upper_to_string(x.sequence) for x in beam]

    #this is the squashing function
    def upper_to_string(self, str):
        if (len(str) == 0):
            return str
        temp = list()
        flag = 0
        for i in range(len(str)):
            if (flag == 1):
                if (str[i] == ' '):
                    continue
                else:
                    flag = 0
                    temp.append(str[i])
            else:
                if (str[i] == ' '):
                    temp.append(str[i])
                    flag = 1
                else:
                    temp.append(str[i])
        # print 'String', str
        # print 'TEMP', temp
        if (temp[0] == ' '):
            return temp[1:]
        else:
            return temp

    def add_log_prob(self, log_x, log_y):
        # handle 0 probabilities
        if log_x <= LOG_ZERO:
            return log_y
        if log_y <= LOG_ZERO:
            return log_x
        # avoid overflow in exp function
        if (log_y - log_x) > 0.0:
            log_y, log_x = log_x, log_y

        # https://en.wikipedia.org/wiki/Log_probability#Addition_in_log_space
        return log_x + math.log(1 + math.exp(log_y - log_x))

    def get_beam_entry(self, candidate_entry, insertion_bonus, lm_weight):
        log_prob = self.add_log_prob(candidate_entry.log_prob_blank, candidate_entry.log_prob_nonblank)
        log_insertion_bonus = -math.log(insertion_bonus) * len(self.upper_to_string(candidate_entry.sequence)) * lm_weight
        key = log_prob + log_insertion_bonus
        return self.__beam_entry(key, candidate_entry.sequence, log_prob, candidate_entry.log_prob_blank, candidate_entry.log_prob_nonblank)