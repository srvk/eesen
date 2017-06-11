from collections import namedtuple
import math
import numpy as np
from itertools import groupby
import pdb
BLANK = u'<eps>'
BLANK_ID = 0
LOG_ZERO = -999999999.0
LOG_ZERO_THRESHOLD = -100.0
LOG_ONE = 0.0

BeamEntry = namedtuple('HeapEntry', ['key', 'sequence', 'log_prob', 'log_prob_blank', 'log_prob_nonblank'], verbose=False)
CandidateEntry = namedtuple('CandidateEntry', ['sequence', 'log_prob_blank', 'log_prob_nonblank'], verbose=False)


def upper_to_string(str):
    if(len(str)==0):
        return str
    temp = list()
    flag = 0
    for i in range(len(str)):
        if(flag == 1):
            if(str[i] == ' '):
                continue
            else:
                flag=0
                temp.append(str[i])
        else:
            if(str[i] == ' '):
                temp.append(str[i])
                flag = 1
            else:
                temp.append(str[i])
    # print 'String', str
    # print 'TEMP', temp
    if(temp[0] == ' '):
        return temp[1:]
    else:
        return temp

def add_log_prob(log_x, log_y):
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


def get_beam_entry(candidate_entry, insertion_bonus, lm_weight):
    log_prob = add_log_prob(candidate_entry.log_prob_blank, candidate_entry.log_prob_nonblank)
    log_insertion_bonus = -math.log(insertion_bonus) * len(upper_to_string(candidate_entry.sequence)) * lm_weight
    key = log_prob + log_insertion_bonus

    return BeamEntry(key, candidate_entry.sequence, log_prob,
                     candidate_entry.log_prob_blank, candidate_entry.log_prob_nonblank)

# log_ctc_prob[time][chars]
def beam_search(log_ctc_prob, get_lm_prob, char_to_int, insertionBonus, lmWeight, beamSize, trie, expansion_chars=[], blank_fudge=0.5):
    # pdb.set_trace()
    beam = []
    candidates = []
    # pdb.set_trace()

    # start element
    candidate_entry = CandidateEntry([], LOG_ONE, LOG_ZERO)
    beam.append(get_beam_entry(candidate_entry, insertionBonus, lmWeight))

    # for each time step
    for t in range(len(log_ctc_prob)):
        # print "DECODING ",t
        for candidate_entry in beam:
            # Handle the case where string stays the same
            # (1) blank symbol
            log_prob_blank = candidate_entry.log_prob + log_ctc_prob[t][BLANK_ID]
            # (2) repeat character
            log_prob_nonblank = LOG_ZERO
            if len(candidate_entry.sequence) > 0 and candidate_entry.sequence[-1] not in expansion_chars:
                char = candidate_entry.sequence[-1]
                if char not in char_to_int.keys():
                    print(candidate_entry.sequence)
                char_id = char_to_int[char]
                log_prob_nonblank = candidate_entry.log_prob + log_ctc_prob[t][char_id]
            candidates.append(CandidateEntry(candidate_entry.sequence, log_prob_blank, log_prob_nonblank))


        # expand string with a space
        for char in expansion_chars:
            for candidate_entry in beam:
                if(len(candidate_entry.sequence)==0):
                    continue
                # pdb.set_trace()
                u_t_s = upper_to_string(candidate_entry.sequence)
                word = u''.join(u_t_s).rsplit(u' ',1)[-1]
                add_space = trie.search(word)
                if(not add_space):
                    continue
                log_lm = math.log(get_lm_prob(char, u_t_s)) * lmWeight
                # pdb.set_trace()
                log_prob_nonblank = candidate_entry.log_prob + log_lm
                log_prob_blank = LOG_ZERO
                candidates.append(CandidateEntry(candidate_entry.sequence + [char], log_prob_blank, log_prob_nonblank))


        # TODO: break if we cannot get better than the worst element in the queue
        for candidate_entry in beam:
            # print(heap_entry.key)
            # handle other chars
            for c in char_to_int.keys():
                if c == BLANK:
                    continue
                u_t_s = upper_to_string(candidate_entry.sequence)
                l_word = u''.join(u_t_s).rsplit(u' ',1)[-1]
                in_vocab = trie.startsWith(l_word+c)
                if(not in_vocab):
                    continue

                # we keep the upper case character in the string, but add a whitespace for the charLM
                # the factor 0.5 is a HACK - look into this (Florian and Siddharth)
                log_lm = math.log(get_lm_prob(c, u_t_s)) * lmWeight * blank_fudge
                # normal case
                ctc_log_prob = log_ctc_prob[t][char_to_int[c]]
                if (len(candidate_entry.sequence) == 0) or (not c == candidate_entry.sequence[-1]):
                    log_prob_nonblank = candidate_entry.log_prob + log_lm + ctc_log_prob
                else:
                    # double character, so we need an epsilon before
                    log_prob_nonblank = candidate_entry.log_prob_blank + log_lm + ctc_log_prob

                # log prob blank is zero because we consider a new character and NO blank
                log_prob_blank = LOG_ZERO

                candidates.append(CandidateEntry(candidate_entry.sequence+[c], log_prob_blank, log_prob_nonblank))

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
            if candidate_entry.sequence != string :
                # pdb.set_trace()
                beam_entry = get_beam_entry(CandidateEntry(string, log_prob_blank, log_prob_nonblank),
                                            insertionBonus, lmWeight)
                beam.append(beam_entry)
                string = candidate_entry.sequence
                log_prob_blank = LOG_ZERO
                log_prob_nonblank = LOG_ZERO
            log_prob_blank = add_log_prob(log_prob_blank, candidate_entry.log_prob_blank)
            log_prob_nonblank = add_log_prob(log_prob_nonblank, candidate_entry.log_prob_nonblank)
            # do not forget the last one
            if i+1 == len(candidates):
                beam_entry = get_beam_entry(candidate_entry, insertionBonus, lmWeight)
                beam.append(beam_entry)

        candidates = []

        beam.sort(key=lambda x: -x.key)

        # apply beam
        beam = beam[:beamSize]
        # pdb.set_trace()

    return [upper_to_string(x.sequence) for x in beam]


def greedy_search(log_ctc_prob, int_to_char):
    # to make if comparision shorter
    ids = []

    for t in range(len(log_ctc_prob)):
        arg_max = np.argmax(log_ctc_prob[t])
        ids.append(arg_max)
    ids = [x[0] for x in groupby(ids)]
    string_ar = [int_to_char.get(x, "") for x in filter(lambda x: x != BLANK_ID, ids)]

    return upper_to_string(string_ar)

