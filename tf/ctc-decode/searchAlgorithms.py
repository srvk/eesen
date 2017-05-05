from collections import namedtuple
import math
import numpy as np
from itertools import groupby

BLANK = "<eps>"
SPACE = " "
EOS = "</s>"
BLANK_ID = 0

LOG_ZERO = -999999999.0
LOG_ZERO_THRESHOLD = -100.0
LOG_ONE = 0.0

BeamEntry = namedtuple('HeapEntry', ['key', 'string', 'log_prob', 'log_prob_blank', 'log_prob_nonblank',''], verbose=False)
CandidateEntry = namedtuple('CandidateEntry', ['string', 'log_prob_blank', 'log_prob_nonblank'], verbose=False)


# ugly as ****
def upper_to_string(str):
    #string_ar = [" " + c.lower() if c.isupper() else c for c in str]
    #string = "".join(string_ar)
    # remove multiple whitespaces, also splits whitespaces in the beginning and end
    #string = ' '.join(string.split())
    # add whitespace at the end
    #if len(str) > 0 and str[-1] == " ":
    #    string += " "
    #string = string.lower()
    #assert not string.isupper()
    return string.lower()


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
    log_insertion_bonus = -math.log(insertion_bonus) * len(upper_to_string(candidate_entry.string)) * lm_weight
    key = log_prob + log_insertion_bonus

    return BeamEntry(key, candidate_entry.string, log_prob,
                     candidate_entry.log_prob_blank, candidate_entry.log_prob_nonblank)


# log_ctc_prob[time][chars]
def beam_search(log_ctc_prob, get_lm_prob, char_to_int, insertionBonus, lmWeight, beamSize, merge=True, expansion_chars=[]):

    beam = []
    candidates = []

    # start element
    candidate_entry = CandidateEntry("", LOG_ONE, LOG_ZERO)
    beam.append(get_beam_entry(candidate_entry, insertionBonus, lmWeight))

    # for each time step
    for t in range(len(log_ctc_prob)):
        for candidate_entry in beam:
            # Handle the case where string stays the same
            # (1) blank symbol
            log_prob_blank = candidate_entry.log_prob + log_ctc_prob[t][BLANK_ID]
            # (2) repeat character
            log_prob_nonblank = LOG_ZERO
            if len(candidate_entry.string) > 0 and candidate_entry.string[-1] != SPACE:
                char = candidate_entry.string[-1]
                if char not in char_to_int.keys():
                    print(candidate_entry.string)
                char_id = char_to_int[char]
                log_prob_nonblank = candidate_entry.log_prob + log_ctc_prob[t][char_id]
            candidates.append(CandidateEntry(candidate_entry.string, log_prob_blank, log_prob_nonblank))

        # expand string with a space
        for char in expansion_chars:
            for candidate_entry in beam:
                log_lm = math.log(get_lm_prob(char, upper_to_string(candidate_entry.string))) * lmWeight
                log_prob_nonblank = candidate_entry.log_prob + log_lm
                log_prob_blank = LOG_ZERO
                candidates.append(CandidateEntry(candidate_entry.string + char, log_prob_blank, log_prob_nonblank))

        # TODO: break if we cannot get better than the worst element in the queue
        for candidate_entry in beam:
            # print(heap_entry.key)
            # handle other chars
            for c in char_to_int.keys():
                if c == BLANK:
                    continue
                # we keep the upper case character in the string, but add a whitespace for the charLM
                lm_string = " " + c.lower() if c.isupper() else c
                log_lm = LOG_ONE
                for i, lm_char in enumerate(lm_string):
                    # we have to add the new part too!
                    context = candidate_entry.string + lm_string[:i]
                    log_lm += math.log(get_lm_prob(lm_char, upper_to_string(context))) * lmWeight
                # normal case
                ctc_log_prob = log_ctc_prob[t][char_to_int[c]]
                if (len(candidate_entry.string) == 0) or (not c == candidate_entry.string[-1]):
                    log_prob_nonblank = candidate_entry.log_prob + log_lm + ctc_log_prob
                else:
                    # double character, so we need an epsilon before
                    log_prob_nonblank = candidate_entry.log_prob_blank + log_lm + ctc_log_prob

                # log prob blank is zero because we consider a new character and NO blank
                log_prob_blank = LOG_ZERO
                candidates.append(CandidateEntry(candidate_entry.string + c, log_prob_blank, log_prob_nonblank))

        # reset beam
        beam = []

        # merge same beam entries
        assert len(candidates) > 0

        candidates.sort(key=lambda x: x.string)

        log_prob_blank = LOG_ZERO
        log_prob_nonblank = LOG_ZERO
        string = candidates[0].string

        for i, candidate_entry in enumerate(candidates):
            if candidate_entry.string != string or not merge:
                beam_entry = get_beam_entry(CandidateEntry(string, log_prob_blank, log_prob_nonblank),
                                            insertionBonus, lmWeight)
                beam.append(beam_entry)
                string = candidate_entry.string
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

    # we are done, just add end of sentence character
    for idx in range(len(beam)):
        e = beam[idx]
        log_lm = math.log(get_lm_prob(EOS, upper_to_string(e.string))) * lmWeight
        beam[idx] = BeamEntry(e.key + log_lm, e.string, e.log_prob, e.log_prob_blank, e.log_prob_nonblank)
    beam.sort(key=lambda x: -x.key)

    # only return the strings
    return [upper_to_string(x.string) for x in beam]


def greedy_search(log_ctc_prob, int_to_char):
    # to make if comparision shorter
    ids = []

    for t in range(len(log_ctc_prob)):
        arg_max = np.argmax(log_ctc_prob[t])
        ids.append(arg_max)
    ids = [x[0] for x in groupby(ids)]
    string_ar = [int_to_char.get(x, "") for x in filter(lambda x: x != BLANK_ID, ids)]

    return upper_to_string(string_ar)
