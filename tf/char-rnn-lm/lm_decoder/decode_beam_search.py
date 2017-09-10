import numpy as np
from itertools import groupby
from collections import namedtuple
import math
import lm_constants


class BeamSearch:

    def __init__(self, config):
        print("construction")
        self.__blank_id = config[lm_constants.CONFIG_TAGS_TEST.BLANK_ID]

    # log_ctc_prob[time][chars]
    def decode(self, log_ctc_prob):

        CandidateEntry = namedtuple('CandidateEntry', ['sequence', 'log_prob_blank', 'log_prob_nonblank'], verbose=False)
        #get_lm_prob, char_to_int, insertionBonus, lmWeight, beamSize, trie, expansion_chars=[]):
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
                log_prob_blank = candidate_entry.log_prob + log_ctc_prob[t][self.__blank_id]
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
                    if (len(candidate_entry.sequence) == 0):
                        continue
                    # pdb.set_trace()
                    u_t_s = upper_to_string(candidate_entry.sequence)
                    try:
                        word = u_t_s[len(u_t_s) - u_t_s[::-1].index(u' '):]
                    except ValueError:
                        word = u_t_s[:]
                    add_space = trie.search(word)
                    if (not add_space):
                        continue
                    log_lm = math.log(get_lm_prob(char, u_t_s)) * lmWeight
                    # pdb.set_trace()
                    log_prob_nonblank = candidate_entry.log_prob + log_lm
                    log_prob_blank = LOG_ZERO
                    candidates.append(
                        CandidateEntry(candidate_entry.sequence + [char], log_prob_blank, log_prob_nonblank))

            # TODO: break if we cannot get better than the worst element in the queue
            for candidate_entry in beam:
                # print(heap_entry.key)
                # handle other chars
                for c in char_to_int.keys():
                    if c == BLANK:
                        continue
                    u_t_s = upper_to_string(candidate_entry.sequence)
                    try:
                        l_word = u_t_s[len(u_t_s) - u_t_s[::-1].index(u' '):]
                    except ValueError:
                        l_word = u_t_s[:]
                    l_word.append(c)
                    in_vocab = trie.startsWith(l_word)
                    if (not in_vocab):
                        continue

                    # we keep the upper case character in the string, but add a whitespace for thel charLM
                    log_lm = math.log(get_lm_prob(c, u_t_s)) * lmWeight * 0.5
                    # normal case
                    ctc_log_prob = log_ctc_prob[t][char_to_int[c]]
                    if (len(candidate_entry.sequence) == 0) or (not c == candidate_entry.sequence[-1]):
                        log_prob_nonblank = candidate_entry.log_prob + log_lm + ctc_log_prob


                    else:
                        # double character, so we need an epsilon before
                        log_prob_nonblank = candidate_entry.log_prob_blank + log_lm + ctc_log_prob

                    # log prob blank is zero because we consider a new character and NO blank
                    log_prob_blank = LOG_ZERO

                    candidates.append(
                        CandidateEntry(candidate_entry.sequence + [c], log_prob_blank, log_prob_nonblank))

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
                if i + 1 == len(candidates):
                    beam_entry = get_beam_entry(candidate_entry, insertionBonus, lmWeight)
                    beam.append(beam_entry)

            candidates = []

            beam.sort(key=lambda x: -x.key)

            # apply beam
            beam = beam[:beamSize]
            # pdb.set_trace()

        return [upper_to_string(x.sequence) for x in beam]

    def decode(self, log_ctc_prob):
        # to make if comparision shorter
        decoded_seq = []

        for t in range(len(log_ctc_prob)):
            arg_max = np.argmax(log_ctc_prob[t])
            decoded_seq.append(arg_max)
        ids = [x[0] for x in groupby(decoded_seq)]
        cleaned_decoded_seq = [x for x in filter(lambda x: x != self.__blank_id, ids)]

        return cleaned_decoded_seq