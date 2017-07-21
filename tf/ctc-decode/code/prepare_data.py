from collections import Counter, defaultdict
from itertools import count
import io
from build_tree import *
import pdb
def prep_data(config):
    w2i = defaultdict(count(0).next)
    sos = u'<s>'
    eos = u'<s>'
    space = u' '
    units = io.open(config['lm_config']['units_file'], encoding='utf-8').readlines()
    w2i[sos]
    for i in units[8:]:
        # [8:]
        w2i[i.strip().split()[0]]
    w2i[eos]
    w2i[space]
    nwords = len(w2i)
    print("Nwords ",nwords)
    print(w2i)
    dict_w2i = dict(w2i)
    config['lm_config']['nwords'] = nwords
    config['w2i'] = dict_w2i
    config['eos'] = eos

    # trie = Trie()

    lexicons = io.open(config['lm_config']['lexicon_file'], encoding='utf-8').readlines()
    lex_dict = dict()
    for lexf in lexicons:
        # [13:]
        lex = lexf.strip().split(' ')
        lex_dict[lex[0]] = lex[1:]

    return config, lex_dict
