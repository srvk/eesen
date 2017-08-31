def decode(mat):
    # sanity check
    check_sum = 0.0
    for x in mat[0]:
        check_sum += math.exp(x)
    # pdb.set_trace()
    assert abs(1.0 - check_sum) < 0.01

    beam = beam_search(mat, lm_function, ch_to_id, config['insertionBonus'], config['lmWeight'], config['beamSize'], trie, expansion_chars=expansion_characters)
    return beam
