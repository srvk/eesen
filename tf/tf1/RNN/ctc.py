import numpy
import theano, theano.tensor as T

def ladd(*x):
    # Add inputs in the log domain (i.e. exp, add, then log)
    M = reduce(T.maximum, x)
    S = M + T.log(sum(T.exp(a - T.switch(T.eq(M, -numpy.inf), 0, M)) for a in x))
    return T.switch(T.eq(M, -numpy.inf), -numpy.inf, S)     # I need to do "switch" again to avoid NaNs in the gradient

def ctc_cost(prob, mask, label):
    # Inputs (all symbolic):
    #   prob: 3D float array (BATCH_SIZE * STREAM_LEN * N_LABELS), output of the neural network
    #   mask: 2D int array (BATCH_SIZE * STREAM_LEN), in which:
    #       1 marks the start of a sequence;
    #       2 marks the end of a sequence;
    #       0 marks the inside of a sequence;
    #       -1 marks parts that are not in any sequence.
    #   label: 2D int array, each row of which is a label sequence, with blanks (0) inserted already.
    #       The label sequences are in the same order as in the mask.

    # Clever way to find out which label sequence each frame corresponds to
    # mask_head = (mask > 0) & (mask & 1 > 0)
    mask_head = T.eq(mask, 1)
    seq_id = (mask_head.ravel().cumsum() - 1).reshape(mask.shape)

    # Clever way to find out the index of the last element (i.e. length - 1) in each label sequence
    label_end = (label > 0).sum(axis = 1) * 2

    # Compute whether each symbol in the label sequences is the same as two symbols before
    # Useful in the computation of the alpha trellis
    # Somehow I have to cast it into float32, otherwise the gradient operator will "illegally output an integer-valued variable"
    leq = T.cast(T.eq(label[:, :-2], label[:, 2:]), theano.config.floatX)

    # Step function to compute the alphas
    rep = lambda x: T.extra_ops.repeat(x.reshape((-1, 1)), label.shape[1], axis = 1)
    alpha_init = T.log(T.eq(T.arange(label.shape[1]), 0)).astype(theano.config.floatX)
    def step_ctc(logP, mask_head, seq_id, alpha_tm1):
        # Inputs:
        #   prob: BATCH_SIZE * N_LABELS
        #   mask_head: BATCH_SIZE
        #   seq_id: BATCH_SIZE
        #   alpha_tm1: BATCH_SIZE * MAX_LABEL_SEQ_LEN
        # Depends on:
        #   label: N_SEQS * MAX_LABEL_SEQ_LEN
        # Outputs:
        #   alpha_t: BATCH_SIZE * MAX_LABEL_SEQ_LEN
        alpha0 = T.switch(rep(mask_head), alpha_init, alpha_tm1)
        alpha1 = T.concatenate([-numpy.inf * T.ones((alpha0.shape[0], 1)), alpha0[:, :-1]], axis = 1)
        alpha2 = T.concatenate([-numpy.inf * T.ones((alpha0.shape[0], 2)), \
                                T.switch(leq[seq_id], -numpy.inf, alpha0[:, :-2])], axis = 1)
        return ladd(alpha0, alpha1, alpha2) + logP[rep(T.arange(logP.shape[0])), label[seq_id, :]]

    # Compute the alphas
    alpha = theano.scan(
        fn = step_ctc,
        sequences = [T.log(prob).dimshuffle((1, 0, 2)), mask_head.T, seq_id.T],
        outputs_info = [T.zeros((mask.shape[0], label.shape[1]), dtype = theano.config.floatX)],
        non_sequences = [],
    )[0]

    # Pick out the alphas at the tail of each sequence, and form cost function
    # mask_tail = ((mask > 0) & (mask & 2 > 0)).nonzero()
    mask_tail = T.eq(mask, 2).nonzero()
    alpha = alpha.dimshuffle((1, 0, 2))[mask_tail]
    alpha_end = alpha[seq_id[mask_tail], label_end]
    alpha_penultimate = T.switch(T.eq(label_end, 0), -numpy.inf, alpha[seq_id[mask_tail], label_end - 1])
    return -ladd(alpha_end, alpha_penultimate).sum() / (mask >= 0).sum()    # Per-frame negative log-likelihood
