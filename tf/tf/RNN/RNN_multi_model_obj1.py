import numpy
import theano, theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from ctc_align import ctc_cost
import sys, os.path
sys.path.append(os.path.expanduser("~/G/coconut"))
from fileutils import smart_open
import dill

relu = lambda x: 0.5 * (x + abs(x))
ACTIVATION = {
    "linear": lambda x: x,
    "sigmoid": T.nnet.sigmoid,
    "tanh": T.tanh,
    "relu": relu,
    "softmax": lambda x: T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape),
}

def zeros(shape):
    return numpy.zeros(shape, dtype = theano.config.floatX)

def full(shape, value):
    return numpy.full(shape, value, dtype = theano.config.floatX)

def rand_init(shape, activation):
    global rng
    limit = numpy.sqrt(6. / (shape[-2] + shape[-1]))
    if activation == "sigmoid":
        limit *= 4
    return rng.uniform(low = -limit, high = limit, size = shape).astype(theano.config.floatX)

class RNN(object):
    def __init__(self,
                 Nbranches = 1,             # number of branches (parallel models to be fused)
                 Nlayers = 1,               # number of layers
                 Ndirs = 1,                 # unidirectional or bidirectional
                 Nx = 100,                  # input size
                 Nh = 100,                  # hidden layer size
                 Ny = 100,                  # output size
                 Ah = "relu",               # hidden unit activation (e.g. relu, tanh, lstm)
                 Ay = "linear",             # output unit activation (e.g. linear, sigmoid, softmax)
                 predictPer = "frame",      # frame or sequence
                 loss = None,               # loss function (e.g. mse, ce, ce_group, hinge, squared_hinge)
                 L1reg = 0.0,               # L1 regularization
                 L2reg = 0.0,               # L2 regularization
                 multiReg = 0.0,            # regularization of agreement of predictions on data of different conditions
                 momentum = 0.0,            # SGD momentum
                 seed = 15213,              # random seed for initializing the weights
                 frontEnd = None,           # a lambda function for transforming the input
                 filename = None,           # initialize from file
                 initParams = None,         # initialize from given dict
                ):

        if filename is not None:            # load parameters from file
            with smart_open(filename, "rb") as f:
                initParams = dill.load(f)
        if initParams is not None:          # load parameters from given dict
            self.paramNames = []
            self.params = []
            for k, v in initParams.iteritems():
                if type(v) is numpy.ndarray:
                    self.addParam(k, v)
                else:
                    setattr(self, k, v)
                    self.paramNames.append(k)
            # F*ck, locals()[k] = v doesn't work; I have to do this statically
            Nbranches, Nlayers, Ndirs, Nx, Nh, Ny, Ah, Ay, predictPer, loss, L1reg, L2reg, momentum, frontEnd \
                = self.Nbranches, self.Nlayers, self.Ndirs, self.Nx, self.Nh, self.Ny, self.Ah, self.Ay, self.predictPer, self.loss, self.L1reg, self.L2reg, self.momentum, self.frontEnd
        else:                           # Initialize parameters randomly
            # Names of parameters to save to file
            self.paramNames = ["Nbranches", "Nlayers", "Ndirs", "Nx", "Nh", "Ny", "Ah", "Ay", "predictPer", "loss", "L1reg", "L2reg", "momentum", "frontEnd"]
            for name in self.paramNames:
                value = locals()[name]
                setattr(self, name, value)

            # Values of parameters for building the computational graph
            self.params = []

            # Initialize random number generators
            global rng
            rng = numpy.random.RandomState(seed)

            # Construct parameter matrices
            Nlstm = 4 if Ah == 'lstm' else 1
            self.addParam("Win", rand_init((Nbranches, Nx, Nh * Ndirs * Nlstm), Ah))
            self.addParam("Wrec", rand_init((Nbranches, Nlayers, Ndirs, Nh, Nh * Nlstm), Ah))
            self.addParam("Wup", rand_init((Nbranches, Nlayers - 1, Nh * Ndirs, Nh * Ndirs * Nlstm), Ah))
            self.addParam("Wout", rand_init((Nbranches, Nh * Ndirs, Ny), Ay))
            if Ah != "lstm":
                self.addParam("Bhid", zeros((Nbranches, Nlayers, Nh * Ndirs)))
            else:
                self.addParam("Bhid", numpy.tile(numpy.concatenate([full((Nbranches, Nlayers, Nh), 1.0),
                                                                    zeros((Nbranches, Nlayers, Nh * 3))], 2), (1, 1, Ndirs)))
            self.addParam("Bout", zeros((Nbranches, Ny)))
            self.addParam("h0", zeros((Nbranches, Nlayers, Ndirs, Nh)))
            if Ah == "lstm":
                self.addParam("c0", zeros((Nbranches, Nlayers, Ndirs, Nh)))

        # Compute total number of parameters
        self.nParams = sum(x.get_value().size for x in self.params)

        # Initialize gradient tensors when using momentum
        if momentum > 0:
            self.dparams = [theano.shared(zeros(x.get_value().shape)) for x in self.params]

        # Build computation graph
        input = T.ftensor3()
        mask = T.imatrix()
        mask_int = [(mask & 1).nonzero(), (mask & 2).nonzero()]
        mask_float = [T.cast((mask & 1).dimshuffle((1, 0)).reshape((mask.shape[1], mask.shape[0], 1)), theano.config.floatX),
                      T.cast(((mask & 2) / 2).dimshuffle((1, 0)).reshape((mask.shape[1], mask.shape[0], 1)), theano.config.floatX)]

        def step_rnn(x_t, mask, h_tm1, W, h0):
            h_tm1 = T.switch(mask, h0, h_tm1)
            return [ACTIVATION[Ah](x_t + h_tm1.dot(W))]

        def step_lstm(x_t, mask, c_tm1, h_tm1, W, c0, h0):
            c_tm1 = T.switch(mask, c0, c_tm1)
            h_tm1 = T.switch(mask, h0, h_tm1)
            a = x_t + h_tm1.dot(W)
            f_t = T.nnet.sigmoid(a[:, :Nh])
            i_t = T.nnet.sigmoid(a[:, Nh : Nh * 2])
            o_t = T.nnet.sigmoid(a[:, Nh * 2 : Nh * 3])
            c_t = T.tanh(a[:, Nh * 3:]) * i_t + c_tm1 * f_t
            h_t = T.tanh(c_t) * o_t
            return [c_t, h_t]

        x = input if frontEnd is None else frontEnd(input)
        outputs = []
        for k in range(Nbranches):
            for i in range(Nlayers):
                h = (x.dimshuffle((1, 0, 2)).dot(self.Win[k]) if i == 0 else h.dot(self.Wup[k, i-1])) + self.Bhid[k, i]
                rep = lambda x: T.extra_ops.repeat(x.reshape((1, -1)), h.shape[1], axis = 0)
                if Ah != "lstm":
                    h = T.concatenate([theano.scan(
                            fn = step_rnn,
                            sequences = [h[:, :, Nh * d : Nh * (d+1)], mask_float[d]],
                            outputs_info = [rep(self.h0[k, i, d])],
                            non_sequences = [self.Wrec[k, i, d], rep(self.h0[k, i, d])],
                            go_backwards = (d == 1),
                        )[0][::(1 if d == 0 else -1)] for d in range(Ndirs)], axis = 2)
                else:
                    h = T.concatenate([theano.scan(
                            fn = step_lstm,
                            sequences = [h[:, :, Nh * 4 * d : Nh * 4 * (d+1)], mask_float[d]],
                            outputs_info = [rep(self.c0[k, i, d]), rep(self.h0[k, i, d])],
                            non_sequences = [self.Wrec[k, i, d], rep(self.c0[k, i, d]), rep(self.h0[k, i, d])],
                            go_backwards = (d == 1),
                        )[0][1][::(1 if d == 0 else -1)] for d in range(Ndirs)], axis = 2)
            h = h.dimshuffle((1, 0, 2))
            if predictPer == "sequence":
                h = T.concatenate([h[mask_int[1 - d]][:, Nh * d : Nh * (d+1)] for d in range(Ndirs)], axis = 1)
            outputs.append(ACTIVATION[Ay](h.dot(self.Wout[k]) + self.Bout[k]))
        output = T.stack(*outputs)      # Deprecated in Theano 0.8 but accepted in Theano 0.7
        output_mean = output.mean(axis = 0)
        output_var = output.var(axis = 0)

        # Compute loss function
        if loss is None:
            loss = {"linear": "mse", "sigmoid": "ce", "softmax": "ce_group"}[self.Ay]
        if loss == "ctc":
            label = T.imatrix()
            label_time = T.imatrix()
            tol = T.iscalar()
            cost = ctc_cost(output_mean, mask, label, label_time, tol)
        else:
            if predictPer == "sequence":
                label = T.fmatrix()
                y = output_mean
                t = label
            elif predictPer == "frame":
                label = T.ftensor3()
                indices = (mask >= 0).nonzero()
                y = output_mean[indices]
                t = label[indices]
            cost = T.mean({
                "ce":               -T.mean(T.log(y) * t + T.log(1 - y) * (1 - t), axis = 1),
                "ce_group":         -T.log((y * t).sum(axis = 1)),
                "mse":              T.mean((y - t) ** 2, axis = 1),
                "hinge":            T.mean(relu(1 - y * (t * 2 - 1)), axis = 1),
                "squared_hinge":    T.mean(relu(1 - y * (t * 2 - 1)) ** 2, axis = 1),
            }[loss])

        # Add regularization
        cost += sum(abs(x).sum() for x in self.params) / self.nParams * L1reg
        cost += sum(T.sqr(x).sum() for x in self.params) / self.nParams * L2reg
        if predictPer == "sequence":
            cost += output_var.mean() * multiReg
        else:
            indices = (mask >= 0).nonzero()
            cost += output_var[indices].mean() * multiReg

        # Compute updates for network parameters
        updates = []
        lrate = T.fscalar()
        clip = T.fscalar()
        grad = T.grad(cost, self.params)
        grad_clipped = [T.maximum(T.minimum(g, clip), -clip) for g in grad]
        if momentum > 0:
            for w, d, g in zip(self.params, self.dparams, grad_clipped):
                updates.append((w, w + momentum * momentum * d - (1 + momentum) * lrate * g))
                updates.append((d, momentum * d - lrate * g))
        else:
            for w, g in zip(self.params, grad_clipped):
                updates.append((w, w - lrate * g))

        # Create functions to be called from outside
        if loss == "ctc":
            inputs = [input, mask, label, label_time, tol, lrate, clip]
        else:
            inputs = [input, mask, label, lrate, clip]
        self.train = theano.function(
                         inputs = inputs,
                         outputs = cost,
                         updates = updates,
                     )

        self.predict = theano.function(inputs = [input, mask], outputs = output)

    def addParam(self, name, value):
        value = theano.shared(value)
        setattr(self, name, value)
        self.params.append(value)
        self.paramNames.append(name)

    def getParams(self):
        params = {}
        for k in self.paramNames:
            v = getattr(self, k)
            try:
                if type(v) is theano.tensor.sharedvar.TensorSharedVariable or type(v) is theano.sandbox.cuda.var.CudaNdarraySharedVariable:
                            # ^ type of matrices when running on a CPU                   ^ type of matrices when running on a GPU
                    v = v.get_value()
            except Exception:
                pass
            params[k] = v
        return params

    def save(self, filename):
        with smart_open(filename, "wb") as f:
            dill.dump(self.getParams(), f)

    def savemat(self, filename):
        scipy.io.savemat(filename, self.getParams())
