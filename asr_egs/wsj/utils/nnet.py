#!/usr/bin/env python

# Copyright 2017       Florian Metze     (Carnegie Mellon University)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


import sys, re, os, numpy, pipes, itertools, functools, struct
import tensorflow as tf
from utils.fileutils.kaldi import readScp, writeArk
from utils.fileutils.kaldi_io import *

def parse_arguments(arg_elements):
    args = {}
    arg_num = len(arg_elements) // 2
    for i in range(arg_num):
        try:
            key = arg_elements[2*i].replace("--","").replace("-", "_");
            args[key] = arg_elements[2*i+1]
        except:
            print("error: ",arg_elements[2*i].replace("--","").replace("-", "_"),arg_elements[2*i+1])
    return args

# this is from kaldi_io (https://github.com/kronos-cm/kaldi-io-for-python/blob/python_2_and_3/kaldi_io.py)

if sys.version_info[0] > 2:
    def str_or_bytes(bytes_string):
        return bytes_string.decode('utf-8') if isinstance(bytes_string, bytes) else bytes_string.encode('utf-8')
else:
    def str_or_bytes(bytes_string):
        return bytes_string

IS_BIN = str_or_bytes('\x0b')
IS_EOL = str_or_bytes('\x04')
IS_SPACE = str_or_bytes(' ')
IS_EMPTY = str_or_bytes('')
FLOAT_VEC = str_or_bytes('FV ')
FLOAT_MAT = str_or_bytes('FM ')
DOUBLE_VEC = str_or_bytes('DV ')
DOUBLE_MAT = str_or_bytes('DM ')

def write_mat_stdout(m, key=IS_EMPTY):
    """ write_mat_stdout(m, key=IS_EMPTY)
    Write a binary kaldi matrix to stdout. Supports 32bit and 64bit floats.
    Arguments:
    m: the matrix to be stored,
    key (optional): used for writing ark-file, the utterance-id gets written before the matrix.
    """
    try:
        #
        #m=numpy.roll(m,-1,axis=0)
        #
        if str_or_bytes(key) != IS_EMPTY: sys.stdout.write(str_or_bytes(key)+IS_SPACE)
        sys.stdout.write(str_or_bytes('\x00'+'B'))  # we write binary!
        # Data-type,
        if   m.dtype == 'float32': sys.stdout.write(FLOAT_MAT)
        elif m.dtype == 'float64': sys.stdout.write(DOUBLE_MAT)
        else: raise MatrixDataTypeError
        # Dims,
        sys.stdout.write(IS_EOL)
        sys.stdout.write(struct.pack('I',m.shape[0]))  # rows
        sys.stdout.write(IS_EOL)
        sys.stdout.write(struct.pack('I',m.shape[1]))  # cols
        # Data,
        sys.stdout.write(m.tobytes())
    finally:
        pass

# end

def load_prior(prior_path,blank_scale=1.0,noise_scale=1.0):
    prior = None
    with open(prior_path, "r") as f:
        for line in f:
            counts = list(map(int, line.split(" ")[1:-1]))
            #counts = parts[1:]
            #counts.append(parts[0])
            #counts = parts
            counts[0] /= blank_scale
            for i in range(1,8):
                counts[i] /= noise_scale
            cnt_sum = functools.reduce(lambda x, y: x + y, counts)
            prior = [float(x) / cnt_sum for x in counts]
    return prior

def load_scale(prior_path):
    prior = None
    with open(prior_path, "r") as f:
        for line in f:
            parts = map(int, line.split(" ")[1:-1])
            #counts = parts[1:]
            #counts.append(parts[0])
            #counts = parts
            #cnt_sum = reduce(lambda x, y: x + y, counts)
            #prior = [float(x) / cnt_sum for x in counts]
    return parts

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = numpy.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - numpy.expand_dims(numpy.max(y, axis = axis), axis)
    
    # exponentiate y
    y = numpy.exp(y)

    # take the sum along the specified axis
    ax_sum = numpy.expand_dims(numpy.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


if __name__ == '__main__':

    """
    Python script to smooth the output of a CTC network. Parameters:
    ------------------
    """

    # parse arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    #print(arg_elements,len(arg_elements),file=sys.stderr)
    arguments = parse_arguments(arg_elements)

    # these arguments are mandatory
    counts='label.counts'
    try:
        counts=arguments['label_counts']
    except:
        pass
    #print("error with label_counts", file=sys.stderr)
    scales='label.scales'
    try:
        scales=arguments['class_scale']
    except:
        pass
    #print("error with class_scale", file=sys.stderr)
    temp=1.0
    try:
        temp=float(arguments['temperature'])
    except:
        pass
    #print("error with temp", file=sys.stderr)
    blank_scale=1.0
    try:
        blank_scale=float(arguments['blank_scale'])
    except:
        pass
    #print("error with blank_scale", file=sys.stderr)
    noise_scale=1.0
    try:
        noise_scale=float(arguments['noise_scale'])
    except:
        pass
    #print("error with noise_scale", file=sys.stderr)

    prior = numpy.array(load_prior(counts,blank_scale=blank_scale,noise_scale=noise_scale), dtype=numpy.float32)
    p = tf.convert_to_tensor(prior)
    #print(prior, prior.shape)
    try:
        scale = numpy.array(load_scale(scales))
        #print(scale, scale.shape)
    except:
        pass
    #print("not loading scale",file=sys.stderr)

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.07)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
    with sess.as_default():

        if True:
            # this uses Yun's code for reading
            features, uttids = readScp("/dev/stdin")
            #of = []
            #ui = []

            for (key,mat) in zip(uttids, features):
                softmax_prob = tf.nn.softmax(mat * temp, dim=-1, name=None)
                log_softmax_prob = tf.log(softmax_prob)
                log_likelihood = log_softmax_prob - tf.log(p)

                #of.append(log_likelihood.eval())
                #ui.append(key)
                out = log_likelihood.eval()
                #print("HI",type(out),type(key),file=sys.stderr)
                write_mat_stdout(out,key=key)

        else:
            # this program acts like a filter
            for (key,mat) in kaldi_io.read_mat_scp("-"):
                #mat = mat / scale[None,:]
                #m = softmax(mat,theta=temp,axis=1)
                #out = m / prior[None,:]
                #tran_logit = mat * float(temp)
                softmax_prob = tf.nn.softmax(mat * temp, dim=-1, name=None)
                log_softmax_prob = tf.log(softmax_prob)
                log_likelihood = log_softmax_prob - tf.log(p)

                out = log_likelihood.eval()
                kaldi_io.write_mat(sys.stdout,out,key=key)

    #writeArk("/dev/stdout", of, ui, tell=False)
