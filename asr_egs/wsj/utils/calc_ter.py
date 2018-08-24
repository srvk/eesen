#!/usr/bin/env python

# Copyright 2018       Eric Fosler-Lussier     (The Ohio State University)

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

# This script takes an output logit/posterior file and computes a greedy CTC decoding.
# It then calculates the levenshtein distance to a label file and puts out the number of erros.

import sys, re, os, numpy, pipes, itertools, functools, struct
from utils.fileutils.kaldi import *
from utils.fileutils.kaldi_io import *
from utils.fileutils.smart_open import smart_open
import itertools
import editdistance

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

def greedy_decode(logits):
    return [i for i,dummy in itertools.groupby(logits.argmax(1)) if i>0]

# based on code from kaldi.py, but creates a generator to process one utterance at a time without storing
def readArkOnline(filename, limit = numpy.inf):
    """
    Reads the features in a Kaldi ark file.
    Returns a list of feature matrices and a list of the utterance IDs.
    """
    count=0
    with smart_open(filename, "rb") as f:
        while True:
            try:
                uttid = readString(f)
            except ValueError:
                break
            feature = readMatrix(f)
            yield feature,uttid
            count+=1
            if count == limit: break


if __name__ == '__main__':

    """
    Python script to calculate token error rate (TER) of a  CTC network. Parameters:
      labels: label file
      ark: archive with logits or posteriors
    ------------------
    """

    # parse arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)

    argerror=False

    # these arguments are mandatory
    verbose=False
    try:
        verbose=arguments['verbose'].lower()=='true'
    except:
        pass

    labels=''
    try:
        labels=arguments['labels']
    except:
        print("labels argument required", file=sys.stderr)
        argerror=True

    ark=''
    try:
        ark=arguments['ark']
    except:
        print("ark argument required", file=sys.stderr)
        argerror=True

    if argerror:
        exit(1)

    ##
    ## Now read in the labels

    uttlabels={}

    with open(labels,'r') as f:
        for line in f:
            tmp=line.rstrip().split()
            uttid=tmp.pop(0)
            uttlabels[uttid]=tmp

    # process the ark file
    total_errors=0
    total_tokens=0

    arkReader=readArkOnline(ark)

    for logits,uttid in arkReader:
        decode=greedy_decode(logits)
        labels=[int(l) for l in uttlabels[uttid]]

        if verbose:
            print(' dec: '+str(decode))
            print(' lab: '+str(labels))

        errors=editdistance.eval(labels,decode)
        tokens=len(uttlabels[uttid])
        if verbose:
            if tokens>0:
                uttpct=100.0*float(errors)/float(tokens)
                print(f'{uttid:15} {errors:-3d} / {tokens:-3d} = {uttpct:.1f}%')
            else:
                print(f'{uttid:15} {errors:-3d} / {tokens:-3d} = INF%')
        total_errors+=errors
        total_tokens+=tokens

    ter=100*float(total_errors)/float(total_tokens)
    print(f'TER = {total_errors} / {total_tokens} = {ter:.1f}')

