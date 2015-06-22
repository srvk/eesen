#!/usr/bin/env python

# Copyright 2015       Yajie Miao    (Carnegie Mellon University)

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

# This python script writes the network topology if we use bi-directionary LSTM layers
# as the building blocks. 

import sys

def parse_arguments(arg_elements):
    args = {}
    arg_num = len(arg_elements) / 2
    for i in xrange(arg_num):
        key = arg_elements[2*i].replace("--","").replace("-", "_");
        args[key] = arg_elements[2*i+1]
    return args


if __name__ == '__main__':

    # parse arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)

    # these 4 arguments are mandatory
    input_feat_dim=int(arguments['input_feat_dim'])
    lstm_layer_num=int(arguments['lstm_layer_num'])
    lstm_cell_dim=int(arguments['lstm_cell_dim'])
    target_num=int(arguments['target_num'])

    # by default, the range of the parameters is set to 0.1; however, you can change it by specifying "--param-range"
    # this means for initialization, model parameters are drawn uniformly from the interval [-0.1, 0.1]
    param_range='0.1'
    if arguments.has_key('param_range'):
        param_range = arguments['param_range']

    print '<NnetProto>'
    lstm_comm = ' <ParamRange> ' + param_range + ' <LearnRateCoef> 1.0 <BiasLearnRateCoef> 1.0 <MaxGrad> 50.0'
    
    # the first layer takes input features
    print '<BiLstmParallel> <InputDim> ' + str(input_feat_dim) + ' <OutputDim> ' + str(2*lstm_cell_dim) + lstm_comm
    # the following bidirectional LSTM layers
    for n in range(1, lstm_layer_num):
         print '<BiLstmParallel> <InputDim> ' + str(2*lstm_cell_dim) + ' <OutputDim> ' + str(2*lstm_cell_dim) + lstm_comm
    # the final affine-transform and softmax layer
    print '<AffineTransform> <InputDim> ' + str(2*lstm_cell_dim) + ' <OutputDim> ' + str(target_num) + ' <ParamRange> ' + param_range
    print '<Softmax> <InputDim> ' + str(target_num) + ' <OutputDim> ' + str(target_num)
    print '</NnetProto>'
