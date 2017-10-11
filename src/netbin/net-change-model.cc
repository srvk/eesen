// netbin/net-change-model.cc

// Copyright 2012-2015  Brno University of Technology (author: Karel Vesely)
//                2015  Yajie Miao
// Copyright 2017 Jayadev Billa

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "net/net.h"

int main(int argc, char *argv[]) {
  try {
    using namespace eesen;
    typedef eesen::int32 int32;

    const char *usage =
        "Change network model with specified options and possibly change binary/text format\n"
        "Note: Model options must be fully specified, options will default to false otherwise\n"
        "Usage:  net-change-model [options] <model-in> <model-out>\n"
        "e.g.:\n"
        " net-change-model --binary=false in.nnet out.nnet\n";


    bool binary_write = true;

    float forwarddrop   = 0.0;
    bool forwardstep    = false;
    bool forwardseq     = false;

    bool rnndrop        = false;
    bool nmldrop        = false;
    float recurrentdrop = 0.0;
    bool recurrentstep  = false;
    bool recurrentseq   = false;

    bool twiddleforward = false;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("forwarddrop", &forwarddrop, "Forward dropout factor (default: 0.0 -- disabled)");
    po.Register("forwardstep", &forwardstep, "Change forward dropout mask every time step (default: false)");
    po.Register("forwardseq",  &forwardseq,  "Change forward dropout mask every sequence (default: false)");
    po.Register("rnndrop",     &rnndrop,     "Apply RNNDrop on recurrent connections (default: false)");
    po.Register("nmldrop",     &nmldrop,     "Apply no memory loss (NML) dropout on recurrent connections (default: false)");
    po.Register("recurrentdrop",  &recurrentdrop, "Recurrent dropout factor (default: 0.0 -- disabled)");
    po.Register("recurrentstep",  &recurrentstep, "Change recurrent dropout mask every time step (default: false)");
    po.Register("recurrentseq",   &recurrentseq,  "Change recurrent dropout mask every sequence (default: false)");
    po.Register("twiddleforward", &twiddleforward, "Apply stochastic dropout combination (default: false)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    // Load the network
    Net net;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      net.Read(ki.Stream(), binary_read);
    }

    net.ChangeDropoutParameters(forwarddrop,forwardstep,forwardseq,rnndrop,nmldrop,
                                recurrentdrop,recurrentstep,recurrentseq,twiddleforward);

    // Store the network
    {
      Output ko(model_out_filename, binary_write);
      net.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

