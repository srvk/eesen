// netbin/net-copy.cc

// Copyright     2015  Hang Su

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
        "Average network parameters over a number of nets\n"
        "Usage:  net-average [options] <model1> <model2> ... <model-out>\n"
        "e.g.:\n"
        " net-average --binary=false 1.nnet 2.nnet 3.nnet final.nnet\n";

    bool binary_write = true;
    bool sum =false;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("sum", &sum, "If true, perform sum instead of average");

    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model1_filename = po.GetArg(1),
        model_out_filename = po.GetArg(po.NumArgs());

    // Load the network
    Net net; 
    {
      bool binary_read;
      Input ki(model1_filename, &binary_read);
      net.Read(ki.Stream(), binary_read);
    }

    int32 num_inputs = po.NumArgs() - 1;
    BaseFloat scale = (sum ? 1.0 : 1.0 / num_inputs);
    net.Scale(scale);

    for (int32 i = 2; i <= num_inputs; i++) {
      bool binary_read;
      Input ki(po.GetArg(i), &binary_read);
      Net net_other;
      net_other.Read(ki.Stream(), binary_read);
      net.AddNet(scale, net_other);
    }

    // Store the network
    {
      Output ko(model_out_filename, binary_write);
      net.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Averaged parameters of " << num_inputs
              << " neural nets, and wrote to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

