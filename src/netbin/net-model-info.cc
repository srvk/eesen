// netbin/net-model-info.cc

// Copyright 2012-2015  Brno University of Technology (author: Karel Vesely)
//                2015  Yajie Miao
//                2017  Jayadev Billa (based on net-copy.cc)

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
        "Print information about network model\n"
        "Usage:  net-model-info <model>\n"
        "e.g.:\n"
        " net-model-info final.nnet\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string net_in_filename = po.GetArg(1);

    // Load the network
    Net net_in;
    KALDI_LOG << "Reading model model from " << net_in_filename;
    {
      bool binary_read;
      Input ki(net_in_filename, &binary_read);
      net_in.Read(ki.Stream(), binary_read);
    }
    //KALDI_LOG << "Pretrained model from " << net_in_filename;
    KALDI_LOG << net_in.Info();

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

