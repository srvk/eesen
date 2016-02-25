// featbin/copy-feats.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2015  Carnegie Mellon University (author: Hao Zhang)

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
#include "cpucompute/matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace eesen;

    const char *usage =
        "Get energy from mfcc features (in which --use-energy=true)\n"
        "Usage: get-energy-from-mfcc [options] (<in-rspecifier> <out-wspecifier> | <in-rxfilename> <out-wxfilename>)\n"
        "e.g.: get-energy-from-mfcc ark:- ark,scp:foo.ark,foo.scp\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Binary-mode output (not relevant if writing "
                "to archive)");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0;
   
    // Copying tables of features.
    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    BaseFloatVectorWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    for (; !kaldi_reader.Done(); kaldi_reader.Next(), num_done++){
	const Matrix<BaseFloat> &feats = kaldi_reader.Value();
        Vector<BaseFloat> energy = Vector<BaseFloat>(feats.NumRows());
	//energy.Resize(feats.NumRows());
        energy.CopyColFromMat(feats,0);	
	kaldi_writer.Write(kaldi_reader.Key(),energy);
    }
    KALDI_LOG << "Copied " << num_done << " feature matrices.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
