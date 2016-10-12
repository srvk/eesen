// featbin/get-spkve-feat.cc

// Copyright 2014  Yajie Miao   Carnegie Mellon University

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
	"Usuage: get-energy-from-ali  ali2phone-rspecifier weightwrespecifier  SILint\n"; 
    // get-energy-statistics  scp:$scp $logdir/spk2utt.JOB $statisticdir
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ali2phone_rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    int32 SIL = atoi(po.GetArg(3).c_str());

    SequentialInt32VectorReader ali2phone_reader(ali2phone_rspecifier);
    BaseFloatVectorWriter kaldi_writer(wspecifier);
    //BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    for (; !ali2phone_reader.Done(); ali2phone_reader.Next()) {
	std::string utt = ali2phone_reader.Key();
	const std::vector<int32> &phone_seq = ali2phone_reader.Value();
	Vector<BaseFloat> weight(phone_seq.size(),kSetZero);
	BaseFloat *weight_data = weight.Data();
	for (MatrixIndexT index=0;index<weight.Dim();index++){
            if(phone_seq[index] != SIL){
		    weight_data[index] = 1;
	    }
	}
	kaldi_writer.Write(utt,weight);
    }
    return 0; 
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  
  }
}

