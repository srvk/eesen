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
	"Usuage: get-energy-statistics  energy-rspecifier spk2utt-rspecifier statisticdir\n"; 
    // get-energy-statistics  scp:$scp $logdir/spk2utt.JOB $statisticdir
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string spk2utt_rspecifier = po.GetArg(2);
    std::string energy_rspecifier = po.GetArg(1);
    std::string outputdir = po.GetArg(3);

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessBaseFloatVectorReader feat_reader(energy_rspecifier);
    //BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    int64  num_utt_err = 0;
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
	std::string spk = spk2utt_reader.Key();
	//std::string filename = "123.txt";
	std::string filename = outputdir+"/"+spk;
	std::ofstream outputFile(filename.c_str(),std::ofstream::out);
        //std::string filename = outputdir+"/"+spk;
        //std::string filename = "123.txt";
	//outputFile.open(string(outputdir+"/"+spk));
	//outputFile.open(filename);
	const std::vector<std::string> &uttlist = spk2utt_reader.Value();
	if (uttlist.empty()) {
            KALDI_ERR << "Speaker with no utterances.";
	}
	for (size_t i = 0; i < uttlist.size(); i++) {
            std::string utt = uttlist[i];
	    if (!feat_reader.HasKey(utt)) {
	        KALDI_WARN << "No feature present in input for utterance " << utt;
		num_utt_err++;
	    } else{
	      const Vector<BaseFloat> &feats = feat_reader.Value(utt);
	      feats.Write(outputFile,false);
	    }
	}
	outputFile.flush();
	outputFile.close();
    }
    return 0; 
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  
  }
}

