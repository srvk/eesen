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
	"Usuage: get-weight-from-energy \n"; 
    // get-energy-statistics  scp:$scp $logdir/spk2utt.JOB $statisticdir
    ParseOptions po(usage);
    bool global = false;
    po.Register("global", &global, "Global threshold");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string spk2utt_rspecifier = po.GetArg(2);
    std::string energy_rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(3);

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    BaseFloatVectorWriter kaldi_writer(wspecifier);
    //BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    int64  num_utt_err = 0;
    if (!global){
      RandomAccessBaseFloatVectorReader feat_reader(energy_rspecifier);	     
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
  	std::string spk = spk2utt_reader.Key();
  	const std::vector<std::string> &uttlist = spk2utt_reader.Value();
  	std::vector<BaseFloat> buffer ;
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
  	      buffer.insert(buffer.end(),feats.Data(),feats.Data()+feats.Dim());
  	    }
  	}
  	//std::cout<<buffer.size()<<std::endl;
  	sort(buffer.begin(),buffer.end());
  	size_t threshold_index = buffer.size() * 0.2;
  	BaseFloat threshold = buffer[threshold_index];
  	//std::vector<BaseFloat>::iterator first = buffer.begin();
          //std::vector<BaseFloat>::iterator last = buffer.end();
  	//while(first!=last){
  	// std::cout << *first << " ";
  	// first++;
  	//}
          
          for (size_t i = 0; i < uttlist.size(); i++) {
  	    std::string utt = uttlist[i];	
  	    if (!feat_reader.HasKey(utt)) {
  		KALDI_WARN << "No feature present in input for utterance " << utt;
  		num_utt_err++;
  	    } else{
  	      const Vector<BaseFloat> &feats = feat_reader.Value(utt);
  	      const BaseFloat *feat_data = feats.Data();
  	      Vector<BaseFloat> weight(feats.Dim(),kSetZero);
  	      BaseFloat *weight_data = weight.Data();
  	      float count = 0;
  	      for (MatrixIndexT index = 0; index<feats.Dim(); index++){
                    if (feat_data[index]>=threshold){
  		     weight_data[index] = 1; 
  		     count++;
  		  } 
  	      }
  	      //std::cout<<count/feats.Dim()<<std::endl;
  	      kaldi_writer.Write(utt,weight);
  	    }
  	}
      }
    } else{
       SequentialBaseFloatVectorReader feat_reader(energy_rspecifier);
       std::vector<BaseFloat> buffer;
       for (; !feat_reader.Done(); feat_reader.Next()) {
         const Vector<BaseFloat> &feats = feat_reader.Value();
	 buffer.insert(buffer.end(),feats.Data(),feats.Data()+feats.Dim());
       }
       sort(buffer.begin(),buffer.end());
       size_t threshold_index = buffer.size() * 0.2;
       BaseFloat threshold = buffer[threshold_index];
       feat_reader.Close();
       SequentialBaseFloatVectorReader feat_reader2(energy_rspecifier);
       for (; !feat_reader2.Done(); feat_reader2.Next()) {
         std::string utt = feat_reader2.Key();
	 const Vector<BaseFloat> &feats = feat_reader2.Value();
	 const BaseFloat *feat_data = feats.Data();
         Vector<BaseFloat> weight(feats.Dim(),kSetZero);
	 BaseFloat *weight_data = weight.Data();
	 float count = 0;
	 for (MatrixIndexT index = 0; index<feats.Dim(); index++){
	   if (feat_data[index]>=threshold){
	       weight_data[index] = 1;
	       count++;
	   }
	 }
         kaldi_writer.Write(utt,weight);
       } 
    }
    return 0; 
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  
  }
}

