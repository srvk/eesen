// netbin/train-ctc.cc

// Copyright     2015  Yajie Miao

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

#include "net/train-opts.h"
#include "net/net.h"
#include "net/ctc-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "gpucompute/cuda-device.h"
#include "fstext/fstext-lib.h"

int main(int argc, char *argv[]) {
  using namespace eesen;
  typedef eesen::int32 int32;  
  using fst::SymbolTable; 
 
  try {
    const char *usage =
        "Perform one iteration of CTC training by SGD.\n"
        "The updates are done per-utternace and by processing a single utterance at one time.\n"
        "\n"
        "Usage: train-ctc [options] <feature-rspecifier> <labels-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        "train-ctc scp:feature.scp ark:labels.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

    std::string token_syms_filename;
    po.Register("token-symbol-table", &token_syms_filename, "Symbol table for tokens [for debug output]");

    int32 report_step=100;
    po.Register("report-step", &report_step, "Step (number of sequences) for status reporting");

    std::string use_gpu="yes";
//    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    po.Read(argc, argv);

    if (po.NumArgs() != 4-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
      model_filename = po.GetArg(3);
        
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    // Read the token symbol table if --token-symbol-table is provided
    fst::SymbolTable *token_syms = NULL;
    if (token_syms_filename != "") {
      token_syms = fst::SymbolTable::ReadText(token_syms_filename);
      if (!token_syms)
        KALDI_ERR << "Could not read symbol table from file "<< token_syms_filename;
    }

    using namespace eesen;
    typedef eesen::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif

    Net net;
    net.Read(model_filename);
    net.SetTrainOptions(trn_opts);

    eesen::int64 total_frames = 0;

    // Initialize feature and labels readers
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader targets_reader(targets_rspecifier);

    // Initialize CTC optimizer
    Ctc ctc;
    ctc.SetReportStep(report_step);

    // net_out : network outputs
    // obj_diff: the errors back-propagated to the network  
    CuMatrix<BaseFloat> net_out, obj_diff;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    for ( ; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      // Check that we have targets
      if (!targets_reader.HasKey(utt)) {
        KALDI_WARN << utt << ", missing targets";
        num_no_tgt_mat++;
        continue;
      }
      // Get feature / target pair
      Matrix<BaseFloat> mat = feature_reader.Value();
      std::vector<int32> targets = targets_reader.Value(utt);

      // Propagation
      net.Propagate(CuMatrix<BaseFloat>(mat), &net_out);

      // CTC training, obtain the errors
      ctc.Eval(net_out, targets, &obj_diff);
      float err = 0.0; std::vector<int32> hyp;
      ctc.ErrorRate(net_out, targets, &err, &hyp);

      // Back-propagation
      if (!crossvalidate) {
        net.Backpropagate(obj_diff, NULL);
      }

      // Print the hypothesis label sequence 
      if (token_syms != NULL) {
        std::cerr << utt << ' ';
        for (size_t i = 0; i < hyp.size(); i++) {
          std::string s = token_syms->Find(hyp[i]);
          if (s == "") { KALDI_ERR << "Token-id " << hyp[i] <<" not in symbol table."; }
          std::cerr << s << " ";
        }
        std::cerr << '\n';
      }
      
      num_done++;
      total_frames += mat.NumRows();
    }
      
    // Print statistics of gradients when training finishes 
    if (!crossvalidate) {
      KALDI_LOG << net.InfoGradient();
    }
    

    if (!crossvalidate) {
      net.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no targets, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";  
    KALDI_LOG << ctc.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
