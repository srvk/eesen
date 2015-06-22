// nnetbin/nnet-ctc-train.cc

// Copyright 2011-2014  Brno University of Technology (Author: Karel Vesely)
//                2015  Yajie Miao

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

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-ctc-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "fstext/fstext-lib.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  using fst::SymbolTable; 
 
  try {
    const char *usage =
        "Perform one iteration of CTC training by SGD..\n"
        "The updates are done per-utternace and by processing a single utterance at one time.\n"
        "\n"
        "Usage: nnet-ctc-train [options] <feature-rspecifier> <labels-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " nnet-ctc-train scp:feature.scp ark:labels.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

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

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader targets_reader(targets_rspecifier);

    // Initialize CTC optimizer
    Ctc ctc;
    ctc.SetReportStep(report_step); 
    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, obj_diff;

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

      // Apply optional feature transform
      nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

      // Propagation and CTC training
      nnet.Propagate(feats_transf, &nnet_out);
      ctc.Eval(nnet_out, targets, &obj_diff);
      float err = 0.0; std::vector<int32> hyp;
      ctc.ErrorRate(nnet_out, targets, &err, &hyp);

      // Backward pass
      if (!crossvalidate) {
        nnet.Backpropagate(obj_diff, NULL);
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
      total_frames += feats_transf.NumRows();
    }
      
    // Check network parameters and gradients when training finishes 
    if (kaldi::g_kaldi_verbose_level >= 1) { // vlog-1
      KALDI_VLOG(1) << "### After " << total_frames << " frames,";
      KALDI_VLOG(1) << nnet.InfoPropagate();
      if (!crossvalidate) {
        KALDI_VLOG(1) << nnet.InfoBackPropagate();
        KALDI_VLOG(1) << nnet.InfoGradient();
      }
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
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
