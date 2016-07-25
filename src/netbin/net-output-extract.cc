// netbin/net-output-extract.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)
//                2015  Yajie Miao

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

#include <limits>

#include "net/net.h"
#include "net/class-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"


int main(int argc, char *argv[]) {
  using namespace eesen;
  typedef eesen::int32 int32;
  try {
    const char *usage =
        "Perform a forward pass through the network for classification/feature extraction.\n"
        "\n"
        "Usage:  net-output-extract [options] <model-in> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        "net-output-extract net ark:features.ark ark:output.ark\n";

    ParseOptions po(usage);

    ClassPriorOptions prior_opts;
    prior_opts.Register(&po);

    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Transform network output to logscale");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);
        
    using namespace eesen;
    typedef eesen::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif

    Net net;
    net.Read(model_filename);

    // Load the counts of the labels/targets, will be used to scale the softmax-layer
    // outputs for ASR decoding
    ClassPrior class_prior(prior_opts);

    eesen::int64 tot_t = 0;   // Keep track of how many frames/data points have been processed

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> net_out;
    Matrix<BaseFloat> net_out_host;

    Timer time;
    int32 num_done = 0;

    // Iterate over all sequences
    for (; !feature_reader.Done(); feature_reader.Next()) {
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      
      // Feed the sequence to the network for a feedforward pass
      net.Feedforward(CuMatrix<BaseFloat>(mat), &net_out);
      
      // Convert posteriors to log-scale, if needed
      if (apply_log) {
        net_out.ApplyLog();
      }
     
      // Subtract log-priors from log-posteriors, which is equivalent to
      // scaling the softmax outputs with the prior distribution
      if (prior_opts.class_frame_counts != "" ) {
        class_prior.SubtractOnLogpost(&net_out);
      }
     
      // Copy from GPU to CPU
      net_out_host.Resize(net_out.NumRows(), net_out.NumCols());
      net_out.CopyToMat(&net_out_host);

      // Write
      feature_writer.Write(feature_reader.Key(), net_out_host);

      num_done++;
      tot_t += mat.NumRows();
    }
    
    // Final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
    if (eesen::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
