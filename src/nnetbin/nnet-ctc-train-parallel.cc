// nnetbin/nnet-ctc-train-parallel.cc

// Copyright 2015   Yajie Miao

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

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of CTC training by SGD.\n"
        "The updates are done per-utternace and by processing multiple utterances in parallel.\n"
        "\n"
        "Usage: nnet-ctc-train-parallel [options] <feature-rspecifier> <labels-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " nnet-ctc-train-parallel scp:feature.scp ark:labels.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;  // training options
    trn_opts.Register(&po); 

    bool binary = true, 
         crossvalidate = false;
    po.Register("binary", &binary, "Write model  in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (no backpropagation)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    int32 num_sequence = 5;
    po.Register("num-sequence", &num_sequence, "Number of sequences processed in parallel");

    double frame_limit = 100000;
    po.Register("frame-limit", &frame_limit, "Max number of frames to be processed");

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

    // Initialize feature ans labels readers
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader targets_reader(targets_rspecifier);

    // Initialize CTC optimizer
    Ctc ctc;
    ctc.SetReportStep(report_step);
    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, obj_diff;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    std::vector< Matrix<BaseFloat> > feats_utt(num_sequence);  // Feature matrix of every utterance
    std::vector< std::vector<int> > labels_utt(num_sequence);  // Label vector of every utterance
    int32 feat_dim = nnet.InputDim();

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    while (1) {

      std::vector<int> frame_num_utt;
      int32 sequence_index = 0, max_frame_num = 0; 

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

        if (max_frame_num < mat.NumRows()) max_frame_num = mat.NumRows();
        feats_utt[sequence_index] = mat;
        labels_utt[sequence_index] = targets;
        frame_num_utt.push_back(mat.NumRows());
        sequence_index++;
        // If the total number of frames reaches frame_limit, then stop adding more sequences, regardless of whether
        // the number of utterances reaches num_sequence or not.
        if (frame_num_utt.size() == num_sequence || frame_num_utt.size() * max_frame_num > frame_limit) {
            feature_reader.Next(); break;
        }
      }
      int32 cur_sequence_num = frame_num_utt.size();
      
      // Create the final feature matrix. Every utterance is padded to the max length within this group of utterances
      Matrix<BaseFloat> feat_mat_host(cur_sequence_num * max_frame_num, feat_dim, kSetZero);
      for (int s = 0; s < cur_sequence_num; s++) {
        Matrix<BaseFloat> mat_tmp = feats_utt[s];
        for (int r = 0; r < frame_num_utt[s]; r++) {
          feat_mat_host.Row(r*cur_sequence_num + s).CopyFromVec(mat_tmp.Row(r));
        }
      }        
      nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat_mat_host), &feats_transf);

      // Set the original lengths of utterances before padding
      nnet.SetSeqLengths(frame_num_utt);

      // Propagation and CTC training
      nnet.Propagate(feats_transf, &nnet_out);
      ctc.EvalParallel(frame_num_utt, nnet_out, labels_utt, &obj_diff);

      // Error rates
      ctc.ErrorRateMSeq(frame_num_utt, nnet_out, labels_utt);

      // Backward pass
      if (!crossvalidate) {
        nnet.Backpropagate(obj_diff, NULL);
      }

      // 1st minibatch : show what happens in network 
      if (kaldi::g_kaldi_verbose_level >= 2 && total_frames == 0) { // vlog-1
        KALDI_VLOG(1) << "### After " << total_frames << " frames,";
        KALDI_VLOG(1) << nnet.InfoPropagate();
        if (!crossvalidate) {
          KALDI_VLOG(1) << nnet.InfoBackPropagate();
          KALDI_VLOG(1) << nnet.InfoGradient();
        }
      }
      
      num_done += cur_sequence_num;
      total_frames += feats_transf.NumRows();

      if (feature_reader.Done()) break; // end loop of while(1)
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
