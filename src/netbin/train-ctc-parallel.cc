// netbin/train-ctc-parallel.cc

// Copyright 2015   Yajie Miao, Hang Su, Mohammad Gowayyed

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
#include "net/communicator.h"

using namespace eesen;
typedef eesen::int32 int32;

int main(int argc, char *argv[]) {
  using namespace eesen;
  typedef eesen::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of CTC training by SGD.\n"
        "The updates are done per-utterance and by processing multiple utterances in parallel.\n"
        "\n"
        "Usage: train-ctc-parallel [options] <feature-rspecifier> <labels-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        "train-ctc-parallel scp:feature.scp ark:labels.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NetTrainOptions trn_opts;  // training options
    trn_opts.Register(&po); 

    bool binary = true, 
         crossvalidate = false;
    po.Register("binary", &binary, "Write model  in binary mode");

    bool block_softmax = false;
    po.Register("block-softmax", &block_softmax, "Whether to use block-softmax or not (default is false). Note that you have to pass this parameter even if the provided model contains a BlockSoftmax layer.");

    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (no backpropagation)");

    std::string sequence_out_file="";
    po.Register("sequence-out-file", &sequence_out_file, "output file for the generated sequence");

    int32 num_sequence = 5;
    po.Register("num-sequence", &num_sequence, "Number of sequences processed in parallel");

    double frame_limit = 100000;
    po.Register("frame-limit", &frame_limit, "Max number of frames to be processed");

    int32 report_step=100;
    po.Register("report-step", &report_step, "Step (number of sequences) for status reporting");

    std::string use_gpu="yes";
//    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    int32 num_jobs = 1;
    po.Register("num-jobs", &num_jobs, "Number subjobs in multi-GPU mode");

    int32 job_id = 1;
    po.Register("job-id", &job_id, "Subjob id in multi-GPU mode");

    int32 utts_per_avg = 500;
    po.Register("utts-per-avg", &utts_per_avg, "Number of utterances to process per average (default is 250)");

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
    std::string base_done_filename = crossvalidate ? model_filename + ".cv" : target_model_filename + ".tr";
    std::string done_filename = comm_done_filename(base_done_filename, job_id);

    if (FileExist(done_filename.c_str())) {
      KALDI_WARN << "Done file already exists! (From a previous run?) Removing.";
      if (std::remove(done_filename.c_str())) {
        KALDI_WARN << "Failed: " << std::strerror(errno);
      }
    }

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
    CuMatrix<BaseFloat> net_out, obj_diff;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";
    if (sequence_out_file.length()) {
      KALDI_LOG << "Sequences will be written to " << sequence_out_file << " in order from feature file";
      std::remove(sequence_out_file.c_str());
    }

    std::vector< Matrix<BaseFloat> > feats_utt(num_sequence);  // Feature matrix of every utterance
    std::vector< std::vector<int> > labels_utt(num_sequence);  // Label vector of every utterance
    int32 feat_dim = net.InputDim();

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0, avg_count = 0;

    std::vector<int> block_softmax_dims(0);
    if (block_softmax)
      block_softmax_dims = net.GetBlockSoftmaxDims();

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

        if (mat.NumRows() > frame_limit) {
          KALDI_WARN << utt << ", has too many frames; ignoring: " << mat.NumRows() << " > " << frame_limit;
          continue;
        }

        int new_max_frame_num = max_frame_num;
        if (mat.NumRows() > new_max_frame_num) {
          new_max_frame_num = mat.NumRows();
        }
        if (new_max_frame_num * (frame_num_utt.size() + 1) > frame_limit) {  // then this utterance doesn't fit in this batch
          break;
        }
        max_frame_num = new_max_frame_num;

        feats_utt[sequence_index] = mat;
        labels_utt[sequence_index] = targets;
        frame_num_utt.push_back(mat.NumRows());
        sequence_index++;
        if (frame_num_utt.size() == num_sequence) {
            feature_reader.Next();
            break;
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

      // Set the original lengths of utterances before padding
      net.SetSeqLengths(frame_num_utt);

      // Propagation and CTC training
      net.Propagate(CuMatrix<BaseFloat>(feat_mat_host), &net_out);
      // ctc.EvalParallel(frame_num_utt, net_out, labels_utt, &obj_diff);

      // I moved the Resize outside the EvalParallel for the block softmax to be convenient
      obj_diff.Resize(net_out.NumRows(), net_out.NumCols());
      obj_diff.Set(0);

      if (block_softmax && block_softmax_dims.size() > 0) {
        int startIdx = 0;

	for (int i = 0; i < block_softmax_dims.size(); i++) {
          // we need to get the submatrix that corresponds to the current block
	  std::vector< std::vector<int> > labels_utt_block(cur_sequence_num);
	  std::vector<int> frame_num_utt_block(cur_sequence_num);
	  // for now, we assume that the original labels use the whole index, so we need to change them to be relative to the current softmax
	  int nonzero_seq = 0;

	  for (int s = 0; s < cur_sequence_num; s++) {
	    // we need to check if this sequence belongs to this block
	    if (labels_utt[s].size() > 0 && labels_utt[s][0] >= startIdx && labels_utt[s][0] < startIdx + block_softmax_dims[i]) {
	      frame_num_utt_block[s] = frame_num_utt[s];
	      for (int r = 0; r < labels_utt[s].size(); r++) {
		KALDI_ASSERT (labels_utt[s][r] >= startIdx && labels_utt[s][r] < startIdx + block_softmax_dims[i]);
		labels_utt_block[s].push_back(labels_utt[s][r] - startIdx);
	      }
	      nonzero_seq++;
	    } else {
	      frame_num_utt_block[s] = 0;
	    }
	  }
	  if (nonzero_seq > 0) {
	    CuSubMatrix<BaseFloat> net_out_block = net_out.ColRange(startIdx, block_softmax_dims[i]);
	    CuSubMatrix<BaseFloat> obj_diff_block = obj_diff.ColRange(startIdx, block_softmax_dims[i]);
            // BlockSoftMax=true may not strictly be needed (Gowayyed's "sequence2" CUDA kernel
            // should work ok also for non-blocksoftmax cases), but just in case let's make this explicit
	    ctc.EvalParallel(frame_num_utt_block, net_out_block, labels_utt_block, &obj_diff_block, true);
	    // Error rates and output
	    ctc.ErrorRateMSeq (frame_num_utt_block, net_out_block, labels_utt_block, sequence_out_file);
	  }
	  startIdx += block_softmax_dims[i];
	}
      } else {
	// We explicitly state that BlockSoftMax=false (see above for the "true" case)
        ctc.EvalParallel(frame_num_utt, net_out, labels_utt, &obj_diff, false);
        // Error rates and output
        ctc.ErrorRateMSeq(frame_num_utt, net_out, labels_utt, sequence_out_file);
      }

      // Backward pass
      if (!crossvalidate) {
	net.Backpropagate(obj_diff, NULL);
        if (num_jobs != 1 && (num_done + cur_sequence_num) / utts_per_avg != num_done / utts_per_avg) {
          comm_avg_weights(net, job_id, num_jobs, avg_count, target_model_filename, base_done_filename);
          avg_count++;
        }
      }
      
      num_done += cur_sequence_num;
      total_frames += feat_mat_host.NumRows();
      
      if (feature_reader.Done()) break; // end loop of while(1)
    }

    if (num_jobs != 1) {
      if (!crossvalidate) {
        comm_avg_weights(net, job_id, num_jobs, avg_count, target_model_filename, base_done_filename);
      }

      comm_touch_done(ctc, job_id, num_jobs, base_done_filename);

      KALDI_LOG << "Total average operations: " << (avg_count + 1);

      if (job_id == 1 && !crossvalidate) {
        std::string avg_model_name = comm_avg_model_name(target_model_filename, avg_count);
        if (std::rename(avg_model_name.c_str(), target_model_filename.c_str())) {
          KALDI_LOG << "Failed to rename " << avg_model_name << " to " << target_model_filename << "; reason: " << std::strerror(errno);
        }
      }
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
