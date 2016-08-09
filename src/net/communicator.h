// net/communicator.h

// Copyright      2015  Hang Su

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

#ifndef EESEN_COMMUNICATOR
#define EESEN_COMMUNICATOR

#include <unistd.h>
#include "net/net.h"
using namespace eesen; 

std::string comm_done_filename(const std::string & base_done_filename, const int & job_id) {
  return base_done_filename + ".done.job" + IntToString(job_id);
}

std::string comm_avg_model_name(const std::string & base_model_filename, const int &count) {
  return base_model_filename + ".avg" + IntToString(count);
}

std::string comm_subjob_model_name(const std::string & base_model_filename, const int & job_id, const int &count) {
  return base_model_filename + ".avg" + IntToString(count) + ".job" + IntToString(job_id);
}

void comm_avg_weights(Net &net, const int &job_id, const int &num_jobs, const int &count,
                      const std::string &base_model_filename, const std::string &base_done_filename) {
  bool binary = true;
  std::string avg_model_filename = comm_avg_model_name(base_model_filename, count);

  if (FileExist(avg_model_filename.c_str())) {
    KALDI_WARN << "Average model already exists! (From a prior run?) Removing " << avg_model_filename;
    if (std::remove(avg_model_filename.c_str())) {
      KALDI_LOG << "Remove failed; it may have been removed by another subjob: " << std::strerror(errno);
    }
  }

  if (job_id == 1) {  // job 1 is responsible for averaging the models
    KALDI_LOG << "Averaging models #" << count;
    int num_nets_added = 1; // always at least one -- this one
    for (int i = 2; i <= num_jobs; i++) {
      std::string done_filename = comm_done_filename(base_done_filename, i);
      std::string subjob_model_filename = comm_subjob_model_name(base_model_filename, i, count);
      while (true) {
        if (FileExist(subjob_model_filename.c_str())) {
          KALDI_LOG << "Found model for job " << i;
          Net net_other;
          net_other.Read(subjob_model_filename);
          net.AddNet(1, net_other);
          num_nets_added += 1;
          break;
        } else if (FileExist(done_filename.c_str())) {   // if subjob done, omit it
          KALDI_LOG << "Skipping job " << i << " because it's finished.";
          break;
        }

        usleep(300);
      }
    }
    KALDI_LOG << "Found " << num_nets_added << " models to average.";

    net.Scale(1.0 / num_nets_added);

    // Write out average model
    std::string tmp_avg_model_filename = avg_model_filename + ".$$";
    net.Write(tmp_avg_model_filename, binary);
    if (std::rename(tmp_avg_model_filename.c_str(), avg_model_filename.c_str())) {
      KALDI_WARN << "Failed to rename temporary average model: " << std::strerror(errno);
    }

    // clean up last average model
    std::string last_avg_model_filename = comm_avg_model_name(base_model_filename, count-1);
    if (FileExist(last_avg_model_filename.c_str())) {
      if (std::remove(last_avg_model_filename.c_str())) {
        KALDI_WARN << "Failed to remove the previous average: " << std::strerror(errno);
      }
    }
  } else {
    std::string main_done_filename = comm_done_filename(base_done_filename, 1);

    // save net to file, for job 1 to collect
    std::string subjob_model_filename = comm_subjob_model_name(base_model_filename, job_id, count);
    std::string tmp_subjob_model_filename = subjob_model_filename + ".$$";
    net.Write(tmp_subjob_model_filename, binary);  // write in binary
    if (std::rename(tmp_subjob_model_filename.c_str(), subjob_model_filename.c_str())) {
      KALDI_WARN << "Failed to rename temporary subjob model: " << std::strerror(errno);
    }

    // There may be more data to process but we give up if the main job
    // finishes first. If we don't do this we will diminish the regularizing
    // effect of averaging multiple nets.
    // FIXME: Training data should be distributed evenly amongst jobs. Or something.
    KALDI_LOG << "Waiting for averaged model at " << avg_model_filename;
    while (!FileExist(avg_model_filename.c_str()) && !FileExist(main_done_filename.c_str())) {
      usleep(500);
    }
    if (FileExist(main_done_filename.c_str())) {
      KALDI_WARN << "Main job finished; dropping this batch!";
      return;
    }
    net.ReRead(avg_model_filename);
    if (std::remove(subjob_model_filename.c_str())) {
      KALDI_WARN << "Failed to remove subjob model: " << std::strerror(errno);
    }
  }
}

void comm_touch_done(Ctc &ctc, const int &job_id, const int &num_jobs, const std::string &base_done_filename) {
  KALDI_LOG << "Writing done file for job" << job_id;

  // Write to done file
  std::string done_filename = comm_done_filename(base_done_filename, job_id);
  bool binary = false;
  Output out(done_filename, binary, false /*no header*/);
  out.Stream() << "Errors " << ctc.NumErrorTokens() << " Refs " << ctc.NumRefTokens();
  out.Close();

  // Collect stats from job 1
  if (job_id == 1) {  // job 1 is responsible for averaging the models
    KALDI_LOG << "Collecting stats from " << num_jobs << " done files";
    std::set<int> stats2collect;
    for (int i = 1; i <= num_jobs; i++) {
      stats2collect.insert(stats2collect.end(), i);
    }
    
    // Loop over all subjob outputs
    float tot_error_num_ = 0;
    int tot_ref_num_ = 0;
    while(!stats2collect.empty()) {
      bool wait = true;
      for (std::set<int>::iterator it = stats2collect.begin(); it != stats2collect.end(); it++) {
        std::string subjob_done_filename = comm_done_filename(base_done_filename, *it);
        if (FileExist(subjob_done_filename.c_str())) {
          Input in(subjob_done_filename, &binary);
          std::string token;
          float error_num_ = 0;
          int ref_num_ = 0;
          ReadToken(in.Stream(), false, &token);
          in.Stream() >> error_num_;
          ReadToken(in.Stream(), false, &token);
          in.Stream() >> ref_num_;
          in.Close();
          tot_error_num_ += error_num_;
          tot_ref_num_ += ref_num_;
          
          stats2collect.erase(it);
          wait = false;
          break;
        }
      }
      if (wait)
        usleep(300);
    }
    KALDI_LOG << "\nTOTAL TOKEN_ACCURACY >> " << 100.0*(1.0 - tot_error_num_ / tot_ref_num_) << "% <<";
  }

}

#endif   // EESEN_COMMUNICATOR
