// base/timer-test.cc

// Copyright 2009-2011  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/timer.h"
#include "base/kaldi-common.h"



namespace eesen {

void TimerTest() {

  Timer timer;
#if defined(_MSC_VER) || defined(MINGW)
  Sleep(1000);
#else
  sleep(1);
#endif
  BaseFloat f = timer.Elapsed();
  std::cout << "time is " << f;
  KALDI_ASSERT(fabs(1.0 - f) < 0.1);
}

}


int main() {
  eesen::TimerTest();
}

