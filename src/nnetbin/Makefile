
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = nnet-forward nnet-copy nnet-initialize nnet-ctc-train nnet-ctc-train-parallel nnet-format-to-nonparal

OBJFILES =



TESTFILES =

ADDLIBS = ../nnet/kaldi-nnet.a ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
