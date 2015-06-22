
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = compute-mfcc-feats compute-plp-feats compute-fbank-feats \
    compute-cmvn-stats add-deltas apply-cmvn copy-feats extract-segments feat-to-len

OBJFILES = 

TESTFILES =

ADDLIBS = ../feat/kaldi-feat.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk

