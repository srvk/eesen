
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../config.mk

BINFILES = compute-mfcc-feats compute-plp-feats compute-fbank-feats \
    compute-cmvn-stats add-deltas apply-cmvn copy-feats extract-segments feat-to-len feat-to-dim \
    compute-kaldi-pitch-feats process-kaldi-pitch-feats paste-feats splice-feats subsample-feats

OBJFILES = 

TESTFILES =

ADDLIBS = ../feat/feat.a ../cpucompute/cpucompute.a ../util/util.a ../base/base.a

include ../makefiles/default_rules.mk
