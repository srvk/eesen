
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = analyze-counts arpa2fst compute-wer decode-faster latgen-faster lattice-best-path lattice-1best lattice-scale

OBJFILES =

ADDLIBS = ../lm/kaldi-lm.a ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a \
	  ../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a


TESTFILES =

include ../makefiles/default_rules.mk

