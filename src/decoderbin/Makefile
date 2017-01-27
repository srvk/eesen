
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../config.mk

BINFILES = analyze-counts arpa2fst compute-wer decode-faster latgen-faster lattice-best-path lattice-1best lattice-to-nbest lattice-scale nbest-to-ctm lattice-prune lattice-to-ctm-conf lattice-add-penalty

OBJFILES =

ADDLIBS = ../lm/lm.a ../decoder/decoder.a ../lat/lat.a \
	  ../cpucompute/cpucompute.a  ../util/util.a ../base/base.a


TESTFILES =

include ../makefiles/default_rules.mk

