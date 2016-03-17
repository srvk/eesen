
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../config.mk

BINFILES = lattice-add-penalty lattice-best-path lattice-prune lattice-to-ctm-conf

OBJFILES =

TESTFILES =

# C++ infuriatingly depends on the order of these, rather than treating
# libraries as a SET
ADDLIBS = ../lat/lat.a ../util/util.a ../lm/lm.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../cpucompute/cpucompute.a ../base/base.a 

include ../makefiles/default_rules.mk
