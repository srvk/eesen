all:

EXTRA_CXXFLAGS = -Wno-sign-compare -O3
include ../kaldi.mk

TESTFILES = 

OBJFILES = lattice-faster-decoder.o faster-decoder.o decoder-wrappers.o

LIBNAME = kaldi-decoder

ADDLIBS = ../lat/kaldi-lat.a ../util/kaldi-util.a ../base/kaldi-base.a ../matrix/kaldi-matrix.a 

include ../makefiles/default_rules.mk

