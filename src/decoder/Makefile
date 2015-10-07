all:

EXTRA_CXXFLAGS = -Wno-sign-compare -O3
include ../config.mk

TESTFILES = 

OBJFILES = lattice-faster-decoder.o faster-decoder.o decoder-wrappers.o

LIBNAME = decoder

ADDLIBS = ../lat/lat.a ../util/util.a ../base/base.a ../cpucompute/cpucompute.a 

include ../makefiles/default_rules.mk

