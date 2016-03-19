
all:

include ../config.mk

EXTRA_CXXFLAGS += -Wno-sign-compare

TESTFILES = kaldi-lattice-test push-lattice-test minimize-lattice-test \
      determinize-lattice-pruned-test 

OBJFILES = kaldi-lattice.o lattice-functions.o \
       push-lattice.o minimize-lattice.o sausages.o \
       determinize-lattice-pruned.o confidence.o

LIBNAME = lat

ADDLIBS = ../cpucompute/cpucompute.a \
          ../util/util.a ../base/base.a


include ../makefiles/default_rules.mk
