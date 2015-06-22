
all:

include ../kaldi.mk

EXTRA_CXXFLAGS += -Wno-sign-compare

TESTFILES = kaldi-lattice-test push-lattice-test minimize-lattice-test \
      determinize-lattice-pruned-test 

OBJFILES = kaldi-lattice.o lattice-functions.o \
       push-lattice.o minimize-lattice.o \
       determinize-lattice-pruned.o confidence.o

LIBNAME = kaldi-lat

ADDLIBS = ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a ../base/kaldi-base.a


include ../makefiles/default_rules.mk
