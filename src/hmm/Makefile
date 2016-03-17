all:


include ../config.mk

TESTFILES = hmm-topology-test hmm-utils-test transition-model-test posterior-test

OBJFILES = hmm-topology.o transition-model.o hmm-utils.o tree-accu.o posterior.o

LIBNAME = kaldi-hmm
ADDLIBS = ../tree/kaldi-tree.a ../cpucompute/cpucompute.a ../util/util.a \
          ../base/base.a

include ../makefiles/default_rules.mk

