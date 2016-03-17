
all:

include ../config.mk

# note, build-tree-utils-test also tests build-tree-questions.cc

TESTFILES = event-map-test context-dep-test build-tree-utils-test \
						cluster-utils-test build-tree-test


OBJFILES = event-map.o context-dep.o clusterable-classes.o cluster-utils.o \
					 build-tree-utils.o build-tree.o build-tree-questions.o tree-renderer.o

LIBNAME = kaldi-tree
ADDLIBS = ../util/util.a ../cpucompute/cpucompute.a ../base/base.a


include ../makefiles/default_rules.mk
