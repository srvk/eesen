EXTRA_CXXFLAGS = -Wno-sign-compare

all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
MATHLIB = NONE

# Uncomment following line to use IRSTLM toolkit installed in ../lmtoolkit
#include ./irstlm.mk

include ../config.mk

TESTFILES =

OBJFILES = const-arpa-lm.o kaldi-lmtable.o kaldi-lm.o

TESTOUTPUTS =

LIBNAME = lm

ADDLIBS = ../base/base.a ../fstext/fstext.a ../util/util.a

include ../makefiles/default_rules.mk
