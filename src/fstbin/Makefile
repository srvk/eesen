
# make "all" the target.
all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../config.mk

BINFILES = fstdeterminizestar  \
           fstrmsymbols fstisstochastic fstminimizeencoded \
           fstaddsubsequentialloop fstaddselfloops  \
           fstrmepslocal fstcomposecontext fsttablecompose fstrand fstfactor \
           fstdeterminizelog fstphicompose fstrhocompose fstpropfinal fstcopy \
	       fstpushspecial fsts-to-transcripts

OBJFILES = 

TESTFILES = 

# actually, this library is currently empty.  Everything is a header.
LIBFILE = 

ADDLIBS = ../fstext/fstext.a ../cpucompute/cpucompute.a ../base/base.a \
          ../util/util.a 

include ../makefiles/default_rules.mk
