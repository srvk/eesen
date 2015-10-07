
# make "all" the target.
all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../config.mk

TESTFILES = determinize-star-test \
      pre-determinize-test trivial-factor-weight-test \
      factor-test table-matcher-test fstext-utils-test \
      remove-eps-local-test rescale-test lattice-weight-test  \
      determinize-lattice-test lattice-utils-test deterministic-fst-test \
      push-special-test epsilon-property-test prune-special-test

OBJFILES = push-special.o


LIBNAME = fstext

# tree and matrix archives needed for test-context-fst
# matrix archive needed for push-special.
ADDLIBS =  ../cpucompute/cpucompute.a \
           ../util/util.a ../base/base.a 

include ../makefiles/default_rules.mk
