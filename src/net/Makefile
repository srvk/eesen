

all:

include ../config.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = 

OBJFILES = net.o layer.o ce-loss.o ctc-loss.o class-prior.o

LIBNAME = net

ADDLIBS = ../gpucompute/gpucompute.a ../cpucompute/cpucompute.a ../base/base.a  ../util/util.a 

include ../makefiles/default_rules.mk

