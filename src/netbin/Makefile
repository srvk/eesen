
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../config.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = net-initialize net-copy format-to-nonparallel \
					 train-ctc train-ctc-parallel train-ce \
					 train-ce-parallel net-output-extract \
					 net-average net-change-model net-model-info

OBJFILES =

TESTFILES =

ADDLIBS = ../net/net.a ../gpucompute/gpucompute.a ../cpucompute/cpucompute.a \
          ../util/util.a ../base/base.a

include ../makefiles/default_rules.mk
