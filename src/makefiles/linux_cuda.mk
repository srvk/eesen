
CUDA_INCLUDE= -I$(CUDATKDIR)/include
CUDA_FLAGS = -g -std=c++11 -Xcompiler -fPIC --verbose --machine 32 -DHAVE_CUDA

CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include 
LDFLAGS += -L$(CUDATKDIR)/lib -Wl,-rpath=$(CUDATKDIR)/lib
LDLIBS += -lcublas -lcudart #LDLIBS : The libs are loaded later than static libs in implicit rule

