

all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS =


include ../config.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES =


OBJFILES = cuda-device.o cuda-math.o cuda-matrix.o cuda-vector.o cuda-common.o cuda-rand.o
ifeq ($(CUDA), true)
  OBJFILES += cuda-kernels.o cuda-randkernels.o
endif

LIBNAME = gpucompute

all:  $(LIBFILE)


ifeq ($(CUDA), true)

  #Default compute capability architectures we compile with
  CUDA_ARCH=-gencode arch=compute_20,code=sm_20

  #Get the CUDA Toolkit version (remove decimal point char)
  CUDA_VERSION=$(shell $(CUDATKDIR)/bin/nvcc -V | grep release | sed -e 's|.*release ||' -e 's|,.*||' -e 's|\.||')
  
  #For toolkit 4.2 or newer, add the compute capability 3.0 
  CUDA_VER_GT_4_2 := $(shell [ $(CUDA_VERSION) -ge 42 ] && echo true)
  ifeq ($(CUDA_VER_GT_4_2), true)
    CUDA_ARCH += -gencode arch=compute_30,code=sm_30
  endif
  
  #For toolkit 5.0 or newer, add the compute capability 3.5 
  CUDA_VER_GT_5_0 := $(shell [ $(CUDA_VERSION) -ge 50 ] && echo true)
  ifeq ($(CUDA_VER_GT_5_0), true)
    CUDA_ARCH += -gencode arch=compute_35,code=sm_35
  endif
  
  #For toolkit 6.0 or newer, add the compute capability 5.0
  CUDA_VER_GT_6_0 := $(shell [ $(CUDA_VERSION) -ge 60 ] && echo true)
  ifeq ($(CUDA_VER_GT_6_0), true)
    CUDA_ARCH += -gencode arch=compute_50,code=sm_50
  endif
  
  #For toolkit 7.0 or newer, add the compute capability 5.3
  CUDA_VER_GT_7_0 := $(shell [ $(CUDA_VERSION) -ge 70 ] && echo true) 
  ifeq ($(CUDA_VER_GT_7_0), true)
    CUDA_ARCH += -gencode arch=compute_53,code=sm_53
  endif

  #For toolkit 8.0 or newer, add the compute capability 6.0, 6.1 and 6.2
  CUDA_VER_GT_8_0 := $(shell [ $(CUDA_VERSION) -ge 80 ] && echo true)
  ifeq ($(CUDA_VER_GT_8_0), true)
    CUDA_ARCH += -gencode arch=compute_60,code=sm_60 \
		 -gencode arch=compute_61,code=sm_61 \
		 -gencode arch=compute_62,code=sm_62
  endif


  #For toolkit older than 6.5, add the compute capability 1.0 and 1.3
  CUDA_VER_GT_6_5 := $(shell [ $(CUDA_VERSION) -ge 65 ] && echo true)
  ifneq ($(CUDA_VER_GT_6_5), true)
    CUDA_ARCH += -gencode arch=compute_13,code=sm_13 \
                 -gencode arch=compute_10,code=sm_10 
  endif

endif


#implicit rule for kernel compilation
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../


ADDLIBS = ../cpucompute/cpucompute.a ../base/base.a  ../util/util.a 

include ../makefiles/default_rules.mk

