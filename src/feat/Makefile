
all:

include ../config.mk

TESTFILES = feature-mfcc-test feature-plp-test feature-fbank-test \
         feature-functions-test pitch-functions-test feature-sdc-test \
         resample-test

OBJFILES = srfft.o cmvn.o feature-functions.o feature-mfcc.o feature-plp.o feature-fbank.o \
           feature-spectrogram.o mel-computations.o wave-reader.o \
           pitch-functions.o resample.o online-feature.o

LIBNAME = feat

ADDLIBS = ../util/util.a ../cpucompute/cpucompute.a ../base/base.a
# ../thread/thread.a

include ../makefiles/default_rules.mk
