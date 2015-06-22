

all:

include ../kaldi.mk

TESTFILES = feature-mfcc-test feature-plp-test feature-fbank-test \
         feature-functions-test pitch-functions-test feature-sdc-test \
         resample-test

OBJFILES = cmvn.o feature-functions.o feature-mfcc.o feature-plp.o feature-fbank.o \
           feature-spectrogram.o mel-computations.o wave-reader.o \
           pitch-functions.o resample.o online-feature.o

LIBNAME = kaldi-feat

ADDLIBS = ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../thread/kaldi-thread.a

include ../makefiles/default_rules.mk

