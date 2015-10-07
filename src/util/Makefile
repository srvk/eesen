

all:


include ../config.mk

TESTFILES = const-integer-set-test stl-utils-test text-utils-test \
    edit-distance-test hash-list-test kaldi-io-test parse-options-test \
    kaldi-table-test simple-options-test

OBJFILES = text-utils.o kaldi-io.o \
         kaldi-table.o parse-options.o simple-options.o simple-io-funcs.o 

LIBNAME = util

ADDLIBS = ../cpucompute/cpucompute.a ../base/base.a

include ../makefiles/default_rules.mk
