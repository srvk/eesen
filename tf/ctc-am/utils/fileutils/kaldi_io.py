#!/usr/bin/env python
# Copyright 2014-2016  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
import os
import sys
import re
import gzip
import struct

if sys.version_info[0] > 2:
    def str_or_bytes(bytes_string):
        return bytes_string.decode('utf-8') if isinstance(bytes_string, bytes) else bytes_string.encode('utf-8')
else:
    def str_or_bytes(bytes_string):
        return bytes_string

#################################################

IS_BIN = str_or_bytes('\x00B')
IS_EOL = str_or_bytes('\x04')
IS_SPACE = str_or_bytes(' ')
IS_EMPTY = str_or_bytes('')
FLOAT_VEC = str_or_bytes('FV ')
FLOAT_MAT = str_or_bytes('FM ')
DOUBLE_VEC = str_or_bytes('DV ')
DOUBLE_MAT = str_or_bytes('DM ')
OPEN_ASCII = str_or_bytes('[')
CLOSE_ASCII = str_or_bytes(']')

#################################################
# Adding kaldi tools to shell path,

# Select kaldi,
if not 'KALDI_ROOT' in os.environ:
    # Default! To change run python with 'export KALDI_ROOT=/some_dir python'
    os.environ['KALDI_ROOT']='/opt/kaldi'

# Add kaldi tools to path,
os.environ['PATH'] = os.popen('echo $KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin/').readline().strip() + ':' + os.environ['PATH']

#################################################
# Data-type independent helper functions,


def open_or_fd(file, mode='rb'):
    """ fd = open_or_fd(file)
    Open file, gzipped file, pipe, or forward the file-descriptor.
    Eventually seeks in the 'file' argument contains ':offset' suffix.
    """
    offset = None
    try:
        # strip 'ark:' prefix from r{x,w}filename (optional),
        if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
            (prefix,file) = file.split(':',1)
        # separate offset from filename (optional),
        if re.search(':[0-9]+$', file):
            (file,offset) = file.rsplit(':',1)
        # is it gzipped?
        if file.split('.')[-1] == 'gz':
            fd = gzip.open(file, mode)
        # input pipe?
        elif file[-1] == '|':
            fd = os.popen(file[:-1], 'rb')
        # output pipe?
        elif file[0] == '|':
            fd = os.popen(file[1:], 'wb')
        # a normal file...
        else:
            fd = open(file, mode)
    except TypeError:
        # 'file' is opened file descriptor,
        fd = file
    # Eventually seek to offset,
    if offset != None: fd.seek(int(offset))
    return fd


def read_key(fd):
    """ [key] = read_key(fd)
    Read the utterance-key from the opened ark/stream descriptor 'fd'.
    """
    key = IS_EMPTY
    while 1:
        char = fd.read(1)
        if char == IS_EMPTY: break
        if char == IS_SPACE: break
        key += char
    key = key.strip()
    if key == IS_EMPTY: return None  # end of file,
    assert(re.match(str_or_bytes('^[\.a-zA-Z0-9_-]+$'), key) is not None), "Non-valid key format"
    return key


#################################################
# Integer vectors (alignments, ...),

def read_ali_ark(file_or_fd):
    """ Alias to 'read_vec_int_ark()' """
    return read_vec_int_ark(file_or_fd)


def read_vec_int_ark(file_or_fd):
    """ generator(key,vec) = read_vec_int_ark(file_or_fd)
    Create generator of (key,vector<int>) tuples, which reads from the ark file/stream.
    file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

    Read ark to a 'dictionary':
    d = { u:d for u,d in kaldi_io.read_vec_int_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_int(fd)
            yield str_or_bytes(key), ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()


def read_vec_int(file_or_fd):
    """ [int-vec] = read_vec_int(file_or_fd)
    Read kaldi integer vector, ascii or binary input,
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2)
    if binary == IS_BIN:
        assert(fd.read(1) == IS_EOL), "Wrong int-size"
        vec_size = struct.unpack('<i', fd.read(4))[0]
        ans = np.zeros(vec_size, dtype=int)
        for i in range(vec_size):
            assert(fd.read(1) == IS_EOL), "Wrong int-size"
            ans[i] = struct.unpack('<i', fd.read(4))[0]  # data
        return ans
    else:  # ascii
        arr = (binary + fd.readline()).strip().split()
        try:
            arr.remove(OPEN_ASCII); arr.remove(CLOSE_ASCII)  # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=int)
    if fd is not file_or_fd : fd.close()  # cleanup
    return ans


# Writing,
def write_vec_int(file_or_fd, v, key=IS_EMPTY):
    """ write_vec_int(f, v, key=IS_EMPTY)
    Write a binary kaldi integer vector to filename or stream.
    Arguments:
    file_or_fd : filename or opened file descriptor for writing,
    v : the vector to be stored,
    key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

    Example of writing single vector:
    kaldi_io.write_vec_int(filename, vec)

    Example of writing arkfile:
    with open(ark_file,'w') as f:
     for key,vec in dict.iteritems():
       kaldi_io.write_vec_flt(f, vec, key=key)
    """
    fd = open_or_fd(file_or_fd, mode='wb')
    key = str_or_bytes(key)
    try:
        if key != IS_EMPTY : fd.write(key+IS_SPACE) # ark-files have keys (utterance-id),
        fd.write(IS_BIN)  # we write binary!
        # dim,
        fd.write(IS_EOL)  # int32 type,
        fd.write(struct.pack('<i', v.shape[0]))
        # data,
        for i in range(len(v)):
            fd.write(IS_EOL)  # int32 type,
        fd.write(struct.pack('<i', v[i]))  # binary,
    finally:
        if fd is not file_or_fd : fd.close()


#################################################
# Float vectors (confidences, ivectors, ...),

def read_vec_flt_ark(file_or_fd):
    """ generator(key,vec) = read_vec_flt_ark(file_or_fd)
    Create generator of (key,vector<float>) tuples, reading from an ark file/stream.
    file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

    Read ark to a 'dictionary':
    d = { u:d for u,d in kaldi_io.read_vec_flt_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_flt(fd)
            yield str_or_bytes(key), ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()


def read_vec_flt(file_or_fd):
    """ [flt-vec] = read_vec_flt(file_or_fd)
    Read kaldi float vector, ascii or binary input,
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2)
    if binary == IS_BIN:  # binary flag
        # Data type,
        fd_type = fd.read(3)
        if fd_type == FLOAT_VEC: sample_size = 4  # floats
        if fd_type == DOUBLE_VEC: sample_size = 8  # doubles
        assert(sample_size > 0)
        # Dimension,
        assert(fd.read(1) == IS_EOL), "Wrong int-size"
        vec_size = struct.unpack('<i', fd.read(4))[0]
        # Read whole vector,
        buf = fd.read(vec_size * sample_size)
        if sample_size == 4: ans = np.frombuffer(buf, dtype='float32')
        elif sample_size == 8: ans = np.frombuffer(buf, dtype='float64')
        else: raise BadSampleSize
        return ans
    else: # ascii,
        arr = (binary + fd.readline()).strip().split()
        try:
            arr.remove(OPEN_ASCII); arr.remove(CLOSE_ASCII)  # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=float)
    if fd is not file_or_fd: fd.close()  # cleanup
    return ans


# Writing,
def write_vec_flt(file_or_fd, v, key=IS_EMPTY):
    """ write_vec_flt(f, v, key=IS_EMPTY)
    Write a binary kaldi vector to filename or stream. Supports 32bit and 64bit floats.
    Arguments:
    file_or_fd: filename or opened file descriptor for writing,
    v: the vector to be stored,
    key (optional): used for writing ark-file, the utterance-id gets written before the vector.

    Example of writing single vector:
    kaldi_io.write_vec_flt(filename, vec)

    Example of writing arkfile:
    with open(ark_file,'w') as f:
     for key,vec in dict.iteritems():
       kaldi_io.write_vec_flt(f, vec, key=key)
    """
    fd = open_or_fd(file_or_fd, mode='wb')
    key = str_or_bytes(key)
    try:
        if key != IS_EMPTY: fd.write(key+IS_SPACE)  # ark-files have keys (utterance-id),
        fd.write(IS_BIN)  # we write binary!
        # Data-type,
        if v.dtype == 'float32': fd.write(FLOAT_VEC)
        elif v.dtype == 'float64': fd.write(DOUBLE_VEC)
        else: raise VectorDataTypeError
        # Dim,
        fd.write(IS_EOL)
        fd.write(struct.pack('I',v.shape[0]))  # dim
        v.tofile(fd, sep="")  # binary
    finally:
        if fd is not file_or_fd: fd.close()

#################################################
# Float matrices (features, transformations, ...),


# Reading,
def read_mat_scp(file_or_fd):
    """ generator(key,mat) = read_mat_scp(file_or_fd)
    Returns generator of (key,matrix) tuples, read according to kaldi scp.
    file_or_fd: scp, gzipped scp, pipe or opened file descriptor.

    Iterate the scp:
    for key,mat in kaldi_io.read_mat_scp(file):
     ...

    Read scp to a 'dictionary':
    d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        for line in fd:
            (key, rxfile) = line.split(IS_SPACE)
            mat = read_mat(str_or_bytes(rxfile))
            yield str_or_bytes(key), mat
    finally:
        if fd is not file_or_fd: fd.close()


def read_mat_ark(file_or_fd):
    """ generator(key,mat) = read_mat_ark(file_or_fd)
    Returns generator of (key,matrix) tuples, read from ark file/stream.
    file_or_fd: scp, gzipped scp, pipe or opened file descriptor.

    Iterate the ark:
    for key,mat in kaldi_io.read_mat_ark(file):
     ...

    Read ark to a 'dictionary':
    d = { key:mat for key,mat in kaldi_io.read_mat_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            mat = read_mat(fd)
            yield str_or_bytes(key), mat
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()


def read_mat(file_or_fd):
    """ [mat] = read_mat(file_or_fd)
    Reads single kaldi matrix, supports ascii and binary.
    file_or_fd: file, gzipped file, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    try:
        binary = fd.read(2)
        if binary == IS_BIN:
            mat = _read_mat_binary(fd)
        else:
            assert(binary == ' [')
            mat = _read_mat_ascii(fd)
    finally:
        if fd is not file_or_fd: fd.close()
    return mat


def _read_mat_binary(fd):
    # Data type
    fd_type = fd.read(3)
    if fd_type == FLOAT_MAT: sample_size = 4  # floats
    if fd_type == DOUBLE_MAT: sample_size = 8  # doubles
    assert(sample_size > 0)
    # Dimensions
    fd.read(1)
    rows = struct.unpack('<i', fd.read(4))[0]
    fd.read(1)
    cols = struct.unpack('<i', fd.read(4))[0]
    # Read whole matrix
    buf = fd.read(rows * cols * sample_size)
    if sample_size == 4: vec = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8: vec = np.frombuffer(buf, dtype='float64')
    else: raise BadSampleSize
    mat = np.reshape(vec,(rows,cols))
    return mat


def _read_mat_ascii(fd):
    rows = []
    while 1:
        line = fd.readline()
        if (len(line) == 0): raise BadInputFormat  # eof, should not happen!
        if len(line.strip()) == 0: continue  # skip empty line
        arr = line.strip().split()
        if arr[-1] != CLOSE_ASCII:
            rows.append(np.array(arr, dtype='float32'))  # not last line
        else:
            rows.append(np.array(arr[:-1], dtype='float32'))  # last line
            mat = np.vstack(rows)
            return mat


# Writing,
def write_mat(file_or_fd, m, key=IS_EMPTY):
    """ write_mat(f, m, key=IS_EMPTY)
    Write a binary kaldi matrix to filename or stream. Supports 32bit and 64bit floats.
    Arguments:
    file_or_fd: filename of opened file descriptor for writing,
    m: the matrix to be stored,
    key (optional): used for writing ark-file, the utterance-id gets written before the matrix.

    Example of writing single matrix:
    kaldi_io.write_mat(filename, mat)

    Example of writing arkfile:
    with open(ark_file,'w') as f:
     for key,mat in dict.iteritems():
       kaldi_io.write_mat(f, mat, key=key)
    """
    fd = open_or_fd(file_or_fd, mode='wb')
    key = str_or_bytes(key)
    try:
        if key != IS_EMPTY: fd.write(key+IS_SPACE)  # ark-files have keys (utterance-id),
        fd.write(IS_BIN)  # we write binary!
        # Data-type,
        if m.dtype == 'float32': fd.write(FLOAT_MAT)
        elif m.dtype == 'float64': fd.write(DOUBLE_MAT)
        else: raise MatrixDataTypeError
        # Dims,
        fd.write(IS_EOL)
        fd.write(struct.pack('I', m.shape[0]))  # rows
        fd.write(IS_EOL)
        fd.write(struct.pack('I', m.shape[1]))  # cols
        # Data,
        m.tofile(fd, sep="")  # binary
    finally:
        if fd is not file_or_fd: fd.close()

#################################################
# 'Posterior' kaldi type (posteriors, confusion network, nnet1 training targets, ...)
# Corresponds to: vector<vector<tuple<int,float> > >
# - outer vector: time axis
# - inner vector: records at the time
# - tuple: int = index, float = value
#


def read_cnet_ark(file_or_fd):
    """ Alias of function 'read_post_ark()', 'cnet' = confusion network """
    return read_post_ark(file_or_fd)


def read_post_ark(file_or_fd):
    """ generator(key,vec<vec<int,float>>) = read_post_ark(file)
    Returns generator of (key,posterior) tuples, read from ark file.
    file_or_fd: ark, gzipped ark, pipe or opened file descriptor.

    Iterate the ark:
    for key,post in kaldi_io.read_post_ark(file):
     ...

    Read ark to a 'dictionary':
    d = { key:post for key,post in kaldi_io.read_post_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            post = read_post(fd)
            yield str_or_bytes(key), post
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()


def read_post(file_or_fd):
    """ [post] = read_post(file_or_fd)
    Reads single kaldi 'Posterior' in binary format.

    The 'Posterior' is C++ type 'vector<vector<tuple<int,float> > >',
    the outer-vector is usually time axis, inner-vector are the records
    at given time,  and the tuple is composed of an 'index' (integer)
    and a 'float-value'. The 'float-value' can represent a probability
    or any other numeric value.

    Returns vector of vectors of tuples.
    """
    fd = open_or_fd(file_or_fd)
    ans=[]
    binary = fd.read(2)
    assert(binary == IS_BIN), "Format is not binary"
    assert(fd.read(1) == IS_EOL), "Wrong int-size"
    outer_vec_size = struct.unpack('<i', fd.read(4))[0]  # number of frames (or bins)

    # Loop over 'outer-vector',
    for i in range(outer_vec_size):
        assert(fd.read(1) == IS_EOL), "Wrong int-size"
        inner_vec_size = struct.unpack('<i', fd.read(4))[0]  # number of records for frame (or bin)
        id = np.zeros(inner_vec_size, dtype=int)  # buffer for integer id's
        post = np.zeros(inner_vec_size, dtype=float)  # buffer for posteriors

    # Loop over 'inner-vector',
    for j in range(inner_vec_size):
        assert(fd.read(1) == IS_EOL), "Wrong int-size"
        id[j] = struct.unpack('<i', fd.read(4))[0]  # id
        assert(fd.read(1) == IS_EOL), "Wrong float-size"
        post[j] = struct.unpack('<f', fd.read(4))[0]  # post

    # Append the 'inner-vector' of tuples into the 'outer-vector'
    ans.append(zip(id, post))

    if fd is not file_or_fd: fd.close()
    return ans


#################################################
# Kaldi Confusion Network bin begin/end times,
# (kaldi stores CNs time info separately from the Posterior).
#

def read_cntime_ark(file_or_fd):
    """ generator(key,vec<tuple<float,float>>) = read_cntime_ark(file_or_fd)
    Returns generator of (key,cntime) tuples, read from ark file.
    file_or_fd: file, gzipped file, pipe or opened file descriptor.

    Iterate the ark:
    for key,time in kaldi_io.read_cntime_ark(file):
     ...

    Read ark to a 'dictionary':
    d = { key:time for key,time in kaldi_io.read_post_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            cntime = read_cntime(fd)
            yield str_or_bytes(key), cntime
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()


def read_cntime(file_or_fd):
    """ [cntime] = read_cntime(file_or_fd)
    Reads single kaldi 'Confusion Network time info', in binary format:
    C++ type: vector<tuple<float,float> >.
    (begin/end times of bins at the confusion network).

    Binary layout is '<num-bins> <beg1> <end1> <beg2> <end2> ...'

    file_or_fd: file, gzipped file, pipe or opened file descriptor.

    Returns vector of tuples.
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2)
    assert(binary == IS_BIN), "Format is not binary"
    assert(fd.read(1) == IS_EOL), "Wrong int-size"
    # Get number of bins,
    vec_size = struct.unpack('<i', fd.read(4))[0]  # number of frames (or bins)
    t_beg = np.zeros(vec_size, dtype=float)
    t_end = np.zeros(vec_size, dtype=float)

    # Loop over number of bins,
    for i in range(vec_size):
        assert(fd.read(1) == IS_EOL), "Wrong float-size"
        t_beg[i] = struct.unpack('<f', fd.read(4))[0]  # begin-time of bin
        assert(fd.read(1) == IS_EOL), "Wrong float-size"
        t_end[i] = struct.unpack('<f', fd.read(4))[0]  # end-time of bin

    # Return vector of tuples,
    ans = zip(t_beg,t_end)
    if fd is not file_or_fd: fd.close()
    return ans
