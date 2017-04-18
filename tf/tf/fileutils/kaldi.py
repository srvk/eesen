import numpy
import struct
from fileutils import smart_open

def readString(f):
    s = ""
    while True:
        c = f.read(1)
        if c == "": raise ValueError("EOF encountered while reading a string.")
        if c == " ": return s
        s += c

def readInteger(f):
    n = ord(f.read(1))
    return reduce(lambda x, y: x * 256 + ord(y), f.read(n)[::-1], 0)

def readMatrix(f):
    header = f.read(2)
    if header != "\0B":
        raise ValueError("Binary mode header ('\0B') not found when attempting to read a matrix.")
    format = readString(f)
    nRows = readInteger(f)
    nCols = readInteger(f)
    if format == "DM":
        data = struct.unpack("<%dd" % (nRows * nCols), f.read(nRows * nCols * 8))
        data = numpy.array(data, dtype = "float64")
    elif format == "FM":
        data = struct.unpack("<%df" % (nRows * nCols), f.read(nRows * nCols * 4))
        data = numpy.array(data, dtype = "float32")
    else:
        raise ValueError("Unknown matrix format '%s' encountered while reading; currently supported formats are DM (float64) and FM (float32)." % format)
    return data.reshape(nRows, nCols)

def writeString(f, s):
    f.write(s + " ")

def writeInteger(f, a):
    s = struct.pack("<i", a)
    f.write(chr(len(s)) + s)

def writeMatrix(f, data):
    f.write("\0B")      # Binary data header
    if str(data.dtype) == "float64":
        writeString(f, "DM")
        writeInteger(f, data.shape[0])
        writeInteger(f, data.shape[1])
        f.write(struct.pack("<%dd" % data.size, *data.ravel()))
    elif str(data.dtype) == "float32":
        writeString(f, "FM")
        writeInteger(f, data.shape[0])
        writeInteger(f, data.shape[1])
        f.write(struct.pack("<%df" % data.size, *data.ravel()))
    else:
        raise ValueError("Unsupported matrix format '%s' for writing; currently supported formats are float64 and float32." % str(data.dtype))

def readArk(filename, limit = numpy.inf):
    """
    Reads the features in a Kaldi ark file.
    Returns a list of feature matrices and a list of the utterance IDs.
    """
    features = []; uttids = []
    with smart_open(filename, "rb") as f:
        while True:
            try:
                uttid = readString(f)
            except ValueError:
                break
            feature = readMatrix(f)
            features.append(feature)
            uttids.append(uttid)
            if len(features) == limit: break
    return features, uttids

def readScp(filename, limit = numpy.inf):
    """
    Reads the features in a Kaldi script file.
    Returns a list of feature matrices and a list of the utterance IDs.
    """
    features = []; uttids = []
    with smart_open(filename, "r") as f:
        for line in f:
            uttid, pointer = line.strip().split()
            p = pointer.rfind(":")
            arkfile, offset = pointer[:p], int(pointer[p+1:])
            with smart_open(arkfile, "rb") as g:
                g.seek(offset)
                feature = readMatrix(g)
            features.append(feature)
            uttids.append(uttid)
            if len(features) == limit: break
    return features, uttids

def writeArk(filename, features, uttids):
    """
    Takes a list of feature matrices and a list of utterance IDs,
      and writes them to a Kaldi ark file.
    Returns a list of strings in the format "filename:offset",
      which can be used to write a Kaldi script file.
    """
    pointers = []
    with smart_open(filename, "wb") as f:
        for feature, uttid in zip(features, uttids):
            writeString(f, uttid)
            pointers.append("%s:%d" % (filename, f.tell()))
            writeMatrix(f, feature)
    return pointers

def writeScp(filename, uttids, pointers):
    """
    Takes a list of utterance IDs and a list of strings in the format "filename:offset",
      and writes them to a Kaldi script file.
    """
    with smart_open(filename, "w") as f:
        for uttid, pointer in zip(uttids, pointers):
            f.write("%s %s\n" % (uttid, pointer))
