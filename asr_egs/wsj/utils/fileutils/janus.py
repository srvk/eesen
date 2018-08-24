import numpy
import struct
from fileutils import smart_open

def readFmatrix(filename):
    """
    Reads a float matrix from a Janus feature file.
    """
    with smart_open(filename, "rb") as f:
        _, rows, cols, _ = struct.unpack(">4i", f.read(16))
        return numpy.array(struct.unpack(">%df" % (rows * cols), f.read())).reshape(rows, cols)

def writeFmatrix(filename, matrix):
    """
    Writes a float matrix to a Janus feature file.
    """
    with smart_open(filename, "wb") as f:
        f.write("FMAT")
        f.write(struct.pack(">3i", matrix.shape[0], matrix.shape[1], 0))
        f.write(struct.pack(">%df" % matrix.size, *matrix.ravel()))
