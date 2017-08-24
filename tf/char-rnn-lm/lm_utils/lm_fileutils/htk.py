import struct

import numpy

from utils.fileutils import smart_open


def readHtk(filename):
    """
    Reads the features in a HTK file, and returns them in a 2-D numpy array.
    """

    with smart_open(filename, "rb") as f:
        # Read header
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
            # sampPeriod and parmKind will be omitted

        # Read data
        data = struct.unpack(">%df" % (nSamples * sampSize / 4), f.read(nSamples * sampSize))
        return numpy.array(data).reshape(nSamples, sampSize / 4)

def writeHtk(filename, feature, sampPeriod, parmKind):
    """
    Writes the features in a 2-D numpy array into a HTK file.
    """
    with smart_open(filename, "wb") as f:
        # Write header
        nSamples = feature.shape[0]
        sampSize = feature.shape[1] * 4
        f.write(struct.pack(">iihh", nSamples, sampPeriod, sampSize, parmKind))

        # Write data
        f.write(struct.pack(">%df" % (nSamples * sampSize / 4), *feature.ravel()))
