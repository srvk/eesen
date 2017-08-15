import numpy
import struct
from fileutils import smart_open

def readPfile(filename):
    """
    Reads the contents of a pfile. Returns a tuple (features, labels), where
    both elements are lists of 2-D numpy arrays. Each element of a list
    corresponds to a sentence; each row of a 2-D array corresponds to a frame.
    In the case where the pfile doesn't contain labels, "labels" will be None.
    """

    with smart_open(filename, "rb") as f:
        # Read header
        # Assuming all data are consistent
        for line in f:
            tokens = line.split()
            if tokens[0] == "-pfile_header":
                headerSize = int(tokens[4])
            elif tokens[0] == "-num_sentences":
                nSentences = int(tokens[1])
            elif tokens[0] == "-num_frames":
                nFrames = int(tokens[1])
            elif tokens[0] == "-first_feature_column":
                cFeature = int(tokens[1])
            elif tokens[0] == "-num_features":
                nFeatures = int(tokens[1])
            elif tokens[0] == "-first_label_column":
                cLabel = int(tokens[1])
            elif tokens[0] == "-num_labels":
                nLabels = int(tokens[1])
            elif tokens[0] == "-format":
                format = tokens[1].replace("d", "i")
            elif tokens[0] == "-end":
                break
        nCols = len(format)
        dataSize = nFrames * nCols

        # Read sentence index
        f.seek(headerSize + dataSize * 4)
        index = struct.unpack(">%di" % (nSentences + 1), f.read(4 * (nSentences + 1)))

        # Read data
        f.seek(headerSize)
        features = []
        labels = []
        sen = 0
        for i in xrange(nFrames):
            if i == index[sen]:
                features.append([])
                labels.append([])
                sen += 1
            data = struct.unpack(">" + format, f.read(4 * nCols))
            features[-1].append(data[cFeature : cFeature + nFeatures])
            labels[-1].append(data[cLabel : cLabel + nLabels])
        features = [numpy.array(x) for x in features]
        labels = [numpy.array(x) for x in labels] if nLabels > 0 else None

    return (features, labels)

def writePfile(filename, features, labels = None):
    """
    Writes "features" and "labels" to a pfile. Both "features" and "labels"
    should be lists of 2-D numpy arrays. Each element of a list corresponds
    to a sentence; each row of a 2-D array corresponds to a frame. In the case
    where there is only one label per frame, the elements of the "labels" list
    can be 1-D arrays.
    """

    nSentences = len(features)
    nFrames = sum(len(x) for x in features)
    nFeatures = len(numpy.array(features[0][0]).ravel())
    nLabels = len(numpy.array(labels[0][0]).ravel()) if labels is not None else 0
    nCols = 2 + nFeatures + nLabels
    headerSize = 32768
    dataSize = nFrames * nCols

    with smart_open(filename, "wb") as f:
        # Write header
        f.write("-pfile_header version 0 size %d\n" % headerSize)
        f.write("-num_sentences %d\n" % nSentences)
        f.write("-num_frames %d\n" % nFrames)
        f.write("-first_feature_column 2\n")
        f.write("-num_features %d\n" % nFeatures)
        f.write("-first_label_column %d\n" % (2 + nFeatures))
        f.write("-num_labels %d\n" % nLabels)
        f.write("-format dd" + "f" * nFeatures + "d" * nLabels + "\n")
        f.write("-data size %d offset 0 ndim 2 nrow %d ncol %d\n" % (dataSize, nFrames, nCols))
        f.write("-sent_table_data size %d offset %d ndim 1\n" % (nSentences + 1, dataSize))
        f.write("-end\n")

        # Write data
        f.seek(headerSize)
        for i in xrange(nSentences):
            for j in xrange(len(features[i])):
                f.write(struct.pack(">2i", i, j))
                f.write(struct.pack(">%df" % nFeatures, *numpy.array(features[i][j]).ravel()))
                if labels is not None:
                    f.write(struct.pack(">%di" % nLabels, *numpy.array(labels[i][j]).ravel()))

        # Write sentence index
        index = numpy.cumsum([0] + [len(x) for x in features])
        f.write(struct.pack(">%di" % (nSentences + 1), *index))
