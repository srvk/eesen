import h5py
import json
import numpy as np
import random

class H5Dataset():


    def parseArg(self, a, b):
        if a is None:
            if b is None:
                return None
            else:
                return b.split(',')
        else:
            return a.split(',')


    def parseMapping(self, a, b):
        mapping = b if a is None else a

        if mapping is not None:
            """ Parse mapping if multiple files have been specified
                _en_,/en_mapping.file,_de_,/de_mapping.file
            """
            if len(mapping.split(',')) > 1:
                self.singleMap = False
                items = mapping.split(',')
                res = {}
                for key, value in zip(items[0::2], items[1::2]):
                    with open(value, 'r') as inFP:
                        res[key] = json.load(inFP)
            else:
                self.singleMap = True
                with open(mapping, 'r') as inFP:
                    res = json.load(inFP)
        else:
            res = None

        return res


    def getLengthAudio(self, element):
        return self.h5[element][self.feats[0]].shape[0]


    def getLengthTranscription(self, element):
        return len(self.h5[element][self.target].value[0].split())


    def isInSpkList(self, element):
        spk = element.split('/')[0]
        # Check if spkList is specified
        if self.spkList is None:
            return True
        # Check is spkList is a list and applies to all utts
        elif isinstance(self.spkList, list):
            if spk in self.spkList:
                return True
        # Check is spkList is dictionary and applies only to certain spks
        elif isinstance(self.spkList, dict):
            for key in self.spkList.keys():
                # Check if spkList is specified for subset
                if spk.find(key) > -1:
                    if spk in self.spkList[key]:
                        return True
                # Subset not specified, use utt
                else:
                    return True
        return False


    def skipUtt(self, element):
        if self.uttSkip is None:
            return False
        else:
            utt = element.split('/')[1]
            if utt in self.uttSkip:
                print("Skipping utt {}".format(utt))
                return True
            else:
                return False

    # def isInSpkList(self, element):
    #     return True


    def readData(self):

        print("Loading h5 data {} ...".format(self.h5.filename))
        # Get all utts
        self.ids = []
        self.totalLength = 0
        for speaker in self.h5.keys() if self.spkList is None else self.spkList:
            for key in self.h5[speaker].keys():
                full_path = "{}/{}".format(speaker, key)
                aLength = self.getLengthAudio(full_path)
                tLength = self.getLengthTranscription(full_path)

                if aLength > self.minLength and tLength < self.maxTransLength and tLength > self.minTransLength and self.isInSpkList(full_path) and not self.skipUtt(full_path):

                    self.ids.append((full_path, aLength, tLength))
                    self.totalLength += aLength

        if self.filter is not None:
            new_ids = []
            for full_path, aLength, tLength in self.ids:
                for filter_string in self.filter:
                    if full_path.find(filter_string) > -1:
                        new_ids.append((full_path, aLength, tLength))
                        break

            self.ids = new_ids

        def sortByLengthLabel(element):
            return element[2]

        # Sort utts by length
        self.ids.sort(key=sortByLengthLabel)

        print("Loading of dataset {} complete.".format(self.h5.filename))
        print("Total length: {:.1f}h".format(float(self.totalLength) / 100.0 / 60.0 / 60.0))

        """ Limit size of set for debugging purposes
        """
        if self.args.debug:
            self.ids = self.ids[:100]

        self.size = len(self.ids)

        return self.ids


    def getTranscript(self, element):

        data = self.h5[element][self.target].value[0]

        temp_transcript = []
        for item in data.split():
            if not self.singleMap:
                for key in self.mapping.keys():
                    if element.find(key) > -1:
                        item = self.mapping[key][item]
                        break
            else:
                item = self.mapping[item]
            temp_transcript.append(item)

        transcript = temp_transcript

        try:
            transcript = [self.labels_map[x] for x in transcript]
        except:
            print("Error in transcript '{}' split={}".format(data, temp_transcript))
            for x in transcript:
                try:
                    _ = "Mapping {} to {}".format(x, self.labels_map[x])
                except:
                    print("Error mapping '{}'".format(x))
                    raise

        return transcript

        res = []
        for item in transcript:
            res.append([item])

        return res


    def apply_context(self, feat, left, right):
        feat = [feat]
        for i in range(left):
            feat.append(np.vstack((feat[-1][0], feat[-1][:-1])))
        feat.reverse()
        for i in range(right):
            feat.append(np.vstack((feat[-1][1:], feat[-1][-1])))
        return np.hstack(feat)


    def getAudio(self, data_path):
        # Merge features if multiple are specified
        spect = []
        for feat in self.feats:
            try:
                temp_feat = np.array(self.h5[data_path][feat], dtype=np.float32)
                spect.append(temp_feat)
            except:
                print("Error accessing {}/{} audio!".format(data_path, feat))

        spect = np.hstack(spect)
        spect = np.array(spect, dtype=np.float32)
        spect = self.apply_context(spect, self.lctx, self.rctx)

        if self.augment_feat is not None:
            # Merge features if multiple are specified
            aug_temp = []
            for feat in self.augment_feat.split(','):
                try:
                    temp_feat = np.array(self.h5[data_path][feat], dtype=np.float32)
                    aug_temp.append(temp_feat)
                except:
                    print("Error accessing {}/{} audio!".format(data_path, feat))

            try:
                aug2 = np.hstack(aug_temp)
            except:
                print("Error stacking augmented features!")
                print("Augment={}".format(self.augment_feat))
                print("{}", len(aug_temp))
                for item in aug_temp:
                    print("shape={}".item.shape)

                raise

            aug = np.lib.stride_tricks.as_strided(aug2, shape=(spect.shape[0], self.augment_size), strides=(0,np.dtype(np.float32).itemsize))

            # Add data to features
            spect = np.concatenate((spect, aug), axis=1)

        return spect


    def make_even_batches(self, batch_size):

        self.batches = []

        # Initialize variables if they are still uninitialized
        if self.ids is None:
            self.readData()

        idx = 0
        L = len(self)
        while idx < L:
            j = idx + 1
            target_len = self.ids[idx][2]
            while j < min(idx + batch_size, L) and self.ids[j][2] == target_len:
                j += 1
            self.batches.append(list(range(idx, j)))
            idx = j

        print("Created {} even sized batches out of {} utts.".format(len(self.batches), len(self)))

        return self.batches


    def read_batch(self, batchIdx):

        feats = []
        targets = []

        batch_len = len(self.batches[batchIdx])
        max_aLen = 0
        max_tLen = 0
        for idx in self.batches[batchIdx]:
            max_aLen = max(max_aLen, self.ids[idx][1])
            max_tLen = max(max_tLen, self.ids[idx][2])

        ares = np.zeros((batch_len, max_aLen, self.input_dim * (self.rctx + self.lctx + 1)), np.float32)

        yidx = []
        yval = []
        yshape = np.array([batch_len, max_tLen], dtype = np.int32)

        for i in range(batch_len):
            idx = self.batches[batchIdx][i]
            element, feat_len, t_len = self.ids[idx]
            ares[i, :feat_len, :] = self.getAudio(element)

            trans = self.getTranscript(element)
            for j in range(t_len):
                yidx.append([i, j])
                yval.append(trans[j])

        yidx = np.asarray(yidx, dtype = np.int32)
        yval = np.asarray(yval, dtype = np.int32)

        # not sure why this is needed
        res = [yidx, yval, yshape]
        return ares, [res]


    def __init__(self, args, input_file=None, h5_pointer=None, labels=None, feature=None, input_dim=None, target=None, access_mode='r', context=1, rctx=None, lctx=None, mapping=None, augment_feat=None, augment_size=None, filter_string=None, spkList=None, uttSkip=None):

        self.args = args

        self.minLength = 100
        self.maxTransLength = 630
        self.minTransLength = 5

        """Init dataset
           Explicit arguments override values from parsed arguments
        """

        self.h5 = h5py.File(input_file, access_mode) if h5_pointer is None else h5_pointer
        self.input_dim = args.h5_input_dim if input_dim is None else input_dim
        self.feats = args.h5_input_feat.split(',') if feature is None else feature.split(',')
        self.target = args.h5_target if target is None else target

        with open(args.h5_labels if labels is None else labels) as label_file:
            self.labels = json.load(label_file)
        self.labels_map = dict([(self.labels[i], i) for i in range(len(self.labels))])

        if uttSkip is None:
            if args.h5_uttSkip is None:
                self.uttSkip = None
            else:
                self.uttSkip = args.h5_uttSkip.split(',')
        else:
            self.uttSkip = uttSkip.split(',')

        self.augment_feat = self.parseArg(augment_feat, args.h5_augment_feat)
        self.augment_size = self.parseArg(augment_size, args.h5_augment_size)
        self.uttSkip = self.parseArg(uttSkip, args.h5_uttSkip)
        self.filter = self.parseArg(filter_string, args.h5_filter)
        self.spkList = self.parseArg(spkList, args.h5_spkList)
        self.mapping = self.parseMapping(mapping, args.h5_mapping)

        self.lctx = context if lctx is None else None
        self.rctx = context if rctx is None else None

        self.size = None
        self.ids = None
        self.batches = None


    def __getitem__(self, index):

        h5_path, _, _ = self.ids[index]

        audio = self.getAudio(h5_path)
        transcript = self.getTranscript(h5_path)
        return audio, transcript


    def __len__(self):
        return self.size


    def load_feat_info(self):
        # this only supports the single softmax case for now
        # why are we adding 1?
        nclass = [len(self.labels) + 1]
        nfeat = self.input_dim * (self.rctx + self.lctx + 1)

        return nclass, nfeat, (None, None, None)


def h5_run_reader(q, dataset, do_shuf):
    batch_shuf = list(range(len(dataset.batches)))

    if do_shuf:
        random.shuffle(batch_shuf)

    for i in batch_shuf:
        feats, targets = dataset.read_batch(i)
        q.put((feats, targets))

    q.put(None)
