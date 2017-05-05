from collections import defaultdict

# P
SOS = '$'
EOS = '#'
UNK = "@"
UNK_ID = 0
TOKENS = " '-abcdefghijklmnopqrstuvwxyz"


class LM_Wrapper:
    def __init__(self, char_to_int, lm):
        self.char_to_int = char_to_int
        self.lm = lm

    def get_prob(self, char, context):
        char_id = self.char_to_int[char]
        context_ids = [self.char_to_int[c] for c in context]
        context_ids.insert(0, self.char_to_int[SOS])
        return self.lm.get_prob(char_id, context_ids)

def read_from_file(filename, char_dict, batch_size, max_line_length, sos = SOS, eos = EOS):
    """ reads and preprocess file

    Parameters
    ----------
    filename: string
        Name of the file to read
    char_dict: dictionary
        for unknown keys it should return a default value: e.g.: defaultdict(lambda: 0)
    batch_size: int
        after preprocessing lines in the same batch will have the same length
        data[i * batch_size, (i+1) * batch_size] will be your batch for an integer i
    max_line_length: int
        len(data[i]) <= max_line_length
    Returns
    -------
    data:
        a list of 'lines'
    """
    assert len(sos) == 1 and "Start of Sentence element SOS has to be a char"
    assert len(eos) == 1 and "End of Sentence element EOS has to be a char"

    # read
    data = open(filename, mode="r", encoding="utf-8").readlines()
    # add SOS, EOS remove \n
    data = list(map(lambda x: [sos] + list(x.lower()[:-1]) + [eos], data))
    # map to chars to ids
    data = [list(map(lambda char: char_dict[char], line)) for line in data]
    # cut each line to max_line_length
    data = list(map(lambda x: x[:max_line_length], data))
    # sort by length
    data.sort(key=lambda x: len(x))
    # Split into batches
    # padding of eos to the longest line in the batch, so that each line in a batch has the same length
    number_of_batches = int(len(data)/batch_size)
    eos_id = char_dict[eos]
    for i in range(number_of_batches):
        line_length = len(data[((i+1) * batch_size) - 1])
        for k in range(batch_size):
            idx = i * batch_size + k
            elements_to_pad = line_length - len(data[idx])
            data[idx] += elements_to_pad * [eos_id]
    return data


def get_char_dicts():
    assert UNK_ID == 0 and "change this function if you want to mess with the UNK_ID 0"

    char_to_int = defaultdict(lambda: UNK_ID)
    tokens = SOS + EOS + TOKENS

    for i, char in enumerate(tokens):
        char_to_int[char] = i + 1

    int_to_char = {v: k for k, v in char_to_int.items()}
    int_to_char[UNK_ID] = UNK

    assert len(TOKENS) + 3 == len(int_to_char)
    return char_to_int, int_to_char





