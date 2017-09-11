import argparse
import sys
import operator


def main_parser():
    parser = argparse.ArgumentParser(description='Create units.txt (if not give) and labels file for AM or LM (needs to be specified by --lm)')

    parser.add_argument('--dict_af', help = "transcription from phnemes to af")

    parser.add_argument('--lexicon_phn', help = "lexicon that contains word transcriptons in phonemes")

    parser.add_argument('--ignore_noises', default = False, action='store_true', help='ignore all noises e.g. [noise], [laughter], [vocalized-noise]')

    parser.add_argument('--input_text', help = "input text that should be transcribed to ")

    parser.add_argument('--output_units', help = "output file for units file")

    parser.add_argument('--output_labels', help = "label output file")


    return parser

def create_config(args):
    config = {
        "dict_af": args.dict_af,
        "lexicon_phn": args.lexicon_phn,
        "ignore_noises": args.ignore_noises,
        "input_text": args.input_text,
        "output_units": args.output_units,
        "output_labels": args.output_labels,
    }
    return config


parser = main_parser()
args = parser.parse_args()
config = create_config(args)

dict_af={}
dict_af_num={}

count = 1
with open(config["dict_af"]) as input_dict_af:
    for line in input_dict_af:
        if (line.split()[1] != "#"):
            dict_af[line.split()[0]] = " ".join(line.split()[1 :])
            for af in line.split()[1:]:
                if(af not in dict_af_num):
                    dict_af_num[af] = count
                    count = count + 1

word_to_num_af = {}

with open(config["lexicon_phn"], "r") as lexicon_phn:
    for phn_line in lexicon_phn:

        word = phn_line.split()[0]
        new_line = ""

        for phn in phn_line.split()[1:]:
            if phn.lower() in dict_af:
                for af_element in dict_af[phn.lower()].split():
                    new_line = new_line + str(dict_af_num[af_element]) + " "

        word_to_num_af[word] = new_line[:-1]

with open(config["input_text"], "r") as input_text, open(config["output_labels"], "w") as output_labels:
    for sentence in input_text:
        new_line  = sentence.split()[0]+ " "

        for word in sentence.split()[1:]:
            if(word in word_to_num_af):
                new_line = new_line + word_to_num_af[word] + " "

        new_line = new_line[:-1] + "\n"
        output_labels.write(new_line)

sorted_dict_af_num = sorted(dict_af_num.items(), key=operator.itemgetter(1))

with open(config["output_units"], "w") as units_file:
    for element in sorted_dict_af_num:
        units_file.write(element[0] + " " + str(element[1]) + "\n")

