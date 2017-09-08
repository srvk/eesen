import argparse
import sys
import operator

def main_parser():
    parser = argparse.ArgumentParser(description='Create units.txt (if not give) and labels file for AM or LM (needs to be specified by --lm)')

    parser.add_argument('--text_file', help = "input text file that will be translated to labels")

    parser.add_argument('--output_word_list', help = "path to out units (optional)")

    parser.add_argument('--ignore_noises', default = False, action='store_true', help='ignore all noises e.g. [noise], [laughter], [vocalized-noise]')

    return parser


def create_config(args):
    config = {
        "text_file": args.text_file,
        "output_word_list": args.output_word_list,
        "ignore_noises": args.ignore_noises,
    }
    return config

parser = main_parser()
args = parser.parse_args()
config = create_config(args)

dict_words={}
with open(config["text_file"]) as input_file:
    for line in input_file:
        for word in line.split()[1:]:
            if(word not in dict_words):
                if("[" in word and "]" in word):
                    if(not config["ignore_noises"]):
                        dict_words[word]=""
                else:
                    dict_words[word]=""


with open(config["output_word_list"], "w") as output_word_list:
    for key in dict_words.keys():
        output_word_list.write(key+"\n")

