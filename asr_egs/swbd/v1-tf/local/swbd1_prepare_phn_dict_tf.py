import argparse
import sys
import operator


def main_parser():
    parser = argparse.ArgumentParser(description='Create units.txt (if not give) and labels file for AM or LM (needs to be specified by --lm)')

    parser.add_argument('--text_file', help = "input text file that will be translated to labels")

    parser.add_argument('--phn_lexicon', help = "path of previous units.txt (most probably from an acoustic model)")

    parser.add_argument('--output_labels', help = "path to output labels")

    parser.add_argument('--output_units', help = "path to output labels")

    parser.add_argument('--ignore_noises', default = False, action='store_true', help='ignore all noises e.g. [noise], [laughter], [vocalized-noise]')

    return parser


def create_config(args):
    config = {
        "text_file":args.text_file,
        "phn_lexicon":args.phn_lexicon,
        "output_units":args.output_units,
        "output_labels":args.output_labels,
        "ignore_noises":args.ignore_noises
    }
    return config


gen_units = False
parser = main_parser()
args = parser.parse_args()
config = create_config(args)

dict_phn={}
lexicon_phn={}
count_phn = 1

with open(config["phn_lexicon"],"r") as f, open(config["output_units"],"w") as output_units:

    for line in f:

            if config["ignore_noises"] and (("[" in  line.split()[0] and "]" in line.split()[0]) or ("<" in  line.split()[0] and ">" in line.split()[0])):
                continue

            new_line = ""
            for phn in line.replace("\n","").split()[1:]:
                if phn not in dict_phn:
                    dict_phn[phn] = count_phn
                    count_phn += 1
                new_line = new_line + " " + str(dict_phn[phn])
            lexicon_phn[line.split()[0]] = new_line

    sorted_dict_phn = sorted(dict_phn.items(), key=operator.itemgetter(1))

    for element in sorted_dict_phn:
        output_units.write(element[0]+" "+str(element[1])+"\n")

with open(config["text_file"],"r") as input_text, open(config["output_labels"],"w") as output_labels:

    for text_line in input_text:
        new_line=text_line.split()[0]

        for word in text_line.split()[1:]:
            if(word in lexicon_phn):
                new_line = new_line  + lexicon_phn[word]

        output_labels.write(new_line+"\n")



