import argparse
import sys
import operator

SPACE="<space>"
EOS="<eos>"

def generate_units()


def main_parser():
    parser = argparse.ArgumentParser(description='Create units.txt (if not give) and labels file for AM or LM (needs to be specified by --lm)')

    parser.add_argument('--text_files', help = "input text file that will be translated to labels")
    parser.add_argument('--input_units', help = "input path to input units.txt (usually units.txt from the AM)")
    parser.add_argument('--output_units', help = "output path for units.txt (in this units we will add <space> <eos> and all different labels that we find in text files")

    parser.add_argument('--lower_case', default = False, action='store_true', help='change to lower case')
    parser.add_argument('--upper_case', default = False, action='store_true', help='change to upper case')

    return parser

def create_config(args):
    config = {
        "text_file": args.text_file,
        "input_units": args.input_units,
        "output_units": args.output_units,
        "output_labels": args.output_labels,
        "is_lm": args.lm,
        "lower_case":args.lower_case,
        "upper_case":args.upper_case,
    }
    return config

dict_units = generate_units(config, config["input_units"], config["output_units"])



















