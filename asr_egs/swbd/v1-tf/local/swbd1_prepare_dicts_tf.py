import argparse
import sys
import operator


SPACE="<space>"
EOS="<eos>"

def main_parser():
    parser = argparse.ArgumentParser(description='Create units.txt (if not give) and labels file for AM or LM (needs to be specified by --lm)')

    parser.add_argument('--text_file', default="", help = "input text file that will be translated to labels")
    parser.add_argument('--input_units', default="", help = "path to input units.txt (optional)")
    parser.add_argument('--output_labels', default="", help = "path to output labels")
    parser.add_argument('--output_units', default="", help = "path to out units (optional)")
    parser.add_argument('--lm', default = False, action='store_true', help='are we working for LM?')

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

def get_units(units_path):
    units_dict={}

    with open(units_path) as f:
        for line in f:
            units_dict[line.split()[0]]=line.split()[1]

    return units_dict

def generate_labels_am(text_path, units_dict, output_labels_path):

    with open(text_path,"r") as input_text, open(output_labels_path,"w") as output_labels:

        for line in input_text:
            utt_id = line.split()[0]
            new_line = utt_id
            for word in  line.split()[1:]:
                if(("[" in word) and ("]" in word)):
                    new_line += " " + str(units_dict[word])
                else:
                    for letter in word:
                        new_line += " " + str(units_dict[letter])
            output_labels.write(new_line+"\n")

def process_string(config, string):

        if(config["lower_case"] and not config["upper_case"]):
            string = string.lower()

        elif(config["upper_case"] and not config["lower_case"]):
            string = string.upper()

        return string

def generate_labels_lm(config, text_path, units_dict, output_labels_path):

    with open(text_path,"r") as input_text, open(output_labels_path,"w") as output_labels:
        for line in input_text:
            utt_id = line.split()[0]
            new_line = utt_id + " " + str(units_dict[EOS])
            for word in line.split()[1:]:
                if(("[" in word) and ("]" in word)):
                    if(word in units_dict):
                        new_line += " " + str(units_dict[word]) + " " + str(units_dict[SPACE])
                else:
                    if(config["upper_case"] or config["lower_case"]):
                        word = process_string(config, word)
                    for letter in word:
                        if(letter in units_dict):
                            new_line += " " + str(units_dict[letter])
                    new_line +=  " " + str(units_dict[SPACE])
            new_line=new_line[:-len(str(units_dict[SPACE]))] + str(units_dict[EOS])
            output_labels.write(new_line+"\n")

def generate_units_am(config, text_path, output_units_path):
    dict_untisid={}
    count_id = 1
    with open(text_path,"r") as input_text, open(output_units_path,"w") as output_labels:
        for line in input_text:

            for word in  line.split()[1:]:
                if(("[" in word) and ("]" in word)):
                    if(word not in dict_untisid):
                        dict_untisid[word]=count_id
                        count_id+=1
                else:
                    if (config["lower_case"] or config["upper case"]):
                        word = process_string(config, word)
                    for letter in word:
                        if(letter not in dict_untisid):
                            dict_untisid[letter]=count_id
                            count_id+=1

        sorted_dict = sorted(dict_untisid.items(), key=operator.itemgetter(1))
        for element in sorted_dict:
            output_labels.write(str(element[0])+" "+str(element[1]) + "\n")

    return dict_untisid

def generate_units_lm(config, input_units_path, output_units_path):

    units_dict={}

    max_count=1
    with open(input_units_path) as f, open(output_units_path,"w") as output_units:

        for line in f:
            units_dict[line.split()[0]]=line.split()[1]
            max_count= max(max_count,int(line.split()[1]))
            output_units.write(line)

        space_idx = max_count + 1
        units_dict[SPACE] = space_idx
        output_units.write(SPACE+" "+str(space_idx)+"\n")

        eos_idx = max_count + 2
        units_dict[EOS] = eos_idx
        output_units.write(EOS+" "+str(eos_idx)+"\n")

    return units_dict

gen_units = False
parser = main_parser()
args = parser.parse_args()
config = create_config(args)

if(not config["text_file"]):
    print("Error: text file is needed in order to generate labels file")
    sys.exit()

if(not config["input_units"]):
    if(not config["output_units"]):
        print("Error: either input_units or output_units should be defined")
        sys.exit()
    else:
        print("generating units file....")
        gen_units = True
#inputs provided
else:
    if(config["output_units"] and config["is_lm"]):
        print("using: "+config["input_units"]+" as units reference and augmenting with <SPACE> and <EOS>")
        gen_units=True
    else:
        print("using: "+config["input_units"]+" as units reference")
        gen_units = False


if(not config["output_labels"]):
    print("Error: output label files is needed in order to write labels somewhere")
    sys.exit()

if(not gen_units):
    dict_units = get_units(config["input_units"])
else:
    if(config["is_lm"]):
        dict_units = generate_units_lm(config, config["input_units"], config["output_units"])
    else:
        dict_units = generate_units_am(config, config["text_file"], config["output_units"])


if(config["is_lm"]):
    generate_labels_lm(config, config["text_file"], dict_units, config["output_labels"])
else:
    generate_labels_am(config, config["text_file"], dict_units, config["output_labels"])



