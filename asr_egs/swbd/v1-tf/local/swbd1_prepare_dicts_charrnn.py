import argparse

def main_parser():
    parser = argparse.ArgumentParser(description='Create units.txt (if not give) and labels file from a char AM for switchboard database')

    parser.add_argument('--text_file', default="", help = "lstm type: cudnn, fuse, native")
    parser.add_argument('--in_units', default="", help = "lstm type: cudnn, fuse, native")
    parser.add_argument('--output_labels', default="", help = "lstm type: cudnn, fuse, native")
    parser.add_argument('--out_units', default="", help = "lstm type: cudnn, fuse, native")

    return parser

def create_config(args):
    config = {
        "text_file": args.text_file,
        "in_units": args.in_units,
        "nepoch": args.output_labels,
        "lr_rate": args.out_units
    }
    return config

def get_units(units_path):
    units_dict={}

    with open(units_path) as f:
        for line in f:
            units_dict[line.split()[0]]=line.split()[1]

    return units_dict

def generate_labels_am(text_path, units_dict, output_labels_path):
    with open(units_path,"r") as input_text, open(output_labels_path,"w") as output_labels:
        for line in input_text:
            utt_id = line.split()[0]
            new_line = utt_id
            for word in  line.split()[1:]:
                if(("[" in word) and ("]" in word))
                    new_line += " " + units_dict[word] 
                else:
                    for letter in word:
                        units_dict[letter] = " " + units_dict[letter]

            output_labels.write(new_line+"\n")

def generate_units_am(text_path, units_dict, output_labels_path):
    dict_units
    with open(units_path,"r") as input_text, open(output_labels_path,"w") as output_labels:
        for line in input_text:
            for word in  line.split()[1:]:
                if(("[" in word) and ("]" in word))
                    if(word not in )
                else:
                    for letter in word:
                        units_dict[letter] = " " + units_dict[letter]

            output_labels.write(new_line+"\n")

def generate_units_lm(text_path, units_dict, output_labels_path):

generate_units = False
parser = main_parser() 
args = parser.parse_args()
config = create_parser(args) 

if(not config["text_file"]):
    print("Error: text file is needed in order to generate labels file")
    sys.exit()

if(not config["in_units"]):
    if(not config["out_units"]):
        print("Error: either in_units or out_units should be defined")
        sys.exit()
    else:
        print("Generating units file....")
        generate_units = True
else:
    if(not config["out_units"]):
        print("Using: "+config["in_units"]+" as units reference")
        generate_units = False

if(not config["output_labels"]):
    print("Error: output label files is needed in order to write labels somewhere")
    sys.exit()

if(not generate_units):
    dict_units = get_units(config["in_units"])
else:
    dict_units = generate_units(config["text_file"])





