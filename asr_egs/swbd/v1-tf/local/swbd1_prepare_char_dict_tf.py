import argparse
import sys
import operator
import re


SPACE="<space>"
EOS="<eos>"
#all numbers in swbd+fisher ['ak-47', '401ks', 'vh1', 'v8', 'v8s', 'm16s', 'dc3s', 'm16', 'y2k', 'mp3', 'f16', 'f16s', '401k', '3d', '90210', 'ak47', 'ak47s', 's2b', '2.', 'espn2', 'u2', '747', '401-', '21', '20/20', '48', '49ers', 'gt1', 'f-16', 'b2', '150', '7-up', 's10', 'rx7', '365', '7-eleven', "them_1's", '128', '60', 'ak-47s', '8088', 'i-30', "635's", '13th', '13ths', 'b6', 'e5', '2', '4h', 'a1', '287', 'i-70', 'i-25', '1200', '101', '1', '100', '7094', 'w-4', 'w-2', '2000', '911', '635', 'v-6', '286', '380', '486', '386', '500', 'v-8', 'h2s', '49er', 's-10', '4-runner', 's-10s', "401k's", '125k', '69', '4o-', '-1k', 'catch-22', '10a', 'pac-10', '40-', 'v-8s', '6s', '302', '5', 'xt6', 'z248', '9050', '990', '1500', '260', '2ci']

dict_numbers={'90210': 'ninety thousand two hundred ten', '9050':'nine thousand fifty', '8088':'eight thousand eighty eight','7094': 'seven thousand ninety four', '2000': 'two thousand', '1500':'one thousand five hundred', '1200':'one thousand two hundred', '990':'nine hundred ninety', '911':'nine eleven', '747': 'seven hundred forty seven', '635': 'six hundred thirty five', '500':'five hundred', '486': 'four hundred eighty six', '386': 'three hundred eighty six', '401': 'four hundred one','386':'three hundred eighty six', '380':'three hundred eighty','365': 'three hundred sixty five','302': 'three hundred two', '287': 'two hundred eighty seven','260': 'two hundred sixty', '248': 'two hundred forty eight', '286': 'two hundred eighty six', '150' : 'one hundred fifty', '128' : 'one hundred twenty eight','125' : 'one hundred twenty five', '101': 'one hundred one', '100': 'one hundred', '70':'seventy','69':'sixty nine', '60' : 'sixty', '49': 'forty nine', '48': 'forty eight', '47': 'forty seven','40': 'forty',  '30': 'thirty', '25':'twenty five','22': 'twenty two', '21': 'twenty one', '20': 'twenty', '16': 'sixteen', '13':'thirteen', '10': 'ten', '8':'eight', '7':'seven', '6':'six', '5':'five', '4':'four',  '3': 'three', '2': 'two', '1': 'one'}

def main_parser():
    parser = argparse.ArgumentParser(description='Create units.txt (if not give) and labels file for AM or LM (needs to be specified by --lm)')

    parser.add_argument('--text_file', help = "input text file that will be translated to labels")
    parser.add_argument('--input_units', help = "path of previous units.txt (most probably from an acoustic model)")
    parser.add_argument('--output_labels', default="", help = "path to output labels")
    parser.add_argument('--output_units', default="", help = "path to out units (optional)")

    parser.add_argument('--lower_case', default = False, action='store_true', help='change to lower case')
    parser.add_argument('--upper_case', default = False, action='store_true', help='change to upper case')

    parser.add_argument('--ignore_characters', default="", help='ignore character listed separated by comma: --ignore_characters \"*,~,_\"')

    parser.add_argument('--ignore_noises', default = False, action='store_true', help='ignore all noises e.g. [noise], [laughter], [vocalized-noise]')

    parser.add_argument('--convert_numbers', default = False, action='store_true', help='ignore numbers and convert them to letters e.g. 2-> two')

    parser.add_argument('--no_space', default = False, action='store_true', help='ignore numbers and convert them to letters e.g. 2-> two')

    parser.add_argument('--convert_and', default = False, action='store_true', help='conver & to letter representation i.e. &-> and')

    return parser

def create_config(args):
    config = {
        "input_units": args.input_units,
        "lower_case":args.lower_case,
        "upper_case":args.upper_case,
        "text_file":args.text_file,
        "output_labels":args.output_labels,
        "output_units":args.output_units,
        "ignore_noises": args.ignore_noises,
        "is_lm": args.lm,
        "ignore_characters": args.ignore_characters,
        "convert_numbers": args.convert_numbers,
        "convert_and": args.convert_and,
	"no_space":args.no_space
    }

    return config

def get_units(units_path):
    units_dict={}

    with open(units_path) as f:
        for line in f:
            if(line.split()[0] == "<space>"):
            	units_dict[" "]=line.split()[1]
            else:
            	units_dict[line.split()[0]]=line.split()[1]

    return units_dict

def generate_labels_am(config, text_path, units_dict, output_labels_path):

    removed_utterances = 0
    total_lines = 0
    clean_lines = 0

    with open(text_path,"r") as input_text, open(output_labels_path,"w") as output_labels:

        for line in input_text:
            total_lines += 1
            utt_id = line.split()[0]
            new_line = utt_id
            final_idx = len(line.split()[1:])-1
            for word_idx, word in enumerate(line.split()[1:]):

                if(("[" in word) and ("]" in word)):
                    if(word in units_dict):
                        new_line += " " + str(units_dict[word])

                else:
                    if(config["convert_and"]):
                        word=word.replace("&"," and ")

                    if(config["convert_numbers"]):

                        numbers = map(int, re.findall(r'\d+', word))

                        if(len(numbers) > 0):
                            for number in numbers:
                                word=word.replace(str(number),dict_numbers[str(number)])

                    if(config["upper_case"] or config["lower_case"]):
                        word = process_string(config, word)

                    if(word_idx != final_idx):
                        word = word+" "
                    for letter in word:
                        if(letter in units_dict):
                            new_line += " " + str(units_dict[letter])


            if(len(new_line.split()) > 1):
                output_labels.write(new_line+"\n")
                clean_lines += 1
            else:
                removed_utterances += 1

        print(80 * "-")
        print("Summary of the conversion to labels: ")
        print(80 * "-")
        print("file cleaned: "+str(text_path))
        print("number total utterances: "+str(total_lines))
        if(removed_utterances > 0):
            print("number utt removed: %d (this is maybe due noises: [laughter], [noise], [vocalized-noise])"% (removed_utterances))
        else:
            print("number utt removed: %d"% (removed_utterances))
        print("number remaining utt: "+str(clean_lines))
        print(80 * "-")



def process_string(config, string):

        if(config["lower_case"] and not config["upper_case"]):
            string = string.lower()

        elif(config["upper_case"] and not config["lower_case"]):
            string = string.upper()

        return string

def generate_units_am(config, text_path, output_units_path, ignore_characters):


    dict_untisid={}
    count_id = 1
    with open(text_path,"r") as input_text, open(output_units_path,"w") as output_labels:
        for line in input_text:
            if(config["convert_and"]):
                line=line.replace("&"," and ")

            for word in line.split()[1:]:
                if(("[" in word) and ("]" in word)):
                    if(word not in dict_untisid and not config["ignore_noises"]):
                        dict_untisid[word]=count_id
                        count_id+=1
                else:
                    if(config["convert_numbers"]):
                        numbers = map(int, re.findall(r'\d+', word))

                        if(len(numbers) > 0):
                            for number in numbers:
                                word=word.replace(str(number),dict_numbers[str(number)])

                    if (config["lower_case"] or config["upper_case"]):
                        word = process_string(config, word)

                    for letter in word:
                        if(letter not in dict_untisid):
                            if(letter not in ignore_characters):
                                    dict_untisid[letter]=count_id
                                    count_id+=1

        sorted_dict = sorted(dict_untisid.items(), key=operator.itemgetter(0))
        new_count = 1
        for element in sorted_dict:
            dict_untisid[element[0]] = new_count
            if(" " == str(element[0])):
                output_labels.write("<space> "+str(new_count) + "\n")
            else:
                output_labels.write(str(element[0])+" "+str(new_count) + "\n")
            new_count += 1

    return dict_untisid

gen_units = False
parser = main_parser()
args = parser.parse_args()
config = create_config(args)

config["ignore_characters"]=config["ignore_characters"].split("|")

if(config["no_space"]):
    config["ignore_characters"].append(" ")


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
    if(config["output_units"]):
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
    dict_units = generate_units_am(config, config["text_file"], config["output_units"], config["ignore_characters"])

generate_labels_am(config, config["text_file"], dict_units, config["output_labels"])



