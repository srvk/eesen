from fileutils.kaldi import read_scp_info_dic
from fileutils.kaldi import read_scp_info
from fileutils.kaldi import readScp
import argparse
import sys

#python ./utils/clean_length.py --scp_in  $tmpdir/cv_local.scp --labels $tmpdir/labels.cv --subsampling 3 --scp_out $tmpdir/cv_local.scp


def main_parser():

    #general arguments
    parser = argparse.ArgumentParser(description='deletes the utterances that have more labels than frames (in video format)')
    parser.add_argument('--scp_in', type=str, help='path to input scp file to be cleaned', required=True)
    parser.add_argument('--labels', type=str, help='path to labels files (txt)', required=True)
    parser.add_argument('--scp_out',type=str, help = 'path to input scp output file', required=True)

    return parser

parser = main_parser()
args = parser.parse_args()


#reading labels file
labels_dict = {}
with open(args.labels, "r") as f:
    for line in f:
        tokens = line.strip().split()
        labels_dict[tokens[0]] = len(tokens[1:])


new_len=0
labels_not_found=0
original_scp_len=0

input_scp_dict={}

with open(args.scp_in, "r") as input_scp:

        for input_line_scp in input_scp:
            utt_id=input_line_scp.split()[0]
            path=input_line_scp.split()[1]
            feat_len=int(input_line_scp.split()[2])
            input_scp_dict[utt_id] = (feat_len, input_line_scp)


with open(args.scp_out, "w") as output_scp:

    for utt_id, element in input_scp_dict.items():

        if(utt_id in labels_dict):
            if(labels_dict[utt_id] < element[0] and labels_dict[utt_id] != 0 and element[0] != 0):
                new_len += 1
                output_scp.write(element[1])
        else:
            labels_not_found += 1
            print("")
            print(80 * "*")
            print(80 * "*")
            print("Warning: " +utt_id+ " has not been found in labels file: "+args.labels)
            print(80 * "*")
            print(80 * "*")
            print("")
        original_scp_len += 1


print("cleaning done:")
print("original scp length: " +str(original_scp_len))
print("scp deleted: "+str(original_scp_len-new_len))
print("final scp length: "+str(new_len))
print("number of labels not found: "+str(labels_not_found))
