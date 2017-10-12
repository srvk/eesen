from utils.fileutils.kaldi import read_scp_info_dic
from utils.fileutils.kaldi import read_scp_info
from utils.fileutils.kaldi import readScp
import argparse
import sys

#python ./utils/clean_length.py --scp_in  $tmpdir/cv_local.scp --labels $tmpdir/labels.cv --subsampling 3 --scp_out $tmpdir/cv_local.scp


def main_parser():

    #general arguments
    parser = argparse.ArgumentParser(description='deletes the utterances that have more labels than frames')
    parser.add_argument('--scp_in', type=str, help='path to input scp file to be cleaned', required=True)
    parser.add_argument('--labels', type=str, help='path to labels files (txt)', required=True)
    parser.add_argument('--subsampling', type=int, default=0, help='augmentation subsampling that will be posteriorly applied ', required=True)
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

#reading scp textfile
dict_text={}
with open(args.scp_in, "r") as f:
    for line in f.readlines():
        dict_text[line.split()[0]] = line.split()[1]

dict_scp=read_scp_info_dic(args.scp_in)
original_scp_len = len(dict_scp)

new_len=0
labels_not_found=0
with open(args.scp_out, "w") as f:
    for key, element in dict_scp.items():
        uttid_aux, arkfile, offset, feat_len, feat_dim  = element
        if(args.subsampling > 1):
            feat_len = int(float(feat_len) / float(args.subsampling))

        if(key in labels_dict):
            if(labels_dict[key] < feat_len and labels_dict[key] != 0):
                new_len += 1
                f.write(str(key)+" "+str(dict_text[key])+"\n")
        else:
            labels_not_found += 1
            print("")
            print(80 * "*")
            print(80 * "*")
            print("Warning: " +key+ " has not been found in labels file: "+args.labels)
            print(80 * "*")
            print(80 * "*")
            print("")



print("cleaning done:")
print("original scp length: " +str(original_scp_len))
print("scp deleted: "+str(original_scp_len-new_len))
print("final scp length: "+str(new_len))
print("number of labels not found: "+str(labels_not_found))
