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
    parser.add_argument('--deduplicate', default=False, help='keep duplicates or not', required=False, action='store_true')
    parser.add_argument('--subsampling', type=int, default=3, help='how much subsampling to take into account', required=False)
    parser.add_argument('--scp_out',type=str, help = 'path to input scp output file', required=True)

    return parser

parser = main_parser()
args = parser.parse_args()

#reading labels file
labels_dict = {}
labels_tcid = {}
with open(args.labels, "r") as f:
    for line in f:
        tokens = line.strip().split()
        key = tokens[0]
        trl = ''.join(map(str, tokens[1:]))
        if args.deduplicate and trl in labels_tcid:
            labels_dict[key] = 0
        else:
            labels_tcid[trl] = key
            labels_dict[key] = len(tokens)-1

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



print("cleaning done: "+args.scp_out)
print("original scp length: " +str(original_scp_len))
print("scp deleted: "+str(original_scp_len-new_len))
print("final scp length: "+str(new_len))
print("number of labels not found: "+str(labels_not_found))
