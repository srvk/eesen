import argparse
import os
import imageio
import sys
import operator


def main_parser():
    parser = argparse.ArgumentParser(description='Create something like a feats.scp for video')

    parser.add_argument('--feats_scp', help = "reference scp that will be used to create feats.video")

    parser.add_argument('--video_dir', help = "directory where all the videos are")

    parser.add_argument('--output_file', help = "file where all videos will be sotred")

    parser.add_argument('--video_format', help = "video format")


    return parser

def create_config(args):
    config = {
        "feats_scp": args.feats_scp,
        "output_file": args.output_file,
        "video_dir": args.video_dir,
        "video_format": args.video_format
    }
    return config

parser = main_parser()
args = parser.parse_args()
config = create_config(args)


count_not_found=0
original_length=0
with open(config["feats_scp"]) as feats_file, open(config["output_file"],"w") as output_file:
    for line in feats_file:
        folder=line.split(".")[0]
        file=line.split("-")[0]
        final_path =os.path.join(config["video_dir"], folder, file)+"."+config["video_format"]
        if os.path.isfile(final_path):
            reader = imageio.get_reader(final_path)
            number_of_frames=len(reader)
            height = list(reader)[0].shape[0]
            width = list(reader)[0].shape[1]
            output_file.write(line.split()[0]+" "+final_path+" "+str(number_of_frames)+" "+str(height)+" "+str(width)+"\n")
        else:
            count_not_found += 1


print("creation video feats done:")
print("original scp length: " +str(original_length))
print("not found: "+str(count_not_found))
print("final video length: "+str(original_length-count_not_found))




