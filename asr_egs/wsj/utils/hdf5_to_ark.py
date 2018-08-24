import argparse
import h5py
from fileutils import kaldi_io

def main_parser():

    parser = argparse.ArgumentParser(description="Reads an hdf5 files file containing a dictionary of utt_ids and float matrices")

    parser.add_argument('--hdf5', type=str, help="hdf5 file to be read", required=True)

    parser.add_argument('--ark', type=str, help="name of the output file, output will be written to ${ark}.ark or to ${ark}${number}.ark if multiple files are written", required=True)

    parser.add_argument('--utts_file', type=int, default=1, help="number of utterances that will be written into each ark file", required=True)

    return parser

if __name__ == "__main__":

    parser = main_parser()
    args = parser.parse_args()

    with h5py.File(args.hdf5, 'r') as h5:
        number_utts = len(h5)
        print("File contains "+str(number_utts)+" utterances")

        if number_utts <= args.utts_file:
            print("Creating a single ark file")
            with open(args.ark + ".ark", 'wb') as f:
                for key, mat in h5.items():
                    kaldi_io.write_mat(f, mat[:], key=key)
        else:
            print("Creating multiple ark files")
            # slow, because we open the files multiple times
            for utt_id, (key, mat) in enumerate(h5.items()):
                ark_id = int(utt_id / args.utts_file)
                with open("{}{}.ark".format(args.ark, ark_id), 'ab') as f:
                    kaldi_io.write_mat(f, mat[:], key=key)

