### for CMU rocks cluster ###

#PBS -q gpu
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -N how_to_concat_char_count
#PBS -V
#PBS -l walltime=48:00:00
#PBS -l nodes=1:ppn=1


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

stage=3

fisher_dirs="/path/to/LDC2004T19/fe_03_p1_tran/ /path/to/LDC2005T19/fe_03_p2_tran/" # Set to "" if you don't have the fisher corpus
eval2000_dirs="/path/to/LDC2002S09/hub5e_00 /path/to/LDC2002T43"

# CMU Rocks
swbd=/data/ASR4/babel/ymiao/CTS/LDC97S62

#TODO fisher datat is currentlu hardcoded (need to deal with that)
fisher_dir_a="/data/ASR5/babel/ymiao/Install/LDC/LDC2004T19/fe_03_p1_tran/"
fisher_dir_b="/data/ASR5/babel/ymiao/Install/LDC/LDC2005T19/fe_03_p2_tran/"


eval2000_dirs="/data/ASR4/babel/ymiao/CTS/LDC2002S09/hub5e_00 /data/ASR4/babel/ymiao/CTS/LDC2002T43"


. parse_options.sh


echo =====================================================================
echo "                Network Training with the 90-Hour Set             "
echo =====================================================================
# Specify network structure and generate the network topology
nlayer=5                 # number of layers
nhidden=200              # cells in each layer and direction
nproj=100
sat_nlayer=2

dir=exp/train_am_char_480h2_l${nlayer}_c${nhidden}_p${nproj}_sat_l${sat_nlayer}_char_count_concat_direct

mkdir -p $dir

echo generating train labels...

#python ./local/swbd1_prepare_char_dict_tf.py --text_file ./data/train_tr95/text --input_units ./data/local/dict_char/units.txt --output_labels $dir/labels.tr --lower_case

echo generating cv labels...

#python ./local/swbd1_prepare_char_dict_tf.py --text_file ./data/train_cv05/text --input_units ./data/local/dict_char/units.txt --output_labels $dir/labels.cv --lower_case

# Train the network with CTC. Refer to the script for details about the arguments
#steps/train_ctc_tf.sh --learn-rate 0.01 --half_after 6  --nlayer $nlayer --nproj $nproj --nhidden $nhidden --batch_size 16 --sat_type concat --sat_stage train_direct --force_lr_epoch_ckpt true --continue_ckpt ./exp/train_am_char_l5_c200_p100/model/epoch08.ckpt --sat_path /data/ASR5/abhinav5/PlacesAlexNet_480h/final_feats  --max_iters 25 --sat_nlayer 2 ./data/train_tr95_fbank/ ./data/train_cv05_fbank/ $dir || exit 1;
#./data/adatped_vectors/PlacesAlexNet_480h.txt

steps/train_ctc_tf.sh --learn-rate 0.02 --half_after 6  --nlayer $nlayer --nproj $nproj --nhidden $nhidden --batch_size 16 --sat_type concat --sat_stage train_direct --sat_path ./data/adatped_vectors/char_counts.txt  --max_iters 25 --sat_nlayer 2 ./data/train_480h_tr95_2/ ./data/train_480h_cv05/ $dir || exit 1;


