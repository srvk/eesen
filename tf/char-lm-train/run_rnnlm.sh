### for CMU rocks cluster ###
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -V
#PBS -l walltime=48:00:00
#PBS -q gpu

module load gcc-4.9.2
module load cuda-8.0
export CUDA_VISIBLE_DEVICES=`qstat -n $PBS_JOBID|awk 'END {split ($NF, a, "/"); printf ("%s\n", a[2])}'`
PATH="/data/ASR5/ramons_2/tools/anaconda2/bin/":$PATH
source activate tensorflow_gpu_1_0

echo $HOSTNAME
echo "Running on gpu"
echo "Device = $CUDA_VISIBLE_DEVICES"

train_fil='./data/turkish_train_text'
dev_fil='./data/turkish_dev_text'
lex_fil='./data/lexicon_char_system.txt'
units_fil='./data/units_char_system.txt'
n_epoch=10
#PARAMETERS
batch_size=16
emb_size=64
hidden_size=1000
num_layers=1
drop_emb=1.0


python code/main_rnn.py --batch_size=${batch_size} --embed_size=${emb_size} --drop_emb=${drop_emb} --hidden_size=${hidden_size} --num_layers=${num_layers} --nepoch=${n_epoch} --lexicon_file=${lex_fil} --units_file=${units_fil} --dev_file=${dev_fil} --train_file=${train_fil}
