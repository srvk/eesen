### for CMU rocks cluster ###
#PBS -j oe
#PBS -k oe
#PBS -o log
#PBS -d .
#PBS -V
#PBS -l walltime=48:00:00
#PBS -N DeepBiRNN 
#PBS -q gpu

module load gcc-4.9.2
module load cuda-8.0
export CUDA_VISIBLE_DEVICES=`qstat -n $PBS_JOBID|awk 'END {split ($NF, a, "/"); printf ("%s\n", a[2])}'`
#PATH="/data/ASR5/ramons_2/tools/anaconda2/bin/":$PATH
#source activate p_3
PATH="/data/ASR1/ramons/anaconda2/bin":$PATH
source activate tensorflow_gpu

echo $HOSTNAME
echo "Running on gpu"
echo "Device = $CUDA_VISIBLE_DEVICES"

# swbd, cudnn, cer around 0.13
python main.py --store_model --nlayer=5 --nproj=120 --nhidden=320

# haitian, cudnn, cer around 0.35
#python main.py --store_model --data_dir=/home/bchen2/Haitian/data --nlayer=6 --nhidden=140 --nproj=60

# haitian, native
#python main.py --store_model --data_dir=/home/bchen2/Haitian/data --nlayer=6 --nhidden=140 --nproj=60 --lstm_type=native

# haitian fuse
#python main.py --store_model --data_dir=/home/bchen2/Haitian/data --nlayer=6 --nhidden=140 --nproj=60 --lstm_type=fuse

#python main.py --eval --eval_model=/home/bchen2/Haitian/log/dbr-run9/model/epoch11.ckpt
