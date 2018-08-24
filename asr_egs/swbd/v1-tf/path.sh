export LC_ALL=C

export TMPDIR=/scratch/tmp/fosler
export EESEN_ROOT=`pwd`/../../../

export PYTHONPATH=$EESEN_ROOT/tf/ctc-am

export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/sph2pipe_v2.5:$PATH

export PATH=/u/drspeech/opt/anaconda3/bin:$PATH
source activate tensorflow_1.3_gpu

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/drspeech/opt/cuda-8.0/lib64:/u/drspeech/opt/cudnn6/lib64


[ -f ${EESEN_ROOT}/tools/env.sh ] && . ${EESEN_ROOT}/tools/env.sh
[ -f ./local.sh ] && . ./local.sh
/bin/true  # so that return is correct
