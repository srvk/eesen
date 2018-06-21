export LC_ALL=C

if [[ `uname -n` =~ ip-* ]]; then
  # AWS instance
  export EESEN_ROOT=/data/ASR5/fmetze/eesen-block-copy
  export LD_LIBRARY_PATH=/data/ASR5/fmetze/eesen-block-copy/src/lib:/data/ASR5/fmetze/eesen-block-copy/tools/openfst/lib:$LD_LIBRARY_PATH
  #export EESEN_ROOT=${HOME}/eesen
  export PATH=${PWD}/meine:${PWD}/utils:${HOME}/kenlm/bin:${EESEN_ROOT}/tools/sph2pipe_v2.5:${EESEN_ROOT}/tools/openfst/bin:${EESEN_ROOT}/src/featbin:${EESEN_ROOT}/src/decoderbin:${EESEN_ROOT}/src/fstbin:${EESEN_ROOT}/src/netbin:$PATH
  export PYTHONPATH=/data/ASR5/fmetze/eesen-tf/tf/ctc-train:/data/ASR5/fmetze/eesen-tf/tf/ctc-decode/code:/data/ASR5/fmetze/eesen-tf/tf/rnn-lm/code

  export BABEL_DATA=/media/s3fs
  #if [ `df -B1073741824 --output=avail /media/ephemeral0 | tail -n 1` -gt 7 ]; then
  #  export TMPDIR=/media/ephemeral0
  #else
  #  export TMPDIR=/dev/shm
  #fi
  #if [ -n "$PWDTMP" ]; then
  #  export TMPDIR=.
  #fi

elif [[ `uname -n` =~ instance ]]; then
  # Google Cloud
  export EESEN_ROOT=/data/ASR5/fmetze/eesen-block-copy
  export KALDI_ROOT=/data/ASR5/fmetze/eesen-block-copy
  export PATH=$PWD/utils/:$EESEN_ROOT/src-google/netbin:$EESEN_ROOT/src-google/featbin:$EESEN_ROOT/src-google/decoderbin:$EESEN_ROOT/src-google/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/sph2pipe_v2.5:$EESEN_ROOT/../kaldi-latest/src/featbin:$PATH

  export TMPDIR=/scratch
  export BABEL_DATA=/data/MM3/babel-corpus

elif [[ `uname -n` =~ bridges ]]; then


  # PSC Bridges cluster
  #module load atlas
  ##module load cuda
  ##module loaf cuda/8.0RC
  ##module cuda/7.5
  #module cuda/9.0RC

  #module load gcc/6.3.0

  export PYTHONPATH=/pylon2/ir3l68p/sanabria/eesen/tf/ctc-am

  export EESEN_ROOT=/pylon2/ir3l68p/metze/eesen/
  export PATH=$PATH:$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/sph2pipe_v2.5:$EESEN_ROOT/../kaldi/src/featbin:$EESEN_ROOT/../sox-14.4.2/src:$PWD

  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/packages/cuda/8.0/lib64/:/opt/packages/cuda/8.0/extras/:/opt/packages/cuda/8.0/include/:/opt/packages/cuda/8.0/targets/x86_64-linux/lib/stubs/:/opt/packages/cuda/8.0/doc/man/man7/


  #export TMPDIR=/pylon1/ir3l68p/metze
  export TMPDIR=$SCRATCH
  #export TMPDIR=.
  export BABEL_DATA=/pylon2/ir3l68p/metze/babel-corpus

  unset CUDA_VISIBLE_DEVICES
  unset GPU_DEVICE_ORDINAL
  #export CUDA_VISIBLE_DEVICES=0




elif [[ `uname -n` =~ comet* ]]; then
  # SDSC Comet cluster
  module load atlas
  module load lapack

  export EESEN_ROOT=../../../../../eesen
  #. /export/babel/data/software/env.sh
  export PATH=$EESEN_ROOT/tools/sph2pipe_v2.5:$EESEN_ROOT/tools/irstlm/bin/:$PWD/utils/:$EESEN_ROOT/tools/sox:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/src/fstbin/:$EESEN_ROOT/src/decoderbin/:$EESEN_ROOT/src/featbin/:$EESEN_ROOT/src/netbin:/home/fmetze/tools/kaldi-trunk/src/featbin:$EESEN_ROOT/tools/srilm/bin/i686-m64:$EESEN_ROOT/tools/srilm/bin:$PATH

  export PATH=./local:./utils:./steps:$PATH
  [ -d /scratch/${USER}/${SLURM_JOBID} ] && export TMPDIR=/scratch/${USER}/${SLURM_JOBID}
  export BABEL_DATA=/oasis/projects/nsf/cmu131/fmetze/babel-corpus

  export PYTHONPATH=/pylon2/ir3l68p/sanabria/eesen/tf/ctc-am


elif [[ `uname -n` =~ islpc* ]]; then

  # islpc-cluster
  export BABEL_DATA=/data/MM3/babel-corpus
  export PYTHONPATH=/data/ASR5/ramons_2/sinbad_projects/eesen_new/tf-exp/ctc-am
  export EESEN_ROOT=/data/MM23/sdalmia/eesen
  export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/sph2pipe_v2.5:/data/ASR5/fmetze/kaldi-latest/src/latbin:/data/ASR5/fmetze/kaldi-latest/src/featbin:$PATH

  if [ `uname -n` == "islpc17" ] || [ `uname -n` == "islpc18" ] ||  [ `uname -n` == "islpc19" ] ||   [ `uname -n` == "islpc20" ] ; then
    source /data/ASR5/ramons_2/sinbad_projects/myenvs/tf16_p27_islpc_neutrum/bin/activate
    export LD_LIBRARY_PATH="/data/ASR5/ramons_2/sinbad_projects/cudnn/cudnn-9.0-linux-x64-v7/lib64/:/usr/local/cuda-9.0/lib64/"

  else
    #this works with: 33, 34, 35, 36
   source /data/ASR5/ramons_2/sinbad_projects/myenvs/tf18_p27_islpc/bin/activate
  export LD_LIBRARY_PATH="/data/ASR5/ramons_2/sinbad_projects/cudnn/cudnn-9.0-linux-x64-v7/lib64/:/usr/local/cuda-9.0/lib64/"

  fi


else
  # CMU Rocks cluster
  module load python27
  module load gcc-4.9.2
  module load cuda-8.0

  [ -n "$PBS_JOBID" ] && export CUDA_VISIBLE_DEVICES=`qstat -n $PBS_JOBID|awk 'END {split ($NF, a, "/"); printf ("%s\n", a[2])}'`
  [ -n "$PBS_JOBID" ] && export THEANO_FLAGS="device=`qstat -n $PBS_JOBID | tail -n 1 | sed 's|.*/|gpu|g'`"

  export TMPDIR=/scratch

  export BABEL_DATA=/data/MM23/sdalmia/eval_lorelei/il5_tig_set1_tts
  export KALDI_ROOT=/data/ASR1/tools/kaldi
  export EESEN_ROOT=/data/MM23/sdalmia/eesen

  export PYTHONPATH=/data/ASR5/ramons_2/sinbad_projects/eesen/tf/ctc-am

  #. /export/babel/data/software/env.sh
  export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/sph2pipe_v2.5:/home/fmetze/tools/kaldi/src/bin:/data/ASR5/fmetze/kaldi-latest/src/latbin:$PATH

  export PATH=/data/ASR5/ramons_2/tools/anaconda2/bin:$PATH
  source activate tensorflow_gpu_1_2

fi

[ -f ${EESEN_ROOT}/tools/env.sh ] && . ${EESEN_ROOT}/tools/env.sh
[ -f ./local.sh ] && . ./local.sh
