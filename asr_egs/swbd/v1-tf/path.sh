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
  export PATH=$PWD/utils/:$EESEN_ROOT/src-google/netbin:$EESEN_ROOT/src-google/featbin:$EESEN_ROOT/src-google/decoderbin:$EESEN_ROOT/src-google/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/sph2pipe_v2.5:$EESEN_ROOT/../kaldi-latest/src/featbin:$PATH

  export TMPDIR=/scratch
  #export TMPDIR=.
  export BABEL_DATA=/data/MM3/babel-corpus

elif [[ `uname -n` =~ bridges ]]; then
  # PSC Bridges cluster
  module load atlas
  module load cuda
  module load gcc/6.3.0


  export EESEN_ROOT=$SCRATCH
  export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/sph2pipe_v2.5:$EESEN_ROOT/../kaldi/src/featbin:$EESEN_ROOT/../sox-14.4.2/src:$PWD:$PATH

  export TMPDIR=/pylon1/ir3l68p/metze
  #export TMPDIR=$LOCAL
  #export TMPDIR=.
  export BABEL_DATA=/pylon2/ir3l68p/metze/babel-corpus

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

elif [[ `uname -n` =~ islpc* ]]; then
  # islpc-cluster
  export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
  export BABEL_DATA=/data/MM3/babel-corpus
  #export PYTHONPATH=/data/ASR5/fmetze/asr-test/lorelei-audio/egs/asr/s5c/201-haitian-flp:
  #export PYTHONPATH=./eesen-tf/tf/ctc-train:./eesen-tf/tf/ctc-decode/code:./eesen-tf/tf/rnn-lm/code:/data/MM23/sdalmia/lorelei-audio/egs/asr/s5c/307-amharic-flp-tf-epitran
  export PYTHONPATH=/data/ASR5/ramons_2/sinbad_projects/youtube_project/am/eesen_20170714/tf/ctc-am:/data/ASR5/ramons_2/sinbad_projects/youtube_project/am/eesen_20170714/tf/char-rnn-lm/
  #export PYTHONPATH=/data/ASR5/fmetze/eesen-tf/tf/ctc-train:/data/ASR5/fmetze/eesen-tf/tf/rnn-lm/code:/data/ASR5/fmetze/eesen-tf/tf/ctc-decode/code
  #export EESEN_ROOT=/data/ASR5/fmetze/eesen-block-copy
  export EESEN_ROOT=/data/MM23/sdalmia/eesen
  export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/sph2pipe_v2.5:/data/ASR5/fmetze/kaldi-latest/src/latbin:/data/ASR5/fmetze/kaldi-latest/src/featbin:$PATH
  #source ~/tf/bin/activate

  unset CUDA_VISIBLE_DEVICES
  unset GPU_DEVICE_ORDINAL

  if [ "$(lsof -n -w -t /dev/nvidia0)" != "" ]; then
          kill -9 $(lsof -n -w -t /dev/nvidia0)
  fi

else
  # CMU Rocks cluster
  module load python27
  module load gcc-4.9.2
  module load cuda-8.0

  [ -n "$PBS_JOBID" ] && export CUDA_VISIBLE_DEVICES=`qstat -n $PBS_JOBID|awk 'END {split ($NF, a, "/"); printf ("%s\n", a[2])}'`
  [ -n "$PBS_JOBID" ] && export THEANO_FLAGS="device=`qstat -n $PBS_JOBID | tail -n 1 | sed 's|.*/|gpu|g'`"

  export TMPDIR=/scratch
  #export TMPDIR=.
  export LD_LIBRARY_PATH=/data/ASR1/tools/sox-14.4.2/install/lib:$LD_LIBRARY_PATH
  export BABEL_DATA=/data/MM23/sdalmia/eval_lorelei/il5_tig_set1_tts
  export KALDI_ROOT=/data/ASR1/tools/kaldi
  export EESEN_ROOT=/data/MM23/sdalmia/eesen
  #export PYTHONPATH=/data/ASR5/fmetze/asr-test/lorelei-audio/egs/asr/s5c/201-haitian-flp:/data/ASR5/fmetze/eesen-tf/tf/tf1
  #export PYTHONPATH=./eesen-tf/tf/ctc-train:./eesen-tf/tf/ctc-decode/code:./eesen-tf/tf/rnn-lm/code
  export PYTHONPATH=/data/ASR5/ramons_2/sinbad_projects/youtube_project/am/eesen_20170714/tf/ctc-am:/data/ASR5/ramons_2/sinbad_projects/youtube_project/am/eesen_20170714/tf/char-rnn-lm/

  #. /export/babel/data/software/env.sh
  export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/sph2pipe_v2.5:/home/fmetze/tools/kaldi/src/bin:/data/ASR5/fmetze/kaldi-latest/src/latbin:$PATH
  #export PATH=$PWD/meine:$PWD/utils/:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$PWD:$PATH
  export PATH=/data/ASR1/tools/sox-14.4.2/install/bin:/data/ASR1/tools/kenlm/bin:$PATH
  #export PATH="/home/fmetze/tools/eesen/src/netbin/:/home/fmetze/tools/eesen/src/decoderbin/:/home/fmetze/tools/eesen/src/fstbin/:/home/fmetze/tools/eesen/src/featbin/:/data/ASR4/babel/sctk-2.4.0/bin/:$PATH"

  #export PATH="/data/ASR1/ramons/anaconda2/bin":$PATH
  #source activate tensorflow_gpu
  #export PATH=/data/ASR5/ramons_2/tools/anaconda2/bin:$PATH
  #source activate tensorflow_gpu_1_0
  export PATH=/data/ASR5/ramons_2/tools/anaconda2/bin:$PATH
  source activate tensorflow_gpu_1_2

  if [ "$(lsof -n -w -t /dev/nvidia`qstat -n $PBS_JOBID|awk 'END {split ($NF, a, "/"); printf ("%s\n", a[2])}'`)" != "" ]; then
    kill -9 $(lsof -n -w -t /dev/nvidia`qstat -n $PBS_JOBID|awk 'END {split ($NF, a, "/"); printf ("%s\n", a[2])}'`)
  fi

fi

[ -f ${EESEN_ROOT}/tools/env.sh ] && . ${EESEN_ROOT}/tools/env.sh
[ -f ./local.sh ] && . ./local.sh
