export EESEN_ROOT=`pwd`/../../..
export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/irstlm/bin/:$PWD:$PATH
export LC_ALL=C

if [[ `uname -n` =~ ip-* ]]; then
  # AWS instance
  export KALDI_ROOT=/home/fmetze/tools/kaldi
  export TMPDIR=/tmp

elif [[ `uname -n` =~ comet* ]]; then
  # comet cluster
  module load atlas
  module load lapack

  export TMPDIR=/scratch/${USER}/${SLURM_JOBID}

elif [[ `uname -n` =~ compute- ]]; then
  # CMU Rocks cluster
  module load python27
  module load gcc-4.9.2

  export TMPDIR=/scratch

  # just in case we're running on a GPU node
  export CUDA_VISIBLE_DEVICES=`qstat -n $PBS_JOBID|awk ' END { split ($NF, a, "/"); printf ("%s\n", a[2]) } '`

else
  # where are we?
  exit 1;
fi

if [[ ! -z ${acwt+x} ]]; then
  # let's assume we're decoding
    export PATH=$EESEN_ROOT/src-nogpu/netbin:$PATH
    echo "Preferring non-gpu netbin code"
fi
