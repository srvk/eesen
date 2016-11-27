export EESEN_ROOT=`pwd`/../../..
export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/irstlm/bin/:$PWD:$PATH
export LC_ALL=C

if [[ `uname -n` =~ comet-* ]]; then
  :
elif [[ `uname -n`=~ bridges ]]; then
  :
elif [[ `uname -n`=~ compute-* ]]; then
  # CMU Rocks cluster
  
  module load gcc-4.9.2
  module load cuda-8.0
  module load python27

  [ -n "$PBS_JOBID" ] && export CUDA_VISIBLE_DEVICES=`qstat -n $PBS_JOBID|awk 'END {split ($NF, a, "/"); printf ("%s\n", a[2])}'`

  export TMPDIR=/scratch
  export LD_LIBRARY_PATH=/data/ASR1/tools/sox-14.4.2/install/lib:$LD_LIBRARY_PATH

else
  echo "Which cluster is this?"
  exit 1;
fi

[ -f ${EESEN_ROOT}/tools/env.sh ] && . ${EESEN_ROOT}/tools/env.sh
[ -f ./local.sh ] && . ./local.sh

# for decoding without requiring CUDA
if [[ "${SLURM_JOB_NAME}" =~ test-* || "${PBS_JOBNAME}" =~ test-* ]]; then
  echo "Setting path to prefer non-gpu code ..."
  export PATH=${EESEN_ROOT}/src-nogpu/netbin:${EESEN_ROOT}/src-nogpu/decoderbin:${PATH}
  export LD_LIBRARY_PATH=${EESEN_ROOT}/tools/openfst/lib:${LD_LIBRARY_PATH}
  export PATH=${PATH}:/data/ASR5/fmetze/kaldi-latest/src/latbin
else
  :
fi
