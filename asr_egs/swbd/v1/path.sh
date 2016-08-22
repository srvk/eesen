export EESEN_ROOT=`pwd`/../../..
<<<<<<< HEAD
export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/../kaldi/src/featbin:$PWD:$PATH
=======
export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$PWD:$PATH
>>>>>>> f417f53c4cf04d67649562230e18d1c03bc6ed38
export LC_ALL=C
export LD_LIBRARY_PATH=$EESEN_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH

. $EESEN_ROOT/tools/env.sh

if [[ `uname -n` =~ comet-* ]]; then
    # SDSC Comet cluster
    export TMPDIR=/scratch/$USER/$SLURM_JOBID

elif [[ `uname -n` =~ br0* ]]; then
    # PSC Bridges cluster
    export TMPDIR=$LOCAL
    
elif [[ `uname -n` =~ compute-* ]]; then
    # CMU Rocks cluster
    module load python27
    module load gcc-4.9.2
    export TMPDIR=/scratch
fi

if [[ ! -z ${acwt+x} ]]; then
    # let's assume we're decoding
    export PATH=$EESEN_ROOT/src-nogpu/netbin:$PATH
    echo "Preferring non-gpu netbin code"
fi
