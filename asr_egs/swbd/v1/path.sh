export EESEN_ROOT=`pwd`/../../..
export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$PWD:$PATH
export LC_ALL=C
export LD_LIBRARY_PATH=$EESEN_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH

. $EESEN_ROOT/tools/env.sh

if [[ `uname -n` =~ comet-* ]]; then
    # SDSC Comet cluster
    export TMPDIR=/scratch/$USER/$SLURM_JOBID
fi

if [[ ! -z ${acwt+x} ]]; then
    # let's assume we're decoding
    export PATH=$EESEN_ROOT/src-nogpu/netbin:$PATH
    echo "Preferring non-gpu netbin code"
fi
