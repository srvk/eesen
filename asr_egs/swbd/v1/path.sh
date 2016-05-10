export EESEN_ROOT=`pwd`/../../..
export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/irstlm/bin/:$PWD:$PATH
export LC_ALL=C

. $EESEN_ROOT/tools/env.sh

if [[ `uname -n` =~ compute-* ]]; then
    # CMU Rocks cluster
    module load python27
    module load gcc-4.9.2
    export TMPDIR=/scratch
fi
