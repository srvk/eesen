
source activate tensorflow_gpu_1_0

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

g++ -std=c++11 -shared ctc_sin_bad.cc -o ctc_sin_bad.so -fPIC -I $TF_INC -O2


