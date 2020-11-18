#!/bin/bash
ulimit -S -s 65536

export MODELS=$HOME/models
export PYTHONPATH=/root/UNIT/python:$PYTHONPATH
export COMMON_PYPATH=$PYTHONPATH

# For Mxnet, best baseline, this is required
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

# For TVM, best numbers, following line is required.
export TVM_BIND_MASTER_THREAD=1 

NUM_ITERS=5
export MODEL_PATH=$MODELS

export PYTHONPATH=/root/tvms/tensorize/python/:/root/tvms/tensorize/topi/python/:$COMMON_PYPATH

python3 profile_mxnet.py --symbol-file=${MODELS}/resnet18_v1-quantized.json --param-file=${MODELS}/resnet18_v1-quantized.params --image-shape=3,224,224 --ctx=cpu --num-inference-batches=5
python3 profile_mxnet.py --symbol-file=${MODELS}/resnet50_v1-quantized.json --param-file=${MODELS}/resnet50_v1-quantized.params --image-shape=3,224,224 --ctx=cpu --num-inference-batches=5
python3 profile_mxnet.py --symbol-file=${MODELS}/resnet50_v1b-quantized.json --param-file=${MODELS}/resnet50_v1b-quantized.params --image-shape=3,224,224 --ctx=cpu --num-inference-batches=5
python3 profile_mxnet.py --symbol-file=${MODELS}/imagenet1k-inception-bn-quantized.json --param-file=${MODELS}/imagenet1k-inception-bn-quantized.params --image-shape=3,224,224 --ctx=cpu --num-inference-batches=5
python3 profile_mxnet.py --symbol-file=${MODELS}/inceptionv3-quantized.json --param-file=${MODELS}/inceptionv3-quantized.params --image-shape=3,299,299 --ctx=cpu --num-inference-batches=5
python3 profile_mxnet.py --symbol-file=${MODELS}/resnet101_v1-quantized.json --param-file=${MODELS}/resnet101_v1-quantized.params --image-shape=3,224,224 --ctx=cpu --num-inference-batches=5
python3 profile_mxnet.py --symbol-file=${MODELS}/imagenet1k-resnet-152-quantized.json --param-file=${MODELS}/imagenet1k-resnet-152-quantized.params --image-shape=3,224,224 --ctx=cpu --num-inference-batches=5
python3 profile_mxnet.py --symbol-file=${MODELS}/mobilenet1.0-quantized.json --param-file=${MODELS}/mobilenet1.0-quantized.params --image-shape=3,224,224 --ctx=cpu --num-inference-batches=5
python3 profile_mxnet.py --symbol-file=${MODELS}/mobilenetv2_1.0-quantized.json --param-file=${MODELS}/mobilenetv2_1.0-quantized.params --image-shape=3,224,224 --ctx=cpu --num-inference-batches=5

