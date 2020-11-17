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

python3 profile_mxnet.py --symbol-file=${MODELS}/resnet18_v1-quantized.json --param-file=${MODELS}/resnet18_v1-quantized.params --image-shape=3,224,224 --ctx=cpu --num-inference-batches=10

#python3 profile_mxnet.py --symbol-file=$MODEL_PATH/resnet18_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet18_v1-quantized-0000.params --num-inference-batches=2000 |& tee -a perf.txt
#
#python3 profile_mxnet.py --symbol-file=$MODEL_PATH/resnet50_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet50_v1-quantized-0000.params --num-inference-batches=2000 |& tee -a perf.txt
#
#for i in $(seq 1 $NUM_ITERS)
#do
#    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/resnet50_v1b-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet50_v1b-quantized-0000.params --num-inference-batches=2000 |& tee -a perf.txt
#    python3 profile_tvm.py --symbol-file=$MODEL_PATH/resnet50_v1b-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet50_v1b-quantized-0000.params --num-inference-batches=2000 |& tee -a perf.txt
#done
#
#for i in $(seq 1 $NUM_ITERS)
#do
#    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/resnet101_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet101_v1-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
#    python3 profile_tvm.py --symbol-file=$MODEL_PATH/resnet101_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet101_v1-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
#done
#
#
#for i in $(seq 1 $NUM_ITERS)
#do
#    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/imagenet1k-resnet-152-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
#    python3 profile_tvm.py --symbol-file=$MODEL_PATH/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/imagenet1k-resnet-152-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
#done
#
#for i in $(seq 1 $NUM_ITERS)
#do
#    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/inceptionv3-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/inceptionv3-quantized-0000.params --image-shape=3,299,299   --num-inference-batches=2000  |& tee -a perf.txt
#    python3 profile_tvm.py --symbol-file=$MODEL_PATH/inceptionv3-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/inceptionv3-quantized-0000.params --image-shape=3,299,299   --num-inference-batches=2000  |& tee -a perf.txt
#done
#
#for i in $(seq 1 $NUM_ITERS)
#do
#    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/imagenet1k-inception-bn-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
#    python3 profile_tvm.py --symbol-file=$MODEL_PATH/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/imagenet1k-inception-bn-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
#done
#
#for i in $(seq 1 $NUM_ITERS)
#do
#    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/mobilenet1.0-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/mobilenet1.0-quantized-0000.params --num-inference-batches=2000  --image-shape=3,224,224 |& tee -a perf.txt
#    python3 profile_tvm.py --symbol-file=$MODEL_PATH/mobilenet1.0-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/mobilenet1.0-quantized-0000.params --num-inference-batches=2000  --image-shape=3,224,224 |& tee -a perf.txt
#done
#
#for i in $(seq 1 $NUM_ITERS)
#do
#    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/mobilenetv2_1.0-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/mobilenetv2_1.0-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
#    python3 profile_tvm.py --symbol-file=$MODEL_PATH/mobilenetv2_1.0-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/mobilenetv2_1.0-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
#done
#
#
