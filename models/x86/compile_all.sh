rm -rf compiled
mkdir compiled

python3 compile.py --symbol-file=${MODELS}/resnet18_v1-quantized.json --param-file=${MODELS}/resnet18_v1-quantized.params --image-shape=3,224,224 --ctx=cpu --libs=$1
python3 compile.py --symbol-file=${MODELS}/resnet50_v1-quantized.json --param-file=${MODELS}/resnet50_v1-quantized.params --image-shape=3,224,224 --ctx=cpu --libs=$1
python3 compile.py --symbol-file=${MODELS}/resnet50_v1b-quantized.json --param-file=${MODELS}/resnet50_v1b-quantized.params --image-shape=3,224,224 --ctx=cpu --libs=$1
python3 compile.py --symbol-file=${MODELS}/imagenet1k-inception-bn-quantized.json --param-file=${MODELS}/imagenet1k-inception-bn-quantized.params --image-shape=3,224,224 --ctx=cpu --libs=$1
python3 compile.py --symbol-file=${MODELS}/inceptionv3-quantized.json --param-file=${MODELS}/inceptionv3-quantized.params --image-shape=3,299,299 --ctx=cpu --libs=$1
python3 compile.py --symbol-file=${MODELS}/resnet101_v1-quantized.json --param-file=${MODELS}/resnet101_v1-quantized.params --image-shape=3,224,224 --ctx=cpu --libs=$1
python3 compile.py --symbol-file=${MODELS}/imagenet1k-resnet-152-quantized.json --param-file=${MODELS}/imagenet1k-resnet-152-quantized.params --image-shape=3,224,224 --ctx=cpu --libs=$1
python3 compile.py --symbol-file=${MODELS}/mobilenet1.0-quantized.json --param-file=${MODELS}/mobilenet1.0-quantized.params --image-shape=3,224,224 --ctx=cpu --libs=$1
python3 compile.py --symbol-file=${MODELS}/mobilenetv2_1.0-quantized.json --param-file=${MODELS}/mobilenetv2_1.0-quantized.params --image-shape=3,224,224 --ctx=cpu --libs=$1
