rm -rf compiled
mkdir compiled

python3 compile.py --symbol-file=${MODEL}/imagenet1k-inception-bn-symbol.json --param-file=${MODEL}/imagenet1k-inception-bn-0000.params --image-shape=3,224,224 --ctx=cpu
python3 compile.py --symbol-file=${MODEL}/inceptionv3-symbol.json --param-file=${MODEL}/inceptionv3-0000.params --image-shape=3,299,299 --ctx=cpu
python3 compile.py --symbol-file=${MODEL}/mobilenet1.0-symbol.json --param-file=${MODEL}/mobilenet1.0-0000.params --image-shape=3,224,224 --ctx=cpu
python3 compile.py --symbol-file=${MODEL}/mobilenetv2_1.0-symbol.json --param-file=${MODEL}/mobilenetv2_1.0-0000.params --image-shape=3,224,224 --ctx=cpu
python3 compile.py --symbol-file=${MODEL}/resnet101_v1-symbol.json --param-file=${MODEL}/resnet101_v1-0000.params --image-shape=3,224,224 --ctx=cpu
python3 compile.py --symbol-file=${MODEL}/imagenet1k-resnet-152-symbol.json --param-file=${MODEL}/imagenet1k-resnet-152-0000.params --image-shape=3,224,224 --ctx=cpu
python3 compile.py --symbol-file=${MODEL}/resnet18_v1-symbol.json --param-file=${MODEL}/resnet18_v1-0000.params --image-shape=3,224,224 --ctx=cpu
python3 compile.py --symbol-file=${MODEL}/resnet50_v1b-symbol.json --param-file=${MODEL}/resnet50_v1b-0000.params --image-shape=3,224,224 --ctx=cpu
python3 compile.py --symbol-file=${MODEL}/resnet50_v1-symbol.json --param-file=${MODEL}/resnet50_v1-0000.params --image-shape=3,224,224 --ctx=cpu
