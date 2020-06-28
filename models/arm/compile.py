# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.relay import expr as _expr
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime
from pathlib import Path

import argparse
import logging
import os
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.contrib.quantization import *
import statistics


target = 'llvm -device=arm_cpu -target=aarch64-linux-gnu -mattr=+v8.2a,+fullfp16,+fp-armv8,+dotprod,+crc,+crypto,+neon'
# target = 'llvm -device=arm_cpu -target=aarch64-linux-gnu'

def load_model(symbol_file, param_file, logger=None):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, symbol_file)
    if logger is not None:
        logger.info('Loading symbol from file %s' % symbol_file_path)
    symbol = mx.sym.load(symbol_file_path)

    param_file_path = os.path.join(cur_path, param_file)
    if logger is not None:
        logger.info('Loading params from file %s' % param_file_path)
    save_dict = nd.load(param_file_path)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return symbol, arg_params, aux_params


def compile_via_tvm(sym, arg_params, aux_params, symbol_file, data_shape):
    tune = False

    input_shape = [1] + list(data_shape)
    input_dict = {'data': input_shape}
    input_name = 'data'

    print('Loading....')
    mod, params = relay.frontend.from_mxnet(sym,
                                            dtype={},
                                            shape=input_dict,
                                            arg_params=arg_params,
                                            aux_params=aux_params)

    model_name = symbol_file.split('/')[-1].replace('.json','')
    log_file = "%s.log" % model_name
    graph_opt_sch_file = "%s_graph_opt.log" % model_name

    Path(log_file).touch()
    Path(graph_opt_sch_file).touch()

    import time
    timing = -1
    def tracer(module, info, is_before):
        global timing
        if bool(is_before):
            timing = time.time()
        else:
            print('Executes: ', info.name, (time.time() - timing) * 1000)

    print('Model Load!')
    import tensorizer
    with tvm.transform.PassContext(config={'tir.add_lower_pass': [(1, tensorizer.rewrite)]},
                                   trace=tracer, opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)
            base = './compiled/' + symbol_file.split('/')[-1].replace('.json','')

            path_lib = base + '_deploy_lib.tar'
            path_graph =  base + '_deploy_graph.json'
            path_params = base + '_deploy_params.params'

            lib.export_library(path_lib)
            with open(path_graph, 'w') as fo:
                fo.write(graph)
            with open(path_params, 'wb') as fo:
                fo.write(relay.save_param_dict(params))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--ctx', type=str, default='gpu')
    parser.add_argument('--symbol-file', type=str, required=True, help='symbol file path')
    parser.add_argument('--param-file', type=str, required=False, help='param file path')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--dataset', type=str, required=False, help='dataset path')
    parser.add_argument('--rgb-mean', type=str, default='0,0,0')
    parser.add_argument('--rgb-std', type=str, default='1,1,1')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=60, help='number of threads for data decoding')
    parser.add_argument('--num-skipped-batches', type=int, default=0, help='skip the number of batches for inference')
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the calibration dataset')
    parser.add_argument('--shuffle-chunk-seed', type=int, default=3982304,
                        help='shuffling chunk seed, see'
                             ' https://mxnet.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--shuffle-seed', type=int, default=48564309,
                        help='shuffling seed, see'
                             ' https://mxnet.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--data-layer-type', type=str, default="float32",
                        choices=['float32', 'int8', 'uint8'],
                        help='data type for data layer')

    args = parser.parse_args()

    if args.ctx == 'gpu':
        ctx = mx.gpu(0)
    elif args.ctx == 'cpu':
        ctx = mx.cpu(0)
    else:
        raise ValueError('ctx %s is not supported in this script' % args.ctx)

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    symbol_file = args.symbol_file
    param_file = args.param_file
    data_nthreads = args.data_nthreads

    batch_size = args.batch_size
    logger.info('batch size = %d for inference' % batch_size)

    rgb_mean = args.rgb_mean
    logger.info('rgb_mean = %s' % rgb_mean)
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}
    rgb_std = args.rgb_std
    logger.info('rgb_std = %s' % rgb_std)
    rgb_std = [float(i) for i in rgb_std.split(',')]
    std_args = {'std_r': rgb_std[0], 'std_g': rgb_std[1], 'std_b': rgb_std[2]}
    combine_mean_std = {}
    combine_mean_std.update(mean_args)
    combine_mean_std.update(std_args)

    label_name = args.label_name
    logger.info('label_name = %s' % label_name)

    image_shape = args.image_shape
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info('Input data shape = %s' % str(data_shape))

    data_layer_type = args.data_layer_type
    # loading model
    sym, arg_params, aux_params = load_model(symbol_file, param_file, logger)
    compile_via_tvm(sym, arg_params, aux_params, symbol_file, data_shape)
