
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


import argparse
import logging
import os
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.contrib.quantization import *
import statistics


def download_dataset(dataset_url, dataset_dir, logger=None):
    if logger is not None:
        logger.info('Downloading dataset for inference from %s to %s' % (dataset_url, dataset_dir))
    mx.test_utils.download(dataset_url, dataset_dir)

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


def advance_data_iter(data_iter, n):
    assert n >= 0
    if n == 0:
        return data_iter
    has_next_batch = True
    while has_next_batch:
        try:
            data_iter.next()
            n -= 1
            if n == 0:
                return data_iter
        except StopIteration:
            has_next_batch = False


def profile(data, symbol_file, num_inference_images, sym, devs, label_name):
    debug = False

    mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name, ])
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)

    # warm up
    warm_up = 0
    for batch in data:
        mod.forward(batch, is_train=False)
        warm_up += 1
        if warm_up == 50:
            break

    counter = 0
    time_mxnet = list()
    for batch in data:
        time0 = time.time()
        mod.forward(batch, is_train=False)
        nd.waitall() #rt_mod.get_output(0).asnumpy()
        time1 = time.time()
        time_mxnet.append(time1 - time0)
        counter += 1
        if counter == num_inference_images:
            break
    
    avg = lambda x : round(1000*sum(x)/len(x), 6)
    std = lambda x: round(statistics.stdev(x), 6)

    total_mxnet = avg(time_mxnet)
    sec_mxnet = total_mxnet/1000
    std_mxnet = std(time_mxnet)
    min_mxnet = round(min(time_mxnet), 6)
    min_mxnet_ms = round(min(time_mxnet)*1000, 6)
    deviation_from_min_mxnet = round(sec_mxnet/min_mxnet*100 - 100, 6) 
    deviation_from_std_mxnet = round(std_mxnet/sec_mxnet*100, 6)


    net_name = symbol_file.split('/')[-1].replace('.json','')
    print("Perf", "Mxnet", net_name, total_mxnet, min_mxnet_ms, std_mxnet, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--ctx', type=str, default='cpu')
    parser.add_argument('--benchmark', type=bool, default=False, help='dummy data benchmark')
    parser.add_argument('--score_tvm', type=bool, default=False, help='score tvm')
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
    parser.add_argument('--num-inference-batches', type=int, required=True, help='number of images used for inference')
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
    if args.benchmark == False:
        dataset = "./data/val_256_q90.rec"
        download_dataset('http://data.mxnet.io/data/val_256_q90.rec', dataset)
        logger.info('Dataset for inference: %s' % dataset)

        # creating data iterator
        data = mx.io.ImageRecordIter(
            path_imgrec=dataset,
            label_width=1,
            preprocess_threads=data_nthreads,
            batch_size=batch_size,
            data_shape=data_shape,
            label_name=label_name,
            rand_crop=False,
            rand_mirror=False,
            shuffle=args.shuffle_dataset,
            shuffle_chunk_seed=args.shuffle_chunk_seed,
            seed=args.shuffle_seed,
            dtype=data_layer_type,
            ctx=args.ctx,
            **combine_mean_std)

        # loading model
        sym, arg_params, aux_params = load_model(symbol_file, param_file, logger)

        # make sure that fp32 inference works on the same images as calibrated quantized model
        logger.info('Skipping the first %d batches' % args.num_skipped_batches)
        data = advance_data_iter(data, args.num_skipped_batches)

        num_inference_images = args.num_inference_batches * batch_size
        logger.info('Running model %s for inference' % symbol_file)
        is_profile = True
        profile(data, symbol_file, num_inference_images, sym, [ctx], label_name)
