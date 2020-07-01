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


target = 'llvm -mcpu=cascadelake'
#################################################################
# Configure tensor tuning settings and create tasks
# -------------------------------------------------
# To get better kernel execution performance on x86 CPU,
# we need to change data layout of convolution kernel from
# "NCHW" to "NCHWc". To deal with this situation, we define
# conv2d_NCHWc operator in topi. We will tune this operator
# instead of plain conv2d.
#
# We will use local mode for tuning configuration. RPC tracker
# mode can be setup similarly to the approach in
# :ref:`tune_relay_arm` tutorial.

# You can skip the implementation of this function for this tutorial.

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


def run_tvm(data, symbol_file, num_inference_images, sym, devs, label_name):
    debug = False
    import tvm
    from tvm.contrib import graph_runtime
    from tvm.contrib.debugger import debug_runtime as debug_runtime

    base = './compiled/' + symbol_file.split('/')[-1].replace('.json','')

    path_lib = base + '_deploy_lib.tar'
    path_graph =  base + '_deploy_graph.json'
    path_params = base + '_deploy_params.params'

    graph = open(path_graph).read()
    lib = tvm.runtime.load_module(path_lib)
    params = bytearray(open(path_params, 'rb').read())

    if debug:
        rt_mod = debug_runtime.create(graph, lib, ctx=tvm.cpu(0))
        mod = mx.mod.Module(symbol=sym, context=devs)
        mod.bind(for_training=False,
                 data_shapes=data.provide_data)
    else:
        rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name, ])
        mod.bind(for_training=False,
                 data_shapes=data.provide_data,
                 label_shapes=data.provide_label)

    rt_mod.load_params(params)
    mod.set_params(arg_params, aux_params)

    counter = 0
    top_1_raw = 0
    top_5_raw = 0
    top_1_raw_mxnet = 0
    top_5_raw_mxnet = 0
    if debug:
        data = advance_data_iter(data, 0)
    for batch in data:
        # Get the original label.
        correct_label = int(batch.label[0].asnumpy()[0])

        rt_mod.set_input('data', batch.data[0].asnumpy())
        rt_mod.run()
        if debug:
            np.set_printoptions(suppress=False)
            for i in rt_mod.debug_datum.get_output_tensors().keys():
                print(i, rt_mod.debug_get_output(i))
            return
        tvm_res = rt_mod.get_output(0).asnumpy()

        mod.forward(batch, is_train=False)
        mxnet_res = mod.get_outputs()[0].asnumpy()

        if debug:
            print("######## MxNet ###########")
            print(mxnet_res[0][0])
            print("######## TVM ###########")
            print(tvm_res[0][0])
            print("############################")
            print("############################")
            print("############################")
            print("############################")
            print("############################")
            print("############################")
            print("############################")
            print("############################")
            print("############################")
            print("######## MxNet ###########")
            print(mxnet_res)
            print("######## TVM ###########")
            print(tvm_res)
            #print("######## Diff ###########")
            # it = np.nditer(mxnet_res, flags=['multi_index'])
            # while not it.finished:
            #     print("%d <%s>" % (it[0], it.multi_index), end='\n')
            #     it.iternext()
            np.testing.assert_allclose(mxnet_res.astype('int32'), tvm_res.astype('int32'), atol=0, verbose=True)
            try:
                np.testing.assert_allclose(mxnet_res.astype('int32'), tvm_res.astype('int32'), atol=0, verbose=True)
            except:
                np.testing.assert_allclose(mxnet_res.astype('int32'), tvm_res.astype('int32'), atol=1, verbose=True)
        else:
            tvm_pred = np.squeeze(tvm_res).argsort()[-5:][::-1]
            mxnet_pred = np.squeeze(mxnet_res).argsort()[-5:][::-1]

            if correct_label == tvm_pred[0]:
                top_1_raw += 1
                top_5_raw += 1
            elif correct_label in tvm_pred:
                top_5_raw += 1


            if correct_label == mxnet_pred[0]:
                top_1_raw_mxnet += 1
                top_5_raw_mxnet += 1
            elif correct_label in mxnet_pred:
                top_5_raw_mxnet += 1

        counter += 1
        if counter == num_inference_images:
            break

    model_name = symbol_file.split('/')[-1].replace('.json','')
    top_1 = float(top_1_raw_mxnet)/float(counter)
    top_5 = float(top_5_raw_mxnet)/float(counter)
    print("Mxnet", model_name, top_1, top_5, sep='\t')


    top_1 = float(top_1_raw)/float(counter)
    top_5 = float(top_5_raw)/float(counter)
    print("Tvm", model_name, top_1, top_5, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--ctx', type=str, default='gpu')
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
        dataset = args.dataset
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
        if args.score_tvm:
            run_tvm(data, symbol_file, num_inference_images, sym, [ctx], label_name)
