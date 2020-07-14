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
import pathlib


target = 'llvm -mcpu=cascadelake'
def tune_kernels(tasks,
                 measure_option,
                 tuner='gridsearch',
                 early_stopping=None,
                 log_filename='tuning.log'):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial=len(task.config_space)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, input_name, use_DP=True):
    target_op = [relay.nn.conv2d]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt, mod, params, data_shape, log_file, graph_opt_sch_file,
        input_name):
    # extract workloads from relay program
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(\
            mod["main"], target=target,
            params=params,
            ops=(relay.op.get("nn.conv2d"), relay.op.get("nn.dense")))

    # run tuning tasks
    print("Tuning...")
    tune_kernels(tasks, **tuning_opt)
    # tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file, input_name)

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


def compile_via_tvm(sym, arg_params, aux_params, symbol_file, data_shape, tune):

    input_shape = [1] + list(data_shape)
    input_dict = {'data': input_shape}
    input_name = 'data'

    batch = 1
    seq_length = 128
    input_dict = {
        'data0': (batch, seq_length),
        'data1': (batch, seq_length),
        'data2': (batch,)
    }
    mod, params = relay.frontend.from_mxnet(sym,
                                            dtype={},
                                            shape=input_dict,
                                            arg_params=arg_params,
                                            aux_params=aux_params)
    print(mod)

    model_name = symbol_file.split('/')[-1].replace('.json','')
    log_dir = os.getcwd() + "/tuned_logs_c5"
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = log_dir + "/" + "%s.log" % model_name
    graph_opt_sch_file = log_dir + "/" + "%s_graph_opt.log" % model_name

    Path(log_file).touch()
    Path(graph_opt_sch_file).touch()

    if tune:
        tuning_option = {
            'log_filename': log_file,
            'tuner': 'random',
            'early_stopping': None,

            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(number=10, repeat=1,
                                           min_repeat_ms=1000),
            ),
        }

        tune_and_evaluate(tuning_option, mod, params, input_shape, log_file,
                graph_opt_sch_file, input_name)

    # with autotvm.apply_graph_best(graph_opt_sch_file):
    # with autotvm.apply_history_best(log_file):
    #     with relay.build_config(opt_level=3):

    import tensorizer
    with tvm.transform.PassContext(config={'tir.add_lower_pass': [(1, tensorizer.rewrite)]},
            opt_level=4):
        graph, lib, params = relay.build_module.build(
            mod, target=target, params=params)

        base_dir = os.getcwd() + "/compiled"
        pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

        base = base_dir + '/' + symbol_file.split('/')[-1].replace('.json','')

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
    parser.add_argument('--symbol-file', type=str, required=True, help='symbol file path')
    parser.add_argument('--param-file', type=str, required=True, help='param file path')
    parser.add_argument('--image-shape', type=str, required=True)
    parser.add_argument('--tune', type=str, required=True, help='yes or no')

    args = parser.parse_args()

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    symbol_file = args.symbol_file
    param_file = args.param_file
    if args.tune == "yes":
        tune = True
    else:
        tune = False

    image_shape = args.image_shape
    data_shape = tuple([int(i) for i in image_shape.split(',')])

    # loading model
    sym, arg_params, aux_params = load_model(symbol_file, param_file, logger)
    compile_via_tvm(sym, arg_params, aux_params, symbol_file, data_shape, tune)
