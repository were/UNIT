import tvm
import tensorizer
import logging
import sys
import numpy as np
from tvm import relay
from tvm import autotvm
import os
import argparse

import topi
from tvm.relay import op

ap = argparse.ArgumentParser()
ap.add_argument('--target', type=str, default='nvptx')
args = ap.parse_args()
target =args.target + ' -libs=cublas,cudnn'
#t0, t1 = eval(input())
#n, c, h, w = map(int, t0)
#oc, ic, kh, kw = map(int, t1)
n, c, h, w, oc, ic, kh, kw, sh, sw = map(int, input().split())

oh = (h - kh) // sh + 1
ow = (w - kw) // sw + 1

var_x = relay.var('x', shape=(n, c, h, w), dtype='float32')
var_w = relay.const(tvm.nd.array((np.random.randn(oc, ic, kh, kw) * 128).astype('float32')))
var_b = relay.const(tvm.nd.array((np.random.randn(1, oc, 1, 1) * 128).astype('float32')))
conv2d = relay.nn.conv2d(var_x, var_w, out_dtype='float32', kernel_size=(kh, kw), channels=oc, strides=(sh, sw))
y = conv2d

func = relay.Function([var_x], y)
module = tvm.IRModule()
module['main'] = func

import time
timing = -1
def tracer(module, info, is_before):
    pass
    #global timing
    #if bool(is_before):
    #    timing = time.time()
    #else:
    #    print('Executes: ', info.name, (time.time() - timing) * 1000)

from tensorizer import tune
tune.enable = False


def run():
    passes = [(1, tensorizer.rewrite)]
    config = {'tir.add_lower_pass': passes} if target.startswith('nvptx') else {}
    with tvm.transform.PassContext(opt_level=3, trace=tracer, config=config):
        graph, lib, params = tvm.relay.build(module, target=target)
        #from tvm.contrib import graph_runtime as runtime
        from tvm.contrib.debugger import debug_runtime as runtime
        func = runtime.create(graph, lib, tvm.gpu())


        x_ =(np.random.randn(n, c, h, w) * 128).astype('float32')
        func.set_input('x', x_)
        timer = func.module.time_evaluator('run', ctx=tvm.gpu(), number=1, repeat=10)
        #timed = []
        #for i in range(10):
        #    func.run()
        #    for node, time in zip(func.debug_datum._nodes_list, func.debug_datum._time_list):
        #        if 'conv2d' in node['name']:
        #            timed.append(time[0])
        timed = timer()
        while np.var(timed.results) > 1e-5:
            timed = timer()
        return timed.mean
        #return np.mean(timed)

base = None
timed = run()
base = timed * 1e6

if target.startswith('cuda'):
    with open(os.getenv('HOME') + '/tune.csv', 'a') as f:
        f.write(','.join(map(str, [n, c, h, w, oc, ic, kh, kw, sh, sw])) + ',' + str(timed * 1e6) + '\n')
    quit()

relay.backend.compile_engine.get().clear()

results = []
for i in [None, 'fuse', 'pad'] if ow < 32 else [None]:
    j = 16
    while True:
        tune.padding = i
        tune.splitk = j
        timed = run()

        results.append(((i, j), timed * 1e6))

        relay.backend.compile_engine.get().clear()
        j <<= 1
        print(tune.total_idx)
        if j > tune.total_idx:
            break


with open(os.getenv('HOME') + '/ablation.log', 'a') as f:
    f.write(f'{tune.ashape} {tune.bshape} {tune.strides} {results}, {base}\n')
    f.write(f'{n} {c} {h} {w} {oc} {ic} {kh} {kw} {sh} {sw} {results}, {base}\n')
