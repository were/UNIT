import tvm
import tensorizer
import logging
import sys
import numpy as np
from tvm import relay
from tvm import autotvm

import topi
from tvm.relay import op

x = relay.var('x', shape=(128, 3072), dtype='float32')
w = relay.var('w', shape=(768, 3072), dtype='float32')
b = relay.var('b', shape=(1, 768), dtype='float32')
conv2d = relay.nn.dense(x, w, out_dtype='float32')
y = relay.add(conv2d, b)

func = relay.Function([x, w, b], y)
module = tvm.IRModule()
module['main'] = func

import time
timing = -1
def tracer(module, info, is_before):
    global timing
    if bool(is_before):
        timing = time.time()
    else:
        print('Executes: ', info.name, (time.time() - timing) * 1000)

with tvm.transform.PassContext(opt_level=4, trace=tracer):
    graph, lib, params = tvm.relay.build(module, target='cuda -libs=cublas,cudnn')
    from tvm.contrib import graph_runtime as runtime
    module = runtime.create(graph, lib, tvm.gpu())

    x_ = tvm.nd.array((np.random.randn(128, 3072) * 128).astype('float32'), tvm.gpu())
    w_ = tvm.nd.array((np.random.randn(768, 3072) * 128).astype('float32'), tvm.gpu())
    b_ = tvm.nd.array((np.random.randn(1, 768) * 128).astype('float32'), tvm.gpu())
    module.set_input('x', x_)
    module.set_input('w', w_)
    module.set_input('b', b_)

    timer = module.module.time_evaluator('run', ctx=tvm.gpu(), number=3, repeat=10)
    timed = timer()

    print((128 * 768 * 3072) / timed.mean / 1e9)
