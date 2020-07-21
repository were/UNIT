import tvm
import tensorizer
import logging
import sys
import numpy as np
from tvm import relay
from tvm import autotvm

import topi
from tvm.relay import op

n, c, h, w = 1, 256, 128, 128
ic, oc, kh, kw = c, 1024, 3, 3

var_x = relay.var('x', shape=(n, c, h, w), dtype='float32')
var_w = relay.var('w', shape=(oc, ic, kh, kw), dtype='float32')
var_b = relay.var('b', shape=(1, oc, 1, 1), dtype='float32')

conv2d = relay.nn.conv2d(var_x, var_w, out_dtype='float32', kernel_size=(3, 3), channels=oc)
y = relay.add(conv2d, var_b)

func = relay.Function([var_x, var_w, var_b], y)
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

    x_ = (np.random.randn(n, c, h, w) * 128).astype('float32')
    w_ = (np.random.randn(oc, ic, kh, kw) * 128).astype('float32')
    b_ = (np.random.randn(1, oc, 1, 1) * 128).astype('float32')
    module.set_input('x', x_)
    module.set_input('w', w_)
    module.set_input('b', b_)

    timer = module.module.time_evaluator('run', ctx=tvm.gpu(), number=3, repeat=10)
    timed = timer()

    print((n * oc * (h - kh + 1) * (w - kw + 1)) * (kh * kw * ic) / timed.mean / 1e9)
