import tvm
import tensorizer
import logging
import sys
import numpy as np
from tvm import relay
from tvm import autotvm

import topi
from tvm.relay import op

print(type(tensorizer.INTRINSICS['vnni']['pattern'].body[0].source[0].a.value))

n, c, h, w = 1, 3, 128, 128
oc, ic, kh, kw = 256, c, 3, 3

var_x = relay.var('x', shape=(n, c, h, w), dtype='int8')
var_w = relay.var('w', shape=(oc, ic, kh, kw), dtype='int8')
var_b = relay.var('b', shape=(1, oc, 1, 1), dtype='int32')
conv2d = relay.nn.conv2d(var_x, var_w, out_dtype='int32', kernel_size=(3, 3), channels=oc)
biased = relay.add(conv2d, var_b)
y = relay.multiply(biased, relay.const(11, 'int32'))

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

with tvm.transform.PassContext(opt_level=3, trace=tracer):
    graph, lib, params = tvm.relay.build(module, target='llvm -mcpu=cascadelake')
    from tvm.contrib import graph_runtime as runtime
    module = runtime.create(graph, lib, tvm.cpu())

    x_ = (np.random.randn(n, c, h, w) * 128).astype('int8')
    w_ = (np.random.randn(oc, ic, kh, kw) * 128).astype('int8')
    b_ = (np.random.randn(1, oc, 1, 1) * 128).astype('int32')
    module.set_input('x', x_)
    module.set_input('w', w_)
    module.set_input('b', b_)

    timer = module.module.time_evaluator('run', ctx=tvm.cpu(0), number=3, repeat=10)
    timed = timer()

    print((n * oc * (h - kh + 1) * (w - kw + 1)) * (kh * kw * ic) / timed.mean / 1e9)
