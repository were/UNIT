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

n, c, h, w, oc, ic, kh, kw, sh, sw = map(int, input().split())

var_x = relay.var('x', shape=(n, c, h, w), dtype='int8')
w_ = (np.random.randn(oc, ic, kh, kw) * 128).astype('int8')
b_ = (np.random.randn(1, oc, 1, 1) * 128).astype('int32')
var_w = relay.const(tvm.nd.array(w_))
var_b = relay.const(tvm.nd.array(b_))
conv2d = relay.nn.conv2d(var_x, var_w, out_dtype='int32', kernel_size=(kh, kw), channels=oc, strides=(sh, sw))
biased = relay.add(conv2d, var_b)
y = relay.multiply(biased, relay.const(11, 'int32'))

func = relay.Function([var_x], y)
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

with tvm.transform.PassContext(opt_level=3, trace=tracer, config={'tir.add_lower_pass': [(1, tensorizer.rewrite)]}):
    graph, lib, params = tvm.relay.build(module, target='llvm -mcpu=cascadelake')
    #from tvm.contrib import graph_runtime as runtime
    from tvm.contrib.debugger import debug_runtime as runtime
    module = runtime.create(graph, lib, tvm.cpu())

    x_ = (np.random.randn(n, c, h, w) * 128).astype('int8')
    module.set_input('x', x_)
    module.run()
    module.run()
    module.run()
    module.run()
    module.run()

    #timer = module.module.time_evaluator('run', ctx=tvm.cpu(0), number=3, repeat=10)
    #timed = timer()

    #print(timed.mean * 1e6)
    #print((n * oc * (h - kh + 1) * (w - kw + 1)) * (kh * kw * ic) / timed.mean / 1e9)
