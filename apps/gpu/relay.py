import tvm
import tensorizer
import logging
import sys
import numpy as np
from tvm import relay
from tvm import autotvm

import topi
from tvm.relay import op


#t0, t1 = eval(input())
#n, c, h, w = map(int, t0)
#oc, ic, kh, kw = map(int, t1)
n, c, h, w = 1, 192, 16, 16
oc, ic, kh, kw = 192, c, 3, 3

var_x = relay.var('x', shape=(n, c, h, w), dtype='float32')
var_w = relay.const(tvm.nd.array((np.random.randn(oc, ic, kh, kw) * 128).astype('float32')))
var_b = relay.const(tvm.nd.array((np.random.randn(1, oc, 1, 1) * 128).astype('float32')))
conv2d = relay.nn.conv2d(var_x, var_w, out_dtype='float32', kernel_size=(kh, kw), channels=oc, strides=(1, 1))
biased = relay.add(conv2d, var_b)
y = relay.multiply(biased, relay.const(123., 'float32'))
#y = conv2d

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

for i in ['fuse', 'pad']:
    for j in [16, 32, 64]:
        from tensorizer import tune
        tune.padding = i
        tune.splitk = j
        passes = [(1, tensorizer.rewrite)]
        with tvm.transform.PassContext(opt_level=3, trace=tracer, config={'tir.add_lower_pass': passes}):
        #with tvm.transform.PassContext(opt_level=4, trace=tracer):
            #graph, lib, params = tvm.relay.build(module, target='cuda -libs=cublas,cudnn')
            graph, lib, params = tvm.relay.build(module, target='nvptx -libs=cublas,cudnn')
            print('compiled')
            from tvm.contrib import graph_runtime as runtime
            #from tvm.contrib.debugger import debug_runtime as runtime
            func = runtime.create(graph, lib, tvm.gpu())

            x_ =(np.random.randn(n, c, h, w) * 128).astype('float32')
            func.set_input('x', x_)
            timer = func.module.time_evaluator('run', ctx=tvm.gpu(), number=1, repeat=10)
            timed = timer()

            print((n * oc * (h - kh + 1) * (w - kw + 1)) * (kh * kw * ic) / timed.mean / 1e9)
            print('%d us' % int(timed.mean * 1e6))
