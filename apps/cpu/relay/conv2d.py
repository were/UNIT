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

x = relay.var('x', shape=(1, 3, 128, 128), dtype='int8')
w = relay.var('w', shape=(256, 3, 3, 3), dtype='int8')
b = relay.var('b', shape=(1, 256, 1, 1), dtype='int32')
conv2d = relay.nn.conv2d(x, w, out_dtype='int32', kernel_size=(3, 3), channels=256)
biased = relay.add(conv2d, b)
y = relay.multiply(biased, relay.const(11, 'int32'))

func = relay.Function([x, w, b], y)
module = tvm.IRModule()
module['main'] = func

def alter(attrs, inputs, tinfos, out_type):
    data, weight = inputs
    new_attrs = dict(attrs)
    new_attrs['data_layout'] = 'NCHW'
    new_attrs['kernel_layout'] = 'HWIO'
    return relay.nn.contrib_conv2d_nchwc(data, weight, **new_attrs)


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

    x_ = (np.random.randn(1, 3, 128, 128) * 128).astype('int8')
    w_ = (np.random.randn(256, 3, 3, 3) * 128).astype('int8')
    b_ = (np.random.randn(1, 256, 1, 1) * 128).astype('int32')
    module.set_input('x', x_)
    module.set_input('w', w_)
    module.set_input('b', b_)

    timer = module.module.time_evaluator('run', ctx=tvm.cpu(0), number=3, repeat=10)
    timed = timer()

    print(timed.mean)
