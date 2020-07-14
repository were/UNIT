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

x = relay.var('x', shape=(128, 768), dtype='float32')
w = relay.var('w', shape=(3072, 768), dtype='float32')
y = relay.nn.dense(x, w, out_dtype='float32')

func = relay.Function([x, w], y)
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
    graph, lib, params = tvm.relay.build(module, target='llvm -mcpu=cascadelake -libs=cblas')
    #graph, lib, params = tvm.relay.build(module, target='llvm -mcpu=cascadelake')
    from tvm.contrib import graph_runtime as runtime
    module = runtime.create(graph, lib, tvm.cpu())

    x_ = (np.random.randn(128, 768) * 128).astype('float')
    w_ = (np.random.randn(3072, 768) * 128).astype('float')
    module.set_input('x', x_)
    module.set_input('w', w_)

    timer = module.module.time_evaluator('run', ctx=tvm.cpu(0), number=3, repeat=10)
    timed = timer()

    print(timed.mean * 1e6)
