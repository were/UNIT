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
n, c, h, w, oc, ic, kh, kw, sh, sw = map(int, input().split())

oh = (h - kh) // sh + 1
ow = (w - kw) // sw + 1

import time
timing = -1

def tracer(module, info, is_before):
    global timing
    if bool(is_before):
        timing = time.time()
    else:
        print('Executes: ', info.name, (time.time() - timing) * 1000)

from tensorizer import tune
tune.enable = False

result = info = 1e9
for i in [None, 'fuse', 'pad'] if ow < 32 else [None]:
    j = 16
    while True:
        diffc = diffoc = diffh = diffw = 0
        #if c % 64:
        #    diffc = 64 - c % 64

        #if oc % 32:
        #    diffoc = 32 - oc % 32

        #can_fuse = can_pad = True
        #if i == 'pad':
        #    can_fuse = False
        #if i == 'fuse':
        #    can_pad = False
        #if not ((oh * ow % 32 == 0 and 32 % ow == 0) or ow % 32 == 0):
        #    first_h = sh - (h - kh) % sh
        #    first_w = sw - (w - kw) % sw
        #    max_diff_h = 32 - oh % 32
        #    max_diff_w = 32 - ow % 32
        #    diffh = diffw = 1e9
        #    for i in range(max_diff_h + 1):
        #        for j in range(max_diff_w + 1):
        #            if (((oh + i) * (ow + j) % 32 == 0 and 32 % (ow + j) == 0 and can_fuse) or ((ow + j) % 32 == 0 and can_pad)) and i + j < diffh + diffw:
        #                def to_pad(padding, first, stride):
        #                    if padding == 0:
        #                        return 0
        #                    assert padding >= 1
        #                    return (padding - 1) * stride + first
        #                diffh, diffw = to_pad(i, first_h, sh), to_pad(j, first_w, sw)
        #    #assert (height + diffh - kh + 1) * (width + diffw - kw + 1) % 32 == 0


        #var_x = relay.var('x', shape=(n, (c + diffc) // 16, (h + diffh), (w + diffw), 16), dtype='float16')
        #var_w = relay.const(tvm.nd.array((np.random.randn((oc + diffoc) // 16, (c + diffc) // 16, kh, kw, 16, 16) * 128).astype('float16')))
        #conv2d = relay.nn.conv2d(var_x, var_w, out_dtype='float32', kernel_size=(kh, kw), channels=oc + diffoc, strides=(sh, sw), data_layout='NCHW16c', kernel_layout='OIHW16i16o')
        #if diffc or diffoc or diffh or diffw:
        #    y = relay.strided_slice(conv2d,
        #                            begin=relay.const(tvm.nd.array([0, 0, 0, 0])),
        #                            end=relay.const(tvm.nd.array([n, oc, oh, ow])))
        #else:
        #    y = conv2d
        var_x = relay.var('x', shape=(n, c, h, w), dtype='float32')
        var_w = relay.const(tvm.nd.array((np.random.randn(oc, ic, kh, kw) * 128).astype('float32')))
        var_b = relay.const(tvm.nd.array((np.random.randn(1, oc, 1, 1) * 128).astype('float32')))
        conv2d = relay.nn.conv2d(var_x, var_w, out_dtype='float32', kernel_size=(kh, kw), channels=oc, strides=(sh, sw), out_layout='NCHW16c')
        y = conv2d

        func = relay.Function([var_x], y)
        module = tvm.IRModule()
        module['main'] = func

        tune.padding = i
        tune.splitk = j
        passes = [(1, tensorizer.rewrite)]
        with tvm.transform.PassContext(opt_level=0, trace=tracer, config={'tir.add_lower_pass': passes}):
        #with tvm.transform.PassContext(opt_level=4, trace=tracer):
            #graph, lib, params = tvm.relay.build(module, target='cuda -libs=cublas,cudnn')
            graph, lib, params = tvm.relay.build(module, target='nvptx -libs=cublas,cudnn')
            from tvm.contrib import graph_runtime as runtime
            from tvm.contrib.debugger import debug_runtime as runtime
            func = runtime.create(graph, lib, tvm.gpu())

            x_ =(np.random.randn(n, c, h, w) * 128).astype('float32')
            func.set_input('x', x_)
            timer = func.module.time_evaluator('run', ctx=tvm.gpu(), number=2, repeat=10)

            timed = timer()
            while np.var(timed.results) > 1e-5:
                timed = timer()

            if timed.mean < result:
                result = timed.mean
                info = (i, j)


        relay.backend.compile_engine.get().clear()
        j <<= 1
        if j > tune.total_idx:
            break