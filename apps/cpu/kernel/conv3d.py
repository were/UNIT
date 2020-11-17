import tvm
import tensorizer
import logging
import sys
from tvm import relay
from tvm import autotvm
import numpy as np
from topi.util import get_const_tuple

import topi
from tvm.relay import op

workloads = [
#(256, 16, 16, 256, 3, 1),
#(512, 9, 9, 512, 3, 1),
(128, 30, 30, 128, 3, 1),
#(64, 56, 56, 128, 1, 2),
#(64, 58, 58, 64, 3, 1),
#(128, 28, 28, 256, 1, 2),
#(256, 16, 16, 512, 3, 2),
#(64, 58, 58, 128, 3, 2),
#(4, 230, 230, 64, 7, 2),
#(128, 30, 30, 256, 3, 2),
#(256, 14, 14, 512, 1, 2)
]

output = []

for ic, h, w, oc, kernel, stride in workloads:

    d = 16
    
    if ic % 4:
        ic += 4 - ic % 4
    
    if oc % 16:
        oc += 16 - oc % 16
    
    if ic % 16 == 0:
        c_chunk = 16
        a = tvm.te.placeholder((1, ic // 16, d, h, w, 16), dtype='int8')
        b = tvm.te.placeholder((oc // 16, ic // 16, kernel, kernel, kernel, 4, 16, 4), dtype='int8')
        rco = tvm.te.reduce_axis((0, ic // 16))
        rcm = tvm.te.reduce_axis((0, 4))
        rci = tvm.te.reduce_axis((0, 4))
    else:
        assert ic % 4 == 0
        c_chunk = 4
        a = tvm.te.placeholder((1, ic // 4, d, h, w, 4), dtype='int8')
        b = tvm.te.placeholder((oc // 16, ic // 4, kernel, kernel, kernel, 1, 16, 4), dtype='int8')
        rco = tvm.te.reduce_axis((0, ic // 4))
        rcm = tvm.te.reduce_axis((0, 1))
        rci = tvm.te.reduce_axis((0, 4))
    
    rd = tvm.te.reduce_axis((0, kernel))
    rh = tvm.te.reduce_axis((0, kernel))
    rw = tvm.te.reduce_axis((0, kernel))
    
    
    c = tvm.te.compute((1, oc // 16, (d - kernel) // stride + 1, (w - kernel) // stride + 1, (h - kernel) // stride + 1, 16),
                       lambda batch, ochunk, x, y, z, oblock: tvm.te.sum(a[batch, rco, stride*x+rd, stride*y+rh, stride*z+rw, rcm*4+rci]
                                                                          .astype('int32') *
                                                                         b[ochunk, rco, rd, rh, rw, rcm, oblock, rci]
                                                                          .astype('int32'), axis=[rco, rd, rh, rw, rcm, rci]))

    print(get_const_tuple(a.shape))
    print(get_const_tuple(b.shape))
    print(get_const_tuple(c.shape))
    #a = tvm.te.placeholder((N, C, H, W, c), dtype='int8')
    #b = tvm.te.placeholder((O, I, KH, KW, e, o, i), dtype='int8')
    
    passes = [(1, tensorizer.rewrite)]
    from tensorizer import tune
    tune.cpu_idx = 0
    target = -1
    results = []
    virgin = True
    while True:
        with tvm.transform.PassContext(opt_level=3, config={'tir.add_lower_pass': passes}), tvm.target.create('llvm -mcpu=cascadelake'):
            sch = tensorizer.INTRINSICS['vnni']['schedule']([c], (stride, stride, stride))
        
            module = tvm.build(sch, [a, b, c], 'llvm -mcpu=cascadelake')
            np_a = np.zeros(get_const_tuple(a.shape), dtype='int8')
            np_b = np.zeros(get_const_tuple(b.shape), dtype='int8')
            np_c = np.zeros(get_const_tuple(c.shape), dtype='int32')
            nd_a = tvm.nd.array(np_a, tvm.cpu())
            nd_b = tvm.nd.array(np_b, tvm.cpu())
            nd_c = tvm.nd.array(np_c, tvm.cpu())
            fte = module.time_evaluator(module.entry_name, ctx=tvm.cpu(), number=3, repeat=10)
            res = fte(nd_a, nd_b, nd_c)
            while np.var(res.results) > 1e-5:
                res = fte(nd_a, nd_b, nd_c)
            import functools, operator
            total = functools.reduce(operator.mul, get_const_tuple(c.shape), 1) * (kernel ** 3) * (ic // c_chunk)
            results.append(res.mean)
        
        relay.backend.compile_engine.get().clear()
        tune.cpu_idx += 1
        #if tune.cpu_idx - target > 8:
        #    break
        if tune.cpu_idx >= tune.total_idx - 1:
            break
    #print(results)
    results = min(results)
    output.append((total, results * 1e6, total / results / 1e9))
    open('res', 'a').write(str(output[-1]) + '\n')

print(*output, sep='\n')
