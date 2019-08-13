import tvm

n = 8192

a = tvm.placeholder((n, ))
b = tvm.placeholder((n, ))
c = tvm.compute((n, ), lambda x: a[x] * b[x])

sch = tvm.create_schedule(c.op)
xo, xi = sch[c].split(c.op.axis[0], 4)
xoo, xoi = sch[c].split(xo, 128)

thrx = tvm.thread_axis('threadIdx.x')
blcx = tvm.thread_axis('blockIdx.x')

sch[c].vectorize(xi)
sch[c].bind(xoi, thrx)
sch[c].bind(xoo, blcx)

func = tvm.lower(sch, [a, b, c], simple_mode=True)

module = tvm.build(sch, [a, b, c], target='cuda')

timer = module.time_evaluator(module.entry_name, ctx=tvm.gpu(0), number=10)

import numpy as np

a = tvm.ndarray.array(np.random.randn(n).astype('float32'), tvm.gpu(0))
b = tvm.ndarray.array(np.random.randn(n).astype('float32'), tvm.gpu(0))
c = tvm.ndarray.array(np.random.randn(n).astype('float32'), tvm.gpu(0))

module(a, b, c)

timer(a, b, c)
