import tvm
from tvm import te
from tensorizer.intrinsics import INTRINSICS
import numpy as np

n, m, k = 128, 768, 3072

a = te.placeholder((n, k), 'float16')
b = te.placeholder((k, m), 'float16')

block_k = 4

rv = te.reduce_axis((0, k // block_k), )

def compute(z, x, y):
    lhs = a[x, z * (k // block_k) + rv].astype('float32')
    rhs = b[rv + z * (k // block_k), y].astype('float32')
    return te.sum(lhs * rhs, axis=[rv])

c = te.compute((block_k, n, m, ), compute)

blkX = tvm.te.thread_axis('blockIdx.x')
blkY = tvm.te.thread_axis('blockIdx.y')
thrY = tvm.te.thread_axis('threadIdx.y')
thrX = tvm.te.thread_axis('threadIdx.x')

sch = te.create_schedule(c.op)

cc = sch.cache_write(c, 'wmma.accumulator')
#cc = sch.cache_write(c, 'local')

rb, x, y = sch[c].op.axis
xo, xi = sch[c].split(x, 16)
yo, yi = sch[c].split(y, 16)

sch[c].reorder(rb, xo, yo, xi, yi)
sch[c].bind(xo, blkY)
sch[c].bind(yo, blkX)
sch[c].bind(rb, thrY)

sch[cc].compute_at(sch[c], yo)
r = sch[cc].op.reduce_axis[0]
ro, ri = sch[cc].split(r, 16)
_, cx, cy = sch[cc].op.axis
sch[cc].reorder(ro, cx, cy, ri)
sch[cc].pragma(cx, 'tensorize', 'tensorcore')
sch[c].pragma(xi, 'tensorize', 'tensorcore')

import tensorizer
with tvm.transform.PassContext(opt_level=4, config={'tir.add_lower_pass': [(1, tensorizer.rewrite)]}):
#with tvm.transform.PassContext(opt_level=4):
    ir = tvm.lower(sch, [a, b, c], simple_mode=True)
    #print(ir)
    #quit()
    module = tvm.build(sch, [a, b, c], 'nvptx')
    #module = tvm.build(sch, [a, b, c], 'cuda')
    #print(module.imported_modules[0].get_source())

np_a = np.random.randn(n, k).astype('float16')
np_b = np.random.randn(k, m).astype('float16')
np_c = np.random.randn(block_k, n, m).astype('float32')

nd_a = tvm.nd.array(np_a, tvm.gpu())
nd_b = tvm.nd.array(np_b, tvm.gpu())
nd_c = tvm.nd.array(np_c, tvm.gpu())

#module(nd_a, nd_b, nd_c)
fte = module.time_evaluator(module.entry_name, ctx=tvm.gpu(), number=10)
print((n * m * k) / fte(nd_a, nd_b, nd_c).mean / 1e9)

for i in range(block_k):
    tmpa = np_a[:,i*(k//block_k):(i+1)*(k//block_k)]
    tmpb = np_b[i*(k//block_k):(i+1)*(k//block_k),:]
    tmpa.dot(tmpb)
    np.testing.assert_allclose(tmpa.dot(tmpb), nd_c.asnumpy()[i,:,:], atol=1e-3, rtol=1e-3)

#ref = np_a.dot(np_b)
#np.testing.assert_allclose(ref, nd_c.asnumpy(), atol=1e-5, rtol=1e-5)
