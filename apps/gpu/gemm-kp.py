import tvm
from tvm import te
from tensorizer.intrinsics import INTRINSICS
import numpy as np

n, m, k = 128, 768, 3072

a = te.placeholder((n, k), 'float16')
b = te.placeholder((k, m), 'float16')

#b = te.placeholder((k, m), 'float16')

block_k = 4

rv = te.reduce_axis((0, k), )

def compute(x, y):
    lhs = a[x, rv].astype('float32')
    rhs = b[rv, y].astype('float32')
    return te.sum(lhs * rhs, axis=[rv])

c = te.compute((n, m), compute)


blkX = tvm.te.thread_axis('blockIdx.x')
blkY = tvm.te.thread_axis('blockIdx.y')
thrY = tvm.te.thread_axis('threadIdx.y')
thrX = tvm.te.thread_axis('threadIdx.x')

sch = te.create_schedule(c.op)

ro, ri = sch[c].split(sch[c].op.reduce_axis[0], 4)
rf = sch.rfactor(c, ri)
c_acc = sch.cache_write(rf, 'wmma.accumulator')

xo, xi = sch[c].split(sch[c].op.axis[0], 16)
yo, yi = sch[c].split(sch[c].op.axis[1], 16)
sch[c].reorder(xo, yo, xi, yi)
sch[c].bind(xo, blkY)
sch[c].bind(yo, blkX)

sch[rf].compute_at(sch[c], yo)
sch[c_acc].compute_at(sch[c], yo)
#sch[c_acc].compute_at(sch[rf], sch[rf].op.axis[0])

ro, ri = sch[c_acc].split(sch[c_acc].op.reduce_axis[0], 16)
acc_xo, acc_xi = sch[c_acc].split(sch[c_acc].op.axis[1], 16)
acc_yo, acc_yi = sch[c_acc].split(sch[c_acc].op.axis[2], 16)
sch[c_acc].reorder(sch[c_acc].op.axis[0], ro, acc_xo, acc_yo, acc_xi, acc_yi, ri)

sch[c_acc].bind(sch[c_acc].op.axis[0], thrY)
#sch[rf].bind(sch[rf].op.axis[0], thrY)

sch[c_acc].pragma(acc_xi, 'tensorize', 'tensorcore')
sch[rf].pragma(sch[rf].op.axis[1], 'tensorize', 'tensorcore')

xio, xii = sch[c].split(xi, nparts=4)
yio, yii = sch[c].split(yi, 2)
fused = sch[c].fuse(xii, yio)
sch[c].bind(xio, thrY)
sch[c].bind(fused, thrX)

import tensorizer
with tvm.transform.PassContext(opt_level=3, config={'tir.add_lower_pass': [(1, tensorizer.rewrite)]}):
#with tvm.transform.PassContext(opt_level=4):
    ir = tvm.lower(sch, [a, b, c], simple_mode=True)
    print(ir)
    #quit()
    module = tvm.build(sch, [a, b, c], 'nvptx')

np_a = np.ones((n, k)).astype('float16')
np_b = np.ones((k, m)).astype('float16')
np_c = np.ones((n, m)).astype('float32')

nd_a = tvm.nd.array(np_a, tvm.gpu())
nd_b = tvm.nd.array(np_b, tvm.gpu())
nd_c = tvm.nd.array(np_c, tvm.gpu())

fte = module.time_evaluator(module.entry_name, ctx=tvm.gpu(), number=10)
print(fte(nd_a, nd_b, nd_c).mean * 1e6)

#print(module.imported_modules[0].get_source())

ref = np_a.dot(np_b)
np.testing.assert_allclose(ref, nd_c.asnumpy(), atol=1e-5, rtol=1e-5)