import tvm
from tvm import te
from tensorizer.intrinsics import INTRINSICS
import numpy as np

n, m, k = 128, 768, 3072

a = te.placeholder((n, k), 'float16')
b = te.placeholder((k, m), 'float16')

block_k = 2

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

ro, ri = sch[c].split(sch[c].op.reduce_axis[0], nparts=block_k)
rf = sch.rfactor(c, ro)
c_acc = sch.cache_write(rf, 'wmma.accumulator')

xo, xi = sch[c].split(sch[c].op.axis[0], 64)
yo, yi = sch[c].split(sch[c].op.axis[1], 64)

rf_xo, rf_xi = sch[rf].split(sch[rf].op.axis[1], 16)
rf_yo, rf_yi = sch[rf].split(sch[rf].op.axis[2], 16)
rf_xoo, rf_xoi = sch[rf].split(rf_xo, 2)
rf_yoo, rf_yoi = sch[rf].split(rf_yo, 2)
sch[rf].reorder(sch[rf].op.axis[0], rf_xoo, rf_yoo, rf_xoi, rf_yoi, rf_xi, rf_yi)
sch[rf].bind(sch[rf].op.axis[0], thrY)
sch[rf].pragma(rf_xoi, 'tensorize', 'tensorcore.store_c')

a_shared = sch.cache_read(a, 'shared', [c_acc])
aa = sch.cache_read(a_shared, 'wmma.matrix_a', [c_acc])
ax0o, ax0i = sch[aa].split(sch[aa].op.axis[0], 16)
ax1o, ax1i = sch[aa].split(sch[aa].op.axis[1], 16)
sch[aa].reorder(ax0o, ax1o, ax0i, ax1i)
sch[aa].pragma(ax0o, 'tensorize', 'tensorcore.load_a')

b_shared = sch.cache_read(b, 'shared', [c_acc])
bb = sch.cache_read(b_shared, 'wmma.matrix_b', [c_acc])
bx0o, bx0i = sch[bb].split(sch[bb].op.axis[0], 16)
bx1o, bx1i = sch[bb].split(sch[bb].op.axis[1], 16)
sch[bb].reorder(bx0o, bx1o, bx0i, bx1i)
sch[bb].pragma(bx0o, 'tensorize', 'tensorcore.load_b')

sch[c].reorder(xo, yo, xi, yi)
sch[c].bind(xo, blkY)
sch[c].bind(yo, blkX)

sch[rf].compute_at(sch[c], yo)
sch[c_acc].compute_at(sch[rf], rf_yoo)

ro, ri = sch[c_acc].split(sch[c_acc].op.reduce_axis[0], 16)
roo, roi = sch[c_acc].split(ro, 4)
roio, roii = sch[c_acc].split(roi, 2)
acc_xo, acc_xi = sch[c_acc].split(sch[c_acc].op.axis[1], 16)
acc_xoo, acc_xoi = sch[c_acc].split(acc_xo, 2)
acc_yo, acc_yi = sch[c_acc].split(sch[c_acc].op.axis[2], 16)
acc_yoo, acc_yoi = sch[c_acc].split(acc_yo, 2)
print(roo, acc_xoo, acc_yoo, roio, acc_xoi, acc_yoi, roii, acc_xi, acc_yi, ri, sep='\n')
sch[c_acc].reorder(roo, acc_xoo, acc_yoo, roio, acc_xoi, acc_yoi, roii, acc_xi, acc_yi, ri)
sch[c_acc].pragma(acc_xoi, 'tensorize', 'tensorcore')

sch[a_shared].compute_at(sch[c_acc], roo)
sch[b_shared].compute_at(sch[c_acc], roo)
sch[aa].compute_at(sch[c_acc], roio)
sch[bb].compute_at(sch[c_acc], roio)

xio, xii = sch[c].split(xi, nparts=block_k)
yio, yii = sch[c].split(yi, 32)
fused = sch[c].fuse(xii, yio)
sch[c].bind(xio, thrY)
sch[c].bind(fused, thrX)

import tensorizer
with tvm.transform.PassContext(opt_level=4, config={'tir.add_lower_pass': [(1, tensorizer.rewrite)]}):
#with tvm.transform.PassContext(opt_level=4):
    ir = tvm.lower(sch, [a, b, c])
    print(ir)
    quit()
    module = tvm.build(sch, [a, b, c], 'nvptx')

#print(module.imported_modules[0].get_source())
np_a = np.random.randn(n, k).astype('float16')
np_b = np.random.randn(k, m).astype('float16')
np_c = np.random.randn(n, m).astype('float32')

#np_a = np.ones((n, k)).astype('float16')
#np_b = np.ones((k, m)).astype('float16')
#np_c = np.ones((n, m)).astype('float32')

#np_a = np.array(np.array(list(range(k)) * n) % 2).astype('float16')
#np_a.shape = (n, k)
#np_b = np.array(np.array(list(range(k)) * m) % 2).astype('float16')
#np_b.shape = (m, k)
#np_b = np_b.T
#np_c = np.random.randn(n, m).astype('float32')

nd_a = tvm.nd.array(np_a, tvm.gpu())
nd_b = tvm.nd.array(np_b, tvm.gpu())
nd_c = tvm.nd.array(np_c, tvm.gpu())

fte = module.time_evaluator(module.entry_name, ctx=tvm.gpu(), number=1, repeat=10)
res = fte(nd_a, nd_b, nd_c).results
print(np.mean(res) * 1e6)
print((n * m * k) / np.mean(res) / 1e9)

ref = np_a.dot(np_b)
np.testing.assert_allclose(ref, nd_c.asnumpy(), atol=1e-3, rtol=1e-3)
