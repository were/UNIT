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

blkY = tvm.te.thread_axis('blockIdx.y')
blkX = tvm.te.thread_axis('blockIdx.x')
thrZ = tvm.te.thread_axis('threadIdx.z')
thrY = tvm.te.thread_axis('threadIdx.y')
thrX = tvm.te.thread_axis('threadIdx.x')

sch = te.create_schedule(c.op)

ro, ri = sch[c].split(sch[c].op.reduce_axis[0], 64)
rio, rii = sch[c].split(ri, 32)
rf = sch.rfactor(c, rio)
xo, xi = sch[c].split(sch[c].op.axis[0], 32)
yo, yi = sch[c].split(sch[c].op.axis[1], 32)
sch[c].reorder(xo, yo, xi, yi)
sch[c].bind(xo, blkY)
sch[c].bind(yo, blkX)
xio, xii = sch[c].split(xi, nparts=2)
sch[c].bind(xio, thrY)

sch[rf].compute_at(sch[c], yo)
c_acc = sch.cache_write(rf, 'wmma.accumulator')

rf_r, rf_x, rf_y = sch[rf].op.axis
rf_xo, rf_xi = sch[rf].split(rf_x, 16)
rf_yo, rf_yi = sch[rf].split(rf_y, 16)
sch[rf].reorder(rf_r, rf_xo, rf_yo, rf_xi, rf_yi)
sch[rf].bind(rf_r, thrY)
sch[rf].pragma(rf_xo, 'tensorize', 'tensorcore.store_c')

c_ro, c_ri = sch[c_acc].op.reduce_axis
_, c_x, c_y = sch[c_acc].op.axis
c_xo, c_xi = sch[c_acc].split(c_x, 16)
c_yo, c_yi = sch[c_acc].split(c_y, 16)
c_roo, c_roi = sch[c_acc].split(c_ro, 2)
c_rio, c_rii = sch[c_acc].split(c_ri, 16)
sch[c_acc].compute_at(sch[rf], rf_r)
sch[c_acc].reorder(c_roo, c_roi, c_rio, c_xo, c_yo, c_xi, c_yi, c_rii)
sch[c_acc].pragma(c_xo, 'tensorize', 'tensorcore')

#a_shared = sch.cache_read(a, 'shared', [c_acc])
a_shared = a
a_frag = sch.cache_read(a_shared, 'wmma.matrix_a', [c_acc])
sch[a_frag].compute_at(sch[c_acc], c_rio)
axo, axi = sch[a_frag].split(sch[a_frag].op.axis[0], 16)
ayo, ayi = sch[a_frag].split(sch[a_frag].op.axis[1], 16)
sch[a_frag].reorder(axo, ayo, axi, ayi)
sch[a_frag].pragma(axo, 'tensorize', 'tensorcore.load_a')
#sch[a_shared].compute_at(sch[c_acc], c_roi)

#b_shared = sch.cache_read(b, 'shared', [c_acc])
b_shared = b
b_frag = sch.cache_read(b_shared, 'wmma.matrix_b', [c_acc])
sch[b_frag].compute_at(sch[c_acc], c_rio)
bxo, bxi = sch[b_frag].split(sch[b_frag].op.axis[0], 16)
byo, byi = sch[b_frag].split(sch[b_frag].op.axis[1], 16)
sch[b_frag].reorder(bxo, byo, bxi, byi)
sch[b_frag].pragma(bxo, 'tensorize', 'tensorcore.load_b')
#sch[b_shared].compute_at(sch[c_acc], c_roi)

import tensorizer
with tvm.transform.PassContext(opt_level=4, config={'tir.add_lower_pass': [(1, tensorizer.rewrite)]}):
#with tvm.transform.PassContext(opt_level=4):
    ir = tvm.lower(sch, [a, b, c])
    module = tvm.build(sch, [a, b, c], 'nvptx')
    print(ir)

#print(module.imported_modules[0].get_source())
np_a = np.random.randn(n, k).astype('float16')
np_b = np.random.randn(k, m).astype('float16')
np_c = np.random.randn(n, m).astype('float32')

#np_a = np.ones((n, k)).astype('float16')
#np_b = np.ones((k, m)).astype('float16')
#np_c = np.ones((n, m)).astype('float32')

np_a = np.array(np.array(list(range(k)) * n) % 3).astype('float16')
np_a.shape = (n, k)
np_b = np.array(np.array(list(range(k)) * m) % 3).astype('float16')
np_b.shape = (m, k)
np_b = np_b.T
np_c = np.random.randn(n, m).astype('float32')

nd_a = tvm.nd.array(np_a, tvm.gpu())
nd_b = tvm.nd.array(np_b, tvm.gpu())
nd_c = tvm.nd.array(np_c, tvm.gpu())

fte = module.time_evaluator(module.entry_name, ctx=tvm.gpu(), number=1, repeat=10)
res = fte(nd_a, nd_b, nd_c).results
print(np.mean(res) * 1e6)
print((n * m * k) / np.mean(res) / 1e9)

ref = np_a.dot(np_b)
np.testing.assert_allclose(ref, nd_c.asnumpy(), atol=1e-3, rtol=1e-3)
