import tvm
from tvm import te, arith
from tensorizer.intrinsics import INTRINSICS
import numpy as np

n, c, h, w = 1, 128, 10, 10
kh, kw, ic, ko = 3, 3, c, 128

a = te.placeholder((n, c // 16, h, w, 16), 'float16')
b = te.placeholder((ko // 16, ic // 16, kh, kw, 16, 16), 'float16')

rc = te.reduce_axis((0, c), 'rc')
rh = te.reduce_axis((0, kh), 'rh')
rw = te.reduce_axis((0, kw), 'rw')

conv = te.compute((n, ko // 16, h - kh + 1, w - kw + 1, 16),
               lambda batch, o_chunk, x, y, ob:
                te.sum(a[batch, rc // 16, x + rh, y + rw, rc % 16].astype('float32') *
                       b[o_chunk, rc // 16, rh, rw, rc % 16, ob].astype('float32'), axis=[rc, rh, rw]))

from tensorizer.intrinsics.pattern import mm_tensorcore

sch = tvm.te.create_schedule(conv.op)
info = list(arith._ffi_api.MatchTensorizer(conv.op, mm_tensorcore()))

#assert info
#print(info)

batch, oc, x, y, ob = list(sch[conv].op.axis)

cc = sch.cache_write(conv, 'wmma.accumulator')
xy = sch[conv].fuse(x, y)
#yo, yi = sch[conv].split(y, 32)
xyo, xyi = sch[conv].split(xy, 32)
oo, oi = sch[conv].split(ob, 16)
xyio, xyii = sch[conv].split(xyi, 16)
oio, oii = sch[conv].split(oi, 16)
oco, oci = sch[conv].split(oc, 2)
#print(batch, x, yo, oco, oci, oo, yio, oio, yii, oii, sep='\n')
sch[conv].reorder(batch, xyo, oco, oo, xyio, oci, oio, xyii, oii)

sch[cc].compute_at(sch[conv], oco)
cb, coc, cx, cy, cob = sch[cc].op.axis
cxy = sch[cc].fuse(cx, cy)
crc, crh, crw = sch[cc].op.reduce_axis
cxyo, cxyi = sch[cc].split(cxy, 16)
crco, crci = sch[cc].split(crc, 16)
crcoo, crcoi = sch[cc].split(crco, 2)
#print(cb, crh, crw, crco, coc, cx, cyo, cyi, cob, crci, sep='\n')
sch[cc].reorder(cb, crcoo, crh, crw, crcoi, cxyo, coc, cxyi, cob, crci)
sch[cc].pragma(cxyo, 'tensorize', 'tensorcore')

a_shared = sch.cache_read(a, 'shared', [cc])
sch[a_shared].compute_at(sch[cc], crcoi)
as23 = sch[a_shared].fuse(sch[a_shared].op.axis[2], sch[a_shared].op.axis[3])
sch[a_shared].bind(as23, te.thread_axis('threadIdx.x'))
as4o, as4i = sch[a_shared].split(sch[a_shared].op.axis[4], 8)
sch[a_shared].vectorize(as4i)

aa = sch.cache_read(a_shared, 'wmma.matrix_a', [cc])
#aa = sch.cache_read(a, 'wmma.matrix_a', [cc])
sch[aa].compute_at(sch[cc], crcoi)
a23 = sch[aa].fuse(sch[aa].op.axis[2], sch[aa].op.axis[3])
a23o, a23i = sch[aa].split(a23, 16)
sch[aa].pragma(a23o, 'tensorize', 'tensorcore.load_a')

#b_shared = sch.cache_read(b, 'shared', [cc])
#bb = sch.cache_read(b_shared, 'wmma.matrix_b', [cc])
bb = sch.cache_read(b, 'wmma.matrix_b', [cc])
sch[bb].compute_at(sch[cc], crcoi)
#sch[b_shared].compute_at(sch[cc], crcoo)
sch[bb].pragma(sch[bb].op.axis[0], 'tensorize', 'tensorcore.load_b')

sch[conv].pragma(xyio, 'tensorize', 'tensorcore.store_c')

sch[conv].bind(xyo, tvm.te.thread_axis('blockIdx.y'))
sch[conv].bind(oco, tvm.te.thread_axis('blockIdx.x'))


def tracer(module, info, is_before):
    import time
    global timing
    if bool(is_before):
        timing = time.time()
    else:
        print('Executes: ', info.name, (time.time() - timing) * 1000)

np_a = np.random.randn(n, c // 16, h, w, 16).astype('float16')
np_b = np.random.randn(ko // 16, ic // 16, kh, kw, 16, 16).astype('float16')
#np_a = (np.arange(n * (c // 16) * h * w * 16) % 7).astype('float16')
#np_b = (np.arange((ko // 16) * kh * kw * ic * 16) % 7).astype('float16')
#np_a.shape = (n, c // 16, h, w, 16)
#np_b.shape = (ko // 16, kh, kw, ic, 16)

np_c = np.random.randn(n, ko // 16, h - kh + 1, w - kw + 1, 16).astype('float32')

nd_a = tvm.nd.array(np_a, tvm.gpu())
nd_b = tvm.nd.array(np_b, tvm.gpu())
nd_c = tvm.nd.array(np_c, tvm.gpu())

import tensorizer
with tvm.transform.PassContext(opt_level=4, config={'tir.add_lower_pass': [(1, tensorizer.rewrite)]}):
#with tvm.transform.PassContext(opt_level=4):
    ir = tvm.lower(sch, [a, b, conv])
    print(ir)
    module = tvm.build(sch, [a, b, conv], 'nvptx')
    fte = module.time_evaluator(module.entry_name, ctx=tvm.gpu(), number=1, repeat=10)
    res = fte(nd_a, nd_b, nd_c).results
    print(np.mean(res) * 1e6)

    import functools, operator
    elem_c = functools.reduce(operator.mul, np_c.shape, 1)
    coef_b = functools.reduce(operator.mul, [ic, kh, kw], 1)
    print(elem_c * coef_b / np.mean(res) / 1e9)

vanilla = tvm.te.create_schedule(conv.op)
print(*vanilla[conv].op.reduce_axis, sep='\n')
vb, vc, vx, vy, vob = vanilla[conv].op.axis
vrc, vrh, vrw = vanilla[conv].op.reduce_axis
vxo, vxi = vanilla[conv].split(vx, 32)
vyo, vyi = vanilla[conv].split(vy, 4)
fusion = vanilla[conv].fuse(vb, vc, vxo)
vanilla[conv].reorder(fusion, vxi, vyo, vrc, vrh, vrw, vyi, vob)
vanilla[conv].unroll(vyi)
vanilla[conv].vectorize(vob)
vanilla[conv].parallel(fusion)

#print(tvm.lower(vanilla, [a, b, conv], simple_mode=True))
vanilla = tvm.build(vanilla, [a, b, conv])
cpu_a = tvm.nd.array(np_a, tvm.cpu())
cpu_b = tvm.nd.array(np_b, tvm.cpu())
cpu_c = tvm.nd.array(np_c, tvm.cpu())
vanilla(cpu_a, cpu_b, cpu_c)

np.testing.assert_allclose(cpu_c.asnumpy(), nd_c.asnumpy(), atol=1e-3, rtol=1e-3)
print('correctness yes!')