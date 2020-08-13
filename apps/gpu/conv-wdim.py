import tvm
from tvm import te, arith
from tensorizer.intrinsics import INTRINSICS
import numpy as np

n, c, h, w = 1, 192, 27, 64
kh, kw, ic, ko = 1, 1, c, 192

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

def _conv2d_schedule_wdim(sch, conv):
    a, b = conv.op.input_tensors

    rc = sch[conv].op.reduce_axis[0]
    rco, rci = sch[conv].split(rc, 64)
    rcio, rcii = sch[conv].split(rci, 16)
    rf = sch.rfactor(conv, rcio)

    batch, oc, x, y, ob = list(sch[conv].op.axis)
    yo, yi = sch[conv].split(y, 32)
    oo, oi = sch[conv].split(ob, 16)
    yio, yii = sch[conv].split(yi, 16)
    oio, oii = sch[conv].split(oi, 16)
    oco, oci = sch[conv].split(oc, 2)
    sch[conv].reorder(batch, x, yo, oco, oo, oci, yio, oio, yii, oii)
    sch[rf].compute_at(sch[conv], oo)

    sch[conv].bind(oco, te.thread_axis('blockIdx.y'))
    sch[conv].bind(x, te.thread_axis('blockIdx.x'))
    fused = sch[conv].fuse(oci, yio, oio)
    sch[conv].bind(fused, te.thread_axis('threadIdx.y'))
    vo, vi = sch[conv].split(oii, 8)
    sch[conv].vectorize(vi)
    fused = sch[conv].fuse(yii, vo)
    sch[conv].bind(fused, te.thread_axis('threadIdx.x'))

    cc = sch.cache_write(rf, 'wmma.accumulator')
    sch[cc].compute_at(sch[rf], sch[rf].op.axis[0])
    sch[rf].bind(sch[rf].op.axis[0], te.thread_axis('threadIdx.y'))

    rc, cb, coc, cx, cy, cob = sch[cc].op.axis
    crh, crw, crco, crci = sch[cc].op.reduce_axis
    cyo, cyi = sch[cc].split(cy, 16)
    sch[cc].reorder(rc, cb, crco, crh, crw, cx, cyo, coc, cyi, cob, crci)
    sch[cc].pragma(cyo, 'tensorize', 'tensorcore')

    aa = sch.cache_read(a, 'wmma.matrix_a', [cc])
    sch[aa].compute_at(sch[cc], crw)
    ao, ai = sch[aa].split(sch[aa].op.axis[3], 16)
    sch[aa].pragma(ao, 'tensorize', 'tensorcore.load_a')
    bb = sch.cache_read(b, 'wmma.matrix_b', [cc])
    sch[bb].compute_at(sch[cc], crw)
    sch[bb].pragma(sch[bb].op.axis[0], 'tensorize', 'tensorcore.load_b')

    rc, batch, oc, x, y, ob = sch[rf].op.axis
    yio, yii = sch[rf].split(y, 16)
    sch[rf].reorder(rc, batch, yio, oc, x, yii, ob)
    sch[rf].pragma(yio, 'tensorize', 'tensorcore.store_c')

    ir = tvm.lower(sch, [a, b, conv])


_conv2d_schedule_wdim(sch, conv)

def tracer(module, info, is_before):
    import time
    global timing
    if bool(is_before):
        timing = time.time()
    else:
        print('Executes: ', info.name, (time.time() - timing) * 1000)

#np_a = np.random.randn(n, c // 16, h, w, 16).astype('float16')
#np_b = np.random.randn(ko // 16, ic // 16, kh, kw, 16, 16).astype('float16')
np_a = (np.arange(n * (c // 16) * h * w * 16) % 7).astype('float16')
np_b = (np.arange((ko // 16) * kh * kw * ic * 16) % 7).astype('float16')
np_a.shape = (n, c // 16, h, w, 16)
np_b.shape = (ko // 16, ic // 16, kh, kw, 16, 16)

np_c = np.random.randn(n, ko // 16, h - kh + 1, w - kw + 1, 16).astype('float32')

nd_a = tvm.nd.array(np_a, tvm.gpu())
nd_b = tvm.nd.array(np_b, tvm.gpu())
nd_c = tvm.nd.array(np_c, tvm.gpu())

import tensorizer
passes = [(1, tensorizer.loop_swizzle), (1, tensorizer.rewrite), (1, tensorizer.inject_sync), (1, tensorizer.sliding_window)]
with tvm.transform.PassContext(opt_level=4, config={'tir.add_lower_pass': passes}):
#with tvm.transform.PassContext(opt_level=4):
    module = tvm.build(sch, [a, b, conv], 'nvptx')
    fte = module.time_evaluator(module.entry_name, ctx=tvm.gpu(), number=3, repeat=10)
    res = fte(nd_a, nd_b, nd_c).results
    print('exec: ', np.mean(res) * 1e6)

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

res = cpu_c.asnumpy()
ref = nd_c.asnumpy()
for ax0 in range(n):
    for ax1 in range(ko // 16):
        for ax2 in range(h - kh + 1):
            for ax3 in range(w - kw + 1):
                for ax4 in range(16):
                    assert abs(res[ax0, ax1, ax2, ax3, ax4] - ref[ax0, ax1, ax2, ax3, ax4]) < 1e-3, \
                           (ax0, ax1, ax2, ax3, ax4, res[ax0, ax1, ax2, ax3, ax4], ref[ax0, ax1, ax2, ax3, ax4])

np.testing.assert_allclose(cpu_c.asnumpy(), nd_c.asnumpy(), atol=1e-3, rtol=1e-3)
print('correctness yes!')
