import tvm

n, h, w, c = 1, 128, 128, 64
o, kc, kh, kw = 64, c, 3, 3

img = tvm.placeholder((n, h, w, c), 'int8', 'input')
knl = tvm.placeholder((kh, kw, o // 16, c // 4, 16, 4), 'int8', 'kernel')

rc, rh, rw = tvm.reduce_axis((0, kc), 'rc'), tvm.reduce_axis((0, kh), 'rh'), tvm.reduce_axis((0, kw), 'rw')

conv = tvm.compute(
        (n, h - kh + 1, w - kw + 1, o),
        lambda bn, x, y, oc:
            tvm.sum(img[bn, x + rh, y + rw, rc].astype('int32') * knl[rh, rw, oc // 16, rc // 4, oc % 16, rc % 4],
        axis=[rc, rh, rw]),
        'conv')

ops = n * (h - kh + 1) * (w - kw + 1) * o * kc * kh * kw / 64

sch = tvm.create_schedule(conv.op)
#vannila = tvm.build(sch, [img, knl, conv], 'llvm -mcpu=cascadelake')

bn, x, y, oc = conv.op.axis
rc, rh, rw = conv.op.reduce_axis
sch[conv].parallel(x)


oco, oci = sch[conv].split(oc, 16)
rco, rci = sch[conv].split(rc, 4)

sch[conv].reorder(bn, x, rh, rw, y, rco, oco, oci, rci)
#sch[conv].unroll(oco)
sch[conv].pragma(oci, 'vnni')
#cached = sch.cache_read(img, 'global', [conv])
#sch[cached].compute_at(sch[conv], y)
#sch[cached].vectorize(cached.op.axis[3])

import vnni
import numpy as np
with tvm.build_config(add_lower_pass= [(1, vnni.vnni_transformation)]):
    module = tvm.build(sch, [img, knl, conv], 'llvm -mcpu=cascadelake')
    print(tvm.lower(sch, [img, knl, conv], simple_mode=True))

    shapes = [i.shape for i in [img, knl]]
    shapes = [list(map(lambda x: x.value, i)) for i in shapes]
    out_shape = list(map(lambda x: x.value, conv.shape)) 
    types = ['int8', 'int8', 'int32']
    args = [tvm.ndarray.array(np.random.randint(0, 127, i, j)) for i, j in zip(shapes, types)]
    out = tvm.ndarray.array(np.zeros(out_shape).astype('int32'))
    ans = tvm.ndarray.array(np.zeros(out_shape).astype('int32'))

    #vannila(args[0], args[1], ans)
    module(args[0], args[1], out)
    module.save('nhwc.ll')
    #tvm.testing.assert_allclose(out.asnumpy(), ans.asnumpy())

    module = module.time_evaluator(module.entry_name, tvm.cpu(0), number=1)
    span = module(args[0], args[1], out).mean
    print('MACs: %d' % ops)
    print('%.3f s' % span)
    print('%.2f GVNNI/s' % (ops / span / 1e9))
