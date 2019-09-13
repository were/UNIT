import tvm
import topi
import numpy as np


n = k = m = 1024
a = tvm.placeholder((n, k), 'int8', name='a')
b = tvm.placeholder((m, k), 'int8', name='b')

packed_a = tvm.compute((n // 16, k // 4, 16, 4), lambda w, x, y, z: a[w * 16 + y, x * 4 + z], name='packed_a')
packed_b = tvm.compute((m // 16, k // 4, 16, 4), lambda w, x, y, z: b[w * 16 + y, x * 4 + z], name='packed_b')

red = tvm.reduce_axis((0, k), name='k')
#c = tvm.compute((n, m),
#        lambda x, y: tvm.sum(a[x, red].astype('int32') * packed_b[y // 16, red // 4, y % 16, red % 4].astype('int32'), axis=red),
#        name='c')

c = tvm.compute((n, m),
        lambda x, y: tvm.sum(packed_a[x // 16, red // 4, x % 16, red % 4].astype('int32') * packed_b[y // 16, red // 4, y % 16, red % 4].astype('int32'), axis=red),
        name='c')

sch = tvm.create_schedule(c.op)

sch[packed_b].vectorize(packed_b.op.axis[-1])
sch[packed_b].unroll(packed_b.op.axis[-2])

cc = sch.cache_write(c, 'global')

x, y = c.op.axis
xo, yo, xi, yi = sch[c].tile(x, y, 32, 16)
#sch[c].parallel(xo)
sch[cc].compute_at(sch[c], yo)

r = cc.op.reduce_axis[0]
ro, ri = sch[cc].split(r, 4)
xc, yc = cc.op.axis
xco, xci = sch[cc].split(xc, 16)
#yco, yci = sch[cc].split(yc, 16)
sch[cc].reorder(xco, ro, xci, yc, ri)
sch[cc].unroll(xci)
sch[cc].pragma(yc, 'vnni')

cached_a = sch.cache_read(packed_a, 'global', [cc])
sch[cached_a].vectorize(cached_a.op.axis[1])
sch[cached_a].compute_at(sch[cc], ro)
fused = sch[cached_a].fuse(cached_a.op.axis[2], cached_a.op.axis[3])
sch[cached_a].vectorize(fused)

print(tvm.lower(sch, [a, b, c], simple_mode=True))

import vnni
with tvm.build_config(add_lower_pass= [(1, vnni.vnni_transformation)]):

    print(tvm.lower(sch, [a, b, c], simple_mode=True))
    module = tvm.build(sch, [a, b, c], target='llvm -mcpu=cascadelake')

    np_a = np.random.randint(0, 64, (n, k), dtype='int8')
    np_b = np.random.randint(0, 64, (m, k), dtype='int8')
    np_c = np.dot(np_a.astype('int32'), np_b.astype('int32').T)

    nd_a = tvm.nd.array(np_a)
    nd_b = tvm.nd.array(np_b)
    nd_c = tvm.nd.array(np.zeros((n, m), dtype='int32'))

    module(nd_a, nd_b, nd_c)
    tvm.testing.assert_allclose(nd_c.asnumpy(), np_c)

    module.save('dense.ll')

    module = module.time_evaluator(module.entry_name, tvm.cpu(0), number=100)
    print('%.2f GVNNI/s' % (n * m * k / 64 / module(nd_a, nd_b, nd_c).mean / 1e9))
