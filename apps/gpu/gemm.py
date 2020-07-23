import tvm
from tvm import te
from tensorizer.intrinsics import INTRINSICS
import numpy as np

n, m, k = 128, 768, 3072

a = te.placeholder((n, k), 'float16')
b = te.placeholder((k, m), 'float16')

rv = te.reduce_axis((0, k), )

c = te.compute((n, m),
               lambda x, y: te.sum(a[x, rv].astype('float32') * b[rv, y].astype('float32'), axis=[rv]))

sch = tvm.te.create_schedule(c.op)
cc = sch.cache_write(c, 'wmma.accumulator')
x, y = sch[c].op.axis
xo, xi = sch[c].split(x, 16)
yo, yi = sch[c].split(y, 16)
sch[c].reorder(xo, yo, xi, yi)
sch[c].bind(xo, tvm.te.thread_axis('blockIdx.y'))
sch[c].bind(yo, tvm.te.thread_axis('blockIdx.x'))
sch[c].pragma(xi, 'tensorize', 'tensorcore')

sch[cc].compute_at(sch[c], yo)
r = sch[cc].op.reduce_axis[0]
cx, cy = sch[cc].op.axis
ro, ri = sch[cc].split(r, 16)
sch[cc].reorder(ro, cx, cy, ri)
sch[cc].pragma(cx, 'tensorize', 'tensorcore')

np_a = np.random.randn(n, k).astype('float16')
np_b = np.random.randn(k, m).astype('float16')
np_c = np.random.randn(n, m).astype('float32')

nd_a = tvm.nd.array(np_a, tvm.gpu())
nd_b = tvm.nd.array(np_b, tvm.gpu())
nd_c = tvm.nd.array(np_c, tvm.gpu())

import tensorizer

#with tvm.transform.PassContext(opt_level=3):
#    ir = tvm.lower(sch, [a, b, c], simple_mode=True)
#    print(ir)
#    quit()

with tvm.transform.PassContext(opt_level=3, config={'tir.add_lower_pass': [(1, tensorizer.rewrite)]}):
    ir = tvm.lower(sch, [a, b, c], simple_mode=True)
    print(ir)

    module = tvm.build(sch, [a, b, c], 'nvptx')
    module(nd_a, nd_b, nd_c)
    fte = module.time_evaluator(module.entry_name, ctx=tvm.gpu(), number=10)
    print('%.2f GFLOP/s' % (((n * m * k) / fte(nd_a, nd_b, nd_c).mean) / 1e9))
    #print(module.imported_modules[0].get_source())


ref = np_a.dot(np_b).astype('float32')

np.testing.assert_allclose(ref, nd_c.asnumpy(), atol=1e-3, rtol=1e-3)
