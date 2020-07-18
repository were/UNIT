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

sch = INTRINSICS['tensorcore']['schedule']([c])

def tracer(module, info, is_before):
    import time
    global timing
    if bool(is_before):
        timing = time.time()
    else:
        print('Executes: ', info.name, (time.time() - timing) * 1000)

#np_a = np.random.randn(n, k).astype('float16')
#np_b = np.random.randn(m, k).astype('float16')
#np_c = np.random.randn(n, m).astype('float32')

np_a = np.ones((n, k)).astype('float16')
np_b = np.ones((k, m)).astype('float16')
np_c = np.ones((n, m)).astype('float32')

nd_a = tvm.nd.array(np_a, tvm.gpu())
nd_b = tvm.nd.array(np_b, tvm.gpu())
nd_c = tvm.nd.array(np_c, tvm.gpu())

print(nd_a, nd_a.dtype)

import tensorizer
from tensorizer.intrinsics.gpu import annotate_warpreduce
with tvm.transform.PassContext(opt_level=3, config={'tir.add_lower_pass': [(1, tensorizer.rewrite), (1, annotate_warpreduce)]}):
    module = tvm.build(sch, [a, b, c], 'nvptx')
    module(nd_a, nd_b, nd_c)
    fte = module.time_evaluator(module.entry_name, ctx=tvm.gpu(), number=10)
    print(fte(nd_a, nd_b, nd_c).mean * 1000)
    #print(module.imported_modules[0].get_source())
    #print(module.get_source())


ref = np_a.dot(np_b).astype('float32')

np.testing.assert_allclose(ref, nd_c.asnumpy(), atol=1e-5, rtol=1e-5)
