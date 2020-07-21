import tvm
from tvm import te
from tensorizer.intrinsics import INTRINSICS
import numpy as np

n, c, h, w = 1, 256, 130, 130
kh, kw, ic, ko = 3, 3, c, 256


a = te.placeholder((n, c, h, w), 'float16')
b = te.placeholder((kh, kw, ic, ko // 256, 16, 16), 'float16')

rc = te.reduce_axis((0, c), )
rh = te.reduce_axis((0, kh), )
rw = te.reduce_axis((0, kw), )

conv = te.compute((n, h - kh + 1, w - kw + 1, ko // 256, 16, 16),
               lambda batch, height, width, o_chunk, ob1, ob2:
                te.sum(a[batch, rc, height+rh, width+rw].astype('float32') *
                       b[rh, rw, rc, o_chunk, ob1, ob2].astype('float32'), axis=[rc, rh, rw]))

print('scheduling')
sch = INTRINSICS['tensorcore']['schedule']([conv])
print('scheduled')

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

np_a = np.ones((n, c, h, w)).astype('float16')
np_b = np.ones((n, ko, kh, kw)).astype('float16')
np_c = np.ones((n, ko, h - kh + 1, w - kw + 1)).astype('float32')

nd_a = tvm.nd.array(np_a, tvm.gpu())
nd_b = tvm.nd.array(np_b, tvm.gpu())
nd_c = tvm.nd.array(np_c, tvm.gpu())

import tensorizer
from tensorizer.intrinsics.gpu import annotate_warpreduce

with tvm.transform.PassContext(opt_level=3):
    ir = tvm.lower(sch, [a, b, conv], simple_mode=True)
    print(ir)
    quit()

print('here?!')
with tvm.transform.PassContext(opt_level=3, config={'tir.add_lower_pass': [(1, tensorizer.rewrite)]}):
    ir = tvm.lower(sch, [a, b, conv], simple_mode=True)
    print(ir)

    #module = tvm.build(sch, [a, b, c], 'nvptx')
    #module(nd_a, nd_b, nd_c)
    #fte = module.time_evaluator(module.entry_name, ctx=tvm.gpu(), number=10)
    #print('%.2f GFLOP/s' % (((n * ko * (h - kh + 1) * (w - kw + 1) * (kh * kw * ic) / fte(nd_a, nd_b, nd_c).mean) / 1e9)))
    #print(module.imported_modules[0].get_source())

#np.testing.assert_allclose(ref, nd_c.asnumpy(), atol=1e-5, rtol=1e-5)