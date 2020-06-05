import tvm
import tensorizer
from tvm import te
import numpy as np

from tvm import relay

n, m, k = 1024, 1024, 1024

a = te.placeholder((n, k), dtype='int8')
b = te.placeholder((m, k), dtype='int8')

r = te.reduce_axis((0, k))
c = te.compute((n, m), lambda x, y: te.sum(a[x, r].astype('int32') * b[y, r].astype('int32'), axis=r))

info = tensorizer.analyze(c.op, tensorizer.vnni.pattern())
print(info)
sch = tensorizer.apply(c.op, info, 'vnni')
#x, y = c.op.axis
#yo, yi = sch[c].split(y, 16)
#ro, ri = sch[c].split(r, 4)
#sch[c].reorder(x, yo, ro, yi, ri)
#
#sch[c].pragma(yi, 'tensorize', 'vnni')
#
with tvm.target.build_config(add_lower_pass=[(1, tensorizer.rewrite)]):
    ir = tvm.lower(sch, [a, b, c], simple_mode=True)
    print(ir)
    module = tvm.build(sch, [a, b, c], target='llvm -mcpu=cascadelake')
    nda = tvm.nd.array((np.random.uniform(0, 1, (n, k)) * 32).astype('int8'))
    ndb = tvm.nd.array((np.random.uniform(0, 1, (m, k)) * 32).astype('int8'))
    ndc = tvm.nd.array((np.random.uniform(0, 1, (n, m)) * 32).astype('int32'))
    ref = tvm.nd.array((np.random.uniform(0, 1, (n, m)) * 32).astype('int32'))
    #print(nda, ndb)
    timer = module.time_evaluator(module.entry_name, tvm.cpu(0))
    print(timer(nda, ndb, ndc).mean)
    #print(ndc)
    #print(module.get_source())

func = tvm.build(te.create_schedule(c.op), [a, b, c], target='llvm')
timer = func.time_evaluator(func.entry_name, tvm.cpu(0))
print(timer(nda, ndb, ref).mean)

np.testing.assert_allclose(ndc.asnumpy(), ref.asnumpy())