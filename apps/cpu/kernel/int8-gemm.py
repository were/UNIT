import tvm
import tensorizer
from tvm import te
import numpy as np
import topi

from tvm import relay

n, m, k = 128, 3072, 768

tiling = 128

a = te.placeholder((n, k), dtype='uint8')
b = te.placeholder((m // tiling, k // 4, tiling, 4), dtype='int8')

c = topi.x86.dense_dotprod(a, b, None, 'int32')

print(c.shape)

from tensorizer.intrinsics import INTRINSICS
from tensorizer.analyzer import analyze_tiling

sch = INTRINSICS['vnni']['schedule']([c])

#sch[c].parallel(sch[c].leaf_iter_vars[2])

with tvm.transform.PassContext(config={'tir.add_lower_pass': [(1, tensorizer.rewrite)]}, opt_level=3):
    ir = tvm.lower(sch, [a, b, c], simple_mode=True)
    module = tvm.build(sch, [a, b, c], target='llvm -mcpu=cascadelake')
    nda = tvm.nd.array((np.random.uniform(0, 1, (n, k)) * 32).astype('uint8'))
    ndb = tvm.nd.array((np.random.uniform(0, 1, (m // tiling, k // 4, tiling, 4)) * 32).astype('int8'))
    ndc = tvm.nd.array((np.random.uniform(0, 1, (n, m)) * 32).astype('int32'))
    ref = tvm.nd.array((np.random.uniform(0, 1, (n, m)) * 32).astype('int32'))
    timer = module.time_evaluator(module.entry_name, tvm.cpu(0), number=10, repeat=10)
    print(timer(nda, ndb, ndc).mean * 1e6)

# func = tvm.build(te.create_schedule(c.op), [a, b, c], target='llvm')
# timer = func.time_evaluator(func.entry_name, tvm.cpu(0))
# print(timer(nda, ndb, ref).mean)
#
# np.testing.assert_allclose(ndc.asnumpy(), ref.asnumpy())
