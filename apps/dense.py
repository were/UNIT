import tvm

n = k = m = 1024

a = tvm.placeholder((n // 16, k // 4, 16, 4), dtype='int8', name='a')
b = tvm.placeholder((m // 16, k // 4, 16, 4), dtype='int8', name='b')
red = tvm.reduce_axis((0, k))
c = tvm.compute((n, m), lambda x, y:
        tvm.sum(a[x // 16, red // 4, x % 16, red % 4].astype('int32') * b[y // 16, red // 4, y % 16, red % 4], axis=red))

import autotensorize

print(c.op.body)

vnni = autotensorize.pattern.vnni()
res = autotensorize.tensorize.preprocessor(c, vnni)

if res is not None:
    sch, outer, inner = res
else:
    quit()

print(tvm.build_module.form_body(sch))

module = tvm.build(sch, [a, b, c])
