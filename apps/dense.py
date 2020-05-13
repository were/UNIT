import tvm
from tvm import te

import tensorizer

n = k = m = 1024

a = te.placeholder((n // 16, k // 4, 16, 4), dtype='int8', name='a')
b = te.placeholder((m // 16, k // 4, 16, 4), dtype='int8', name='b')
red = te.reduce_axis((0, k))
c = te.compute((n, m), lambda x, y: te.sum(a[x // 16, red // 4, x % 16, red % 4].astype('int32') *
                                           b[y // 16, red // 4, y % 16, red % 4], axis=red))


tensorizer.vnni.pattern()
