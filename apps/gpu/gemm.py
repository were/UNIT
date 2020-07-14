import tvm
from tvm import te

a = te.placeholder((1000, 1000), 'float32')
b = te.placeholder((1000, 1000), 'float32')

rv = te.reduce_axis((0, 1000), )

c = te.compute((1000, 1000), lambda x, y: te.sum(a[x, rv] * b[y, rv], axis=[rv]))

sch = te.create_schedule(c.op)

bx = te.thread_axis('blockIdx.x')
tx = te.thread_axis('threadIdx.x')

sch[c].bind(c.op.axis[1], bx)
sch[c].bind(c.op.axis[0], tx)

ir = tvm.lower(sch, [a, b, c], simple_mode=True)

print(ir)

func = tvm.build(sch, [a, b, c], 'nvptx')

print(func.get_source())
