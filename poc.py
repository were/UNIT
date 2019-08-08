import tvm

n = 8192 #tvm.var('n')
a = tvm.placeholder((n, ))
b = tvm.placeholder((8, ))
c = tvm.compute((n, ), lambda x: a[x] * b[x % 8])

sch = tvm.create_schedule(c.op)
o, i = sch[c.op].split(c.op.axis[0], 16)
sch[c.op].vectorize(i)

func = tvm.lower(sch, [a, b])

a = tvm.build(sch, [a, b], target='c')

print(a.get_source())
