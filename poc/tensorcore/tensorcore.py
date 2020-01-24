import tvm

n = 16
k = 16
m = 16

a = tvm.placeholder((n, k), dtype='float16', name='a')
b = tvm.placeholder((m, k), dtype='float16', name='b')
red = tvm.reduce_axis((0, k))
c = tvm.compute((m, n),
        lambda x, y: tvm.sum(a[x, red].astype('float32') * b[y, red].astype('float32'), axis=red),
        name='c')

sch = tvm.create_schedule(c.op)

x, y = c.op.axis

xo, xi = sch[c].split(x, 16)
yo, yi = sch[c].split(y, 16)

blcx = tvm.thread_axis('blockIdx.x')
thrx = tvm.thread_axis('threadIdx.x')

yio, yii = sch[c].split(yi, 8)
sch[c].reorder(xo, yo, xi, yio, yii, red)
xy = sch[c].fuse(xi, yio)

print(tvm.lower(sch, [a, b, c], simple_mode=True))
