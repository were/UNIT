import tvm

n = 4096
k = 4096
m = 4096

a = tvm.placeholder((n, k), dtype='float16', name='a')
b = tvm.placeholder((m, k), dtype='float16', name='b')
red = tvm.reduce_axis((0, k))
c = tvm.compute((m, n),
        lambda x, y: tvm.sum(a[x, red].astype('float32') * b[y, red].astype('float32'), axis=red),
        name='c')

sch = tvm.create_schedule(c.op)

c_write = sch.cache_write(c, 'local')

x, y = c.op.axis

xo, xi = sch[c].split(x, 16)
xoo, xoi = sch[c].split(xo, 4)
yo, yi = sch[c].split(y, 16)
yoo, yoi = sch[c].split(yo, 4)

blcx = tvm.thread_axis('blockIdx.x')
thrx = tvm.thread_axis('threadIdx.x')

sch[c].bind(xoo, blcx)
sch[c].bind(yoo, thrx)

red_axis = c_write.op.reduce_axis[0]
ro, ri = sch[c_write].split(red, 16)
sch[c_write].reorder(ro, c_write.op.axis[0], c_write.op.axis[1], ri)

#ax0, ax1 = a_shared.op.axis
#ax1o, ax1i = sch[a_shared].split(ax1, 4)
#fused = sch[a_shared].fuse(ax0, ax1o)
#sch[a_shared].bind(fused, thrx)

sch[c].reorder(xoo, yoo, xoi, yoi, xi, yi)

#a_shared = sch.cache_read(a, 'shared', [c_write])
#sch[a_shared].compute_at(sch[c_write], ro)

sch[c_write].compute_at(sch[c], yoi)

b_shared = sch.cache_read(b, 'shared', [c_write])
sch[b_shared].compute_at(sch[c_write], ri)

def toucher(op):
    if isinstance(op, tvm.stmt.For):
        print(op.loop_var)
        print('a: ', tvm.arith.DomainTouched(op, a, True, True))
        print('b: ', tvm.arith.DomainTouched(op, b, True, True))
        print('c: ', tvm.arith.DomainTouched(op, c, True, True))
        print('c.local: ', tvm.arith.DomainTouched(op, c_write, True, True))

tvm.ir_pass.PostOrderVisit(tvm.build_module.form_body(sch), toucher)

ir = tvm.lower(sch, [a, b, c], simple_mode=True)
print(ir)

module = tvm.build(sch, [a, b, c], target='cuda')
print(module.imported_modules[0].get_source())

module.imported_modules[0].save('gemm.cu')

#import numpy as np
#
#nda = tvm.ndarray.array(np.random.randn(n, k).astype('float16'), tvm.gpu(0))
#ndb = tvm.ndarray.array(np.random.randn(m, k).astype('float16'), tvm.gpu(0))
#ndc = tvm.ndarray.array(np.zeros((n, m), dtype='float32'), tvm.gpu(0))

#timer = module.time_evaluator(module.entry_name, tvm.gpu(0), number=10)
#print(timer(nda, ndb, ndc).mean)
