import tvm
import topi


image = tvm.placeholder((1, 3, 127, 127), dtype='int8', name='input')
kernel = tvm.placeholder((8, 3, 32, 32), dtype='int8', name='kernel')

conv = topi.nn.conv2d_nchw(image, kernel, stride=(1, 1), padding=0, dilation=(1, 1),
                           out_dtype='int32')

n, c, h, w = conv.op.axis
ro, rh, rw = conv.op.reduce_axis

print(ro, rh, rw)
print(n, c, h, w)

sch = tvm.create_schedule(conv.op)

rho, rhi = sch[conv.op].split(rh, 16)
rwo, rwi = sch[conv.op].split(rw, 16)

sch[conv.op].reorder(n, c, ro, h, w, rho, rwo, rhi, rwi)

ir = tvm.lower(sch, [image, kernel], simple_mode=True)
print(ir)

#module = tvm.build(sch, [image, kernel], target='c', name='generated')
#print(module.get_source())

module.save('generated.c')

