import tvm
from tvm import te, arith
from tensorizer.intrinsics import INTRINSICS
import numpy as np
from topi.util import get_const_tuple

n, c, h, w = 1, 192, 18, 34
kh, kw, ic, ko = 3, 3, c, 192
stride_h = stride_w = 1

a = te.placeholder((n, c // 16, h, w, 16), 'float16')
b = te.placeholder((ko // 16, ic // 16, kh, kw, 16, 16), 'float16')

from tensorizer.ops.gpu import _conv2d_NCHW16c_OHWI16o_impl, _conv2d_schedule_wdim

conv = _conv2d_NCHW16c_OHWI16o_impl(a, b, stride_h, stride_w, 'float32')

sch = tvm.te.create_schedule(conv.op)

_conv2d_schedule_wdim(sch, conv, conv.op, stride_h, stride_w)


def tracer(module, info, is_before):
    import time
    global timing
    if bool(is_before):
        timing = time.time()
    else:
        print('Executes: ', info.name, (time.time() - timing) * 1000)

#np_a = np.random.randn(n, c // 16, h, w, 16).astype('float16')
#np_b = np.random.randn(ko // 16, ic // 16, kh, kw, 16, 16).astype('float16')
np_a = (np.arange(n * (c // 16) * h * w * 16) % 7).astype('float16')
np_b = (np.arange((ko // 16) * kh * kw * ic * 16) % 7).astype('float16')
np_a.shape = (n, c // 16, h, w, 16)
np_b.shape = (ko // 16, ic // 16, kh, kw, 16, 16)

np_c = np.random.randn(*get_const_tuple(conv.shape)).astype('float32')

nd_a = tvm.nd.array(np_a, tvm.gpu())
nd_b = tvm.nd.array(np_b, tvm.gpu())
nd_c = tvm.nd.array(np_c, tvm.gpu())

import tensorizer
passes = [(1, tensorizer.loop_swizzle), (1, tensorizer.rewrite), (1, tensorizer.inject_sync), (1, tensorizer.sliding_window)]
with tvm.transform.PassContext(opt_level=4, config={'tir.add_lower_pass': passes}):
#with tvm.transform.PassContext(opt_level=4):
    module = tvm.build(sch, [a, b, conv], 'nvptx')
    fte = module.time_evaluator(module.entry_name, ctx=tvm.gpu(), number=3, repeat=10)
    res = fte(nd_a, nd_b, nd_c).results
    print('exec: ', np.mean(res) * 1e6)

    import functools, operator
    elem_c = functools.reduce(operator.mul, np_c.shape, 1)
    coef_b = functools.reduce(operator.mul, [ic, kh, kw], 1)
    print(elem_c * coef_b / np.mean(res) / 1e9)

vanilla = tvm.te.create_schedule(conv.op)
print(*vanilla[conv].op.reduce_axis, sep='\n')
vb, vc, vx, vy, vob = vanilla[conv].op.axis
vrc, vrh, vrw = vanilla[conv].op.reduce_axis
vxo, vxi = vanilla[conv].split(vx, 32)
vyo, vyi = vanilla[conv].split(vy, 4)
fusion = vanilla[conv].fuse(vb, vc, vxo)
vanilla[conv].reorder(fusion, vxi, vyo, vrc, vrh, vrw, vyi, vob)
vanilla[conv].unroll(vyi)
vanilla[conv].vectorize(vob)
vanilla[conv].parallel(fusion)

#print(tvm.lower(vanilla, [a, b, conv], simple_mode=True))
vanilla = tvm.build(vanilla, [a, b, conv])
cpu_a = tvm.nd.array(np_a, tvm.cpu())
cpu_b = tvm.nd.array(np_b, tvm.cpu())
cpu_c = tvm.nd.array(np_c, tvm.cpu())
vanilla(cpu_a, cpu_b, cpu_c)

#res = cpu_c.asnumpy()
#ref = nd_c.asnumpy()
#for ax0 in range(n):
#    for ax1 in range(ko // 16):
#        for ax2 in range(h - kh + 1):
#            for ax3 in range(w - kw + 1):
#                for ax4 in range(16):
#                    assert abs(res[ax0, ax1, ax2, ax3, ax4] - ref[ax0, ax1, ax2, ax3, ax4]) < 1e-3, \
#                           (ax0, ax1, ax2, ax3, ax4, res[ax0, ax1, ax2, ax3, ax4], ref[ax0, ax1, ax2, ax3, ax4])
#
np.testing.assert_allclose(cpu_c.asnumpy(), nd_c.asnumpy(), atol=1e-3, rtol=1e-3)
print('correctness yes!')
