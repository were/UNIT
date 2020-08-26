import tvm
import tensorizer
import logging
import sys
from tvm import relay
from tvm import autotvm
import numpy as np
from topi.util import get_const_tuple

import topi
from tvm.relay import op

ic, h, w, oc, _, kh, kw, sh, sw = map(int, input().split())

if ic % 4:
    ic += 4 - ic % 4

if oc % 16:
    oc += 16 - oc % 16

a = tvm.te.placeholder((1, ic // 16, h, w, 16), dtype='int8')
if ic % 16 == 0:
    b = tvm.te.placeholder((oc // 16, ic // 16, kh, kw, 4, 16, 4), dtype='int8')
else:
    assert ic % 4 == 0
    a = tvm.te.placeholder((1, ic // 4, h, w, 4), dtype='int8')
    b = tvm.te.placeholder((oc // 16, ic // 4, kh, kw, 1, 16, 4), dtype='int8')


passes = [(1, tensorizer.rewrite)]
with tvm.transform.PassContext(opt_level=3, config={'tir.add_lower_pass': passes}), tvm.target.create('llvm -mcpu=cascadelake'):
    conv = topi.nn.conv2d_NCHWc_int8(a, b, stride=(sh, sw), padding=0, dilation=1, out_dtype='int32',
                                     layout='NCHW4c', out_layout='NCHW16c')
    sch = tensorizer.INTRINSICS['vnni']['schedule']([conv])

    module = tvm.build(sch, [a, b, conv], 'llvm -mcpu=cascadelake')
    np_a = np.zeros(get_const_tuple(a.shape), dtype='int8')
    np_b = np.zeros(get_const_tuple(b.shape), dtype='int8')
    np_c = np.zeros(get_const_tuple(conv.shape), dtype='int32')
    nd_a = tvm.nd.array(np_a, tvm.cpu())
    nd_b = tvm.nd.array(np_b, tvm.cpu())
    nd_c = tvm.nd.array(np_c, tvm.cpu())
    fte = module.time_evaluator(module.entry_name, ctx=tvm.cpu(), number=3, repeat=10)
    res = fte(nd_a, nd_b, nd_c).results
    print('exec: ', np.mean(res) * 1e6)