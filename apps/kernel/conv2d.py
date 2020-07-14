import tvm
import tensorizer
import logging
import sys
from tvm import relay
from tvm import autotvm

import topi
from tvm.relay import op

x = tvm.te.placeholder((1, 4, 128, 128), dtype='int8')
w = tvm.te.placeholder((256, 4, 3, 3), dtype='int8')
y = topi.nn.conv2d(x, w, strides=1, padding=0, dilation=1, out_dtype='int32')

sch = tensorizer.INTRINSICS['vnni']['schedule'](y.op)

with tvm.target.build_config(add_lower_pass=[(1, tensorizer.rewrite)]):
    ir = tvm.lower(sch, [x, w, y], simple_mode=True)
    print(ir)

#x = tvm.te.placeholder((1, 4, 128, 128, 128), dtype='int8')
#w = tvm.te.placeholder((256, 4, 3, 3, 3), dtype='int8')
#y = topi.nn.conv3d_ncdhw(x, w, stride=1, padding=0, dilation=1, out_dtype='int32')
#
#sch = tensorizer.INTRINSICS['vnni']['schedule'](y.op)
#
#with tvm.target.build_config(add_lower_pass=[(1, tensorizer.rewrite)]):
#    ir = tvm.lower(sch, [x, w, y], simple_mode=True)
#    print(ir)