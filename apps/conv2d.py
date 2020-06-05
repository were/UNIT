import tvm
import tensorizer
import logging
import sys
from tvm import relay
from tvm import autotvm

import topi
from tvm.relay import op

x = relay.var('x', shape=(1, 3, 128, 128), dtype='int8')
w = relay.var('w', shape=(256, 3, 3, 3), dtype='int8')
b = relay.var('b', shape=(1, 256, 1, 1), dtype='int32')
conv2d = relay.nn.conv2d(x, w, out_dtype='int32', kernel_size=(3, 3), channels=256)
biased = relay.add(conv2d, b)
y = relay.multiply(biased, relay.const(11, 'int32'))

func = relay.Function([x, w, b], y)
module = tvm.IRModule()
module['main'] = func


def alter(attrs, inputs, tinfos, out_type):
    data, weight = inputs
    new_attrs = dict(attrs)
    new_attrs['data_layout'] = 'NHWC16c'
    new_attrs['kernel_layout'] = 'HWOI16o'

with tvm.target.build_config(add_lower_pass=[(1, tensorizer.rewrite)]), \
     relay.build_config(opt_level=3, disabled_pass=['AlterOpLayout']),  \
     tensorizer.AlterOpLayout('nn.conv2d', 'FTVMAlterOpLayout', alter):
    graph, mod, params = tvm.relay.build(module, target='llvm -mcpu=cascadelake')