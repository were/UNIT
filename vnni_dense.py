import tvm
import topi
import numpy as np


with tvm.target.create('llvm'):
    a = tvm.placeholder((4096, 4096), 'int8', name='a')
    b = tvm.placeholder((4096, 4096), 'int8', name='b')
    
    c = topi.nn.dense(a, b, out_dtype='int32')
    
    sch = tvm.create_schedule(c.op)

    y, x = c.op.axis
    r = c.op.reduce_axis[0]
    xo, xi = sch[c].split(x, 16)
    ro, ri = sch[c].split(r, 4)

    sch[c].reorder(y, xo, ro, xi, ri)

    print(tvm.lower(sch, [a, b, c], simple_mode=True))
