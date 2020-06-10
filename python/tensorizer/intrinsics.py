from tvm import te

def _vnni():
    """ Define the stencil of VNNI. """
    a = te.placeholder((64, ), dtype='int8', name='a')
    b = te.placeholder((64, ), dtype='int8', name='b')
    red = te.reduce_axis((0, 4), name='red')
    c = te.compute((16, ),
            lambda x: te.sum(a[x * 4 + red].astype('int32') * b[x * 4 + red].astype('int32'),
                             axis=red),
            name='c')
    return c.op
    sch = te.create_schedule(c.op)
    return sch, [a, b], tvm.driver.build_module.get_binds([a, b])[0]


INTRINSICS = {
  'vnni': {
    'pattern': _vnni(),
    'call': 'llvm.x86.avx512.vpdpbusd.512'
  },
}

