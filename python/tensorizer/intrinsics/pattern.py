from tvm import te

def vector_dotprod(out_lanes, reduce_lanes, a_dtype, b_dtype, out_dtype):
    """ Define the stencil of VNNI. """
    a = te.placeholder((reduce_lanes * out_lanes, ), dtype=a_dtype, name='a')
    b = te.placeholder((reduce_lanes * out_lanes, ), dtype=b_dtype, name='b')
    red = te.reduce_axis((0, reduce_lanes), name='red')
    c = te.compute((out_lanes, ),
            lambda x: te.sum(a[x * reduce_lanes + red].astype('int32') *
                             b[x * reduce_lanes + red].astype(out_dtype),
                             axis=red),
            name='c')
    return c.op

arm_sdot128_i8i16 = vector_dotprod(4, 4, 'int8', 'int8', 'int32')
x86_vnni = vector_dotprod(16, 4, 'uint8', 'int8', 'int32')

def mm_tensorcore(n=16, m=16, k=16, ta='n', tb='n', atype='float16', btype='float16', otype='float32'):
    a = te.placeholder((n, k) if ta == 'n' else (k, n), dtype=atype)
    b = te.placeholder((k, m) if tb == 'n' else (m, k), dtype=atype)
    rv = te.reduce_axis((0, k))

    def func(*args):
        x, y = args
        idxa = (x, rv) if ta == 'n' else (rv, x)
        idxb = (rv, y) if ta == 'n' else (y, rv)
        #return te.sum(a[x, rv].astype(otype) * b[rv, y].astype(otype),
        #              axis=[rv])

        return te.sum(a.__getitem__(idxa).astype(otype) * b.__getitem__(idxb).astype(otype),
                      axis=[rv])

    c = te.compute((n, m), func)

    return c.op

nv_tensorcore_m16n16k16_fp16fp32 = mm_tensorcore()