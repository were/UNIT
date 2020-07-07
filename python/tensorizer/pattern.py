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