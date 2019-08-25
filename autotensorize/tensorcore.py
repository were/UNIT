import tvm

# TODO(@were): support different transpose patterns
def pattern(itype='float16', otype='float32', trans_a=False, trans_b=False):
    a = tvm.placeholder((16, 16), dtype=itype, name='lhs')
    b = tvm.placeholder((16, 16), dtype=itype, name='rhs')
    red = tvm.reduce_axis((0, 16), 'red')
    c = tvm.placeholder((16, 16),
            lambda x, y: tvm.sum(a[x, red].astype(otype) * b[red, y].astype(otype)),
            name='tensorcore')
    return c
