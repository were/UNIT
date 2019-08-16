import tvm

def test_gemm(a, b, c, sch):
    shapes = [(i.shape[0].value, i.shape[1].value) for i in [a, b, c]]
    import numpy as np
    np_a = np.random.randint(0, 63, shapes[0], 'int8')
    np_b = np.random.randint(0, 63, shapes[1], 'int8')
    np_c = np.zeros(shapes[2], 'int32')
    module = tvm.build(sch, [a, b, c])
    nds = [tvm.nd.array(i) for i in [np_a, np_b, np_c]]
    module(*nds)
    np_c = np.dot(np_a.astype('int32'), np_b.T.astype('int32'))
    tvm.testing.assert_allclose(nds[-1].asnumpy(), np_c)

a = tvm.placeholder((64, 64), 'int8', 'a')
b = tvm.placeholder((64, 64), 'int8', 'b')
red = tvm.reduce_axis((0, 64), 'r')
c = tvm.compute((64, 64),
        lambda x, y: tvm.sum(a[x, red].astype('int32') * b[y, red].astype('int32'), axis=red),
        'c')

sch = tvm.create_schedule(c.op)

test_gemm(a, b, c, sch)
