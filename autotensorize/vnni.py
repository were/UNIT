import tvm
from . import util

def customized_pass(stmt):

    to_vectorize = []
    outer_loops = []

    def add_loop_level(op):
        if isinstance(op, tvm.stmt.AttrStmt):
            if op.attr_key == 'pragma_vnni':
                to_vectorize.append(op.node.var)
        elif isinstance(op, tvm.stmt.For):
            outer_loops.append(op)
        return None


    def vectorize(op):
        if isinstance(op, tvm.stmt.For):
            outer_loops.pop()
            if to_vectorize:
                if str(op.loop_var) == f'{to_vectorize[-1]}.init':
                    return tvm.make.For(op.loop_var, op.min, op.extent, tvm.stmt.For.Vectorized,
                                        op.device_api, op.body)
                elif str(op.loop_var) == str(to_vectorize[-1]):
                    loops = []
                    loads = []
                    store = [None]
                    guard = [None]
                    def get_loops(op):
                        if isinstance(op, tvm.stmt.For):
                            loops.append(op)
                        elif isinstance(op, tvm.expr.Load):
                            loads.append(op)
                        elif isinstance(op, tvm.stmt.Store):
                            assert store[0] is None
                            store[0] = op
                        elif isinstance(op, tvm.stmt.IfThenElse):
                            guard[0] = op

                    tvm.ir_pass.PostOrderVisit(op, get_loops)
                    inner, outer = loops
                    loops = loops[::-1]

                    inner_ext = util.as_const_int(inner.extent)
                    outer_ext = util.as_const_int(outer.extent)
                    assert inner_ext is not None and outer_ext is not None
                    assert outer_ext ==16 and inner_ext == 4

                    empty = {outer.loop_var: tvm.const(0, 'int32'),
                             inner.loop_var: tvm.const(0, 'int32')}

                    operands = []
                    indeces = []
                    for elem in loads:
                        iters = [i.loop_var for i in outer_loops + loops]
                        coef = tvm.arith.DetectLinearEquation(elem.index, iters)
                        base_index = sum(i * j for i, j in zip(iters[:-2], coef)) + coef[-1]
                        inner_stride = util.as_const_int(coef[-2])
                        outer_stride = util.as_const_int(coef[-3])
                        assert inner_stride is not None and outer_stride is not None

                        if tvm.ir_pass.Equal(elem.buffer_var, store[0].buffer_var):
                            index = tvm.make.Ramp(base_index, tvm.const(1, 'int32'), 16)
                            continue

                        indeces = []
                        for i in range(outer_ext):
                            for j in range(inner_ext):
                                indeces.append(i * outer_stride + j * inner_stride)
                        bound = max(indeces) + 1
                        to_load = tvm.make.Ramp(base_index, tvm.const(1, 'int32'), bound)
                        value = tvm.make.Load(elem.dtype + 'x%d' % bound, elem.buffer_var, to_load,
                                              tvm.const(1, 'int32x%d' % bound))
                        assert 64 % bound == 0

                        operands.append(tvm.make.Shuffle([value] * (64 // bound), [tvm.const(i, 'int32') for i in indeces]))

                    buffer_var = store[0].buffer_var

                    operands = [tvm.make.Load('int32x16', buffer_var, index, tvm.const(1, 'int32x16'))] + operands

                    operands = [tvm.call_pure_intrin('int32x16', 'reinterpret', i) for i in operands]

                    res = tvm.call_llvm_intrin('int32x16', 'llvm.x86.avx512.vpdpbusd.512',
                                               tvm.const(0, 'uint32'),
                                               *operands)

                    res = tvm.make.Store(buffer_var, res, index, tvm.const(1, 'int32x16'))
                    if guard[0] is not None:
                        res = tvm.make.IfThenElse(guard[0].condition, res, None)
                    return res
        elif isinstance(op, tvm.stmt.AttrStmt):
            if not to_vectorize:
                return None
            if tvm.ir_pass.Equal(op.node.var, to_vectorize[-1]):
                to_vectorize.pop()
                return op.body
        return None

    return tvm.ir_pass.IRTransform(stmt, add_loop_level, vectorize, ['AttrStmt', 'For'])


def pattern():
    """ Define the stencil of VNNI. """
    a = tvm.placeholder((64, ), dtype='int8', name='a')
    b = tvm.placeholder((64, ), dtype='int8', name='b')
    red = tvm.reduce_axis((0, 4), name='red')
    c = tvm.compute((16, ),
            lambda x: tvm.sum(a[x * 4 + red].astype('int32') * b[x * 4 + red].astype('int32'),
                              axis=red),
            name='b')
    return c