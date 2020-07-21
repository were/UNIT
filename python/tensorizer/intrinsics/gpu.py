import functools
import operator

import tvm

from .pattern import mm_tensorcore

def noop():
    return tvm.tir.Evaluate(tvm.tir.const(0, 'int32'))

threadIdx_x = tvm.te.thread_axis('threadIdx.x')

def schedule(outs):
    c = outs[0]

    pattern = mm_tensorcore()

    sch = tvm.te.create_schedule(c.op)
    info = list(tvm.arith._ffi_api.MatchTensorizer(c.op, pattern))
    assert info

    o_outers, o_inners = [], []
    c_outers, c_inners = [], []

    cc = sch.cache_write(c, 'wmma.accumulator')

    for i, j in zip(info[0::2], info[1::2]):
        if i in c.op.axis:
            outer, inner = sch[c].split(i, j.dom.extent.value)
            o_outers.append(outer)
            o_inners.append(inner)
        else:
            idx = list(c.op.reduce_axis).index(i)
            outer, inner = sch[cc].split(cc.op.reduce_axis[idx], j.dom.extent.value)
            c_outers.append(outer)
            c_inners.append(inner)
    sch[cc].compute_at(sch[c], o_outers[-1])

    outer, inner = sch[cc].split(c_outers[0], 4)
    c_outers = [outer, inner] + c_outers[1:]

    #aa = sch.cache_read(c.op.input_tensors[0], 'shared', [cc])
    #sch[aa].compute_at(sch[cc], outer)
    #outer, inner = sch[aa].split(aa.op.axis[0], nparts=8)
    #sch[aa].bind(outer, tvm.te.thread_axis('threadIdx.y'))
    #outer, inner = sch[aa].split(aa.op.axis[1], nparts=32)
    #sch[aa].bind(outer, tvm.te.thread_axis('threadIdx.x'))

    cc_axis = list(cc.op.axis)

    sch[cc].reorder(*(c_outers + cc_axis + c_inners))
    sch[cc].pragma(cc_axis[0], 'tensorize', 'tensorcore')

    blkx = tvm.te.thread_axis('blockIdx.x')

    sch[c].reorder(*(o_outers + o_inners))

    sch[c].bind(o_outers[1], blkx)
    #sch[c].bind(o_outers[0], tvm.te.thread_axis('threadIdx.y'))
    fused = sch[c].fuse(o_outers[0], o_outers[1])
    sch[c].bind(fused, blkx)
    sch[c].pragma(o_inners[0], 'tensorize', 'tensorcore')

    return sch

def loader(load, axis):
    return tvm.tir.call_intrin('handle', 'tir.address_of', load)

def writeback(store, axis, operands):

    ripper = {i[0]: tvm.tir.const(0, 'int32') for i in axis[:3]}

    coef = tvm.arith.detect_linear_equation(operands[1].args[0].index, [i[0] for i in axis[:3]])
    aval = tvm.tir.call_llvm_intrin('handle', 'llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0i32',
        tvm.tir.const(2, 'int32'), tvm.tir.stmt_functor.substitute(operands[1], ripper), coef[2])
    avar = tvm.tir.Var('aval', 'handle')
    coef = tvm.arith.detect_linear_equation(operands[2].args[0].index, [i[0] for i in axis[:3]])
    bval = tvm.tir.call_llvm_intrin('handle', 'llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0i32',
       tvm.tir.const(2, 'int32'), tvm.tir.stmt_functor.substitute(operands[2], ripper), coef[0])
    bvar = tvm.tir.Var('bval', 'handle')
    dtype = operands[0].args[0].dtype
    buffer_var = operands[0].args[0].buffer_var
    operands = []

    idx = _flatten_index(axis[3:])

    # TODO(@were): Only one 1 trip count is supported for now.
    for j in range(8):
        operands.append(tvm.tir.call_intrin('float16x2', 'tir.tvm_struct_get', avar, idx,
                                            tvm.tir.const(j, 'int32')))
    for j in range(8):
        operands.append(tvm.tir.call_intrin('float16x2', 'tir.tvm_struct_get', bvar, idx,
                                            tvm.tir.const(j, 'int32')))
    for j in range(8):
        operands.append(tvm.tir.call_intrin(dtype, 'tir.tvm_struct_get', buffer_var, idx,
                                            tvm.tir.const(j, 'int32')))

    res = tvm.tir.call_llvm_intrin('handle', 'llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32',
                                   tvm.tir.const(0, 'int32'), *operands)

    itm = tvm.tir.Var('intermediate', 'handle')
    stmts = []
    for j in range(8):
        val = tvm.tir.call_intrin(dtype, 'tir.tvm_struct_get', itm, tvm.tir.const(0, 'int32'), tvm.tir.const(j, 'int32'))
        set_val = tvm.tir.call_intrin(dtype, 'tir.tvm_struct_set', buffer_var, tvm.tir.const(0, 'int32'), tvm.tir.const(j, 'int32'), val)
        stmts.append(tvm.tir.Evaluate(set_val))

    seq = tvm.tir.SeqStmt(stmts)

    res = tvm.tir.LetStmt(itm, res, seq)
    res = tvm.tir.LetStmt(bvar, bval, res)
    res = tvm.tir.LetStmt(avar, aval, res)
    res = _wrap_unroll(axis[3:], res)

    return tvm.tir.AttrStmt(threadIdx_x, 'thread_extent', tvm.tir.const(32, 'int32'), res)


def _wrap_unroll(axis, stmt):
    for i, min_, ext_ in axis:
        stmt = tvm.tir.For(i, tvm.tir.const(min_, 'int32'), tvm.tir.const(ext_, 'int32'), 3, 0, stmt)
    return stmt

def _flatten_index(axis):
    res = tvm.tir.const(0, 'int32')
    prod = tvm.tir.const(1, 'int32')
    for i, _, ext_ in axis:
        res = res + prod * i
        prod = prod * tvm.tir.const(ext_, 'int32')
    return res

def initializer(store, axis):
    stmts = []

    idx = _flatten_index(axis[2:])
    for j in range(8):
        struct_set = tvm.tir.call_intrin('handle', 'tir.tvm_struct_set', store.buffer_var, idx,
                                         tvm.tir.const(j, 'int32'), tvm.tir.const(0, store.value.dtype))
        stmts.append(tvm.tir.Evaluate(struct_set))

    res = _wrap_unroll(axis[2:], tvm.tir.SeqStmt(stmts))
    return res

def cleanup(store, loads, axis):
    coef = tvm.arith.detect_linear_equation(store.index, [i[0] for i in axis[:2]])
    ripper = {i[0]: tvm.tir.const(0, 'int32') for i in axis}
    addr = tvm.tir.call_intrin('handle', 'tir.address_of',
                               tvm.tir.Load(store.value.dtype, store.buffer_var,
                                            tvm.tir.stmt_functor.substitute(store.index, ripper),
                                            store.predicate))

    args = [addr]

    idx = _flatten_index(axis[2:])
    dtype = loads[0].dtype
    buffer_var = loads[0].buffer_var
    for j in range(8):
        args.append(tvm.tir.call_intrin(dtype, 'tir.tvm_struct_get', buffer_var, idx,
                                             tvm.tir.const(j, 'int32')))
    args.append(coef[1])
    res = tvm.tir.Evaluate(
        tvm.tir.call_llvm_intrin('handle', 'llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0f32',
                                 tvm.tir.const(10, 'int32'), *args))

    res = _wrap_unroll(axis[2:], res)

    return tvm.tir.AttrStmt(threadIdx_x, 'thread_extent', tvm.tir.const(32, 'int32'), res)
