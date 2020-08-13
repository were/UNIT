import functools
import operator

import tvm

from .pattern import mm_tensorcore

def noop():
    return tvm.tir.Evaluate(tvm.tir.const(0, 'int32'))

threadIdx_x = tvm.te.thread_axis('threadIdx.x')
threadIdx_y = tvm.te.thread_axis('threadIdx.y')
threadIdx_z = tvm.te.thread_axis('threadIdx.z')

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

def operand_loader(load, axis, dtype, role, reduce):
    res = []
    axis_ = axis[2+reduce:]
    if len(axis_) == 3 or len(axis_) == 2:
        axis_ = axis[2+reduce:]
        if reduce and len(axis_) == 3:
            r, _, r_ext = axis_[0]
            axis_ = axis[1:]
        else:
            r, _, r_ext = tvm.tir.const(0, 'int32'), None, tvm.tir.const(1, 'int32')
        y, _, y_ext = axis_[0]
        x, _, x_ext = axis_[1]
        if role == 'a':
            # x, r
            index = x * r_ext + r
        elif role == 'b':
            # r, y
            index = r * y_ext + y
        elif role == 'c':
            # x, y
            index = x * y_ext + y
    elif len(axis_) == 0:
        index = tvm.tir.const(0, 'int32')
    else:
        assert False
    for i in range(8):
        attr = tvm.tir.const(i, 'int32')
        val = tvm.tir.call_intrin(dtype, 'tir.tvm_struct_get', load.buffer_var, index, attr)
        res.append(val)
    return res

operand_a_fp16x2 = functools.partial(operand_loader, dtype='float16x2', role='a', reduce=True)
operand_b_fp16x2 = functools.partial(operand_loader, dtype='float16x2', role='b', reduce=True)
operand_c_fp32_compute = functools.partial(operand_loader, dtype='float32', role='c', reduce=True)
operand_c_fp32_store = functools.partial(operand_loader, dtype='float32', role='c', reduce=False)

def load_ab(load, axis, ab):
    addr = tvm.tir.stmt_functor.substitute(load, {i[0]: tvm.tir.const(0, 'int32') for i in axis[:2]})
    addr = tvm.tir.call_intrin('handle', 'tir.address_of', addr)
    coef = tvm.arith.detect_linear_equation(load.index, [i[0] for i in axis[:3]])
    stride = coef[1]
    intrin = f'llvm.nvvm.wmma.m16n16k16.load.{ab}.row.stride.f16.p0i32'
    return tvm.tir.call_llvm_intrin('handle', intrin, tvm.tir.const(2, 'int32'), addr, stride)

load_a = functools.partial(load_ab, ab='a')
load_b = functools.partial(load_ab, ab='b')

def store_fragment(store, axis, operands, dtype):
    assert len(operands) == 1
    if axis[2:]:
        axis_ = axis[2:]
        index = _flatten_index(axis_)
    else:
        index = tvm.tir.const(0, 'int32')
    operand = operands[0]
    item = tvm.tir.Var('item', 'handle')
    attr = tvm.tir.Var('attr', 'int32')
    val = tvm.tir.call_intrin(dtype, 'tir.tvm_struct_get', item, 0, attr)
    res = tvm.tir.call_intrin(dtype, 'tir.tvm_struct_set', store.buffer_var, index, attr, val)
    res = tvm.tir.Evaluate(res)
    res = tvm.tir.For(attr, tvm.tir.const(0, 'int32'), tvm.tir.const(8, 'int32'), 3, 0, res)
    res = tvm.tir.LetStmt(item, operand, res)
    res = _wrap_unroll(axis[2:], res)
    return tvm.tir.AttrStmt(threadIdx_x, 'thread_extent', tvm.tir.const(32, 'int32'), res)

store_ab = functools.partial(store_fragment, dtype='float16x2')

# i * stride + j -> stride
# (io * 4 + ii) * stride + j -> stride * 4

def writeback(store, axis, operands, dtype):

    res = tvm.tir.call_llvm_intrin('handle', 'llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32',
                                   tvm.tir.const(0, 'int32'), *(operands[8:] + operands[:8]))

    itm = tvm.tir.Var('intermediate', 'handle')
    if axis[3:]:
        axis_ = axis[3:]
        if len(axis_) == 3:
            y, _, y_ext = axis_[1]
            x, _, x_ext = axis_[2]
        elif len(axis_) == 2:
            y, _, y_ext = axis_[0]
            x, _, x_ext = axis_[1]
        else:
            assert False
        # x, y
        index = x * y_ext + y
    else:
        index = tvm.tir.const(0, 'int32')
    attr = tvm.tir.Var('attr', 'int32')
    val = tvm.tir.call_intrin(dtype, 'tir.tvm_struct_get', itm, tvm.tir.const(0, 'int32'), attr)
    set_val = tvm.tir.call_intrin(dtype, 'tir.tvm_struct_set', store.buffer_var, index, attr, val)
    stmt = tvm.tir.Evaluate(set_val)
    stmt = tvm.tir.For(attr, tvm.tir.const(0, 'int32'), tvm.tir.const(8, 'int32'), 3, 0, stmt)

    res = tvm.tir.LetStmt(itm, res, stmt)
    res = _wrap_unroll(axis[3:], res)

    return tvm.tir.AttrStmt(threadIdx_x, 'thread_extent', tvm.tir.const(32, 'int32'), res)

write_fp32 = functools.partial(writeback, dtype='float32')

def _wrap_unroll(axis, stmt):
    for elem in axis:
        idx, _, ext = elem
        stmt = tvm.tir.For(idx, tvm.tir.const(0, 'int32'), tvm.tir.const(ext, 'int32'), 0, 3, stmt)
    return stmt

def _flatten_index(axis):
    res = tvm.tir.const(0, 'int32')
    prod = tvm.tir.const(1, 'int32')
    for i, _, ext_ in axis:
        res = res + prod * i
        prod = prod * tvm.tir.const(ext_, 'int32')
    return res

def initializer(store, axis):

    idx = _flatten_index(axis[2:])
    attr = tvm.tir.Var('attr', 'int32')
    stmt = tvm.tir.call_intrin('handle', 'tir.tvm_struct_set', store.buffer_var, idx,
                               attr, tvm.tir.const(0, store.value.dtype))
    stmt = tvm.tir.Evaluate(stmt)
    stmt = tvm.tir.For(attr, tvm.tir.const(0, 'int32'), tvm.tir.const(8, 'int32'), 3, 0, stmt)

    res = _wrap_unroll(axis[2:], stmt)
    return res

def cleanup(store, axis, operands):
    coef = tvm.arith.detect_linear_equation(store.index, [i[0] for i in axis[:2]])
    ripper = {i[0]: tvm.tir.const(0, 'int32') for i in axis[:2]}
    addr = tvm.tir.call_intrin('handle', 'tir.address_of',
                               tvm.tir.Load(store.value.dtype, store.buffer_var,
                                            tvm.tir.stmt_functor.substitute(store.index, ripper),
                                            store.predicate))

    args = [addr]+ operands
    args.append(coef[1])
    res = tvm.tir.Evaluate(
        tvm.tir.call_llvm_intrin('handle', 'llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0f32',
                                 tvm.tir.const(10, 'int32'), *args))

    res = _wrap_unroll(axis[2:], res)
    res = tvm.tir.AttrStmt(threadIdx_x, 'thread_extent', tvm.tir.const(32, 'int32'), res)
    res = [res, tvm.tir.Evaluate(tvm.tir.call_llvm_intrin('handle', 'llvm.nvvm.barrier0', tvm.tir.const(0, 'int32')))]

    res = tvm.tir.SeqStmt(res)
    return res