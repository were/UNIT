import functools
import operator

import tvm

from .pattern import mm_tensorcore

def noop():
    return tvm.tir.Evaluate(tvm.tir.const(0, 'int32'))

def schedule(outs):
    c = outs[0]

    pattern = mm_tensorcore()

    sch = tvm.te.create_schedule(c.op)
    info = list(tvm.arith._ffi_api.MatchTensorizer(c.op, pattern))
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
    sch[cc].reorder(*(c_outers + list(cc.op.axis) + c_inners))

    sch[cc].pragma(cc.op.axis[0], 'tensorize', 'tensorcore')
    sch[c].pragma(o_inners[0], 'tensorize', 'tensorcore')

    blkx = tvm.te.thread_axis('blockIdx.x')
    #thrx = tvm.te.thread_axis('threadIdx.y')

    sch[c].reorder(*(o_outers + o_inners))

    fused = sch[c].fuse(o_outers[0], o_outers[1])
    sch[c].bind(fused, blkx)
    #sch[c].bind(o_outers[0], blkx)
    #sch[c].bind(o_outers[1], thrx)

    return sch

def loader(load, axis):
    return tvm.tir.call_intrin('handle', 'tir.address_of', load)

def writeback(store, axis, operands):

    ripper = {i[0]: tvm.tir.const(0, 'int32') for i in axis}

    coef = tvm.arith.detect_linear_equation(operands[1].args[0].index, [i[0] for i in axis])
    print(coef)
    aval = tvm.tir.call_llvm_intrin('handle', 'llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0i32',
        tvm.tir.const(2, 'int32'), tvm.tir.stmt_functor.substitute(operands[1], ripper), coef[2])
    avar = tvm.tir.Var('aval', 'handle')
    coef = tvm.arith.detect_linear_equation(operands[2].args[0].index, [i[0] for i in axis])
    print(coef)
    bval = tvm.tir.call_llvm_intrin('handle', 'llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0i32',
       tvm.tir.const(2, 'int32'), tvm.tir.stmt_functor.substitute(operands[2], ripper), coef[0])
    bvar = tvm.tir.Var('bval', 'handle')
    dtype = operands[0].args[0].dtype
    buffer_var = operands[0].args[0].buffer_var
    operands = []

    # TODO(@were): Only one 1 trip count is supported for now.
    for j in range(8):
        operands.append(tvm.tir.call_intrin('float16x2', 'tir.tvm_struct_get', avar, tvm.tir.const(0, 'int32'),
                                            tvm.tir.const(j, 'int32')))
    for j in range(8):
        operands.append(tvm.tir.call_intrin('float16x2', 'tir.tvm_struct_get', bvar, tvm.tir.const(0, 'int32'),
                                            tvm.tir.const(j, 'int32')))
    for j in range(8):
        operands.append(tvm.tir.call_intrin(dtype, 'tir.tvm_struct_get', buffer_var, tvm.tir.const(0, 'int32'),
                                            tvm.tir.const(j, 'int32')))

    res = tvm.tir.call_llvm_intrin('handle', 'llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32',
                                   tvm.tir.const(0, 'int32'), *operands)

    itm = tvm.tir.Var('intermediate', 'handle')
    stmts = []
    for i in range(functools.reduce(operator.mul, [i[2] for i in axis[1:]], 1) // 32 // 8):
        for j in range(8):
            val = tvm.tir.call_intrin(dtype, 'tir.tvm_struct_get', itm, tvm.tir.const(i, 'int32'), tvm.tir.const(j, 'int32'))
            set_val = tvm.tir.call_intrin(dtype, 'tir.tvm_struct_set', buffer_var, tvm.tir.const(i, 'int32'), tvm.tir.const(j, 'int32'), val)
            stmts.append(tvm.tir.Evaluate(set_val))

    seq = tvm.tir.SeqStmt(stmts)

    res = tvm.tir.LetStmt(itm, res, seq)
    res = tvm.tir.LetStmt(bvar, bval, res)
    res = tvm.tir.LetStmt(avar, aval, res)


    return res


def initializer(store, axis):
    stmts = []
    for j in range(8):
        struct_set = tvm.tir.call_intrin('handle', 'tir.tvm_struct_set', store.buffer_var, tvm.tir.const(0, 'int32'),
                                         tvm.tir.const(j, 'int32'), tvm.tir.const(0, store.value.dtype))

        stmts.append(tvm.tir.Evaluate(struct_set))
    return tvm.tir.SeqStmt(stmts)

def cleanup(store, loads, axis):
    coef = tvm.arith.detect_linear_equation(store.index, [i[0] for i in axis])
    ripper = {i[0]: tvm.tir.const(0, 'int32') for i in axis}
    addr = tvm.tir.call_intrin('handle', 'tir.address_of',
                               tvm.tir.Load(store.value.dtype, store.buffer_var,
                                            tvm.tir.stmt_functor.substitute(store.index, ripper),
                                            store.predicate))
    args = [addr]

    dtype = loads[0].dtype
    buffer_var = loads[0].buffer_var
    for j in range(8):
        args.append(tvm.tir.call_intrin(dtype, 'tir.tvm_struct_get', buffer_var, tvm.tir.const(0, 'int32'),
                                             tvm.tir.const(j, 'int32')))
    args.append(coef[1])
    return tvm.tir.Evaluate(
        tvm.tir.call_llvm_intrin('handle', 'llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0f32',
                                 tvm.tir.const(10, 'int32'), *args))


@tvm.tir.transform.prim_func_pass(opt_level=0)
def annotate_warpreduce(f, mod, ctx):

    def visitor(op):
        if isinstance(op, tvm.tir.AttrStmt):
            if op.attr_key == 'storage_scope' and op.value.value == 'wmma.accumulator':
                thrx = tvm.te.thread_axis('threadIdx.x')
                return tvm.tir.AttrStmt(thrx, 'thread_extent', tvm.tir.const(32, 'int32'), op)
        return None
    
    res = f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, None, visitor, ['tir.AttrStmt']))

    return res