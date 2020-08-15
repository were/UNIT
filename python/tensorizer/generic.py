import tvm

def _gather_loop_trip_counts(stmt):
    axis_dom = []

    def visitor(op):
        nonlocal axis_dom
        if isinstance(op, tvm.tir.For):
            assert isinstance(op.min, tvm.tir.IntImm)
            assert isinstance(op.extent, tvm.tir.IntImm), op.extent
            axis_dom.append((op.loop_var, op.min.value, op.extent.value))
        if isinstance(op, tvm.tir.AttrStmt) and op.attr_key == 'thread_extent':
            assert isinstance(op.value, tvm.tir.IntImm)
            assert isinstance(op.node, tvm.tir.IterVar)
            axis_dom.append((op.node.var, 0, op.value.value))

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)

    return axis_dom

def _gather_memory_operations(stmt):
    loads = []
    store = []

    def visitor(op):
        if isinstance(op, tvm.tir.Load):
            loads.append(op)
        if isinstance(op, tvm.tir.Store):
            store.append(op)

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)
    assert len(store) == 1
    return loads, store[0]

def _gather_condition(stmt, axis):
    cond = []
    def visitor_gather_cond(op):
        if isinstance(op, tvm.tir.IfThenElse):
            cond.append(op.condition)
            assert op.else_case is None
    tvm.tir.stmt_functor.post_order_visit(stmt, visitor_gather_cond)

    if not cond:
        return cond

    pred = [None]
    def visitor_uses_var(op):
        if isinstance(op, tvm.tir.Var):
            for elem in axis:
                if tvm.tir.analysis.expr_deep_equal(elem[0], op):
                    pred[0] = op

    for i in cond:
        tvm.tir.stmt_functor.post_order_visit(i, visitor_uses_var)
    assert pred[0] is None, ("Predication not supported yet!", cond, pred[0])

    return cond

@tvm.tir.transform.prim_func_pass(opt_level=0)
def inject_sync(f, mod, ctx):

    def visitor(op):
        if isinstance(op, tvm.tir.AttrStmt):
            if op.attr_key == 'pragma_sync':
                res = [op.body, tvm.tir.Evaluate(tvm.tir.call_intrin('handle', 'tir.tvm_storage_sync', op.value.value))]
                return tvm.tir.SeqStmt(res)

        return None
    
    res = f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, None, visitor, ['tir.AttrStmt']))

    return res

@tvm.tir.transform.prim_func_pass(opt_level=0)
def loop_swizzle(f, mod, ctx):

    def visitor(op):
        if isinstance(op, tvm.tir.AttrStmt):
            if op.attr_key == 'pragma_swizzle':
                loop_body = op.body.body
                replace = {op.body.loop_var: (op.body.loop_var + 1) % op.body.extent}
                loop_body = tvm.tir.stmt_functor.substitute(loop_body, replace)
                return tvm.tir.For(op.body.loop_var, op.body.min, op.body.extent, 3, 0, loop_body)

        return None
    
    res = f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, None, visitor, ['tir.AttrStmt']))

    return res

@tvm.tir.transform.prim_func_pass(opt_level=0)
def sliding_window(f, mod, ctx):
    shift = []

    def detector(op):
        nonlocal shift
        if isinstance(op, tvm.tir.AttrStmt):
            if op.attr_key == 'pragma_sliding_window':
                print(op)
                if op.value.value == 'shift':
                    shift.append(op.body.loop_var)

        return None

    def visitor(op):
        nonlocal shift
        if isinstance(op, tvm.tir.AttrStmt):
            if op.attr_key == 'pragma_sliding_window':
                if op.value.value == 'rewrite':
                    threadIdx_x = tvm.te.thread_axis('threadIdx.x')
                    axis = shift[0]
                    ax0_var, ax0_ext = op.body.loop_var, op.body.extent - 1
                    ax1_var, ax1_ext = op.body.body.loop_var, op.body.body.extent
                    then_body = tvm.tir.stmt_functor.substitute(op.body.body, {ax0_var: threadIdx_x.var})
                    load_store = op.body.body.body
                    store_index = load_store.index
                    load_index = tvm.tir.stmt_functor.substitute(store_index, {ax0_var: ax0_var + 1})
                    loop_body = tvm.tir.Store(load_store.buffer_var, tvm.tir.Load(load_store.value.dtype, load_store.buffer_var, load_index), store_index)
                    loop_body = tvm.tir.For(ax1_var, tvm.tir.const(0, 'int32'), ax1_ext, 0, 0, loop_body)
                    loop_body = tvm.tir.stmt_functor.substitute(loop_body, {ax0_var: threadIdx_x.var})
                    loop_body = tvm.tir.IfThenElse(threadIdx_x.var < ax0_ext, loop_body, None)
                    postlog = tvm.tir.stmt_functor.substitute(op.body.body, {ax0_var: ax0_ext})
                    postlog = tvm.tir.stmt_functor.substitute(postlog, {ax1_var: threadIdx_x.var})
                    postlog = tvm.tir.IfThenElse(threadIdx_x.var < ax1_ext, postlog, None)
                    else_body = tvm.tir.SeqStmt([loop_body, postlog])
                    res = tvm.tir.IfThenElse(axis == tvm.tir.const(0, 'int32'), then_body, else_body)
                    res = tvm.tir.AttrStmt(threadIdx_x, 'thread_extent', tvm.tir.const(32, 'int32'), res)
                    return res
                else:
                    return op.body

        return None
    
    res = f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, detector, visitor, ['tir.AttrStmt']))
    print(res)

    return res

@tvm.tir.transform.prim_func_pass(opt_level=0)
def rewrite(f, mod, ctx):
    is_init = [False]
    stmt = f.body
    print(stmt)

    def detector(op):
        nonlocal is_init
        if isinstance(op, tvm.tir.For):
            is_init[0] = ('.init:') in str(op.loop_var)

    def visitor(op):
        nonlocal is_init
        if isinstance(op, tvm.tir.AttrStmt):
            if op.attr_key == 'pragma_tensorize':
                from .intrinsics import INTRINSICS
                loads, store = _gather_memory_operations(op)
                axis = _gather_loop_trip_counts(op)
                cond = _gather_condition(op, axis)

                if not is_init[0]:
                    encoded_operands = []
                    for i in zip(loads, INTRINSICS[op.value.value]['operands']):
                        tmp = i[1](i[0], axis)
                        if isinstance(tmp, (list, tuple)):
                            encoded_operands += list(tmp)
                        else:
                            encoded_operands.append(tmp)
                    xform = INTRINSICS[op.value.value]['write'](store, axis, encoded_operands)
                else:
                    is_init[0] = False
                    xform = INTRINSICS[op.value.value]['init'](store, axis)
                
                for elem in cond:
                    xform = tvm.tir.IfThenElse(elem, xform, None)
                return xform

        return None
    
    res = f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, detector, visitor, ['tir.For', 'tir.AttrStmt']))

    return res
