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
def rewrite(f, mod, ctx):
    is_init = [False]
    stmt = f.body

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
    # print(res)

    return res
