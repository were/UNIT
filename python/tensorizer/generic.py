import tvm

def _gather_loop_trip_counts(stmt):
    axis_dom = []

    def visitor(op):
        nonlocal axis_dom
        if isinstance(op, tvm.tir.For):
            assert isinstance(op.min, tvm.tir.IntImm)
            assert isinstance(op.extent, tvm.tir.IntImm)
            axis_dom.append((op.loop_var, op.min.value, op.extent.value))

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

@tvm.tir.transform.prim_func_pass(opt_level=0)
def rewrite(f, mod, ctx):
    is_init = [False]
    stmt = f.body
    #print(stmt)

    def detector(op):
        nonlocal is_init
        if isinstance(op, tvm.tir.For):
            is_init[0] = ('.init:') in str(op.loop_var)

    def visitor(op):
        nonlocal is_init
        if isinstance(op, tvm.tir.AttrStmt):
            if op.attr_key == 'pragma_tensorize':
                if not is_init[0]:
                    from .intrinsics import INTRINSICS
                    loads, store = _gather_memory_operations(op)
                    axis = _gather_loop_trip_counts(op)
                    encoded_operands = []
                    for i in zip(loads, INTRINSICS[op.value.value]['operands']):
                        encoded_operands.append(i[1](i[0], axis))
                    return INTRINSICS[op.value.value]['write'](store, axis, encoded_operands)
                else:
                    from .intrinsics import INTRINSICS
                    loads, store = _gather_memory_operations(op)
                    axis = _gather_loop_trip_counts(op)
                    return INTRINSICS[op.value.value]['init'](store, axis)

                    is_init[0] = False
        return None
    
    res = f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, detector, visitor, ['For', 'AttrStmt']))
    print(res)
    return res

def analyze(op, stencil):
    info = list(tvm.arith._ffi_api.MatchTensorizer(op, stencil))
    res = {}
    for i, j in zip(info[::2], info[1::2]):
        res[i] = j
    return res

def apply(op, loops, pragma):
    sch = tvm.te.create_schedule(op)

    axis = list(op.axis)
    reduce_axis = list(op.reduce_axis)
    inners = []
    dom = {}

    for i in axis:
        dom[i] = i.dom.extent.value
    for i in reduce_axis:
        dom[i] = i.dom.extent.value

    def process(axis):
        for i in range(len(axis)):
            if axis[i] in loops.keys():
                outer, inner = sch[op].split(axis[i], loops[axis[i]].dom.extent.value)
                inners.append(inner)
                axis[i] = outer

    process(axis)
    process(reduce_axis)

    sch[op].reorder(*(axis + reduce_axis + inners))
    sch[op].pragma(inners[0], 'tensorize', pragma)

    return sch
