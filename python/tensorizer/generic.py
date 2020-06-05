import tvm

def unroller(stmt):
    axis_dom = []

    def visitor(op):
        nonlocal axis_dom
        if isinstance(op, tvm.tir.For):
            assert isinstance(op.min, tvm.tir.IntImm)
            assert isinstance(op.extent, tvm.tir.IntImm)
            axis_dom.append((op.loop_var, op.min.value, op.extent.value))

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)

    return axis_dom

def _iter_axis_dom(axis_dom):
    assignment = [m for _, m, _ in axis_dom]
    if assignment:
        while assignment[-1] != axis_dom[-1][2]:
            yield {i[0]: j for i, j in zip(axis_dom, assignment)}
            assignment[0] += 1
            for j in range(len(assignment) - 1):
                if assignment[j] == axis_dom[j][2]:
                    assignment[j] = 0
                    assignment[j + 1] += 1
                else:
                    break
    else:
        yield {}

def _gather_loop_levels(stmt):

    res = [[]]

    def visitor(op):
        if isinstance(op, tvm.tir.For):
            res[0].append(op)
    
    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)

    return res

def _vnni():
    from tvm import te
    """ Define the stencil of VNNI. """
    a = te.placeholder((64, ), dtype='int8', name='a')
    b = te.placeholder((64, ), dtype='int8', name='b')
    red = te.reduce_axis((0, 4), name='red')
    c = te.compute((16, ),
            lambda x: te.sum(a[x * 4 + red].astype('int32') * b[x * 4 + red].astype('int32'),
                             axis=red),
            name='c')
    sch = te.create_schedule(c.op)
    return tvm.lower(sch, [a, b], simple_mode=True)

PATTERN = {
    'vnni': _vnni
}

def _coalesce_memory(dtype, buffer_var, index, axis):
    coef = tvm.arith.detect_linear_equation(index, [i[0] for i in axis])
    assert list(coef)
    x = 0
    base_dict = {}
    while x < len(axis) and isinstance(coef[x], tvm.tir.IntImm) and coef[x].value == 0:
        base_dict[axis[x][0]] = tvm.tir.IntImm('int32', 0)
        x += 1
    base_dict[axis[x][0]] = tvm.tir.IntImm('int32', 0)
    ramps = []
    
    stride = coef[x]
    trips = axis[x][2]

    if isinstance(stride, tvm.tir.IntImm):
        y = x + 1
        while y < len(axis) and isinstance(coef[y], tvm.tir.IntImm) and coef[y].value == stride.value * trips:
            trips *= axis[y][2]
            base_dict[axis[y][0]] = tvm.tir.IntImm('int32', 0)
            y += 1
        if y == len(axis):
            base_index = tvm.tir.stmt_functor.substitute(index, base_dict)
            ramp = tvm.tir.Ramp(base_index, stride, trips)
            dtype = dtype + ('x%d' % trips)
            return [tvm.tir.Load(dtype, buffer_var, ramp)]

    dtype = dtype + ('x%d' % trips)

    # TODO(were): Ramp only
    for i in _iter_axis_dom(axis[y:]):
        m = base_dict.copy()
        m.update(i)
        base_index = tvm.tir.stmt_functor.substitute(index, m)
        ramp = tvm.tir.Ramp(base_index, stride, trips)
        ramp = tvm.arith.Analyzer().canonical_simplify(ramp)
        ramps.append(tvm.tir.Load(dtype, buffer_var, ramp))
    return ramps

def _parse_lanes(dtype):
    try:
        return int(dtype.split('x')[1])
    except:
        return 1

def prepare_operand(stmt):
    res = []
    axis = unroller(stmt)

    def visitor(op):
        if isinstance(op, tvm.tir.Load):
            ramps = _coalesce_memory(op.dtype, op.buffer_var, op.index, axis)
            total = len(ramps) * _parse_lanes(ramps[0].dtype)
            ramps = tvm.tir.Shuffle(ramps, list(range(total))) if len(ramps) != 1 else ramps[0]
            res.append(ramps)
        if isinstance(op, tvm.tir.Store):
            ramps = _coalesce_memory(op.value.dtype, op.buffer_var, op.index, axis)
            assert len(ramps) == 1
            res.append(ramps[0])

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)

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
                if not is_init[0]:
                    operands = prepare_operand(op)
                    buffer_var = operands[0].buffer_var
                    a, b, c = operands[0], operands[1], operands[2]
                    a = tvm.tir.call_pure_intrin('int32x16', 'reinterpret', a)
                    b = tvm.tir.call_pure_intrin('int32x16', 'reinterpret', b)
                    c = tvm.tir.call_pure_intrin('int32x16', 'reinterpret', c)
                    vnni = tvm.tir.call_llvm_intrin('int32x16', 'llvm.x86.avx512.vpdpbusd.512',
                                                    tvm.tir.const(0, 'uint32'), a, b, c)
                    return tvm.tir.Store(buffer_var, vnni, operands[0].index)
                else:
                    operands = prepare_operand(op)
                    value = operands[0]
                    return tvm.tir.Store(value.buffer_var, tvm.tir.const(0, value.dtype), value.index)
                    is_init[0] = False
        return None
    
    res = f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, detector, visitor, ['For', 'AttrStmt']))
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
