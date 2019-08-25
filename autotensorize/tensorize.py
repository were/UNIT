import tvm

def _orders(body):
    first, post = [], []

    import functools

    def visit(lst, op):
        if isinstance(op, tvm.expr.Call):
            lst.append(op.dtype)
            return op
        if isinstance(op, tvm.expr.Cast):
            lst.append(op.dtype)
            lst.append(op.value.dtype)
        else:
            lst.append(type(op))
        return None

    first_visit = functools.partial(visit, first)
    post_visit = functools.partial(visit, post)

    tvm.ir_pass.IRTransform(body, first_visit, post_visit,
            ['Add', 'Sub', 'Div', 'Mul', 'Cast', 'Call', 'Reduce'])

    return first, post

def tensorizable(a, stencil):
    assert isinstance(a.op, tvm.tensor.ComputeOp)
    assert isinstance(stencil.op, tvm.tensor.ComputeOp)
    assert len(a.op.body) == len(stencil.op.body)
    n = len(a.op.body)

    for i in range(n):
        src0, src1 = _orders(tvm.make.Evaluate(a.op.body[i]))
        tgt0, tgt1 = _orders(tvm.make.Evaluate(stencil.op.body[i]))
        if src0 != tgt0 or src1 != tgt1:
            return False

    return True

def preprocessor(a, stencil):
    """ If compute node `a` can be tensorized, return the tuple of
        (schedule, outer loops, inner loops). O.w. return None. """

    if not tensorizable(a, stencil):
        return None

    sch = tvm.create_schedule(a.op)
    outers = []
    inners = []

    assert len(stencil.op.axis) == len(stencil.shape)

    def split(axis, template):
        n, m = len(axis), len(template)
        for i in range(n - m):
            outers.append(axis[i])
        for i in range(m):
            assert isinstance(template[i].dom.extent, (tvm.expr.IntImm, tvm.expr.UIntImm))
            outer, inner = sch[a].split(axis[i + n - m], template[i].dom.extent.value)
            outers.append(outer)
            inners.append(inner)

    split(a.op.axis, stencil.op.axis)
    split(a.op.reduce_axis, stencil.op.reduce_axis)

    sch[a].reorder(*(outers + inners))

    return sch, outers, inners
