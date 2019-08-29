import tvm
from . import util

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
            lst.append(op.dtype)
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

    assert len(stencil.op.axis) == len(stencil.shape)
    sch = tvm.create_schedule(a.op)
    inners = []

    dom_map = {}
    for i in list(a.op.axis) + list(a.op.reduce_axis):
        dom_map[i] = util.as_const_int(i.dom.extent)

    def update_split(map_, stage, axis, factor, is_stencil=False, inners=None):
        extent = map_[axis]
        map_.pop(axis)
        outer, inner = stage.split(axis, factor)
        map_[outer] = extent // factor
        # If this belongs to the stencil, it will be reordered as the inner most loops.
        if is_stencil:
            assert inners is not None
            inners.append(inner)
        else:
            map_[inner] = factor


    def split(axis, template):
        n, m = len(axis), len(template)
        for i in range(m):
            update_split(dom_map, sch[a], axis[i + n - m],
                    util.as_const_int(template[i].dom.extent), True, inners)

    split(a.op.axis, stencil.op.axis)
    split(a.op.reduce_axis, stencil.op.reduce_axis)

    to_tile = util.extract_tiling(tvm.build_module.form_body(sch), list(dom_map.keys()))

    for i in list(dom_map.keys()):
        if str(i.var) in to_tile.keys():
            update_split(dom_map, sch[a], i, to_tile[str(i.var)])

    sch[a].reorder(*(list(dom_map.keys()) + inners))

    return sch, dom_map, inners
