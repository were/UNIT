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


from tvm import autotvm

def define_space(comp, stencil, args, pragma):

    cfg = autotvm.get_config()
    sch = tvm.create_schedule(comp.op)

    def define_split(a, b, outers, inners):
        n, m = len(a), len(b)
        for i in range(n - m):
            cfg.define_split('split_%s' % a[i], a[i], num_outputs=3)
            axes = cfg['split_%s' % a[i]].apply(sch, comp, a[i])
            outers += list(axes)

        for i in range(m):
            cfg.define_split('split_%s' % a[i + n - m], a[i + n - m],
                policy='all', num_outputs=4,
                filter=lambda x: x.size[-1] == autotvm.util.get_const_int(b[i].dom.extent.value))
            axes = cfg['split_%s' % a[i + n - m]].apply(sch, comp, a[i + n - m])
            outers += list(axes)[:-1]
            inners += [axes[-1]]

    tensorized = []
    to_reorder = []
    define_split(comp.op.axis, stencil.op.axis, to_reorder, tensorized)
    define_split(comp.op.reduce_axis, stencil.op.reduce_axis, to_reorder, tensorized)

    cfg.define_reorder('reorder', to_reorder, policy='all')
    sch[comp].reorder(*([to_reorder[i] for i in cfg['reorder'].perm] + tensorized))

    sch[comp].pragma(tensorized[0], pragma)

    return sch, args


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
