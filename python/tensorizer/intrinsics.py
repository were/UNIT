""" Tensorization code generation """
import functools

import tvm

def _index2ramps(index, axis, dtype=None):
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

    # coalesce dimension
    if isinstance(stride, tvm.tir.IntImm):
        y = x + 1
        while y < len(axis) and isinstance(coef[y], tvm.tir.IntImm) and coef[y].value == stride.value * trips:
            trips *= axis[y][2]
            base_dict[axis[y][0]] = tvm.tir.IntImm('int32', 0)
            y += 1
        if y == len(axis):
            base_index = tvm.tir.stmt_functor.substitute(index, base_dict)
            ramp = tvm.tir.Ramp(base_index, stride, trips)
            return [ramp]

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

    is_broadcast = True
    for i in range(y, len(axis)):
        if isinstance(coef[i], tvm.tir.IntImm) and coef[i].value == 0:
            pass
        else:
            is_broadcast = False

    cnt = 0
    # TODO(@were): Ramp only
    for i in _iter_axis_dom(axis[y:]):
        m = base_dict.copy()
        m.update(i)
        base_index = tvm.tir.stmt_functor.substitute(index, m)
        ramp = tvm.tir.Ramp(base_index, stride, trips)
        ramp = tvm.arith.Analyzer().canonical_simplify(ramp)
        ramps.append(ramp)
        cnt += 1

    if is_broadcast and dtype is not None:
        lanes = int(str(ramps[0].dtype).split('x')[1])
        for i in [8, 16, 32, 64]:
            if dtype.endswith(str(i)):
                bits = i
                dtype = dtype[:-len(str(i))]
                break
        total = bits * lanes
        if (total & -total) != total or total > 64:
            return ramps
        return [ramps[0], '%s%d' % (dtype, total), cnt]
    
    return ramps

def _load_concatenator(load, axis, cast_type=None):
    ramps = _index2ramps(load.index, axis, load.dtype)
    assert 'x' not in load.dtype
    loads = []
    total_lanes = 0
    is_broadcast = False
    if len(ramps) == 3 and isinstance(ramps[1], str) and isinstance(ramps[2], int):
        is_broadcast = True
        ri_cast = ramps[1]
        br_lanes = ramps[2]
        ramps = ramps[:1]
    for ramp in ramps:
        lanes = int(ramp.dtype.split('x')[1])
        dtype = load.dtype + 'x' + str(lanes)
        total_lanes += lanes
        loads.append(tvm.tir.Load(dtype, load.buffer_var, ramp))
    if is_broadcast:
        assert len(loads) == 1
        res = tvm.tir.call_pure_intrin(ri_cast, 'reinterpret', loads[0])
        res = tvm.tir.Broadcast(res, br_lanes)
    elif len(loads) == 1:
        res = loads[0]
    else:
        res = tvm.tir.Shuffle(loads, list(range(total_lanes)))
    if cast_type is not None:
        res = tvm.tir.call_pure_intrin(cast_type, 'reinterpret', res)
    return res

def _vnni_write(store, axis, operands):
    ramps = _index2ramps(store.index, axis)
    assert 'x' not in store.value.dtype
    assert len(ramps) == 1
    llvm_intrin = 'llvm.x86.avx512.vpdpbusd.512'
    vnni = tvm.tir.call_llvm_intrin('int32x16', llvm_intrin,
                                     tvm.tir.const(0, 'uint32'),
                                     *operands)
    return tvm.tir.Store(store.buffer_var, vnni, ramps[0])

def _vec_init(store, axis, dtype, lanes):
    ramps = _index2ramps(store.index, axis)
    assert 'x' not in store.value.dtype
    assert len(ramps) == 1
    return tvm.tir.Store(store.buffer_var, tvm.tir.const(0, '%sx%d' % (dtype, lanes)), ramps[0])

def _vdot_write(store, axis, operands):
    ramps = _index2ramps(store.index, axis)
    assert 'x' not in store.value.dtype
    assert len(ramps) == 1
    llvm_intrin = 'llvm.aarch64.neon.sdot.v4i32.v16i8'
    vdot = tvm.tir.call_llvm_intrin('int32x4', llvm_intrin,
                                     tvm.tir.const(3, 'uint32'),
                                     *operands)
    return tvm.tir.Store(store.buffer_var, vdot, ramps[0])

def _schedule_vdot(outs, pattern, pragma, max_threads):

    from topi.util import traverse_inline
    sch = tvm.te.create_schedule([i.op for i in outs])
    output = outs[0].op

    def callback(op):
        if len(list(op.reduce_axis)):
            from .analyzer import analyze_tiling
            points = list(analyze_tiling(op, pattern))
            fobj = lambda x: (2 ** -x[0]) * (2 ** -x[1]) * x[2] * (x[3] * x[3] if 2 <= x[3] <= 8 else 1.0 / x[3])
            points.sort(key=fobj)
            for x in points[::-1]:
                print((2 ** -x[0]), (2 ** -x[1]), x[2], (x[3] * x[3] if 2 <= x[3] <= 8 else 1.0 / x[3]))
                print(x[-1])
            to_apply = points[-1][-1]
            to_schedule = output
            loops = []
            parallel_level = None
            for i in range(len(output.axis)):

                if isinstance(to_apply[i][0], tuple) and to_apply[i][0][1] == 'parallel':
                    to_schedule = op
                    if str(op) != str(output):
                        outer, inner = sch[output].split(output.axis[i], nparts=to_apply[i][0][0])
                        parallel_level = outer
                        sch[op].compute_at(sch[output], outer)
                        if i == len(output.axis) - 1:
                            sch[output].vectorize(inner)
                        else:
                            sch[output].vectorize(output.axis[-1])

                to_append = []
                to_split = to_schedule.axis[i]

                for j in to_apply[i][1:][::-1]:
                    if isinstance(j, int):
                        outer, inner = sch[to_schedule].split(to_split, j)
                        to_split = outer
                    else:
                        outer, inner = sch[to_schedule].split(to_split, j[0])
                        to_split = outer

                    to_append = [inner] + to_append
                to_append = [to_split] + to_append
                loops += to_append

            for i in range(len(op.reduce_axis)):
                to_split = op.reduce_axis[i]
                to_append = []
                for j in to_apply[i + len(op.axis)][1:][::-1]:
                    if isinstance(j, int):
                        outer, inner = sch[op].split(to_split, j)
                        to_split = outer
                    else:
                        outer, inner = sch[op].split(to_split, j[0])
                        to_split = outer
                    to_append = [inner] + to_append
                to_append = [to_split] + to_append
                loops += to_append

            annot = []
            for i, elem in enumerate(to_apply):
                for j in elem:
                    if isinstance(j, int):
                        annot.append(None if i < len(op.axis) else 'reduce')
                    else:
                        annot.append(j[1])
            assert len(annot) == len(loops), '%d != %d' % (len(annot), len(loops))


            unroll, stencil, simple, reduction = [], [], [], []
            for i, elem in enumerate(zip(annot, loops)):
                # print(elem)
                hint, axis = elem
                if unroll and hint is None:
                    unroll.append(axis)
                elif hint == 'parallel':
                    fusion = sch[output].fuse(*(simple + [parallel_level if parallel_level is not None else axis]))
                    sch[output].parallel(fusion)
                    if str(op) != str(output):
                        sch[op].compute_at(sch[output], fusion)
                    simple = []
                elif hint == 'unroll':
                    unroll.append(axis)
                elif hint == 'offload':
                    stencil.append(axis)
                elif hint == 'reduction':
                    reduction.append(axis)
                else:
                    simple.append(axis)
            for i in unroll:
                sch[op].unroll(i)
            sch[op].pragma(stencil[0], 'tensorize', pragma)
            if str(op) != str(output):
                # print(simple, reduction, unroll, stencil, sep='\n')
                sch[op].reorder(*(simple + reduction + unroll + stencil))
            else:
                sch[op].reorder(*([fusion] + simple + reduction + unroll + stencil))

    traverse_inline(sch, output, callback)

    return sch

from .pattern import vector_dotprod

INTRINSICS = {
  'vnni': {
    'operands': [
        functools.partial(_load_concatenator, cast_type='int32x16'),
        functools.partial(_load_concatenator, cast_type='int32x16'),
        functools.partial(_load_concatenator, cast_type='int32x16')
    ],
    'write': _vnni_write,
    'init': functools.partial(_vec_init, dtype='int32', lanes=16),
    'schedule': functools.partial(_schedule_vdot,
                                  pattern=vector_dotprod(16, 4, 'uint8', 'int8', 'int32'),
                                  pragma='vnni', max_threads=10000)
  },
  'vdot': {
      'operands': [
          functools.partial(_load_concatenator, cast_type='int32x4'),
          functools.partial(_load_concatenator, cast_type='int8x16'),
          functools.partial(_load_concatenator, cast_type='int8x16')
      ],
      'write': _vdot_write,
      'init': functools.partial(_vec_init, dtype='int32', lanes=4),
      'schedule': functools.partial(_schedule_vdot,
                                    pattern=vector_dotprod(4, 4, 'int8', 'int8', 'int32'),
                                    pragma='vdot', max_threads=10000)
  }
}
