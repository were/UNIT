from tvm import te
import tvm
import functools
import operator

def dots(out_lanes, reduce_lanes, a_dtype, b_dtype, out_dtype):
    """ Define the stencil of VNNI. """
    a = te.placeholder((reduce_lanes * out_lanes, ), dtype=a_dtype, name='a')
    b = te.placeholder((reduce_lanes * out_lanes, ), dtype=b_dtype, name='b')
    red = te.reduce_axis((0, reduce_lanes), name='red')
    c = te.compute((out_lanes, ),
            lambda x: te.sum(a[x * reduce_lanes + red].astype('int32') *
                             b[x * reduce_lanes + red].astype(out_dtype),
                             axis=red),
            name='c')
    return c.op

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
            print('coal')
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
    print('analyzing', load)
    ramps = _index2ramps(load.index, axis, load.dtype)
    assert 'x' not in load.dtype
    loads = []
    total_lanes = 0
    print('#ramps:', len(ramps))
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
        print(len(loads))
        print(total_lanes)
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
    print(vdot)
    return tvm.tir.Store(store.buffer_var, vdot, ramps[0])

def _schedule_vdot(outs, pattern, pragma, max_threads):

    from topi.util import traverse_inline
    sch = tvm.te.create_schedule([i.op for i in outs])
    output = outs[0].op

    def callback(op):
        if len(list(op.reduce_axis)):
            info = list(tvm.arith._ffi_api.MatchTensorizer(op, pattern))
            assert info, op
            loops = {}
            for i, j in zip(info[::2], info[1::2]):
                loops[i] = j

            axis = list(op.axis)
            reduce_axis = list(op.reduce_axis)
            inners = []
            dom = {}

            o_axis = list(output.axis)

            for i in axis:
                dom[i] = i.dom.extent.value
            for i in reduce_axis:
                dom[i] = i.dom.extent.value

            def process(axis, is_reduce):
                is_firstsplit = True
                for i in range(len(axis)):
                    if axis[i] in loops.keys():
                        outer, inner = sch[op].split(axis[i], loops[axis[i]].dom.extent.value)
                        inners.append(inner)
                        dom[inner] = axis[i].dom.extent.value
                        dom[outer] = axis[i].dom.extent.value // dom[inner]
                        dom.pop(axis[i])
                        axis[i] = outer

                        if is_firstsplit and not is_reduce:
                            is_firstsplit = False
                            prod = 1
                            for j in range(i - 1, 0, -1):
                                prod *= o_axis[j].dom.extent.value
                                print(prod, o_axis[j])
                                if prod > 1:
                                    tiled = None
                                    for k in range(8, 2, -1):
                                        if tiled == None or \
                                           o_axis[j].dom.extent.value % k < o_axis[j].dom.extent.value % tiled:
                                            tiled = k
                                    k = tiled
                                    oj_outer_o, oj_outer_i = sch[output].split(o_axis[j], k)
                                    sch[op].compute_at(sch[output], oj_outer_o)

                                    #fused = sch[output].fuse(*(o_axis[:j]))
                                    #sch[output].parallel(fused)

                                    outer_prod = 1
                                    to_fuse = []
                                    for k in range(j):
                                        if outer_prod * o_axis[k].dom.extent.value <= max_threads:
                                            outer_prod *= o_axis[k].dom.extent.value
                                            to_fuse.append(o_axis[k])
                                            print('fuse: ', o_axis[k])
                                        else:
                                            if outer_prod * 2 > max_threads:
                                                break
                                            factor, ext = 2, o_axis[k].dom.extent.value
                                            tiling = None
                                            while factor < ext:
                                                if outer_prod * (ext // factor) <= max_threads:
                                                    if (tiling == None) or (ext % factor < ext % tiling):
                                                        tiling = factor
                                                factor += 1
                                            oo, oi = sch[output].split(o_axis[k], tiling)
                                            to_fuse.append(oo)
                                            print('split: ', o_axis[k], 'by ', tiling)
                                            break

                                    fused = sch[output].fuse(*(to_fuse))
                                    sch[output].parallel(fused)
                                    sch[output].vectorize(o_axis[-1])

                                    #for k in range(j, 0, -1):
                                    #    if functools.reduce(operator.mul, o_doms[:k]) <= max_threads:
                                    #        fused = sch[output].fuse(*(o_axis[:k]))
                                    #        sch[output].parallel(fused)
                                    #        break

                                    break

            process(axis, False)
            process(reduce_axis, True)

            print('reorder:')
            print(axis[:-2])
            print(reduce_axis)
            print(axis[-2:])
            print(inners)
            print(op.body)

            sch[op].reorder(*(axis[:-2] + reduce_axis + axis[-2:] + inners))
            sch[op].unroll(axis[-1])
            sch[op].unroll(axis[-2])
            sch[op].pragma(inners[0], 'tensorize', pragma)

    traverse_inline(sch, output, callback)

    return sch

INTRINSICS = {
  'vnni': {
    'pattern': dots(16, 4, 'uint8', 'int8', 'int32'),
    'operands': [
        functools.partial(_load_concatenator, cast_type='int32x16'),
        functools.partial(_load_concatenator, cast_type='int32x16'),
        functools.partial(_load_concatenator, cast_type='int32x16')
    ],
    'write': _vnni_write,
    'init': functools.partial(_vec_init, dtype='int32', lanes=16),
    'schedule': functools.partial(_schedule_vdot, pattern=dots(16, 4, 'uint8', 'int8', 'int32'),
                                  pragma='vnni', max_threads=10000)
  },
  'vdot': {
      'pattern': dots(4, 4, 'int8', 'int8', 'int32'),
      'operands': [
          functools.partial(_load_concatenator, cast_type='int32x4'),
          functools.partial(_load_concatenator, cast_type='int8x16'),
          functools.partial(_load_concatenator, cast_type='int8x16')
      ],
      'write': _vdot_write,
      'init': functools.partial(_vec_init, dtype='int32', lanes=4),
      'schedule': functools.partial(_schedule_vdot, pattern=dots(4, 4, 'int8', 'int8', 'int32'),
                                    pragma='vdot', max_threads=10000)
  }
}