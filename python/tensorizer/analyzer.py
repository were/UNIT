import functools
import operator

import tvm

def _factors(x):
    res = []
    for i in range(2, x):
        if x % i == 0:
            res.append(i)
            res.append(x // i)
        if i * i > x:
            break
    return [1, x] + sorted(res) if res else sorted(list(range(2, 9)), key=lambda v: x % v)

def _ceil_div(a, b):
    return (a - 1) // b + 1


def analyze_tiling(op, pattern, max_unroll=32, max_parallel=10000):

    info = list(tvm.arith._ffi_api.MatchTensorizer(op, pattern))
    assert info
    loops = {}
    for i, j in zip(info[::2], info[1::2]):
        loops[i] = j

    dom = {}
    split = []
    for i in op.axis:
        split.append([i.dom.extent.value])
    for i in op.reduce_axis:
        split.append([i.dom.extent.value])
    
    def tiling_stencil(axis, offset):
        nonlocal loops
        for i, j in enumerate(axis):
            if j in loops.keys():
                factor = loops[j].dom.extent.value
                split[i + offset] = [j.dom.extent.value // factor, (factor, 'offload')]

    tiling_stencil(op.axis, 0)
    tiling_stencil(op.reduce_axis, len(op.axis))

    # from outer to inner enumerate the loop levels to be parallelized
    for parallel in range(len(op.axis)):
        for tile_parallel in _factors(split[parallel][0]):
            copy_split = split[:]
            fused_prod = functools.reduce(
                operator.mul,
                [j for i in copy_split[:parallel] for j in i if isinstance(j, int)],
                1) * _ceil_div(copy_split[parallel][0], tile_parallel)
            if fused_prod > max_parallel:
                continue
            copy_split[parallel] = [(_ceil_div(copy_split[parallel][0], tile_parallel), 'parallel'),
                                    tile_parallel] + copy_split[parallel][1:]
            for unroll in range(parallel, len(op.axis)):
                j = 1 if unroll == parallel else 0
                for tile_unroll in _factors(copy_split[unroll][j]):
                    yield_split = copy_split[:]
                    yield_split[unroll] = yield_split[unroll][:j] + [
                        _ceil_div(yield_split[unroll][j], tile_unroll),
                        (tile_unroll, 'unroll')] + yield_split[unroll][j+1:]
                    unroll_prod = functools.reduce(
                        operator.mul,
                        [j for i in yield_split[unroll+1:len(op.axis)] for j in i if isinstance(j, int)],
                        1) * tile_unroll
                    if unroll_prod > max_unroll:
                        break
                    yield [split[parallel][0] % tile_parallel,
                           copy_split[unroll][j] % tile_unroll,
                           fused_prod, unroll_prod, yield_split]
        if len(split[parallel]) != 1:
            break