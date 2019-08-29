"""Design space automatic exploration"""
import tvm
from .util import as_const_int


def factorize(n):
    res = []
    i = 2
    while i * i <= n:
        if n % i == 0:
            res.append((i, n // i))
        i += 1
    return res


def coef_accumulation(stmt, axis):
    n = [len(axis)]
    res = [0] * len(axis)

    def visit(op):
        if isinstance(op, (tvm.expr.Load, tvm.stmt.Store)):
            coef = tvm.arith.DetectLinearEquation(op.index, axis)
            for i in range(n[0]):
                res[i] += as_const_int(coef[i])

    tvm.ir_pass.PostOrderVisit(stmt, visit)

    return res


def enumerate_factor(dom_map):
    factors = []
    for _, i in dom_map.items():
        factors.append(factorize(i) + [1])
    idx = len(factors) * [0]
    while idx[0] != len(factors[0]):
        yield [j[i] for i, j in zip(idx, factors)]
        idx[-1] += 1
        i = len(idx) - 1
        while i >= 1 and idx[i] == len(factors[i]):
            idx[i] = 0
            idx[i - 1] += 1
            i -= 1


def enumerate_inject(order, to_inject):
    idx = list(range(len(to_inject)))
    while idx[0] != len(order) - 1:
        res = order[:]
        for i in range(len(to_inject)):
            res.insert(idx[i] + i + 1, to_inject[i])
        yield res
        idx[-1] += 1
        while i >= 1 and idx[i] >= len(order):
            idx[i] = -1
            idx[i - 1] += 1
            i -= 1
        for j in range(i + 1, len(idx)):
            idx[j] = idx[j - 1] + 1
