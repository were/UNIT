import tvm
import numpy as np

def as_const_int(expr):
    if isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        return expr.value
    return None


def extract_tiling(stmt, axes):
    res = {}

    # TODO(@were): Using name as the keys of variables is not good.
    def visit(op):
        if (isinstance(op, tvm.expr.Div) or isinstance(op, tvm.expr.Mod)) \
                and isinstance(op.a, tvm.expr.Var):
            val = as_const_int(op.b)
            if val is None:
                res[str(op.a)] = None
            elif op.a not in res.keys():
                res[str(op.a)] = val
            elif res[op.a] != val:
                res[str(op.a)] = None

    tvm.ir_pass.PostOrderVisit(stmt, visit)

    return res


def random_tvm_nd(a, is_output=False):
    shape = [as_const_int(i) for i in a.shape]
    if is_output:
        np_array = np.zeros(shape, dtype=a.dtype)
    elif a.dtype.startswith('int'):
        np_array = np.random.randint(0, 127, shape, a.dtype)
    else:
        np_array = np.random.randn(*shape).astype(a.dtype)
    return tvm.nd.array(np_array)
