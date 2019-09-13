import tvm
from tvm import autotvm

import autotensorize

n = k = m = 1024

@autotvm.template
def packed_matmul_vnni(n, k, m):
    a = tvm.placeholder((n // 16, k // 4, 16, 4), dtype='int8', name='a')
    b = tvm.placeholder((m // 16, k // 4, 16, 4), dtype='int8', name='b')
    red = tvm.reduce_axis((0, k))
    c = tvm.compute((n, m), lambda x, y:
            tvm.sum(
                a[x // 16, red // 4, x % 16, red % 4].astype('int32') * b[y // 16, red // 4, y % 16, red % 4],
                axis=red))

    vnni = autotensorize.vnni.pattern()
    sch, args = autotensorize.tensorize.define_space(c, vnni, [a, b, c], 'vnni')

    return sch, args


task = autotvm.task.create(packed_matmul_vnni, [n, k, m], target='llvm -mcpu=cascadelake')

print(task.config_space)

tuner = autotvm.tuner.XGBTuner(task)

measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))

import logging, sys

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

with tvm.build_config(add_lower_pass= [(1, autotensorize.vnni.customized_pass)]):
    tuner.tune(n_trial=10000, measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file('T_T.log')])

    with autotvm.apply_history_best('T_T.log'):
        with tvm.target.create('llvm -mcpu=cascadelake'):
            sch, args = packed_matmul_vnni(n, k, m)
            print(tvm.lower(sch, args, simple_mode=True))
