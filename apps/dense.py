import tvm

n = k = m = 1024

a = tvm.placeholder((n // 16, k // 4, 16, 4), dtype='int8', name='a')
b = tvm.placeholder((m // 16, k // 4, 16, 4), dtype='int8', name='b')
red = tvm.reduce_axis((0, k))
c = tvm.compute((n, m), lambda x, y:
        tvm.sum(
            a[x // 16, red // 4, x % 16, red % 4].astype('int32') * b[y // 16, red // 4, y % 16, red % 4],
            axis=red))

import autotensorize, random

vnni = autotensorize.vnni.pattern()
best_GVNNIs = 0.
best_schedule = None

from tvm import autotvm

@autotvm.template
def dense_auto():
    sch, doms, inner = autotensorize.tensorize.preprocessor(c, vnni)

    cfg = autotvm.get_config()

    for var, ext in doms.items():
        cfg.define_split('split_%s' % var, var,
                policy='candidate',
                candidate=autotensorize.dse.factorize(ext) + [(1, ext)],
                num_outputs=2)

    axes = []
    for var in doms.keys():
        vo, vi = cfg['split_%s' % var].apply(sch, c, var)
        axes.append(vo)
        axes.append(vi)

    cfg.define_reorder('reorder', axes, 'all')

    order = [axes[i] for i in cfg['reorder'].perm] + inner
    sch[c].reorder(*order)
    sch[c].pragma(inner[0], 'vnni')

    return sch, [a, b, c]


task = autotvm.task.create(dense_auto, args=[], target='llvm -mcpu=cascadelake')
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
            sch, args = dense_auto()
            print(tvm.lower(sch, args, simple_mode=True))
