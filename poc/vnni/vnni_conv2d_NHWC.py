import tvm

from tvm import relay

#@relay.op.register_compute("relay.op.nn.conv2d_vnni")
#def compute(attr, inputs, out_type, target):
#    img = inputs[0]
#    knl = inputs[1]
#    n, h, w, c = img.shape
#    kc = c
#    kh, kw, o_, _, xo, _ = knl.shape
#    o = o_ * xo
#    rc, rh, rw = tvm.reduce_axis((0, kc), 'rc'), tvm.reduce_axis((0, kh), 'rh'), tvm.reduce_axis((0, kw), 'rw')
#    conv = tvm.compute(
#            (n, h - kh + 1, w - kw + 1, o),
#            lambda bn, x, y, oc:
#                tvm.sum(img[bn, x + rh, y + rw, rc].astype('int32') *
#                knl[rh, rw, oc // 16, rc // 4, oc % 16, rc % 4],
#            axis=[rc, rh, rw]),
#            'conv')
#    return [conv]
#
#@relay.op.register_schedule("relay.op.nn.conv2d_vnni")
#def schedule(attrs, outputs, target):
#    conv = outputs[0]
#    sch = tvm.create_schedule(conv.op)
#    sch = tvm.create_schedule(conv.op)
#    #vannila = tvm.build(sch, [img, knl, conv], 'llvm -mcpu=cascadelake')
#
#    bn, x, y, oc = conv.op.axis
#    rc, rh, rw = conv.op.reduce_axis
#    #sch[conv].parallel(x)
#
#    oco, oci = sch[conv].split(oc, 16)
#    rco, rci = sch[conv].split(rc, 4)
#
#    sch[conv].reorder(bn, x, rh, rw, y, rco, oco, oci, rci)
#    #sch[conv].unroll(oco)
#    sch[conv].pragma(oci, 'vnni')
#    #cached = sch.cache_read(img, 'global', [conv])
#    #sch[cached].compute_at(sch[conv], y)
#    #sch[cached].vectorize(cached.op.axis[3])
#    return sch

#relay.op.register_pattern("relay.op.nn.conv2d_vnni", relay.op.OpPattern.OUT_ELEMWISE_FUSABLE)

n, h, w, c = 1, 130, 130, 128
o, kc, kh, kw = 128, c, 3, 3

img = relay.var('x', relay.ty.TensorType((n, h, w, c), 'int8'))
knl = relay.var('w', relay.ty.TensorType((kh, kw, o // 16, c // 4, 16, 4), 'int8'))

conv2d_vnni = relay.op.nn.conv2d_vnni(img, knl, strides=1, padding=0)
func = relay.Function([img, knl], conv2d_vnni)

ops = n * (h - kh + 1) * (w - kw + 1) * o * kc * kh * kw / 64

import vnni
import numpy as np
module = tvm.IRModule.from_expr(func)
with tvm.build_config(add_lower_pass= [(1, vnni.vnni_transformation)]):
    graph, module, params = relay.build(func, target='llvm -mcpu=cascadelake')
    x_ = tvm.nd.array((np.random.randn(n, h, w, c) * 255).astype('int8'), ctx=tvm.cpu())
    w_ = tvm.nd.array((np.random.randn(kw, kh, o // 16, c // 4, 16, 4) * 255).astype('int8'),
                     ctx=tvm.cpu())
    y_ = tvm.nd.array((np.random.randn(n, h - kh + 1, w - kw + 1, o) * 255).astype('int32'),
                      ctx=tvm.cpu())

    module = module.time_evaluator(module.entry_name, tvm.cpu(), number=5)
    span = module(x_, w_, y_).mean
    print('Exec Time: ', span)
    print('%.2f GVNNI/s' % (ops / span / 1e9))

    #module = tvm.contrib.graph_runtime.create(graph, module, tvm.cpu())
    #module.set_input('x', x)
    #module.set_input('w', w)
    #module.run()
