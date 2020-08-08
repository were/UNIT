from topi.util import get_const_tuple, get_const_int
from topi.cuda.injective import schedule_injective_from_existing
from tvm import te
from tvm import autotvm
import tvm

@autotvm.register_topi_compute('conv2d_NCHW16c_OHWI16o.nvptx')
def _conv2d_NCHW16c_OHWI16o_impl(cfg, a, b, stride_h, stride_w, out_type):
    need_pack = False
    a, b
    if len(a.shape) == 4 and len(b.shape) == 4:
        n, c, h, w = get_const_tuple(a.shape)
        ko, ic, kh, kw = get_const_tuple(b.shape)
        need_pack = True
    elif len(a.shape) == 5 and len(b.shape) == 6:
        n, c, h, w, _ = get_const_tuple(a.shape)
        c *= 16
        ko, _, kh, kw, ic, _ = get_const_tuple(b.shape)
        ko *= 16
        assert ic == c
    else:
        assert False

    if need_pack:
        packed_a = te.compute((n, c // 16, h, w, 16),
                              lambda batch, oc, x, y, ob: a[batch, oc * 16 + ob, x, y],
                              tag='packed_kernel')
        packed_b = te.compute((ko // 16, 1, kh, kw, ic, 16),
                              lambda oc, ic, x, y, ib, ob: b[oc * 16 + ob, ib, x, y],
                              tag='packed_kernel')
        a = packed_a
        b = packed_b

    rc = te.reduce_axis((0, c), )
    rh = te.reduce_axis((0, kh), )
    rw = te.reduce_axis((0, kw), )

    def compute(batch, o_chunk, x, y, ob):
        A = a[batch, rc // 16, stride_h * x + rh, stride_w * y + rw, rc % 16].astype(out_type)
        # TODO(@were): Maybe later we support different oc.
        B = b[o_chunk, 0, rh, rw, rc, ob].astype(out_type)
        return te.sum(A * B, axis=[rc, rh, rw])
                      
    conv = te.compute((n, ko // 16, (h - kh + 1) // stride_h, (w - kw + 1) // stride_w, 16), compute)

    return conv

def conv2d_NCHW16c_OHWI16o_compute(attrs, inputs, out_type):
    stride_h, stride_w = attrs.get_int_tuple('strides')
    assert stride_h == 1 and stride_w == 1
    return [_conv2d_NCHW16c_OHWI16o_impl(inputs[0], inputs[1], stride_h, stride_w, out_type.dtype)]

def conv2d_NCHW16c_OHWI16o_schedule(attrs, outs, target):

    from topi.util import traverse_inline
    sch = te.create_schedule([i.op for i in outs])
    output = outs[0].op

    def callback(op):
        nonlocal sch
        if len(list(op.reduce_axis)):
            a, b = op.input_tensors

            if isinstance(a.op, te.ComputeOp):
                schedule_injective_from_existing(sch, a)
            if isinstance(b.op, te.ComputeOp):
                schedule_injective_from_existing(sch, b)

            conv = op.output(0)

            # TODO(@were): Uncomment this later
            #info = list(tvm.arith._ffi_api.MatchTensorizer(op, ))
            #assert info

            batch, oc, x, y, ob = list(sch[conv].op.axis)

            cc = sch.cache_write(conv, 'wmma.accumulator')
            yo, yi = sch[conv].split(y, 32)
            oo, oi = sch[conv].split(ob, 16)
            yio, yii = sch[conv].split(yi, 16)
            oio, oii = sch[conv].split(oi, 16)
            oco, oci = sch[conv].split(oc, 2)
            sch[conv].reorder(batch, x, yo, oco, oo, yio, oci, oio, yii, oii)

            sch[cc].compute_at(sch[conv], oco)
            cb, coc, cx, cy, cob = sch[cc].op.axis
            crc, crh, crw = sch[cc].op.reduce_axis
            cyo, cyi = sch[cc].split(cy, 16)
            crco, crci = sch[cc].split(crc, 16)
            #print(cb, crh, crw, crco, coc, cx, cyo, cyi, cob, crci, sep='\n')
            sch[cc].reorder(cb, crh, crw, crco, cx, cyo, coc, cyi, cob, crci)
            sch[cc].pragma(cyo, 'tensorize', 'tensorcore')

            aa = sch.cache_read(a, 'wmma.matrix_a', [cc])
            sch[aa].compute_at(sch[cc], crco)
            ao, ai = sch[aa].split(sch[aa].op.axis[3], 16)
            sch[aa].pragma(ao, 'tensorize', 'tensorcore.load_a')
            bb = sch.cache_read(b, 'wmma.matrix_b', [cc])
            sch[bb].compute_at(sch[cc], crco)
            sch[bb].pragma(sch[bb].op.axis[0], 'tensorize', 'tensorcore.load_b')

            sch[conv].pragma(yio, 'tensorize', 'tensorcore.store_c')

            sch[conv].bind(x, te.thread_axis('blockIdx.x'))
            sch[conv].bind(oco, te.thread_axis('blockIdx.y'))

    traverse_inline(sch, output, callback)

    return sch