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
        ko, ib, kh, kw, ic, _ = get_const_tuple(b.shape)
        ko *= 16
        assert ib * ic == c
    else:
        assert False

    if need_pack:
        packed_a = te.compute((n, c // 16, h, w, 16),
                              lambda batch, oc, x, y, ob: a[batch, oc * 16 + ob, x, y],
                              tag='packed_kernel')
        packed_b = te.compute((ko // 16, ic // 16, kh, kw, 16, 16),
                              lambda oc, ic, x, y, ib, ob: b[oc * 16 + ob, ic * 16 + ib, x, y],
                              tag='packed_kernel')
        a = packed_a
        b = packed_b

    rc = te.reduce_axis((0, c), )
    rh = te.reduce_axis((0, kh), )
    rw = te.reduce_axis((0, kw), )

    def compute(batch, o_chunk, x, y, ob):
        A = a[batch, rc // 16, stride_h * x + rh, stride_w * y + rw, rc % 16].astype(out_type)
        B = b[o_chunk, rc // 16, rh, rw, rc % 16, ob].astype(out_type)
        return te.sum(A * B, axis=[rc, rh, rw])
                      
    conv = te.compute((n, ko // 16, (h - kh + 1) // stride_h, (w - kw + 1) // stride_w, 16), compute)

    return conv

def conv2d_NCHW16c_OHWI16o_compute(attrs, inputs, out_type):
    stride_h, stride_w = attrs.get_int_tuple('strides')
    assert stride_h == 1 and stride_w == 1
    return [_conv2d_NCHW16c_OHWI16o_impl(inputs[0], inputs[1], stride_h, stride_w, out_type.dtype)]

def _conv2d_schedule_fused(sch, conv):
    a, b = conv.op.input_tensors

    def schedule_fetcher(sch, buffer, y, x):
        axes = list(sch[buffer].op.axis)
        fused = sch[buffer].fuse(*axes[:-1])
        yo, yi = sch[buffer].split(fused, nparts=y)
        yio, yii = sch[buffer].split(yi, nparts=x)
        sch[buffer].bind(yo, te.thread_axis('threadIdx.y'))
        sch[buffer].bind(yio, te.thread_axis('threadIdx.x'))
        xo, xi = sch[buffer].split(axes[-1], 8)
        sch[buffer].vectorize(xi)

    rc = sch[conv].op.reduce_axis[0]
    rco, rci = sch[conv].split(rc, 32)
    rcio, rcii = sch[conv].split(rci, 16)
    rf = sch.rfactor(conv, rcio)
    cc = sch.cache_write(rf, 'wmma.accumulator')

    batch, oc, x, y, ob = list(sch[conv].op.axis)
    xy = sch[conv].fuse(x, y)
    xyo, xyi = sch[conv].split(xy, 32)
    oco, oci = sch[conv].split(oc, 2)
    sch[conv].bind(oci, te.thread_axis('threadIdx.y'))
    sch[conv].bind(xyi, te.thread_axis('threadIdx.x'))
    obo, obi = sch[conv].split(ob, 4)
    sch[conv].vectorize(obi)
    sch[conv].reorder(batch, oco, xyo, oci, xyi)
    sch[conv].bind(oco, te.thread_axis('blockIdx.y'))
    sch[conv].bind(xyo, te.thread_axis('blockIdx.x'))
    sch[rf].compute_at(sch[conv], xyo)

    rco, batch, oc, x, y, ob = list(sch[rf].op.axis)

    xy = sch[rf].fuse(x, y)
    xyo, xyi = sch[rf].split(xy, 32)
    oo, oi = sch[rf].split(ob, 16)
    xyio, xyii = sch[rf].split(xyi, 16)
    oio, oii = sch[rf].split(oi, 16)
    oco, oci = sch[rf].split(oc, 2)
    sch[rf].reorder(batch, xyo, oco, rco, oo, xyio, oci, oio, xyii, oii)
    sch[rf].pragma(xyio, 'tensorize', 'tensorcore.store_c')
    sch[rf].bind(rco, te.thread_axis('threadIdx.y'))

    sch[cc].compute_at(sch[rf], rco)
    cri, cb, coc, cx, cy, cob = sch[cc].op.axis
    cxy = sch[cc].fuse(cx, cy)
    crh, crw, crco, crci = sch[cc].op.reduce_axis
    cxyo, cxyi = sch[cc].split(cxy, 16)
    crcio, crcii = sch[cc].split(crci, 16)
    #print(cb, crh, crw, crco, coc, cx, cyo, cyi, cob, crci, sep='\n')
    sch[cc].reorder(cb, crco, crcio, crh, crw, cxyo, coc, cxyi, cob, crcii)
    sch[cc].pragma(cxyo, 'tensorize', 'tensorcore')

    a_reuse = sch.cache_read(a, 'shared', [cc])
    sch[a_reuse].compute_at(sch[cc], crcio)
    ar23 = sch[a_reuse].fuse(sch[a_reuse].op.axis[2], sch[a_reuse].op.axis[3])
    ar23o, ar23i = sch[a_reuse].split(ar23, 32)
    sch[a_reuse].bind(ar23o, tvm.te.thread_axis('threadIdx.y'))
    sch[a_reuse].bind(ar23i, tvm.te.thread_axis('threadIdx.x'))
    ar4o, ar4i = sch[a_reuse].split(sch[a_reuse].op.axis[4], 8)
    sch[a_reuse].vectorize(ar4i)

    a_shared = sch.cache_read(a_reuse, 'shared', [cc])
    sch[a_shared].compute_at(sch[cc], crw)
    schedule_fetcher(sch, a_shared, 2, 32)
    ##sch[a_shared].bind(sch[a_shared].op.axis[1], tvm.te.thread_axis('threadIdx.y'))
    #as23 = sch[a_shared].fuse(sch[a_shared].op.axis[2], sch[a_shared].op.axis[3])
    #sch[a_shared].bind(as23, te.thread_axis('threadIdx.x'))
    #as4o, as4i = sch[a_shared].split(sch[a_shared].op.axis[4], 8)
    #sch[a_shared].vectorize(as4i)

    aa = sch.cache_read(a_shared, 'wmma.matrix_a', [cc])
    #aa = sch.cache_read(a, 'wmma.matrix_a', [cc])
    sch[aa].compute_at(sch[cc], crw)
    a23 = sch[aa].fuse(sch[aa].op.axis[2], sch[aa].op.axis[3])
    a23o, a23i = sch[aa].split(a23, 16)
    sch[aa].pragma(a23o, 'tensorize', 'tensorcore.load_a')

    #b_shared = sch.cache_read(b, 'shared', [cc])
    #sch[b_shared].compute_at(sch[cc], crcio)
    #fused = sch[b_shared].fuse(*list(sch[b_shared].op.axis)[1:5])
    #fusedo, fusedi = sch[b_shared].split(fused, nparts=32)
    #sch[b_shared].bind(fusedo, te.thread_axis('threadIdx.x'))
    #vio, vii = sch[b_shared].split(sch[b_shared].op.axis[5], 8)
    #sch[b_shared].vectorize(vii)
    ##sch[b_shared].bind(sch[b_shared].op.axis[0], te.thread_axis('threadIdx.y'))

    bb = sch.cache_read(b, 'wmma.matrix_b', [cc])
    #bb = sch.cache_read(b, 'wmma.matrix_b', [cc])
    sch[bb].compute_at(sch[cc], crw)
    #sch[b_shared].compute_at(sch[cc], crcoo)
    sch[bb].pragma(sch[bb].op.axis[0], 'tensorize', 'tensorcore.load_b')

def _conv2d_schedule_wdim(sch, conv):
    a, b = conv.op.input_tensors

    rc = sch[conv].op.reduce_axis[0]
    rco, rci = sch[conv].split(rc, 64)
    rcio, rcii = sch[conv].split(rci, 16)
    rf = sch.rfactor(conv, rcio)

    batch, oc, x, y, ob = list(sch[conv].op.axis)
    yo, yi = sch[conv].split(y, 32)
    oo, oi = sch[conv].split(ob, 16)
    yio, yii = sch[conv].split(yi, 16)
    oio, oii = sch[conv].split(oi, 16)
    oco, oci = sch[conv].split(oc, 2)
    sch[conv].reorder(batch, x, yo, oco, oo, oci, yio, oio, yii, oii)
    sch[rf].compute_at(sch[conv], oo)

    sch[conv].bind(oco, te.thread_axis('blockIdx.y'))
    sch[conv].bind(x, te.thread_axis('blockIdx.x'))
    fused = sch[conv].fuse(oci, yio, oio)
    sch[conv].bind(fused, te.thread_axis('threadIdx.y'))
    vo, vi = sch[conv].split(oii, 8)
    sch[conv].vectorize(vi)
    fused = sch[conv].fuse(yii, vo)
    sch[conv].bind(fused, te.thread_axis('threadIdx.x'))

    cc = sch.cache_write(rf, 'wmma.accumulator')
    sch[cc].compute_at(sch[rf], sch[rf].op.axis[0])
    sch[rf].bind(sch[rf].op.axis[0], te.thread_axis('threadIdx.y'))

    rc, cb, coc, cx, cy, cob = sch[cc].op.axis
    crh, crw, crco, crci = sch[cc].op.reduce_axis
    cyo, cyi = sch[cc].split(cy, 16)
    sch[cc].reorder(rc, cb, crco, crh, crw, cx, cyo, coc, cyi, cob, crci)
    sch[cc].pragma(cyo, 'tensorize', 'tensorcore')

    aa = sch.cache_read(a, 'wmma.matrix_a', [cc])
    sch[aa].compute_at(sch[cc], crw)
    ao, ai = sch[aa].split(sch[aa].op.axis[3], 16)
    sch[aa].pragma(ao, 'tensorize', 'tensorcore.load_a')
    bb = sch.cache_read(b, 'wmma.matrix_b', [cc])
    sch[bb].compute_at(sch[cc], crw)
    sch[bb].pragma(sch[bb].op.axis[0], 'tensorize', 'tensorcore.load_b')

    rc, batch, oc, x, y, ob = sch[rf].op.axis
    yio, yii = sch[rf].split(y, 16)
    sch[rf].reorder(rc, batch, yio, oc, x, yii, ob)
    sch[rf].pragma(yio, 'tensorize', 'tensorcore.store_c')

    ir = tvm.lower(sch, [a, b, conv])



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
            n, c, h, w, _ = get_const_tuple(conv.shape)
            if w % 32 == 0:
                _conv2d_schedule_wdim(sch, conv)
            else:
                print('fused dimensions:')
                print(get_const_tuple(a.shape))
                print(get_const_tuple(b.shape))
                print(get_const_tuple(conv.shape))
                assert h * w % 32 == 0 and 32 % w == 0
                _conv2d_schedule_fused(sch, conv)

    traverse_inline(sch, output, callback)

    return sch