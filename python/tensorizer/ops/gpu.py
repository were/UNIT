from topi.util import get_const_tuple, get_const_int
from topi.cuda.injective import schedule_injective_from_existing
from tvm import te
from tvm import autotvm
import tvm

@autotvm.register_topi_compute('conv2d_NCHW16c_OHWI16o.nvptx')
def _conv2d_NCHW16c_OHWI16o_impl(cfg, a, b, stride_h, stride_w, out_type):
    need_pack = False

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

    rc = te.reduce_axis((0, c), 'rc')
    rh = te.reduce_axis((0, kh), 'rh')
    rw = te.reduce_axis((0, kw), 'rw')

    if (stride_h, stride_w) == (1, 1):
        def compute(batch, o_chunk, x, y, ob):
            A = a[batch, rc // 16, stride_h * x + rh, stride_w * y + rw, rc % 16].astype(out_type)
            B = b[o_chunk, rc // 16, rh, rw, rc % 16, ob].astype(out_type)
            return te.sum(A * B, axis=[rc, rh, rw])
        conv = te.compute((n, ko // 16, (h - kh) // stride_h + 1, (w - kw) // stride_w + 1, 16), compute)
        return conv

    a_icol = te.compute((n, c // 16, (h - kh) // stride_h + 1, (w - kw) // stride_w + 1, kh, kw, 16),
                        lambda batch, i_chunck, ox, oy, rx, ry, i_block: a[batch, i_chunck, ox * stride_h + rx, oy * stride_w + ry, i_block],
                        name='a_i2c')
    conv = te.compute((n, ko // 16, (h - kh) // stride_h + 1, (w - kw) // stride_w + 1, 16),
                      lambda batch, o_chunk, x, y, ob: te.sum(a_icol[batch, rc // 16, x, y, rh, rw, rc % 16].astype('float32') *
                                                              b[o_chunk, rc // 16, rh, rw, rc % 16, ob].astype('float32'), axis=[rc, rh, rw]))
    return conv


def conv2d_NCHW16c_OHWI16o_compute(attrs, inputs, out_type):
    strides = attrs.get_int_tuple('strides')
    return [_conv2d_NCHW16c_OHWI16o_impl(inputs[0], inputs[1], strides[0], strides[1], out_type.dtype)]

def _conv2d_schedule_fused(sch, conv, output, stride_h, stride_w):
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
    rco, rci = sch[conv].split(rc, 64)
    rcio, rcii = sch[conv].split(rci, 16)
    rf = sch.rfactor(conv, rcio)
    cc = sch.cache_write(rf, 'wmma.accumulator')


    batch, oc, x, y, ob = list(sch[conv].op.axis)
    xy = sch[conv].fuse(x, y)
    oco, oci = sch[conv].split(oc, 2)
    xyo, xyi = sch[conv].split(xy, 32)
    xyio, xyii = sch[conv].split(xyi, 16)
    obo, obi = sch[conv].split(ob, 8)
    sch[conv].reorder(batch, oco, xyo, oci, xyio, xyii, obo, obi)
    sch[conv].bind(sch[conv].fuse(oci, xyio), te.thread_axis('threadIdx.y'))
    sch[conv].bind(sch[conv].fuse(xyii, obo), te.thread_axis('threadIdx.x'))
    sch[conv].vectorize(obi)
    sch[rf].compute_at(sch[conv], xyo)

    if conv.op == output:
        sch[conv].bind(oco, te.thread_axis('blockIdx.y'))
        sch[conv].bind(xyo, te.thread_axis('blockIdx.x'))
    else:
        batch, oc, x, y, ob = list(sch[output].op.axis)
        xy = sch[output].fuse(x, y)
        oco, oci = sch[output].split(oc, 2)
        xyo, xyi = sch[output].split(xy, 32)
        xyio, xyii = sch[output].split(xyi, 16)
        obo, obi = sch[output].split(ob, 8)
        sch[output].reorder(batch, oco, xyo, oci, xyio, xyii, obo, obi)
        sch[output].bind(sch[output].fuse(oci, xyio), te.thread_axis('threadIdx.y'))
        sch[output].bind(sch[output].fuse(xyii, obo), te.thread_axis('threadIdx.x'))
        sch[output].vectorize(obi)
        sch[output].bind(oco, te.thread_axis('blockIdx.y'))
        sch[output].bind(xyo, te.thread_axis('blockIdx.x'))
        sch[conv].compute_at(sch[output], xyo)

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
    
    if (stride_h, stride_w) != (1, 1):
        a_icol = a
        aaii = sch.cache_write(a_icol, 'shared')
        sch[aaii].compute_at(sch[cc], crw)
        sch[a_icol].compute_inline()
        fused = sch[aaii].fuse(sch[aaii].op.axis[1], sch[aaii].op.axis[2], sch[aaii].op.axis[3])
        fo, fi = sch[aaii].split(fused, nparts=4)
        fio, fii = sch[aaii].split(fi, nparts=32)
        sch[aaii].bind(fo, te.thread_axis('threadIdx.y'))
        sch[aaii].bind(fio, te.thread_axis('threadIdx.x'))
        sch[aaii].vectorize(sch[aaii].op.axis[6])
        a_shared = a_icol
    else:
        a_reuse = sch.cache_read(a, 'shared', [cc])
        sch[a_reuse].compute_at(sch[cc], crcio)
        schedule_fetcher(sch, a_reuse, 4, 32)
        a_shared = sch.cache_read(a_reuse, 'shared', [cc])
        sch[a_shared].compute_at(sch[cc], crw)
        schedule_fetcher(sch, a_shared, 4, 32)
    
    aa = sch.cache_read(a_shared, 'wmma.matrix_a', [cc])
    #aa = sch.cache_read(a, 'wmma.matrix_a', [cc])
    sch[aa].compute_at(sch[cc], crw)
    a23 = sch[aa].fuse(sch[aa].op.axis[2], sch[aa].op.axis[3])
    a23o, a23i = sch[aa].split(a23, 16)
    sch[aa].pragma(a23o, 'tensorize', 'tensorcore.load_a')
    
    
    bb = sch.cache_read(b, 'wmma.matrix_b', [cc])
    sch[bb].compute_at(sch[cc], crw)
    sch[bb].pragma(sch[bb].op.axis[0], 'tensorize', 'tensorcore.load_b')

def _conv2d_schedule_wdim(sch, conv, output, stride_h, stride_w):
    a, b = conv.op.input_tensors

    rc = sch[conv].op.reduce_axis[0]
    for i in [16, 32, 64]:
        if rc.dom.extent.value % i == 0:
            split_k = i

    rc = sch[conv].op.reduce_axis[0]
    rco, rci = sch[conv].split(rc, split_k)
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
    fused = sch[conv].fuse(oci, yio, oio)
    sch[conv].bind(fused, te.thread_axis('threadIdx.y'))
    vo, vi = sch[conv].split(oii, 8)
    sch[conv].vectorize(vi)
    fused = sch[conv].fuse(yii, vo)
    sch[conv].bind(fused, te.thread_axis('threadIdx.x'))

    if conv.op == output:
        fused = sch[conv].fuse(yo, oco)
        sch[conv].bind(fused, te.thread_axis('blockIdx.y'))
        sch[conv].bind(x, te.thread_axis('blockIdx.x'))
    else:
        batch, oc, x, y, ob = list(sch[output].op.axis)
        yo, yi = sch[output].split(y, 32)
        oo, oi = sch[output].split(ob, 16)
        yio, yii = sch[output].split(yi, 16)
        oio, oii = sch[output].split(oi, 16)
        oco, oci = sch[output].split(oc, 2)
        sch[output].reorder(batch, x, yo, oco, oo, oci, yio, oio, yii, oii)
        sch[conv].compute_at(sch[output], oo)
        fused = sch[output].fuse(yo, oco)
        sch[output].bind(fused, te.thread_axis('blockIdx.y'))
        sch[output].bind(x, te.thread_axis('blockIdx.x'))

        fused = sch[output].fuse(oci, yio, oio)
        if rf_rw:
            fo, fi = sch[output].split(fused, nparts=conv.op.reduce_axis[2].dom.extent.value)
        else:
            fo, fi = sch[output].split(fused, nparts=(split_k // 16))
        sch[output].bind(fo, te.thread_axis('threadIdx.y'))
        vo, vi = sch[output].split(oii, 8)
        sch[output].vectorize(vi)
        fused = sch[output].fuse(yii, vo)
        sch[output].bind(fused, te.thread_axis('threadIdx.x'))


    cc = sch.cache_write(rf, 'wmma.accumulator')
    sch[cc].compute_at(sch[rf], sch[rf].op.axis[0])
    sch[rf].bind(sch[rf].op.axis[0], te.thread_axis('threadIdx.y'))

    rc, cb, coc, cx, cy, cob = sch[cc].op.axis
    crh, crw, crco, crci = sch[cc].op.reduce_axis
    cyo, cyi = sch[cc].split(cy, 16)
    sch[cc].reorder(rc, cb, crco, crh, crw, cx, cyo, coc, cyi, cob, crci)
    sch[cc].pragma(cyo, 'tensorize', 'tensorcore')

    if (stride_h, stride_w) != (1, 1):
        assert isinstance(a.op, tvm.te.ComputeOp), type(a)
        a_icol = a
        aaii = sch.cache_write(a_icol, 'shared')
        sch[aaii].compute_at(sch[cc], crw)
        sch[a_icol].compute_inline()
        fused = sch[aaii].fuse(sch[aaii].op.axis[1], sch[aaii].op.axis[2], sch[aaii].op.axis[3])
        fo, fi = sch[aaii].split(fused, nparts=(split_k // 16))
        sch[aaii].bind(fo, te.thread_axis('threadIdx.y'))
        fio, fii = sch[aaii].split(fi, nparts=32)
        sch[aaii].bind(fio, te.thread_axis('threadIdx.x'))
        sch[aaii].vectorize(sch[aaii].op.axis[6])
        aa = sch.cache_read(a_icol, 'wmma.matrix_a', [cc])
    else:
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



def conv2d_NCHW16c_OHWI16o_schedule(attrs, outs, target):

    from topi.util import traverse_inline
    sch = te.create_schedule([i.op for i in outs])
    output = outs[0].op

    def callback(op):
        nonlocal sch
        if len(list(op.reduce_axis)):
            a, b = op.input_tensors

            conv = op.output(0)
            n, c, h, w, _ = get_const_tuple(conv.shape)
            stride_h, stride_w = attrs.get_int_tuple('strides')
            if w % 32 == 0:
                _conv2d_schedule_wdim(sch, conv, output, stride_h, stride_w)
            else:
                assert h * w % 32 == 0 and 32 % w == 0
                _conv2d_schedule_fused(sch, conv, output, stride_h, stride_w)

    traverse_inline(sch, output, callback)

    return sch
