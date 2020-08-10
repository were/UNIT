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
    batch, oc, x, y, ob = list(sch[conv].op.axis)

    cc = sch.cache_write(conv, 'wmma.accumulator')
    xy = sch[conv].fuse(x, y)
    #yo, yi = sch[conv].split(y, 32)
    xyo, xyi = sch[conv].split(xy, 32)
    oo, oi = sch[conv].split(ob, 16)
    xyio, xyii = sch[conv].split(xyi, 16)
    oio, oii = sch[conv].split(oi, 16)
    oco, oci = sch[conv].split(oc, 2)
    #print(batch, x, yo, oco, oci, oo, yio, oio, yii, oii, sep='\n')
    sch[conv].reorder(batch, xyo, oco, oo, xyio, oci, oio, xyii, oii)

    sch[cc].compute_at(sch[conv], oco)
    cb, coc, cx, cy, cob = sch[cc].op.axis
    cxy = sch[cc].fuse(cx, cy)
    crc, crh, crw = sch[cc].op.reduce_axis
    cxyo, cxyi = sch[cc].split(cxy, 16)
    crco, crci = sch[cc].split(crc, 16)
    crcoo, crcoi = sch[cc].split(crco, 2)
    #print(cb, crh, crw, crco, coc, cx, cyo, cyi, cob, crci, sep='\n')
    sch[cc].reorder(cb, crcoo, crh, crw, crcoi, cxyo, coc, cxyi, cob, crci)
    sch[cc].pragma(cxyo, 'tensorize', 'tensorcore')

    a_shared = sch.cache_read(a, 'local', [cc])
    sch[a_shared].compute_at(sch[cc], crcoi)
    as23 = sch[a_shared].fuse(sch[a_shared].op.axis[2], sch[a_shared].op.axis[3])
    sch[a_shared].bind(as23, te.thread_axis('threadIdx.x'))
    as4o, as4i = sch[a_shared].split(sch[a_shared].op.axis[4], 8)
    sch[a_shared].vectorize(as4i)

    aa = sch.cache_read(a_shared, 'wmma.matrix_a', [cc])
    #aa = sch.cache_read(a, 'wmma.matrix_a', [cc])
    sch[aa].compute_at(sch[cc], crcoi)
    a23 = sch[aa].fuse(sch[aa].op.axis[2], sch[aa].op.axis[3])
    a23o, a23i = sch[aa].split(a23, 16)
    sch[aa].pragma(a23o, 'tensorize', 'tensorcore.load_a')

    #b_shared = sch.cache_read(b, 'shared', [cc])
    #bb = sch.cache_read(b_shared, 'wmma.matrix_b', [cc])
    bb = sch.cache_read(b, 'wmma.matrix_b', [cc])
    sch[bb].compute_at(sch[cc], crcoi)
    #sch[b_shared].compute_at(sch[cc], crcoo)
    sch[bb].pragma(sch[bb].op.axis[0], 'tensorize', 'tensorcore.load_b')

    sch[conv].pragma(xyio, 'tensorize', 'tensorcore.store_c')

    sch[conv].bind(xyo, tvm.te.thread_axis('blockIdx.y'))
    sch[conv].bind(oco, tvm.te.thread_axis('blockIdx.x'))

def _conv2d_schedule_wdim(sch, conv):
    a, b = conv.op.input_tensors
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
                assert h * w % 32 == 0 and 32 % w == 0
                _conv2d_schedule_fused(sch, conv)

    traverse_inline(sch, output, callback)

    return sch