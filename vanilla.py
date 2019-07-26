import tvm
import topi
import numpy as np

def vectorize_init(stmt):

    to_vectorize = []

    def add_loop_level(op):
        if isinstance(op, tvm.stmt.AttrStmt):
            if op.attr_key == 'pragma_vnni compute':
                to_vectorize.append(op.node.var)
        return None



    def vectorize(op):
        if isinstance(op, tvm.stmt.For):
            if to_vectorize:
                if str(op.loop_var) == f'{to_vectorize[-1]}.init':
                    return tvm.make.For(op.loop_var, op.min, op.extent, tvm.stmt.For.Vectorized,
                                        op.device_api, op.body)
                elif str(op.loop_var) == str(to_vectorize[-1]):

                    loops = []
                    loads = []
                    def get_loops(op):
                        if isinstance(op, tvm.stmt.For):
                            loops.append(op.loop_var)
                        elif isinstance(op, tvm.expr.Load):
                            loads.append(op)

                    tvm.ir_pass.PostOrderVisit(op, get_loops)

                    inner, outer = loops
                    empty = {outer: tvm.const(0, 'int32'), inner: tvm.const(0, 'int32')}

                    new_loads = []

                    for load in loads:
                        index = tvm.ir_pass.Substitute(load.index, empty)
                        index = tvm.ir_pass.CanonicalSimplify(index)

                        if load.dtype == 'int32':
                            lhs = tvm.make.Broadcast(index, 16)
                            rhs = tvm.make.Ramp(tvm.const(0, 'int32'), tvm.const(1, 'int32'), 16)
                            dtype = 'int32x16'
                            index = lhs + rhs
                            new_load = tvm.make.Load(dtype, load.buffer_var, index,
                                                     tvm.const(1, index.dtype))
                        elif sum(tvm.ir_pass.ExprUseVar(load, var) for var in loops) == 1:
                            values = []
                            dtype = 'int8x64'
                            loads = []
                            for i in range(4):
                                new_load = tvm.make.Load('int8', load.buffer_var, index + tvm.const(i, 'int32'))
                                loads.append(tvm.make.Broadcast(new_load, 16))
                            #index = tvm.make.Broadcast(index, 4)
                            #index = index + tvm.make.Ramp(tvm.const(0, 'int32'), tvm.const(1, 'int32'), 4)
                            #aword = tvm.make.Load('int8x4', load.buffer_var,
                            #                      index,
                            #                      tvm.const(1, 'int32x4'))
                            new_load = tvm.make.Shuffle(loads, [tvm.const(i, 'int32') for i in range(64)])
                            print(new_load)
                        else:
                            lhs = tvm.make.Broadcast(index, 64)
                            rhs = tvm.make.Ramp(tvm.const(0, 'int32'), tvm.const(1, 'int32'), 64)
                            dtype = 'int8x64'
                            index = lhs + rhs
                            new_load = tvm.make.Load(dtype, load.buffer_var, index,
                                                     tvm.const(1, index.dtype))
                            #indeces = [tvm.const((i % 4) * 16 + i // 4, 'int32') for i in range(64)]
                            #new_load = tvm.make.Shuffle([new_load], indeces)
                        new_loads.append(new_load)

                    buffer_var = new_loads[0].buffer_var
                    index = new_loads[0].index

                    #new_loads = [tvm.const(0, 'int8x32')] * 3

                    new_loads = [tvm.call_pure_intrin('int32x16', 'reinterpret', i) for i in new_loads]

                    #index = tvm.const(0, 'int32x16')
                    #new_loads[0] = tvm.const(0, 'int32x16')
                    #new_loads[1] = tvm.const(0, 'int32x16')
                    #new_loads[2] = tvm.const(0, 'int32x16')

                    res = tvm.call_llvm_intrin('int32x16', 'llvm.x86.avx512.vpdpbusd.512',
                                               tvm.const(0, 'uint32'),
                                               *new_loads)

                    res = tvm.make.Store(buffer_var, res, index, tvm.const(1, 'int32x16'))
                    return res
        elif isinstance(op, tvm.stmt.AttrStmt):
            if op.attr_key == 'pragma_vnni memory':
                return op.body
            if not to_vectorize:
                return None
            if op.node.var == to_vectorize[-1]:
                to_vectorize.pop()
                return op.body
        return None

    return tvm.ir_pass.IRTransform(stmt, add_loop_level, vectorize, ['AttrStmt', 'For'])

print('???')

with tvm.target.create('llvm'):
    image = tvm.placeholder((1, 2, 192, 192, 4), dtype='int8', name='input')
    kernel = tvm.placeholder((1, 2, 32, 32, 4, 16), dtype='int8', name='kernel')

    conv = topi.nn.conv2d_NCHWc(image, kernel, stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                                layout='NCHW4c', out_layout = 'NCHW4c',
                                out_dtype='int32')
    print(conv.shape)
    print(kernel.shape)

    sch = tvm.create_schedule(conv.op)

    n, c0, h, w, c1 = conv.op.axis
    ro, rh, rw = conv.op.reduce_axis
    rwo, rwi = sch[conv.op].split(rw, 16)
    roo, roi = sch[conv.op].split(ro, 4)

    sch[conv.op].reorder(n, c0, h, w, rh, rwo, roo, rwi, c1, roi)

    #sch[conv.op].pragma(rwi, 'vnni memory')
    sch[conv.op].pragma(c1, 'vnni compute')

    raw = tvm.create_schedule(conv.op)
    func = tvm.lower(raw, [image, kernel, conv])
    answer_ref = tvm.build(raw, [image, kernel, conv])

    print(tvm.lower(raw, [image, kernel, conv], simple_mode=True))
    print(tvm.lower(sch, [image, kernel, conv], simple_mode=True))

    with tvm.build_config(add_lower_pass=[(1, vectorize_init)]):
        ir = tvm.lower(sch, [image, kernel, conv], simple_mode=True)
        print(ir)
        module = tvm.build(sch, [image, kernel, conv], 'llvm -mcpu=cascadelake')
        shapes = [i.shape for i in [image, kernel]]
        shapes = [list(map(lambda x: x.value, i)) for i in shapes]
        out_shape = list(map(lambda x: x.value, conv.shape)) 
        types = ['int8', 'int8', 'int32']
        args = [tvm.ndarray.array(np.random.randint(1, 2, i, j)) for i, j in zip(shapes, types)]
        out = tvm.ndarray.array(np.zeros(out_shape).astype('int32'))
        ans = tvm.ndarray.array(np.zeros(out_shape).astype('int32'))

        module.save('vnni.ll')
        module(args[0], args[1], out)
        answer_ref(args[0], args[1], ans)
        tvm.testing.assert_allclose(out.asnumpy(), ans.asnumpy())
        #print('Correctness pass!')

        vannila = answer_ref.time_evaluator(answer_ref.entry_name, tvm.cpu(0), number=10)
        vnni = module.time_evaluator(module.entry_name, tvm.cpu(0), number=10)
        print(vannila(args[0], args[1], ans).mean)
        print(vnni(args[0], args[1], out).mean)
