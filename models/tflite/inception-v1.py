import tflite
import tvm
from tvm import relay
from PIL import Image
import numpy as np
import tensorizer


buffer = open('1.tflite', 'rb').read()
image = Image.open('./data.jpg').resize((224, 224))
x = np.array(image).astype('uint8')
data = np.reshape(x, (1, 224, 224, 3))

model = tflite.Model.GetRootAsModel(buffer, 0)
mod, params = relay.frontend.from_tflite(model,
                                         shape_dict={'input': data.shape},
                                         dtype_dict={'input': data.dtype.name})
print('load model')

mod = relay.transform.ConvertLayout({'qnn.conv2d': ['NCHW', 'OIHW']})(mod)
print('convert layout')

with tvm.transform.PassContext(config={'tir.add_lower_pass': [(1, tensorizer.rewrite)]}), \
     tvm.transform.PassContext(opt_level=3):

    graph, lib, params = relay.build_module.build(mod, target='llvm -mcpu=cascadelake', params=params)
    print('built')

    from tvm.contrib import graph_runtime as runtime
    module = runtime.create(graph, lib, tvm.cpu())
    module.set_input(**params)
    module.set_input('input', data)
    timer = module.module.time_evaluator('run', ctx=tvm.cpu(0), number=10)
    timed = timer()
    print('%.2fms' % (timed.mean * 1000))
