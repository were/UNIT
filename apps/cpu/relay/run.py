import subprocess

workloads = [(1024,16,16,2048,1024,1,1,1,1),
(960,8,8,160,960,1,1,1,1),
(256,56,56,512,256,1,1,1,1),
(64,56,56,128,64,1,1,1,1),
(512,10,10,512,512,3,3,1,1),
(192,16,16,160,192,3,3,1,1),
(64,65,65,128,64,3,3,1,1),
(320,65,65,384,320,3,3,1,1),
(384,8,8,384,384,1,3,1,1),
(192,23,23,192,192,7,1,1,1),
(3,224,224,64,3,7,7,2,2),
(1024,15,15,2048,1024,1,1,2,2),
(128,65,65,256,128,3,3,2,2)]

for i in workloads:
    exec_time = []
    with open('input', 'w') as f:
        f.write('1 ' + ' '.join(map(str, i)))
    try:
        avg = []
        for j in range(10):
            res = subprocess.check_output('python ./conv2d.py < input', shell=True).decode('utf-8')
            mns = []
            for k in res.split('\n'):
                if 'fused_nn_contrib_conv2d_NCHWc_subtract_add_multiply' in k:
                    mns.append(float(k.split()[2]))
            avg.append(min(mns))
        exec_time.append(sum(avg) / len(avg))
    except:
        print(i, 'fails')
    print('!!!', i, min(exec_time))

