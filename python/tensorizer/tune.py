import os

enable = False

cpu_idx = None
total_idx = None
ashape = None
bshape = None
strides = None

padding = None
splitk = None
x86 = {}

HOME = os.getenv("HOME")

def load_x86():
    try:
        f = open(HOME + '/Tensorization-PoC/cpu-tune.log')
    except:
        f = open(HOME + '/UNIT/cpu-tune.log')
    for i in f.readlines():
        i = i.replace(') ', '), ')
        try:
            a, b, s, v = eval(i)
        except:
            a, b, s, v, _, _ = eval(i)
        x86[(a, b, s)] = v

cuda_kernel = {}
cuda_relay = {}
def load_cuda():
    try:
        f = open(HOME + '/Tensorization-PoC/gpu-tune.log')
    except:
        f = open(HOME + '/UNIT/gpu-tune.log')
    raw = f.readlines()
    for i in raw[::2]:
        i = i.replace(') ', '), ')
        a, b, s, v, _ = eval(i)
        cuda_kernel[(a, b, s)] = v
    for i in raw[1::2]:
        N, C, H, W, O, I, KH, KW, SH, SW, v, _ = i.split()
        v = v.strip(',')
        cuda_relay[tuple(map(int, (N, C, H, W, O, I, KH, KW, SH, SW)))] = v

load_x86()
load_cuda()
