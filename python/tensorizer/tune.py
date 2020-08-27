cpu_idx = None
total_idx = None
ashape = None
bshape = None
strides = None

padding = None
splitk = None
x86 = {}

def load_x86():
    for i in open('/home/ubuntu/Tensorization-PoC/cpu-tune.log').readlines():
        i = i.replace(') ', '), ')
        try:
            a, b, s, v = eval(i)
        except:
            a, b, s, v, _, _ = eval(i)
        x86[(a, b, s)] = v

load_x86()