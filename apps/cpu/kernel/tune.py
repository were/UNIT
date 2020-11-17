import subprocess

with open('./cpu-shapes.log') as f:
    shapes = []
    for i in f.readlines():
        try:
            a = i.split()
            i = (''.join(a[:-1])).replace(')', '), ')
            val = eval(i)
            if isinstance(val[2], tuple):
                shapes.append(tuple(list(val) + [a[-1].rstrip()]))
        except:
            pass
    shapes = set(shapes)
    print(len(shapes))

    for elem in shapes:
        if len(elem) != 4:
            continue
        a, b, s, f = elem
        N, C, H, W, c = a
        O, I, KH, KW, e, o, i = b
        args = f'{N} {C} {H} {W} {c} {O} {I} {KH} {KW} {e} {o} {i} {s[0]} {s[1]}'
        #print(f.split('/')[-1], N, C * c, H, W, O * o, I * i * e, KH, KW, s[0], s[1])
        with open('input', 'w') as f:
            f.write(args)
        print('tuning:', args)
        subprocess.check_output('python conv2d.py < input', shell=True)
