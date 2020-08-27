import subprocess

with open('../../../cpu-shapes.log') as f:
    shapes = []
    for i in f.readlines():
        try:
            i = i.replace(') ', '), ')
            exec(f'a = {i}')
            if isinstance(a[2], tuple):
                shapes.append(a)
        except:
            pass
    shapes = set(shapes)

    for a, b, s in shapes:
        N, C, H, W, c = a
        O, I, KH, KW, e, o, i = b
        print(a, b, s)
        args = f'{N} {C} {H} {W} {c} {O} {I} {KH} {KW} {e} {o} {i} {s[0]} {s[1]}'
        with open('input', 'w') as f:
            f.write(args)
        print('tuning:', args)
        subprocess.check_output('python conv2d.py < input', shell=True)
