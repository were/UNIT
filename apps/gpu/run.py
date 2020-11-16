import subprocess

with open('./intersect', 'r') as f:
    shapes = []
    for i in f.readlines():
        a = eval(i)
        shapes.append(' '.join(map(str, a[:-2])))
    shapes = set(shapes)
    for i in shapes:
        with open('input', 'w') as f:
            f.write(i)
        print('tuning:', i)
        subprocess.check_output('python relay.py < input', shell=True)
