import subprocess
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--target', type=str, default='nvptx')
args = ap.parse_args()

#with open('/home/ubuntu/shapes.raw', 'r') as f:
#    shapes = []
#    for i in f.readlines():
#        shapes.append(i)
#    shapes = set(shapes)
#    for i in shapes:
#        with open('input', 'w') as f:
#            f.write(i)
#        print('tuning:', i)
#        subprocess.check_output('python relay.py < input', shell=True)
#

shapes = [
        (1, 160, 9, 9, 224, 160, 3, 3, 1, 1),
        (1, 1056, 7, 7, 192, 1056, 1, 1, 1, 1),
        (1, 128, 16, 16, 128, 128, 3, 3, 1, 1),
        (1, 192, 16, 16, 192, 192, 3, 3, 1, 1),
        (1, 256, 16, 16, 256, 256, 3, 3, 1, 1),
        (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1),
        (1, 128, 16, 16, 160, 128, 3, 3, 1, 1),
        (1, 576, 14, 14, 192, 576, 1, 1, 1, 1),
        (1, 96, 16, 16, 128, 96, 3, 3, 1, 1),
        (1, 1024, 14, 14, 256, 1024, 1, 1, 1, 1),
        (1, 576, 14, 14, 128, 576, 1, 1, 1, 1),
        (1, 64, 29, 29, 96, 64, 3, 3, 1, 1),
        (1, 64, 56, 56, 128, 64, 1, 1, 2, 2),
        (1, 608, 14, 14, 192, 608, 1, 1, 1, 1),
        (1, 288, 35, 35, 384, 288, 3, 3, 2, 2),
        (1, 80, 73, 73, 192, 80, 3, 3, 1, 1),
        ]

target = args.target

for i in shapes:
    with open('input', 'w') as f:
        f.write(' '.join(map(str, i)))
    print('tuning:', i)
    subprocess.check_output(f'python relay.py --target={target} < input', shell=True)
