res = []
for i in open('input', 'r').readlines():
    a, b, s = eval(i)
    N, C, H, W, c = a
    K, _, KH, KW, _, k, _ = b
    res.append((C * c, H, W, K * k, KH, s[0]))

res = set(res)
print(*res, sep=',\n')
