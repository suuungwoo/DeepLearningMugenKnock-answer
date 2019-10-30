import numpy as np

xs = np.array(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.float32)
ts = np.array(((-1), (-1), (-1), (1)), dtype=np.float32)

np.random.seed(0)
w = np.random.normal(0., 1, (3))
print("weight >>", w)

_xs = np.hstack((xs, [[1] for _ in xs]))
lr = .1
ite = 0

while True:
    ys = np.dot(_xs, w)
    ite += 1
    print("iteration:", ite, "y >>", ys)
    if len(np.where(ys * ts < 0)[0]) < 1:
        break
    _ys = ys.copy()
    _ts = ts.copy()
    _ys[ys * ts >= 0] = 0
    _ts[ys * ts >= 0] = 0
    En = np.dot(_ts, _xs)
    w += lr * En

print("training finished!")
print("weight >>", w)

for x in _xs:
    print("in >>", x, "out >>", np.dot(x, w))
