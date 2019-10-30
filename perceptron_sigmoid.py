import numpy as np

xs = np.array(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.float32)
ts = np.array(((0), (0), (0), (1)), dtype=np.float32)

np.random.seed(0)
w = np.random.normal(0., 1, (3))
print("weight >>", w)

_xs = np.hstack((xs, [[1] for _ in xs]))
lr = .1
ite = 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for ite in range(5000):
    ite += 1
    ys = sigmoid(np.dot(_xs, w))
    # print("iteration:", ite, "y >>", ys)

    En = -(ts - ys) * ys * (1 - ys)
    grad_w = np.dot(_xs.T, En)
    w -= lr * grad_w

print("training finished!")
print("weight >>", w)

for x in _xs:
    ys = sigmoid(np.dot(x, w))
    print("in >>", x, "out >>", ys)
