import numpy as np

xs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)

np.random.seed(0)
w1s = np.random.normal(0, 1, [2, 2])
b1s = np.random.normal(0, 1, 2)
w2s = np.random.normal(0, 1, [2, 1])
b2s = np.random.normal(0, 1, 1)

print("weight >>", w1s)
print("bias >>", b1s)
print("weight_out >>", w2s)
print("bias_out >>", b2s)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


zs = xs

for i, x in enumerate(xs):
    z2s = sigmoid(np.dot(zs[i], w1s) + b1s)
    out = sigmoid(np.dot(z2s, w2s) + b2s)
    print("in >>", x, "out >>", out)
