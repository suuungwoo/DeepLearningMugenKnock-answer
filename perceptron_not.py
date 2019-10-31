import numpy as np

np.random.seed(0)

xs = np.array([[0], [1]], dtype=np.float32)
ts = np.array([1, 0], dtype=np.float32)

w = np.random.normal(0., 1, (1))
b = np.random.normal(0., 1, (1))
print("weight >>", w)
print("bias >>", b)

z1 = xs
lr = .1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for ite in range(5000):
    ite += 1
    ys = sigmoid(np.dot(z1, w) + b)
    # print("iteration:", ite, "y >>", ys)

    En = -(ts - ys) * ys * (1 - ys)
    grad_w = np.dot(z1.T, En)
    grad_b = np.dot(np.ones([En.shape[0]]), En)
    w -= lr * grad_w
    b -= lr * grad_b

print("training finished!")
print("weight >>", w)
print("bias >>", b)

for x in xs:
    ys = sigmoid(np.dot(x, w) + b)[0]
    print("in >>", x, "out >>", ys)
