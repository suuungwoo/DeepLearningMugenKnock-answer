import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
t = np.array([[0], [1], [1], [0]])

np.random.seed(0)
w1 = np.random.normal(0, 1, [2, 2])
b1 = np.random.normal(0, 1, [2])
w2 = np.random.normal(0, 1, [2, 1])
b2 = np.random.normal(0, 1, [1])

print("weight >>", w1)
print("bias >>", b1)
print("weight_out >>", w2)
print("bias_out >>", b2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


z1 = x
lr = .1

for ite in range(5000):
    ite += 1
    z2 = sigmoid(np.dot(z1, w1) + b1)
    out = sigmoid(np.dot(z2, w2) + b2)

    En = (out - t) * out * (1 - out)
    grad_w2 = np.dot(z2.T, En)
    grad_b2 = np.dot(np.ones([En.shape[0]]), En)
    w2 -= lr * grad_w2
    b2 -= lr * grad_b2

    grad_u1 = np.dot(En, w2.T) * z2 * (1 - z2)
    grad_w1 = np.dot(z1.T, grad_u1)
    grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
    w1 -= lr * grad_w1
    b1 -= lr * grad_b1

print("weight >>", w1)
print("bias >>", b1)
print("weight_out >>", w2)
print("bias_out >>", b2)

for i, _x in enumerate(x):
    z2 = sigmoid(np.dot(z1[i], w1) + b1)
    out = sigmoid(np.dot(z2, w2) + b2)
    print("in >>", _x, "out >>", out)
