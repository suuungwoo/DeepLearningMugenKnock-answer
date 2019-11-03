import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
t = np.array([[0], [1], [1], [0]])

np.random.seed(0)
w1 = np.random.normal(0, 1, [2, 2])
b1 = np.random.normal(0, 1, [2])
w2 = np.random.normal(0, 1, [2, 2])
b2 = np.random.normal(0, 1, [2])
w3 = np.random.normal(0, 1, [2, 1])
b3 = np.random.normal(0, 1, [1])

print("weight >>\n", w1)
print("bias >>\n", b1)
print("weight_out >>\n", w3)
print("bias_out >>\n", b3)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


z0 = x
lr = .1

for ite in range(10000):
    ite += 1

    # feed forward
    z1 = sigmoid(np.dot(z0, w1) + b1)
    z2 = sigmoid(np.dot(z1, w2) + b2)
    z3 = sigmoid(np.dot(z2, w3) + b3)

    # back propagate
    En = (z3 - t) * z3 * (1 - z3)
    grad_w3 = np.dot(z2.T, En)
    grad_b3 = np.dot(np.ones([En.shape[0]]), En)
    w3 -= lr * grad_w3
    b3 -= lr * grad_b3

    grad_u2 = np.dot(En, w3.T) * z2 * (1 - z2)
    grad_w2 = np.dot(z1.T, grad_u2)
    grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
    w2 -= lr * grad_w2
    b2 -= lr * grad_b2

    grad_u1 = np.dot(grad_u2, w2.T) * z1 * (1 - z1)
    grad_w1 = np.dot(z0.T, grad_u1)
    grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
    w1 -= lr * grad_w1
    b1 -= lr * grad_b1

print("--------------train---------------")
print("weight >>\n", w1)
print("bias >>\n", b1)
print("weight_out >>\n", w3)
print("bias_out >>\n", b3)

for i, _x in enumerate(x):
    z1 = sigmoid(np.dot(z0[i], w1) + b1)
    z2 = sigmoid(np.dot(z1, w2) + b2)
    z3 = sigmoid(np.dot(z2, w3) + b3)
    print("in >>", _x, "z3 >>", z3)
