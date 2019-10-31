import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
t = np.array([[0], [1], [1], [0]])

np.random.seed(0)
w1 = np.random.normal(0, 1, [2, 2])
b1 = np.random.normal(0, 1, [2])
w2 = np.random.normal(0, 1, [2, 2])
b2 = np.random.normal(0, 1, [2])
w_out = np.random.normal(0, 1, [2, 1])
b_out = np.random.normal(0, 1, [1])

print("weight >>\n", w1)
print("bias >>\n", b1)
print("weight_out >>\n", w_out)
print("bias_out >>\n", b_out)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


z1 = x
lr = .1

for ite in range(10000):
    ite += 1
    z2 = sigmoid(np.dot(z1, w1) + b1)
    z3 = sigmoid(np.dot(z2, w2) + b2)
    out = sigmoid(np.dot(z3, w_out) + b_out)

    En = (out - t) * out * (1 - out)
    grad_w_out = np.dot(z2.T, En)
    grad_b_out = np.dot(np.ones([En.shape[0]]), En)
    w_out -= lr * grad_w_out
    b_out -= lr * grad_b_out

    grad_u2 = np.dot(En, w_out.T) * z3 * (1 - z3)
    grad_w2 = np.dot(z2.T, grad_u2)
    grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
    w2 -= lr * grad_w2
    b2 -= lr * grad_b2

    grad_u1 = np.dot(En, w_out.T) * z2 * (1 - z2)
    grad_w1 = np.dot(z1.T, grad_u1)
    grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
    w1 -= lr * grad_w1
    b1 -= lr * grad_b1

print("--------------train---------------")
print("weight >>\n", w1)
print("bias >>\n", b1)
print("weight_out >>\n", w_out)
print("bias_out >>\n", b_out)

for i, _x in enumerate(x):
    z2 = sigmoid(np.dot(z1[i], w1) + b1)
    z3 = sigmoid(np.dot(z2, w2) + b2)
    out = sigmoid(np.dot(z3, w_out) + b_out)
    print("in >>", _x, "out >>", out)
