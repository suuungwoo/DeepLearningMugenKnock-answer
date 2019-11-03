import numpy as np

np.random.seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FullyConnectedLayer():
    def __init__(self, in_n, out_n, use_bias=True, activation=None):
        self.w = np.random.normal(0, 1, [in_n, out_n])
        self.b = np.random.normal(0, 1, [out_n]) if use_bias else None
        self.activation = activation

    def set_lr(self, lr):
        self.lr = lr

    def forward(self, x):
        self.x_in = x
        x = np.dot(x, self.w) + self.b

        if self.activation:
            x = self.activation(x)
        self.x_out = x
        return x

    def backward(self, w_pro, grad_pro):
        grad = np.dot(grad_pro, w_pro.T)
        if self.activation is sigmoid:
            grad *= self.x_out * (1 - self.x_out)
        grad_w = np.dot(self.x_in.T, grad)
        self.w -= self.lr * grad_w

        if self.b is not None:
            grad_b = np.dot(np.ones([grad.shape[0]]), grad)
            self.b -= self.lr * grad_b

        return grad


class Model():
    def __init__(self, *args, lr):
        self.layers = args
        for layer in self.layers:
            layer.set_lr(lr=lr)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.output = x
        return x

    def backward(self, t):
        En = (self.output - t) * self.output * (1 - self.output)
        grad_pro = En
        w_pro = np.eye(En.shape[-1])
        for layer in self.layers[::-1]:
            grad_pro = layer.backward(w_pro=w_pro, grad_pro=grad_pro)
            w_pro = layer.w


model = Model(
    FullyConnectedLayer(in_n=2, out_n=64, activation=sigmoid),
    FullyConnectedLayer(in_n=64, out_n=32, activation=sigmoid),
    FullyConnectedLayer(in_n=32, out_n=1, activation=sigmoid),
    lr=.1
)

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
t = np.array([[0], [1], [1], [0]])

for ite in range(10000):
    model.forward(x)
    model.backward(t)

for _x in x:
    out = model.forward(_x)
    print("in >>", _x, ", out >>", out)
