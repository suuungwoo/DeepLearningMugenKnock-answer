import matplotlib.pyplot as plt
import numpy as np

xs = np.array(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.float32)
ts = np.array(((-1), (-1), (-1), (1)), dtype=np.float32)

_xs = np.hstack((xs, [[1] for _ in xs]))
lrs = (.1, .01)
linestyles = ('solid', 'dashed')
plts = []

for i, lr in enumerate(lrs):
    np.random.seed(0)
    w = np.random.normal(0., 1, (3))
    print("weight >>", w)
    w1, w2, w3 = [], [], []
    ite = 0

    while True:
        ite += 1
        ys = np.dot(_xs, w)
        print("iteration:", ite, "y >>", ys)

        if len(np.where(ys * ts < 0)[0]) < 1:
            break
        _ys = ys.copy()
        _ts = ts.copy()
        _ys[ys * ts >= 0] = 0
        _ts[ys * ts >= 0] = 0

        En = np.dot(_ts, _xs)
        w1.append(w[0])
        w2.append(w[1])
        w3.append(w[2])
        w += lr * En

    plt.plot(list(range(ite - 1)), w1, linestyle=linestyles[i])
    plt.plot(list(range(ite - 1)), w2, linestyle=linestyles[i])
    plt.plot(list(range(ite - 1)), w3, linestyle=linestyles[i])
    plt.legend(["w1:lr=0.1", "w2:lr=0.1", "w3:lr=0.1",
                "w1:lr=0.01", "w2:lr=0.01", "w3:lr=0.01"], loc="best")
plt.savefig("answer_perceptron3.png")
plt.show()
print("training finished!")
print("weight >>", w)

for x in _xs:
    print("in >>", x, "out >>", np.dot(x, w))
