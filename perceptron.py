import numpy as np

xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
ts = np.array([[-1], [-1], [-1], [1]], dtype=np.float32)

np.random.seed(0)
w = np.random.normal(0., 1, (3))
print("weight >>", w)

_xs = np.hstack((xs, [[1] for _ in xs]))
for x in _xs:
    print("in >>", x, "y >>", np.dot(x, w))
