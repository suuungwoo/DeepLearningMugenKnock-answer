import cv2
import glob
import numpy as np

num_classes = 2
img_height, img_width = 64, 64
CLS = ["akahara", "madara"]


def data_load(path):
    paths = glob.glob(path + "*/*")
    xs = []
    ts = []
    for path in paths:
        x = cv2.imread(path)
        x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
        x /= 255.
        xs.append(x)

        for i, cls in enumerate(CLS):
            if cls in path:
                t = i
        ts.append(t)

    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.float32)
    return xs, ts, paths


xs, ts, paths = data_load("Dataset/train/images/")
xs = xs.transpose(0, 3, 1, 2)
mb = 3
mbi = 0
train_ind = np.arange(len(xs))
np.random.seed(0)
np.random.shuffle(train_ind)

for i in range(10):
    remain_num = len(xs) - mbi
    if remain_num >= mb:
        mb_ind = train_ind[mbi:mbi + mb]
        mbi += mb
    else:
        mb_ind = train_ind[mbi:]
        np.random.shuffle(train_ind)
        mb_ind = np.hstack((mb_ind, train_ind[:(mb - remain_num)]))
        mbi = mb - remain_num
    print(mb_ind)
