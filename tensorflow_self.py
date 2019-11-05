import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import argparse

num_classes = 2
img_height, img_width = 64, 64


def conv2d(
        x,
        k=3,
        in_num=1,
        out_num=32,
        strides=1,
        activ=None,
        bias=True,
        name="conv"):
    w = tf.compat.v1.Variable(tf.random.normal(
        [k, k, in_num, out_num]), name=name + "_w")
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding="SAME")
    tf.compat.v1.add_to_collections("vars", w)
    if bias:
        b = tf.Variable(tf.random.normal([out_num]), name=name + "_b")
        tf.compat.v1.add_to_collections("vars", b)
        x = tf.nn.bias_add(x, b)
    if activ is not None:
        x = activ(x)
    return x


def maxpool2d(x, k=2):
    return tf.nn.max_pool2d(
        x, ksize=[
            1, k, k, 1], strides=[
            1, k, k, 1], padding="SAME")


def fc(x, in_num=100, out_num=100, bias=True, activ=None, name="fc"):
    w = tf.Variable(tf.random.normal([in_num, out_num]), name=name + "_w")
    x = tf.matmul(x, w)
    tf.compat.v1.add_to_collections("vars", w)
    if bias:
        b = tf.Variable(tf.random.normal([out_num]), name=name + "_b")
        tf.compat.v1.add_to_collections("vars", b)
        x = tf.add(x, b)
    if activ is not None:
        x = activ(x)
    return x


def Mynet(x, keep_prob):
    x = conv2d(x, k=3, in_num=3, out_num=32, activ=tf.nn.relu, name="conv1_1")
    x = conv2d(x, k=3, in_num=32, out_num=32, activ=tf.nn.relu, name="conv1_2")
    x = maxpool2d(x, k=2)
    x = conv2d(x, k=3, in_num=32, out_num=64, activ=tf.nn.relu, name='conv2_1')
    x = conv2d(x, k=3, in_num=64, out_num=64, activ=tf.nn.relu, name='conv2_2')
    x = maxpool2d(x, k=2)
    x = conv2d(
        x,
        k=3,
        in_num=64,
        out_num=128,
        activ=tf.nn.relu,
        name='conv3_1')
    x = conv2d(
        x,
        k=3,
        in_num=128,
        out_num=128,
        activ=tf.nn.relu,
        name='conv3_2')
    x = maxpool2d(x, k=2)

    mb, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h * w * c])
    x = fc(x, in_num=w * h * c, out_num=1024, activ=tf.nn.relu, name='fc1')
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, rate=(1 - keep_prob))
    x = fc(x, in_num=1024, out_num=num_classes, name='fc_out')
    return x


def data_load(path, hf=False, vf=False):
    xs = np.ndarray((0, img_height, img_width, 3), dtype=np.float32)
    ts = np.ndarray((0, num_classes))
    paths = []

    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            xs = np.r_[xs, x[None, ...]]

            t = np.zeros((num_classes))
            if 'akahara' in path:
                t[0] = 1
            elif 'madara' in path:
                t[1] = 1
            t = t[None, ...]
            ts = np.r_[ts, t]

            paths.append(path)

            if hf:
                _x = x[:, ::-1]
                xs = np.r_[xs, _x[None, ...]]
                ts = np.r_[ts, t]
                paths.append(path)

            if vf:
                _x = x[::-1]
                xs = np.r_[xs, _x[None, ...]]
                ts = np.r_[ts, t]
                paths.append(path)

            if hf and vf:
                _x = x[::-1, ::-1]
                xs = np.r_[xs, _x[None, ...]]
                ts = np.r_[ts, t]
                paths.append(path)

    return xs, ts, paths


def train():
    tf.compat.v1.reset_default_graph()

    X = tf.compat.v1.placeholder(tf.float32, [None, img_height, img_width, 3])
    Y = tf.compat.v1.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.compat.v1.placeholder(tf.float32)

    logits = Mynet(X, keep_prob)

    preds = tf.nn.softmax(logits)
    loss = tf.reduce_mean(
        tf.compat.v1.losses.softmax_cross_entropy(
            logits=logits,
            onehot_labels=Y))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    xs, ts, paths = data_load('Dataset/train/images/', hf=True, vf=True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"

    mb = 3
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(100):
            if mbi + mb > len(xs):
                mb_ind = train_ind[mbi:]
                np.random.shuffle(train_ind)
                mb_ind = np.hstack(
                    (mb_ind, train_ind[:(mb - (len(xs) - mbi))]))
            else:
                mb_ind = train_ind[mbi: mbi + mb]
                mbi += mb

            x = xs[mb_ind]
            t = ts[mb_ind]

            _, acc, los = sess.run([train, accuracy, loss],
                                   feed_dict={X: x, Y: t, keep_prob: 0.5})

        saver = tf.compat.v1.train.Saver()
        saver.save(sess, "./cnn.ckpt")


def test():
    tf.compat.v1.reset_default_graph()

    X = tf.compat.v1.placeholder(tf.float32, [None, img_height, img_width, 3])
    keep_prob = tf.compat.v1.placeholder(tf.float32)

    logits = Mynet(X, keep_prob)
    out = tf.nn.softmax(logits)

    xs, ts, paths = data_load('Dataset/train/images/')

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"

    with tf.compat.v1.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./cnn.ckpt")

        for i in range(len(paths)):
            x = xs[i]
            path = paths[i]

            x = np.expand_dims(x, axis=0)

            pred = sess.run([out], feed_dict={X: x, keep_prob: 1.})[0]
            pred = pred[0]
            print("in {}, predicted probabilities >> {}".format(path, pred))


def arg_parse():
    parser = argparse.ArgumentParser(description="CNN implemented with Keras")
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parse()

    if args.train:
        train()
    if args.test:
        test()

    if not(args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
