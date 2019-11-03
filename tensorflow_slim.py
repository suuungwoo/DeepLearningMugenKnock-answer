import tensorflow as tf
from tensorflow.contrib import slim

num_classes = 2
img_height, img_width = 64, 64


def Mynet(x, train=False):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0., 0.01)):
        x = slim.conv2d(x, 64, [3, 3], scope="conv1")
        x = slim.batch_norm(x, is_training=train, scope="bn1")
        x = slim.max_pool2d(x, [2, 2], scope="pool1")
        x = slim.conv2d(x, 64, [3, 3], scope="conv2")
        x = slim.batch_norm(x, is_training=train, scope="bn2")
        x = slim.max_pool2d(x, [2, 2], scope="pool2")
        x = slim.flatten(x)
        x = slim.fully_connected(x, 256, scope="fc1")
        x = slim.dropout(x, 0.25, scope="drop1")
        x = slim.fully_connected(x, 256, scope="fc2")
        x = slim.dropout(x, 0.25, scope="drop2")
        x = slim.fully_connected(x, num_classes, scope="fc_cls")
    return x
