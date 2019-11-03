import tensorflow as tf

num_classes = 2
img_height, img_width = 64, 64


def Mynet(x, train=False):
    x = tf.layers.conv2d(
        inputs=x, filters=32, kernel_size=[3, 3],
        padiding="same", activation=tf.nn.relu, name="conv1")
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name="conv2")
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

    mb, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h * w * c])
    x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu, name="fc1")
    x = tf.layers.dropout(inputs=x, rate=0.25, traininig=train)
    x = tf.layers.dense(inputs=x, units=num_classes, name="fc_cls")

    return x
