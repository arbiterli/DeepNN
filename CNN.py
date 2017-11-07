import tensorflow as tf

class cnn:
    def __init__(self, batch_size=128):
        g = tf.Graph()
        with g.as_default():
            self.X = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
            self.build_graph()

    def build_graph(self):
        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv2d(inputs=self.X,
                                     filters=64,
                                     kernel_size=[5,5],
                                     padding='same',
                                     activation=tf.nn.relu)

        with tf.variable_scope('pooling1'):
            pooling1 = tf.layers.max_pooling2d(inputs=conv1,
                                               pool_size=[2,2],
                                               strides=2)


