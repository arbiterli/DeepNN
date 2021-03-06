import tensorflow as tf


class CNN:
    def __init__(self, batch_size=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images = tf.placeholder(tf.float32, shape=(batch_size, 28 * 28), name='input_images')
            self.labels = tf.placeholder(tf.float32, shape=(batch_size,), name='input_labels')
            self.X = tf.reshape(self.images, [-1, 28, 28, 1])
            self.Y = tf.one_hot(indices=tf.cast(self.labels, tf.int32), depth=10)
            self.build_graph()
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.Y, logits=self.output)
            self.optimize()

    def build_graph(self):
        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv2d(inputs=self.X,
                                     filters=64,
                                     kernel_size=[5,5],
                                     padding='same',
                                     activation=tf.nn.relu)

        with tf.variable_scope('pooling1'):
            pooling1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

        with tf.variable_scope('conv2'):
            conv2 = tf.layers.conv2d(inputs=pooling1,
                                     filters=64,
                                     kernel_size=[5,5],
                                     padding='same',
                                     activation=tf.nn.relu)

        with tf.variable_scope('pooling2'):
            pooling2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

        pool_flat = tf.reshape(pooling2, [-1, 7*7*64])
        with tf.variable_scope('dense'):
            dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)

        with tf.variable_scope('dropout'):
            dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

        with tf.variable_scope('output'):
            self.output = tf.layers.dense(inputs=dropout, units=10)

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

