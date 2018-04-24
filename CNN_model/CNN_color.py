import tensorflow as tf


class CNNColor:
    def __init__(self, image_width, image_height, channels=3, batch_size=None):
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = channels
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, channels), name='X')
            self.labels = tf.placeholder(tf.float32, shape=(batch_size,), name='input_labels')
            self.Y = tf.one_hot(indices=tf.cast(self.labels, tf.int32), depth=10)
            self.build_graph()
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.Y, logits=self.output)
            self.optimize()


    def inception_layer(self, inputs, filters):
        conv1x1 = tf.layers.conv2d(inputs=inputs,
                                   filters=filters,
                                   kernel_size=[1, 1],
                                   padding='same',
                                   activation=tf.nn.relu)

        conv3x3 = tf.layers.conv2d(inputs=inputs,
                                 filters=filters,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.relu)

        conv5x5 = tf.layers.conv2d(inputs=inputs,
                                 filters=filters,
                                 kernel_size=[5, 5],
                                 padding='same',
                                 activation=tf.nn.relu)

        avg_pooling = tf.layers.average_pooling2d(inputs=inputs, pool_size=[2,2], strides=1, padding='same')

        return tf.concat([conv1x1, conv3x3, conv5x5, avg_pooling], axis=3)


    def inception_conv_graph(self):
        with tf.variable_scope('incep1'):
            incep1 = self.inception_layer(inputs=self.X, filters=64)

        with tf.variable_scope('incep2'):
            incep2 = self.inception_layer(inputs=incep1, filters=64)

        with tf.variable_scope('incep3'):
            incep3 = self.inception_layer(inputs=incep2, filters=64)

        with tf.variable_scope('max_pooling'):
            pooling = tf.layers.max_pooling2d(inputs=incep3, pool_size=[2,2], strides=2)

        self.pool_flat = tf.reshape(pooling, [-1, 148224])
        with tf.variable_scope('dense'):
            dense = tf.layers.dense(inputs=self.pool_flat, units=2048, activation=tf.nn.relu)

        with tf.variable_scope('dropout'):
            dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=True)

        with tf.variable_scope('output'):
            self.output = tf.layers.dense(inputs=dropout, units=10)


    # conv-conv-pooling (same padding) X 2, conv-conv-pooling (2 strides) X 2
    def normal_conv_graph(self):
        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv2d(inputs=self.X,
                                     filters=64,
                                     kernel_size=[4,4],
                                     padding='same',
                                     activation=tf.nn.relu)

        with tf.variable_scope('conv2'):
            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=64,
                                     kernel_size=[4,4],
                                     padding='same',
                                     activation=tf.nn.relu)

        with tf.variable_scope('pooling1'):
            pooling1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=1, padding='same') # strides = 1


        with tf.variable_scope('conv3'):
            conv3 = tf.layers.conv2d(inputs=pooling1,
                                     filters=64,
                                     kernel_size=[4,4],
                                     padding='same',
                                     activation=tf.nn.relu)

        with tf.variable_scope('conv4'):
            conv4 = tf.layers.conv2d(inputs=conv3,
                                     filters=64,
                                     kernel_size=[4,4],
                                     padding='same',
                                     activation=tf.nn.relu)

        with tf.variable_scope('pooling2'):
            pooling2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=1, padding='same') # strides = 1

        with tf.variable_scope('conv5'):
            conv5 = tf.layers.conv2d(inputs=pooling2,
                                     filters=64,
                                     kernel_size=[4,4],
                                     padding='same',
                                     activation=tf.nn.relu)

        with tf.variable_scope('conv6'):
            conv6 = tf.layers.conv2d(inputs=conv5,
                                     filters=64,
                                     kernel_size=[4,4],
                                     padding='same',
                                     activation=tf.nn.relu)

        with tf.variable_scope('pooling3'):
            pooling3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2,2], strides=2)


        with tf.variable_scope('conv7'):
            conv7 = tf.layers.conv2d(inputs=pooling3,
                                     filters=64,
                                     kernel_size=[4,4],
                                     padding='same',
                                     activation=tf.nn.relu)

        with tf.variable_scope('conv8'):
            conv8 = tf.layers.conv2d(inputs=conv7,
                                     filters=64,
                                     kernel_size=[4,4],
                                     padding='same',
                                     activation=tf.nn.relu)

        with tf.variable_scope('pooling3'):
            pooling4 = tf.layers.max_pooling2d(inputs=conv8, pool_size=[2,2], strides=2)

        self.pool_flat = tf.reshape(pooling4, [-1, int((self.image_width/4) * (self.image_height/4) * 64)])
        with tf.variable_scope('dense'):
            dense = tf.layers.dense(inputs=self.pool_flat, units=1024, activation=tf.nn.relu)

        with tf.variable_scope('dropout'):
            dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=True)

        with tf.variable_scope('output'):
            self.output = tf.layers.dense(inputs=dropout, units=10)

    def build_graph(self):
        # self.normal_conv_graph()
        self.inception_conv_graph()

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


